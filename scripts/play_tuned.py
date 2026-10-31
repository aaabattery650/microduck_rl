#!/usr/bin/env python
"""Play a checkpoint with CONTROLLED pushes (magnitude / frequency / off).

Same env + policy loading path as mjlab's `play`, but exposes the push
disturbance so you can stress-test a balance policy interactively:

  uv run scripts/play_tuned.py --task Mjlab-Stand-Flat-MicroDuck \
      --checkpoint-file logs/rsl_rl/stand/<run>/model_1499.pt --gpu

  --push 1.0            pushes up to ±1.0 m/s (default 0.3, the training value)
  --push-every 3.0      one push every 3 s (play default: every 0.5–1 s)
  --no-push             disable pushes entirely (watch pure balance)
  --no-terminations     falls don't reset the episode
  --envs 4              four robots

Viewer is always viser (browser) — the WSL-friendly path.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

from mjlab_microduck.tasks import MicroduckOnPolicyRunner  # registry runner


class PushableViewer(ViserPlayViewer):
    """ViserPlayViewer + a Manual push panel and click-to-shove.

    GUI: strength / direction sliders + PUSH and Random buttons, applying the
    same velocity-kick the training-time push event uses. Click anywhere on
    the ground: the duck is shoved horizontally TOWARD the clicked point with
    the current strength.
    """

    def setup(self) -> None:
        super().setup()
        env = self.env
        asset = env.scene["robot"]

        def kick(vx: float, vy: float, vz: float = 0.0):
            vel = asset.data.root_link_vel_w.clone()
            vel[0, 0] += vx
            vel[0, 1] += vy
            vel[0, 2] += vz
            asset.write_root_link_velocity_to_sim(vel, env_ids=torch.tensor([0], device=env.device))
            print(f"[push] vx={vx:+.2f} vy={vy:+.2f} m/s")

        with self._server.gui.add_folder("Manual push 🫸"):
            mag = self._server.gui.add_slider("Strength (m/s)", min=0.05, max=1.5, step=0.05, initial_value=0.5)
            ang = self._server.gui.add_slider("Direction (deg)", min=-180, max=180, step=5, initial_value=0)
            btn = self._server.gui.add_button("PUSH")
            rnd = self._server.gui.add_button("Random push")

            @btn.on_click
            def _push(_ev):
                a = math.radians(ang.value)
                kick(mag.value * math.cos(a), mag.value * math.sin(a))

            @rnd.on_click
            def _rand(_ev):
                a = 2 * math.pi * float(np.random.rand())
                kick(mag.value * math.cos(a), mag.value * math.sin(a))

            note = self._server.gui.add_markdown(
                "*or click the ground to shove the duck toward that point*"
            )
            del note

        @self._server.scene.on_click()
        def _click(ev):
            o = np.array(ev.ray_origin)
            d = np.array(ev.ray_direction)
            target_z = float(asset.data.root_link_pos_w[0, 2].cpu())
            if abs(d[2]) < 1e-6:
                return
            t = (target_z - o[2]) / d[2]
            hit = o + t * d
            robot_xy = asset.data.root_link_pos_w[0, :2].cpu().numpy()
            delta = hit[:2] - robot_xy
            n = np.linalg.norm(delta)
            if n < 1e-3:
                return
            kick(mag.value * delta[0] / n, mag.value * delta[1] / n)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint-file", required=True)
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--gpu", action="store_true", help="use cuda:0 (default: auto)")
    p.add_argument("--cpu", action="store_true", help="force cpu")
    p.add_argument("--device", default=None, help="explicit device (overrides --gpu/--cpu)")
    p.add_argument("--push", type=float, default=0.3, help="max push speed m/s (training value 0.3)")
    p.add_argument("--push-every", type=float, default=None, help="seconds between pushes (default: play 0.5-1.0)")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--no-terminations", action="store_true")
    args = p.parse_args()

    device = (
        args.device
        or ("cuda:0" if args.gpu else "cpu" if args.cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    )

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.envs
    if args.no_terminations:
        env_cfg.terminations = {}

    if "push_robot" in env_cfg.events:
        if args.no_push:
            env_cfg.events.pop("push_robot")
            print("[push] disabled")
        else:
            push = env_cfg.events["push_robot"]
            push.params["velocity_range"]["x"] = (-args.push, args.push)
            push.params["velocity_range"]["y"] = (-args.push, args.push)
            if args.push_every is not None:
                push.interval_range_s = (args.push_every, args.push_every)
            print(
                f"[push] ±{args.push} m/s every "
                f"{args.push_every if args.push_every else '0.5–1.0'} s "
                f"(training was ±0.3 every 3–6 s)"
            )

    resume_path = Path(args.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(resume_path)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(args.task)
    rsl_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MicroduckOnPolicyRunner
    runner = runner_cls(rsl_env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    ckpt_dir = resume_path.parent

    def fetch_available():
        now = time.time()
        entries = []
        for f in sorted(ckpt_dir.glob("*.pt")):
            try:
                step = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                step = 0
            entries.append((f.name, format_time_ago(int(now - f.stat().st_mtime)), step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

    def reload_policy(name: str):
        runner.load(str(ckpt_dir / name), load_cfg={"actor": True}, strict=True, map_location=device)
        return runner.get_inference_policy(device=device)

    ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available,
        load_checkpoint=reload_policy,
    )
    PushableViewer(env, policy, checkpoint_manager=ckpt_manager).run()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
