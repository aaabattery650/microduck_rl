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
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

from mjlab_microduck.tasks import MicroduckOnPolicyRunner  # registry runner


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
    runner_cls = load_runner_cls(args.task) or MicroduckOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
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
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
