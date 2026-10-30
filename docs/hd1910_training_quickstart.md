# HD-1910 training quickstart

The exact command list to go from a fresh PC to a trained, viewable, exportable
Microduck policy with the Feetech HD-1910 actuator model. Tested paths: Linux
x86_64 CUDA (WSL2 included) for training, macOS/CPU for tests and viewing.

## 1. One-time setup (Windows PC with NVIDIA GPU, via WSL2)

WSL2 is the supported Windows path: PyPI's native-Windows torch wheel is
CPU-only (training dies with `IndexError` in `select_gpus()`), while the
Linux x86_64 wheel bundles CUDA — no repo changes needed.

```powershell
# PowerShell (admin): install WSL2 + Ubuntu, and the NVIDIA Game Driver
# on the WINDOWS side only (WSL uses it; never install a Linux driver).
wsl --install -d Ubuntu-24.04
```

```bash
# Inside Ubuntu:
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_HTTP_TIMEOUT=600          # first sync pulls ~2 GB of CUDA wheels

git clone https://github.com/aaabattery650/microduck_rl
cd microduck_rl
uv sync

# Sanity: must print "True" and your GPU name before anything else.
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

GPU sizing: 16 GB VRAM → `--env.scene.num-envs 4096`; 8 GB → start at 2048
and raise while `nvidia-smi` shows headroom.

## 2. Smoke test — ALWAYS first (~2 minutes)

A 5-iteration / 64-env run catches ~95 % of config errors for cents:

```bash
uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

Expected: it builds the env (actor 61-D / critic 76-D), runs 5 iterations,
saves `model_0..4.pt`, all reward terms print, no NaNs. The robot falls over
constantly at this stage — that is expected and fine.

## 2b. First real policy: balance (`Mjlab-Stand-Flat-MicroDuck`)

The simplest policy in the family — **stay standing** at (near-)zero command
under initial tilt, pushes and head motion. Best first run: fast, dead-simple
success criterion, and it validates the whole HD-1910 pipeline end to end
before you invest in gaits.

```bash
uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096
```

Budget **500–1500 iterations** (~15–60 min on a modern GPU). Healthy run:
`Mean episode length` climbing toward the timeout fast, `fell_over` → 0,
`upright`/`stand_height`/`pose` all growing. Gait rewards are absent by
design — nothing pays for stepping. The exported ONNX drops into the
runtime's `stand` slot (`uv run publish ... --slot stand`).

Then view it:

```bash
uv run play Mjlab-Stand-Flat-MicroDuck --checkpoint-file logs/rsl_rl/stand/<ts>/model_1000.pt
```

## 3. Real training (walking)

```bash
wandb login                     # or: export WANDB_MODE=offline
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

- Budget: **4000–6000 iterations** (~1–3 h on a modern GPU) for a usable gait;
  episodic tricks need ~1000.
- Checkpoints: `logs/rsl_rl/velocity/<timestamp>/model_XXXX.pt` (every ~50 iters).
- Resume after an interruption:
  `uv run train <TASK> --env.scene.num-envs 4096 --agent.load-checkpoint model_29999.pt --agent.resume True`
- No GPU at all? Append `--hf-jobs` to any train command (Hugging Face Jobs).
- Watching progress (wandb, project `mjlab_microduck`): mean reward rising,
  episode length climbing toward the 1000-step timeout, `fell_over` → 0,
  and **every `Episode_Reward/<penalty>` ≤ 0** (a positive penalty = a
  reward-hacked run — stop and fix, don't deploy).

## 4. View the policy

```bash
# Faithful viewer (exact training physics: HD-1910 actuator + DR). Interactive 3D.
uv run play Mjlab-Velocity-Flat-MicroDuck --checkpoint-file logs/rsl_rl/velocity/<ts>/model_4000.pt
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity>/mjlab_microduck/<run_id>   # from wandb
# Options: --num-envs 4 (slow machines), --no-terminations (watch falls), --video.

# Headless video clip (no display needed):
uv run scripts/render_policy.py output.onnx --seconds 10 --video media/walk.mp4
```

The `play` command auto-falls back to CPU — you can copy a `model_XXXX.pt` to
the Mac and view it there the same way.

## 5. Export for deployment

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity>/mjlab_microduck/<run_id>
# → output.onnx  ([1,61] → [1,14], obs normalizer baked in — the only safe export path)

# Runtime rehearsal on CPU (keyboard-driven; use --no-bam for non-XL330 physics):
uv run scripts/infer_policy.py --walking output.onnx --no-bam
```

## What gets trained

Each task id trains ONE policy (an ONNX, hot-swappable at runtime through the
shared 61-D observation contract). Start with the walking task:

| Task id | Policy learns | Budget (iters @4096) |
|---|---|---|
| `Mjlab-Stand-Flat-MicroDuck` | **Start here.** Balance: hold the stand at zero command under tilt spawns, pushes and head motion | **0.5–1.5 k** |
| `Mjlab-Velocity-Flat-MicroDuck` | Walking gaits: track commanded forward/side speed + turn (incl. spin-in-place), head-pose commands, upright idle at zero command | 4–6 k |
| `Mjlab-Velocity-Rough-MicroDuck` | Same, on rough terrain | 4–6 k |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | Walking + fall recovery in one policy | 4–6 k |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | Stand up from face-down/face-up/sitting, then hold + body-pose control | ~1 k |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | Commanded sit ↔ stand, gentle, head commandable | ~1 k |
| `Mjlab-GroundPick-*`, `Mjlab-BallKick-Flat-*`, `Mjlab-Roulade-Flat-*` | Episodic tricks: mouth-to-ground, kick, forward roll | ~1 k each |
| `*-Rollers / Swizzle / RollerCrouch / RollerSlope / RollerStandUp / Spin` | Roller-skate family (needs the wheel attachments) | varies |

Robustness is baked into every run via domain randomization (HD-1910 bus
voltage 6.2–8.4 V scaling torque AND speed, firmware gains ±15 %/±10 %,
gearbox friction ×0.5–1.5, mass/inertia/CoM, armature, delays, pushes,
encoder bias, IMU misalignment, obs noise) plus the printed spacer mass and
the beak — so the policy transfers to the physical robot inside that envelope
(see AGENTS.md for the limits: systematic IMU bias is a runtime calibration,
not a trained behavior).

The jaw servo (15th HD-1910) is deliberately outside every policy: it is
simulated (`passive_jaw`, beak) but daemon-commanded at deployment.

`uv run list-envs` prints the live registry, including the `-Backlash-`
twins (currently XL330-study only — do not use them for HD-1910 runs).
