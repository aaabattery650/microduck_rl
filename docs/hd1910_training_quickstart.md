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

**Before you leave it running:** (a) Windows Settings → System → Power →
**Sleep = Never** when plugged in (sleep suspends WSL and kills training),
(b) don't close the terminal windows, (c) always pass `--agent.max_iterations`
— without it training runs to 50 000.

**Terminal 1 — training** (the `tee` copies the output to a file so the
dashboard can read it; `1500` is the full budget for this task, ~15–60 min
on a modern GPU):

```bash
cd ~/microduck_rl
uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.max_iterations 1500 2>&1 | tee /tmp/train_stand.log
```

**Terminal 2 — live dashboard** (opens http://localhost:8765: graphs
refresh every 5 s; the ▶ button opens the 3D viewer on the latest
checkpoint):

```bash
cd ~/microduck_rl
uv run scripts/train_monitor.py --task Mjlab-Stand-Flat-MicroDuck --log /tmp/train_stand.log
```

You can leave this running unattended — training stops by itself at 1500.

**After it finishes (or the next morning):**

```bash
LATEST=$(ls -t logs/rsl_rl/stand/*/model_*.pt | head -1)   # newest checkpoint
uv run play Mjlab-Stand-Flat-MicroDuck --checkpoint-file "$LATEST" --num-envs 1
```

Healthy run: `Mean episode length` climbing toward the timeout fast,
`fell_over` → 0, `upright`/`stand_height`/`pose` all growing. Gait rewards
are absent by design — nothing pays for stepping. The exported ONNX drops
into the runtime's `stand` slot (`uv run publish ... --slot stand`).

## 3. Real training (walking)

Same two-terminal pattern — swap the task id, log name, and budget
(**4000–6000 iterations**, ~1–3 h: the true overnight run):

```bash
# terminal 1:
wandb login                     # optional; or export WANDB_MODE=offline
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.max_iterations 5000 2>&1 | tee /tmp/train_vel.log

# terminal 2:
uv run scripts/train_monitor.py --task Mjlab-Velocity-Flat-MicroDuck --log /tmp/train_vel.log
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

## Watching it train: live dashboard

Terminal 1 runs training with its output teed to a file; terminal 2 runs a
small local dashboard (graphs + a button that opens the 3D viewer on the
latest checkpoint — the fully accurate model):

```bash
# terminal 1 — training, tee'd:
uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096 2>&1 | tee /tmp/train_stand.log

# terminal 2 — dashboard (opens in your browser at http://localhost:8765):
uv run scripts/train_monitor.py --task Mjlab-Stand-Flat-MicroDuck --log /tmp/train_stand.log
```

The dashboard live-plots mean reward, episode length, `fell_over`, the task
reward terms and the penalties, and its **Watch robot** button runs
`play --checkpoint-file <latest>` — checkpoints refresh every 250 iterations.
No GUI toolkit needed (it's a local web page), works on WSL2. Alternatively
`wandb login` gives the full hosted dashboard, and
`uv run play <TASK> --checkpoint-file <...>` opens the viewer directly.

## 4. View the trained policy

**One command** (from the clone where you trained — finds the newest
checkpoint itself and picks the right task):

```bash
bash scripts/watch.sh
```

Wait ~1 min, open the printed `http://localhost:PORT` URL in the Windows
browser. Drag = orbit, scroll = zoom, `Ctrl+C` closes. Variants:

```bash
bash scripts/watch.sh --envs 4              # four robots at once
bash scripts/watch.sh --no-terminations     # inspect falls without auto-reset
bash scripts/watch.sh --task Mjlab-Velocity-Flat-MicroDuck   # override task
```

(On WSL2 this always uses the browser viewer on CPU — full speed in the
browser, no VRAM contention with a running training. Use the native window
on Mac/Linux desktops: same play command without `--viewer viser`.)

<details><summary>Manual / other-machine variants</summary>

```bash
LATEST=$(ls -t logs/rsl_rl/stand/*/model_*.pt | head -1)     # or .../velocity/...
uv run play Mjlab-Stand-Flat-MicroDuck --checkpoint-file "$LATEST" --num-envs 1

# from a wandb run:
uv run play Mjlab-Stand-Flat-MicroDuck --wandb-run-path <entity>/mjlab_microduck/<run_id>

# headless video clip (needs the exported ONNX):
uv run scripts/render_policy.py output.onnx --seconds 10 --video media/stand.mp4
```
</details>

**What a successful stand policy looks like:** starts standing (from a random
tilt) and stays up; staggers and recovers from the periodic pushes; moves its
head on command without falling. Face-planting, growing oscillation, or a
persistent crouch = not converged yet — train longer or check the dashboard
graphs.

## 5. Export for deployment

```bash
LATEST=$(ls -t logs/rsl_rl/stand/*/model_*.pt | head -1)     # or .../velocity/...
uv run scripts/export.py Mjlab-Stand-Flat-MicroDuck --checkpoint-file "$LATEST"
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
