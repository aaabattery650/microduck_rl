#!/usr/bin/env bash
# One-command policy viewer: finds the NEWEST checkpoint in this repo's logs/,
# derives the task id from the experiment folder, launches the browser viewer.
#
#   bash scripts/watch.sh                              # newest checkpoint
#   bash scripts/watch.sh --envs 4                     # four robots
#   bash scripts/watch.sh --no-terminations            # watch falls, no reset
#   bash scripts/watch.sh --task Mjlab-Stand-Flat-MicroDuck   # override task
#
# Then: wait ~1 min, open the printed http://localhost:PORT URL in your
# browser. Ctrl+C closes the viewer. (Always viser+CPU: full-speed in the
# browser, no VRAM contention with training, works on WSL2.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENVS=1
TASK=""
DEVICE=""
DRY=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task)  TASK="$2"; shift 2 ;;
    --envs)  ENVS="$2"; shift 2 ;;
    --cpu)   DEVICE=cpu; shift ;;
    --gpu)   DEVICE=cuda:0; shift ;;
    --dry)   DRY=1; shift ;;
    --info)  INFO=1; shift ;;
    *)       EXTRA+=("$1"); shift ;;
  esac
done

# Device: GPU when it's free (fast, real-time); CPU when a training run owns
# the GPU (avoids the CUDA-OOM viewer crash) or no GPU exists.
if [ -z "$DEVICE" ]; then
  if pgrep -f "bin/train" >/dev/null 2>&1; then
    DEVICE=cpu
    echo "Training is running -> viewer on CPU (slow motion, but no VRAM conflict)."
  elif uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda:0
    echo "GPU is free -> viewer on GPU (real-time)."
  else
    DEVICE=cpu
    echo "No GPU available -> viewer on CPU (slow motion)."
  fi
fi

CKPT="$(ls -t logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1 || true)"

if [ "${INFO:-0}" = 1 ]; then
  echo "repo:     $ROOT"
  echo "ckpt:     ${CKPT:-<none found>}"
  echo "training: $(pgrep -af 'bin/train' || echo 'not running')"
  echo "cuda:     $(uv run python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
  echo "device:   ${DEVICE:-auto}"
  echo "run without --info to launch the viewer"
  exit 0
fi

if [ -z "${CKPT:-}" ]; then
  echo "No checkpoints under $ROOT/logs — train something first."
  exit 1
fi
echo "Newest checkpoint: $CKPT"

if [ -z "$TASK" ]; then
  EXP="$(basename "$(dirname "$(dirname "$CKPT")")")"
  case "$EXP" in
    stand)    TASK=Mjlab-Stand-Flat-MicroDuck ;;
    velocity) TASK=Mjlab-Velocity-Flat-MicroDuck ;;
    velstand) TASK=Mjlab-VelStand-Flat-MicroDuck ;;
    *)
      echo "Experiment '$EXP' has no task mapping — rerun with:  --task <task-id>"
      exit 1
      ;;
  esac
fi
echo "Task: $TASK"

pkill -f "bin/play" 2>/dev/null || true   # don't stack viewers

CMD=(uv run play "$TASK" --checkpoint-file "$CKPT" --num-envs "$ENVS"
     --viewer viser --device "$DEVICE" "${EXTRA[@]+"${EXTRA[@]}"}")
echo "Running: ${CMD[*]}"
if [ "$DRY" = 1 ]; then exit 0; fi
exec "${CMD[@]}"
