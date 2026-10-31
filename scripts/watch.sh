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
DRY=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task)  TASK="$2"; shift 2 ;;
    --envs)  ENVS="$2"; shift 2 ;;
    --dry)   DRY=1; shift ;;
    *)       EXTRA+=("$1"); shift ;;
  esac
done

CKPT="$(ls -t logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1 || true)"
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
     --viewer viser --device cpu "${EXTRA[@]+"${EXTRA[@]}"}")
echo "Running: ${CMD[*]}"
if [ "$DRY" = 1 ]; then exit 0; fi
exec "${CMD[@]}"
