#!/usr/bin/env bash
# One-command policy viewer: finds the NEWEST checkpoint across every known
# clone of this repo, derives the task id from the experiment folder, picks a
# sensible device, and launches the browser viewer.
#
#   bash scripts/watch.sh                              # newest checkpoint anywhere
#   bash scripts/watch.sh --envs 4                     # four robots
#   bash scripts/watch.sh --no-terminations            # watch falls, no reset
#   bash scripts/watch.sh --task Mjlab-Stand-Flat-MicroDuck   # override task
#   bash scripts/watch.sh --ckpt <path/to/model.pt>    # explicit checkpoint
#   bash scripts/watch.sh --info                       # diagnostics only
#
# Then: wait ~1 min, open the printed http://localhost:PORT URL in your
# browser. Ctrl+C closes the viewer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PUSH_ARGS=()
ENVS=1
TASK=""
CKPT_ARG=""
DEVICE=""
DRY=0
INFO=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task)  TASK="$2"; shift 2 ;;
    --envs)  ENVS="$2"; shift 2 ;;
    --ckpt)  CKPT_ARG="$2"; shift 2 ;;
    --cpu)   DEVICE=cpu; shift ;;
    --gpu)   DEVICE=cuda:0; shift ;;
    --dry)   DRY=1; shift ;;
    --info)  INFO=1; shift ;;
    --push)  PUSH_ARGS+=(--push "$2"); shift 2 ;;
    --push-every) PUSH_ARGS+=(--push-every "$2"); shift 2 ;;
    --no-push)   PUSH_ARGS+=(--no-push); shift ;;
    *)       EXTRA+=("$1"); shift ;;
  esac
done

# ── Checkpoint discovery: search EVERY known clone ──────────────────────────
# The user may have several clones (~/microduck_rl, /mnt/c/...Documents/...);
# the newest model_*.pt across all of them wins. play then runs from the clone
# that OWNS the checkpoint (its venv resolves the task registry).
search_roots() {
  local r
  echo "$ROOT"
  for r in "$HOME/microduck_rl" /mnt/c/Users/*/Documents/microduck_r*; do
    [ -d "$r/logs/rsl_rl" ] && echo "$r"
  done
}

find_ckpt() {
  local matches=() r f
  while IFS= read -r r; do
    while IFS= read -r f; do
      matches+=("$r/$f")
    done < <(cd "$r" && ls logs/rsl_rl/*/*/model_*.pt 2>/dev/null || true)
  done < <(search_roots)
  [ ${#matches[@]} -gt 0 ] || return 0
  ls -t "${matches[@]}" 2>/dev/null | head -1 || true
}

if [ -n "$CKPT_ARG" ]; then
  CKPT="$(cd "$(dirname "$CKPT_ARG")" && pwd)/$(basename "$CKPT_ARG")"
else
  CKPT="$(find_ckpt)"
fi

# Reposition to the clone that owns the checkpoint (its venv runs play).
if [ -n "${CKPT:-}" ]; then
  RROOT="$(cd "$(dirname "$(dirname "$(dirname "$CKPT")")")" 2>/dev/null && pwd || true)"
  if [ -n "${RROOT:-}" ] && [ -d "$RROOT" ]; then cd "$RROOT"; fi
fi

# ── Diagnostics mode ────────────────────────────────────────────────────────
if [ "$INFO" = 1 ]; then
  echo "searched:"
  search_roots | sed 's/^/  /'
  echo "ckpt:     ${CKPT:-<none found>}"
  echo "repo:     $(pwd)"
  echo "training: $(pgrep -af 'bin/train' || echo 'not running')"
  echo "cuda:     $(uv run python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
  echo "run without --info to launch the viewer"
  exit 0
fi

if [ -z "${CKPT:-}" ]; then
  echo "No model_*.pt found in any known clone — train something first,"
  echo "or pass one explicitly:  bash scripts/watch.sh --ckpt /path/to/model_XXXX.pt"
  exit 1
fi
echo "Newest checkpoint: $CKPT"

# ── Device: GPU when free; CPU when training owns the GPU (OOM safety) ──────
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

# ── Task derivation from the experiment folder name ─────────────────────────
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
pkill -f "play_tuned.py" 2>/dev/null || true

if [ ${#PUSH_ARGS[@]} -gt 0 ]; then
  # Push-controlled playback (own script: mjlab's play has no env overrides).
  CMD=(uv run scripts/play_tuned.py --task "$TASK" --checkpoint-file "$CKPT"
       --envs "$ENVS" --device "$DEVICE" "${PUSH_ARGS[@]}"
       ${EXTRA[@]+"${EXTRA[@]}"})
else
  CMD=(uv run play "$TASK" --checkpoint-file "$CKPT" --num-envs "$ENVS"
       --viewer viser --device "$DEVICE" "${EXTRA[@]+"${EXTRA[@]}"}")
fi
echo "Running: ${CMD[*]}"
if [ "$DRY" = 1 ]; then exit 0; fi
exec "${CMD[@]}"
