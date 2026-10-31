# Windows/WSL2 setup & troubleshooting for microduck_rl

Everything around the repo that isn't the repo: WSL basics, the one-command
launcher, and a troubleshooting table built from every failure actually hit on
this setup. For training/viewing commands see
[hd1910_training_quickstart.md](hd1910_training_quickstart.md).

## One-time setup

1. PowerShell (admin): `wsl --install -d Ubuntu-24.04`, install the NVIDIA
   Game Driver on the **Windows** side only, reboot, launch "Ubuntu 24.04 LTS"
   from the Start menu, create the UNIX username/password.
2. Inside Ubuntu:
   ```bash
   sudo apt install -y curl gh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   exec $SHELL -l                 # or open a new terminal
   gh auth login                  # only if the repo is private
   git clone https://github.com/aaabattery650/microduck_rl ~/microduck_rl
   cd ~/microduck_rl && export UV_HTTP_TIMEOUT=600 && uv sync
   ```
3. Windows Settings → System → Power → **Sleep = Never** when plugged in
   (sleep suspends WSL and freezes training; screen-off is fine).
4. Make the launcher (once):
   ```bash
   cat > ~/duck << 'EOF'
   #!/usr/bin/env bash
   REPO="${DUCK_REPO:-$HOME/microduck_rl}"
   exec bash "$REPO/scripts/watch.sh" "$@"
   EOF
   chmod +x ~/duck
   ```
   Now `~/duck`, `~/duck --gpu`, `~/duck --envs 4` work from any terminal.

> Prefer the repo in `~/microduck_rl` (Linux filesystem). A clone on
> `/mnt/c/...` works but is much slower (I/O through the Windows bridge) and
> stretches every load/compile. Move with `mv /mnt/c/Users/<you>/Documents/microduck_rl ~/`.

## Daily driving

- New terminals open in `~` (home). `~/duck` works anywhere; repo commands
  need `cd ~/microduck_rl` first.
- Windows files from Ubuntu: `/mnt/c/Users/<you>/Desktop` (Desktop may live
  under `OneDrive/`). Ubuntu files from Windows: `\\wsl$\Ubuntu-24.04\home\...`.
- There is no pip and it's not needed: `uv sync` builds everything,
  `uv run <cmd>` runs inside it.
- Copy/paste in the terminal: `Ctrl+Shift+C/V` (plain Ctrl+C kills a program).

## Watching from anywhere on your LAN (phone, laptop)

All the web UIs (training dashboard :8765, reward lab :8766, lab 3D :8767,
policy viewer :8080) listen on all interfaces — but WSL2 sits behind Windows'
NAT, so other devices need a one-time port-forward. In **admin PowerShell**,
from the repo's Windows checkout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\wsl_lan_access.ps1
```

It prints the LAN URLs (e.g. `http://192.168.1.42:8765`) — open them on any
device on the same network. **Rerun after each PC reboot** (WSL's internal IP
changes). Alternative for Windows 11 22H2+: `networkingMode=mirrored` in
`%USERPROFILE%\.wslconfig` makes WSL share the PC's interfaces directly, then
no forwarding is needed at all (then `wsl --shutdown` and reopen Ubuntu).

## Troubleshooting — every error seen on this machine

| Symptom | Cause | Fix |
|---|---|---|
| `uv: command not found` | fresh terminal without uv's path | `source ~/.local/bin/env` or new terminal |
| `bash: scripts/watch.sh: No such file or directory` | not inside the repo (new terminal opens in `~`) or clone not updated | `cd ~/microduck_rl && git pull`, or just `~/duck` |
| `git clone`: `Permission denied (publickey)` | SSH URL | use the HTTPS URL |
| `git clone`: `could not read Username` / 403 | private repo, not logged in | `gh auth login` |
| Terminal 2 (monitor) spams `ValueError: invalid literal for int()... '?t='` | old train_monitor before the fix | `git pull`, restart monitor |
| Viewer (native window) ~4 fps | WSL software OpenGL | always `--viewer viser` (browser) — `~/duck` does this |
| Viewer in browser runs in slow motion (~5 fps) | viewer on CPU — either training owns the GPU (deliberate fallback) or auto-detect failed | `~/duck --gpu` when no training is running; check `pgrep -af "bin/train"` |
| Viewer crashes with CUDA error while training runs | VRAM contention | that's why the auto fallback exists: view on CPU (`~/duck`) or wait; `--gpu` risks only the viewer, never training |
| Nothing happens for ~1–2 min after launching the viewer | normal: warp kernel compile (first time) + model load + checkpoint | first run is the slowest; later runs <1 min. Terminal still printing lines = healthy |
| Browser didn't open automatically | WSL can't always launch the Windows browser | open the printed `http://localhost:PORT` URL by hand |
| Training died overnight / no progress | PC slept | Sleep = Never (see setup); resume with `--agent.load-checkpoint model_XXXX.pt --agent.resume True` |
| `No checkpoints under .../logs` | wrong clone (old `/mnt/c` copy) or training never ran there | `find ~/microduck_rl -name "model_*.pt"` to locate them |
