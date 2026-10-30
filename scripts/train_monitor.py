#!/usr/bin/env python
"""Microduck training monitor — a small live dashboard for a running training.

Opens in your browser: live graphs (reward, episode length, falls, per-term
rewards/penalties) plus a "Watch robot" button that launches the interactive
3D viewer on the LATEST checkpoint (the real `play` command → the fully
accurate model: HD-1910 actuator physics, housings, beak, spacer mass).

No GUI toolkit needed — it's a local web page (matplotlib renders the charts
headlessly). Works on WSL2, Linux, macOS.

Usage — terminal 1 (training, output teed to a file):

    uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096 \
        2>&1 | tee /tmp/train_stand.log

Usage — terminal 2 (this dashboard):

    uv run scripts/train_monitor.py --task Mjlab-Stand-Flat-MicroDuck \
        --log /tmp/train_stand.log

    (both flags optional: without --log it watches the newest
    /tmp/train_*.log; --port defaults to 8765)
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

FLOAT = r"([-\d.eE+]+)"
PATTERNS = {
    "iteration": re.compile(r"Learning iteration\s+" + FLOAT),
    "reward": re.compile(r"Mean reward:\s*" + FLOAT),
    "ep_len": re.compile(r"Mean episode length:\s*" + FLOAT),
    "term": re.compile(r"Episode_Termination/(\w+):\s*" + FLOAT),
    "rew": re.compile(r"Episode_Reward/([\w]+):\s*" + FLOAT),
}
POSITIVE_TERMS = ("upright", "stand_height", "pose", "head_pose_tracking")
PENALTY_TERMS = ("action_rate_l2", "body_ang_vel", "self_collisions")

DATA: dict[str, list] = {}
LOCK = threading.Lock()
LAST_MESSAGE = "waiting for metrics…"
TASK = "Mjlab-Stand-Flat-MicroDuck"
LOG_PATH: Path | None = None


def parse_line(line: str, data: dict) -> None:
    if m := PATTERNS["iteration"].search(line):
        data.setdefault("iteration", []).append(int(float(m.group(1))))
    if m := PATTERNS["reward"].search(line):
        data.setdefault("reward", []).append(float(m.group(1)))
    if m := PATTERNS["ep_len"].search(line):
        data.setdefault("ep_len", []).append(float(m.group(1)))
    if m := PATTERNS["term"].search(line):
        data.setdefault(f"term/{m.group(1)}", []).append(float(m.group(2)))
    if m := PATTERNS["rew"].search(line):
        data.setdefault(f"rew/{m.group(1)}", []).append(float(m.group(2)))


def newest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    return Path(hits[-1]) if hits else None


def tail_loop():
    global LAST_MESSAGE
    handle = None
    while True:
        try:
            if handle is None and LOG_PATH is not None and LOG_PATH.exists():
                handle = LOG_PATH.open("r", errors="replace")
            if handle is not None:
                lines = handle.readlines()
                with LOCK:
                    for line in lines:
                        parse_line(line, DATA)
                    n = len(DATA.get("reward", []))
                    if n:
                        it = DATA.get("iteration", [])
                        LAST_MESSAGE = (
                            f"iter {it[-1] if it else n}   "
                            f"reward {DATA['reward'][-1]:+.3f}   "
                            f"ep_len {DATA.get('ep_len', [float('nan')])[-1]:.1f}   "
                            f"fell_over {DATA.get('term/fell_over', [float('nan')])[-1]:.2f}"
                        )
        except OSError:
            pass
        time.sleep(2.0)


def chart_png(panel: int) -> bytes:
    with LOCK:
        data = {k: list(v) for k, v in DATA.items()}
    it = data.get("iteration", [])
    n = len(data.get("reward", []))
    if n and len(it) < n:
        it = list(range(1, n + 1))

    panels = [
        ("Mean reward", [("reward", "C0")]),
        ("Episode length / fell_over", [("ep_len", "C2"), ("term/fell_over", "C3")]),
        ("Task rewards (weighted, per-term)",
         [(f"rew/{t}", None) for t in POSITIVE_TERMS if f"rew/{t}" in data]),
        ("Penalties (must stay ≤ 0)",
         [(f"rew/{t}", None) for t in PENALTY_TERMS if f"rew/{t}" in data]),
    ]
    title, series = panels[panel]
    fig = Figure(figsize=(6.2, 3.1), dpi=110)
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    for key, color in series:
        ys = data.get(key)
        if not ys:
            continue
        xs = it[-len(ys):] if len(it) == len(ys) else list(range(1, len(ys) + 1))
        ax.plot(xs, ys, label=key.split("/")[-1], color=color, linewidth=1.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Microduck training monitor</title>
<style>
 body{{font-family: system-ui, sans-serif; margin: 24px; background: #fafafa}}
 h1{{font-size: 20px; margin: 0 0 4px}}
 #status{{color: #444; font-size: 15px; margin-bottom: 14px}}
 button{{font-size: 15px; padding: 8px 18px; cursor: pointer}}
 #msg{{color: #080; margin-left: 12px}}
 table{{border-spacing: 14px}}
 img{{background: #fff; border: 1px solid #ddd; border-radius: 6px}}
</style></head><body>
<h1>Microduck training</h1>
<div id="status">{status}</div>
<button onclick="fetch('/watch').then(r=>r.text()).then(t=>document.getElementById('msg').textContent=t)">
 &#9654; Watch robot (latest checkpoint)</button><span id="msg"></span>
<table><tr>
 <td><img src="/chart/0" width="640"></td><td><img src="/chart/1" width="640"></td>
</tr><tr>
 <td><img src="/chart/2" width="640"></td><td><img src="/chart/3" width="640"></td>
</tr></table>
<div style="color:#777;font-size:12px">log: {log} &nbsp;|&nbsp; viewer task: {task}
 &nbsp;|&nbsp; charts auto-refresh every 5 s</div>
<script>setInterval(()=>{{for(const im of document.images) im.src=im.src.split('?')[0]+'?t='+Date.now();
 fetch('/status').then(r=>r.text()).then(t=>document.getElementById('status').textContent=t);}},5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request noise
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global LAST_MESSAGE
        if self.path.startswith("/chart/"):
            self._send(chart_png(int(self.path.rsplit("/", 1)[1])), "image/png")
        elif self.path == "/status":
            self._send(LAST_MESSAGE.encode(), "text/plain")
        elif self.path == "/watch":
            ckpt = newest(str(REPO_ROOT / "logs" / "rsl_rl" / "*" / "*" / "model_*.pt"))
            if ckpt is None:
                self._send(b"no checkpoint yet - train a little longer", "text/plain")
                return
            subprocess.Popen(
                [
                    "uv", "run", "play", TASK,
                    "--checkpoint-file", str(ckpt),
                    "--num-envs", "1",
                    # Browser-based viewer: WSL's translated OpenGL is often
                    # software-rendered (~4 fps); viser renders in the browser.
                    "--viewer", "viser",
                ],
                cwd=str(REPO_ROOT),
            )
            self._send(f"opening viewer on {ckpt.name} ...".encode(), "text/plain")
        else:
            body = PAGE.format(status=LAST_MESSAGE, log=LOG_PATH, task=TASK).encode()
            self._send(body, "text/html")


def main() -> int:
    global LOG_PATH, TASK
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, default=None, help="training tee log (default: newest /tmp/train_*.log)")
    p.add_argument("--task", default=TASK, help="task id the Watch button plays")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    LOG_PATH = args.log or newest("/tmp/train_*.log")
    TASK = args.task
    if LOG_PATH is None:
        print("No log file found. Run training with:  ... 2>&1 | tee /tmp/train_stand.log")
        return 1

    threading.Thread(target=tail_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"[train_monitor] {url}   (watching {LOG_PATH}; Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
