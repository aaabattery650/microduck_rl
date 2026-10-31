"""train_monitor regression tests — CPU-only.

Born from a shipped bug: the dashboard's auto-refresh appends a cache-buster
(``?t=...``) to chart URLs and the server crashed parsing ``2?t=...`` as an
int — a traceback every 5 s in the user's terminal. These tests parse sample
training output AND exercise the real HTTP server with cache-busted URLs.
"""

import importlib.util
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "train_monitor", Path(__file__).resolve().parents[1] / "scripts" / "train_monitor.py"
)
tm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tm)

SAMPLE = """
                    Learning iteration 3/5000
                        Mean reward: -1.11
                    Mean episode length: 23.39
          Episode_Reward/upright: 0.0105
          Episode_Reward/stand_height: 0.0129
          Episode_Reward/action_rate_l2: -0.0664
          Episode_Termination/fell_over: 2.6250
"""


def test_parse_line_extracts_all_metric_kinds():
    data = {}
    for line in SAMPLE.splitlines():
        tm.parse_line(line, data)
    assert data["iteration"][-1] == 3
    assert data["reward"][-1] == -1.11
    assert data["ep_len"][-1] == 23.39
    assert data["rew/stand_height"][-1] == 0.0129
    assert data["rew/action_rate_l2"][-1] == -0.0664
    assert data["term/fell_over"][-1] == 2.625


def test_server_survives_cache_busted_chart_urls():
    # The regression: /chart/2?t=1793380312360 must serve, not crash.
    server = ThreadingHTTPServer(("127.0.0.1", 0), tm.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/", "/status", "/chart/0", "/chart/2?t=1793380312360", "/chart/3?x=y"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
                assert resp.status == 200, path
    finally:
        server.shutdown()
