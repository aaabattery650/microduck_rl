#!/usr/bin/env python
"""Reward Lab — a browser UI for shaping the Stand (balance) task's rewards.

    uv run scripts/reward_lab.py            # → http://localhost:8766

What you get:
  - Sliders for every reward parameter (tilt tolerance, stand-height gate,
    per-joint-group pose bands, weights).
  - LIVE curves: reward vs tilt, reward vs trunk height, reward vs joint
    error — updated as you drag, so you see what each number means.
  - A joint chart: all 14 servos' XML ranges, HOME angles, and your chosen
    pose band (HOME ± std) per joint group.
  - Apply → writes configs/stand_rewards.json, which the Stand task cfg
    loads on every build. Retrain to feel the difference.
  - A sign guard: penalty terms refuse positive weights (the classic
    reward-hacking footgun — see AGENTS.md).

The policy you already trained is unaffected until you retrain.
"""

from __future__ import annotations

import argparse
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import io  # noqa: E402

import math  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES = REPO_ROOT / "configs" / "stand_rewards.json"

# ── Current defaults + editability, straight from the real cfg ───────────────
from mjlab_microduck.tasks.microduck_stand_env_cfg import (  # noqa: E402
    make_microduck_stand_env_cfg,
)

CFG = make_microduck_stand_env_cfg()

PENALTY_TERMS = {"body_ang_vel", "action_rate_l2", "self_collisions", "dof_pos_limits"}
ZERO_OK_TERMS = {"body_pose_tracking", "head_pose_bias"}

SLIDERS = {
    "upright": [
        ("weight", 0.0, 6.0, CFG.rewards["upright"].weight, "× overall pull toward level"),
        ("std", 0.05, 0.6, CFG.rewards["upright"].params["std"], "tilt tolerance (rad; smaller = stricter)"),
    ],
    "stand_height": [
        ("weight", 0.0, 6.0, CFG.rewards["stand_height"].weight, "× upright-at-height pull"),
        ("std", 0.05, 0.6, CFG.rewards["stand_height"].params["std"], "tilt tolerance (rad)"),
        ("height_high", 0.10, 0.14, CFG.rewards["stand_height"].params["height_high"], "full reward from this trunk z (m)"),
        ("height_low", 0.05, 0.11, CFG.rewards["stand_height"].params["height_low"], "no reward below this trunk z (m)"),
    ],
    "pose": [
        ("weight", 0.0, 6.0, CFG.rewards["pose"].weight, "× hold the HOME joint pose"),
    ],
    "head_pose_tracking": [
        ("weight", 0.0, 6.0, CFG.rewards["head_pose_tracking"].weight, "× follow head commands"),
    ],
    "body_ang_vel": [("weight", -2.0, -0.001, CFG.rewards["body_ang_vel"].weight, "penalty: trunk angular velocity")],
    "action_rate_l2": [("weight", -2.0, -0.001, CFG.rewards["action_rate_l2"].weight, "penalty: joint-speed smoothness")],
    "self_collisions": [("weight", -4.0, -0.001, CFG.rewards["self_collisions"].weight, "penalty: parts touching")],
    "dof_pos_limits": [("weight", -4.0, -0.001, CFG.rewards["dof_pos_limits"].weight, "penalty: near joint limits")],
}

POSE_GROUPS = list(CFG.rewards["pose"].params["std_standing"].keys())
POSE_GROUP_DESC = {
    r".*hip_yaw.*": "hip yaw", r".*hip_roll.*": "hip roll", r".*hip_pitch.*": "hip pitch",
    r".*knee.*": "knee", r".*ankle.*": "ankle",
}

# Joints: name → (range_min, range_max, home, group) from the real model.
import mujoco  # noqa: E402
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_walk_spec  # noqa: E402

JOINTS: dict[str, tuple[float, float, float, str]] = {}
_spec = get_walk_spec()
for _j in _spec.joints:
    if not _j.name or _j.type == mujoco.mjtJoint.mjJNT_FREE or _j.name.startswith("passive_"):
        continue
    home = 0.0
    for _pat, _val in HOME_FRAME.joint_pos.items():
        if re.fullmatch(_pat, _j.name):
            home = _val
            break
    grp = next((g for g in POSE_GROUPS if re.fullmatch(g, _j.name)), "")
    JOINTS[_j.name] = (float(_j.range[0]), float(_j.range[1]), home, grp)
SETTLED_Z = 0.1164  # measured stand height (see stand cfg)


# ── curves ───────────────────────────────────────────────────────────────────
def curve_tilt(std: float, weight: float) -> bytes:
    t = np.linspace(0, 0.6, 200)
    r = np.exp(-np.square(t) / std**2) * weight
    fig = Figure(figsize=(5.4, 2.7), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(np.degrees(t), r, "C0")
    for deg, lbl in ((5, "5°"), (10, "10°"), (20, "20°")):
        ax.axvline(deg, color="gray", ls=":", lw=0.8)
        ax.annotate(f"{lbl}: {math.exp(-math.radians(deg)**2/std**2)*weight:.2f}", (deg, 0.02),
                    fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("trunk tilt (deg)"); ax.set_ylabel("reward / step")
    ax.set_title(f"upright reward — std={std:.3f} rad, weight={weight:.2f}", fontsize=9)
    ax.grid(alpha=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); return buf.getvalue()


def curve_height(std: float, weight: float, lo: float, hi: float) -> bytes:
    z = np.linspace(0.05, 0.15, 250)
    tilt = 0.03  # assume small residual tilt
    up = np.exp(-(tilt**2) / std**2)
    s = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    gate = s * s * (3 - 2 * s)
    r = up * gate * weight
    fig = Figure(figsize=(5.4, 2.7), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(z, r, "C2", label="reward")
    ax.plot(z, gate * weight, "C1", ls="--", lw=0.9, label="height gate")
    ax.axvline(SETTLED_Z, color="gray", ls=":", lw=1)
    ax.annotate("settled z=0.116", (SETTLED_Z, 0.02), fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("trunk height z (m)"); ax.set_ylabel("reward / step")
    ax.set_title(f"stand_height — gate {lo:.3f}→{hi:.3f} m, weight={weight:.2f}", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); return buf.getvalue()


def curve_pose(std: float) -> bytes:
    e = np.linspace(0, 0.6, 200)
    r = np.exp(-np.square(e) / std**2)
    fig = Figure(figsize=(5.4, 2.7), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(np.degrees(e), r, "C0")
    ax.axvline(math.degrees(std), color="gray", ls=":", lw=1)
    ax.annotate(f"±1σ = ±{math.degrees(std):.1f}°", (math.degrees(std), 0.4), fontsize=8, rotation=90)
    ax.set_xlabel("joint deviation from target (deg)"); ax.set_ylabel("reward (per joint)")
    ax.set_title(f"pose reward shape — std={std:.3f} rad", fontsize=9)
    ax.grid(alpha=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); return buf.getvalue()


def joint_chart(stds: dict[str, float]) -> bytes:
    fig = Figure(figsize=(7.6, 4.6), dpi=110)
    ax = fig.add_subplot(111)
    names = [n for n in JOINTS if JOINTS[n][3]]
    for i, n in enumerate(sorted(names)):
        lo, hi, home, grp = JOINTS[n]
        std = stds.get(grp, 0.1)
        ax.barh(i, hi - lo, left=lo, color="#ddd", height=0.6)
        ax.barh(i, 2 * std, left=home - std, color="#7fc97f", height=0.6,
                label="pose band ±std" if i == 0 else None)
        ax.plot(home, i, "k|", ms=12, label="HOME" if i == 0 else None)
        ax.annotate(f"{math.degrees(std):.0f}°", (hi, i), fontsize=7, va="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(sorted(names), fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("joint angle (rad)")
    ax.set_title("14 servos: XML range (grey), HOME (|), reward band (green)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="x")
    buf = io.BytesIO(); fig.savefig(buf, format="png"); return buf.getvalue()


# ── 3D duck preview (viser): optimal pose + band ghosts + height gate ────────
import threading  # noqa: E402

import viser  # noqa: E402

from mjlab_microduck.robot.microduck_constants import get_walk_spec  # noqa: E402
from mjlab_microduck.robot.microduck.make_hd1910_model import load_stl_binary  # noqa: E402

_PREVIEW: dict = {}


def _stl(path: Path):
    tris = load_stl_binary(path)
    verts = tris.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    return verts, faces


def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _init_preview(port: int):
    server = viser.ViserServer(host="0.0.0.0", port=port)
    model = get_walk_spec().compile()
    data = mujoco.MjData(model)

    # HOME qpos: free joint at settled height/level + joints at HOME angles.
    qpos = np.zeros(model.nq)
    home = {n: v for n, v in _home_pairs()}
    for name in JOINTS:
        adr = model.jnt_qposadr[model.joint(name).id]
        qpos[adr] = home.get(name, 0.0)
    qpos[2] = SETTLED_Z
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    # Visual geoms (mesh-carrying, non-collision) → viser meshes, 3 variants.
    robot_dir = Path(microduck_xml())
    variants = {"optimal": (0.25, 0.8, 1.0), "ghost_lo": (1.0, 0.55, 0.55), "ghost_hi": (0.55, 1.0, 0.55)}
    _PREVIEW["server"] = server
    _PREVIEW["model"] = model
    _PREVIEW["data"] = data
    _PREVIEW["handles"] = {k: [] for k in variants}
    _PREVIEW["mesh_cache"] = {}

    for vid, color in variants.items():
        for g in range(model.ngeom):
            mid = model.geom_dataid[g]
            if mid < 0 or model.geom_bodyid[g] == 0:
                continue
            if model.geom_contype[g] or model.geom_conaffinity[g]:
                continue  # collision geoms → skip (visual only)
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            if mesh_name not in _PREVIEW["mesh_cache"]:
                _PREVIEW["mesh_cache"][mesh_name] = _stl(robot_dir / "assets" / f"{mesh_name}.stl")
            verts, faces = _PREVIEW["mesh_cache"][mesh_name]
            h = server.scene.add_mesh_simple(f"{vid}_{g}", vertices=verts, faces=faces, color=color)
            _PREVIEW["handles"][vid].append((g, h))

    # Height-gate discs (updated on slider change).
    theta = np.linspace(0, 2 * np.pi, 40)
    ring = np.stack([0.09 * np.cos(theta), 0.09 * np.sin(theta), np.zeros(40)], axis=1)
    disc_v = np.concatenate([np.zeros((1, 3)), ring])
    disc_f = np.array([[0, i + 1, i + 2] for i in range(39)], dtype=np.int32)
    _PREVIEW["gate"] = {
        "gate_low": server.scene.add_mesh_simple("gate_low", vertices=disc_v, faces=disc_f, color=(1.0, 0.6, 0.2)),
        "gate_high": server.scene.add_mesh_simple("gate_high", vertices=disc_v, faces=disc_f, color=(0.2, 0.9, 0.3)),
    }
    _update_preview({"height_low": 0.08, "height_high": 0.114})


def microduck_xml() -> str:
    from mjlab_microduck.robot import microduck_constants as C
    return str(C.MICRODUCK_WALK_XML.parent)


def _home_pairs():
    for pat, val in HOME_FRAME.joint_pos.items():
        for name in JOINTS:
            if re.fullmatch(pat, name):
                yield name, val


def _pose_for_offsets(leg_offsets) -> np.ndarray:
    model, data = _PREVIEW["model"], _PREVIEW["data"]
    qpos = np.zeros(model.nq)
    for name, (lo, hi, home, grp) in JOINTS.items():
        adr = model.jnt_qposadr[model.joint(name).id]
        qpos[adr] = home + (leg_offsets.get(grp, 0.0) if grp else 0.0)
    qpos[2] = SETTLED_Z
    return qpos


def _apply_pose(variant: str, qpos: np.ndarray):
    model, data = _PREVIEW["model"], _PREVIEW["data"]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for g, h in _PREVIEW["handles"][variant]:
        bid = model.geom_bodyid[g]
        bw, gw = data.xquat[bid], model.geom_quat[g]
        w = _quat_mul([bw[0], bw[1], bw[2], bw[3]], [gw[0], gw[1], gw[2], gw[3]])
        p = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ model.geom_pos[g]
        h.position = tuple(p)
        h.wxyz = tuple(w)


def _update_preview(params: dict):
    if not _PREVIEW:
        return
    stds = params.get("pose_std_standing", {})
    lo = params.get("height_low", 0.08)
    hi = params.get("height_high", 0.114)
    _apply_pose("optimal", _pose_for_offsets({}))
    ghost_lo, ghost_hi = {}, {}
    for grp, std in stds.items():
        ghost_lo[grp], ghost_hi[grp] = -float(std), float(std)
    _apply_pose("ghost_lo", _pose_for_offsets(ghost_lo))
    _apply_pose("ghost_hi", _pose_for_offsets(ghost_hi))
    for name, z in (("gate_low", lo), ("gate_high", hi)):
        _PREVIEW["gate"][name].position = (0.0, 0.0, z)


def _f(q, k, d):
    try:
        return float(q.get(k, [d])[0])
    except (TypeError, ValueError, IndexError):
        return d


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        if path == "/curve/tilt":
            self._send(curve_tilt(_f(q, "std", 0.22), _f(q, "weight", 2.0)), "image/png")
        elif path == "/curve/height":
            self._send(curve_height(_f(q, "std", 0.22), _f(q, "weight", 1.5),
                                    _f(q, "height_low", 0.08), _f(q, "height_high", 0.114)), "image/png")
        elif path == "/curve/pose":
            self._send(curve_pose(_f(q, "std", 0.1)), "image/png")
        elif path == "/joints":
            stds = {g: _f(q, g.replace(".*", "").replace("*", ""), CFG.rewards["pose"].params["std_standing"][g])
                    for g in POSE_GROUPS}
            self._send(joint_chart(stds), "image/png")
        else:
            self._send(PAGE.encode(), "text/html")

    def do_POST(self):
        if self.path == "/preview":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            _update_preview(payload)
            up_w = payload.get("rewards", {}).get("upright", {}).get("weight", 2.0)
            sh_w = payload.get("rewards", {}).get("stand_height", {}).get("weight", 1.5)
            self._send(
                f"3D updated. Reward ceiling at the shown pose: upright {up_w:.2f} + "
                f"stand_height {sh_w:.2f} (+pose/head terms at their weights, penalties 0).".encode(),
                "text/plain",
            )
        elif self.path == "/apply":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            errors = []
            for name, params in payload.get("rewards", {}).items():
                if name in PENALTY_TERMS and params.get("weight", -1) > 0:
                    errors.append(f"{name}: penalty weights must stay NEGATIVE (got {params['weight']})")
            if errors:
                self._send(("REFUSED: " + "; ".join(errors)).encode(), "text/plain")
                return
            OVERRIDES.parent.mkdir(exist_ok=True)
            OVERRIDES.write_text(json.dumps(payload, indent=2))
            self._send(
                f"Saved {OVERRIDES.relative_to(REPO_ROOT)} — the Stand task now trains with these. "
                "Retrain:  uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096 "
                "--agent.max_iterations 1500".encode(),
                "text/plain",
            )
        else:
            self._send(b"not found", "text/plain")


def _slider_html(term: str, key: str, lo: float, hi: float, val: float, desc: str) -> str:
    step = 0.001 if hi - lo < 3 else 0.05
    return (
        f'<div class="ctl"><label><b>{term}.{key}</b> <span id="v_{term}_{key}">{val:.3f}</span>'
        f'<small> — {desc}</small></label><br>'
        f'<input type="range" id="{term}_{key}" min="{lo}" max="{hi}" step="{step}" value="{val}" '
        f'oninput="onSlide(\'{term}\',\'{key}\',this.value)"></div>'
    )


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Stand Reward Lab</title>
<style>
 body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa}
 h1{font-size:20px;margin:0 0 2px} .sub{color:#666;margin-bottom:18px}
 .cols{display:flex;gap:28px;flex-wrap:wrap}
 .ctl{margin:10px 0} label{font-size:13px} small{color:#777}
 input[type=range]{width:330px}
 img{background:#fff;border:1px solid #ddd;border-radius:6px}
 button{font-size:15px;padding:8px 22px;cursor:pointer;margin-top:14px}
 #msg{margin-left:12px;font-size:14px}
 .pen{background:#fff3f3;padding:6px 10px;border-radius:6px}
 .pos{background:#f3fff5;padding:6px 10px;border-radius:6px}
</style></head><body>
<h1>Stand Reward Lab</h1>
<div class="sub">Drag sliders — curves update live. Apply writes
<code>configs/stand_rewards.json</code>; the Stand task reads it on every build. Retrain to feel it.</div>
<div class="cols">
__SLIDERS__
 <div>
  <h3>Joint bands (pose reward std per group)</h3>
  __GROUPS__
  <img id="joints" width="760"><br>
  <img id="curve_pose" width="540">
 </div>
</div>
<h3>Reward shapes</h3>
<img id="curve_tilt" width="540"> <img id="curve_height" width="540">
<h3>3D preview — <a href="http://localhost:__VISER_PORT__" target="_blank">open the duck</a></h3>
<div class="sub">Blue duck = reward-optimal pose (level trunk, settled height, joints at HOME).
Red/green ghosts = your pose-band edges (all leg groups ±std). Orange disc = height_low
(no reward below), green disc = height_high (full reward above). They move as you drag.
Open the 3D link once, keep it in a second tab/window.</div>
<div id="preview_msg" class="sub"></div>
<br><button onclick="apply()">Apply overrides</button><span id="msg"></span>
<script>
const groups = __GROUP_JSON__;
function onSlide(term,key,val){
  document.getElementById('v_'+term+'_'+key).textContent=(+val).toFixed(3);
  refresh();
}
function refresh(){
  const g = (t,k)=>document.getElementById(t+'_'+k)?document.getElementById(t+'_'+k).value:null;
  const q=[];
  if(g('upright','std')) q.push('std='+g('upright','std'),'weight='+g('upright','weight'));
  document.getElementById('curve_tilt').src='/curve/tilt?'+q.join('&')+'&t='+Date.now();
  document.getElementById('curve_height').src='/curve/height?std='+(g('stand_height','std')||0.22)
    +'&weight='+(g('stand_height','weight')||1.5)+'&height_low='+(g('stand_height','height_low')||0.08)
    +'&height_high='+(g('stand_height','height_high')||0.114)+'&t='+Date.now();
  let pq='/curve/pose?std='+(groups[Object.keys(groups)[0]]||0.1);
  document.getElementById('curve_pose').src=pq+'&t='+Date.now();
  const jq=[]; for(const [k,v] of Object.entries(groups)){
    const el=document.getElementById('grp_'+k.replace(/[^a-z]/gi,''));
    if(el) { groups[k]=+el.value; jq.push(k.replace(/\\.\\*/g,'')+'='+el.value); }
  }
  document.getElementById('joints').src='/joints?'+jq.join('&')+'&t='+Date.now();
  // live 3D preview update
  fetch('/preview',{method:'POST',body:JSON.stringify({
    pose_std_standing:groups,
    height_low:+(g('stand_height','height_low')||0.08),
    height_high:+(g('stand_height','height_high')||0.114),
    rewards:{upright:{weight:+(g('upright','weight')||2)},
             stand_height:{weight:+(g('stand_height','weight')||1.5)}}
  })}).then(r=>r.text()).then(t=>document.getElementById('preview_msg').textContent=t);
}
function apply(){
  const out={rewards:{},pose_std_standing:{}};
  for(const el of document.querySelectorAll('input[type=range]')){
    const [term,key]=el.id.split('___') ;
  }
  // collect by known ids
  const val=(t,k)=>{const e=document.getElementById(t+'_'+k); return e?+e.value:null};
  const terms={};
  """ + "".join([]) + """
  const defs = __IDS_JSON__;
  for(const [t,keys] of Object.entries(defs)){
    terms[t]={}; for(const k of keys){ const v=val(t,k); if(v!==null) terms[t][k]=v; }
  }
  out.rewards=terms; out.pose_std_standing=groups;
  fetch('/apply',{method:'POST',body:JSON.stringify(out)})
    .then(r=>r.text()).then(t=>document.getElementById('msg').textContent=t);
}
refresh();
</script></body></html>"""


def build_page() -> str:
    ids = {}
    html = []
    for term, sliders in SLIDERS.items():
        cls = "pen" if term in PENALTY_TERMS else "pos"
        block = [f'<div class="{cls}"><h3>{term}</h3>']
        ids[term] = []
        for key, lo, hi, val, desc in sliders:
            block.append(_slider_html(term, key, lo, hi, val, desc))
            ids[term].append(key)
        html.append(" ".join(block) + "</div>")
    groups_html = []
    group_defaults = {}
    for g in POSE_GROUPS:
        gid = re.sub(r"[^a-z]", "", g.lower())
        v = CFG.rewards["pose"].params["std_standing"][g]
        group_defaults[g] = v
        groups_html.append(
            f'<div class="ctl"><label><b>{POSE_GROUP_DESC.get(g, g)}</b> '
            f'<span id="v_grp_{gid}">{v:.3f}</span><small> rad band</small></label><br>'
            f'<input type="range" id="grp_{gid}" min="0.02" max="0.5" step="0.005" value="{v}" '
            f'oninput="document.getElementById(\'v_grp_{gid}\').textContent=(+this.value).toFixed(3);refresh()"></div>'
        )
    page = PAGE.replace("__SLIDERS__", " ".join(html))
    page = page.replace("__GROUPS__", " ".join(groups_html))
    page = page.replace("__GROUP_JSON__", json.dumps(group_defaults))
    page = page.replace("__IDS_JSON__", json.dumps(ids))
    return page


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--viser-port", type=int, default=8767)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    global PAGE
    PAGE = build_page().replace("__VISER_PORT__", str(args.viser_port))
    _init_preview(args.viser_port)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    if OVERRIDES.exists():
        print(f"[reward_lab] NOTE: existing overrides loaded by the cfg: {OVERRIDES}")
    print(f"[reward_lab] {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
