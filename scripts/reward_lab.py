#!/usr/bin/env python
"""Reward Lab — beginner-friendly reward shaping with a live 3D duck.

    uv run scripts/reward_lab.py            # → http://localhost:8766

One page:
  ┌──────────────┬──────────────────────────┬───────────────┐
  │ WHAT TO TUNE │      THE DUCK (3D)       │  WHAT IT PAYS │
  │  IMU         │  blue  = optimal pose    │  live curves  │
  │  14 motors   │  ghosts = rewarded range │  plain ° text │
  │  stand height│  discs = height gate     │               │
  └──────────────┴──────────────────────────┴───────────────┘

Select IMU → the ghosts tilt to ±tolerance; select a motor → its group's
band shows as ghost poses and the part highlights orange; select Stand
height → the gate discs ride the duck. Apply → writes
configs/stand_rewards.json (the Stand task trains with it); penalties refuse
positive weights; the trained policy is untouched until you retrain.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib

matplotlib.use("Agg")

import math  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import viser  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES = REPO_ROOT / "configs" / "stand_rewards.json"

from mjlab_microduck.robot import microduck_constants as C  # noqa: E402
from mjlab_microduck.tasks.microduck_stand_env_cfg import make_microduck_stand_env_cfg  # noqa: E402

CFG = make_microduck_stand_env_cfg()
SETTLED_Z = 0.1164  # measured (stand cfg)

PENALTY_TERMS = {"body_ang_vel", "action_rate_l2", "self_collisions", "dof_pos_limits"}
POSE_GROUPS = list(CFG.rewards["pose"].params["std_standing"].keys())
DEFAULTS = {
    "upright_w": CFG.rewards["upright"].weight,
    "upright_std": CFG.rewards["upright"].params["std"],
    "height_w": CFG.rewards["stand_height"].weight,
    "height_low": CFG.rewards["stand_height"].params["height_low"],
    "height_high": CFG.rewards["stand_height"].params["height_high"],
    "pose_w": CFG.rewards["pose"].weight,
    "groups": dict(CFG.rewards["pose"].params["std_standing"]),
}

# Friendly names -------------------------------------------------------------
PRETTY = {"left": "Left", "right": "Right", "hip": "hip", "yaw": "yaw", "pitch": "pitch",
          "roll": "roll", "knee": "knee", "ankle": "ankle", "neck": "neck", "head": "head"}


def pretty(name: str) -> str:
    return " ".join(PRETTY.get(p, p) for p in name.split("_"))


# Joints: name -> (lo, hi, home, group) from the real model ------------------
_model = mujoco.MjSpec.from_file(str(C.MICRODUCK_WALK_XML))
JOINTS: dict[str, tuple[float, float, float, str]] = {}
_home_map = dict(C.HOME_FRAME.joint_pos)
for _j in _model.joints:
    if not _j.name or _j.type == mujoco.mjtJoint.mjJNT_FREE or _j.name.startswith("passive_"):
        continue
    home = 0.0
    for _pat, _val in _home_map.items():
        if re.fullmatch(_pat, _j.name):
            home = _val
            break
    grp = next((g for g in POSE_GROUPS if re.fullmatch(g, _j.name)), "")
    JOINTS[_j.name] = (float(_j.range[0]), float(_j.range[1]), home, grp)

GHOSTS = ["g1", "g2", "g3", "g4"]
GHOST_COLORS = {"g1": (1.0, 0.45, 0.45), "g2": (0.45, 1.0, 0.45),
                "g3": (1.0, 0.8, 0.3), "g4": (0.55, 0.7, 1.0)}
PREVIEW: dict = {}


# ═══ 3D scene ════════════════════════════════════════════════════════════════
def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def _axis_quat(ax, ang):
    return np.array([math.cos(ang / 2), *(math.sin(ang / 2) * np.array(ax))])


def _mesh_buffers(model, mid: int):
    """Compiled mesh vertices/faces — already in MuJoCo's canonical mesh
    frame, consistent with geom_pos/geom_quat. (Raw STL files are NOT: the
    compiler re-centers meshes via mesh_pos/mesh_quat, so placing raw
    vertices scatters the robot.)"""
    vadr, vnum = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = np.array(model.mesh_vert[vadr : vadr + vnum], dtype=np.float32)
    fadr, fnum = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    faces = np.array(model.mesh_face[fadr : fadr + fnum], dtype=np.int32)
    return verts, faces


def init_preview(port: int):
    server = viser.ViserServer(host="0.0.0.0", port=port)
    model = mujoco.MjSpec.from_file(str(C.MICRODUCK_WALK_XML)).compile()
    data = mujoco.MjData(model)

    PREVIEW.update(server=server, model=model, data=data, mesh_handles=[], gate={},
                   joint_geoms={}, base_colors={}, ghost_handles={})

    variants = {"optimal": (0.35, 0.75, 1.0), **GHOST_COLORS}
    cache = {}
    for vid, color in variants.items():
        for g in range(model.ngeom):
            mid = model.geom_dataid[g]
            bid = model.geom_bodyid[g]
            if mid < 0 or bid == 0 or model.geom_contype[g] or model.geom_conaffinity[g]:
                continue
            if mid not in cache:
                cache[mid] = _mesh_buffers(model, mid)
            v, f = cache[mid]
            h = server.scene.add_mesh_simple(f"{vid}_{g}", vertices=v, faces=f, color=color)
            if vid == "optimal":
                PREVIEW["mesh_handles"].append((g, h))
                PREVIEW["base_colors"][g] = color
            else:
                PREVIEW["ghost_handles"].setdefault(vid, []).append((g, h))

    # joint -> [(geom, handle)] of the optimal duck (for click-highlight)
    for jn in JOINTS:
        jbid = model.jnt_bodyid[model.joint(jn).id]
        PREVIEW["joint_geoms"][jn] = [(g, h) for g, h in PREVIEW["mesh_handles"]
                                      if model.geom_bodyid[g] == jbid]

    th = np.linspace(0, 2 * np.pi, 40)
    disc_v = np.concatenate([np.zeros((1, 3)), np.stack([0.09*np.cos(th), 0.09*np.sin(th), np.zeros(40)], 1)])
    disc_f = np.arange(40).reshape(-1, 2)
    disc_f = np.array([[0, a + 1, a + 2] for a in range(39)], dtype=np.int32)
    PREVIEW["gate"]["low"] = server.scene.add_mesh_simple(
        "gate_low", vertices=disc_v, faces=disc_f, color=(1.0, 0.6, 0.2))
    PREVIEW["gate"]["high"] = server.scene.add_mesh_simple(
        "gate_high", vertices=disc_v, faces=disc_f, color=(0.2, 0.9, 0.3))


def _qpos(joint_offsets: dict[str, float] | None = None, trunk_quat=None) -> np.ndarray:
    model = PREVIEW["model"]
    q = np.zeros(model.nq)
    for name, (_, _, home, _) in JOINTS.items():
        q[model.jnt_qposadr[model.joint(name).id]] = home + (joint_offsets or {}).get(name, 0.0)
    q[2] = SETTLED_Z
    if trunk_quat is not None:
        q[3:7] = trunk_quat
    return q


def _place(handles, qpos):
    model, data = PREVIEW["model"], PREVIEW["data"]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for g, h in handles:
        bid = model.geom_bodyid[g]
        bw, gw = data.xquat[bid], model.geom_quat[g]
        w = _quat_mul([bw[0], bw[1], bw[2], bw[3]], [gw[0], gw[1], gw[2], gw[3]])
        p = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ model.geom_pos[g]
        h.position = tuple(p)
        h.wxyz = tuple(w)


def _place_ghosts(vid, qpos):
    handles = PREVIEW.get("ghost_handles", {}).get(vid, [])
    if not handles:
        return
    model, data = PREVIEW["model"], PREVIEW["data"]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for g, h in handles:
        bid = model.geom_bodyid[g]
        bw, gw = data.xquat[bid], model.geom_quat[g]
        w = _quat_mul([bw[0], bw[1], bw[2], bw[3]], [gw[0], gw[1], gw[2], gw[3]])
        p = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ model.geom_pos[g]
        h.position = tuple(p)
        h.wxyz = tuple(w)


SELECTION = {"type": "imu", "id": None}  # type: imu | motor | height


def update_scene(params: dict):
    """Reposition everything for the current selection + slider values."""
    if not PREVIEW:
        return
    # targets ride along into every pose (blue duck = the rewarded pose)
    targets = {k: float(v) for k, v in params.get("targets", {}).items()}
    _place(PREVIEW["mesh_handles"], _qpos(targets))
    sel = SELECTION
    groups = params.get("groups", DEFAULTS["groups"])
    up_std = params.get("upright_std", DEFAULTS["upright_std"])
    # highlight selected joint orange, others back to base (never compare
    # viser handles — their __eq__ broadcasts vertex arrays)
    sel_joint = sel["id"] if sel["type"] == "motor" else None
    for jn, pairs in PREVIEW["joint_geoms"].items():
        for g, h in pairs:
            try:
                h.color = (1.0, 0.55, 0.1) if jn == sel_joint else PREVIEW["base_colors"][g]
            except Exception:
                pass

    def qpos_for(joint_offsets=None, trunk_quat=None):
        return _qpos(joint_offsets, trunk_quat)

    if sel["type"] == "motor":
        jname = sel["id"]
        grp = JOINTS.get(jname, (0, 0, 0, ""))[3]
        std = float(groups.get(grp, 0.1))
        base = dict(targets)
        for vid, sgn in (("g1", -1.0), ("g2", 1.0)):
            _place_ghosts(vid, _qpos({**base, jname: base.get(jname, 0.0) + sgn * std}))
    elif sel["type"] == "imu":
        a = float(up_std)
        _place_ghosts("g1", qpos_for(dict(targets), trunk_quat=_axis_quat((1, 0, 0), a)))
        _place_ghosts("g2", qpos_for(dict(targets), trunk_quat=_axis_quat((1, 0, 0), -a)))
        _place_ghosts("g3", qpos_for(dict(targets), trunk_quat=_axis_quat((0, 1, 0), a)))
        _place_ghosts("g4", qpos_for(dict(targets), trunk_quat=_axis_quat((0, 1, 0), -a)))
    # height selection: ghosts parked; discs always track sliders
    PREVIEW["gate"]["low"].position = (0, 0, params.get("height_low", DEFAULTS["height_low"]))
    PREVIEW["gate"]["high"].position = (0, 0, params.get("height_high", DEFAULTS["height_high"]))


# ═══ curves ═════════════════════════════════════════════════════════════════
def png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def curve_tilt(std, weight) -> bytes:
    t = np.linspace(0, 0.6, 200)
    fig = Figure(figsize=(5.2, 2.6), dpi=105)
    ax = fig.add_subplot(111)
    ax.plot(np.degrees(t), np.exp(-np.square(t) / std**2) * weight, "C0")
    for deg in (5, 10, 20):
        ax.axvline(deg, color="gray", ls=":", lw=0.8)
        ax.annotate(f"{deg}°→{math.exp(-math.radians(deg)**2/std**2)*weight:.2f}",
                    (deg, 0.03), fontsize=7, rotation=90)
    ax.set_xlabel("lean (degrees)"); ax.set_ylabel("reward / step")
    ax.set_title(f"IMU balance: fully rewarded at 0° lean, ±{math.degrees(std):.0f}° tolerance", fontsize=9)
    ax.grid(alpha=0.3)
    return png(fig)


def curve_height(w, lo, hi) -> bytes:
    z = np.linspace(0.05, 0.15, 250)
    s = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    fig = Figure(figsize=(5.2, 2.6), dpi=105)
    ax = fig.add_subplot(111)
    ax.plot(z * 100, (s * s * (3 - 2 * s)) * w, "C2")
    ax.axvline(SETTLED_Z * 100, color="gray", ls=":", lw=1)
    ax.annotate("duck stands here", (SETTLED_Z * 100 + 0.2, 0.05), fontsize=8, rotation=90)
    ax.set_xlabel("trunk height (cm)"); ax.set_ylabel("reward / step")
    ax.set_title(f"Stand height: 0 below {lo*100:.0f} cm, full above {hi*100:.0f} cm", fontsize=9)
    ax.grid(alpha=0.3)
    return png(fig)


def curve_pose(std) -> bytes:
    e = np.linspace(0, 0.6, 200)
    fig = Figure(figsize=(5.2, 2.6), dpi=105)
    ax = fig.add_subplot(111)
    ax.plot(np.degrees(e), np.exp(-np.square(e) / std**2), "C0")
    ax.set_xlabel("how far the motor is from its home angle (degrees)")
    ax.set_ylabel("reward")
    ax.set_title(f"Motor band: ±{math.degrees(std):.0f}° from home keeps most of the reward", fontsize=9)
    ax.grid(alpha=0.3)
    return png(fig)


# ═══ page ═══════════════════════════════════════════════════════════════════
MOTORS_JS = json.dumps(
    [{"id": n, "label": pretty(n), "group": JOINTS[n][3],
      "lo": JOINTS[n][0], "hi": JOINTS[n][1], "home": JOINTS[n][2]} for n in sorted(JOINTS)]
)
GROUPS_JS = json.dumps(DEFAULTS["groups"])
DEFAULTS_JS = json.dumps(DEFAULTS)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Duck Reward Lab</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#222}
 header{background:#24344d;color:#fff;padding:12px 22px;font-size:18px}
 header small{opacity:.75;margin-left:12px}
 .layout{display:flex;gap:14px;padding:14px;height:calc(100vh - 60px);box-sizing:border-box}
 .col{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px;overflow:auto}
 .left{width:250px}.mid{flex:1.4;min-width:340px}.right{flex:1;min-width:330px}
 h3{margin:8px 0;font-size:14px;text-transform:uppercase;color:#558}
 .item{display:block;width:100%;text-align:left;margin:3px 0;padding:7px 10px;font-size:14px;
       border:1px solid #ddd;border-radius:8px;background:#fafafa;cursor:pointer}
 .item.sel{background:#ffb45c;border-color:#e07b00;font-weight:600}
 iframe{width:100%;height:82%;border:0;border-radius:8px}
 .val{font-size:22px;font-weight:700;color:#0a6c2e}
 .slide{margin:8px 0}
 input[type=range]{width:100%}
 .note{font-size:12.5px;color:#555;margin:4px 0 10px}
 button.apply{font-size:15px;padding:9px 20px;background:#2e7d32;color:#fff;border:0;
              border-radius:8px;cursor:pointer}
 #msg{font-size:13px;margin-left:10px}
 img{width:100%;background:#fff;border:1px solid #eee;border-radius:6px;margin-top:6px}
</style></head><body>
<header>🦆 Duck Reward Lab <small>pick a part → see what it's rewarded for → tune → Apply → retrain</small></header>
<div class="layout">
 <div class="col left"><h3>What do you want to tune?</h3>
  <button class="item sel" id="sel_imu"   onclick="select('imu')">🧭 IMU (balance)</button>
  <button class="item"      id="sel_height" onclick="select('height')">📏 Stand height</button>
  <h3>Motors (click one)</h3>
  <div id="motors"></div>
 </div>
 <div class="col mid"><h3>The duck</h3>
  <iframe id="duck3d"></iframe>
  <div class="note" id="scene_note"></div>
 </div>
 <div class="col right"><h3 id="rtitle"></h3>
  <div id="rbody"></div>
  <img id="curve">
  <h3>All weights (advanced)</h3>
  <div id="toggles"></div>
  <div class="slide">upright weight <input type="range" id="upright_w" min="0" max="6" step="0.05" value="__UPW__" oninput="push()"></div>
  <div class="slide">stand-height weight <input type="range" id="height_w" min="0" max="6" step="0.05" value="__HW__" oninput="push()"></div>
  <div class="slide">pose weight <input type="range" id="pose_w" min="0" max="6" step="0.05" value="__PW__" oninput="push()"></div>
  <br><button class="apply" onclick="apply()">Apply overrides</button><span id="msg"></span>
 </div>
</div>
<script>
const MOTORS=__MOTORS__, GROUPS=__GROUPS__, DEF=__DEFAULTS__;
let sel={type:'imu',id:null};
let groups={...GROUPS};
let targets={};          // joint name -> rad offset from home
let disabled=new Set();
document.getElementById('motors').innerHTML = MOTORS.map(m=>
 `<button class="item" id="sel_${m.id}" onclick="select('motor','${m.id}')">${m.label}</button>`).join('');
document.getElementById('duck3d').src = 'http://'+location.hostname+':__VISER_PORT__';
document.getElementById('toggles').innerHTML = ['upright','stand_height','pose','head_pose_tracking',
  'body_ang_vel','action_rate_l2','self_collisions','dof_pos_limits'].map(t=>
 `<label style="font-size:13px;display:inline-block;margin:2px 8px"><input type="checkbox" checked
   onchange="toggle('${t}',this)"> ${t}</label>`).join('');
function select(type,id=null){
  sel={type,id};
  for(const el of document.querySelectorAll('.left .item')) el.classList.remove('sel');
  document.getElementById('sel_'+(id||type)).classList.add('sel');
  render();
}
function render(){
  const R=document.getElementById('rbody'), T=document.getElementById('rtitle'), N=document.getElementById('scene_note');
  if(sel.type==='imu'){
    T.textContent='IMU — how level must the duck stay?';
    const std=DEF.upright_std;
    R.innerHTML=`<div class="val">±${(std*180/Math.PI).toFixed(0)}° tolerance</div>
     <div class="slide">tolerance <input type="range" id="upright_std" min="0.05" max="0.6" step="0.005"
       value="${std}" oninput="DEF.upright_std=+this.value;push()"></div>
     <div class="note">Red/yellow ghosts = the duck leaning to the tolerance limit. Smaller = stricter balance.</div>`;
    document.getElementById('curve').src='/curve/tilt?std='+DEF.upright_std+'&weight='+(+document.getElementById('upright_w').value)+'&t='+Date.now();
    N.textContent='Ghosts: trunk pitched/rolled to ±tolerance.';
  } else if(sel.type==='height'){
    T.textContent='Stand height — how tall must it stand?';
    R.innerHTML=`<div class="val">${(DEF.height_high*100).toFixed(1)} cm goal</div>
     <div class="slide">full reward above (cm) <input type="range" id="height_high" min="10" max="13.5" step="0.1"
       value="${DEF.height_high*100}" oninput="DEF.height_high=this.value/100;push()"></div>
     <div class="slide">no reward below (cm) <input type="range" id="height_low" min="5" max="11" step="0.1"
       value="${DEF.height_low*100}" oninput="DEF.height_low=this.value/100;push()"></div>
     <div class="note">Discs on the duck: orange = no reward below, green = full reward above.</div>`;
    document.getElementById('curve').src='/curve/height?weight='+(+document.getElementById('height_w').value)
      +'&height_low='+DEF.height_low+'&height_high='+DEF.height_high+'&t='+Date.now();
    N.textContent='Discs track your sliders on the duck.';
  } else {
    const m=MOTORS.find(x=>x.id===sel.id); const std=groups[m.group];
    const loD=(m.lo-m.home)*180/Math.PI, hiD=(m.hi-m.home)*180/Math.PI;
    const cur=(targets[m.id]||0)*180/Math.PI;
    T.textContent=m.label;
    R.innerHTML=`<div class="val">rewarded within ±${(std*180/Math.PI).toFixed(0)}° of its target</div>
     <div class="slide"><b>Target angle: <span id="tgt_v">${cur.toFixed(0)}°</span> from home</b>
       <input type="range" id="tgt_${m.id}" min="${loD.toFixed(1)}" max="${hiD.toFixed(1)}" step="1"
       value="${cur.toFixed(1)}" oninput="targets['${m.id}']=this.value*Math.PI/180;
       document.getElementById('tgt_v').textContent=this.value+'°';push()"></div>
     <div class="slide">Band width (tolerance) <input type="range" id="g_${m.group}" min="0.02" max="0.5" step="0.005"
       value="${std}" oninput="groups['${m.group}']=+this.value;push()"></div>
     <div class="note">Blue duck now poses at YOUR target; ghosts = target ± band. Home is 0°.</div>`;
    document.getElementById('curve').src='/curve/pose?std='+std+'&t='+Date.now();
    N.textContent='Orange part = selected motor. Ghosts = band edges around the target.';
  }
  push();
}
function push(){
  fetch('/select',{method:'POST',body:JSON.stringify({sel,groups,targets,
    upright_std:DEF.upright_std,height_low:DEF.height_low,height_high:DEF.height_high})});
  render._t && clearTimeout(render._t);
  render._t=setTimeout(render,400);
}
const DISABLEABLE=['upright','stand_height','pose','head_pose_tracking','body_ang_vel','action_rate_l2','self_collisions','dof_pos_limits'];
function toggle(term,el){
  if(el.checked) disabled.delete(term); else disabled.add(term);
  push();
}
function apply(){
  const val=id=>document.getElementById(id)?+document.getElementById(id).value:null;
  const w={};
  const ws={upright:'upright_w',stand_height:'height_w',pose:'pose_w'};
  for(const [t,id] of Object.entries(ws)) w[t]={weight:disabled.has(t)?0:val(id)};
  fetch('/apply',{method:'POST',body:JSON.stringify({rewards:Object.assign(w,{
      upright:Object.assign(w.upright||{},{std:DEF.upright_std}),
      stand_height:Object.assign(w.stand_height||{},{height_low:DEF.height_low,height_high:DEF.height_high})}),
      pose_std_standing:groups, pose_targets:targets, disabled:[...disabled]})})
    .then(r=>r.text()).then(t=>document.getElementById('msg').textContent=t);
}
select('imu');
</script></body></html>"""


# ═══ server ═════════════════════════════════════════════════════════════════
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
        q = parse_qs(urlparse(self.path).query)

        def f(k, d):
            try:
                return float(q.get(k, [d])[0])
            except (TypeError, ValueError):
                return d

        if path == "/curve/tilt":
            self._send(curve_tilt(f("std", 0.22), f("weight", 2.0)), "image/png")
        elif path == "/curve/height":
            self._send(curve_height(f("weight", 1.5), f("height_low", 0.08), f("height_high", 0.114)), "image/png")
        elif path == "/curve/pose":
            self._send(curve_pose(f("std", 0.1)), "image/png")
        else:
            self._send(PAGE.encode(), "text/html")

    def do_POST(self):
        global SELECTION
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/select":
            SELECTION = {"type": payload.get("sel", {}).get("type", "imu"),
                         "id": payload.get("sel", {}).get("id")}
            update_scene(payload)
            self._send(b"ok", "text/plain")
        elif self.path == "/apply":
            errors = [f"{t}: penalty weights must stay negative"
                      for t, p in payload.get("rewards", {}).items()
                      if t in PENALTY_TERMS and p.get("weight", -1) > 0]
            if errors:
                self._send(("REFUSED: " + "; ".join(errors)).encode(), "text/plain")
                return
            OVERRIDES.parent.mkdir(exist_ok=True)
            OVERRIDES.write_text(json.dumps(payload, indent=2))
            self._send(
                f"Saved! The Stand task now trains with these. Retrain:  uv run train "
                f"Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 1500".encode(),
                "text/plain",
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--viser-port", type=int, default=8767)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    global PAGE
    PAGE = (PAGE.replace("__VISER_PORT__", str(args.viser_port))
                .replace("__MOTORS__", MOTORS_JS)
                .replace("__GROUPS__", GROUPS_JS)
                .replace("__DEFAULTS__", DEFAULTS_JS)
                .replace("__UPW__", str(DEFAULTS["upright_w"]))
                .replace("__HW__", str(DEFAULTS["height_w"]))
                .replace("__PW__", str(DEFAULTS["pose_w"])))

    init_preview(args.viser_port)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    import socket as _s
    try:
        _sk = _s.socket(_s.AF_INET, _s.SOCK_DGRAM); _sk.connect(("8.8.8.8", 80))
        lan = f"http://{_sk.getsockname()[0]}:{args.port}"; _sk.close()
    except OSError:
        lan = None
    if OVERRIDES.exists():
        print(f"[reward_lab] NOTE: overrides exist and will be loaded by the cfg: {OVERRIDES}")
    print(f"[reward_lab] {url}   3D inside the page  (Ctrl+C to stop)")
    if lan:
        print(f"[reward_lab] from other LAN devices: {lan} (3D on the same hostname, port {args.viser_port})")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
