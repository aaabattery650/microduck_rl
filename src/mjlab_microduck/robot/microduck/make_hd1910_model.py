#!/usr/bin/env python
"""Regenerate the microduck MJCF models for the Feetech HD-1910-C001 swap.

Run from anywhere:  uv run python src/mjlab_microduck/robot/microduck/make_hd1910_model.py

What it does (idempotent — safe to re-run):

1. Generates ``assets/hd1910.stl``: a parametric housing ENVELOPE (34x20x23 mm
   case) axis-aligned to the existing ``xl330.stl`` so the swap cannot be
   rotated wrong. The axis carrying the XL330's 28.5 mm height carries 23 mm;
   the 34/20 mm plan axes are unchanged (identical footprint). MuJoCo
   convex-hulls mesh geoms for collision, so an envelope box is the right
   fidelity: joint axes come from the kinematics (<joint>/<body pos>), never
   from the mesh. Bosses/spline are NOT modeled — their in-plane position is a
   VERIFY-against-datasheet quantity; the printed spacer restores the spline
   axis physically.

2. In all EIGHT model XMLs (4 base + 4 backlash twins — the twins are edited
   in place, NOT regenerated, to preserve their onshape provenance):
   - repoint every ``mesh="xl330"`` geom reference to ``mesh="hd1910"`` and
     swap the asset declaration ``file="xl330.stl"`` -> ``file="hd1910.stl"``
     (MuJoCo derives the mesh name from the filename).
   - insert the jaw as a new jointed body ``jaw_link`` under ``jaw_soft``
     (anchored at the ``mouth_tip`` site of that file), hinged via
     ``passive_jaw``. The ``passive_`` prefix makes every obs/action/reward
     selector in the repo ignore it, so the 61-D obs / 14-D action contract
     is untouched; the real jaw servo is daemon-commanded at deployment.
     Axis/range/mass are PLACEHOLDERS until the printed linkage is measured.

3. Prints a summary. Body masses are deliberately NOT rescaled: bodies merge
   servo + bracket parts, the real delta is ~2 g/servo (inside the +/-5%
   mass DR), and the Onshape re-export will set exact values.

Manual companions (kept out of scripts on purpose):
  - microduck_constants.py: SERVO_MESH_NAME = "hd1910"
  - microduck_constants.py: HOME_FRAME gains an anchored ``^passive_jaw$: 0.0``
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent
ASSETS = DIR / "assets"

MODELS = [
    "robot_walk.xml",
    "robot_groundcontact.xml",
    "robot_groundcontact_rollers.xml",
    "robot_allcollisions.xml",
    "robot_walk_backlash.xml",
    "robot_groundcontact_backlash.xml",
    "robot_groundcontact_rollers_backlash.xml",
    "robot_allcollisions_backlash.xml",
]

# -- HD-1910-C001 datasheet (meters); the 34x20 footprint matches the XL330 --
HD1910_LONG = 0.034   # case length (matches XL330)
HD1910_WIDTH = 0.020  # case width  (matches XL330)
HD1910_HEIGHT = 0.023  # case height (XL330: 28.5 mm — the axis that changes)

# -- Beak (lower, moving) placeholders — VERIFY once the printed linkage exists --
JAW_RANGE = (-0.15, 0.15)   # rad, ~±8.6°
JAW_DAMPING = 0.02
JAW_FRICTIONLOSS = 0.01
JAW_ARMATURE = 0.0005
JAW_MASS = 0.005            # kg — beak shovel (~3 g PLA) + pin/clevis hardware
BEAK_LEN = 0.030            # m, hinge -> tip
BEAK_W_REAR = 0.022
BEAK_W_FRONT = 0.016
BEAK_T_REAR = 0.006
BEAK_T_FRONT = 0.004
BEAK_HINGE_BACK_OFFSET = 0.022  # hinge sits this far behind the mouth_tip anchor
JAW_AXIS = "0 1 0"          # lateral axis in jaw_soft's local frame

# Mass of one printed spacer (kg), added to every servo-carrying body so the
# sim matches the spacer build. VERIFY: weigh a printed spacer (4 perimeters +
# ~30% infill ≈ 2 g; solid ≈ 4.6 g) and set the exact value. Applied once per
# body — guarded by a marker comment, so re-runs never accumulate.
SPACER_MASS_KG = 0.002


def load_stl_binary(path: Path) -> np.ndarray:
    """Vertices (n, 3, 3) of a binary STL."""
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw[84 : 84 + n * 50], dtype=np.uint8).reshape(n, 50)
    tris = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(3):
        tris[:, i, :] = rec[:, 12 + i * 12 : 12 + i * 12 + 12].copy().view(np.float32).reshape(n, 3)
    return tris


def write_stl(path: Path, tris: np.ndarray) -> None:
    """Write a binary STL (millimetre-agnostic: units are whatever you pass)."""
    n = len(tris)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals = normals / lens
    with path.open("wb") as f:
        f.write(b"hd1910 parametric envelope (microduck_rl make_hd1910_model)".ljust(80, b"\0"))
        f.write(struct.pack("<I", n))
        for t, nm in zip(tris, normals):
            f.write(struct.pack("<3f", *nm.astype(float)))
            for v in t:
                f.write(struct.pack("<3f", *v.astype(float)))
            f.write(struct.pack("<H", 0))


def box_tris(center: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """12 triangles of an axis-aligned box."""
    h = extents / 2.0
    c = center
    P = np.array([
        c + [-h[0], -h[1], -h[2]], c + [h[0], -h[1], -h[2]], c + [h[0], h[1], -h[2]], c + [-h[0], h[1], -h[2]],
        c + [-h[0], -h[1], h[2]], c + [h[0], -h[1], h[2]], c + [h[0], h[1], h[2]], c + [-h[0], h[1], h[2]],
    ])
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return np.array([[P[a], P[b], P[c_]] for a, b, c_ in F], dtype=np.float32)


def _rounded_rect_ring(w: float, t: float, r: float, n: int = 4) -> np.ndarray:
    """Convex rounded-rect ring in (thickness, width) plane, CCW."""
    pts = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        cx, cy = sx * (w / 2 - r), sy * (t / 2 - r)
        a0 = np.arctan2(sy, sx)
        for k in range(n + 1):
            a = a0 + (np.pi / 2) * k / n
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def loft_tris(ring_a: np.ndarray, z_a: float, ring_b: np.ndarray, z_b: float) -> np.ndarray:
    """Loft two convex rings (same vertex count) along z; caps fan from centroid."""
    tris = []
    n = len(ring_a)
    assert len(ring_b) == n
    for k in range(n):
        p0a, p1a = ring_a[k], ring_a[(k + 1) % n]
        p0b, p1b = ring_b[k], ring_b[(k + 1) % n]
        tris.append(((p0a[0], p0a[1], z_a), (p1a[0], p1a[1], z_a), (p1b[0], p1b[1], z_b)))
        tris.append(((p0a[0], p0a[1], z_a), (p1b[0], p1b[1], z_b), (p0b[0], p0b[1], z_b)))
    for ring, z, flip in ((ring_a, z_a, True), (ring_b, z_b, False)):
        c = ring.mean(axis=0)
        for k in range(n):
            p0, p1 = ring[k], ring[(k + 1) % n]
            t = ((p0[0], p0[1], z), (p1[0], p1[1], z), (c[0], c[1], z))
            tris.append(t if flip else (t[0], t[2], t[1]))
    return np.array(tris, dtype=np.float32)


def make_beak_mesh() -> list[str]:
    """Lower-beak wedge (meters): hinge at origin, tip at -z (face-local frame;
    the part is rotated to the robot at assembly). Rear ring at z=0, tip ring at
    z=-BEAK_LEN; width is lateral (y), thickness through (x). Envelope fidelity
    only — MuJoCo hulls it for collision."""
    rear = _rounded_rect_ring(BEAK_W_REAR, BEAK_T_REAR, 0.002)
    tip = _rounded_rect_ring(BEAK_W_FRONT, BEAK_T_FRONT, 0.0015)
    write_stl(ASSETS / "beak_lower.stl", loft_tris(rear, 0.0, tip, -BEAK_LEN))
    return [
        f"beak_lower.stl: wedge {BEAK_W_REAR*1e3:.0f}mm rear -> {BEAK_W_FRONT*1e3:.0f}mm tip, "
        f"{BEAK_LEN*1e3:.0f}mm long (visual + collision hull for jaw_link)"
    ]


def make_hd1910_mesh() -> tuple[list[str], list[float]]:
    xl = load_stl_binary(ASSETS / "xl330.stl")
    flat = xl.reshape(-1, 3)
    mn, mx = flat.min(axis=0), flat.max(axis=0)
    ext, ctr = mx - mn, (mx + mn) / 2.0
    order = np.argsort(ext)  # ascending: [width(20), height(28.5), length(34)]
    hd = np.array([HD1910_WIDTH, HD1910_HEIGHT, HD1910_LONG], dtype=np.float64)
    new_ext = np.zeros(3)
    new_ext[order[0]] = hd[0]
    new_ext[order[1]] = hd[1]
    new_ext[order[2]] = hd[2]
    write_stl(ASSETS / "hd1910.stl", box_tris(ctr.astype(np.float32), new_ext.astype(np.float32)))
    log = [
        f"xl330.stl bbox (m): {np.round(ext, 5).tolist()} -> hd1910.stl extents: {np.round(new_ext, 5).tolist()}",
        f"hd1910.stl centroid aligned to xl330.stl centroid {np.round(ctr, 5).tolist()}",
    ]
    return log, ext.tolist()


def jaw_block(indent: str, pos: tuple[float, float, float], collision_class: bool) -> str:
    geom_cls = 'class="collision" ' if collision_class else ""
    return (
        f"{indent}<!-- HD-1910 jaw servo (15th): lower-beak wedge, policy-invisible\n"
        f"{indent}     (passive_ prefix), daemon-commanded at deployment. Geometry/mass are\n"
        f"{indent}     PLACEHOLDERS until the printed beak is measured (make_hd1910_model.py). -->\n"
        f'{indent}<body name="jaw_link" pos="{pos[0]:.8g} {pos[1]:.8g} {pos[2]:.8g}">\n'
        f'{indent}  <joint name="passive_jaw" type="hinge" axis="{JAW_AXIS}" '
        f'range="{JAW_RANGE[0]} {JAW_RANGE[1]}" damping="{JAW_DAMPING}" '
        f'frictionloss="{JAW_FRICTIONLOSS}" armature="{JAW_ARMATURE}"/>\n'
        f'{indent}  <inertial pos="0 0 -0.012" mass="{JAW_MASS}" '
        f'diaginertia="2.1e-7 3.9e-7 5.8e-7"/>\n'
        f'{indent}  <geom name="jaw_link_collision" type="mesh" {geom_cls}'
        f'mesh="beak_lower" rgba="0.9 0.72 0.1 1"/>\n'
        f"{indent}</body>\n"
    )


def bake_spacer_mass(text: str) -> tuple[str, int]:
    """Add SPACER_MASS_KG to the inertial of every body carrying an hd1910
    housing geom (onshape exports put <inertial> right before the body's
    geoms, so the nearest preceding inertial is the right one). Scales the
    inertia tensor with the mass (same geometry). Idempotent via marker."""
    out, count = [], 0
    marker = "<!-- hd1910-spacer-mass -->"
    for m in re.finditer(r'mesh="hd1910"', text):
        inert = None
        for im in re.finditer(r"<inertial[^>]*/>", text[: m.start()]):
            inert = im
        if inert is None:
            continue
        if text[max(0, inert.start() - 60) : inert.start()].find("hd1910-spacer-mass") >= 0:
            continue  # already baked
        tag = inert.group(0)
        mass = float(re.search(r'mass="([^"]+)"', tag).group(1))
        new_mass = mass + SPACER_MASS_KG
        ratio = new_mass / mass
        new_tag = tag.replace(f'mass="{mass}"', f'mass="{new_mass:.8g}"')
        new_tag = re.sub(
            r'(fullinertia|diaginertia)="([^"]+)"',
            lambda mm: mm.group(1)
            + '="'
            + " ".join(f"{float(v) * ratio:.6g}" for v in mm.group(2).split())
            + '"',
            new_tag,
        )
        out.append((inert.start(), inert.end(), f"{marker}{new_tag}"))
        count += 1
    for start, end, repl in reversed(out):
        text = text[:start] + repl + text[end:]
    return text, count


def process_xml(path: Path) -> list[str]:
    text = path.read_text()
    log = []
    n_refs = text.count('mesh="xl330"')
    if n_refs:
        text = text.replace('mesh="xl330"', 'mesh="hd1910"')
        log.append(f"repointed {n_refs} mesh refs xl330 -> hd1910")
    if '<mesh file="xl330.stl"/>' in text:
        text = text.replace('<mesh file="xl330.stl"/>', '<mesh file="hd1910.stl"/>')
        log.append("swapped asset declaration xl330.stl -> hd1910.stl")
    if '<mesh file="beak_lower.stl"/>' not in text:
        text = text.replace(
            '<mesh file="hd1910.stl"/>',
            '<mesh file="hd1910.stl"/>\n    <mesh file="beak_lower.stl"/>',
            1,
        )
        log.append("added beak_lower.stl asset")
    # Replace any previously inserted jaw block (template evolves with the beak).
    if re.search(r"<!-- HD-1910 jaw servo.*?</body>", text, re.S):
        text = re.sub(r"\n?<!-- HD-1910 jaw servo.*?</body>\n?", "\n", text, flags=re.S)
        log.append("replaced previous jaw_link block")
    if '<body name="jaw_soft"' in text:
        m = re.search(r'<site[^>]*name="mouth_tip"[^>]*pos="([^"]+)"', text)
        if m:
            site = [float(x) for x in m.group(1).split()]
            # Hinge sits BEHIND the mouth tip along the face's local +z; the beak
            # mesh extends forward (-z) from the hinge. VERIFY against the print.
            pos = (site[0], site[1], site[2] + BEAK_HINGE_BACK_OFFSET)
            anchor = re.search(r'^.*name="mouth_tip".*$', text, re.M)
            indent = anchor.group(0)[: len(anchor.group(0)) - len(anchor.group(0).lstrip())]
            block = jaw_block(indent, pos, 'class="collision"' in text)
            text = text[: anchor.end()] + "\n" + block.rstrip("\n") + text[anchor.end():]
            log.append(f"inserted jaw_link (beak) hinged {BEAK_HINGE_BACK_OFFSET*1e3:.0f}mm behind mouth_tip")
        else:
            log.append("WARNING: no mouth_tip site — jaw NOT inserted")
    text, n_baked = bake_spacer_mass(text)
    if n_baked:
        log.append(f"baked +{SPACER_MASS_KG*1e3:.1f} g spacer mass into {n_baked} servo bodies")
    if log:
        path.write_text(text)
    return log


def main() -> int:
    print("== HD-1910 model regeneration ==")
    mesh_log, xl_ext = make_hd1910_mesh()
    for line in mesh_log + make_beak_mesh():
        print(" ", line)
    if max(xl_ext) > 1.0:
        print("  ERROR: xl330.stl appears to be in mm — mesh scale mismatch, aborting.")
        return 1
    for name in MODELS:
        path = DIR / name
        if not path.exists():
            print(f"  {name}: MISSING — skipped")
            continue
        log = process_xml(path)
        print(f"  {name}: {'; '.join(log) if log else 'no changes needed'}")
    print(
        "\nManual companions (microduck_constants.py):\n"
        '  SERVO_MESH_NAME = "hd1910"\n'
        "  HOME_FRAME: add r\"^passive_jaw$\": 0.0 as the FIRST rule"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
