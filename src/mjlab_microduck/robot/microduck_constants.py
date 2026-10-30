import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg, XmlActuatorCfg
from mjlab.actuator.dc_actuator import DcMotorActuator
from mjlab_microduck.actuator import (
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuatorCfg,
    VoltageDRDcServoActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
# Ground-contact model (formerly "allcollisions"): curated collision set for
# the parts that touch the floor in body-on-ground tasks (soles, legs, trunk
# shells, head shells, jaw, battery, hips) — NOT every geom. Shared by
# standup / ground-pick / sitstand / roulade / walk-rollers tasks.
MICRODUCK_GROUNDCONTACT_XML: Path = _ROBOT_DIR / "robot_groundcontact.xml"
# TRUE all-collisions model: every part carries a collision geom (70 geoms,
# 37 meshes; power_support demoted to self_collision_only like every variant).
# No task uses it yet — exported 2026-09 for future envs needing full contact.
MICRODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
# 70mm / 15g ball prop for the BallKick task.
MICRODUCK_BALL_XML: Path = _ROBOT_DIR / "ball.xml"
# Roller-skate model: 14 actuated joints + passive wheel hinges (passive_*wheel).
MICRODUCK_GROUNDCONTACT_ROLLERS_XML: Path = _ROBOT_DIR / "robot_groundcontact_rollers.xml"
# Backlash models: every servo joint gets an unactuated passive_<joint>_backlash
# hinge in series (±1° play, 2° total). Exported via
# config_mjcf_{groundcontact,walk}_backlash.json (add_backlash.py post-processor).
MICRODUCK_GROUNDCONTACT_BACKLASH_XML: Path = _ROBOT_DIR / "robot_groundcontact_backlash.xml"
MICRODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_groundcontact_rollers_backlash.xml"
MICRODUCK_ALLCOLLISIONS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_GROUNDCONTACT_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_XML}"
assert MICRODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_XML}"
assert MICRODUCK_BALL_XML.exists(), f"XML not found: {MICRODUCK_BALL_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_XML}"
assert MICRODUCK_GROUNDCONTACT_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_BACKLASH_XML}"
assert MICRODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_WALK_BACKLASH_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML}"
assert MICRODUCK_ALLCOLLISIONS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_BACKLASH_XML}"


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    # NOTE: was loading robot_groundcontact.xml (no wheels) — the roller env
    # silently ran on the wheel-less standup model.
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_XML))


SERVO_MESH_NAME = "hd1910"  # Feetech HD-1910 housing envelope (was "xl330"; see make_hd1910_model.py)
SERVO_GEOM_SUFFIX = "_servo_collision"


def name_servo_collision_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Name every collision geom that is an XL330 housing mesh.

    The onshape-to-robot export leaves collision geoms unnamed (only the two
    foot soles are named by the export config). The protective-fall tasks need a
    contact sensor on the servo housings specifically — a servo hitting the floor
    is the event that strips gears — so this names each housing collision geom
    ``<body>_<k>_servo_collision``. The ``_collision`` suffix puts them under
    FULL_COLLISION (condim 1) like the other named collision geoms.
    """
    counts: dict[str, int] = {}
    for geom in spec.geoms:
        if geom.meshname != SERVO_MESH_NAME or (geom.contype == 0 and geom.conaffinity == 0):
            continue
        body = geom.parent.name
        k = counts.get(body, 0)
        counts[body] = k + 1
        geom.name = f"{body}_{k}{SERVO_GEOM_SUFFIX}"
    return spec


def get_allcollisions_spec() -> mujoco.MjSpec:
    return name_servo_collision_geoms(mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML)))


def get_allcollisions_backlash_spec() -> mujoco.MjSpec:
    return name_servo_collision_geoms(mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_BACKLASH_XML)))


def get_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))


def get_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_BACKLASH_XML))


def get_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_BACKLASH_XML))


def get_rollers_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Anchored first rules (first-match-wins): pin the passive jaw hinge
        # (HD-1910 15th servo — see make_hd1910_model.py) at 0 so the unanchored
        # patterns below can never catch it.
        r"^passive_jaw$": 0.0,
        # Lower body — STAND2 pose: trunk shifted ~5mm forward over the feet so
        # the CoM sits over the ankle axis (was ~5mm behind it at the old HOME,
        # which biased the robot backward and made the standup policy droop its
        # head forward as a counterweight). Leg pitch chain leaned forward:
        # hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°. Matches the
        # STAND keyframe in scene.xml / scene_walk.xml.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        # Head
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# -- Old actuator (XML position, MuJoCo built-in PD + friction) --
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
# )

# -- BAM M6 actuator (full voltage control + load-dependent friction) --
# Exclude passive_* joints (jaw linkage in the new model has no XML actuator).
# Voltage domain randomization (mirrors mjlab_microban):
#   - vin_range: per-env battery voltage sampled at startup (replaces fixed vin)
#   - vin_drop_gain_range: load-dependent voltage sag V_drop = gain * sum(|tau|)
#   - vin_min: hard floor on the effective voltage after sag
# kp_fw kept at 200 (microduck's preserved firmware stiffness; microban uses 125).
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=200.0,  # microduck's preserved firmware stiffness (microban uses 125)
    # vin_range=(6.9, 7.9),
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    # max_current=1.75,
    delay_min_lag=3,
    delay_max_lag=6,
)

# -- MG90S hobby servo (DC motor model: stall torque + no-load speed + PD) --
# Values are datasheet starting guesses at 6 V; replace them with numbers
# measured on the pendulum rig once the servos arrive.


class MG90SActuator(DcMotorActuator):
    """mjlab's DC-motor servo model, reusing the robot XML's actuators.

    The robot XML already defines a <position> actuator on every servo joint.
    mjlab's stock DcMotorActuator adds a second actuator with the same name,
    which MuJoCo rejects. This converts the existing XML actuators into torque
    motors in place (the same thing Pollen's BAM actuator does) and lets the
    DC-motor model compute the torque each step.
    """

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        cfg = self.cfg
        by_target = {}
        for act in spec.actuators:
            tgt = act.target if isinstance(act.target, str) else getattr(act.target, "name", None)
            by_target[tgt] = act

        missing = []
        for name in target_names:
            act = by_target.get(name)
            if act is None:
                missing.append(name)
                continue
            act.set_to_motor()
            act.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            act.forcelimited = True
            act.forcerange = [-cfg.effort_limit, cfg.effort_limit]
            act.ctrllimited = True
            act.ctrlrange = [-cfg.effort_limit, cfg.effort_limit]
            joint = spec.joint(name)
            if cfg.armature is not None:
                joint.armature = cfg.armature
            if cfg.frictionloss is not None:
                joint.frictionloss = cfg.frictionloss
            if cfg.viscous_damping is not None:
                joint.damping[0] = cfg.viscous_damping
            self._mjs_actuators.append(act)

        # Joints with no XML actuator get a fresh one (stock mjlab behaviour).
        if missing:
            super().edit_spec(spec, missing)


@dataclass(kw_only=True)
class MG90SActuatorCfg(DcMotorActuatorCfg):
    def build(self, entity, target_ids, target_names) -> MG90SActuator:
        return MG90SActuator(self, entity, target_ids, target_names)


MG90S_ACTUATOR = MG90SActuatorCfg(
    target_names_expr=(r"^(?!passive_).*",),
    stiffness=2.0,  # P gain, N·m/rad (guess: full torque at ~6° error)
    damping=0.05,  # D gain, N·m·s/rad (guess)
    saturation_effort=0.22,  # stall torque: 2.2 kg·cm = 0.22 N·m at 6 V
    effort_limit=0.22,  # same as stall: servo is allowed short bursts
    velocity_limit=13.0,  # no-load speed: 0.08 s/60° ≈ 13 rad/s at 6 V
    viscous_damping=0.0,  # speed limit above already models the motor
    frictionloss=0.01,  # gear friction, N·m (guess)
    # armature left at the XML value (XL330's 0.0018) until measured
    delay_min_lag=2,  # physics steps (~5 ms each): PCA9685 PWM + servo lag
    delay_max_lag=6,
)

# -- Feetech HD-1910-C001 bus servo (DC motor model + firmware PD + voltage DR) --
# Datasheet @7.4 V (2S nominal): stall 15 kg·cm = 1.47 N·m, no-load 113 rpm =
# 11.8 rad/s, rated (continuous) 3.7 kg·cm = 0.363 N·m. Coreless motor, metal
# gearbox 1:320, ball bearings, backlash <=0.5°, 12-bit magnetic encoder,
# 4–8.4 V operating range, 21 g. Torque/speed/voltage numbers are datasheet
# values; the firmware PD gains, gearbox Coulomb friction and reflected rotor
# inertia are engineering estimates — replace them with pendulum-rig
# measurements when the servos arrive (see AGENTS.md workflow step 2).
HD1910_ACTUATOR = VoltageDRDcServoActuatorCfg(
    target_names_expr=(r"^(?!passive_).*",),
    stiffness=8.0,  # firmware P gain, N·m/rad (guess: stall torque at ~10.5° error — Feetech mode-4 PD, unmeasured)
    damping=0.15,  # firmware D gain, N·m·s/rad (guess)
    saturation_effort=1.47,  # stall torque @7.4 V (15 kg·cm)
    effort_limit=0.6,  # continuous cap: ~1.65× the 0.363 N·m rating (bursts above rated are thermally fine; the servo itself cuts above 80 °C)
    velocity_limit=11.8,  # no-load speed @7.4 V (113 rpm)
    nominal_voltage=7.4,  # reference for the per-episode bus-voltage DR event
    viscous_damping=0.0,  # the torque-speed envelope above already caps speed
    frictionloss=0.06,  # gearbox Coulomb friction, N·m (estimate; DR'd 0.5–1.5× by the env cfg)
    armature=0.0005,  # reflected rotor inertia J·320² (the XML's XL330 value 0.0018 assumes ~2× the gear ratio)
    delay_min_lag=3,  # 15 ms min: Feetech TTL bus + firmware pipeline (XL330 range kept)
    delay_max_lag=6,  # 30 ms max
)

# Bus-voltage DR range for the HD-1910 (volts at the servo, per episode): a 2S
# pack sagging under 15 servos' load → fresh off the charger. Both ends stay
# inside the 4–8.4 V operating range. Scales stall torque AND no-load speed.
HD1910_VOLTAGE_RANGE = (6.2, 8.4)
# Gearbox-friction DR for the DC-servo path (multiplies dof_frictionloss,
# restore-then-scale): wide because 0.06 N·m is an unmeasured estimate for the
# 1:320 metal train. Tighten after pendulum-rig measurements.
HD1910_FRICTION_SCALE_RANGE = (0.5, 1.5)

# Which servo model every task trains against:
#   "hd1910"    — Feetech HD-1910-C001 (DC-motor model + voltage/friction DR)
#   "mg90s"     — MG90S hobby servo (DC-motor model, no voltage DR)
#   "bam_xl330" — Pollen's measured Dynamixel XL330 (BAM M6 voltage model)
# Override per shell without editing code: MICRODUCK_ACTUATOR=bam_xl330 uv run train ...
ACTUATOR_KIND = os.environ.get("MICRODUCK_ACTUATOR", "hd1910")

if ACTUATOR_KIND == "hd1910":
    actuators = HD1910_ACTUATOR
elif ACTUATOR_KIND == "mg90s":
    actuators = MG90S_ACTUATOR
elif ACTUATOR_KIND == "bam_xl330":
    actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)
else:
    raise ValueError(
        f"Unknown ACTUATOR_KIND {ACTUATOR_KIND!r}; expected one of "
        "'hd1910', 'mg90s', 'bam_xl330' (or set MICRODUCK_ACTUATOR)."
    )

# True when the selected actuator is a DC-servo model (HD-1910 / MG90S):
# gearbox friction is native dof_frictionloss (randomizable) and the
# bus-voltage DR event applies. False under BAM (its own vin/friction DR).
IS_DC_SERVO = ACTUATOR_KIND in ("hd1910", "mg90s")

# Same BAM actuator, but the firmware position loop reads the encoder THROUGH
# the passive_<joint>_backlash hinges (the real encoder is on the output side
# of the gear play). Only for the backlash model; the target regex already
# excludes the passive_* backlash joints from actuation.
# NOTE: the backlash twins stay on the XL330 BAM path regardless of
# ACTUATOR_KIND — they are XL330-study infrastructure. An HD-1910 backlash
# twin would regenerate the XMLs with `add_backlash.py <xml> --backlash-deg
# 0.5` (datasheet play) and add an encoder-through-backlash DC-servo subclass.
backlash_actuators = BacklashEncoderBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# -- BAM M4 actuator
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=make_bam_m4_actuator_cfg(),
# )

# HOME frame for the backlash model. HOME_FRAME's unanchored patterns
# (e.g. r".*left_hip_roll.*") would also match passive_left_hip_roll_backlash
# and try to initialize it at -0.0873 rad — outside its ±1° range. Pattern
# matching is first-match-wins in declaration order, so the anchored backlash
# rule placed FIRST pins every backlash joint at 0 and the servo joints fall
# through to the normal HOME values.
BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# True full-collision model (every part collides, servo housings named — see
# name_servo_collision_geoms). For tasks where the robot falls onto arbitrary
# parts and the reward must know WHICH part hit: VelStand / protective fall.
MICRODUCK_ALLCOLLISIONS_ROBOT_CFG = EntityCfg(
    spec_fn=get_allcollisions_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Backlash robots: base model + ±1° serial backlash hinge per servo.
# Encoder reads through the backlash (BacklashEncoderBamActuator feedback +
# joint_pos/vel_rel_backlash observations — see tasks/backlash.py).
# Groundcontact variant → VelStand/StandUp backlash tasks (mirrors
# MICRODUCK_STANDUP_ROBOT_CFG); walk variant → Velocity backlash
# tasks (mirrors MICRODUCK_WALK_ROBOT_CFG, keeps backlash-vs-base comparisons
# unconfounded by the collision model).
MICRODUCK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# All-collisions backlash robot → VelStand backlash tasks (mirrors
# MICRODUCK_ALLCOLLISIONS_ROBOT_CFG).
MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_allcollisions_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Roller-skate backlash robot: wheels stay free (passive_*wheel untouched by
# add_backlash.py). collisions=() mirrors MICRODUCK_WALK_ROLLERS_ROBOT_CFG —
# roller wheel collision geoms have no explicit names; XML defaults apply.
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_rollers_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Free-floating, non-articulated ball prop for the BallKick task. Position is
# set each episode by the reset_ball_in_front_of_foot event; the init pos here
# only matters for the pristine pre-first-reset state.
MICRODUCK_BALL_CFG = EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)),
)

# Roller skate robot: the 4 passive wheel joints (passive_*wheel) have no XML
# actuators; the BAM cfg's target regex already excludes them, so the action
# space stays 14-dimensional. Uses the SAME canonical BAM actuator as every
# other variant (was a plain XmlActuatorCfg PD — an actuator-physics mismatch
# vs the rest of the family, and joint-friction DR was impossible).
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # roller wheel collision geoms have no explicit names; XML defaults apply
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
