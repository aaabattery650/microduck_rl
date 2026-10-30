"""HD-1910 (Feetech) actuator invariants — CPU-only, no GPU needed.

Locks in: the default actuator selection, the datasheet numbers at the
nominal voltage, the joint targeting (14 servos, passive_* excluded), the
voltage-DR envelope math (non-accumulating), and the DR event wiring in the
task cfgs. See AGENTS.md "Building a new env" for why cfg tests exist.
"""

import re

import mujoco
import pytest
import torch

from mjlab_microduck.actuator.voltage_dr_dc import (
    VoltageDRDcServoActuator,
    VoltageDRDcServoActuatorCfg,
)
from mjlab_microduck.robot import microduck_constants as consts


def test_default_actuator_is_hd1910():
    # The repo trains against the Feetech HD-1910 by default; XL330 (BAM) and
    # MG90S remain selectable via MICRODUCK_ACTUATOR for A/B comparisons.
    assert consts.ACTUATOR_KIND == "hd1910"
    assert isinstance(consts.actuators, VoltageDRDcServoActuatorCfg)
    assert consts.IS_DC_SERVO is True


def test_hd1910_numbers_match_datasheet_at_7v4():
    cfg = consts.HD1910_ACTUATOR
    # Datasheet @7.4 V: stall 15 kg·cm = 1.47 N·m, no-load 113 rpm = 11.8 rad/s,
    # rated (continuous) 3.7 kg·cm = 0.363 N·m, operating range 4–8.4 V.
    assert cfg.saturation_effort == pytest.approx(1.47, abs=0.01)
    assert cfg.velocity_limit == pytest.approx(11.8, abs=0.1)
    assert cfg.nominal_voltage == pytest.approx(7.4)
    # Continuous cap between the rated load and stall (bursts allowed).
    assert 0.363 < cfg.effort_limit < cfg.saturation_effort
    # Bus + firmware latency: 3–6 physics steps (15–30 ms at 5 ms).
    assert 0 <= cfg.delay_min_lag <= cfg.delay_max_lag
    # Positive gearbox friction and reflected rotor inertia.
    assert cfg.frictionloss > 0.0
    assert cfg.armature > 0.0


def test_hd1910_voltage_dr_range_stays_inside_operating_range():
    lo, hi = consts.HD1910_VOLTAGE_RANGE
    assert 4.0 <= lo < hi <= 8.4  # Feetech HD-1910 operating range
    lo, hi = consts.HD1910_FRICTION_SCALE_RANGE
    assert 0.0 < lo < 1.0 < hi  # straddles the nominal estimate


def test_hd1910_targets_the_14_servo_joints_only():
    # Resolve the target regex on the actual walk model: the passive_* joints
    # (jaw hinge / backlash hinges / wheels) must never be actuated, and the
    # root free joint is not an articulation drive target (mjlab entities
    # resolve actuator targets over the hinge joints).
    spec = consts.get_walk_spec()
    names = [
        j.name
        for j in spec.joints
        if j.name and j.type != mujoco.mjtJoint.mjJNT_FREE
    ]
    matched = [n for n in names if re.fullmatch(r"^(?!passive_).*", n)]
    assert len(matched) == 14, matched


def test_jaw_is_present_and_policy_invisible():
    # The HD-1910 15th servo: a hinged jaw_link child of jaw_soft (at the
    # mouth_tip anchor), named passive_* so every obs/action/reward selector
    # excludes it — the 61-D/14-D policy contract is untouched.
    spec = consts.get_walk_spec()
    jaw = spec.joint("passive_jaw")
    assert jaw.type == mujoco.mjtJoint.mjJNT_HINGE
    body = spec.body("jaw_link")
    assert body.parent.name == "jaw_soft"
    # Sits between head_roll and the right leg in joint order — regex-selected
    # code never notices, hardcoded indices do (see roller_standup resolver).
    order = [j.name for j in spec.joints if j.name]
    assert order.index("passive_jaw") > order.index("head_roll")
    assert order.index("passive_jaw") < order.index("right_hip_yaw")



def _make_bare_actuator(n_envs=4, n_joints=3):
    """A VoltageDRDcServoActuator with hand-set tensors (no mjlab runtime)."""
    act = object.__new__(VoltageDRDcServoActuator)
    act.cfg = VoltageDRDcServoActuatorCfg(
        target_names_expr=(r"^(?!passive_).*",),
        stiffness=8.0,
        damping=0.15,
        saturation_effort=2.0,
        effort_limit=1.0,
        velocity_limit=10.0,
        nominal_voltage=7.4,
    )
    act.saturation_effort = torch.full((n_envs, n_joints), 2.0)
    act.velocity_limit_motor = torch.full((n_envs, n_joints), 10.0)
    act.force_limit = torch.full((n_envs, n_joints), 1.0)
    # Corner velocity at nominal: v_nl * (1 + tau_cont / tau_stall) = 10 * 1.5.
    act._vel_at_effort_lim = torch.full((n_envs, n_joints), 15.0)
    act._default_saturation_effort = act.saturation_effort.clone()
    act._default_velocity_limit = act.velocity_limit_motor.clone()
    act.voltage_scale = torch.ones(n_envs, 1)
    return act


def test_voltage_scale_scales_envelope_and_corner_velocity():
    act = _make_bare_actuator()
    ids = torch.tensor([0, 2])
    scales = torch.tensor([[6.2 / 7.4], [8.4 / 7.4]])  # 0.8378, 1.1351
    act.set_voltage_scale(ids, scales)

    for row, s in zip(ids.tolist(), scales.flatten().tolist()):
        assert act.saturation_effort[row].min() == pytest.approx(2.0 * s)
        assert act.velocity_limit_motor[row].min() == pytest.approx(10.0 * s)
        # Corner velocity recomputed from the scaled values.
        expected = 10.0 * s * (1.0 + 1.0 / (2.0 * s))
        assert act._vel_at_effort_lim[row].min() == pytest.approx(expected)
        assert act.voltage_scale[row].min() == pytest.approx(s)
    # Untouched envs keep the nominal envelope.
    assert act.saturation_effort[1].min() == pytest.approx(2.0)
    assert act.voltage_scale[1].min() == pytest.approx(1.0)


def test_voltage_scale_is_non_accumulating():
    # Two consecutive applications: the second must depend only on its own
    # sample, not on the first (the restore-then-apply contract).
    act = _make_bare_actuator()
    ids = torch.tensor([1])
    act.set_voltage_scale(ids, torch.tensor([[8.4 / 7.4]]))
    act.set_voltage_scale(ids, torch.tensor([[6.2 / 7.4]]))
    assert act.saturation_effort[1].min() == pytest.approx(2.0 * 6.2 / 7.4)
    act.reset_voltage_scale(ids)
    assert act.saturation_effort[1].min() == pytest.approx(2.0)
    assert act.velocity_limit_motor[1].min() == pytest.approx(10.0)


def test_velocity_env_wires_hd1910_dr_events():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg()
    assert "randomize_motor_voltage" in cfg.events
    assert "randomize_joint_friction_dc" in cfg.events
    v = cfg.events["randomize_motor_voltage"].params["voltage_range"]
    assert 4.0 <= v[0] <= v[1] <= 8.4
    # The DC friction event must not touch passive_* joints.
    pattern = cfg.events["randomize_joint_friction_dc"].params["asset_cfg"].joint_names
    assert all(re.fullmatch(r"^(?!passive_).*", p) for p in pattern)
    # Firmware gain DR is wired and non-degenerate (HD-1910 gains are guesses).
    assert "randomize_motor_gains" in cfg.events
    assert cfg.events["randomize_motor_gains"].params["kp_range"][0] < 1.0


def test_velocity_env_actor_obs_stays_61d():
    # The actuator swap must not leak into the observation contract: the actor
    # keeps the shared 61D layout — 48 proprioception + the 13D command block
    # [twist(3), head_pose(4), body_pose(6)] with every slot present.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg()
    actor = cfg.observations["actor"].terms
    assert "base_lin_vel" not in actor
    assert "height_scan" not in actor
    for slot in ("command", "head_command", "body_command"):
        assert slot in actor, slot
    assert actor["head_command"].params["command_name"] == "head_pose"
    assert actor["body_command"].params["command_name"] == "body_pose"
