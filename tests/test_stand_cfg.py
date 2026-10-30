"""Stand/balance task cfg invariants — CPU-only.

Locks in: the 61-D obs parity with the velocity family (runtime hot-swap),
zero-ish twist commands with the exact-zero bucket, gait rewards absent,
the height-gated balance term at the MEASURED stand height, and the
inherited HD-1910 DR + fall termination.
"""

import math

from mjlab_microduck.tasks.microduck_stand_env_cfg import (
    HEIGHT_HIGH,
    STAND_Z_SETTLED,
    make_microduck_stand_env_cfg,
    MicroduckStandRlCfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def _stand():
    return make_microduck_stand_env_cfg()


def test_obs_parity_with_velocity_family():
    # Exact term-for-term parity: otherwise the ONNX won't load in the runtime
    # slot alongside the other policies.
    stand = _stand()
    vel = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(stand.observations[grp].terms.keys()) == list(
            vel.observations[grp].terms.keys()
        ), f"obs layout diverged on {grp}"


def test_twist_command_is_zero_dominant():
    cmd = _stand().commands["twist"]
    assert cmd.rel_standing_envs == 0.5  # explicit exact-zero bucket
    assert getattr(cmd, "rel_turn_in_place_envs", 0.0) == 0.0
    for attr in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        lo, hi = getattr(cmd.ranges, attr)
        assert -0.06 < lo <= hi < 0.06  # tiny noise, never a gait command


def test_gait_rewards_are_gone_and_balance_stack_is_present():
    cfg = _stand()
    for gone in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "angular_momentum",
    ):
        assert gone not in cfg.rewards, gone
    # Positive task terms, right order of magnitude vs the inherited stack.
    assert cfg.rewards["upright"].weight == 2.0
    assert cfg.rewards["stand_height"].weight == 1.5
    assert cfg.rewards["pose"].weight == 1.0
    assert cfg.rewards["head_pose_tracking"].weight > 0.0
    # Every remaining penalty must keep its sign (self-negating style > 0).
    for name in ("body_ang_vel", "action_rate_l2", "self_collisions", "dof_pos_limits"):
        assert cfg.rewards[name].weight < 0, name


def test_stand_height_gate_matches_measured_settle():
    params = _stand().rewards["stand_height"].params
    assert params["height_high"] == HEIGHT_HIGH
    # Full reward reachable just below the measured settled height; crouch
    # gate well below it (AGENTS STAND_Z rule: measured, not carried over).
    assert HEIGHT_HIGH <= STAND_Z_SETTLED
    assert params["height_low"] < params["height_high"] - 0.03
    assert params["std"] == math.sqrt(0.05)  # same tilt width as `upright`


def test_balance_perturbations_are_wired():
    cfg = _stand()
    # Start tilted: balance must be reactive, not memorized.
    orient = cfg.events["randomize_base_orientation"]
    assert 0.0 < orient.params["max_pitch_deg"] <= 10.0
    assert orient.mode == "reset"
    # Mid-episode pushes inherited from the recipe.
    assert "push_robot" in cfg.events
    # HD-1910 DR inherited (voltage + native gearbox friction).
    assert "randomize_motor_voltage" in cfg.events
    assert "randomize_joint_friction_dc" in cfg.events


def test_fall_termination_present_and_curriculum_pruned():
    cfg = _stand()
    assert "fell_over" in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "standing_envs" not in cfg.curriculum  # no-op with ~zero commands
    for kept in ("com_range", "action_rate_weight", "head_pose_range"):
        assert kept in cfg.curriculum, kept


def test_own_experiment_name():
    assert MicroduckStandRlCfg.experiment_name == "stand"


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    assert "Mjlab-Stand-Flat-MicroDuck" in list_tasks()
