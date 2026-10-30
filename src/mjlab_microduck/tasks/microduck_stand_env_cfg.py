"""Microduck stand/balance environment — the simplest policy in the family.

One skill: **stay standing.** Built ON the velocity recipe
(`make_microduck_velocity_env_cfg`) so the HD-1910 actuator, all DR, obs
noise, delays and NaN guards stay in sync for free; then:

  - twist command ≈ zero (tiny noise keeps the slot alive; deployment idle
    feeds exact zeros — half the envs get exact-zero explicitly, per the
    zero-command rule in AGENTS.md),
  - gait rewards REMOVED (track_linear/angular, air_time, foot_clearance,
    foot_swing_height, foot_slip) — nothing rewards stepping,
  - balance stack: `upright` (tilt Gaussian) + `stand_height`
    (upright_gaussian_at_height: upright only pays at standing height — no
    "crouch low and vertical" local optimum; height_high measured in sim
    under the HD-1910 firmware PD: trunk settles at 0.116 m — measured, not
    carried across model revisions) + tight `pose` + `head_pose_tracking`
    (balancing while the ~38 %-mass head moves on command) + regularizers,
  - initial tilt randomization + the velocity pushes inherited from the
    recipe → REACTIVE balance, not memorized stillness.

Head/body pose command slots stay alive (head motion is the perturbation
spectrum; body_pose stays weight-0 parity padding). The 61-D/14-D runtime
contract is untouched — the exported ONNX drops into the runtime's `stand`
slot. ~500-1500 iterations at 4096 envs.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Measured in sim (CPU probe, HD-1910 firmware PD, zero action, 1.2 s settle):
# trunk z mean 0.1164 m, tilt ~0.006 rad → STAND pose is a stable equilibrium.
STAND_Z_SETTLED = 0.1164
HEIGHT_HIGH = 0.114  # full upright reward from here up (settled mean minus margin)
HEIGHT_LOW = 0.080   # no upright-at-height reward below (crouch gate)

# Initial-tilt randomization band (rad): the policy must recover INTO balance,
# not start memorized-perfect. Deliberately smaller than the standup task's —
# episodes always start standing.
INIT_MAX_PITCH_DEG = 6.0
INIT_MAX_ROLL_DEG = 4.0

# Rewards to DELETE from the velocity recipe (everything that pays for motion).
GAIT_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "angular_momentum",
)


def make_microduck_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Balance-only task: stand at (near-)zero command, handle head motion,
    initial tilt and pushes. Flat terrain only."""
    cfg = make_microduck_velocity_env_cfg(play=play)

    # ── Commands: zero twist, live slots ──────────────────────────────────────
    # deepcopy first: the velocity factory's command object is shared mutable
    # state (see its own comment). Half the envs get EXACT zero (deployment
    # idle), the rest tiny noise so the twist input neurons see variance.
    command = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.5
    command.rel_turn_in_place_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    command.resampling_time_range = (10.0, 10.0)
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # head_pose / body_pose commands inherited untouched: head tracking is a
    # balance-relevant objective; body_pose stays weight-0 parity padding.

    # ── Rewards: strip the gait, keep and sharpen the balance stack ───────────
    for name in GAIT_REWARDS:
        cfg.rewards.pop(name, None)

    # Height-gated upright: closes the crouch local optimum of a pure tilt
    # Gaussian (see mdp.upright_gaussian_at_height docstring).
    cfg.rewards["stand_height"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=1.5,
        params={
            "std": math.sqrt(0.05),  # same tilt width as `upright`
            "height_low": HEIGHT_LOW,
            "height_high": HEIGHT_HIGH,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # `pose` inherits the velocity tight std_standing (command≈0 → standing
    # branch). `upright` (2.0), `self_collisions`, `dof_pos_limits`,
    # `action_rate_l2` (curriculum-ramped) and `head_pose_tracking` are
    # inherited as-is. body_ang_vel (-0.05) damps balance jitter.

    # ── Events: start standing, but never perfectly ───────────────────────────
    # The velocity module ships randomize_base_orientation behind a False
    # toggle; balance is the task that wants it on.
    cfg.events["randomize_base_orientation"] = EventTermCfg(
        func=microduck_mdp.randomize_base_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_pitch_deg": INIT_MAX_PITCH_DEG,
            "max_roll_deg": INIT_MAX_ROLL_DEG,
        },
    )
    # Velocity pushes (inherited, ±0.3 m/s every 3-6 s) supply the mid-episode
    # perturbations. HD-1910 DR (voltage/friction/gains) inherited via the
    # velocity recipe's add_dc_servo_dr_events.

    # ── Terminations: fell_over + nan_state + time_out all inherited ──────────

    # ── Curricula: inherited where meaningful ────────────────────────────────
    # com_range / head_com_range (widen CoM DR — robustness), action_rate_weight
    # (smoothness ramp), head_pose_range (widen head commands) all apply.
    # standing_envs is a no-op here (commands are ~zero already) and is dropped
    # to keep the wandb panel honest.
    cfg.curriculum.pop("standing_envs", None)

    return cfg


MicroduckStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,  # balance is symmetric, but keep parity with velocity
    ),
    wandb_project="mjlab_microduck",
    experiment_name="stand",  # Directory name
    run_name="stand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
