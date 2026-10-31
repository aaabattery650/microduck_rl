"""Stand reward-override invariants — CPU-only.

Locks the configs/stand_rewards.json hook (written by scripts/reward_lab.py):
values are applied, unknown names fail loud, and the sign-convention footgun
(positive weight on a self-negating penalty) is REFUSED at cfg build.
"""

import json

import pytest

from mjlab_microduck.tasks import microduck_stand_env_cfg as stand
from mjlab_microduck.tasks.microduck_stand_env_cfg import (
    _NEGATIVE_WEIGHT_TERMS,
    _apply_reward_overrides,
    make_microduck_stand_env_cfg,
)

OVERRIDES = stand._REWARD_OVERRIDES


def _write(f, payload):
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(payload))


def test_no_override_file_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", tmp_path / "absent.json")
    cfg = make_microduck_stand_env_cfg()
    assert cfg.rewards["upright"].weight == 2.0  # untouched defaults


def test_overrides_are_applied(tmp_path, monkeypatch):
    f = tmp_path / "stand_rewards.json"
    _write(f, {
        "rewards": {"upright": {"weight": 3.5, "std": 0.3}},
        "pose_std_standing": {r".*knee.*": 0.2},
    })
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", f)
    cfg = make_microduck_stand_env_cfg()
    assert cfg.rewards["upright"].weight == 3.5
    assert cfg.rewards["upright"].params["std"] == 0.3
    assert cfg.rewards["pose"].params["std_standing"][r".*knee.*"] == 0.2


def test_positive_penalty_weight_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "stand_rewards.json"
    _write(f, {"rewards": {"self_collisions": {"weight": 1.0}}})
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", f)
    with pytest.raises(ValueError, match="negative"):
        make_microduck_stand_env_cfg()


def test_unknown_names_fail_loud(tmp_path, monkeypatch):
    f = tmp_path / "stand_rewards.json"
    _write(f, {"rewards": {"no_such_term": {"weight": 1.0}}})
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", f)
    with pytest.raises(ValueError, match="unknown reward term"):
        make_microduck_stand_env_cfg()
    _write(f, {"pose_std_standing": {r".*elbow.*": 0.1}})
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", f)
    with pytest.raises(ValueError, match="unknown pose group"):
        make_microduck_stand_env_cfg()


def test_penalty_term_list_matches_the_sign_guard():
    cfg = make_microduck_stand_env_cfg()
    for name in _NEGATIVE_WEIGHT_TERMS:
        assert cfg.rewards[name].weight < 0, name


def test_pose_targets_swap_the_reward_function(tmp_path, monkeypatch):
    f = tmp_path / "stand_rewards.json"
    _write(f, {"pose_targets": {"left_knee": 0.2}})
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", f)
    cfg = make_microduck_stand_env_cfg()
    from mjlab_microduck.tasks.mdp import pose_targets_gaussian

    assert cfg.rewards["pose"].func is pose_targets_gaussian
    assert cfg.rewards["pose"].params["targets"] == {"left_knee": 0.2}
    assert cfg.rewards["pose"].weight == 1.0  # weight preserved


def test_unknown_pose_target_joint_fails_loud(tmp_path, monkeypatch):
    f = tmp_path / "stand_rewards.json"
    _write(f, {"pose_targets": {"left_elbow": 0.2}})
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", f)
    with pytest.raises(ValueError, match="unknown joint name"):
        make_microduck_stand_env_cfg()


def test_disabled_terms_get_zero_weight(tmp_path, monkeypatch):
    f = tmp_path / "stand_rewards.json"
    _write(f, {"disabled": ["stand_height", "self_collisions"]})
    monkeypatch.setattr(stand, "_REWARD_OVERRIDES", f)
    cfg = make_microduck_stand_env_cfg()
    assert cfg.rewards["stand_height"].weight == 0.0
    assert cfg.rewards["self_collisions"].weight == 0.0  # zero = off, allowed
    assert cfg.rewards["upright"].weight == 2.0  # others untouched
