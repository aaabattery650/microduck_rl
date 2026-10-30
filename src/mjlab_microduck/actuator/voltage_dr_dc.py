"""DC bus-servo actuator (Feetech HD-1910) with per-env supply-voltage DR.

The HD-1910-C001 (like the MG90S before it) is modeled with mjlab's
``DcMotorActuator``: a firmware position-PD whose output is clipped by the
DC-motor torque-speed envelope (full stall torque at zero speed, zero torque
at no-load speed, continuous effort limit on top). On top of that this
subclass adds:

1. In-place conversion of the robot XML's existing ``<position>`` actuators
   into torque motors (MuJoCo rejects a second actuator with the same name —
   the same trick as ``MG90SActuator`` and Pollen's BAM actuator).
2. A per-env ``voltage_scale`` that scales stall torque AND no-load speed
   together — both track the supply voltage ~linearly on a DC servo
   (torque via KT·V/R at stall, speed via back-EMF). This is the analog of
   BAM's ``vin_range`` DR: battery sag and pack state become per-episode
   randomization instead of a fixed datasheet operating point.

The continuous effort limit (thermal rating) is deliberately NOT
voltage-scaled: it is a gearbox/motor heating property, not a bus-voltage
property. Over-temperature protection lives in the real servo anyway
(HD-1910 cuts torque above 80 °C).

Non-accumulating: ``voltage_scale`` is reset to 1.0 then set to a fresh
sample each episode by the ``randomize_dc_servo_voltage`` event (see
tasks/mdp.py), exactly like the BAM friction DR.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg


class VoltageDRDcServoActuator(DcMotorActuator):
    """DcMotorActuator built from the XML actuators + per-env voltage scaling.

    The robot XML already defines a ``<position>`` actuator on every servo
    joint; mjlab's stock DcMotorActuator would add a second actuator with the
    same name, which MuJoCo rejects. ``edit_spec`` converts the existing
    actuators into torque motors in place (gear 1, forcerange/ctrlrange =
    ±effort_limit) and lets the DC-motor model compute the torque each step,
    then applies armature/frictionloss/damping from the cfg onto the joints.
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

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        # Nominal envelopes (scale = 1.0); the DR event restores these before
        # applying a fresh voltage scale, so repeated resets never accumulate.
        self._default_saturation_effort = self.saturation_effort.clone()
        self._default_velocity_limit = self.velocity_limit_motor.clone()
        self.voltage_scale = torch.ones(
            self.saturation_effort.shape[0],
            1,
            dtype=self.saturation_effort.dtype,
            device=self.saturation_effort.device,
        )

    def set_voltage_scale(self, env_ids, voltage_scale: torch.Tensor) -> None:
        """Scale the torque-speed envelope for ``env_ids`` by ``voltage_scale``.

        ``voltage_scale`` is (N, 1) — one factor per env, applied to every
        joint of this actuator (one bus, one battery). Stall torque and
        no-load speed scale together; the corner velocity where the
        torque-speed line meets the continuous effort limit is recomputed
        (mirrors DcMotorActuator.initialize).
        """
        sat = self._default_saturation_effort[env_ids] * voltage_scale
        vel = self._default_velocity_limit[env_ids] * voltage_scale
        self.saturation_effort[env_ids] = sat
        self.velocity_limit_motor[env_ids] = vel
        self._vel_at_effort_lim[env_ids] = vel * (
            1.0 + self.force_limit[env_ids] / sat
        )
        self.voltage_scale[env_ids] = voltage_scale

    def reset_voltage_scale(self, env_ids) -> None:
        ones = torch.ones(
            len(env_ids), 1,
            dtype=self.saturation_effort.dtype,
            device=self.saturation_effort.device,
        )
        self.set_voltage_scale(env_ids, ones)

    def reset_gains(self, env_ids) -> None:
        """Restore nominal firmware PD gains (for non-accumulating gain DR)."""
        self.stiffness[env_ids] = self.default_stiffness[env_ids]
        self.damping[env_ids] = self.default_damping[env_ids]


@dataclass(kw_only=True)
class VoltageDRDcServoActuatorCfg(DcMotorActuatorCfg):
    """DcMotorActuatorCfg that builds from the XML actuators + voltage DR."""

    nominal_voltage: float = 7.4
    """Bus voltage the torque/speed numbers are quoted at (scale = 1.0)."""

    def build(self, entity, target_ids, target_names) -> VoltageDRDcServoActuator:
        return VoltageDRDcServoActuator(self, entity, target_ids, target_names)
