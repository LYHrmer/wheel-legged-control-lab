"""Explicit 06 body-speed controller transfer over the unchanged rolling plant.

This module is imported only in the root-prepaid physical process. Construct the
original rolling environments and run the strict original checkpoint loader
before calling :func:`install_body_controller` on either environment.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from body_speed_controller_06 import (
    BODY_CONTROL_SCHEMA,
    BODY_TASK_SCHEMA,
    BodyCommonPRollingController,
    assert_cooperative_mro,
)
from drive_fresh_lifecycle_05 import assert_never_reset_inactive

from scripts.d1_rolling_residual_env import D1RollingResidualEnv
from scripts.d1_rolling_residual_task import ROLLING_CONTROL_SCHEMA, ROLLING_TASK_SCHEMA
from wheel_legged_control.d1.control_loop import D1WheelLegControllerAdapter


def install_body_controller(original: D1RollingResidualEnv) -> dict[str, Any]:
    """Replace only the inactive original controller adapter, never plant/data."""
    if not isinstance(original, D1RollingResidualEnv):
        raise TypeError("transfer requires the original scalar rolling environment")
    inner = original.unwrapped
    before_lifecycle = assert_never_reset_inactive(inner)
    old = inner._controller.controller
    if old.control_schema != ROLLING_CONTROL_SCHEMA:
        raise RuntimeError("strict original checkpoint view must precede controller transfer")
    plant = inner.plant
    model, data = plant.model, plant.data
    model_address, data_address = int(model._address), int(data._address)
    mro = assert_cooperative_mro()
    new = BodyCommonPRollingController(enabled=True, **asdict(inner.wheel_leg_control))
    if new.control_schema != BODY_CONTROL_SCHEMA:
        raise RuntimeError("transferred controller schema differs")
    adapter = D1WheelLegControllerAdapter(new)
    inner._controller = adapter
    if (plant is not inner.plant or model is not inner.plant.model or data is not inner.plant.data
            or model_address != int(inner.plant.model._address)
            or data_address != int(inner.plant.data._address)):
        raise RuntimeError("controller transfer changed model/data, time, or lifecycle")
    after_lifecycle = assert_never_reset_inactive(inner)
    if after_lifecycle != before_lifecycle:
        raise RuntimeError("controller transfer changed the fresh lifecycle shape")
    return {
        "schema": "d1-rolling-controller-law-transfer-v1",
        "historical_training_control_schema": ROLLING_CONTROL_SCHEMA,
        "transferred_control_schema": BODY_CONTROL_SCHEMA,
        "historical_training_task_schema": ROLLING_TASK_SCHEMA,
        "transferred_task_schema": BODY_TASK_SCHEMA,
        "controller_mro": mro,
        "model_address_unchanged": True,
        "data_address_unchanged": True,
        "physical_time_zero_before_first_reset": True,
        "fresh_lifecycle_before": before_lifecycle,
        "fresh_lifecycle_after": after_lifecycle,
        "controller_replaced_after_strict_original_checkpoint_load": True,
        "new_raw_forward_gate": "bound_raw_forward_velocity_mps != 0.0",
        "drive_leg_delta_wheels_exact_zero": True,
        "body_common_p_leg_delta_exact_zero": True,
        "body_common_p_uses_original_pi_once": True,
    }


class BodyCommonPTransferredEnv(gym.Wrapper):
    """One-scalar actor view with honest new task metadata and pre-control taps."""

    task_schema = BODY_TASK_SCHEMA

    def __init__(self, original: D1RollingResidualEnv, transfer_receipt: dict[str, Any]) -> None:
        if (original.unwrapped._controller.controller.control_schema
                != BODY_CONTROL_SCHEMA):
            raise RuntimeError("wrapper requires completed explicit controller transfer")
        super().__init__(original)
        self.transfer_receipt = dict(transfer_receipt)
        self.pre_control_states: list[dict[str, Any]] = []
        self.observation_space = original.observation_space
        self.action_space = original.action_space

    @property
    def episode_metadata(self) -> dict[str, Any]:
        metadata = dict(self.env.episode_metadata)
        historical_definition = metadata.pop("rolling_task_definition")
        metadata.update({
            "task_schema": BODY_TASK_SCHEMA,
            "controller_schema": BODY_CONTROL_SCHEMA,
            "historical_training_task_schema": ROLLING_TASK_SCHEMA,
            "historical_training_control_schema": ROLLING_CONTROL_SCHEMA,
            "historical_training_task_definition": historical_definition,
            "rolling_task_definition": {
                **historical_definition,
                "task_schema": BODY_TASK_SCHEMA,
                "control_schema": BODY_CONTROL_SCHEMA,
                "controller_schema": BODY_CONTROL_SCHEMA,
                "controller_law_transfer":
                    "fixed_drive_jx_and_body_com_speed_common_wheel_p_on_bound_raw_forward",
            },
            "controller_law_transfer": dict(self.transfer_receipt),
            "policy_action_mapping_unchanged": True,
            "observation_schema_unchanged": True,
        })
        return metadata

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.pre_control_states = []
        observation, info = self.env.reset(seed=seed, options=options)
        result = dict(info)
        result["episode_metadata"] = self.episode_metadata
        return observation, result

    def configure_next_episode(self, spec: Any) -> None:
        self.env.configure_next_episode(spec)

    def step(self, action: Any):
        inner = self.env.unwrapped
        context = inner.heading_decision
        if context is None or inner.decision is None or context.tick != inner._steps:
            raise RuntimeError("pre-control tap lacks the exact prepared decision")
        state = context.decision.context.state
        plant = inner.plant
        qpos = np.asarray(plant.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(plant.data.qvel, dtype=np.float64).copy()
        qadr = np.asarray(plant.qpos_addresses, dtype=np.int64)
        vadr = np.asarray(plant.dof_addresses, dtype=np.int64)
        before = {
            "tick": int(context.tick),
            "provider_state_sequence": int(state.sequence),
            "provider_state_age_s": float(state.age_s),
            "provider_control_time_s": float(state.control_time_s),
            "provider_base_rotation": np.asarray(state.base_rotation).copy(),
            "provider_base_linear_velocity_body":
                np.asarray(state.base_linear_velocity_body).copy(),
            "provider_foot_jacobian": np.asarray(state.foot_jacobian).copy(),
            "provider_joint_position": np.asarray(state.joint_position).copy(),
            "provider_joint_velocity": np.asarray(state.joint_velocity).copy(),
            "raw_precontrol_qpos": qpos,
            "raw_precontrol_qvel": qvel,
            "raw_joint_position": qpos[qadr].copy(),
            "raw_joint_velocity": qvel[vadr].copy(),
        }
        result = self.env.step(action)
        self.pre_control_states.append(before)
        return result
