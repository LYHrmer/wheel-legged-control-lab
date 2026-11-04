"""Opt-in coherent sampling for the frozen legacy fusion experiment.

Only the sensor sampling phase changes. The live plant, its sampling mode,
the original noise draws, delayed-packet queue and legacy observer Jacobians
remain unchanged. This is not a CV predictor and does not make delayed data
current. No archived source or default runtime is replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields as dataclass_fields

import mujoco
import numpy as np

from .sensor_estimation import D1MujocoSensorSource, D1SensorNoise, D1SensorStateSource
from .state_estimation import (
    D1LatencyCompensationResult,
    D1StateEstimate,
    prepare_d1_control_state,
)

_STATE_INPUTS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
    "userdata",
)
_PREDICTED_GEOMETRY = (
    "base_position",
    "base_rotation",
    "joint_position",
    "foot_position",
    "foot_jacobian",
)
_HELD_VELOCITIES = (
    "base_linear_velocity_body",
    "base_angular_velocity_body",
    "joint_velocity",
)
_ROTATED_VELOCITIES = ("base_linear_velocity_world", "base_angular_velocity_world")
_CONTACT_FIELDS = (
    "wheel_contact",
    "wheel_contact_point",
    "wheel_contact_normal",
    "wheel_contact_jacobian",
    "undesired_ground_contacts",
)


class _SensorView83:
    """The narrow plant interface used by the unmodified sensor collector."""

    def __init__(self, plant, sample_data) -> None:
        self.model = plant.model
        self.measurement_data = sample_data
        self.base_body_id = plant.base_body_id
        self.wheel_body_ids_by_leg = plant.wheel_body_ids_by_leg
        self.terrain_geom_ids = plant.terrain_geom_ids
        self.qpos_addresses = plant.qpos_addresses
        self.dof_addresses = plant.dof_addresses

    @property
    def joint_position(self) -> np.ndarray:
        return self.measurement_data.qpos[self.qpos_addresses].copy()

    @property
    def joint_velocity(self) -> np.ndarray:
        return self.measurement_data.qvel[self.dof_addresses].copy()


class D1CoherentMujocoSensorSource83(D1MujocoSensorSource):
    """Run forward on a private copy, then use the original sensor/noise code.

    With ``held_external_wrench=None`` the caller explicitly declares a
    zero-external-wrench experiment, as in the registered 49/50 tasks. The live
    plant clears external forces after stepping, so a nonzero push cannot be
    recovered from that cleared buffer. Such a caller must provide the known
    held wrench, indexed by the sample's end time, with shape (nbody, 6).
    This is an input ledger, not a current truth measurement sent to fusion.

    At most one forward refresh occurs per physical timestamp. Repeated reads
    still draw noise/increment sequence exactly as the original collector.
    A changed integration input at the same timestamp is rejected; reset is
    required after a deliberate same-time state edit. No live data is written.
    """

    def __init__(
        self,
        plant,
        *,
        noise: D1SensorNoise | None = None,
        seed: int | None = None,
        held_external_wrench: Callable[[float], np.ndarray] | None = None,
    ) -> None:
        if plant.sampling_mode != "legacy_mixed":
            raise ValueError("candidate83 requires the unchanged legacy_mixed plant")
        self._live_plant = plant
        self._sample_data = mujoco.MjData(plant.model)
        self._held_external_wrench = held_external_wrench
        self._sample_time: float | None = None
        self._input_snapshot: dict[str, np.ndarray] = {}
        self._held_wrench_snapshot: np.ndarray | None = None
        self.sampling_refresh_count = 0
        self.sensor_capture_count = 0
        super().__init__(_SensorView83(plant, self._sample_data), noise=noise, seed=seed)

    def reset(self, *, seed: int | None = None):
        self._sample_time = None
        self._input_snapshot = {}
        self._held_wrench_snapshot = None
        self.sampling_refresh_count = 0
        self.sensor_capture_count = 0
        return super().reset(seed=seed)

    def _refresh_sampling_copy(self) -> None:
        plant, live = self._live_plant, self._live_plant.data
        sample_time = float(live.time)
        if not np.isfinite(sample_time):
            raise ValueError("sample time must be finite")
        if self._held_external_wrench is None and np.any(live.xfrc_applied):
            raise ValueError("nonzero live wrench contradicts the declared zero-wrench protocol")
        wrench = (
            np.zeros_like(live.xfrc_applied)
            if self._held_external_wrench is None
            else np.asarray(self._held_external_wrench(sample_time), dtype=np.float64)
        )
        if wrench.shape != live.xfrc_applied.shape or not np.isfinite(wrench).all():
            raise ValueError("held external wrench must be finite with shape (nbody, 6)")
        if self._sample_time is not None:
            if sample_time < self._sample_time:
                raise ValueError("physical clock moved backwards; reset the source")
            if sample_time == self._sample_time:
                unchanged = all(
                    np.array_equal(getattr(live, name), previous)
                    for name, previous in self._input_snapshot.items()
                ) and np.array_equal(wrench, self._held_wrench_snapshot)
                if not unchanged:
                    raise ValueError("integration input changed at the same sample time; reset")
                return
        inputs = {name: getattr(live, name).copy() for name in _STATE_INPUTS}
        mujoco.mj_copyData(self._sample_data, plant.model, live)
        self._sample_data.xfrc_applied[:] = wrench
        mujoco.mj_forward(plant.model, self._sample_data)
        self.sampling_refresh_count += 1
        self._sample_time = sample_time
        self._input_snapshot = inputs
        self._held_wrench_snapshot = wrench.copy()

    def _capture(self):
        self._refresh_sampling_copy()
        packet = super()._capture()
        self.sensor_capture_count += 1
        return packet

    @property
    def sample_provenance(self) -> dict:
        if self._sample_time is None:
            raise RuntimeError("reset must precede reading sample provenance")
        return {
            "schema": "d1-coherent-sensor-sample83-v1",
            "sample_time_s": self._sample_time,
            "field_sample_times_s": {
                name: self._sample_time
                for name in (
                    "gyro_rad_s",
                    "specific_force_m_s2",
                    "joint_position_rad",
                    "joint_velocity_rad_s",
                    "wheel_contact",
                )
            },
            "sampling_phase": "post_integration_forward_on_private_copy",
            "live_plant_sampling_mode": self._live_plant.sampling_mode,
            "external_wrench_source": (
                "declared_zero_external_wrench"
                if self._held_external_wrench is None
                else "explicit_held_input_ledger"
            ),
            "sampling_forward_calls": self.sampling_refresh_count,
            "sensor_rne_post_constraint_calls": self.sensor_capture_count,
            "source_integration_steps": 0,
            "count_boundary": "successful source calls since reset; excludes model compilation "
            "and observer FK; in-flight exception counts are unknown",
        }


class D1CoherentSensorStateSource83(D1SensorStateSource):
    """Original queue/fusion behavior with only a different sensor collector."""

    def __init__(
        self,
        plant,
        *,
        noise: D1SensorNoise | None = None,
        seed: int | None = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.455),
        initial_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        delay_steps: int = 0,
        held_external_wrench: Callable[[float], np.ndarray] | None = None,
    ) -> None:
        if plant.sampling_mode != "legacy_mixed":
            raise ValueError("candidate83 requires the unchanged legacy_mixed plant")
        # Parent allocates exactly one original observer and retains validation,
        # reset/read/queue/ground-reference implementations. Its discarded plain
        # sensor collector allocates no model/data and performs no sensor draws.
        super().__init__(
            plant,
            noise=noise,
            seed=seed,
            initial_position=initial_position,
            initial_rpy=initial_rpy,
            delay_steps=delay_steps,
        )
        if self.estimator.wheel_center_jacobians:
            raise ValueError("candidate83 must retain the legacy observer Jacobian branch")
        self.measurements = D1CoherentMujocoSensorSource83(
            plant,
            noise=noise,
            seed=seed,
            held_external_wrench=held_external_wrench,
        )
        self.schema = "d1-coherent-sampling-legacy-fusion83-v1"


def control_state_times83(
    measured: D1StateEstimate,
    result: D1LatencyCompensationResult,
    *,
    coherent_measurement_packet: bool,
) -> dict:
    """Separate information time, prediction time and retained contact geometry.

    Supports unmodified none/full-CV results only. A constant body velocity is
    still an old measurement; treating its value as a current estimate requires
    the explicitly named constant-twist model. Measured contact geometry is
    never relabelled as current. This function changes no control values.
    """
    if result.method not in ("none", "constant_body_twist_first_order"):
        raise ValueError("unsupported compensation result")
    state = result.control_state
    if (
        state.control_time_s != measured.control_time_s
        or state.measurement_time_s != measured.measurement_time_s
        or state.age_s != measured.age_s
    ):
        raise ValueError("compensation changed measurement provenance")
    applied = result.status == "applied"
    sample_time, control_time = measured.measurement_time_s, measured.control_time_s
    predicted_time = control_time if applied else sample_time
    if result.estimate_time_s != predicted_time or result.applied_horizon_s != (
        measured.age_s if applied else 0.0
    ):
        raise ValueError("compensation result has inconsistent prediction time")
    # A method string alone cannot certify full CV: candidate50 deliberately
    # kept that string while restoring measured leg q. Reject such transforms
    # rather than assigning a falsely current timestamp to their leg positions.
    expected = prepare_d1_control_state(
        measured,
        latency_compensation="none" if result.method == "none" else "constant_velocity",
    )
    for field in dataclass_fields(expected):
        if field.name != "control_state" and getattr(result, field.name) != getattr(
            expected, field.name
        ):
            raise ValueError("result is not the unmodified registered none/full-CV treatment")
    for field in dataclass_fields(measured):
        actual = np.asarray(getattr(state, field.name))
        reference = np.asarray(getattr(expected.control_state, field.name))
        if (
            actual.dtype != reference.dtype
            or actual.shape != reference.shape
            or actual.tobytes() != reference.tobytes()
        ):
            raise ValueError("control fields differ from unmodified none/full-CV treatment")
    fields = {}
    groups = (
        (_PREDICTED_GEOMETRY, "first_order_geometry_prediction", predicted_time),
        (_HELD_VELOCITIES, "held_constant_by_twist_and_joint_velocity_model", predicted_time),
        (_ROTATED_VELOCITIES, "reexpressed_in_predicted_rotation", predicted_time),
        (_CONTACT_FIELDS, "retained_measurement", sample_time),
    )
    for names, prediction, estimate_time in groups:
        for name in names:
            fields[name] = {
                "information_time_s": sample_time if coherent_measurement_packet else None,
                "reported_packet_time_s": sample_time,
                "estimate_time_s": estimate_time,
                "operation": prediction if applied else "retained_measurement",
                "fresh_measurement": (
                    bool(coherent_measurement_packet) and sample_time == control_time
                ),
            }
    return {
        "schema": "d1-field-time-contract83-v1",
        "measurement_time_s": sample_time,
        "control_time_s": control_time,
        "measurement_age_s": measured.age_s,
        "coherent_measurement_packet": bool(coherent_measurement_packet),
        "information_time_status": (
            "coherent_packet"
            if coherent_measurement_packet
            else "legacy_packet_contains_mixed_sampling_phases"
        ),
        "field_times": fields,
        "mixed_estimate_times": len({v["estimate_time_s"] for v in fields.values()}) > 1,
        "whole_state_current": bool(coherent_measurement_packet)
        and all(v["estimate_time_s"] == control_time for v in fields.values()),
    }
