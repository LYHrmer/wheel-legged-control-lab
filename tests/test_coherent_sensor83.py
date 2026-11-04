"""Sampling-phase and provenance checks for the independent candidate83 source."""

from dataclasses import fields, replace

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.coherent_sensor83 import (
    D1CoherentMujocoSensorSource83,
    D1CoherentSensorStateSource83,
    control_state_times83,
)
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import (
    D1MujocoSensorSource,
    D1SensorNoise,
    D1SensorStateSource,
)
from wheel_legged_control.d1.state_estimation import (
    D1MujocoTruthStateSource,
    prepare_d1_control_state,
)


def _live_state(plant):
    names = (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "qacc",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "eq_active",
        "userdata",
    )
    return {"time": plant.data.time, **{name: getattr(plant.data, name).copy() for name in names}}


def _assert_live_equal(before, after):
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)


def test_copy_refresh_resolves_the_specific_legacy_final_substep_phase_error():
    plant = D1Plant()
    plant.step(np.zeros(16))
    # The frozen legacy path demonstrably combines old derived pose and new qpos.
    assert np.max(np.abs(plant.base_position - plant.data.qpos[:3])) > 1e-5
    np.testing.assert_allclose(
        plant.base_position,
        plant.data.qpos[:3] - 0.002 * plant.data.qvel[:3],
        atol=1e-12,
    )
    before = _live_state(plant)
    source = D1CoherentMujocoSensorSource83(plant)
    packet = source.reset()
    copied = source._sample_data
    np.testing.assert_array_equal(copied.xpos[plant.base_body_id], plant.data.qpos[:3])
    expected_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(expected_rotation, plant.data.qpos[3:7])
    np.testing.assert_allclose(copied.xmat[plant.base_body_id], expected_rotation, atol=1e-15)
    np.testing.assert_array_equal(packet.joint_position_rad, plant.joint_position)
    np.testing.assert_array_equal(packet.joint_velocity_rad_s, plant.joint_velocity)
    assert packet.time_s == plant.data.time
    assert source.sample_provenance["field_sample_times_s"] == {
        field: packet.time_s
        for field in (
            "gyro_rad_s",
            "specific_force_m_s2",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "wheel_contact",
        )
    }
    assert plant.sampling_mode == "legacy_mixed" and plant.measurement_data is plant.data
    _assert_live_equal(before, _live_state(plant))


def test_same_tick_reads_do_not_repeat_forward_or_change_live_state(monkeypatch):
    plant = D1Plant()
    source = D1CoherentMujocoSensorSource83(plant)
    forward = mujoco.mj_forward
    calls = []

    def counted_forward(model, data):
        assert data is not plant.data
        calls.append(data)
        return forward(model, data)

    monkeypatch.setattr(mujoco, "mj_forward", counted_forward)
    before = _live_state(plant)
    packets = [source.reset(), source.read(), source.read()]
    assert [packet.sequence for packet in packets] == [0, 1, 2]
    assert len(calls) == source.sampling_refresh_count == 1
    assert source.sensor_capture_count == 3
    _assert_live_equal(before, _live_state(plant))
    plant.step(np.zeros(16))
    before = _live_state(plant)
    source.read()
    assert len(calls) == source.sampling_refresh_count == 2
    _assert_live_equal(before, _live_state(plant))


def test_original_collector_noise_and_rng_stream_are_identical_on_the_same_sample():
    plant = D1Plant()
    noise = D1SensorNoise(
        gyro_std_rad_s=0.01,
        accelerometer_std_m_s2=0.03,
        encoder_position_std_rad=0.001,
        encoder_velocity_std_rad_s=0.02,
        gyro_bias_rad_s=(0.001, 0.002, 0.003),
        accelerometer_bias_m_s2=(0.01, 0.02, 0.03),
    )
    source = D1CoherentMujocoSensorSource83(plant, noise=noise, seed=17)
    # Same frozen collector, explicitly given the refreshed sensor-only view.
    reference = D1MujocoSensorSource(source.plant, noise=noise, seed=17)
    for step in range(5):
        if step:
            plant.step(np.full(16, 0.1))
        coherent = source.reset() if step == 0 else source.read()
        original = reference.reset() if step == 0 else reference.read()
        for field in fields(coherent):
            np.testing.assert_array_equal(
                getattr(coherent, field.name),
                getattr(original, field.name),
            )
        assert source._rng.bit_generator.state == reference._rng.bit_generator.state


def test_extra_sensor_reads_leave_open_loop_integration_and_warmstart_unchanged():
    reference, observed = D1Plant(), D1Plant()
    source = D1CoherentMujocoSensorSource83(observed, seed=29)
    source.reset()
    for step in range(10):
        torque = np.sin(np.arange(16) + step) * 0.2
        reference.step(torque)
        observed.step(torque)
        source.read()
        source.read()
        _assert_live_equal(_live_state(reference), _live_state(observed))
    assert source.sampling_refresh_count == 11
    assert source.sample_provenance["source_integration_steps"] == 0


def test_original_delay_queue_and_legacy_observer_branch_are_preserved():
    plant = D1Plant()
    original = D1SensorStateSource(plant, delay_steps=2, seed=43)
    source = D1CoherentSensorStateSource83(plant, delay_steps=2, seed=43)
    original.reset()
    source.reset()
    states = []
    for _ in range(3):
        plant.step(np.zeros(16))
        old, new = original.read(), source.read()
        states.append(new)
        assert new.sequence == old.sequence
        assert new.measurement_time_s == old.measurement_time_s
        assert new.control_time_s == old.control_time_s
        assert (
            source.measurements._rng.bit_generator.state
            == original.measurements._rng.bit_generator.state
        )
    assert [state.measurement_time_s for state in states] == pytest.approx([0, 0, 0.01])
    assert [state.age_s for state in states] == pytest.approx([0.01, 0.02, 0.02])
    assert not source.estimator.wheel_center_jacobians
    assert source.estimator.schema == original.estimator.schema
    assert source.estimator.attitude_time_constant_s == original.estimator.attitude_time_constant_s
    assert source.estimator.velocity_time_constant_s == original.estimator.velocity_time_constant_s
    assert source.schema != original.schema


def test_changed_same_time_input_and_backwards_clock_require_reset():
    plant = D1Plant()
    source = D1CoherentMujocoSensorSource83(plant)
    source.reset()
    plant.data.ctrl[0] = 0.1
    with pytest.raises(ValueError, match="same sample time"):
        source.read()
    source.reset()
    plant.step(np.zeros(16))
    source.read()
    plant.reset()
    with pytest.raises(ValueError, match="backwards"):
        source.read()
    assert source.reset().time_s == 0


def test_synchronized_plant_is_rejected_before_silent_jacobian_switch():
    plant = D1Plant(sampling_mode="synchronized")
    with pytest.raises(ValueError, match="legacy_mixed"):
        D1CoherentSensorStateSource83(plant)
    with pytest.raises(ValueError, match="legacy_mixed"):
        D1CoherentMujocoSensorSource83(plant)


def test_held_external_input_is_explicit_and_never_written_to_live_data():
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, 2.0)))
    force = np.asarray((30.0, 0.0, 0.0))
    plant.step(np.zeros(16), push_force_world_n=force)
    assert not plant.data.xfrc_applied.any()
    wrench = np.zeros_like(plant.data.xfrc_applied)
    wrench[plant.base_body_id, :3] = force
    source = D1CoherentMujocoSensorSource83(plant, held_external_wrench=lambda time: wrench)
    before = _live_state(plant)
    held = source.reset()
    zero = D1CoherentMujocoSensorSource83(plant).reset()
    assert np.linalg.norm(held.specific_force_m_s2 - zero.specific_force_m_s2) > 0.1
    assert source.sample_provenance["external_wrench_source"] == "explicit_held_input_ledger"
    _assert_live_equal(before, _live_state(plant))


def test_cv_metadata_never_calls_measured_contacts_or_old_information_current():
    raw = D1MujocoTruthStateSource(D1Plant()).reset()
    raw = replace(raw, control_time_s=0.02)
    result = prepare_d1_control_state(raw, latency_compensation="constant_velocity")
    info = control_state_times83(raw, result, coherent_measurement_packet=True)
    assert info["mixed_estimate_times"] and not info["whole_state_current"]
    assert info["measurement_age_s"] == result.control_state.age_s == 0.02
    for field in ("wheel_contact", "wheel_contact_point", "wheel_contact_jacobian"):
        assert info["field_times"][field]["estimate_time_s"] == 0
        assert not info["field_times"][field]["fresh_measurement"]
    velocity = info["field_times"]["base_linear_velocity_body"]
    assert velocity["information_time_s"] == 0 and velocity["estimate_time_s"] == 0.02
    assert velocity["operation"] == "held_constant_by_twist_and_joint_velocity_model"
    assert not velocity["fresh_measurement"]
    disabled = prepare_d1_control_state(raw)
    old = control_state_times83(raw, disabled, coherent_measurement_packet=True)
    assert not old["mixed_estimate_times"] and not old["whole_state_current"]
    current = replace(raw, measurement_time_s=0.02)
    result = prepare_d1_control_state(current)
    coherent = control_state_times83(current, result, coherent_measurement_packet=True)
    legacy = control_state_times83(current, result, coherent_measurement_packet=False)
    assert coherent["whole_state_current"] and not legacy["whole_state_current"]
    assert legacy["field_times"]["base_rotation"]["information_time_s"] is None
    assert not legacy["field_times"]["base_rotation"]["fresh_measurement"]
    with pytest.raises(ValueError, match="prediction time"):
        control_state_times83(
            raw, replace(disabled, estimate_time_s=0.02), coherent_measurement_packet=True
        )


def test_metadata_rejects_the_old_measured_leg_q_transform_instead_of_mislabeling_it():
    raw = D1MujocoTruthStateSource(D1Plant()).reset()
    velocity = np.zeros(16)
    velocity[0] = 0.1
    raw = replace(raw, control_time_s=0.02, joint_velocity=velocity)
    cv = prepare_d1_control_state(raw, latency_compensation="constant_velocity")
    assert cv.status == "applied"
    position = cv.control_state.joint_position.copy()
    position[0] = raw.joint_position[0]
    mixed = replace(cv, control_state=replace(cv.control_state, joint_position=position))
    with pytest.raises(ValueError, match="control fields differ"):
        control_state_times83(raw, mixed, coherent_measurement_packet=True)
