from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise
from wheel_legged_control.d1.state_estimation import D1EstimatorImpairments, D1StateEstimate
from wheel_legged_control.d1.state_provider import (
    D1StateProvider,
    D1StateProviderConfig,
    build_d1_state_provider,
)
from wheel_legged_control.d1.training_terrain import TrainingGroundReference


def state_at(tick=0, time_s=0.0):
    return D1StateEstimate(
        sequence=tick,
        control_time_s=time_s,
        measurement_time_s=time_s,
        base_position=np.asarray((0.0, 0.0, 0.455)),
        base_rotation=np.eye(3),
        base_linear_velocity_body=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        base_linear_velocity_world=np.zeros(3),
        base_angular_velocity_world=np.zeros(3),
        joint_position=np.zeros(16),
        joint_velocity=np.zeros(16),
        foot_position=np.zeros((4, 3)),
        foot_jacobian=np.zeros((4, 3, 4)),
        wheel_contact=np.zeros(4, dtype=bool),
        wheel_contact_point=np.zeros((4, 3)),
        wheel_contact_normal=np.zeros((4, 3)),
        wheel_contact_jacobian=np.zeros((4, 3, 4)),
        undesired_ground_contacts=0,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "estimated"},
        {"kind": "oracle", "sensor_noise": D1SensorNoise()},
        {"kind": "oracle", "impairments": D1EstimatorImpairments()},
        {"kind": "truth_impairment"},
        {"kind": "imu_encoder_fusion"},
        {"kind": "imu_encoder_fusion", "initial_position_m": (0, 0, 0.455)},
        {"kind": "oracle", "sensor_delay_steps": True},
        {"kind": "oracle", "sensor_delay_steps": -1},
        {"kind": "oracle", "initial_rpy_rad": (0, 0, 0)},
        {"kind": "imu_encoder_fusion", "impairments": D1EstimatorImpairments()},
    ],
)
def test_provider_configuration_rejects_ambiguous_source_meanings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        D1StateProviderConfig(**kwargs)


class CountingSource:
    def __init__(self, clock):
        self.clock = clock
        self.read_calls = 0

    def reset(self, *, seed=None):
        self.sequence = 0
        self.rng = np.random.default_rng(seed)
        return self.capture()

    def capture(self):
        return replace(
            state_at(self.sequence, self.clock.time),
            base_position=np.asarray((self.rng.normal(), 0, 0.455)),
        )

    def read(self):
        self.read_calls += 1
        self.sequence += 1
        return self.capture()


def test_cached_read_and_duplicate_advance_neither_capture_nor_consume_rng():
    clock = SimpleNamespace(time=0.0)
    plant = SimpleNamespace(measurement_data=clock, control_dt=0.01)
    source = CountingSource(clock)
    ground_calls = []

    def ground_query(state):
        ground_calls.append(state.sequence)
        return TrainingGroundReference(state.base_position[0], 0, 0)

    provider = D1StateProvider(plant, D1StateProviderConfig("oracle"), source, ground_query)
    with pytest.raises(RuntimeError, match="reset"):
        provider.read()
    initial = provider.reset(seed=53)
    rng = np.random.default_rng(53)
    assert initial.base_position[0] == rng.normal()
    for _ in range(10):
        assert provider.read() is initial
        assert provider.advance(0) is initial
        assert provider.ground_reference().height_m == initial.base_position[0]
    assert ground_calls == [0] and source.read_calls == 0
    clock.time = 0.01
    next_state = provider.advance(1)
    assert next_state.base_position[0] == rng.normal()
    assert provider.advance(1) is next_state
    assert source.read_calls == 1 and ground_calls == [0, 1]
    clock.time = 0.02
    with pytest.raises(ValueError, match="without a new"):
        provider.advance(1)
    assert provider.read() is next_state  # Explicitly the last publication.
    assert source.read_calls == 1


def test_clock_and_tick_failures_happen_before_source_advancement():
    clock = SimpleNamespace(time=0.0)
    source = CountingSource(clock)
    provider = D1StateProvider(
        SimpleNamespace(measurement_data=clock, control_dt=0.01),
        D1StateProviderConfig("oracle"),
        source,
        lambda _: TrainingGroundReference(0, 0, 0),
    )
    provider.reset(seed=0)
    for tick in (-1, 2):
        with pytest.raises(ValueError, match="consecutive"):
            provider.advance(tick)
    with pytest.raises(TypeError, match="integer"):
        provider.advance(True)
    with pytest.raises(ValueError, match="exactly one"):
        provider.advance(1)
    assert source.read_calls == 0


def fusion_config(delay=0):
    return D1StateProviderConfig(
        "imu_encoder_fusion",
        sensor_noise=D1SensorNoise(gyro_std_rad_s=0.002, accelerometer_std_m_s2=0.03),
        sensor_delay_steps=delay,
        initial_position_m=(0, 0, 0.455),
        initial_rpy_rad=(0, 0, 0),
    )


@pytest.mark.parametrize("kind", ("oracle", "truth_impairment", "imu_encoder_fusion"))
def test_real_provider_ticks_are_idempotent_and_reset_replays_noise(kind):
    plant = D1Plant(sampling_mode="synchronized")
    config = (
        fusion_config()
        if kind == "imu_encoder_fusion"
        else D1StateProviderConfig(
            kind, impairments=D1EstimatorImpairments(base_position_std_m=0.001)
        )
        if kind == "truth_impairment"
        else D1StateProviderConfig(kind)
    )
    ground_query = (
        None if kind == "imu_encoder_fusion" else lambda x, y: TrainingGroundReference(0, 0, 0)
    )
    provider = build_d1_state_provider(plant, config, oracle_ground_query=ground_query)

    def rollout(extra_reads):
        plant.reset()
        result = [provider.reset(seed=617)]
        for tick in range(1, 5):
            plant.step(np.zeros(16))
            current = provider.advance(tick)
            result.append(current)
            for _ in range(extra_reads):
                assert provider.advance(tick) is current
                assert provider.read() is current
                assert provider.ground_reference() is provider.ground_reference()
        return result

    first, second = rollout(0), rollout(4)
    for a, b in zip(first, second, strict=True):
        for field in fields(a):
            np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name))
    assert provider.tick == 4 and provider.read().control_time_s == pytest.approx(0.04)
    assert kind.replace("_", "-") in provider.source_schema


def test_fusion_factory_rejects_oracle_query_and_legacy_sampling():
    plant = D1Plant(sampling_mode="synchronized")
    with pytest.raises(ValueError, match="oracle ground query"):
        build_d1_state_provider(plant, fusion_config(), oracle_ground_query=lambda x, y: None)
    with pytest.raises(ValueError, match="explicit oracle_ground_query"):
        build_d1_state_provider(plant, D1StateProviderConfig("oracle"))
    plant.sampling_mode = "legacy_mixed"
    with pytest.raises(ValueError, match="synchronized"):
        build_d1_state_provider(plant, fusion_config())


def test_fusion_delays_measurements_not_the_publication_clock():
    plant = D1Plant(sampling_mode="synchronized")
    provider = build_d1_state_provider(plant, fusion_config(delay=2))
    provider.reset(seed=43)
    ages = []
    for tick in range(1, 5):
        plant.step(np.zeros(16))
        state = provider.advance(tick)
        assert state.sequence == tick
        ages.append(state.age_s)
    np.testing.assert_allclose(ages, (0.01, 0.02, 0.02, 0.02), atol=1e-12)


def test_fusion_does_not_query_base_truth_properties(monkeypatch):
    plant = D1Plant(sampling_mode="synchronized")
    provider = build_d1_state_provider(plant, fusion_config())
    comparison = build_d1_state_provider(plant, fusion_config())
    clean = provider.reset(seed=29)
    comparison.reset(seed=29)
    plant.step(np.zeros(16))
    clean_next = comparison.advance(1)

    def forbidden(*args, **kwargs):
        pytest.fail("base truth property leaked into sensor fusion")

    monkeypatch.setattr(D1Plant, "base_position", property(forbidden))
    monkeypatch.setattr(D1Plant, "base_velocity", forbidden)
    monkeypatch.setattr(D1Plant, "training_ground_reference", forbidden)
    assert provider.read() is clean
    poisoned_next = provider.advance(1)
    for field in fields(clean_next):
        np.testing.assert_array_equal(
            getattr(clean_next, field.name), getattr(poisoned_next, field.name)
        )
    assert provider.ground_reference() == comparison.ground_reference()
