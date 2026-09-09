"""Tick-owned state providers for the synchronized D1 runtime.

``advance(tick)`` publishes at most one new estimate per physical control tick.
``read()`` and ``ground_reference()`` only return that publication; neither
advances the filter nor draws noise. The existing three source implementations
remain distinct Adapters, selected by an explicit, validated configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .model import D1Plant
from .sensor_estimation import D1SensorNoise, D1SensorStateSource
from .state_estimation import (
    D1EstimatorImpairments,
    D1MujocoTruthStateSource,
    D1NoisyDelayedStateSource,
    D1StateEstimate,
)
from .training_terrain import TrainingGroundReference

PROVIDER_SCHEMAS = {
    "oracle": "d1-synchronized-oracle-v1",
    "truth_impairment": "d1-synchronized-truth-impairment-v1",
    "imu_encoder_fusion": "d1-synchronized-imu-encoder-fusion-v1",
}


@dataclass(frozen=True, slots=True)
class D1StateProviderConfig:
    kind: str
    impairments: D1EstimatorImpairments | None = None
    sensor_noise: D1SensorNoise | None = None
    sensor_delay_steps: int = 0
    initial_position_m: tuple[float, float, float] | None = None
    initial_rpy_rad: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in PROVIDER_SCHEMAS:
            raise ValueError(f"provider kind must be one of {tuple(PROVIDER_SCHEMAS)}")
        if self.impairments is not None and not isinstance(
            self.impairments, D1EstimatorImpairments
        ):
            raise TypeError("impairments must be D1EstimatorImpairments")
        if self.sensor_noise is not None and not isinstance(self.sensor_noise, D1SensorNoise):
            raise TypeError("sensor_noise must be D1SensorNoise")
        if isinstance(self.sensor_delay_steps, bool) or not isinstance(
            self.sensor_delay_steps, Integral
        ):
            raise TypeError("sensor_delay_steps must be a non-negative integer")
        if self.sensor_delay_steps < 0:
            raise ValueError("sensor_delay_steps must be a non-negative integer")
        object.__setattr__(self, "sensor_delay_steps", int(self.sensor_delay_steps))
        if self.kind == "imu_encoder_fusion":
            if self.impairments is not None:
                raise ValueError("fusion cannot consume truth-level impairments")
            for field in ("initial_position_m", "initial_rpy_rad"):
                value = getattr(self, field)
                if value is None:
                    raise ValueError(
                        "fusion requires an explicit initial position and attitude prior"
                    )
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (3,) or not np.isfinite(array).all():
                    raise ValueError(f"{field} must be three finite values")
                object.__setattr__(self, field, tuple(float(x) for x in array))
        else:
            if (
                any(
                    value is not None
                    for value in (self.sensor_noise, self.initial_position_m, self.initial_rpy_rad)
                )
                or self.sensor_delay_steps
            ):
                raise ValueError(
                    "sensor noise, delay and placement priors require imu_encoder_fusion"
                )
            if self.kind == "oracle" and self.impairments is not None:
                raise ValueError("oracle cannot consume truth-level impairments")
            if self.kind == "truth_impairment" and self.impairments is None:
                raise ValueError("truth_impairment requires an explicit impairment configuration")

    @property
    def source_schema(self) -> str:
        return PROVIDER_SCHEMAS[self.kind]


class D1StateProvider:
    """Publish one cached state/ground pair after the caller advances physics.

    Construct through ``build_d1_state_provider``. Plant reset/step belongs to
    the runtime, not this module. Repeating ``advance`` at the same tick is
    idempotent only while the physical clock still names that same tick.
    """

    def __init__(self, plant, config, source, ground_query):
        self.config = config
        self._plant, self._source, self._ground_query = plant, source, ground_query
        self._state: D1StateEstimate | None = None
        self._ground: TrainingGroundReference | None = None
        self._tick = -1

    @property
    def source_schema(self) -> str:
        return self.config.source_schema

    @property
    def tick(self) -> int:
        return self._tick

    def _clock(self) -> float:
        return float(self._plant.measurement_data.time)

    def _publish(self, state: D1StateEstimate, tick: int) -> D1StateEstimate:
        if not np.isclose(state.control_time_s, self._clock(), rtol=0, atol=1e-10):
            raise ValueError("source publication time differs from the synchronized physical clock")
        if state.sequence != tick:
            raise ValueError("source sequence differs from the control tick")
        ground = self._ground_query(state)
        if not isinstance(ground, TrainingGroundReference):
            raise TypeError("ground query must return TrainingGroundReference")
        if not np.isfinite((ground.height_m, ground.pitch_rad, ground.roll_rad)).all():
            raise ValueError("ground reference must be finite")
        self._state, self._ground, self._tick = state, ground, tick
        return state

    def reset(self, *, seed: int | None = None) -> D1StateEstimate:
        self._state, self._ground, self._tick = None, None, -1
        return self._publish(self._source.reset(seed=seed), 0)

    def advance(self, tick: int) -> D1StateEstimate:
        state = self.read()
        if isinstance(tick, bool) or not isinstance(tick, Integral):
            raise TypeError("tick must be an integer")
        if tick == self._tick:
            if not np.isclose(self._clock(), state.control_time_s, rtol=0, atol=1e-10):
                raise ValueError("physical time advanced without a new control tick")
            return state
        if tick != self._tick + 1:
            raise ValueError("control ticks must be consecutive; resets require reset()")
        expected_time = state.control_time_s + self._plant.control_dt
        if not np.isclose(self._clock(), expected_time, rtol=0, atol=1e-10):
            raise ValueError("advance requires exactly one completed physical control interval")
        return self._publish(self._source.read(), int(tick))

    def read(self) -> D1StateEstimate:
        if self._state is None:
            raise RuntimeError("provider reset must precede read/advance")
        return self._state

    def ground_reference(self) -> TrainingGroundReference:
        self.read()
        return self._ground


def build_d1_state_provider(
    plant: D1Plant,
    config: D1StateProviderConfig,
    *,
    oracle_ground_query: Callable[[float, float], TrainingGroundReference] | None = None,
) -> D1StateProvider:
    """Bind a real source Adapter; never silently replace fusion with truth.

    Oracle ground queries are explicit for both oracle-based kinds. Fusion
    rejects such a query and takes ground from its own measured support plane.
    """
    if not isinstance(config, D1StateProviderConfig):
        raise TypeError("config must be D1StateProviderConfig")
    if plant.sampling_mode != "synchronized":
        raise ValueError("v3 providers require sampling_mode='synchronized'")
    if config.kind == "imu_encoder_fusion":
        if oracle_ground_query is not None:
            raise ValueError("fusion must not be given an oracle ground query")
        source = D1SensorStateSource(
            plant,
            noise=config.sensor_noise,
            delay_steps=config.sensor_delay_steps,
            initial_position=config.initial_position_m,
            initial_rpy=config.initial_rpy_rad,
        )
        ground_query = source.ground_reference
    else:
        if not callable(oracle_ground_query):
            raise ValueError("oracle-based providers require an explicit oracle_ground_query")
        source = (
            D1MujocoTruthStateSource(plant)
            if config.kind == "oracle"
            else D1NoisyDelayedStateSource(plant, impairments=config.impairments)
        )

        def ground_query(state):
            return oracle_ground_query(float(state.base_position[0]), float(state.base_position[1]))

    return D1StateProvider(plant, config, source, ground_query)
