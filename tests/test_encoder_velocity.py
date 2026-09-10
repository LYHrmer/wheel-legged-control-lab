"""Tests for the encoder velocity observer, exercised through its public API.

No mocks and no plant model: every expectation is either hand-derived from the
documented recursions or a property of the estimator itself.
"""

from __future__ import annotations

import dataclasses
import math
import random

import numpy as np
import pytest

from wheel_legged_control.encoder_velocity import (
    METHODS,
    EncoderVelocityConfig,
    EncoderVelocityEstimate,
    EncoderVelocityObserver,
)

DT = 0.002
T0 = 12.5  # non-zero time origin everywhere


def observer(method: str, **overrides: float) -> EncoderVelocityObserver:
    return EncoderVelocityObserver(EncoderVelocityConfig(method=method, **overrides))


def started(method: str, position_rad: float = 0.0, **overrides: float):
    obs = observer(method, **overrides)
    obs.reset(T0, position_rad, known_initial_rest=True)
    return obs


def replay(obs: EncoderVelocityObserver, positions, t0: float = T0):
    dt = obs.config.dt
    return [
        obs.update(t0 + (k + 1) * dt, q) for k, q in enumerate(positions)
    ]


# --------------------------------------------------------------------------
# configuration contract
# --------------------------------------------------------------------------


def test_default_config_matches_contract():
    cfg = EncoderVelocityConfig()
    assert (cfg.method, cfg.dt, cfg.cutoff_hz, cfg.alpha, cfg.beta) == (
        "difference_lowpass",
        0.002,
        20.0,
        0.25,
        0.04,
    )


def test_config_and_estimate_are_frozen():
    cfg = EncoderVelocityConfig()
    est = EncoderVelocityObserver(cfg).reset(T0, 1.0, known_initial_rest=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dt = 0.001
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.velocity_rad_s = 3.0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "field, value",
    [
        ("dt", True),
        ("dt", "0.002"),
        ("dt", complex(0.002, 0.0)),
        ("dt", [0.002]),
        ("cutoff_hz", False),
        ("cutoff_hz", "20"),
        ("cutoff_hz", complex(20.0, 1.0)),
        ("cutoff_hz", (20.0,)),
        ("alpha", True),
        ("alpha", "0.25"),
        ("alpha", complex(0.25, 0.0)),
        ("beta", False),
        ("beta", b"0.04"),
        ("beta", [0.04, 0.04]),
    ],
)
def test_config_rejects_non_scalar_numeric_fields(method, field, value):
    with pytest.raises(TypeError):
        EncoderVelocityConfig(method=method, **{field: value})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "field",
    ["dt", "cutoff_hz", "alpha", "beta"],
)
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_config_rejects_non_finite_fields(method, field, value):
    with pytest.raises(ValueError):
        EncoderVelocityConfig(method=method, **{field: value})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "overrides",
    [
        {"dt": 0.0},
        {"dt": -0.002},
        {"cutoff_hz": 0.0},
        {"cutoff_hz": -20.0},
        {"cutoff_hz": 250.0},  # exactly Nyquist for dt=0.002
        {"cutoff_hz": 400.0},  # above Nyquist
        {"alpha": 0.0},
        {"alpha": -0.1},
        {"alpha": 1.5},
        {"beta": 0.0},
        {"beta": -0.04},
        {"beta": 3.5},  # >= 4 - 2*alpha = 3.5
    ],
)
def test_config_rejects_out_of_range_values_for_both_methods(method, overrides):
    with pytest.raises(ValueError):
        EncoderVelocityConfig(method=method, **overrides)


@pytest.mark.parametrize("method", ["", "kalman", "difference", "ALPHA_BETA"])
def test_config_rejects_unknown_method(method):
    with pytest.raises(ValueError):
        EncoderVelocityConfig(method=method)


@pytest.mark.parametrize("method", [None, 1.0, ["alpha_beta"]])
def test_config_rejects_non_string_method(method):
    with pytest.raises(TypeError):
        EncoderVelocityConfig(method=method)


def test_alpha_one_and_beta_near_limit_are_accepted():
    cfg = EncoderVelocityConfig(method="alpha_beta", alpha=1.0, beta=1.999)
    assert (cfg.alpha, cfg.beta) == (1.0, 1.999)


# --------------------------------------------------------------------------
# hand-computed recursions
# --------------------------------------------------------------------------


def test_difference_lowpass_hand_computed_first_two_steps():
    # r = exp(-2*pi*20*0.002); q rises by 0.002 rad each 0.002 s, so raw = 1 rad/s.
    # v1 = r*0 + (1-r)*1 = 1-r ;  v2 = r*(1-r) + (1-r)*1 = 1-r**2
    r = math.exp(-2.0 * math.pi * 20.0 * DT)
    obs = started("difference_lowpass", 1.0)
    first, second = replay(obs, [1.002, 1.004])

    assert first.time_s == pytest.approx(T0 + DT, abs=0.0, rel=1e-15)
    assert first.position_rad == 1.002  # measured position is reported as-is
    assert first.velocity_rad_s == pytest.approx(1.0 - r, abs=1e-15)
    assert second.position_rad == 1.004
    assert second.velocity_rad_s == pytest.approx(1.0 - r * r, abs=1e-15)


def test_alpha_beta_hand_computed_first_two_steps():
    # alpha=0.25, beta=0.04, dt=0.002, qhat0=1.0, vhat0=0
    # step 1: qpred=1.0, e=0.002 -> qhat=1.0005, vhat=0.04
    # step 2: qpred=1.00058, e=0.00342 -> qhat=1.001435, vhat=0.1084
    obs = started("alpha_beta", 1.0)
    first, second = replay(obs, [1.002, 1.004])

    assert first.position_rad == pytest.approx(1.0005, abs=1e-15)
    assert first.velocity_rad_s == pytest.approx(0.04, abs=1e-15)
    assert second.position_rad == pytest.approx(1.001435, abs=1e-15)
    assert second.velocity_rad_s == pytest.approx(0.1084, abs=1e-15)


@pytest.mark.parametrize("method", METHODS)
def test_reset_reports_zero_velocity_at_the_reset_instant(method):
    est = started(method, -0.75).state
    assert est == EncoderVelocityEstimate(T0, -0.75, 0.0)


# --------------------------------------------------------------------------
# causality / statelessness properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_future_suffix_cannot_change_the_shared_prefix(method):
    prefix = [0.001 * k for k in range(1, 21)]
    suffix_a = [0.02 + 0.001 * k for k in range(1, 11)]
    suffix_b = [0.02 - 0.05 * k for k in range(1, 11)]

    run_a = replay(started(method), prefix + suffix_a)
    run_b = replay(started(method), prefix + suffix_b)

    assert run_a[: len(prefix)] == run_b[: len(prefix)]
    assert run_a[len(prefix) :] != run_b[len(prefix) :]


@pytest.mark.parametrize("method", METHODS)
def test_deterministic_replay(method):
    rng = random.Random(4242)
    positions = [rng.uniform(-2.0, 2.0) for _ in range(50)]
    first = replay(started(method, 0.3), positions)
    second = replay(started(method, 0.3), positions)
    assert first == second


@pytest.mark.parametrize("method", METHODS)
def test_repeated_state_reads_do_not_advance(method):
    obs = started(method, 0.5)
    replay(obs, [0.5 + 0.001 * k for k in range(1, 6)])
    snapshots = [obs.state for _ in range(4)]
    assert snapshots.count(snapshots[0]) == len(snapshots)


def test_state_requires_reset():
    with pytest.raises(RuntimeError):
        _ = observer("difference_lowpass").state


@pytest.mark.parametrize("method", METHODS)
def test_update_requires_reset(method):
    with pytest.raises(RuntimeError):
        observer(method).update(T0 + DT, 0.0)


# --------------------------------------------------------------------------
# reset semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_reset_clears_all_history(method):
    obs = started(method, 0.0)
    replay(obs, [0.01 * k for k in range(1, 31)])  # build up a large velocity
    assert abs(obs.state.velocity_rad_s) > 1.0

    fresh = obs.reset(3.0, 0.0, known_initial_rest=True)
    assert fresh == EncoderVelocityEstimate(3.0, 0.0, 0.0)

    after_reset = replay(obs, [0.002, 0.004], t0=3.0)
    reference = replay(started(method, 0.0), [0.002, 0.004])
    assert [e.velocity_rad_s for e in after_reset] == [
        e.velocity_rad_s for e in reference
    ]


@pytest.mark.parametrize("method", METHODS)
def test_reset_requires_explicit_keyword_rest_flag(method):
    obs = observer(method)
    with pytest.raises(TypeError):
        obs.reset(T0, 0.0)
    with pytest.raises(TypeError):
        obs.reset(T0, 0.0, True)


@pytest.mark.parametrize("flag", [1, 1.0, "yes", [True], None])
def test_reset_rejects_non_bool_rest_flag(flag):
    with pytest.raises(TypeError):
        observer("alpha_beta").reset(T0, 0.0, known_initial_rest=flag)


def test_reset_rejects_false_rest_flag():
    with pytest.raises(ValueError):
        observer("alpha_beta").reset(T0, 0.0, known_initial_rest=False)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad_time", [-1e-9, -T0, math.inf, math.nan])
def test_reset_rejects_bad_sample_time(method, bad_time):
    with pytest.raises(ValueError):
        observer(method).reset(bad_time, 0.0, known_initial_rest=True)


@pytest.mark.parametrize("method", METHODS)
def test_reset_accepts_zero_and_nonzero_time_origin(method):
    assert observer(method).reset(0.0, 0.0, known_initial_rest=True).time_s == 0.0
    assert observer(method).reset(987.25, 0.0, known_initial_rest=True).time_s == 987.25


# --------------------------------------------------------------------------
# clock and sample validation, state preservation on rejection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("jitter", [1e-11, -1e-11, 9.9e-11, -9.9e-11, 0.0])
def test_clock_accepts_jitter_within_absolute_tolerance(method, jitter):
    obs = started(method)
    est = obs.update(T0 + DT + jitter, 0.001)
    assert est.time_s == T0 + DT + jitter


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "dt_multiple, extra",
    [
        (0, 0.0),  # repeated timestamp
        (-1, 0.0),  # reversed
        (-2, 0.0),
        (2, 0.0),  # skipped beat
        (3, 0.0),
        (1, 1.1e-10),  # just outside the absolute tolerance
        (1, -1.1e-10),
    ],
)
def test_bad_clock_is_rejected_without_changing_state(method, dt_multiple, extra):
    obs = started(method, 0.4)
    replay(obs, [0.401, 0.402])
    before = obs.state
    with pytest.raises(ValueError):
        obs.update(before.time_s + dt_multiple * DT + extra, 0.403)
    assert obs.state == before


def test_clock_tolerance_is_absolute_not_relative():
    # A large time origin must not buy a proportionally larger jitter budget.
    obs = observer("difference_lowpass")
    obs.reset(1.0e6, 0.0, known_initial_rest=True)
    with pytest.raises(ValueError):
        obs.update(1.0e6 + DT + 1.0e-6, 0.001)
    assert obs.state.time_s == 1.0e6


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "bad_position",
    [math.inf, -math.inf, math.nan],
)
def test_non_finite_position_is_rejected_without_changing_state(method, bad_position):
    obs = started(method, 0.4)
    before = obs.state
    with pytest.raises(ValueError):
        obs.update(T0 + DT, bad_position)
    assert obs.state == before


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad_position", [True, "0.4", complex(0.4, 0.0), [0.4]])
def test_non_scalar_position_is_rejected_without_changing_state(method, bad_position):
    obs = started(method, 0.4)
    before = obs.state
    with pytest.raises(TypeError):
        obs.update(T0 + DT, bad_position)
    assert obs.state == before


@pytest.mark.parametrize("method", METHODS)
def test_huge_int_position_is_rejected_without_changing_state(method):
    obs = started(method, 0.4)
    before = obs.state
    with pytest.raises(ValueError):
        obs.update(T0 + DT, 10**400)
    assert obs.state == before


@pytest.mark.parametrize("method", METHODS)
def test_arithmetic_overflow_leaves_no_partial_update(method):
    # dt is tiny, so a huge position step overflows the velocity arithmetic.
    obs = observer(method, dt=1e-9)
    obs.reset(0.0, 0.0, known_initial_rest=True)
    before = obs.state
    with pytest.raises(ValueError):
        obs.update(1e-9, 1e308)
    assert obs.state == before

    # The observer is still usable at the very same expected instant.
    good = obs.update(1e-9, 1e-9)
    assert math.isfinite(good.velocity_rad_s)
    assert math.isfinite(good.position_rad)


# --------------------------------------------------------------------------
# estimation behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("held_position", [0.0, -4.25, 1234.5])
def test_identical_positions_keep_velocity_at_zero(method, held_position):
    # Purely a statement about the estimator input/output: an unchanging encoder
    # reading yields zero estimated velocity, with no actuator assumption.
    obs = started(method, held_position)
    estimates = replay(obs, [held_position] * 30)
    assert all(e.velocity_rad_s == 0.0 for e in estimates)
    assert all(e.position_rad == pytest.approx(held_position, abs=1e-12) for e in estimates)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("velocity", [0.5, -3.0, 12.0])
def test_constant_velocity_converges(method, velocity):
    obs = started(method, 0.0)
    positions = [velocity * k * DT for k in range(1, 601)]
    estimates = replay(obs, positions)

    assert abs(estimates[-1].velocity_rad_s - velocity) < 1e-3 * max(1.0, abs(velocity))
    # monotone-ish improvement: the tail is much closer than the first samples
    assert abs(estimates[-1].velocity_rad_s - velocity) < abs(
        estimates[0].velocity_rad_s - velocity
    )
    assert estimates[-1].position_rad == pytest.approx(positions[-1], abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_velocity_noise_is_lower_than_raw_difference(method):
    """Estimator-level noise metric only; no closed-loop claim is implied."""
    rng = random.Random(20240501)
    truth_velocity = 2.0
    noise = [rng.gauss(0.0, 1e-4) for _ in range(400)]
    positions = [truth_velocity * (k + 1) * DT + n for k, n in enumerate(noise)]

    obs = started(method, 0.0)
    estimates = replay(obs, positions)

    warmup = 150
    raw = [
        (positions[k] - positions[k - 1]) / DT for k in range(warmup, len(positions))
    ]
    est = [e.velocity_rad_s for e in estimates[warmup:]]

    def rms_error(values):
        return math.sqrt(
            sum((v - truth_velocity) ** 2 for v in values) / len(values)
        )

    assert rms_error(est) < 0.5 * rms_error(raw)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("offset", [0.0, 1000.0 * math.pi, -500.0 * math.tau])
def test_large_angles_are_not_wrapped(method, offset):
    obs = started(method, offset)
    positions = [offset + 0.004 * k for k in range(1, 601)]
    estimates = replay(obs, positions)

    assert estimates[-1].position_rad == pytest.approx(positions[-1], abs=1e-5)
    # a wrapped estimate would collapse into [-pi, pi]; this one tracks the
    # multi-revolution measurement instead
    assert abs(estimates[-1].position_rad) >= abs(positions[-1]) - 1e-5
    assert estimates[-1].velocity_rad_s == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("method", METHODS)
def test_estimate_timestamp_is_the_current_sample_instant(method):
    obs = started(method)
    estimates = replay(obs, [0.001 * k for k in range(1, 11)])
    for k, est in enumerate(estimates, start=1):
        assert est.time_s == pytest.approx(T0 + k * DT, abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "dt,t0,rejected_time",
    [(1e-12, 0.0, 0.0), (1e-12, 2e-12, 1e-12), (DT, 1e16, 1e16)],
)
def test_clock_order_is_strict_even_inside_absolute_tolerance(method, dt, t0, rejected_time):
    obs = observer(method, dt=dt)
    before = obs.reset(t0, 0.0, known_initial_rest=True)
    with pytest.raises(ValueError):
        obs.update(rejected_time, 0.0)
    assert obs.state is before


def test_finite_extreme_cutoff_has_correct_bounded_frequency_product():
    # fc*dt=0.01 is bounded by Nyquist, although evaluating 2*pi*fc first
    # overflows.  This step has unit raw velocity; no private fields are read.
    cfg = EncoderVelocityConfig(dt=1e-310, cutoff_hz=1e308)
    obs = EncoderVelocityObserver(cfg)
    obs.reset(0.0, 0.0, known_initial_rest=True)
    estimate = obs.update(cfg.dt, cfg.dt)
    expected = 1.0 - math.exp(-2.0 * math.pi * (cfg.cutoff_hz * cfg.dt))
    assert estimate.velocity_rad_s == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_numpy_real_scalars_are_valid_inputs_without_adding_production_dependency(method):
    cfg = EncoderVelocityConfig(
        method=method, dt=np.float32(DT), cutoff_hz=np.int64(20),
        alpha=np.float32(0.25), beta=np.float64(0.04),
    )
    obs = EncoderVelocityObserver(cfg)
    obs.reset(np.int64(0), np.float32(2.0), known_initial_rest=True)
    result = obs.update(np.float64(cfg.dt), np.float32(2.001))
    assert math.isfinite(result.velocity_rad_s)
    assert result.time_s == cfg.dt


@pytest.mark.parametrize("invalid", [np.bool_(True), np.array(0.002), np.array([0.002])])
def test_numpy_bool_and_arrays_are_not_real_scalars(invalid):
    with pytest.raises(TypeError):
        EncoderVelocityConfig(dt=invalid)


@pytest.mark.parametrize("method", METHODS)
def test_failed_reset_does_not_discard_previous_history(method):
    obs = started(method)
    before = obs.update(T0 + DT, 0.003)
    with pytest.raises(ValueError):
        obs.reset(0.0, math.nan, known_initial_rest=True)
    assert obs.state is before
    with pytest.raises(TypeError):
        obs.reset(0.0, 0.0, known_initial_rest=np.bool_(True))
    assert obs.state is before


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("period_multiple", [0.5, 2.0])
def test_small_sample_period_cannot_hide_half_or_skipped_tick(method, period_multiple):
    cfg = EncoderVelocityConfig(method=method, dt=1e-12)
    obs = EncoderVelocityObserver(cfg)
    before = obs.reset(0.0, 0.0, known_initial_rest=True)
    with pytest.raises(ValueError):
        obs.update(period_multiple * cfg.dt, 0.0)
    assert obs.state is before
    assert obs.update(cfg.dt, 0.0).time_s == cfg.dt


@pytest.mark.parametrize("method", METHODS)
def test_small_sample_period_accepts_jitter_inside_capped_tolerance(method):
    cfg = EncoderVelocityConfig(method=method, dt=1e-12)
    obs = EncoderVelocityObserver(cfg)
    obs.reset(0.0, 0.0, known_initial_rest=True)
    assert obs.update(1.1 * cfg.dt, 0.0).time_s == 1.1 * cfg.dt
