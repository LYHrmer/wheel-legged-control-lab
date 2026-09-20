"""Nonintegrating tests of the pure jump task definitions.

Every expectation here is tabulated or computed independently of
``scripts/d1_jump_ppo_task.py``: the shifted height schedule is re-derived from
the frozen contract table, the reward is re-summed from explicit synthetic
``parent_terms``, and progress/flight accounting is driven by explicit synthetic
:class:`JumpEndpointMetrics` fixtures.  No model, plant, controller, PPO object
or MuJoCo integration is constructed.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

try:  # portable: the work tree root is on sys.path for the fixed verification
    from scripts.d1_jump_ppo_task import (
        BONUS_WINDOW_TICKS,
        JUMP_HORIZON_TICKS,
        JUMP_OBSERVATION_SCHEMA,
        JUMP_OBSERVATION_SIZE,
        JUMP_PARENT_OBSERVATION_SIZE,
        JUMP_PROGRESS_SCHEMA,
        JUMP_TASK_DEFINITION,
        JumpEndpointMetrics,
        JumpEpisodeSpec,
        JumpProgress,
        JumpProgressSnapshot,
        append_jump_observation,
        compose_jump_reward,
        jump_task_definition,
        phase_at_tick,
        raw_command_at_tick,
    )
except ModuleNotFoundError:  # pragma: no cover - flat scripts/ layout fallback
    from d1_jump_ppo_task import (
        BONUS_WINDOW_TICKS,
        JUMP_HORIZON_TICKS,
        JUMP_OBSERVATION_SCHEMA,
        JUMP_OBSERVATION_SIZE,
        JUMP_PARENT_OBSERVATION_SIZE,
        JUMP_PROGRESS_SCHEMA,
        JUMP_TASK_DEFINITION,
        JumpEndpointMetrics,
        JumpEpisodeSpec,
        JumpProgress,
        JumpProgressSnapshot,
        append_jump_observation,
        compose_jump_reward,
        jump_task_definition,
        phase_at_tick,
        raw_command_at_tick,
    )

# --- independent contract constants (never imported from the module) -------------------
BASELINE_M = 0.455
MARGIN_M = 0.001
CONTROL_DT_S = 0.01
GOAL_M = 0.020

#: Independently tabulated raw height scaffold relative to the request tick s.
EXPECTED_SEGMENTS = ((0, 0.405), (25, 0.500), (40, 0.455), (75, 0.435), (120, 0.455))

#: Explicit synthetic original reward terms; every value is a binary fraction so
#: the expected sums below are exact.  Sum = 0.5234375, sum without height = 0.5859375.
PARENT_TERMS = {
    "velocity_tracking": 0.5,
    "yaw_tracking": 0.25,
    "attitude": -0.125,
    "height": -0.0625,
    "mechanical_power": -0.03125,
    "action_change": -0.015625,
    "heading_goal": 0.0078125,
    "termination": 0.0,
}
PARENT_SUM = 0.5234375
PARENT_SUM_WITHOUT_HEIGHT = 0.5859375


def expected_height_m(offset: int) -> float:
    """Independently tabulated commanded clearance at ``offset = k - s``."""
    if offset < 0:
        return BASELINE_M
    height = BASELINE_M
    for first_offset, value in EXPECTED_SEGMENTS:
        if offset >= first_offset:
            height = value
    return height


def loaded_metrics(**overrides) -> JumpEndpointMetrics:
    """Supported endpoint: four wheels on the plane, zero gap, positive load."""
    fields = {
        "simultaneous_minimum_gap_m": 0.0,
        "contact_margin_m": MARGIN_M,
        "wheel_bottom_gap_m": [0.0] * 4,
        "active_sample_fraction_by_wheel": [1.0] * 4,
        "wheel_force_world_n": [[0.0, 0.0, 60.0]] * 4,
        "physics_sample_count": 5,
        "base_position_m": [0.0, 0.0, BASELINE_M],
        "base_rpy_rad": [0.0, 0.0, 0.0],
        "body_forward_velocity_mps": 0.0,
        "com_position_m": [0.0, 0.0, 0.30],
        "com_velocity_mps": [0.0, 0.0, 0.0],
        "initial_origin_xy_m": [0.0, 0.0],
        "heading_error_rad": 0.0,
    }
    fields.update(overrides)
    return JumpEndpointMetrics(**fields)


def unloaded_metrics(net_gap_m: float, com_vz_mps: float = 0.4, **overrides):
    """Fully unloaded endpoint with net useful gap ``c = g - margin``."""
    fields = {
        "simultaneous_minimum_gap_m": MARGIN_M + net_gap_m,
        "wheel_bottom_gap_m": [MARGIN_M + net_gap_m] * 4,
        "active_sample_fraction_by_wheel": [0.0] * 4,
        "wheel_force_world_n": [[0.0, 0.0, 0.0]] * 4,
        "com_velocity_mps": [0.0, 0.0, com_vz_mps],
        "base_position_m": [0.0, 0.0, BASELINE_M + net_gap_m],
    }
    fields.update(overrides)
    return loaded_metrics(**fields)


def synthetic_snapshot(
    progress: float = 0.0,
    flight_seen: bool = False,
    unloaded_native_samples: int = 0,
) -> JumpProgressSnapshot:
    """Explicit synthetic accounting snapshot, independent of :class:`JumpProgress`."""
    return JumpProgressSnapshot(
        schema=JUMP_PROGRESS_SCHEMA,
        progress=progress,
        flight_seen=flight_seen,
        consecutive_unloaded_intervals=0,
        unloaded_native_samples=unloaded_native_samples,
        airborne_progress=1.0 if flight_seen else min(1.0, unloaded_native_samples / 10.0),
        run_upward_onset=False,
        run_maximum_net_gap_m=0.0,
        certified_run_maximum_net_gap_m=0.0,
        completed_intervals=0,
    )


def published_state(contacts=(True, True, True, True), base_xy=(0.0, 0.0)):
    """Minimal stand-in for the published oracle state fields the encoder reads."""
    return SimpleNamespace(
        wheel_contact=np.array(contacts, dtype=np.bool_),
        base_position=np.array([base_xy[0], base_xy[1], BASELINE_M], dtype=np.float64),
    )


def parent_base85() -> np.ndarray:
    """Deterministic parent 85-entry float32 encoding."""
    return (np.arange(JUMP_PARENT_OBSERVATION_SIZE, dtype=np.float32) * 0.01) - 0.4


# --- shifted command schedule ---------------------------------------------------------


@pytest.mark.parametrize("request_tick", [0, 150, 175, 200, 225, 250, 480])
def test_shifted_height_schedule_matches_independent_table(request_tick):
    """Every prepared tick 0..600 reproduces the independently tabulated scaffold."""
    spec = JumpEpisodeSpec(request_tick, GOAL_M)
    for tick in range(601):
        command = raw_command_at_tick(spec, tick)
        assert command.clearance_m == pytest.approx(expected_height_m(tick - request_tick))
        assert command.forward_velocity_mps == 0.0
        assert command.yaw_rate_rps == 0.0


def test_prepared_terminal_tick_and_hold_stay_at_baseline():
    """Prepared tick 600 is valid and a no-jump hold never leaves 0.455 m."""
    hold = JumpEpisodeSpec(None, 0)
    for tick in (0, 199, 200, 320, 599, 600):
        command = raw_command_at_tick(hold, tick)
        assert command.clearance_m == pytest.approx(BASELINE_M)
        assert phase_at_tick(hold, tick) == ("settle" if tick < 200 else "hold")
    late = JumpEpisodeSpec(480, GOAL_M)
    assert raw_command_at_tick(late, 600).clearance_m == pytest.approx(BASELINE_M)
    assert phase_at_tick(late, 600) == "hold"
    with pytest.raises(ValueError):
        raw_command_at_tick(late, 601)


def test_phase_labels_shift_with_the_request_tick():
    """Command phase labels are the approved labels shifted by s, nothing measured."""
    spec = JumpEpisodeSpec(175, GOAL_M)
    assert phase_at_tick(spec, 174) == "settle"
    assert phase_at_tick(spec, 175) == "crouch_request"
    assert phase_at_tick(spec, 199) == "crouch_request"
    assert phase_at_tick(spec, 200) == "extension_request"
    assert phase_at_tick(spec, 214) == "extension_request"
    assert phase_at_tick(spec, 215) == "return_request"
    assert phase_at_tick(spec, 249) == "return_request"
    assert phase_at_tick(spec, 250) == "lower_request"
    assert phase_at_tick(spec, 294) == "lower_request"
    assert phase_at_tick(spec, 295) == "hold"


def test_episode_spec_windows_and_rejected_settings():
    """Bonus/mask windows are [s, s+120) and [s+25, s+120); holds carry no window."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    assert spec.bonus_window_ticks == (200, 320)
    assert spec.height_mask_window_ticks == (225, 320)
    assert [spec.in_bonus_window(t) for t in (199, 200, 319, 320)] == [False, True, True, False]
    assert [spec.masks_height_term(t) for t in (224, 225, 319, 320)] == [
        False,
        True,
        True,
        False,
    ]
    hold = JumpEpisodeSpec(None, 0)
    assert hold.is_hold and hold.condition == "hold"
    assert hold.bonus_window_ticks is None and hold.height_mask_window_ticks is None
    assert not hold.in_bonus_window(250) and not hold.masks_height_term(250)
    with pytest.raises(ValueError):
        JumpEpisodeSpec(None, GOAL_M)
    with pytest.raises(ValueError):
        JumpEpisodeSpec(200, 0.015)
    with pytest.raises(ValueError):
        JumpEpisodeSpec(JUMP_HORIZON_TICKS - BONUS_WINDOW_TICKS + 1, GOAL_M)
    with pytest.raises(ValueError):
        JumpEpisodeSpec(200, GOAL_M, 1.10)
    with pytest.raises(TypeError):
        JumpEpisodeSpec(200.0, GOAL_M)


# --- observation 95 -------------------------------------------------------------------


def test_observation_preserves_the_parent_85_prefix_byte_for_byte():
    """The 95 encoding is float32 and keeps the parent prefix unchanged."""
    base = parent_base85()
    obs = append_jump_observation(
        base, published_state(), 300, JumpEpisodeSpec(200, GOAL_M), JumpProgress(), [0.0, 0.0]
    )
    assert obs.shape == (JUMP_OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert obs[:JUMP_PARENT_OBSERVATION_SIZE].tobytes() == base.tobytes()
    assert JUMP_OBSERVATION_SCHEMA == "d1-jump-oracle-task95-v1"


def test_observation_never_leaks_a_future_request_clock_or_goal():
    """Entries 85/86/91 stay -1/0/0 before s, then follow clip((k-s)/120, 0, 1)."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = synthetic_snapshot(progress=0.6, flight_seen=True)
    base = parent_base85()
    for tick in (0, 100, 199):
        obs = append_jump_observation(base, published_state(), tick, spec, progress, [0.0, 0.0])
        assert obs[85] == pytest.approx(-1.0)
        assert obs[86] == pytest.approx(0.0)
        assert obs[91] == pytest.approx(0.0)
    expected_clock = {200: 0.0, 260: 0.5, 319: 119.0 / 120.0, 320: 1.0, 599: 1.0}
    for tick, clock in expected_clock.items():
        obs = append_jump_observation(base, published_state(), tick, spec, progress, [0.0, 0.0])
        assert obs[85] == pytest.approx(clock, rel=1e-6)
        assert obs[86] == pytest.approx(0.020 / 0.04)
        assert obs[91] == pytest.approx(0.6, rel=1e-6)


def test_observation_hold_hides_clock_goal_and_progress_at_every_tick():
    """A no-jump hold publishes -1 / 0 / 0 throughout, including prepared tick 600."""
    hold = JumpEpisodeSpec(None, 0)
    progress = synthetic_snapshot(progress=0.9, flight_seen=True)
    for tick in (0, 200, 320, 599, 600):
        obs = append_jump_observation(
            parent_base85(), published_state(), tick, hold, progress, [0.0, 0.0]
        )
        assert obs[85] == pytest.approx(-1.0)
        assert obs[86] == pytest.approx(0.0)
        assert obs[91] == pytest.approx(0.0)


def test_observation_contacts_airborne_progress_and_planar_offset():
    """Entries 87..90 are the published contacts; 92..94 the oracle task features."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 210, CONTROL_DT_S, unloaded_metrics(0.010), False, False)
    obs = append_jump_observation(
        parent_base85(),
        published_state(contacts=(True, False, True, False), base_xy=(0.03, -0.02)),
        211,
        spec,
        progress,
        [0.01, 0.0],
    )
    assert list(obs[87:91]) == [1.0, 0.0, 1.0, 0.0]
    assert obs[92] == pytest.approx(0.5)  # five of ten unloaded native samples
    assert obs[93] == pytest.approx(0.2, rel=1e-6)  # (0.03 - 0.01) / 0.10
    assert obs[94] == pytest.approx(-0.2, rel=1e-6)


def test_observation_clips_only_to_the_declared_range():
    """A large planar offset saturates at the declared [-5, 5] clip."""
    obs = append_jump_observation(
        parent_base85(),
        published_state(base_xy=(1.0, -2.0)),
        100,
        JumpEpisodeSpec(200, GOAL_M),
        JumpProgress(),
        [0.0, 0.0],
    )
    assert obs[93] == pytest.approx(5.0)
    assert obs[94] == pytest.approx(-5.0)


# --- measured progress and flight certification ---------------------------------------


def test_loaded_interval_with_positive_gap_is_never_flight():
    """Load on any sample, or nonzero measured force, denies the unloaded claim."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    partly_active = unloaded_metrics(0.030, active_sample_fraction_by_wheel=[0.0, 0.2, 0.0, 0.0])
    residual_force = unloaded_metrics(0.030, wheel_force_world_n=[[0.0, 0.0, 0.5]] * 4)
    for metrics in (partly_active, residual_force):
        assert metrics.net_gap_m == pytest.approx(0.030)
        assert metrics.above_contact_margin
        assert not metrics.fully_unloaded
        progress = JumpProgress()
        for tick in (205, 206, 207):
            event = progress.advance(spec, tick, CONTROL_DT_S, metrics, False, False)
            assert not event.fully_unloaded
            assert not event.newly_certified_flight
        assert progress.progress == 0.0
        assert progress.flight_seen is False
        assert progress.airborne_progress == 0.0


def test_endpoint_at_the_margin_is_not_unloaded_and_short_summaries_are_rejected():
    """g must be strictly above m, and one interval must carry exactly five samples."""
    at_margin = unloaded_metrics(0.0)
    assert at_margin.net_gap_m == pytest.approx(0.0)
    assert not at_margin.above_contact_margin
    assert not at_margin.fully_unloaded
    with pytest.raises(ValueError):
        unloaded_metrics(0.030, physics_sample_count=4)
    with pytest.raises(ValueError):
        unloaded_metrics(0.030, physics_sample_count=10)


def test_one_unloaded_interval_is_insufficient_for_flight():
    """Five native samples are half of the conservative twenty-millisecond quantum."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    event = progress.advance(spec, 205, CONTROL_DT_S, unloaded_metrics(0.030), False, False)
    assert event.fully_unloaded and event.run_onset
    assert not event.newly_certified_flight
    assert event.credited_net_gap_m == 0.0
    assert event.clearance_progress == 0.0
    assert progress.snapshot.unloaded_native_samples == 5
    assert progress.airborne_progress == pytest.approx(0.5)
    assert progress.progress == 0.0
    assert progress.flight_seen is False


def test_two_consecutive_upward_intervals_certify_ten_native_samples():
    """Flight is certified exactly on the second unloaded interval of the run."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    first = progress.advance(spec, 205, CONTROL_DT_S, unloaded_metrics(0.010), False, False)
    second = progress.advance(spec, 206, CONTROL_DT_S, unloaded_metrics(0.010), False, False)
    assert not first.newly_certified_flight
    assert second.newly_certified_flight
    assert second.flight_once == 1.0
    assert progress.flight_seen is True
    assert progress.snapshot.unloaded_native_samples == 10
    assert progress.snapshot.consecutive_unloaded_intervals == 2
    assert progress.airborne_progress == 1.0
    assert progress.progress == pytest.approx(0.010 / GOAL_M)
    third = progress.advance(spec, 207, CONTROL_DT_S, unloaded_metrics(0.010), False, False)
    assert not third.newly_certified_flight
    assert third.flight_once == 0.0


def test_disrupted_unloaded_run_never_certifies_flight():
    """A loaded interval between two unloaded ones resets the run and the count."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 205, CONTROL_DT_S, unloaded_metrics(0.030), False, False)
    progress.advance(spec, 206, CONTROL_DT_S, loaded_metrics(), False, False)
    assert progress.snapshot.unloaded_native_samples == 0
    assert progress.snapshot.consecutive_unloaded_intervals == 0
    event = progress.advance(spec, 207, CONTROL_DT_S, unloaded_metrics(0.030), False, False)
    assert not event.newly_certified_flight
    assert progress.flight_seen is False
    assert progress.progress == 0.0
    assert progress.airborne_progress == pytest.approx(0.5)


def test_downward_onset_prevents_flight_certification():
    """A run beginning without a rising whole-robot COM can never be certified."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    onset = progress.advance(
        spec, 205, CONTROL_DT_S, unloaded_metrics(0.030, com_vz_mps=-0.2), False, False
    )
    assert onset.run_onset and onset.com_vertical_velocity_mps == pytest.approx(-0.2)
    for tick in (206, 207, 208):
        event = progress.advance(
            spec, tick, CONTROL_DT_S, unloaded_metrics(0.030, com_vz_mps=0.5), False, False
        )
        assert not event.newly_certified_flight
    assert progress.flight_seen is False
    assert progress.progress == 0.0


def test_first_certified_run_credits_the_run_maximum_net_gap():
    """The just-certified two-interval run credits its own maximum net gap once."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 205, CONTROL_DT_S, unloaded_metrics(0.012), False, False)
    certifying = progress.advance(
        spec, 206, CONTROL_DT_S, unloaded_metrics(0.004), False, False
    )
    assert certifying.newly_certified_flight
    assert certifying.net_gap_m == pytest.approx(0.004)
    assert certifying.credited_net_gap_m == pytest.approx(0.012)
    assert progress.snapshot.certified_run_maximum_net_gap_m == pytest.approx(0.012)
    assert progress.progress == pytest.approx(0.012 / GOAL_M)
    assert certifying.clearance_progress == pytest.approx(2.0 * 0.012 / GOAL_M)


def test_rebound_and_supported_intervals_cannot_increase_progress():
    """Progress is monotone, bounded by one and frozen by support or the window end."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 205, CONTROL_DT_S, unloaded_metrics(0.012), False, False)
    progress.advance(spec, 206, CONTROL_DT_S, unloaded_metrics(0.012), False, False)
    assert progress.progress == pytest.approx(0.6, rel=1e-9)
    for tick, metrics in (
        (207, loaded_metrics(simultaneous_minimum_gap_m=0.050)),
        (208, unloaded_metrics(0.002)),
        (209, loaded_metrics()),
    ):
        event = progress.advance(spec, tick, CONTROL_DT_S, metrics, False, False)
        assert event.clearance_progress == 0.0
    assert progress.progress == pytest.approx(0.6, rel=1e-9)
    big = progress.advance(spec, 210, CONTROL_DT_S, unloaded_metrics(0.500), False, False)
    assert progress.progress == pytest.approx(0.6, rel=1e-9)
    assert big.clearance_progress == 0.0
    frozen = progress.advance(spec, 330, CONTROL_DT_S, unloaded_metrics(0.500), False, False)
    assert not frozen.in_bonus_window
    assert progress.progress == pytest.approx(0.6, rel=1e-9)
    assert frozen.clearance_progress == 0.0


def test_pre_request_flight_cannot_be_retroactively_certified():
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 199, CONTROL_DT_S, unloaded_metrics(0.024), False, False)
    event = progress.advance(spec, 200, CONTROL_DT_S,
                             unloaded_metrics(0.002, com_velocity_mps=[0., 0., -.1]),
                             False, False)
    assert not event.newly_certified_flight
    assert not progress.flight_seen and progress.progress == 0.
    assert progress.snapshot.consecutive_unloaded_intervals == 1


def test_same_certified_run_can_continue_to_full_clearance():
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 200, CONTROL_DT_S, unloaded_metrics(.005), False, False)
    progress.advance(spec, 201, CONTROL_DT_S, unloaded_metrics(.005), False, False)
    event = progress.advance(spec, 202, CONTROL_DT_S, unloaded_metrics(.03), False, False)
    assert progress.progress == 1.
    assert event.clearance_progress == pytest.approx(1.5)
    assert progress.snapshot.credited_run_active


@pytest.mark.parametrize(
    ("spec", "ticks"),
    [
        (JumpEpisodeSpec(200, GOAL_M), (100, 101, 199)),
        (JumpEpisodeSpec(200, GOAL_M), (320, 321, 400)),
        (JumpEpisodeSpec(None, 0), (100, 205, 206)),
    ],
)
def test_no_progress_before_the_request_on_a_hold_or_outside_the_window(spec, ticks):
    """Pre-request activity, holds and post-window flight never earn progress."""
    progress = JumpProgress()
    for tick in ticks:
        event = progress.advance(
            spec, tick, CONTROL_DT_S, unloaded_metrics(0.030), False, False
        )
        assert not event.in_bonus_window
        assert not event.newly_certified_flight
        assert event.credited_net_gap_m == 0.0
        assert event.clearance_progress == 0.0
    assert progress.progress == 0.0
    assert progress.flight_seen is False


def test_new_episode_reset_clears_progress_flight_and_run_state():
    """``reset`` returns the accounting to a fresh episode's zero state."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 205, CONTROL_DT_S, unloaded_metrics(0.030), False, False)
    progress.advance(spec, 206, CONTROL_DT_S, unloaded_metrics(0.030), False, False)
    assert progress.flight_seen and progress.progress == 1.0
    progress.reset()
    snapshot = progress.snapshot
    assert snapshot.progress == 0.0
    assert snapshot.flight_seen is False
    assert snapshot.unloaded_native_samples == 0
    assert snapshot.consecutive_unloaded_intervals == 0
    assert snapshot.run_maximum_net_gap_m == 0.0
    assert snapshot.certified_run_maximum_net_gap_m == 0.0
    assert snapshot.completed_intervals == 0
    assert progress.airborne_progress == 0.0


# --- fixed reward composition ---------------------------------------------------------


def test_reward_keeps_every_original_term_and_the_original_sum():
    """Unmasked intervals add the original sum plus only the declared new terms."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    zero = synthetic_snapshot()
    breakdown = compose_jump_reward(
        PARENT_TERMS, 100, spec, CONTROL_DT_S, zero, zero, loaded_metrics(), False, False, 101
    )
    assert dict(breakdown.parent_reward_terms) == PARENT_TERMS
    assert breakdown.parent_reward == pytest.approx(PARENT_SUM)
    assert breakdown.height_term_masked is False
    assert breakdown.height_term_effective == pytest.approx(PARENT_TERMS["height"])
    assert breakdown.clearance_progress == 0.0
    assert breakdown.flight_once == 0.0
    assert breakdown.landing_once == 0.0
    assert breakdown.stationary_offset == pytest.approx(0.0)
    assert breakdown.reward == pytest.approx(PARENT_SUM)
    assert set(breakdown.reward_terms) == set(PARENT_TERMS) | {
        "clearance_progress",
        "flight_once",
        "stationary_offset",
        "landing_once",
    }
    with pytest.raises(ValueError):
        compose_jump_reward(
            {k: v for k, v in PARENT_TERMS.items() if k != "height"},
            100,
            spec,
            CONTROL_DT_S,
            zero,
            zero,
            loaded_metrics(),
            False,
            False,
            101,
        )


@pytest.mark.parametrize(
    ("tick", "masked"),
    [(199, False), (200, False), (224, False), (225, True), (319, True), (320, False)],
)
def test_height_mask_timing_is_exactly_the_predeclared_command_window(tick, masked):
    """Only execution ticks [s+25, s+120) of a requested jump drop the height term."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    zero = synthetic_snapshot()
    breakdown = compose_jump_reward(
        PARENT_TERMS, tick, spec, CONTROL_DT_S, zero, zero, loaded_metrics(), False, False,
        tick + 1,
    )
    assert breakdown.height_term_masked is masked
    assert breakdown.height_term_original == pytest.approx(PARENT_TERMS["height"])
    assert breakdown.height_term_effective == pytest.approx(
        0.0 if masked else PARENT_TERMS["height"]
    )
    assert breakdown.reward == pytest.approx(
        PARENT_SUM_WITHOUT_HEIGHT if masked else PARENT_SUM
    )
    assert breakdown.parent_reward == pytest.approx(PARENT_SUM)
    hold = compose_jump_reward(
        PARENT_TERMS, tick, JumpEpisodeSpec(None, 0), CONTROL_DT_S, zero, zero,
        loaded_metrics(), False, False, tick + 1,
    )
    assert hold.height_term_masked is False
    assert hold.reward == pytest.approx(PARENT_SUM)


def test_stationary_offset_follows_the_declared_formula_and_saturates():
    """-dt * 0.5 * min((d / 0.10)^2, 4), applied on holds as well."""
    zero = synthetic_snapshot()
    for displacement, expected in ((0.0, 0.0), (0.05, -0.00125), (0.10, -0.005), (0.5, -0.02)):
        metrics = loaded_metrics(base_position_m=[displacement, 0.0, BASELINE_M])
        assert metrics.planar_displacement_m == pytest.approx(displacement)
        for spec in (JumpEpisodeSpec(200, GOAL_M), JumpEpisodeSpec(None, 0)):
            breakdown = compose_jump_reward(
                PARENT_TERMS, 100, spec, CONTROL_DT_S, zero, zero, metrics, False, False, 101
            )
            assert breakdown.stationary_offset == pytest.approx(expected)
            assert breakdown.reward == pytest.approx(PARENT_SUM + expected)


def test_positive_jump_event_total_is_bounded_by_two_one_two():
    """One synthetic episode earns at most 2 progress + 1 flight + 2 landing."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    plan = [
        (200, loaded_metrics(), False),
        (205, unloaded_metrics(0.012), False),
        (206, unloaded_metrics(0.025), False),
        (207, unloaded_metrics(0.030), False),
        (208, loaded_metrics(), False),
        (300, unloaded_metrics(0.030), False),
        (301, unloaded_metrics(0.030), False),
        (599, loaded_metrics(), True),
    ]
    clearance_total = 0.0
    flight_total = 0.0
    landing_total = 0.0
    for tick, metrics, final in plan:
        before = progress.snapshot
        progress.advance(spec, tick, CONTROL_DT_S, metrics, False, final)
        breakdown = compose_jump_reward(
            PARENT_TERMS, tick, spec, CONTROL_DT_S, before, progress, metrics,
            False, final, tick + 1,
        )
        clearance_total += breakdown.clearance_progress
        flight_total += breakdown.flight_once
        landing_total += breakdown.landing_once
    assert clearance_total == pytest.approx(2.0)
    assert flight_total == pytest.approx(1.0)
    assert landing_total == pytest.approx(2.0)
    assert clearance_total + flight_total + landing_total == pytest.approx(5.0)


def test_landing_once_requires_the_genuine_final_transition():
    """The completed 600th truncated transition with full progress earns exactly 2."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    landed = synthetic_snapshot(progress=1.0, flight_seen=True)
    breakdown = compose_jump_reward(
        PARENT_TERMS, 599, spec, CONTROL_DT_S, landed, landed, loaded_metrics(),
        False, True, 600,
    )
    assert breakdown.landing_once == pytest.approx(2.0)
    assert all(breakdown.landing_conditions.values())
    assert breakdown.reward == pytest.approx(PARENT_SUM + 2.0)


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("terminated", {"terminated": True, "truncated": False}),
        ("not_final_tick", {"executed_tick": 400, "endpoint_tick": 401}),
        ("incomplete_trace", {"truncated": False}),
        ("no_flight", {"snapshot": synthetic_snapshot(progress=1.0, flight_seen=False)}),
        ("partial_progress", {"snapshot": synthetic_snapshot(progress=0.6, flight_seen=True)}),
        ("hold", {"spec": JumpEpisodeSpec(None, 0)}),
        ("wheel_unloaded", {"metrics": loaded_metrics(wheel_force_world_n=[
            [0.0, 0.0, 60.0], [0.0, 0.0, 0.0], [0.0, 0.0, 60.0], [0.0, 0.0, 60.0]])}),
        ("base_height", {"metrics": loaded_metrics(base_position_m=[0.0, 0.0, 0.48])}),
        ("body_vx", {"metrics": loaded_metrics(body_forward_velocity_mps=0.05)}),
        ("com_vz", {"metrics": loaded_metrics(com_velocity_mps=[0.0, 0.0, -0.04])}),
        ("roll_pitch", {"metrics": loaded_metrics(base_rpy_rad=[0.0, 0.25, 0.0])}),
        ("heading", {"metrics": loaded_metrics(heading_error_rad=math.radians(6.0))}),
        ("displacement", {"metrics": loaded_metrics(base_position_m=[0.2, 0.0, BASELINE_M])}),
    ],
)
def test_no_landing_bonus_when_any_declared_condition_fails(label, kwargs):
    """Early termination, holds, missing flight or partial progress earn no bonus."""
    spec = kwargs.get("spec", JumpEpisodeSpec(200, GOAL_M))
    snapshot = kwargs.get("snapshot", synthetic_snapshot(progress=1.0, flight_seen=True))
    metrics = kwargs.get("metrics", loaded_metrics())
    tick = kwargs.get("executed_tick", 599)
    endpoint = kwargs.get("endpoint_tick", 600)
    breakdown = compose_jump_reward(
        PARENT_TERMS, tick, spec, CONTROL_DT_S, snapshot, snapshot, metrics,
        kwargs.get("terminated", False), kwargs.get("truncated", True), endpoint,
    )
    assert breakdown.landing_once == 0.0, label
    assert not all(breakdown.landing_conditions.values()), label
    assert breakdown.reward == pytest.approx(
        PARENT_SUM + breakdown.stationary_offset
    )


def test_repeated_composition_is_pure_and_never_mutates_progress():
    """Composition reads snapshots only: repeated calls give identical results."""
    spec = JumpEpisodeSpec(200, GOAL_M)
    progress = JumpProgress()
    progress.advance(spec, 205, CONTROL_DT_S, unloaded_metrics(0.012), False, False)
    before = progress.snapshot
    progress.advance(spec, 206, CONTROL_DT_S, unloaded_metrics(0.012), False, False)
    after = progress.snapshot
    metrics = unloaded_metrics(0.012)
    rewards = []
    for _ in range(3):
        breakdown = compose_jump_reward(
            PARENT_TERMS, 206, spec, CONTROL_DT_S, before, progress, metrics, False, False, 207
        )
        rewards.append(breakdown.reward)
        assert breakdown.clearance_progress == pytest.approx(2.0 * 0.6, rel=1e-9)
        assert breakdown.flight_once == 1.0
    assert rewards[0] == rewards[1] == rewards[2]
    assert progress.snapshot.as_dict() == after.as_dict()
    assert progress.progress == pytest.approx(0.6, rel=1e-9)
    assert progress.snapshot.completed_intervals == 2
    with pytest.raises(ValueError):
        compose_jump_reward(
            PARENT_TERMS, 206, spec, CONTROL_DT_S, after, before, metrics, False, False, 207
        )


def test_task_definition_is_a_frozen_json_ready_copy():
    """The published definition is immutable and its fresh copy is JSON-ready."""
    copy = jump_task_definition()
    assert copy["observation"]["total_size"] == JUMP_OBSERVATION_SIZE
    assert copy["observation"]["parent_size"] == JUMP_PARENT_OBSERVATION_SIZE
    assert copy["reward"]["maximum_event_total"] == pytest.approx(5.0)
    assert copy["reward"]["height_mask_window_ticks"] == [25, 120]
    assert copy["episode"]["duration_s"] == pytest.approx(6.0)
    copy["reward"]["maximum_event_total"] = 99.0
    assert jump_task_definition()["reward"]["maximum_event_total"] == pytest.approx(5.0)
    with pytest.raises(TypeError):
        JUMP_TASK_DEFINITION["reward"] = {}
    assert json.loads(json.dumps(copy))["schema"] == copy["schema"]
