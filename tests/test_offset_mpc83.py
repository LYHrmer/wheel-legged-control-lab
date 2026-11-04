"""Synthetic observer/coordinate invariants; no physical plant or rollout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("mujoco")
from wheel_legged_control import controllers
from wheel_legged_control.controllers import TrackingCommand
from wheel_legged_control.delay_mpc82 import DelayMPC82
from wheel_legged_control.offset_mpc83 import OffsetMPC83


class NominalFixture:
    def __init__(self):
        self.a = np.diag([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
        self.b = np.array(
            [[0.02, 0.0], [0.01, 0.0], [0.0, 0.0002],
             [0.04, 0.0], [0.02, 0.0], [0.0, 0.002]]
        )
        self.control_dt = 0.02
        self.equilibrium_control = np.array([0.0, 100.0])
        self.actuator_low = np.array([-80.0, 0.0])
        self.actuator_high = np.array([80.0, 260.0])
        self.linearize_calls = 0

    def linearize(self):
        self.linearize_calls += 1
        return self.a.copy(), self.b.copy()


def keep_warm_start(monkeypatch):
    calls = []

    def minimize(_objective, initial, *, jac, bounds, method, options):
        calls.append((initial.copy(), np.asarray(bounds), method, options.copy()))
        return SimpleNamespace(x=initial.copy(), nit=0)

    monkeypatch.setattr(controllers, "minimize", minimize)
    return calls


@pytest.mark.parametrize("delay", [0, 2])
def test_zero_innovation_is_exact_delay82_and_never_mutates_source_plant(delay):
    plant = NominalFixture()
    candidate = OffsetMPC83(plant, delay_steps=delay)
    baseline = DelayMPC82(plant, delay_steps=delay)
    for _ in range(3):
        np.testing.assert_array_equal(
            candidate.compute(np.zeros(6), TrackingCommand()),
            baseline.compute(np.zeros(6), TrackingCommand()),
        )
        np.testing.assert_array_equal(candidate._solution, baseline._solution)
    assert candidate.disturbance_force_n == 0.0
    np.testing.assert_array_equal(plant.equilibrium_control, [0.0, 100.0])
    for name in ("a", "b", "q", "r", "terminal_q", "_qbar", "_rbar", "_sx", "_su", "_hessian"):
        np.testing.assert_array_equal(getattr(candidate, name), getattr(baseline, name))
    assert plant.linearize_calls == 2


@pytest.mark.parametrize("delay", [0, 2])
def test_observer_uses_delayed_absolute_force_and_fixed_nominal_origin(monkeypatch, delay):
    keep_warm_start(monkeypatch)
    plant = NominalFixture()
    candidate = OffsetMPC83(plant, delay_steps=delay)
    state = np.zeros(6)
    queue = [plant.equilibrium_control.copy() for _ in range(delay)]
    expected_disturbance = 0.0
    applied = None
    for step in range(4):
        # Deliberately issue a different absolute force every call: the newest
        # issue must never be mistaken for the previous transition's applied one.
        candidate._solution.reshape(-1, 2)[:] = [float(step), 5.0 * step]
        issued = candidate.compute(state, TrackingCommand())
        if step:
            expected_disturbance = (
                (1 - candidate.observer_alpha) * expected_disturbance
                + candidate.observer_alpha * -20.0
            )
            assert candidate.disturbance_force_n == pytest.approx(expected_disturbance)
            assert candidate.last_raw_disturbance_force_n == pytest.approx(-20.0)
            np.testing.assert_array_equal(candidate.last_transition_applied_control, applied)
        queue.append(issued.copy())
        applied = queue.pop(0)
        state = plant.a @ state + plant.b @ (applied - plant.equilibrium_control + [0.0, -20.0])
    np.testing.assert_array_equal(plant.equilibrium_control, [0.0, 100.0])
    assert plant.linearize_calls == 1


def test_new_equilibrium_rebases_cost_bounds_fifo_and_warm_start_together(monkeypatch):
    calls = keep_warm_start(monkeypatch)
    candidate = OffsetMPC83(NominalFixture(), delay_steps=2)
    candidate.compute(np.zeros(6), TrackingCommand())
    before_pending = candidate.pending_controls
    candidate._solution.reshape(-1, 2)[:] = [3.0, 7.0]
    previous_absolute_plan = candidate._solution.reshape(-1, 2) + [0.0, 100.0]
    state = candidate.b[:, 1] * -30.0
    candidate.compute(state, TrackingCommand())
    equilibrium = candidate.estimated_equilibrium_control
    np.testing.assert_allclose(calls[-1][0].reshape(-1, 2) + equilibrium, previous_absolute_plan)
    np.testing.assert_allclose(calls[-1][1][:2, 0] + equilibrium, [-80.0, 0.0])
    np.testing.assert_allclose(calls[-1][1][:2, 1] + equilibrium, [80.0, 260.0])
    np.testing.assert_array_equal(candidate.last_pending_before, before_pending)
    # The first prediction is driven by the already pending physical command,
    # and equals nominal prediction plus the inferred constant input offset.
    extended = np.r_[state, (before_pending - equilibrium).ravel()]
    first_prediction = (candidate._sx @ extended)[:6]
    expected = candidate.a @ state + candidate.b @ (
        before_pending[0] - candidate.nominal_equilibrium_control
        + [0.0, candidate.disturbance_force_n]
    )
    np.testing.assert_allclose(first_prediction, expected, rtol=1e-13, atol=1e-14)
    assert calls[-1][2:] == ("L-BFGS-B", {"maxiter": 35, "ftol": 1e-8, "gtol": 1e-6})


@pytest.mark.parametrize("endpoint", ["previous", "current"])
@pytest.mark.parametrize("boundary", [-0.115, 0.115])
def test_limit_reaction_is_recorded_but_never_learned(monkeypatch, endpoint, boundary):
    keep_warm_start(monkeypatch)
    candidate = OffsetMPC83(NominalFixture(), delay_steps=2)
    first = np.zeros(6)
    second = np.zeros(6)
    (first if endpoint == "previous" else second)[2] = boundary
    second[5] = 20.0  # A large limit reaction must not drive the observer to its cap.
    candidate.compute(first, TrackingCommand())
    candidate.compute(second, TrackingCommand())
    assert candidate.disturbance_force_n == 0.0 and not candidate.observer_updated
    assert candidate.observer_freeze_reason == "measured_extension_near_limit"
    assert candidate.last_raw_disturbance_force_n == pytest.approx(10000.0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_estimate_clips_after_filter_and_reset_discards_all_history(monkeypatch, sign):
    keep_warm_start(monkeypatch)
    candidate = OffsetMPC83(NominalFixture(), delay_steps=2)
    candidate.compute(np.zeros(6), TrackingCommand())
    measured = np.zeros(6)
    measured[5] = sign * 20.0
    candidate.compute(measured, TrackingCommand())
    assert candidate.disturbance_force_n == sign * 60.0
    candidate.reset()
    assert candidate.disturbance_force_n == 0.0
    assert candidate.last_residual is None and candidate.last_nominal_prediction is None
    assert candidate.last_transition_applied_control is None
    assert candidate.last_raw_disturbance_force_n is None and not candidate.observer_updated
    np.testing.assert_array_equal(candidate.pending_controls, [[0.0, 100.0], [0.0, 100.0]])
    np.testing.assert_array_equal(candidate.estimated_equilibrium_control, [0.0, 100.0])
    candidate.compute(np.zeros(6), TrackingCommand())
    assert candidate.last_residual is None and candidate.disturbance_force_n == 0.0


def test_nonfinite_input_fails_before_observer_or_fifo_changes(monkeypatch):
    keep_warm_start(monkeypatch)
    candidate = OffsetMPC83(NominalFixture(), delay_steps=2)
    candidate.compute(np.zeros(6), TrackingCommand())
    queue = candidate.pending_controls
    with pytest.raises(ValueError, match="physical6"):
        candidate.compute([0.0, 0.0, 0.0, 0.0, 0.0, float("nan")], TrackingCommand())
    assert candidate.disturbance_force_n == 0.0
    np.testing.assert_array_equal(candidate.pending_controls, queue)
