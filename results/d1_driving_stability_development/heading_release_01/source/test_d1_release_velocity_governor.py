"""Release-reference clock, intent preservation, and raw-score regression boundaries."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scripts.d1_release_velocity_governor import ReleaseVelocityGovernor
from scripts.probe_d1_heading_g1 import command_at_tick
from scripts.probe_d1_heading_release import (
    PreparedReleaseSource,
    bitwise_equal,
    both_stop_cases_pass,
    paired_check,
)
from wheel_legged_control.d1.control_loop import D1MotionCommand


@pytest.mark.parametrize("direction", (-1, 1))
def test_original_user_release_and_symmetric_finite_tail(direction):
    spec = {"forward_target_mps": direction * .25, "settle_ticks": 50, "ramp_ticks": 50,
                "stop_tick": 400, "yaw_pulse": None, "height_m": .455}
    governor = ReleaseVelocityGovernor()
    records = [governor.advance(command_at_tick(spec, k), k * .01) for k in range(801)]
    assert all(r.served_command is r.raw_command for r in records[:400])
    assert records[0].actual_dt_s == 0
    assert records[400].served_command.forward_velocity_mps == pytest.approx(direction * .245)
    assert all(r.raw_command.forward_velocity_mps == 0 for r in records[400:])
    assert records[449].served_command.forward_velocity_mps == 0
    assert not records[449].release_active
    assert all(r.served_command.forward_velocity_mps == 0 for r in records[449:])
    assert all(r.served_command.yaw_rate_rps == r.raw_command.yaw_rate_rps
               and r.served_command.clearance_m == r.raw_command.clearance_m for r in records)


def test_idempotence_invalid_call_atomicity_restart_and_preemption():
    governor = ReleaseVelocityGovernor()
    first = D1MotionCommand(.25, .1, .47)
    a = governor.advance(first, 3.99)
    assert a.served_command is first
    zero = D1MotionCommand(0, -.2, .46)
    b = governor.advance(zero, 4)
    assert governor.advance(D1MotionCommand(0, -.2, .46), 4) is b
    with pytest.raises((ValueError, TypeError)):
        governor.advance(first, 4)
    with pytest.raises((ValueError, TypeError)):
        governor.advance(zero, 3)
    assert governor.advance(zero, 4) is b
    reverse = D1MotionCommand(-.25)
    assert governor.advance(reverse, 4.01).served_command is reverse
    with pytest.raises(FrozenInstanceError):
        b.release_active = False
    governor.reset()
    reset = governor.advance(zero, 0)
    assert reset.actual_dt_s == 0 and reset.served_command is zero


def test_bypass_preserves_object_and_yaw_clearance_during_tail():
    governor = ReleaseVelocityGovernor(None)
    for k, command in enumerate((D1MotionCommand(.25), D1MotionCommand(0, -.7, .48))):
        assert governor.advance(command, k * .01).served_command is command
    governor = ReleaseVelocityGovernor()
    governor.advance(D1MotionCommand(.25), 0)
    command = D1MotionCommand(0, -.7, .48)
    record = governor.advance(command, .01)
    assert record.served_command.yaw_rate_rps == command.yaw_rate_rps
    assert record.served_command.clearance_m == command.clearance_m
    assert record.raw_command is command


@pytest.mark.parametrize("value", (True, np.bool_(False), 0, -1, float("nan"), float("inf"), "0.5"))
def test_invalid_config(value):
    with pytest.raises((TypeError, ValueError)):
        ReleaseVelocityGovernor(value)


@pytest.mark.parametrize("value", (True, np.bool_(False), -1, float("nan"), float("inf"), "0"))
def test_invalid_clock_leaves_governor_unused(value):
    governor = ReleaseVelocityGovernor()
    with pytest.raises((TypeError, ValueError)):
        governor.advance(D1MotionCommand(), value)
    assert governor.advance(D1MotionCommand(), 0).actual_dt_s == 0


def test_forged_malformed_command_is_rejected_without_consuming_clock():
    governor = ReleaseVelocityGovernor()
    bad = D1MotionCommand()
    object.__setattr__(bad, "forward_velocity_mps", True)
    with pytest.raises((TypeError, ValueError)):
        governor.advance(bad, 0)
    with pytest.raises((TypeError, ValueError)):
        governor.advance(object(), 0)
    assert governor.advance(D1MotionCommand(), 0).actual_dt_s == 0


def test_adapter_consumes_source_once_per_prepared_tick():
    called = []

    def raw(seconds):
        called.append(seconds)
        return D1MotionCommand(.25 if seconds == 0 else 0)

    source = PreparedReleaseSource(raw, .5)
    assert source(0) is source(0)
    a = source(.01)
    assert source(.01) is a
    source(.02)  # Extra terminal prepared record; no physical interval claimed.
    assert called == [0, .01, .02]
    assert len(source.records) == 3
    with pytest.raises(ValueError):
        source(.04)


def test_strict_bitwise_prefix_and_early_failure_not_silently_excluded():
    assert not bitwise_equal(np.array([0.]), np.array([-0.]))
    baseline = {"qpos": np.zeros((801, 1)), "observations": np.zeros((801, 1))}
    candidate = {k: v.copy() for k, v in baseline.items()}
    candidate["observations"][400] = 1  # Different reference before any changed physics.
    candidate["qpos"][401] = 1
    case = {"name": "flat_forward_stop", "command": {"stop_tick": 400}}
    assert paired_check(case, baseline, candidate)["passed"]
    candidate["qpos"][400] = 1
    assert not paired_check(case, baseline, candidate)["passed"]
    summaries = [{"case": name, "condition": "release_0p5", "gates": {"passed": ok},
                      "release_diagnostics": None} for name, ok in
                 (("flat_forward_stop", False), ("flat_reverse_stop", True))]
    assert not both_stop_cases_pass(summaries)
    assert not both_stop_cases_pass(summaries[1:])
