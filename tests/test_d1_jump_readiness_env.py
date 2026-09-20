"""Unit tests for :mod:`scripts.d1_jump_readiness_env`.

Every test drives ``JumpReadinessEnv`` through a hand-built instance created with
``object.__new__`` so that no plant, no controller and no MuJoCo integrator is
ever constructed.  ``StopTurnEnv.step`` is replaced by a fake that returns a
control endpoint without integrating anything, and ``mj_step``/``mj_step1``/
``mj_step2`` are poisoned so an accidental real step fails loudly.
"""
from __future__ import annotations

import io
import json
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.d1_jump_readiness_env as env_mod

JumpReadinessEnv = env_mod.JumpReadinessEnv
CONDITION = "profile"
BOUNDARY_TICKS = (199, 200, 224, 225, 239, 240, 274, 275, 319, 320, 600)
BOUNDARY_PAIRS = ((199, 200), (224, 225), (239, 240), (274, 275), (319, 320))


def zero8() -> np.ndarray:
    return np.zeros(8, dtype=np.float32)


class RecordingStream(io.StringIO):
    """StringIO that snapshots its contents at close time."""

    closed_value: str | None = None

    def close(self) -> None:  # pragma: no cover - exercised via ExitStack
        if not self.closed:
            self.closed_value = self.getvalue()
        super().close()


@pytest.fixture(autouse=True)
def forbid_integration(monkeypatch):
    """Any real physics stepping is a test failure, not a silent pass."""
    import mujoco

    def forbidden(*args, **kwargs):
        raise AssertionError("MuJoCo integration is forbidden in this unit test")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        if hasattr(mujoco, name):
            monkeypatch.setattr(mujoco, name, forbidden)


@pytest.fixture
def parent_calls(monkeypatch):
    """Replace the parent step with a non-integrating endpoint producer."""
    calls: list[np.ndarray] = []

    def fake_step(self, action):
        calls.append(np.asarray(action).copy())
        spec = self._fake
        if spec.reenter:
            spec.reenter = False
            self.step(zero8())
        if spec.error is not None:
            raise spec.error
        applied = action if spec.applied is None else spec.applied
        info = {"applied_action": applied}
        return (np.zeros(2, dtype=np.float32), 0.0, spec.terminated, spec.truncated, info)

    monkeypatch.setattr(env_mod.StopTurnEnv, "step", fake_step, raising=True)
    return calls


def make_record(pi_before=(0.5, -0.25, 0.125, 0.0), pi_after=(0.75, -0.5, 0.25, 0.1)):
    def vec(offset):
        return np.arange(4, dtype=np.float64) + offset

    return SimpleNamespace(
        nominal_joint_target_rad=vec(1.0),
        joint_target_rad=vec(2.0),
        joint_target_rate_limited=np.zeros(4),
        leg_extension_target_m=vec(0.1),
        requested_extension_m=vec(0.2),
        wheel_speed_target_rad_s=np.zeros(4),
        leg_pd_nm=vec(3.0),
        support_nm=vec(4.0),
        support_force_n=vec(5.0),
        wheel_nm=np.zeros(4),
        requested_torque_nm=vec(6.0),
        torque_nm=vec(7.0),
        torque_limited=np.zeros(4),
        effective_yaw_request_rps=0.0,
        memory_before=SimpleNamespace(wheel_integral_nm=np.asarray(pi_before, dtype=np.float64)),
        memory_after=SimpleNamespace(wheel_integral_nm=np.asarray(pi_after, dtype=np.float64)),
    )


def make_env(tick=225, condition=CONDITION, record=None):
    env = object.__new__(JumpReadinessEnv)
    command, _label = env_mod.fixed_height_command(tick, condition)
    clearance = float(command.clearance_m)
    loop = object()
    decision = SimpleNamespace(
        context=SimpleNamespace(state=SimpleNamespace(
            joint_position=np.array([0.1, 0.2, 0.3, 0.4]),
            joint_velocity=np.zeros(4))),
        world_command=SimpleNamespace(base_height_m=0.2 + clearance),
        ground=SimpleNamespace(height_m=0.2),
    )
    before = SimpleNamespace(
        loop=loop,
        tick=tick,
        control_time_s=tick * env_mod.CONTROL_DT_S,
        decision=decision,
        user_command=SimpleNamespace(clearance_m=clearance,
                                     forward_velocity_mps=0.0,
                                     yaw_rate_rps=0.0),
        servo_command=SimpleNamespace(clearance_m=clearance),
        heading_error_rad=0.0,
    )
    env.condition = condition
    env.loop = loop
    env._heading_context = before
    env._controller = SimpleNamespace(controller=SimpleNamespace(
        last_damping=SimpleNamespace(active=False),
        last_authority=SimpleNamespace(active=False),
        last_result=make_record() if record is None else record))
    env.plant = SimpleNamespace(data=SimpleNamespace(
        qpos=np.array([0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0]),
        xfrc_applied=np.zeros((3, 6)),
        qfrc_applied=np.zeros(9)))
    env.last_transition = SimpleNamespace(truth=SimpleNamespace(base_rpy=np.zeros(3)))
    env._prepared = {}
    env.height_callback_count = 0
    env.parent_step_calls = 0
    env.executed_ticks = []
    env.last_phase_label = None
    env.last_readiness_row = None
    env._in_step = False
    env._active = True
    env._readiness_files = ExitStack()
    env._readiness_stream = env._readiness_files.enter_context(RecordingStream())
    env._readiness_closed = False
    env._fake = SimpleNamespace(terminated=False, truncated=False, applied=None,
                                error=None, reenter=False)
    return env


def test_fixed_command_boundary_table_is_prepared_once_per_tick():
    seen = {}
    for tick in BOUNDARY_TICKS:
        env = make_env(tick=tick)
        command = env._fixed_command(tick * env_mod.CONTROL_DT_S)
        expected, label = env_mod.fixed_height_command(tick, CONDITION)
        assert float(command.clearance_m) == float(expected.clearance_m)
        assert env._prepared == {tick: label}
        assert env.height_callback_count == 1
        assert env.last_phase_label == label
        seen[tick] = (float(expected.clearance_m), label)
    for low, high in BOUNDARY_PAIRS:
        assert seen[low] != seen[high], f"ticks {low}/{high} are not a schedule boundary"
    assert env_mod.TERMINAL_PREPARED_TICK == 600


def test_fixed_command_rejects_duplicate_offtick_and_out_of_range():
    env = make_env()
    env._fixed_command(2.25)
    with pytest.raises(RuntimeError, match="already consumed"):
        env._fixed_command(2.25)
    with pytest.raises(RuntimeError, match="integer control tick"):
        env._fixed_command(2.005)
    with pytest.raises(RuntimeError, match="outside"):
        env._fixed_command((env_mod.TERMINAL_PREPARED_TICK + 1) * env_mod.CONTROL_DT_S)
    with pytest.raises(RuntimeError, match="outside"):
        env._fixed_command(-env_mod.CONTROL_DT_S)
    assert env.height_callback_count == 1
    assert list(env._prepared) == [225]


def test_step_delegates_once_and_passes_zero_action_through(parent_calls):
    env = make_env(tick=240)
    action = zero8()
    result = env.step(action)
    assert len(parent_calls) == 1
    assert np.array_equal(parent_calls[0], action)
    assert env.parent_step_calls == 1
    assert env.executed_ticks == [240]
    assert env._in_step is False
    assert result[1] == 0.0 and result[2] is False and result[3] is False
    assert np.array_equal(np.asarray(result[4]["applied_action"]), action)

    row = env.last_readiness_row
    _expected, label = env_mod.fixed_height_command(240, CONDITION)
    assert row["probe_schema"] == env_mod.READINESS_PROBE_SCHEMA
    assert row["condition"] == CONDITION
    assert row["tick"] == 240 and row["endpoint_tick"] == 241
    assert row["control_time_s"] == pytest.approx(2.40)
    assert row["display_phase"] == label
    assert row["action"] == [0.0] * 8
    assert row["raw_forward_mps"] == 0.0 and row["raw_yaw_rps"] == 0.0
    assert row["stop_latch_active"] is False and row["authority_gate_active"] is False
    assert row["quaternion_attitude_error_rad"] == 0.0
    assert row["unknown_wrench"] is False
    assert json.loads(env._readiness_stream.getvalue().strip())["tick"] == 240
    assert env.height_callback_count == 0


def test_reentrant_parent_step_is_refused(parent_calls):
    env = make_env()
    env._fake.reenter = True
    with pytest.raises(RuntimeError, match="exactly once"):
        env.step(zero8())
    assert len(parent_calls) == 1
    assert env._active is False
    assert env.last_readiness_row is None


def test_wrong_world_height_at_tick_225_rejected_even_with_correct_raw(parent_calls):
    env = make_env(tick=225)
    before = env._heading_context
    expected, _ = env_mod.fixed_height_command(225, CONDITION)
    before.decision.world_command.base_height_m = 0.2 + float(expected.clearance_m) + 1e-6
    assert float(before.user_command.clearance_m) == float(expected.clearance_m)
    assert float(before.servo_command.clearance_m) == float(expected.clearance_m)
    with pytest.raises(RuntimeError, match="world command height"):
        env.step(zero8())
    assert len(parent_calls) == 1
    assert env._active is False
    assert env._readiness_stream.getvalue() == ""


@pytest.mark.parametrize("attribute", ["last_damping", "last_authority"])
def test_actually_active_mechanism_is_rejected(parent_calls, attribute):
    env = make_env()
    setattr(env._controller.controller, attribute, SimpleNamespace(active=True))
    with pytest.raises(RuntimeError, match="inactive"):
        env.step(zero8())
    assert env._active is False
    assert env.last_readiness_row is None


def test_pi_memories_are_recorded_exactly(parent_calls):
    record = make_record(pi_before=(1.5, -2.5, 0.0, 3.25),
                         pi_after=(1.75, -2.0, 0.5, 3.0))
    env = make_env(record=record)
    env.step(zero8())
    row = env.last_readiness_row
    assert row["pi_memory_before"] == [1.5, -2.5, 0.0, 3.25]
    assert row["pi_memory_after"] == [1.75, -2.0, 0.5, 3.0]


@pytest.mark.parametrize("record", [
    make_record(pi_before=(0.1, 0.2, 0.3)),
    make_record(pi_after=(0.1, 0.2, 0.3, 0.4, 0.5)),
    None,
])
def test_missing_or_misshaped_controller_diagnostics_refused(parent_calls, record):
    env = make_env(record=record if record is not None else make_record())
    if record is None:
        env._controller.controller.last_result = None
        pattern = "diagnostics are required"
    else:
        pattern = "four entries"
    with pytest.raises(RuntimeError, match=pattern):
        env.step(zero8())
    assert env._active is False
    assert env._readiness_stream.getvalue() == ""


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
def test_final_step_closes_child_stream_only_after_writing_row(parent_calls, flag):
    env = make_env(tick=599)
    setattr(env._fake, flag, True)
    stream = env._readiness_stream
    env.step(zero8())
    assert stream.closed is True
    assert env.last_readiness_row is not None
    written = json.loads(stream.closed_value.strip())
    assert written["tick"] == 599
    assert written == env.last_readiness_row


def test_parent_error_and_post_parent_error_deactivate_without_retry(parent_calls):
    failing = make_env()
    failing._fake.error = RuntimeError("parent integration failed")
    with pytest.raises(RuntimeError, match="parent integration failed"):
        failing.step(zero8())
    assert len(parent_calls) == 1
    assert failing.parent_step_calls == 0
    assert failing.executed_ticks == []
    assert failing._active is False

    parent_calls.clear()
    transformed = make_env()
    transformed._fake.applied = np.full(8, 0.5, dtype=np.float32)
    with pytest.raises(RuntimeError, match="transformed"):
        transformed.step(zero8())
    assert len(parent_calls) == 1
    assert transformed.parent_step_calls == 1
    assert transformed._active is False
    assert transformed.last_readiness_row is None
    assert transformed._readiness_stream.getvalue() == ""
    assert transformed._readiness_stream.closed is False


def test_foreign_checkpoint_metadata_is_rejected_before_deserialization():
    with pytest.raises(TypeError, match="mapping"):
        env_mod.reject_foreign_checkpoint_metadata(["task_schema"])
    with pytest.raises(ValueError, match="before deserialization"):
        env_mod.reject_foreign_checkpoint_metadata({"task_schema": "d1-stop-turn-v1"})
    with pytest.raises(ValueError, match="no policy or checkpoint load"):
        env_mod.reject_foreign_checkpoint_metadata(
            {"task_schema": env_mod.READINESS_TASK_SCHEMA})


@pytest.mark.parametrize("action, pattern", [
    (np.full(8, 1e-7, dtype=np.float32), "exactly zero"),
    (np.zeros(8, dtype=np.float64), r"float32"),
    (np.zeros(7, dtype=np.float32), r"shape \(8,\)"),
])
def test_guards_reject_before_any_parent_delegation(parent_calls, action, pattern):
    env = make_env()
    with pytest.raises(ValueError, match=pattern):
        env.step(action)
    assert parent_calls == []
    assert env.parent_step_calls == 0
    assert env._in_step is False
    assert env._active is True


def test_stale_heading_context_blocks_delegation(parent_calls):
    env = make_env()
    env._heading_context.loop = object()
    with pytest.raises(RuntimeError, match="heading_decision"):
        env.step(zero8())
    assert parent_calls == []

    env = make_env()
    env._heading_context = None
    with pytest.raises(RuntimeError, match="heading_decision"):
        env.step(zero8())
    assert parent_calls == []
    assert env.last_readiness_row is None
