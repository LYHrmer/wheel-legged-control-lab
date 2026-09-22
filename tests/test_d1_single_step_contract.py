"""Pure mocks and static guards: these tests never integrate a real plant."""
from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.d1_single_step_env import _require_zero_action, raw_command_at_tick
from scripts.d1_single_step_geometry import geom_world_bounds
from scripts.d1_single_step_records import PhysicsCallLedger, StreamingNativeObserver


@pytest.mark.parametrize(("tick", "forward"), [(0, 0.), (199, 0.), (200, .2), (999, .2), (1000, 0.), (1199, 0.), (1200, 0.)])
def test_raw_boundary(tick, forward):
    assert raw_command_at_tick(tick) == {"tick": tick, "forward_velocity_mps": forward, "yaw_rate_rps": 0., "world_height_m": .455}


@pytest.mark.parametrize("action", [np.ones(8), np.zeros(7), np.zeros(8, dtype=bool), ["0"]*8, [0j]*8, [float("nan")]*8])
def test_nonzero_or_invalid_action_rejected(action):
    with pytest.raises((TypeError, ValueError)):
        _require_zero_action(action)


def test_cylinder_tilt_matches_analytic_support():
    # Axis tilted 45 degrees in x-z; radius 2, half-length 3.
    a = 1/np.sqrt(2.)
    rotation = np.array([[a, 0., a], [0., 1., 0.], [-a, 0., a]])
    model = SimpleNamespace(ngeom=1, geom_type=[int(mujoco.mjtGeom.mjGEOM_CYLINDER)], geom_size=[[2., 3., 0.]])
    data = SimpleNamespace(geom_xpos=[[1., 2., 3.]], geom_xmat=[rotation.ravel()])
    lo, hi = geom_world_bounds(model, data, 0)
    np.testing.assert_allclose(hi, [1+5*a, 4., 3+5*a], rtol=0., atol=1e-14)
    np.testing.assert_allclose(lo, [1-5*a, 0., 3-5*a], rtol=0., atol=1e-14)


def fake_pair():
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
    data = SimpleNamespace(time=0., qpos=np.r_[np.zeros(3), 1., np.zeros(19)],
        qvel=np.zeros(22), ctrl=np.zeros(16), qacc_warmstart=np.zeros(22),
        xfrc_applied=np.zeros((18, 6)), qfrc_applied=np.zeros(22), act=np.zeros(0))
    return model, data


def test_ledger_refuses_unbound_bulk_foreign_and_over_budget(monkeypatch):
    calls = []
    def fake(model, data):
        calls.append(1); data.time += .002
    monkeypatch.setattr(mujoco, "mj_step", fake)
    model, data = fake_pair()
    with PhysicsCallLedger(native_limit=1) as ledger:
        with pytest.raises(RuntimeError):
            mujoco.mj_step(model, data)
        ledger.allowed = (model, data)
        for args, kwargs in [((model, data, 2), {}), ((model, data), {"nstep": 2}), ((object(), data), {})]:
            with pytest.raises(RuntimeError):
                mujoco.mj_step(*args, **kwargs)
        mujoco.mj_step(model, data)
        with pytest.raises(RuntimeError):
            mujoco.mj_step(model, data)
    assert len(calls) == ledger.returned == ledger.advanced_substeps == 1
    assert ledger.forbidden_calls == 5


@pytest.mark.parametrize("advance", [False, True])
def test_exception_attempt_and_advanced_count_separate(monkeypatch, advance):
    def fake(model, data):
        if advance:
            data.time += .002
        raise RuntimeError("mock native error")
    monkeypatch.setattr(mujoco, "mj_step", fake)
    model, data = fake_pair()
    with PhysicsCallLedger(native_limit=5) as ledger:
        ledger.allowed = (model, data)
        with pytest.raises(RuntimeError):
            mujoco.mj_step(model, data)
    assert (ledger.attempted, ledger.returned, ledger.failed) == (1, 0, 1)
    assert ledger.advanced_substeps == int(advance)


def test_streaming_observer_keeps_partial_failure(monkeypatch, tmp_path):
    model, data = fake_pair()
    calls = []
    def fake(model, data):
        calls.append(1); data.time += .002
        if len(calls) == 3:
            raise RuntimeError("mock interruption in a control interval")
    monkeypatch.setattr(mujoco, "mj_step", fake)
    monkeypatch.setattr("scripts.d1_single_step_records.sample_step_contacts", lambda _: {"contacts": [], "geometric_box_contact": False})
    observer = StreamingNativeObserver(SimpleNamespace(model=model, data=data), tmp_path)
    with pytest.raises(RuntimeError), observer:
        for _ in range(5):
            mujoco.mj_step(model, data)
    assert observer.attempted_calls == 3 and observer.returned_calls == 2
    assert mujoco.mj_step is fake
    with gzip.open(tmp_path/"native.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 3 and rows[-1]["returned"] is False
    assert rows[-1]["qpos_after_error"] == data.qpos.tolist()
    assert rows[-1]["end_time_s"] == data.time


def test_precontact_divergence_stops_before_delegate(monkeypatch, tmp_path):
    model, data = fake_pair()
    calls = []
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: calls.append(1))
    baseline = [{"qpos_before": data.qpos+1., "qvel_before": data.qvel, "ctrl_nm": data.ctrl}]
    observer = StreamingNativeObserver(SimpleNamespace(model=model, data=data), tmp_path, baseline=baseline)
    with pytest.raises(RuntimeError, match="divergence"), observer:
        mujoco.mj_step(model, data)
    assert not calls and not observer.entries and observer.prefix_errors


def test_world_height_conversion_and_cached_callback_without_plant():
    # Exercise only the actual bound callback on a minimal published provider.
    from scripts.d1_single_step_env import D1SingleStepEnv
    bindings = []
    env = object.__new__(D1SingleStepEnv)
    env._steps = 200; env.raw_callback_count = 0; env.command_records = []
    env.loop = SimpleNamespace(provider=SimpleNamespace(ground_reference=lambda: SimpleNamespace(height_m=.015)))
    env._controller = SimpleNamespace(controller=SimpleNamespace(bind_raw_command=lambda **kw: bindings.append(kw)))
    command = env._command_source(2.)
    assert command.clearance_m == .455-.015
    assert command.clearance_m+.015 == .455 and command.forward_velocity_mps == .2
    assert bindings == [{"forward_velocity_mps": .2, "yaw_rate_rps": 0., "control_time_s": 2.}]
    assert env.raw_callback_count == 1
