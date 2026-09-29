"""Four new pure checks for compact entry/return, native gates and display dispatch."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


native = load("compact_native_q08")
# The frozen keyboard viewer imports a terrain drawing helper at module scope.
# This dispatch test never draws: give only that helper an explicit failing
# double so collection cannot load MuJoCo. The engine import guard stays active.
terrain_name = "scripts.d1_terrain_display"
terrain_previous = sys.modules.get(terrain_name)
terrain_double = ModuleType(terrain_name)


def unexpected_terrain_draw(*args, **kwargs):
    raise AssertionError("pure viewer dispatch test attempted terrain drawing")


terrain_double.append_terrain_grid = unexpected_terrain_draw
sys.modules[terrain_name] = terrain_double
try:
    view = load("compact_viewer_q08")
finally:
    if terrain_previous is None:
        del sys.modules[terrain_name]
    else:
        sys.modules[terrain_name] = terrain_previous
worker = load("run_compact_q08")


def test_native_buffer_five_real_returns_cap_and_unreturned_partial(tmp_path):
    data = NS(
        qpos=np.arange(3, dtype=float),
        qvel=np.arange(2, dtype=float),
        qacc_warmstart=np.ones(2),
        ctrl=np.array([3.0]),
        time=0.0,
    )
    buf = native.NativeBuffer(5, 3, 2, 1)
    for step in range(5):
        before = data.qpos.copy()
        index = buf.entry(data)
        data.qpos += 0.125
        data.time += 0.002
        buf.exit(data, index)
        assert np.array_equal(buf.arrays["qpos_before"][step], before)
    assert (buf.attempted, buf.returned) == (5, 5)
    with pytest.raises(RuntimeError, match="capacity"):
        buf.entry(data)
    partial = native.NativeBuffer(5, 3, 2, 1)
    partial.entry(data)
    with pytest.raises(RuntimeError, match="unreturned"):
        partial.entry(data)
    path = tmp_path / "partial.npz"
    partial.save(path)
    with np.load(path, allow_pickle=False) as stored:
        assert stored["returned"].tolist() == [False]
        assert stored["qpos_before"].shape == (1, 3)
        assert np.array_equal(stored["qpos_before"][0], data.qpos)


def test_native_guard_rejects_nonfinite_invalid_frame_and_nonwheel():
    frame = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contact = NS(
        geom1=0,
        geom2=1,
        frame=frame.ravel(),
        pos=np.zeros(3),
        friction=np.ones(5),
        solref=np.zeros(2),
        solreffriction=np.zeros(2),
        solimp=np.ones(5),
        dist=0.0,
        includemargin=0.0,
    )
    model = NS(ngeom=2, geom_bodyid=np.array([0, 1]))
    data = NS(geom_xpos=np.zeros((2, 3)), geom_xmat=np.tile(np.eye(3).ravel(), (2, 1)))
    plant = NS(
        model=model,
        data=data,
        terrain_geom_ids={0},
        floor_geom_id=0,
        box_geom_id=-1,
        _wheel_index_by_body_id={1: 0},
    )
    assert native.audit_native_contact(plant, contact) is False
    contact.pos[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        native.audit_native_contact(plant, contact)
    contact.pos[0] = 0.0
    contact.frame = frame.ravel() * 2
    with pytest.raises(ValueError, match="orthonormal"):
        native.audit_native_contact(plant, contact)
    contact.frame = frame.ravel()
    plant._wheel_index_by_body_id = {}
    with pytest.raises(ValueError, match="nonwheel"):
        native.audit_native_contact(plant, contact)
    assert native.posture_degrees(np.array([0, 0, 0, 1, 0, 0, 0])) == (0.0, 0.0)
    with pytest.raises(ValueError, match="quaternion"):
        native.posture_degrees(np.zeros(7))


def test_viewer_strided_skip_makes_no_parent_call_and_capture_forces_draw(monkeypatch):
    calls = []

    def parent(self, plant, runtime, *, source_tick):
        assert self._last_render == float("-inf")
        calls.append((plant, runtime, source_tick))
        return True

    monkeypatch.setattr(view.LatestRLViewer, "copy_and_sync", parent)
    obj = view.CompactViewer.__new__(view.CompactViewer)
    obj._capture_requests = []
    obj.source_ticks_skipped = []
    obj.draw_calls = []
    obj.skipped_frames = 0
    for tick in range(21):
        obj.copy_and_sync("plant", "ledger", source_tick=tick)
    assert [x[2] for x in calls] == [0, 10, 20]
    assert len(obj.source_ticks_skipped) == 18
    obj._capture_requests = [("forced", "unused")]
    assert obj.copy_and_sync("plant", "ledger", source_tick=23) is True
    assert calls[-1][2] == 23 and obj.draw_calls[-1]["forced_capture"]
    assert all(row["returned"] and row["rendered"] for row in obj.draw_calls)


def test_worker_preflight_rejects_source_drift_before_engine(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("original")
    session_path = tmp_path / "session.json"
    argv = [str(HERE / "run_compact_q08.py"), "--session", str(session_path)]
    monkeypatch.setattr(sys, "argv", argv)
    session = {
        "schema": worker.SCHEMA,
        "contract_sha256": worker.CONTRACT_SHA,
        "control_limit": 800,
        "normal_native_limit": 4000,
        "compiler_native_limit": 3,
        "segments": [400, 400],
        "seed": 77351,
        "actor": "final_policy",
        "terrain": "box",
        "retry_permitted": False,
        "argv": argv,
        "source_hashes": {
            str(source): {"bytes": source.stat().st_size, "sha256": "0" * 64}
        },
    }
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        worker.preflight(session, session_path)
    assert not any(
        name in sys.modules for name in ("mujoco", "glfw", "torch", "gymnasium")
    )
