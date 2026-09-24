"""Bounded pure tests for LatestRLViewer.copy_and_sync display isolation.

Nothing real is imported: MuJoCo, GLFW, torch, the engine binding and even
scripts.d1_latest_rl_viewer itself stay out of this process. Instead the actual
class body of scripts/d1_latest_rl_viewer.py is read, parsed with ast and
compiled against a fake KeyboardViewer base, so the real copy_and_sync bytecode
executes over fake MjData/plant/runtime objects. The guard formula in the source
is never copied or restated here; only its observable outcome is asserted.

Source assumptions (scripts/d1_latest_rl_viewer.py at the time of writing):
  * class LatestRLViewer(KeyboardViewer)                            (line 23)
  * __init__ only calls super().__init__(model, data, commands) and then sets
    its own record fields, so a no-op fake base is enough             (line 26)
  * copy_and_sync(plant, runtime, *, source_tick) -> bool           (line 125)
    reads self.model / self.data / self.mujoco.mj_copyData / self.cam.lookat,
    plant.model / plant.data / plant.measurement_data / plant.sampling_mode,
    self._raw_c_state(runtime) and integer counters on runtime, then calls
    self.sync() and appends to self.frame_receipts
  * _raw_c_state is the only path to the native DSO and is overridden here
    by a pure tuple, so ctypes/engine_binding are never touched      (line 114)
  * sync() owns GLFW/render/screenshot work and is overridden here with a
    test-controlled action; the real sync increments rendered_frames  (line 43)
The runtime ledger field names are recovered from the parsed method instead of
being hardcoded, so a renamed or added counter is exposed rather than skipped.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "d1_latest_rl_viewer.py"
_CLASS_NAME = "LatestRLViewer"
_METHOD_NAME = "copy_and_sync"

# Plant state channels the contract requires to survive every rendered frame.
_LIVE_CHANNELS = ("qpos", "qvel", "ctrl", "qacc_warmstart")
_MEASUREMENT_CHANNELS = ("qpos", "qvel", "ctrl")
_NQ, _NV, _NU = 7, 6, 4
_ADDRESSES = itertools.count(0x7F0000001000, 0x1000)


class _FakeModel:
    """Opaque compiled-model stand-in; only object identity is meaningful."""


class _FakeData:
    """Minimal MjData: dedicated arrays, a float time and a unique address."""

    def __init__(self, tag: str, *, time_s: float = 0.0) -> None:
        self.tag = tag
        self.qpos = np.zeros(_NQ)
        self.qvel = np.zeros(_NV)
        self.ctrl = np.zeros(_NU)
        self.qacc_warmstart = np.zeros(_NV)
        self.time = float(time_s)
        self._address = next(_ADDRESSES)

    def clone(self, tag: str) -> _FakeData:
        other = _FakeData(tag, time_s=self.time)
        for name in _LIVE_CHANNELS:
            getattr(other, name)[:] = getattr(self, name)
        return other


class _FakeMuJoCo:
    """Only mj_copyData is reachable from copy_and_sync; it never integrates."""

    def __init__(self) -> None:
        self.copy_calls: list[tuple[str, str, Any]] = []

    def mj_copyData(self, dest: _FakeData, model: Any, src: _FakeData) -> _FakeData:
        if dest is src or dest._address == src._address:
            raise AssertionError("fake mj_copyData refuses to write onto its source")
        self.copy_calls.append((dest.tag, src.tag, model))
        for name in _LIVE_CHANNELS:
            getattr(dest, name)[:] = getattr(src, name)
        dest.time = float(src.time)
        return dest


class _FakeCamera:
    def __init__(self) -> None:
        self.lookat = np.zeros(3)


class _FakeKeyboardViewer:
    """Stands in for the frozen GLFW base: no window, no context, no physics."""

    def __init__(self, model: Any, data: Any, commands: Any, **kwargs: Any) -> None:
        self.model = model
        self.data = data
        self.commands = commands
        self.mujoco = _FakeMuJoCo()
        self.cam = _FakeCamera()


class _FakePlant:
    def __init__(self, model: Any, live: _FakeData, measurement: _FakeData) -> None:
        self.model = model
        self.data = live
        self.measurement_data = measurement
        self.sampling_mode = "synchronized"


class _FakeRuntime:
    """Python ledger with exactly the counters the real method reads."""

    def __init__(self, names: tuple[str, ...]) -> None:
        for index, name in enumerate(names):
            setattr(self, name, 10 * (index + 1))


def _load_actual_class() -> tuple[type, ast.FunctionDef]:
    """Compile the actual LatestRLViewer body against the fake base class."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == _CLASS_NAME]
    assert len(classes) == 1, f"{_CLASS_NAME} not found exactly once in {_SOURCE}"
    methods = [node for node in classes[0].body
               if isinstance(node, ast.FunctionDef) and node.name == _METHOD_NAME]
    assert len(methods) == 1, f"{_METHOD_NAME} not found exactly once in {_CLASS_NAME}"
    # Keep annotations unevaluated so the fake namespace needs no real types.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0)
    module = ast.fix_missing_locations(
        ast.Module(body=[future, classes[0]], type_ignores=[]))
    namespace: dict[str, Any] = {
        "KeyboardViewer": _FakeKeyboardViewer, "np": np, "hashlib": hashlib,
    }
    exec(compile(module, str(_SOURCE), "exec"), namespace)  # noqa: S102 - vetted source
    return namespace[_CLASS_NAME], methods[0]


def _ledger_names(method: ast.FunctionDef) -> tuple[str, ...]:
    """Every runtime.<attr> the real method snapshots, in source order."""
    names: list[str] = []
    for node in ast.walk(method):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "runtime" and node.attr not in names):
            names.append(node.attr)
    return tuple(names)


_ACTUAL_VIEWER, _METHOD_NODE = _load_actual_class()
_LEDGER_NAMES = _ledger_names(_METHOD_NODE)


class _StubViewer(_ACTUAL_VIEWER):  # type: ignore[misc, valid-type]
    """Real copy_and_sync over a pure native snapshot and a pure sync action."""

    def __init__(self, model: Any, data: Any,
                 *, c_state: tuple[int, ...] = (3, 6000, 0)) -> None:
        super().__init__(model, data, object())
        self._c_state = c_state
        self._action: Any = None
        self.sync_calls = 0

    def _raw_c_state(self, runtime: Any) -> tuple[int, ...]:
        return tuple(self._c_state)

    def sync(self) -> bool:
        self.sync_calls += 1
        if self._action is not None:
            self._action()
        self.rendered_frames += 1
        return True


def _build_case() -> tuple[_StubViewer, _FakePlant, _FakeRuntime]:
    """A synchronized endpoint: measurement equals live, display is separate."""
    model = _FakeModel()
    live = _FakeData("live", time_s=1.75)
    live.qpos[:] = [0.512, -0.004, 0.455, 0.9995, 0.0, 0.0316, 0.0]
    live.qvel[:] = [0.2031, -0.0012, 0.0024, 0.0, 0.0041, 0.0]
    live.ctrl[:] = [0.31, -0.29, 0.30, -0.28]
    live.qacc_warmstart[:] = [0.021, 0.0, -0.013, 0.0, 0.0033, 0.0]
    measurement = live.clone("measurement")
    viewer = _StubViewer(model, _FakeData("display"))
    return viewer, _FakePlant(model, live, measurement), _FakeRuntime(_LEDGER_NAMES)


def _snapshot(data: _FakeData) -> tuple[dict[str, np.ndarray], float]:
    return ({name: getattr(data, name).copy() for name in _LIVE_CHANNELS},
            float(data.time))


def _assert_unchanged(data: _FakeData, snapshot: tuple[dict[str, np.ndarray], float],
                      label: str) -> None:
    arrays, stamp = snapshot
    for name, value in arrays.items():
        assert np.array_equal(getattr(data, name), value), f"{label}.{name} changed"
    assert float(data.time) == stamp, f"{label}.time changed"


def _ledger(runtime: _FakeRuntime) -> tuple[int, ...]:
    return tuple(getattr(runtime, name) for name in _LEDGER_NAMES)


def test_copy_and_sync_renders_a_synchronized_endpoint_and_rejects_bad_sources():
    viewer, plant, runtime = _build_case()
    live_before = _snapshot(plant.data)
    measurement_before = _snapshot(plant.measurement_data)
    ledger_before = _ledger(runtime)

    assert viewer.copy_and_sync(plant, runtime, source_tick=613) is True

    # The display got its own copy of the measurement endpoint, once.
    assert viewer.sync_calls == 1
    assert viewer.mujoco.copy_calls == [("display", "measurement", plant.model)]
    assert viewer.data is not plant.data
    assert viewer.data is not plant.measurement_data
    for name in _LIVE_CHANNELS:
        assert np.array_equal(getattr(viewer.data, name),
                              getattr(plant.measurement_data, name))
    assert float(viewer.data.time) == float(plant.measurement_data.time)
    assert np.array_equal(viewer.cam.lookat, plant.measurement_data.qpos[:3])
    assert viewer._source_tick == 613

    # The integrated plant, the measurement endpoint and the ledger are intact.
    _assert_unchanged(plant.data, live_before, "live")
    _assert_unchanged(plant.measurement_data, measurement_before, "measurement")
    assert _ledger(runtime) == ledger_before

    assert viewer.rendered_frames == 1
    assert viewer.skipped_frames == 0
    assert len(viewer.frame_receipts) == 1
    receipt = viewer.frame_receipts[0]
    assert receipt["frame_index"] == 0
    assert receipt["source_tick"] == 613
    assert receipt["source_time_s"] == pytest.approx(1.75)
    assert receipt["native_counter_delta"] == 0
    assert receipt["python_counter_delta"] == 0
    assert receipt["live_integrator_unchanged"] is True
    assert receipt["measurement_unchanged"] is True
    assert receipt["screenshot_count"] == 0

    def desync_array(name: str):
        def mutate(viewer_: _StubViewer, plant_: _FakePlant) -> None:
            getattr(plant_.measurement_data, name)[0] += 1e-9
        return mutate

    def desync_time(viewer_: _StubViewer, plant_: _FakePlant) -> None:
        plant_.measurement_data.time += 1e-9

    def wrong_model(viewer_: _StubViewer, plant_: _FakePlant) -> None:
        viewer_.model = _FakeModel()

    def unsynchronized_sampling(viewer_: _StubViewer, plant_: _FakePlant) -> None:
        plant_.sampling_mode = "free_running"

    def display_is_live(viewer_: _StubViewer, plant_: _FakePlant) -> None:
        viewer_.data = plant_.data

    def display_is_measurement(viewer_: _StubViewer, plant_: _FakePlant) -> None:
        viewer_.data = plant_.measurement_data

    def display_aliases_live(viewer_: _StubViewer, plant_: _FakePlant) -> None:
        viewer_.data._address = plant_.data._address

    def display_aliases_measurement(viewer_: _StubViewer, plant_: _FakePlant) -> None:
        viewer_.data._address = plant_.measurement_data._address

    rejections = [("measurement qpos desynchronized", desync_array("qpos")),
                  ("measurement qvel desynchronized", desync_array("qvel")),
                  ("measurement ctrl desynchronized", desync_array("ctrl")),
                  ("measurement time desynchronized", desync_time),
                  ("display uses a different model", wrong_model),
                  ("plant is not sampling synchronized", unsynchronized_sampling),
                  ("display data is the live data", display_is_live),
                  ("display data is the measurement data", display_is_measurement),
                  ("display data aliases the live allocation", display_aliases_live),
                  ("display data aliases the measurement allocation",
                   display_aliases_measurement)]
    for label, mutate in rejections:
        bad_viewer, bad_plant, bad_runtime = _build_case()
        mutate(bad_viewer, bad_plant)
        live_state = _snapshot(bad_plant.data)
        measurement_state = _snapshot(bad_plant.measurement_data)
        with pytest.raises(RuntimeError):
            bad_viewer.copy_and_sync(bad_plant, bad_runtime, source_tick=614)
        # A rejected source is refused before any copy, render or receipt.
        assert bad_viewer.sync_calls == 0, label
        assert bad_viewer.mujoco.copy_calls == [], label
        assert bad_viewer.rendered_frames == 0, label
        assert bad_viewer.frame_receipts == [], label
        _assert_unchanged(bad_plant.data, live_state, f"live ({label})")
        _assert_unchanged(bad_plant.measurement_data, measurement_state,
                          f"measurement ({label})")


def test_copy_and_sync_fails_on_any_state_or_counter_mutation_during_render():
    # The contract requires the Python ledger snapshot to cover these counters.
    assert _LEDGER_NAMES, "copy_and_sync snapshots no runtime counter at all"
    for required in ("forward_calls", "setconst_calls", "attempted", "returned"):
        assert required in _LEDGER_NAMES, f"runtime.{required} is not snapshotted"

    def mutate_data_array(target: str, name: str):
        def inject(viewer: _StubViewer, plant: _FakePlant, runtime: _FakeRuntime) -> None:
            getattr(getattr(plant, target), name)[0] += 1e-9
        return inject

    def mutate_data_time(target: str):
        def inject(viewer: _StubViewer, plant: _FakePlant, runtime: _FakeRuntime) -> None:
            getattr(plant, target).time += 1e-9
        return inject

    def mutate_native(viewer: _StubViewer, plant: _FakePlant,
                      runtime: _FakeRuntime) -> None:
        viewer._c_state = tuple(value + 1 for value in viewer._c_state)

    def mutate_ledger(name: str):
        def inject(viewer: _StubViewer, plant: _FakePlant, runtime: _FakeRuntime) -> None:
            setattr(runtime, name, getattr(runtime, name) + 1)
        return inject

    injections: list[tuple[str, Any]] = []
    for name in _LIVE_CHANNELS:
        injections.append((f"live {name}", mutate_data_array("data", name)))
    injections.append(("live time", mutate_data_time("data")))
    for name in _MEASUREMENT_CHANNELS:
        injections.append((f"measurement {name}",
                           mutate_data_array("measurement_data", name)))
    injections.append(("measurement time", mutate_data_time("measurement_data")))
    injections.append(("native C snapshot", mutate_native))
    for name in _LEDGER_NAMES:
        injections.append((f"runtime.{name}", mutate_ledger(name)))

    for label, inject in injections:
        viewer, plant, runtime = _build_case()
        viewer._action = lambda v=viewer, p=plant, r=runtime, f=inject: f(v, p, r)
        live_before = _snapshot(plant.data)
        measurement_before = _snapshot(plant.measurement_data)
        ledger_before = _ledger(runtime)
        native_before = tuple(viewer._c_state)

        # The isolation guard, not an input precondition, must reject the frame.
        with pytest.raises(RuntimeError, match="rendering modified"):
            viewer.copy_and_sync(plant, runtime, source_tick=615)

        assert viewer.sync_calls == 1, label
        assert viewer.frame_receipts == [], label
        # The injection really happened, so the guard caught an actual change.
        changed = (not np.array_equal(
                       np.concatenate([getattr(plant.data, n).ravel()
                                       for n in _LIVE_CHANNELS]),
                       np.concatenate([live_before[0][n].ravel()
                                       for n in _LIVE_CHANNELS]))
                   or float(plant.data.time) != live_before[1]
                   or not np.array_equal(
                       np.concatenate([getattr(plant.measurement_data, n).ravel()
                                       for n in _MEASUREMENT_CHANNELS]),
                       np.concatenate([measurement_before[0][n].ravel()
                                       for n in _MEASUREMENT_CHANNELS]))
                   or float(plant.measurement_data.time) != measurement_before[1]
                   or _ledger(runtime) != ledger_before
                   or tuple(viewer._c_state) != native_before)
        assert changed, f"{label}: injection did not mutate anything"
