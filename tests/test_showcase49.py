"""Source and cleanup failure boundaries; no models, rendering or real children."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import render_b22_showcase49 as showcase
from scripts import resource_guard42


def file_identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def save_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    source = root / "src/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("archived model source")
    monkeypatch.setattr(showcase, "ROOT", root)
    run = root / "run"
    stage = run / "flat_1p6_B"
    (stage / "episode_0").mkdir(parents=True)
    (stage / "episode_0/states.npz").write_bytes(b"bound-state-bytes-not-loaded-by-these-tests")
    save_json(stage / "command_identity.json", {
        "actor": "B", "profile": "flat_1p6", "control_limit": 1600,
    })
    save_json(stage / "construction_receipt.json", {"geometry_receipt": "bound"})
    save_json(run / "release.json", {"inputs": {str(source): file_identity(source)}})
    readback = {
        "status": "COMPLETE", "cases": [{"completed_controls": 1600,
            "record": {"record_valid": True, "full_horizon": True},
            "safety": {"passed": True}, "final_window": {"passed": True}}],
        "release_identity": file_identity(run / "release.json"),
        "artifacts": {name: file_identity(stage / name) for name in (
            "episode_0/states.npz", "command_identity.json", "construction_receipt.json")},
    }
    suite = {"status": "COMPLETE", "failure": None,
             "finite_suite_task_qualification_passed": True, "stages": []}

    def bind():
        path = run / "flat_1p6_B_readback.json"
        save_json(path, readback)
        suite["stages"] = [{"name": "flat_1p6_B", "report": file_identity(path)}]
        save_json(run / "report.json", suite)

    bind()
    return SimpleNamespace(run=run, stage=stage, source=source, readback=readback,
                           suite=suite, bind=bind)


@pytest.mark.parametrize("change", ["partial_suite", "failed_suite", "short_case", "unsafe_case"])
def test_source_gate_rejects_partial_and_failed_archive(archive, change):
    if change == "partial_suite":
        archive.suite["status"] = "PARTIAL"
    elif change == "failed_suite":
        archive.suite["failure"] = "episode failed"
    elif change == "short_case":
        archive.readback["cases"][0]["completed_controls"] = 1599
    else:
        archive.readback["cases"][0]["safety"]["passed"] = False
    archive.bind()
    with pytest.raises(ValueError, match="complete|COMPLETE"):
        showcase.verify_source(archive.run, "flat_1p6")


@pytest.mark.parametrize("artifact", ["episode_0/states.npz", "command_identity.json"])
def test_changed_recording_or_actor_identity_cannot_be_presented(archive, artifact):
    showcase.verify_source(archive.run, "flat_1p6")
    with (archive.stage / artifact).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed archived artifact"):
        showcase.verify_source(archive.run, "flat_1p6")


def test_changed_render_model_source_is_rejected(archive):
    archive.source.write_text("different robot geometry")
    with pytest.raises(ValueError, match="render model source changed"):
        showcase.verify_source(archive.run, "flat_1p6")


@pytest.mark.parametrize("artifact", ["episode_0/states.npz", "command_identity.json",
                                      "construction_receipt.json"])
def test_source_gate_requires_bindings_for_every_consumed_artifact(archive, artifact):
    del archive.readback["artifacts"][artifact]
    archive.bind()
    with pytest.raises(ValueError):
        showcase.verify_source(archive.run, "flat_1p6")


def test_source_gate_rejects_artifact_escape(archive):
    escape = archive.run / "outside.json"
    escape.write_text("not inside this stage")
    archive.readback["artifacts"]["../outside.json"] = file_identity(escape)
    archive.bind()
    with pytest.raises(ValueError, match="unsafe artifact path"):
        showcase.verify_source(archive.run, "flat_1p6")


def test_render_rejection_retains_failure_and_never_reuses_output(archive, tmp_path, monkeypatch):
    archive.suite["status"] = "PARTIAL"
    archive.bind()
    monkeypatch.setattr(showcase.signal, "signal", lambda *_: None)
    output = tmp_path / "render"
    with pytest.raises(ValueError):
        showcase.render(archive.run, "flat_1p6", output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "INCOMPLETE"
    assert report["model_compilations"] == report["physics_step_calls"] == 0
    assert report["error"]
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        showcase.render(archive.run, "flat_1p6", output)
    assert (output / "report.json").read_bytes() == before


@pytest.fixture
def bounded(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource_guard42", resource_guard42)
    path = Path(__file__).resolve().parents[1] / "scripts/run_bounded49.py"
    spec = importlib.util.spec_from_file_location("bounded49_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def leader_already_exited():
    return SimpleNamespace(pid=4242, returncode=0, poll=lambda: 0, wait=lambda: 0)


def test_cleanup_waits_for_descendants_after_leader_exited(bounded, monkeypatch):
    process = leader_already_exited()
    clock = FakeClock()
    state = {"alive": True, "reaped": False}
    sent = []

    def killpg(pgid, sig):
        assert pgid == process.pid  # Never signal another process group.
        if not state["alive"]:
            raise ProcessLookupError
        if sig:
            sent.append(sig)
        if sig == signal.SIGKILL:
            state["alive"] = False

    def waitpid(pid, options):
        assert pid == -process.pid and options == bounded.os.WNOHANG
        if state["alive"]:
            return 0, 0
        if not state["reaped"]:
            state["reaped"] = True
            return 4243, 0
        raise ChildProcessError

    monkeypatch.setattr(bounded.os, "killpg", killpg)
    monkeypatch.setattr(bounded.os, "waitpid", waitpid)
    monkeypatch.setattr(bounded.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bounded.time, "sleep", clock.sleep)
    report = {"status": "COMPLETE", "signals": []}
    bounded.close_owned_group(process, report)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert state["reaped"]
    assert report["owned_process_group_empty"] and report["child_reaped"]


def test_cleanup_handles_group_disappearing_before_signal(bounded, monkeypatch):
    process = leader_already_exited()
    alive = [True]

    def killpg(pgid, sig):
        assert pgid == process.pid
        if sig or not alive[0]:
            alive[0] = False
            raise ProcessLookupError

    def waitpid(*_):
        raise ChildProcessError

    monkeypatch.setattr(bounded.os, "killpg", killpg)
    monkeypatch.setattr(bounded.os, "waitpid", waitpid)
    report = {"status": "COMPLETE", "signals": []}
    bounded.close_owned_group(process, report)
    assert report["owned_process_group_empty"] and report["child_reaped"]


def test_surviving_group_cannot_receive_complete_status(bounded, monkeypatch):
    process = leader_already_exited()
    clock = FakeClock()

    def killpg(pgid, _sig):
        assert pgid == process.pid

    monkeypatch.setattr(bounded.os, "killpg", killpg)
    monkeypatch.setattr(bounded.os, "waitpid", lambda *_: (0, 0))
    monkeypatch.setattr(bounded.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bounded.time, "sleep", clock.sleep)
    report = {"status": "COMPLETE", "signals": []}
    bounded.close_owned_group(process, report)
    assert not report["owned_process_group_empty"]
    assert report["status"] == "INCOMPLETE"
    assert "survived" in report["error"]


def test_cleanup_exception_is_retained_as_incomplete_report(bounded, monkeypatch, tmp_path):
    clock = FakeClock()
    output = tmp_path / "job"
    monkeypatch.setattr(sys, "argv", ["run_bounded49.py", "--scope", "/owned.service",
                                     "--output", str(output), "--timeout", "60", "--", "fake"])
    monkeypatch.setattr(bounded.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bounded.time, "sleep", clock.sleep)
    monkeypatch.setattr(bounded.signal, "signal", lambda *_: None)
    monkeypatch.setattr(bounded.ctypes, "CDLL", lambda *_, **__: SimpleNamespace(prctl=lambda *_: 0))
    monkeypatch.setattr(bounded.os, "getpriority", lambda *_: 10)
    monkeypatch.setattr(bounded.os, "sched_getaffinity", lambda *_: {0})
    monkeypatch.setattr(bounded.os, "sched_setaffinity", lambda *_: None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "LP_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(bounded, "snapshot_memory", lambda _: {
        "errors": [], "effective_available_bytes": 5 << 30,
        "owned": {"current_bytes": 100, "peak_bytes": 100, "events": {}},
    })
    monkeypatch.setattr(bounded.subprocess, "Popen", lambda *_, **__: leader_already_exited())

    def cleanup_failure(*_):
        raise PermissionError("cleanup denied by test fixture")

    monkeypatch.setattr(bounded, "close_owned_group", cleanup_failure)
    assert bounded.main() == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "INCOMPLETE"
    assert "PermissionError" in report["cleanup_error"]
    assert report["child_started"]
