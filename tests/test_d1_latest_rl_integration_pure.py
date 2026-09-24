"""Pure checks of actual 07 command preparation and frozen budget reservation."""

from __future__ import annotations

import ast
import copy
import json
import sys
import types
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.d1_latest_rl_controls import LatestRLCommands
from scripts.d1_latest_rl_validation import ValidationDriver
from scripts.run_d1_latest_rl import InteractiveCommandSource


def test_actual_command_callback_consumes_next_tick_intent_and_settles_after_reset(monkeypatch):
    # Only replace the command DTO import; execute the actual production callback,
    # actual commands and actual acceptance driver. No plant or model is imported.
    interface = types.ModuleType("wheel_legged_control.d1.control_loop")
    interface.D1MotionCommand = namedtuple(
        "D1MotionCommand", "forward_velocity_mps yaw_rate_rps clearance_m")
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    bindings = []
    inner = SimpleNamespace(
        _steps=0, raw_callback_count=0, command_records=[],
        loop=SimpleNamespace(provider=SimpleNamespace(
            ground_reference=lambda: SimpleNamespace(height_m=.015))),
        _controller=SimpleNamespace(controller=SimpleNamespace(
            bind_raw_command=lambda **value: bindings.append(value))),
    )
    commands = LatestRLCommands()
    driver = ValidationDriver("headless_zero_plane", commands)
    source = InteractiveCommandSource(inner, commands)
    schedules = []
    for segment, limit in enumerate((1000, 200)):
        commands.reset_for_new_segment()
        inner._steps, inner.raw_callback_count, inner.command_records = 0, 0, []
        prepared = source(0.0)
        executed = []
        for tick in range(limit + 1):
            driver.before_poll(segment, tick)
            driver.after_poll(segment, tick)
            if tick == limit:
                assert commands.consume_reset_request() if segment == 0 else commands.stopped
                break
            executed.append(prepared.forward_velocity_mps)
            assert prepared.yaw_rate_rps == 0.0
            assert prepared.clearance_m == pytest.approx(.440, abs=1e-15)
            assert inner.command_records[-1]["tick"] == tick
            assert bindings[-1]["forward_velocity_mps"] == executed[-1]
            inner._steps = tick + 1
            prepared = source((tick + 1) * .01)
        assert inner.raw_callback_count == limit + 1
        assert len(inner.command_records) == limit + 1
        schedules.append(executed)
    assert schedules[0] == [0.] * 175 + [.20] * 226 + [.25] * 400 + [0.] * 199
    assert schedules[1] == [0.] * 175 + [.20] * 25
    assert driver.report()["passed"] and driver.report()["complete"]
    previous_calls = len(bindings)
    with pytest.raises(RuntimeError, match="tick/time mismatch"):
        source(.0005)
    assert len(bindings) == previous_calls
    inner._steps = 1200
    assert source(12.).forward_velocity_mps == 0.0
    assert inner.command_records[-1]["terminal_prepared_only"] is True


def test_actual_budget_and_execution_gate_reject_refunds_or_stale_evidence(tmp_path):
    # Run the unchanged runtime method through its pure reservation seam.
    # Importing the runtime itself would import MuJoCo, which this test forbids.
    path = Path(__file__).resolve().parents[1] / "scripts/d1_rolling_engine_runtime.py"
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "RollingEngineRuntime")
    method = next(node for node in definition.body
                  if isinstance(node, ast.FunctionDef) and node.name == "start_segment")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    runtime = SimpleNamespace(_fatal=False, _inside_control=False,
                              segments=[], control_limit=1200, _segment=None)
    start = types.MethodType(namespace["start_segment"], runtime)
    start("segment_00", 1000)
    runtime._segment["attempted"] = runtime._segment["completed"] = 1
    start("segment_01", 200)
    assert sum(row["limit"] for row in runtime.segments) == 1200
    assert runtime.segments[0]["completed"] == 1
    with pytest.raises(RuntimeError, match="reservations exceed"):
        start("segment_02", 1)
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        start("segment_00", 1)
    for invalid in (True, 0, -1, 1.0):
        with pytest.raises(ValueError):
            start("invalid", invalid)
    runtime._fatal = True
    with pytest.raises(RuntimeError, match="after failure"):
        start("fatal", 1)

    from scripts.play_d1_latest_rl import (
        CONTRACT_SHA256,
        identity,
        validate_execution_evidence,
    )

    repo = path.parents[1]
    tested_paths = list((repo / "scripts").glob("*latest_rl*.py")) + [
        repo / "tests" / name for name in (
            "test_d1_latest_rl_controls_opus.py", "test_d1_latest_rl_render_opus.py",
            "test_d1_latest_rl_integration_pure.py")]
    hashes = {str(item): identity(item) for item in tested_paths}
    contract = repo / "docs/latest_rl_gui_contract_07.md"
    hashes[str(contract)] = identity(contract)
    receipt_path, go_path = tmp_path / "synthetic_receipt.json", tmp_path / "synthetic_go.json"
    receipt = {
        "all_passed": True, "exit_code": 0, "collected": 12,
        "blocked_engine_imports": [], "sources_changed_during_tests": [],
        "results": [{"nodeid": f"synthetic_fixture_{i}", "when": "call", "outcome": "passed"}
                    for i in range(12)],
        "source_sha256": {str(item): hashes[str(item)]["sha256"] for item in tested_paths},
        "file_sha256": {str(item): hashes[str(item)]["sha256"] for item in tested_paths
                        if item.parent.name == "tests"},
    }
    receipt_path.write_text(json.dumps(receipt))
    hashes[str(receipt_path)] = identity(receipt_path)
    go = {"schema": "d1-latest-rl-go-v1", "decision": "GO",
          "contract_sha256": CONTRACT_SHA256, "inputs": copy.deepcopy(hashes),
          "pure_test_receipt_sha256": identity(receipt_path)["sha256"]}
    go_path.write_text(json.dumps(go))
    validate_execution_evidence(receipt_path, go_path, copy.deepcopy(hashes))
    # These temporary synthetic fixtures test the real gate only. They never
    # authorize a process and are not written into an execution output directory.
    for field, bad in (("all_passed", False), ("collected", 11),
                       ("blocked_engine_imports", ["mujoco"]),
                       ("sources_changed_during_tests", ["changed_source"])):
        invalid = {**receipt, field: bad}
        receipt_path.write_text(json.dumps(invalid))
        with pytest.raises(RuntimeError):
            validate_execution_evidence(receipt_path, go_path, copy.deepcopy(hashes))
    receipt_path.write_text(json.dumps(receipt))
    for field, bad in (("decision", "WAIT"), ("inputs", {}),
                       ("pure_test_receipt_sha256", "0" * 64)):
        go_path.write_text(json.dumps({**go, field: bad}))
        with pytest.raises(RuntimeError):
            validate_execution_evidence(receipt_path, go_path, copy.deepcopy(hashes))
    go_path.write_text(json.dumps(go))
    stale = copy.deepcopy(receipt)
    stale["source_sha256"][str(tested_paths[0])] = "0" * 64
    receipt_path.write_text(json.dumps(stale))
    with pytest.raises(RuntimeError, match="source changed"):
        validate_execution_evidence(receipt_path, go_path, copy.deepcopy(hashes))
