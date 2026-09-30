"""Saved-only regression coverage for ordered real OS callback adjudication."""
from pathlib import Path
import gzip
import json
import importlib.util

SOURCE = Path(__file__).with_name("reader25.py")
spec = importlib.util.spec_from_file_location("reader25_under_test", SOURCE)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def test_saved_keyboard27_missing_only_redundant_510_release():
    run = SOURCE.parents[2]/"continuation23/keyboard_gui_01"
    sent = [e for e in json.loads((run/"x11_events_receipt.json").read_text())["events"]
            if e["operation"] != "initial_focus"]
    callbacks = []
    with gzip.open(run/"input_snapshots.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            for event in row["poll"]["events"]:
                if event["type"] == "key" and event["action"] in (0,1):
                    callbacks.append({"operation":"key_down" if event["action"] == 1 else "key_up",
                                      "key":reader.GLFW_KEYS[event["key"]],"wall_ns":event["wall_ns"]})
                elif event["type"] == "focus":
                    callbacks.append({"operation":"focus_return" if event["focused"] else "focus_lost",
                                      "wall_ns":event["wall_ns"]})
            callbacks.extend({"operation":"button_click","button":"stop" if e["key"] == "space" else "reset",
                              "wall_ns":e["wall_ns"]} for e in row["raw_button_events"])
    result = reader.ordered_callback_matches(sent, callbacks)
    assert result["passed"] is False
    assert len(result["matches"]) == 26
    assert [e["tick"] for e in result["unmatched_events"]] == [510]
    matched = next(e for e in result["matches"] if e["planned_tick"] == 590)
    assert matched["callback_wall_ns"] == 95378593810449
    assert matched["latency_ms"] == 42.095848


def test_two_edges_in_one_poll_after_both_injections_are_preserved():
    sent = [{"operation":"key_down","key":"space","tick":400,"wall_ns":1_000_000_000},
            {"operation":"key_up","key":"space","tick":405,"wall_ns":1_050_000_000}]
    callbacks = [{"operation":"key_up","key":"space","wall_ns":1_071_000_000},
                 {"operation":"key_down","key":"space","wall_ns":1_070_000_000}]
    result = reader.ordered_callback_matches(sent, callbacks)
    assert result["passed"] is True
    assert [e["callback_index"] for e in result["matches"]] == [1,0]


def test_one_callback_cannot_satisfy_two_injections():
    sent = [{"operation":"key_up","key":"w","wall_ns":1_000_000_000},
            {"operation":"key_up","key":"w","wall_ns":1_100_000_000}]
    result = reader.ordered_callback_matches(sent,[{"operation":"key_up","key":"w","wall_ns":1_110_000_000}])
    assert result["passed"] is False
    assert len(result["matches"]) == 1
    assert result["matches"][0]["event_index"] == 1


def test_reversed_callback_actions_cannot_pass_as_ordered_events():
    sent = [{"operation":"key_down","key":"w","wall_ns":1_000_000_000},
            {"operation":"key_up","key":"w","wall_ns":1_050_000_000}]
    callbacks = [{"operation":"key_up","key":"w","wall_ns":1_070_000_000},
                 {"operation":"key_down","key":"w","wall_ns":1_080_000_000}]
    result = reader.ordered_callback_matches(sent,callbacks)
    assert result["passed"] is False
    assert len(result["matches"]) == 1


def test_original_clock_bounds_remain_strict():
    sent = [{"operation":"key_down","key":"w","wall_ns":2_000_000_000}]
    for when in (1_949_999_999,3_000_000_001):
        result = reader.ordered_callback_matches(sent,[{"operation":"key_down","key":"w","wall_ns":when}])
        assert result["passed"] is False
