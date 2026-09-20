"""Fixed physical-window views, retaining original raw references and G1 limits."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from scripts import probe_d1_heading_g1 as g1


def _segment(original_case, rows, positions, gates, first, end):
    """Score a read-only time-translated view; never forge a truncation flag."""
    selected = deepcopy(rows[first:end])
    selected_positions = np.asarray(positions)[first:end+1]
    coverage = (len(selected) == end-first and len(selected_positions) == end-first+1
        and all(row["tick"] == first+i and row["endpoint_tick"] == first+i+1
                and not row["terminated"] for i, row in enumerate(selected)))
    if not selected:
        return {"physical_row_slice": [first, end], "segment_coverage": False,
                "gates": {"checks": {"segment_coverage": False}, "passed": False,
                          "failed": ["segment_coverage"]}, "metrics": {}}
    for row in selected:
        row["tick"] -= first
        row["endpoint_tick"] -= first
    metrics, gate = g1.gates_and_metrics(original_case, selected, selected_positions, gates)
    # The stop prefix is not an independently truncated episode. Keep that fact
    # visible while testing coverage, separate from global real completion.
    whole_episode_predicate = metrics.pop("completed_duration")
    gate["checks"].pop("complete_duration")
    gate["checks"]["segment_coverage"] = coverage
    gate["passed"] = all(gate["checks"].values())
    gate["failed"] = [name for name, passed in gate["checks"].items() if not passed]
    return {"original_case": original_case["name"], "physical_row_slice": [first, end],
            "physical_position_slice": [first, end+1], "local_tick_shift": -first,
            "original_duration_predicate_on_view": whole_episode_predicate,
            "actual_last_row_truncated": bool(selected[-1]["truncated"]),
            "segment_coverage": coverage, "reference_reset": False,
            "metrics": metrics, "gates": gate}


def score_handoff(case, rows, positions, original_cases, gates):
    """Score global 1400, stop 0:800 and turn 600:1400 with original numbers."""
    if case["name"] not in ("forward_stop_then_left_turn", "reverse_stop_then_right_turn"):
        raise ValueError("only the two frozen handoff cases are supported")
    if case["max_transitions"] != 1400 or case["gate_groups"] != ["all_cases"]:
        raise ValueError("handoff must retain the fixed global horizon and gate group")
    original = {c["name"]: c for c in original_cases}
    positive = case["command"]["forward_target_mps"] > 0.
    stop = original["flat_forward_stop" if positive else "flat_reverse_stop"]
    turn = original["stationary_turn_left_hold" if positive else "stationary_turn_right_hold"]
    metric, global_gate = g1.gates_and_metrics(case, rows, positions, gates)
    stopped = _segment(stop, rows, positions, gates, 0, 800)
    turned = _segment(turn, rows, positions, gates, 600, 1400)
    return {"schema": "d1-stop-turn-fixed-handoff-scores-v1", "case": case["name"],
            "global": {"metrics": metric, "gates": global_gate},
            "stop_segment": stopped, "turn_segment": turned,
            "passed": global_gate["passed"] and stopped["gates"]["passed"] and turned["gates"]["passed"],
            "interpretation": "Original raw reference and numerical gates; stop coverage is not an invented horizon truncation."}
