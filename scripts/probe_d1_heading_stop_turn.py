"""One fixed eight-case qualification of the independent stop/turn composition."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from scripts import probe_d1_heading_g1 as g1
from scripts.d1_native_contact_diagnostics import sample_native_contacts
from scripts.d1_probe_archive import run_archived_g1_episode
from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_NSPM
from scripts.d1_stop_turn_env import StopTurnEnv
from scripts.d1_stop_turn_scoring import score_handoff
from scripts.probe_d1_heading_flat_plane import ARRAY_FIELDS, prefix_check
from scripts.probe_d1_heading_plane_stop_damping import extra_pair_arrays
from scripts.probe_d1_heading_turn_yaw_authority import load_rows
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest

SCHEMA = "d1-heading-plane-stop-turn-composition-probe-v1"


def comparison_arrays(folder, *, stop_baseline=False, handoff_prefix=False):
    with np.load(folder/"states.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    diagnostics = load_rows(folder/("damping_trace.jsonl.gz" if stop_baseline else "plane_diagnostics.jsonl.gz"))
    arrays.update({k: np.asarray([d[k] for d in diagnostics]) for k in ARRAY_FIELDS})
    if not stop_baseline:
        arrays.update({k: np.asarray([d[k] for d in diagnostics]) for k in
                       ("wheel_target_rad_s", "effective_yaw_request_rps", "wheel_error_before_rad_s")})
    arrays.update(extra_pair_arrays(folder, candidate=stop_baseline))
    trace = load_rows(folder/"trace.jsonl.gz")
    digests = []
    for k, original in enumerate(trace):
        row = deepcopy(original)
        if handoff_prefix:
            # Phase is a display label based on the presence of a future pulse.
            row.pop("phase")
            if k == 799:
                for key in ("terminated", "truncated", "terminal_reason"):
                    row.pop(key)
                heading = row["heading_task"]
                for key in list(heading):
                    if key.startswith(("appended_observation_", "terminal_observation_")) or key == "bootstrap_next_observation":
                        heading.pop(key)
        digests.append(hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).digest())
    arrays["physical_trace_sha256"] = np.frombuffer(b"".join(digests), dtype=np.uint8).reshape(len(trace), 32)
    return arrays


def run_case(folder, case, gates):
    summary, trace = run_archived_g1_episode(folder, case, "stop_turn_composition", gates,
        lambda **kwargs: StopTurnEnv(diagnostic_output=folder, **kwargs),
        contact_sampler=sample_native_contacts)
    diagnostics = load_rows(folder/"plane_diagnostics.jsonl.gz")
    authority = load_rows(folder/"turn_authority.jsonl.gz")
    stopped = load_rows(folder/"stop_composition.jsonl.gz")
    native = load_rows(folder/"native_physics_entries.jsonl.gz")
    count = len(trace)
    if len(diagnostics) != count or len(authority) != count or len(stopped) != count or len(native) != 5*count:
        raise RuntimeError("composition evidence lengths disagree")
    previous, latched = None, False
    for k, (d, ar, sr) in enumerate(zip(diagnostics, authority, stopped)):
        a, s = ar["record"], sr["record"]
        raw = g1.command_at_tick(case["command"], k)
        if raw.forward_velocity_mps != 0.:
            latched = False
        elif previous is not None and previous != 0.:
            latched = True
        turn = raw.forward_velocity_mps == 0. and raw.yaw_rate_rps != 0.
        if (s["previous_forward_mps"] != previous or s["active"] != latched or a["active"] != turn
                or sr["tick"] != k or ar["tick"] != k or sr["endpoint_tick"] != k+1):
            raise RuntimeError("executed mode history or record timing changed")
        previous = raw.forward_velocity_mps
        if ar["raw_callback_count_after_prepare"] != k+2:
            raise RuntimeError("raw callback must be consumed once per prepare")
        if a["raw_forward_mps"] != raw.forward_velocity_mps or a["raw_yaw_rate_rps"] != raw.yaw_rate_rps:
            raise RuntimeError("authority did not consume the original raw command")
        if s["forward_command_mps"] != raw.forward_velocity_mps or s["damping_nspm"] != STOP_LEG_DAMPING_NSPM:
            raise RuntimeError("fixed executed-forward damping changed")
        if np.any(np.asarray(s["delta_torque_nm"])[3::4]) or s["joint_power_w"] > 1e-12:
            raise RuntimeError("damping wheel isolation or sampled-power invariant failed")
        if not np.array_equal(a["wheel_torque_nm"], np.asarray(d["requested_torque_nm"])[3::4]):
            raise RuntimeError("authority snapshot does not match final wheel torque")
        formula = a["servo_yaw_rate_rps"]+4.*(a["servo_yaw_rate_rps"]-a["body_yaw_rate_rps"])
        expected = formula if turn else float(np.clip(formula, -.6, .6))
        if abs(a["effective_yaw_request_rps"]-expected) > 1e-12 or a["inner_cap_applied"] != (not turn):
            raise RuntimeError("original yaw formula or actual cap semantics changed")
        if d["inner_yaw_limit_occupied"] != a["inner_cap_occupied"]:
            raise RuntimeError("cap diagnostic differs from executed mechanism")
    semantics = all(e["returned"] and e["contacts"]["max_horizontal_normal"] <= 1e-12
                    and e["contacts"]["max_vertical_normal_error"] <= 1e-12 for e in native)
    if not semantics:
        raise RuntimeError("native plane contact invariant failed")
    summary["plant_semantics_valid"] = semantics
    summary["composition_diagnostics"] = {
        "stop_active_intervals": sum(s["record"]["active"] for s in stopped),
        "authority_active_intervals": sum(a["record"]["active"] for a in authority),
        "overlap_intervals": sum(s["record"]["active"] and a["record"]["active"] for s, a in zip(stopped, authority)),
        "raw_callback_count": authority[-1]["raw_callback_count_after_prepare"],
        "peak_abs_wheel_target_rad_s": max(abs(v) for a in authority for v in a["record"]["wheel_target_rad_s"]),
        "peak_abs_unlimited_wheel_request_nm": max(abs(v) for a in authority for v in a["record"]["wheel_request_nm"]),
        "peak_abs_actual_wheel_torque_nm": max(abs(v) for a in authority for v in a["record"]["wheel_torque_nm"]),
        "peak_abs_stop_delta_nm": max(abs(v) for s in stopped for v in s["record"]["delta_torque_nm"]),
        "maximum_algebraic_stop_power_w": max(s["record"]["joint_power_w"] for s in stopped),
        "maximum_protected_increment_power_w": max(s["protected_increment_joint_power_w"] for s in stopped),
        "target_clipped_wheel_intervals": sum(sum(a["record"]["wheel_target_clipped"]) for a in authority),
        "protected_joint_intervals": sum(sum(d["torque_protected"]) for d in diagnostics),
        "power_interpretation": "Decision-time algebra, not held-interval work or stability proof."}
    return summary, trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("g1-protocol", "plan", "stop-baseline", "turn-baseline", "preflight", "contract", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256(args.g1_protocol) != g1.PROTOCOL_SHA256:
        raise ValueError("original G1 revision2 required")
    original = json.loads(args.g1_protocol.read_text())
    plan = json.loads((args.plan/"cases.json").read_text())
    if plan["original_cases"] != original["cases"] or plan["original_numeric_thresholds"] != original["proposed_gates"]:
        raise ValueError("original commands or gates changed")
    preflight = json.loads(args.preflight.read_text())
    if not preflight["passed"] or not preflight["root_execution_authorized"]:
        raise ValueError("root-reviewed preflight required")
    inputs = {str((ROOT/p).resolve()): h for p, h in source_hashes().items()}
    for path, digest in preflight["input_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("preflight input changed: "+path)
        inputs[path] = digest
    for p in (args.g1_protocol, args.plan/"cases.json", args.preflight, args.contract):
        inputs[str(p.resolve())] = sha256(p)
    frozen = json.loads((ROOT/"results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(sha256(ROOT/p) != h for p, h in frozen.items()):
        raise ValueError("frozen77 changed")
    cases = plan["original_cases"]+[h["case"] for h in plan["handoff_cases"]]
    if len(cases) != 8 or sum(c["max_transitions"] for c in cases) != 8400:
        raise ValueError("fixed composition budget changed")
    handoffs = {h["case"]["name"]: h for h in plan["handoff_cases"]}
    baseline_folders, baselines, baseline_summaries = {}, {}, {}
    for i, case in enumerate(plan["original_cases"]):
        stop_source = i not in (2, 3)
        source = args.stop_baseline if stop_source else args.turn_baseline
        folder = source/case["name"]
        manifest = json.loads((folder/"complete_manifest.json").read_text())
        for p, value in manifest.items():
            if sha256(folder/p) != value["sha256"]:
                raise ValueError("baseline evidence changed: "+str(folder/p))
        for p in folder.iterdir():
            if p.is_file():
                inputs[str(p.resolve())] = sha256(p)
        baseline_folders[case["name"]] = (folder, stop_source)
        baselines[case["name"]] = comparison_arrays(folder, stop_baseline=stop_source)
        baseline_summaries[case["name"]] = json.loads((folder/"summary.json").read_text())
    for source in (args.stop_baseline, args.turn_baseline):
        for name in ("protocol.json", "summary.json"):
            inputs[str((source/name).resolve())] = sha256(source/name)
    for name, handoff in handoffs.items():
        folder, stop_source = baseline_folders[handoff["prefix"]["baseline_case"]]
        baselines[name] = comparison_arrays(folder, stop_baseline=stop_source, handoff_prefix=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"protocol.json", {"schema": SCHEMA, "cases": cases,
        "handoff_specs": plan["handoff_cases"], "original_gates": original["proposed_gates"],
        "input_sha256": inputs, "maximum_new_control_transitions": 8400,
        "maximum_new_native_substeps": 42000, "unique_reused_baseline_control_transitions": 5600,
        "new_training_steps": 0, "default_changed": False, "zero_only": True,
        "pair_target_evidence": "Turn baseline has explicit target/effective/error arrays, compared bitwise. Old stop logs lack those fields; equal states/raw/servo/PI/requests and unchanged inactive formula support equivalence, not a direct archived target-field comparison."})
    summaries, pairs, handoff_scores = [], [], []
    try:
        for case in cases:
            folder = args.output/case["name"]
            summary, rows = run_case(folder, case, original["proposed_gates"])
            handoff = case["name"] in handoffs
            arrays = comparison_arrays(folder, handoff_prefix=handoff)
            count = 800 if handoff else case["max_transitions"]
            pair = prefix_check(case["name"], baselines[case["name"]], arrays, count,
                                next_observation_equal=not handoff)
            if handoff:
                score = score_handoff(case, rows, arrays["truth_positions_world_m"], original["cases"], original["proposed_gates"])
                write_json(folder/"handoff_scores.json", score)
                handoff_scores.append(score)
                summary["handoff_passed"] = score["passed"]
            else:
                base = baseline_summaries[case["name"]]
                pair["checks"]["all_original_summary_fields_except_model"] = all(summary.get(k) == v for k, v in base.items() if k != "model")
                pair["passed"] = all(pair["checks"].values())
            summaries.append(summary)
            pairs.append(pair)
            write_json(folder/"candidate_summary.json", summary)
            write_json(folder/"complete_manifest.json", {str(p.relative_to(folder)):
                {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(folder.rglob("*")) if p.is_file()})
            write_json(args.output/f"pair_{case['name']}.json", pair)
            if not pair["passed"]:
                raise RuntimeError("composition whole/prefix invariance failed: "+case["name"])
            print(json.dumps({"case": case["name"], "gates": summary["gates"],
                "handoff_passed": summary.get("handoff_passed"), "diagnostics": summary["composition_diagnostics"]}), flush=True)
        if any(sha256(Path(p)) != h for p, h in inputs.items()) or any(sha256(ROOT/p) != h for p, h in frozen.items()):
            raise RuntimeError("sources or baseline changed during evaluation")
        passed = all(s["gates"]["passed"] for s in summaries) and all(h["passed"] for h in handoff_scores)
        write_json(args.output/"summary.json", {"schema": SCHEMA, "episodes": summaries, "pairs": pairs,
            "handoff_scores": handoff_scores, "comparison_valid": all(p["passed"] for p in pairs),
            "candidate_passed_fixed_domain": passed,
            "actual_new_control_transitions": sum(s["actual_transitions"] for s in summaries),
            "actual_new_native_substeps": sum(s["actual_observed_physics_substeps"] for s in summaries),
            "frozen77_unchanged": True, "full_driving_goal_complete": False, "default_changed": False})
        write_manifest(args.output)
    except BaseException as error:
        receipts = [json.loads(p.read_text()) for p in args.output.glob("*/plane_execution_receipt.json")]
        write_json(args.output/"batch_failure.json", {"type": type(error).__name__, "message": str(error),
            "actual_new_native_substeps_from_clock": sum(r["actual_physics_substeps_from_clock"] for r in receipts),
            "actual_new_completed_control_intervals": sum(r["actual_completed_control_intervals"] for r in receipts),
            "partial_interval_substeps": sum(r["partial_interval_physics_substeps"] for r in receipts),
            "no_retry_or_padding": True, "remaining_cases_stopped": True})
        write_json(args.output/"failure_manifest.json", {str(p.relative_to(args.output)):
            {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(args.output.rglob("*")) if p.is_file()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
