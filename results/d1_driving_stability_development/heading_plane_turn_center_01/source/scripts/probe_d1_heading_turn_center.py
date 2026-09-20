"""Fixed wheel-center turn compensation on the retained native-plane cases."""
from __future__ import annotations

import argparse
import gzip
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts import probe_d1_heading_g1 as g1
from scripts.d1_native_contact_diagnostics import sample_native_contacts
from scripts.d1_probe_archive import run_archived_g1_episode
from scripts.d1_turn_center_compensation import (
    TURN_CENTER_COMPENSATION_SCHEMA,
    TurnCenterCompensationController,
)
from scripts.probe_d1_heading_flat_plane import (
    PlaneDiagnosticEnv,
    prefix_check,
    read_completed_arrays,
)
from scripts.probe_d1_heading_plane_stop_damping import extra_pair_arrays
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest
from wheel_legged_control.d1.control_loop import D1MotionCommand, D1WheelLegControllerAdapter

SCHEMA = "d1-heading-plane-turn-center-probe-v1"


class TurnCenterEnv(PlaneDiagnosticEnv):
    """Bind the unchanged raw callback before the original heading preparation."""

    task_schema = "d1-heading-plane-turn-center-zero-task-v1"

    def __init__(self, *, command_source, diagnostic_output, **kwargs):
        if not callable(command_source):
            raise TypeError("turn candidate requires an external raw command callback")
        self.raw_callback_count = 0

        def tracked(time_s):
            raw = command_source(time_s)
            if not isinstance(raw, D1MotionCommand):
                raise TypeError("raw callback must return D1MotionCommand")
            self._controller.controller.bind_raw_command(
                forward_velocity_mps=raw.forward_velocity_mps,
                yaw_rate_rps=raw.yaw_rate_rps, control_time_s=time_s)
            self.raw_callback_count += 1
            return raw

        super().__init__(command_source=tracked, diagnostic_output=diagnostic_output, **kwargs)
        self._controller = D1WheelLegControllerAdapter(TurnCenterCompensationController(
            enabled=True, **asdict(self.wheel_leg_control)))
        self._extra_files = ExitStack()
        self._compensation_stream = self._extra_files.enter_context(
            gzip.open(diagnostic_output / "turn_compensation.jsonl.gz", "xt")  # noqa: SIM115
        )
        self._candidate_closed = False

    def _build_heading_task_config(self):
        config = super()._build_heading_task_config()
        config["task_schema"] = self.task_schema
        config["turn_center_compensation"] = {
            "schema": TURN_CENTER_COMPENSATION_SCHEMA, "coefficient": 1., "wheel_radius_m": .087,
            "gate": "raw forward exactly zero and raw yaw nonzero; no latch",
            "projection": "horizontal heading; wheel-center leg Jacobian only",
            "original_inner_yaw_cap_rps": .6, "final_wheel_speed_clip_rad_s": 30.,
            "zero_only": True, "external_raw_callback_required": True,
            "stop_leg_damping": False}
        return config

    def reset(self, **kwargs):
        self.raw_callback_count = 0
        return super().reset(**kwargs)

    def step(self, action):
        self._require_zero_action(action)
        before = self.heading_decision
        try:
            result = super().step(action)
            record = self._controller.controller.last_compensation
            if (record.raw_forward_mps != before.user_command.forward_velocity_mps
                    or record.raw_yaw_rate_rps != before.user_command.yaw_rate_rps
                    or abs(record.control_time_s-before.control_time_s) > 1e-12):
                raise RuntimeError("turn record does not describe the executed raw binding")
            payload = asdict(record)
            payload = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in payload.items()}
            self._compensation_stream.write(json.dumps({"tick": before.tick,
                "endpoint_tick": before.tick+1, "record": payload,
                "raw_callback_count_after_prepare": self.raw_callback_count}, allow_nan=False)+"\n")
            if result[2] or result[3]:
                self._extra_files.close()
            return result
        except BaseException as error:
            self._error = {"type": type(error).__name__, "message": str(error)}
            raise

    def close(self):
        if self._candidate_closed:
            return
        self._candidate_closed = True
        try:
            self._extra_files.close()
        finally:
            super().close()


def load_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def run_candidate(folder, case, gates):
    summary, trace = run_archived_g1_episode(folder, case, "turn_center_coefficient1", gates,
        lambda **kwargs: TurnCenterEnv(diagnostic_output=folder, **kwargs),
        contact_sampler=sample_native_contacts)
    rows = load_rows(folder / "turn_compensation.jsonl.gz")
    native = load_rows(folder / "native_physics_entries.jsonl.gz")
    diag = load_rows(folder / "plane_diagnostics.jsonl.gz")
    if len(rows) != len(trace) or len(diag) != len(trace) or len(native) != 5*len(trace):
        raise RuntimeError("turn evidence lengths disagree")
    pulse = case["command"]["yaw_pulse"]
    for k, row in enumerate(rows):
        active = pulse is not None and pulse["start_tick"] <= k < pulse["end_tick_exclusive"]
        if row["tick"] != k or row["record"]["active"] != active:
            raise RuntimeError("turn gate differs from the executed raw command")
        record = row["record"]
        increment = np.asarray(record["correction_unclipped_rad_s"])
        expected = np.asarray(record["leg_center_forward_mps"])/.087 if active else np.zeros(4)
        if not np.allclose(increment, expected, rtol=0, atol=1e-12):
            raise RuntimeError("turn correction differs from fixed +u/r")
        if not active and np.any(increment):
            raise RuntimeError("inactive turn correction is not exactly zero")
        corrected = np.asarray(record["base_unclipped_rad_s"])+increment
        if not np.allclose(record["corrected_unclipped_rad_s"], corrected, rtol=0, atol=1e-12):
            raise RuntimeError("turn target addition changed")
        if not np.allclose(record["corrected_target_rad_s"], np.clip(corrected, -30., 30.), rtol=0, atol=1e-12):
            raise RuntimeError("turn final target clip changed")
        # G1 prepares one extra terminal decision too; it never executes it.
        if row["raw_callback_count_after_prepare"] != k+2:
            raise RuntimeError("raw callback was skipped or consumed more than once")
    semantics = all(e["returned"] and e["contacts"]["max_horizontal_normal"] <= 1e-12
                    and e["contacts"]["max_vertical_normal_error"] <= 1e-12 for e in native)
    summary["plant_semantics_valid"] = semantics
    summary["turn_compensation_diagnostics"] = {
        "active_intervals": sum(r["record"]["active"] for r in rows),
        "peak_abs_correction_rad_s": max(abs(v) for r in rows for v in r["record"]["correction_unclipped_rad_s"]),
        "peak_abs_target_rad_s": max(abs(v) for r in rows for v in r["record"]["corrected_target_rad_s"]),
        "peak_abs_target_equivalent_yaw_rps": max(abs(r["record"]["target_equivalent_yaw_rps"]) for r in rows),
        "target_clipped_wheel_intervals": sum(sum(r["record"]["target_clipped"]) for r in rows),
        "torque_protected_joint_intervals": sum(sum(r["torque_protected"]) for r in diag),
        "raw_callback_count": rows[-1]["raw_callback_count_after_prepare"]}
    write_json(folder / "candidate_summary.json", summary)
    write_json(folder / "complete_manifest.json", {str(p.relative_to(folder)):
        {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(folder.rglob("*")) if p.is_file()})
    if not semantics:
        raise RuntimeError("native plane contact invariant failed")
    arrays = read_completed_arrays(folder)
    arrays.update(extra_pair_arrays(folder, candidate=False))
    return summary, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-protocol", type=Path, required=True)
    parser.add_argument("--plane-baseline", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256(args.g1_protocol) != g1.PROTOCOL_SHA256:
        raise ValueError("original raw G1 protocol revision2 required")
    protocol = json.loads(args.g1_protocol.read_text())
    cases = [protocol["cases"][i] for i in (2,3,0,1)]
    frozen = json.loads((ROOT / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(sha256(ROOT/p) != h for p,h in frozen.items()):
        raise ValueError("frozen77 changed")
    base_protocol = json.loads((args.plane_baseline / "protocol.json").read_text())
    base_summary = json.loads((args.plane_baseline / "summary.json").read_text())
    if not base_summary["plant_semantics_valid"] or not base_summary["execution_evidence_valid"]:
        raise ValueError("validated plane-zero baselines required")
    inputs = {str((ROOT/p).resolve()): h for p,h in source_hashes().items()}
    scripts = ("d1_turn_center_compensation.py", "probe_d1_heading_turn_center.py", "d1_flat_plane_env.py",
        "d1_probe_archive.py", "d1_native_contact_diagnostics.py", "d1_turn_contact_diagnostics.py",
        "probe_d1_heading_flat_plane.py", "probe_d1_heading_plane_stop_damping.py", "probe_d1_heading_g1.py",
        "probe_d1_heading_stop_damping.py", "d1_stop_leg_damping.py", "probe_d1_heading_release.py",
        "d1_release_velocity_governor.py")
    for name in ("d1_flat_plane_env.py", "d1_probe_archive.py", "d1_native_contact_diagnostics.py", "probe_d1_heading_g1.py"):
        path = ROOT / "scripts" / name
        if sha256(path) != base_protocol["input_sha256"][str(path)]:
            raise ValueError("baseline plant or physics instrumentation changed")
    for path in (args.g1_protocol, args.contract, args.diagnosis_report,
                 args.diagnosis_report.parent/"audit_turn.py", args.diagnosis_report.parent/"next_contract.md",
                 args.plane_baseline/"protocol.json", args.plane_baseline/"summary.json",
                 *(ROOT/"scripts"/name for name in scripts)):
        inputs[str(path.resolve())] = sha256(path)
    baselines = {}
    for case in cases:
        folder = args.plane_baseline / case["name"]
        manifest = json.loads((folder / "complete_manifest.json").read_text())
        if any(sha256(folder/p) != v["sha256"] for p,v in manifest.items()):
            raise ValueError("baseline evidence changed")
        for p in folder.iterdir():
            if p.is_file():
                inputs[str(p.resolve())] = sha256(p)
        baselines[case["name"]] = read_completed_arrays(folder)
        baselines[case["name"]].update(extra_pair_arrays(folder, candidate=False))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"protocol.json", {"schema": SCHEMA, "cases": cases,
        "input_sha256": inputs, "original_gates": protocol["proposed_gates"],
        "coefficient": 1., "original_inner_yaw_cap_rps": .6, "wheel_speed_clip_rad_s": 30.,
        "maximum_new_control_transitions": 3200, "maximum_new_physics_substeps": 16000,
        "reused_baseline_control_transitions": 3200, "reused_baseline_physics_substeps": 16000,
        "new_training_steps": 0, "stop_damping": False, "release_shaping": False,
        "default_changed": False, "baseline_source": str(args.plane_baseline.resolve())})
    summaries, pairs = [], []
    try:
        for case in cases:
            summary, arrays = run_candidate(args.output/case["name"], case, protocol["proposed_gates"])
            summaries.append(summary)
            pulse = case["command"]["yaw_pulse"]
            count = pulse["start_tick"] if pulse else case["max_transitions"]
            pair = prefix_check(case["name"], baselines[case["name"]], arrays, count,
                                next_observation_equal=pulse is None)
            if pulse is None:
                base = json.loads((args.plane_baseline/case["name"]/"summary.json").read_text())
                pair["checks"]["original_failed_gates_unchanged"] = summary["gates"] == base["gates"]
                pair["checks"]["all_original_summary_fields_unchanged_except_model"] = all(
                    summary.get(k) == v for k,v in base.items() if k != "model")
                pair["passed"] = all(pair["checks"].values())
            pairs.append(pair)
            write_json(args.output/f"pair_{case['name']}.json", pair)
            if not pair["passed"]:
                raise RuntimeError("turn inactive prefix or whole no-op trajectory changed")
            print(json.dumps({"case": case["name"], "gates": summary["gates"],
                "heading_peak_rad": summary["heading_peak_rad"],
                "diagnostics": summary["turn_compensation_diagnostics"]}), flush=True)
        if any(sha256(Path(p)) != h for p,h in inputs.items()) or any(sha256(ROOT/p) != h for p,h in frozen.items()):
            raise RuntimeError("sources or baseline changed during evaluation")
        write_json(args.output/"summary.json", {"schema": SCHEMA, "episodes": summaries, "pairs": pairs,
            "comparison_valid": all(p["passed"] for p in pairs),
            "candidate_passed_both_turn_cases": all(s["gates"]["passed"] for s in summaries[:2]),
            "no_op_stop_failures_preserved": True,
            "actual_new_control_transitions": sum(s["actual_transitions"] for s in summaries),
            "actual_new_physics_substeps": sum(s["actual_observed_physics_substeps"] for s in summaries),
            "frozen77_unchanged": True, "default_changed": False, "full_driving_goal_complete": False})
        write_manifest(args.output)
    except BaseException as error:
        receipts = [json.loads(p.read_text()) for p in args.output.glob("*/plane_execution_receipt.json")]
        write_json(args.output/"batch_failure.json", {"type": type(error).__name__, "message": str(error),
            "actual_new_physics_substeps_from_clock": sum(r["actual_physics_substeps_from_clock"] for r in receipts),
            "actual_new_completed_control_intervals": sum(r["actual_completed_control_intervals"] for r in receipts),
            "partial_interval_substeps": sum(r["partial_interval_physics_substeps"] for r in receipts),
            "no_retry_or_padding": True, "remaining_cases_stopped": True})
        write_json(args.output/"failure_manifest.json", {str(p.relative_to(args.output)):
            {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(args.output.rglob("*")) if p.is_file()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
