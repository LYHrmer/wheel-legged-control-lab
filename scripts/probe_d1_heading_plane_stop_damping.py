"""One fixed post-stop damping candidate against retained native-plane baselines."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from scripts import probe_d1_heading_g1 as g1
from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv
from scripts.d1_native_contact_diagnostics import sample_native_contacts
from scripts.d1_probe_archive import run_archived_g1_episode
from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_NSPM
from scripts.d1_turn_contact_diagnostics import sample_turn_contacts
from scripts.probe_d1_heading_flat_plane import ARRAY_FIELDS, prefix_check, read_completed_arrays
from scripts.probe_d1_heading_stop_damping import StopDampingEnv
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)

SCHEMA = "d1-heading-plane-stop-leg-damping-probe-v1"


class PlaneStopDampingEnv(StopDampingEnv, D1FlatPlaneHeadingEnv):
    """Compose the reviewed stop instrumentation with the reviewed plane seam.

    StopDampingEnv's cooperative super chain reaches D1FlatPlaneHeadingEnv before
    D1HeadingTrackingEnv. Plane construction finishes first, then the existing
    stop adapter installs its sole controller change before the first reset.
    Neither base execution chain is copied or run twice.
    """

    task_schema = "d1-heading-native-plane-stop-leg-damping-zero-task-v1"

    def __init__(self, **kwargs):
        super().__init__(damping_enabled=True, **kwargs)
        self._details = ExitStack()
        output = kwargs.get("diagnostic_output")
        self._detail_stream = None if output is None else self._details.enter_context(
            gzip.open(output / "damping_contact_power.jsonl.gz", "xt")  # noqa: SIM115
        )
        self._candidate_closed = False

    def _build_heading_task_config(self):
        config = super()._build_heading_task_config()
        config["task_schema"] = self.task_schema
        return config

    def step(self, action):
        self._require_zero_action(action)
        state = self.decision.context.state
        before = self.heading_decision
        try:
            result = super().step(action)
            record = self._controller.controller.last_damping
            if not (record.forward_command_mps == before.servo_command.forward_velocity_mps
                    == before.user_command.forward_velocity_mps):
                raise RuntimeError("damper did not consume the raw forward command")
            base_safe = np.clip(record.base_requested_torque_nm, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
            blocked = (((state.joint_position >= JOINT_POSITION_HIGH) & (base_safe > 0.))
                | ((state.joint_position <= JOINT_POSITION_LOW) & (base_safe < 0.))
                | ((np.abs(state.joint_velocity) >= JOINT_VELOCITY_LIMIT) & (base_safe*state.joint_velocity > 0.)))
            base_safe[blocked] = 0.
            physical_before = (self.plant.data.qpos.copy(), self.plant.data.qvel.copy(), self.plant.data.qacc_warmstart.copy())
            endpoint = sample_turn_contacts(self.plant)
            physical_after = (self.plant.data.qpos, self.plant.data.qvel, self.plant.data.qacc_warmstart)
            if any(not np.array_equal(a, b) for a, b in zip(physical_before, physical_after)):
                raise RuntimeError("endpoint contact observer changed integrator state")
            row = {"tick": before.tick, "endpoint_tick": before.tick+1,
                "joint_velocity_before_rad_s": state.joint_velocity.tolist(),
                "base_safe_torque_nm_same_state": base_safe.tolist(),
                "protected_increment_joint_power_w": float((record.safe_torque_nm-base_safe) @ state.joint_velocity),
                "unprotected_algebraic_joint_power_w": record.joint_power_w,
                "power_phase": "decision-time velocity; not an integrated physical-interval work measurement",
                "endpoint_contacts": endpoint}
            if self._detail_stream is not None:
                self._detail_stream.write(json.dumps(row, allow_nan=False)+"\n")
            if result[2] or result[3]:
                self._details.close()
            return result
        except BaseException as error:
            self._error = {"type": type(error).__name__, "message": str(error)}
            raise

    def close(self):
        if self._candidate_closed:
            return
        self._candidate_closed = True
        try:
            self._details.close()
        finally:
            super().close()


def extra_pair_arrays(folder, *, candidate):
    name = "damping_trace.jsonl.gz" if candidate else "plane_diagnostics.jsonl.gz"
    with gzip.open(folder / name, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    with gzip.open(folder / "native_physics_entries.jsonl.gz", "rt") as stream:
        native = [json.loads(line) for line in stream]
    fields = ("forward_velocity_mps", "yaw_rate_rps", "clearance_m")
    arrays = {k: np.asarray([[d[k][f] for f in fields] for d in rows])
              for k in ("raw_user_command", "servo_command")}
    arrays["unlimited_torque_nm"] = np.asarray([
        d["damping"]["total_requested_torque_nm"] if candidate else d["unlimited_torque_nm"] for d in rows])
    arrays["native_ctrl_nm"] = np.asarray([e["ctrl_nm"] for e in native]).reshape(len(rows), 5, 16)
    arrays["native_wrench"] = np.asarray([e["wrench_world"] for e in native]).reshape(len(rows), 5, 6)
    # Canonical-entry digests preserve all native contact fields as well, while
    # retaining a fixed dtype/shape irrespective of later trajectory divergence.
    digest = b"".join(hashlib.sha256(json.dumps(e, sort_keys=True, separators=(",", ":")).encode()).digest() for e in native)
    arrays["native_entry_sha256"] = np.frombuffer(digest, dtype=np.uint8).reshape(len(rows), 5, 32)
    return arrays


def run_candidate(output, case, gates):
    summary, rows = run_archived_g1_episode(output, case, "plane_stop_damping", gates,
        lambda **kwargs: PlaneStopDampingEnv(diagnostic_output=output, **kwargs),
        contact_sampler=sample_native_contacts)
    with gzip.open(output / "damping_trace.jsonl.gz", "rt") as stream:
        diagnostics = [json.loads(line) for line in stream]
    with gzip.open(output / "native_physics_entries.jsonl.gz", "rt") as stream:
        native = [json.loads(line) for line in stream]
    with gzip.open(output / "damping_contact_power.jsonl.gz", "rt") as stream:
        powers = [json.loads(line) for line in stream]
    if len(diagnostics) != len(rows) or len(powers) != len(rows) or len(native) != 5*len(rows):
        raise RuntimeError("candidate evidence lengths disagree")
    stop = case["command"]["stop_tick"]
    for k, d in enumerate(diagnostics):
        record = d["damping"]
        if record["active"] != (stop is not None and k >= stop):
            raise RuntimeError("damper activation differs from the executed raw stop transition")
        delta = np.asarray(record["delta_torque_nm"])
        if np.any(delta[[3, 7, 11, 15]]) or record["joint_power_w"] > 1e-12:
            raise RuntimeError("damper wheel isolation or algebraic power invariant failed")
        if record["damping_nspm"] != STOP_LEG_DAMPING_NSPM:
            raise RuntimeError("damper gain changed")
        if powers[k]["tick"] != k or powers[k]["endpoint_tick"] != k+1:
            raise RuntimeError("power evidence tick differs from executed interval")
        if powers[k]["unprotected_algebraic_joint_power_w"] != record["joint_power_w"]:
            raise RuntimeError("power evidence differs from controller record")
    semantics = all(e["returned"] and e["contacts"]["max_horizontal_normal"] <= 1e-12
                    and e["contacts"]["max_vertical_normal_error"] <= 1e-12 for e in native)
    summary["plant_semantics_valid"] = semantics
    summary["damping_diagnostics"] = {
        "active_intervals": sum(d["damping"]["active"] for d in diagnostics),
        "fixed_gain_nspm": STOP_LEG_DAMPING_NSPM,
        "peak_abs_delta_nm": max(abs(v) for d in diagnostics for v in d["damping"]["delta_torque_nm"]),
        "max_algebraic_joint_power_w": max(d["damping"]["joint_power_w"] for d in diagnostics),
        "min_algebraic_joint_power_w": min(d["damping"]["joint_power_w"] for d in diagnostics),
        "max_protected_increment_joint_power_w": max(p["protected_increment_joint_power_w"] for p in powers),
        "min_protected_increment_joint_power_w": min(p["protected_increment_joint_power_w"] for p in powers),
        "protected_joint_intervals": sum(sum(d["torque_protected"]) for d in diagnostics),
        "power_limit": "pre-protection sampled algebraic power, not closed-loop interval dissipation"}
    write_json(output / "candidate_summary.json", summary)
    write_json(output / "complete_manifest.json", {
        str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*")) if p.is_file()})
    if not semantics:
        raise RuntimeError("native plane contact invariant failed")
    with np.load(output / "states.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays.update({k: np.asarray([d[k] for d in diagnostics]) for k in ARRAY_FIELDS})
    arrays.update(extra_pair_arrays(output, candidate=True))
    return summary, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-protocol", type=Path, required=True)
    parser.add_argument("--plane-baseline", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    parser.add_argument("--gain-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256(args.g1_protocol) != g1.PROTOCOL_SHA256:
        raise ValueError("original G1 revision2 required")
    protocol = json.loads(args.g1_protocol.read_text())
    gain = json.loads(args.gain_provenance.read_text())["hypothetical_damper_gain_provenance"]
    if gain["proposed_extra_per_leg_damping_Nspm"] != STOP_LEG_DAMPING_NSPM:
        raise ValueError("gain differs from the original fixed nominal-mode derivation")
    cases = [protocol["cases"][i] for i in (0, 1, 4, 5)]
    frozen = json.loads((ROOT / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(sha256(ROOT/p) != h for p, h in frozen.items()):
        raise ValueError("frozen77 changed")
    baseline_summary = json.loads((args.plane_baseline / "summary.json").read_text())
    if not baseline_summary["plant_semantics_valid"] or not baseline_summary["execution_evidence_valid"]:
        raise ValueError("valid plane baseline evidence required")
    baseline_protocol = json.loads((args.plane_baseline / "protocol.json").read_text())
    inputs = {str((ROOT / p).resolve()): h for p, h in source_hashes().items()}
    scripts = ("d1_stop_leg_damping.py", "d1_probe_archive.py", "d1_flat_plane_env.py",
        "d1_native_contact_diagnostics.py", "d1_turn_contact_diagnostics.py", "d1_release_velocity_governor.py",
        "probe_d1_heading_g1.py", "probe_d1_heading_release.py", "probe_d1_heading_flat_plane.py",
        "probe_d1_heading_stop_damping.py", "probe_d1_heading_plane_stop_damping.py")
    for name in ("d1_flat_plane_env.py", "d1_probe_archive.py", "d1_native_contact_diagnostics.py", "probe_d1_heading_g1.py"):
        p = ROOT / "scripts" / name
        if sha256(p) != baseline_protocol["input_sha256"][str(p)]:
            raise ValueError("plane or physics instrumentation changed since baseline")
    for p in [args.g1_protocol, args.contract, args.diagnosis_report, args.gain_provenance,
              args.diagnosis_report.parent / "audit_stop.py",
              args.diagnosis_report.parent / "next_contract.md",
              *(ROOT / "scripts" / name for name in scripts),
              args.plane_baseline / "protocol.json", args.plane_baseline / "summary.json"]:
        inputs[str(p.resolve())] = sha256(p)
    baselines = {}
    for case in cases:
        folder = args.plane_baseline / case["name"]
        manifest = json.loads((folder / "complete_manifest.json").read_text())
        if any(sha256(folder/p) != v["sha256"] for p, v in manifest.items()):
            raise ValueError("baseline evidence changed")
        for p in folder.iterdir():
            if p.is_file():
                inputs[str(p.resolve())] = sha256(p)
        baselines[case["name"]] = read_completed_arrays(folder)
        baselines[case["name"]].update(extra_pair_arrays(folder, candidate=False))
        if len(baselines[case["name"]]["qpos"]) != case["max_transitions"]+1:
            raise ValueError("full baseline required")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"schema": SCHEMA, "cases": cases,
        "input_sha256": inputs, "original_gates": protocol["proposed_gates"],
        "fixed_damping_nspm": STOP_LEG_DAMPING_NSPM, "raw_stop_tick": 400,
        "reused_baseline_control_transitions": 4000, "reused_baseline_physics_substeps": 20000,
        "maximum_new_control_transitions": 4000, "maximum_new_physics_substeps": 20000,
        "baseline_source": str(args.plane_baseline.resolve()), "new_training_steps": 0,
        "release_shaping": False, "inner_yaw_limit_rps": .6, "zero_only": True,
        "default_changed": False, "gain_sweep": False})
    summaries, pairs = [], []
    try:
        for case in cases:
            summary, arrays = run_candidate(args.output / case["name"], case, protocol["proposed_gates"])
            summaries.append(summary)
            stop = case["command"]["stop_tick"]
            pair = prefix_check(case["name"], baselines[case["name"]], arrays,
                case["max_transitions"] if stop is None else stop, next_observation_equal=True)
            pairs.append(pair)
            write_json(args.output / f"pair_{case['name']}.json", pair)
            if not pair["passed"]:
                raise RuntimeError("inactive prefix or no-stop full trajectory changed")
            print(json.dumps({"case": case["name"], "gates": summary["gates"],
                "velocity_rmse_mps": summary["velocity_rmse_mps"],
                "late_speed_mps": summary.get("late_stop_max_abs_body_vx_mps"),
                "late_path_m": summary.get("late_stop_cumulative_planar_path_m"),
                "damping": summary["damping_diagnostics"]}), flush=True)
        if any(sha256(Path(p)) != h for p, h in inputs.items()) or any(sha256(ROOT/p) != h for p, h in frozen.items()):
            raise RuntimeError("sources changed during comparison")
        write_json(args.output / "summary.json", {"schema": SCHEMA, "episodes": summaries, "pairs": pairs,
            "comparison_valid": all(p["passed"] for p in pairs),
            "candidate_passed_both_stop_cases": all(s["gates"]["passed"] for s in summaries[:2]),
            "actual_new_control_transitions": sum(s["actual_transitions"] for s in summaries),
            "actual_new_physics_substeps": sum(s["actual_observed_physics_substeps"] for s in summaries),
            "reused_baseline_control_transitions": 4000, "reused_baseline_physics_substeps": 20000,
            "frozen77_unchanged": True, "default_changed": False, "full_driving_goal_complete": False})
        write_manifest(args.output)
    except BaseException as error:
        receipts = [json.loads(p.read_text()) for p in args.output.glob("*/damping_execution_receipt.json")]
        write_json(args.output / "batch_failure.json", {"type": type(error).__name__, "message": str(error),
            "actual_new_physics_substeps_from_clock": sum(r["actual_physics_substeps_from_clock"] for r in receipts),
            "actual_new_completed_control_intervals": sum(r["actual_completed_control_intervals"] for r in receipts),
            "partial_interval_substeps": sum(r["partial_interval_physics_substeps"] for r in receipts),
            "reused_baseline_control_transitions": 4000, "reused_baseline_physics_substeps": 20000,
            "no_retry_or_padding": True, "remaining_cases_stopped": True})
        write_json(args.output / "failure_manifest.json", {
            str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(args.output.rglob("*")) if p.is_file()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
