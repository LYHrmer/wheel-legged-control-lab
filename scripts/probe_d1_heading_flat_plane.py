"""Six fixed zero-residual G1 command profiles on a separately identified plane."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts import probe_d1_heading_g1 as g1
from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv
from scripts.d1_native_contact_diagnostics import sample_native_contacts
from scripts.d1_probe_archive import ProbeStateArchive, run_archived_g1_episode
from scripts.d1_turn_contact_diagnostics import sample_turn_contacts
from scripts.probe_d1_heading_release import bitwise_equal
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest

SCHEMA = "d1-heading-flat-plane-zero-probe-v1"
ARRAY_FIELDS = ("requested_torque_nm", "applied_torque_nm", "wheel_integral_before_nm",
                "wheel_integral_after_nm")


class PlaneDiagnosticEnv(D1FlatPlaneHeadingEnv):
    def __init__(self, *, diagnostic_output, **kwargs):
        super().__init__(**kwargs)
        self.output = diagnostic_output
        self._files = ExitStack()
        self._stream = self._files.enter_context(
            gzip.open(self.output / "plane_diagnostics.jsonl.gz", "xt")  # noqa: SIM115
        )
        self._states = ProbeStateArchive(self.plant, self.output)
        self._error, self._count, self._closed = None, 0, False

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._states.capture("reset")
        write_json(self.output / "initial_controller_memory.json", {
            "wheel_integral_nm": self._controller.controller.control_memory.wheel_integral_nm.tolist()
        })
        return result

    def step(self, action):
        self._require_zero_action(action)
        before = self.heading_decision
        state = self.decision.context.state
        try:
            result = super().step(action)
            control = self._controller.controller.last_result
            qpos, qvel = self.plant.data.qpos.copy(), self.plant.data.qvel.copy()
            endpoint = sample_turn_contacts(self.plant)
            if not bitwise_equal(qpos, self.plant.data.qpos) or not bitwise_equal(qvel, self.plant.data.qvel):
                raise RuntimeError("endpoint observer changed integrator state")
            jx = np.einsum("i,kij->kj", state.base_rotation[:, 0], state.foot_jacobian[:, :, :3])
            leg_velocity = state.joint_velocity.reshape(4, 4)[:, :3]
            row = {"tick": before.tick, "endpoint_tick": before.tick+1,
                "raw_user_command": asdict(before.user_command), "servo_command": asdict(before.servo_command),
                "body_forward_before_mps": float(state.base_linear_velocity_body[0]),
                "body_yaw_rate_before_rps": float(state.base_angular_velocity_body[2]),
                "wheel_rolling_before_mps": float(np.mean(state.joint_velocity[[3, 7, 11, 15]])*.087),
                "leg_forward_velocity_before_mps": np.einsum("ki,ki->k", jx, leg_velocity).tolist(),
                "wheel_target_rad_s": control.wheel_speed_target_rad_s.tolist(),
                "wheel_error_before_rad_s": (control.wheel_speed_target_rad_s-state.joint_velocity[[3, 7, 11, 15]]).tolist(),
                "effective_yaw_request_rps": control.effective_yaw_request_rps,
                "inner_yaw_limit_occupied": abs(control.effective_yaw_request_rps) >= .6-1e-12,
                "unlimited_torque_nm": control.requested_torque_nm.tolist(),
                "requested_torque_nm": self.last_transition.requested_torque_nm.tolist(),
                "applied_torque_nm": [v.applied_nm.tolist() for v in self.plant.last_control_interval_actuator_traces],
                "wheel_integral_before_nm": control.memory_before.wheel_integral_nm.tolist(),
                "wheel_integral_after_nm": control.memory_after.wheel_integral_nm.tolist(),
                "torque_protected": control.torque_limited.tolist(), "endpoint_contacts": endpoint}
            self._stream.write(json.dumps(row, allow_nan=False)+"\n")
            self._count += 1
            if result[2] or result[3]:
                self._files.close()
            return result
        except BaseException as error:
            self._error = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            self._states.capture("failed_step" if self._error else "completed_step")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._files.close()
        self._states.close()
        substeps = round(float(self.plant.data.time)/float(self.plant.model.opt.timestep))
        write_json(self.output / "plane_execution_receipt.json", {
            "actual_physics_substeps_from_clock": substeps,
            "actual_completed_control_intervals": substeps//self.plant.physics_steps,
            "partial_interval_physics_substeps": substeps % self.plant.physics_steps,
            "recorded_diagnostic_intervals": self._count, "error": self._error,
            "collision_terrain": self.plant.collision_terrain_metadata,
        })
        return super().close()


def run_episode(output, case, gates):
    summary, rows = run_archived_g1_episode(output, case, "plane_zero", gates,
        lambda **kwargs: PlaneDiagnosticEnv(diagnostic_output=output, **kwargs),
        contact_sampler=sample_native_contacts)
    with gzip.open(output / "plane_diagnostics.jsonl.gz", "rt") as stream:
        diagnostics = [json.loads(line) for line in stream]
    with gzip.open(output / "native_physics_entries.jsonl.gz", "rt") as stream:
        entries = [json.loads(line) for line in stream]
    if len(diagnostics) != len(rows) or len(entries) != len(rows)*5:
        raise RuntimeError("native/diagnostic lengths disagree with scored control intervals")
    invariant = all(e["returned"] and e["contacts"]["max_horizontal_normal"] <= 1e-12
                    and e["contacts"]["max_vertical_normal_error"] <= 1e-12 for e in entries)
    summary["plant_semantics_valid"] = invariant
    summary["native_contact_diagnostics"] = {
        "substeps": len(entries),
        "max_horizontal_normal": max(e["contacts"]["max_horizontal_normal"] for e in entries),
        "no_wheel_contact_substeps": sum(e["contacts"]["geometric_wheel_contact_count"] == 0 for e in entries),
        "max_active_nonwheel_contacts": max(e["contacts"]["active_nonwheel_contacts"] for e in entries),
        "cache_phase": "native_step_solved_cache_not_synchronized_endpoint"}
    summary["controller_diagnostics"] = {
        "inner_yaw_limit_occupied_intervals": sum(d["inner_yaw_limit_occupied"] for d in diagnostics),
        "protected_joint_intervals": sum(sum(d["torque_protected"]) for d in diagnostics),
        "peak_abs_torque_nm": max(abs(v) for d in diagnostics for v in d["requested_torque_nm"])}
    write_json(output / "plane_summary.json", summary)
    write_json(output / "complete_manifest.json", {
        str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*")) if p.is_file()})
    if not invariant:
        raise RuntimeError("plane normal invariant failed; remaining physics stopped")
    with np.load(output / "states.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays.update({k: np.asarray([d[k] for d in diagnostics]) for k in ARRAY_FIELDS})
    return summary, arrays


def prefix_check(name, a, b, transitions, *, next_observation_equal):
    checks = {}
    for key in a:
        states = key in ("qpos", "qvel", "truth_positions_world_m")
        count = transitions+int(states or (key == "observations" and next_observation_equal))
        checks[key] = len(a[key]) >= count and len(b[key]) >= count and bitwise_equal(a[key][:count], b[key][:count])
    return {"comparison": name, "transitions": transitions, "checks": checks,
            "passed": all(checks.values())}


def initial_check(name, first, arrays):
    # The original reverse schedule produces -0.0 in its initial command and
    # three observation fields. Preserve raw evidence and canonicalize exactly
    # zero only for this cross-direction initial observation comparison.
    a, b = first["observations"][:1].copy(), arrays["observations"][:1].copy()
    checks = {k: bool(len(first[k]) > 0 and len(arrays[k]) > 0
              and np.isfinite(first[k][:1]).all() and np.isfinite(arrays[k][:1]).all()
              and bitwise_equal(first[k][:1], arrays[k][:1])) for k in ("qpos", "qvel")}
    checks["observation_shape_dtype_finite"] = bool(
        a.shape == b.shape == (1, 85) and a.dtype == b.dtype == np.float32
        and np.isfinite(a).all() and np.isfinite(b).all())
    if not checks["observation_shape_dtype_finite"]:
        return {"comparison": name+"_initial", "checks": checks, "passed": False,
                "raw_observations_bitwise_equal": bitwise_equal(a, b),
                "normalization": "not applied to invalid observation"}
    raw_equal = bitwise_equal(a, b)
    sign_difference = np.signbit(a[0]) != np.signbit(b[0])
    differences = np.flatnonzero(sign_difference).tolist()
    signed_zero = np.flatnonzero(sign_difference & (a[0] == 0.) & (b[0] == 0.)).tolist()
    numeric_difference = np.flatnonzero(a[0] != b[0]).tolist()
    a[a == 0.] = 0.
    b[b == 0.] = 0.
    checks["observations_equal_mod_signed_zero"] = bitwise_equal(a, b)
    return {"comparison": name+"_initial", "checks": checks, "passed": all(checks.values()),
            "comparison_rule": "initial_observation_exact_zero_canonicalization_v1",
            "raw_observations_bitwise_equal": raw_equal,
            "raw_observation_signbit_differences": differences,
            "signed_zero_difference_indices": signed_zero,
            "numeric_difference_indices": numeric_difference,
            "normalization": "exact zero sign only, initial observations only; raw files unchanged"}


def read_completed_arrays(folder):
    with np.load(folder / "states.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    with gzip.open(folder / "plane_diagnostics.jsonl.gz", "rt") as stream:
        diagnostics = [json.loads(line) for line in stream]
    arrays.update({k: np.asarray([d[k] for d in diagnostics]) for k in ARRAY_FIELDS})
    return arrays


def validate_completed_prefix(folder, cases):
    """Accept only the complete two-case signed-zero interruption, never replay."""
    old = json.loads((folder / "protocol.json").read_text())
    failure = json.loads((folder / "batch_failure.json").read_text())
    manifest = json.loads((folder / "failure_manifest.json").read_text())
    if any(sha256(folder / p) != item["sha256"] for p, item in manifest.items()):
        raise ValueError("interrupted batch evidence changed")
    if failure["type"] != "RuntimeError" or failure["message"] != "same-command prefix mismatch; remaining physics stopped":
        raise ValueError("unsupported interruption reason")
    if old["schema"] != SCHEMA or old["cases"] != cases or failure["completed_control_intervals_from_clock"] != 1600:
        raise ValueError("expected original two complete plane-stop cases")
    if failure["actual_physics_substeps_from_clock"] != 8000 or failure["partial_interval_substeps"] != 0:
        raise ValueError("partial physics cannot be carried as a completed case")
    for p, h in old["input_sha256"].items():
        if Path(p).resolve() != Path(__file__).resolve() and sha256(Path(p)) != h:
            raise ValueError("physical input changed since completed prefix")
    selected = [c["name"] for c in cases[:2]]
    if sorted(p.parent.name for p in folder.glob("*/plane_summary.json")) != sorted(selected):
        raise ValueError("unexpected completed episode set")
    for name in selected:
        base = folder / name
        manifest = json.loads((base / "complete_manifest.json").read_text())
        if any(sha256(base / p) != item["sha256"] for p, item in manifest.items()):
            raise ValueError("completed evidence hash changed")
        summary = json.loads((base / "plane_summary.json").read_text())
        receipt = json.loads((base / "native_entry_receipt.json").read_text())
        if (summary["actual_transitions"] != 800 or not summary["plant_semantics_valid"]
                or receipt["observed_native_calls"] != 4000 or receipt["returned_native_calls"] != 4000
                or receipt["failure"] is not None or receipt["archival_error"] is not None):
            raise ValueError("completed prefix lacks full valid native evidence")
    arrays = [read_completed_arrays(folder / name) for name in selected]
    check = initial_check("carried_reverse", arrays[0], arrays[1])
    if not check["passed"] or check["raw_observation_signbit_differences"] != [38, 56, 58]:
        raise ValueError("original interruption is not solely the recorded signed-zero mismatch")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-protocol", type=Path, required=True)
    parser.add_argument("--old-episodes", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-prefix", type=Path)
    parser.add_argument("--resume-contract", type=Path)
    args = parser.parse_args(argv)
    if sha256(args.g1_protocol) != g1.PROTOCOL_SHA256:
        raise ValueError("original G1 rev2 protocol required")
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("passed") is not True:
        raise ValueError("a passed nonphysical preflight report is required")
    if any(sha256(Path(p)) != h for p, h in preflight["input_sha256"].items()):
        raise ValueError("nonphysical preflight inputs changed")
    protocol = json.loads(args.g1_protocol.read_text())
    cases = protocol["cases"][:6]
    if sum(c["max_transitions"] for c in cases) != 5600:
        raise ValueError("unexpected six-case budget")
    carried = []
    if args.completed_prefix is not None:
        if args.resume_contract is None:
            raise ValueError("completed prefix requires its separate resume contract")
        carried = validate_completed_prefix(args.completed_prefix, cases)
    frozen = json.loads((ROOT / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(sha256(ROOT / p) != h for p, h in frozen.items()):
        raise RuntimeError("frozen77 changed")
    inputs = {str((ROOT / p).resolve()): h for p, h in source_hashes().items()}
    for p in [Path(__file__), args.g1_protocol, args.contract, args.preflight,
              *(ROOT / "scripts" / name for name in ("d1_flat_plane_env.py", "d1_native_contact_diagnostics.py",
                "d1_probe_archive.py", "d1_turn_contact_diagnostics.py", "probe_d1_heading_g1.py",
                "probe_d1_heading_release.py", "d1_release_velocity_governor.py"))]:
        inputs[str(p.resolve())] = sha256(p)
    if carried:
        inputs[str(args.resume_contract.resolve())] = sha256(args.resume_contract)
        for p in sorted(args.completed_prefix.rglob("*")):
            if p.is_file():
                inputs[str(p.resolve())] = sha256(p)
    old = {}
    for case in cases:
        p = args.old_episodes / case["name"] / "zero/summary.json"
        old[case["name"]] = json.loads(p.read_text())
        inputs[str(p.resolve())] = sha256(p)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"schema": SCHEMA, "cases": cases,
        "condition": "plane_zero", "original_gates": protocol["proposed_gates"],
        "input_sha256": inputs, "max_control_transitions": 5600, "max_physics_substeps": 28000,
        "carried_control_transitions": 1600 if carried else 0,
        "max_new_control_transitions": 4000 if carried else 5600,
        "max_new_physics_substeps": 20000 if carried else 28000,
        "comparison_rule": "initial_observation_exact_zero_canonicalization_v1",
        "new_training_steps": 0, "default_changed": False, "old_hfield_replayed": False,
        "raw_stop_tick": 400, "release_shaping": False, "leg_damping": False,
        "inner_yaw_limit_rps": .6, "score_against_original_raw_commands": True})
    write_json(args.output / "execution_carry.json", {"completed_cases": carried,
        "source": str(args.completed_prefix) if carried else None,
        "carried_control_transitions": 1600 if carried else 0,
        "maximum_new_control_transitions": 4000 if carried else 5600,
        "maximum_new_physics_substeps": 20000 if carried else 28000,
        "source_runner_sha256": json.loads((args.completed_prefix / "protocol.json").read_text())["input_sha256"][str(Path(__file__).resolve())] if carried else None,
        "current_runner_sha256": sha256(Path(__file__)),
        "normalization": "initial observations only, exact signed zero; no trajectory or score changes"})
    try:
        return execute_batch(args, protocol, cases, inputs, frozen, old, carried)
    except BaseException as error:
        receipts = [{**json.loads(p.read_text()), "case": p.parent.name}
                    for p in args.output.glob("*/plane_execution_receipt.json")]
        native_receipts = [json.loads(p.read_text()) for p in args.output.glob("*/native_entry_receipt.json")]
        new_receipts = [r for r in receipts if r["case"] not in carried]
        new_native_calls = sum(json.loads(p.read_text())["observed_native_calls"]
                               for p in args.output.glob("*/native_entry_receipt.json") if p.parent.name not in carried)
        write_json(args.output / "batch_failure.json", {
            "type": type(error).__name__, "message": str(error),
            "actual_physics_substeps_from_clock": (8000 if carried else 0)+sum(r["actual_physics_substeps_from_clock"] for r in new_receipts),
            "completed_control_intervals_from_clock": (1600 if carried else 0)+sum(r["actual_completed_control_intervals"] for r in new_receipts),
            "partial_interval_substeps": sum(r["partial_interval_physics_substeps"] for r in new_receipts),
            "native_call_count": (8000 if carried else 0)+new_native_calls,
            "carried_cases": carried,
            "new_physics_substeps_from_clock": sum(r["actual_physics_substeps_from_clock"] for r in receipts if r["case"] not in carried),
            "carried_physics_substeps": 8000 if carried else 0,
            "copied_receipt_native_call_count": sum(r["observed_native_calls"] for r in native_receipts),
            "phase": "episode execution or postprocessing; see episode receipts and traceback",
            "no_retry_or_padding": True, "remaining_cases_stopped": True,
        })
        write_json(args.output / "failure_manifest.json", {
            str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(args.output.rglob("*")) if p.is_file()})
        raise


def execute_batch(args, protocol, cases, inputs, frozen, old, carried):
    summaries, records, pairs = [], {}, []
    for case in cases:
        name = case["name"]
        if name in carried:
            shutil.copytree(args.completed_prefix / name, args.output / name)
            manifest = json.loads((args.output / name / "complete_manifest.json").read_text())
            if any(sha256(args.output / name / p) != item["sha256"] for p, item in manifest.items()):
                raise ValueError("copied prefix evidence differs from its original manifest")
            summary = json.loads((args.output / name / "plane_summary.json").read_text())
            arrays = read_completed_arrays(args.output / name)
        else:
            summary, arrays = run_episode(args.output / name, case, protocol["proposed_gates"])
        origin = (args.completed_prefix if name in carried else args.output) / name
        summary = {**summary, "reused_without_physics": name in carried,
                   "execution_origin": str(origin), "origin_complete_manifest_sha256": sha256(origin / "complete_manifest.json")}
        summaries.append(summary)
        records[name] = arrays
        if len(records) > 1:
            first = records[cases[0]["name"]]
            pairs.append(initial_check(name, first, arrays))
        if name == cases[3]["name"]:
            pairs.append(prefix_check("turn_prefix", records[cases[2]["name"]], arrays, 200, next_observation_equal=False))
        if name in (cases[4]["name"], cases[5]["name"]):
            pairs.append(prefix_check(name+"_stop_prefix", records[cases[0]["name"]], arrays, 400, next_observation_equal=False))
        if name == cases[5]["name"]:
            pairs.append(prefix_check("impulse_prefix", records[cases[4]["name"]], arrays, 600, next_observation_equal=True))
        write_json(args.output / f"consistency_after_{name}.json", pairs)
        if not all(p["passed"] for p in pairs):
            raise RuntimeError("same-command prefix mismatch; remaining physics stopped")
        print(json.dumps({"case": name, "actual_transitions": summary["actual_transitions"],
            "reused_without_physics": name in carried,
            "gates": summary["gates"], "heading_peak_rad": summary["heading_peak_rad"],
            "velocity_rmse_mps": summary["velocity_rmse_mps"],
            "late_speed_mps": summary.get("late_stop_max_abs_body_vx_mps"),
            "late_path_m": summary.get("late_stop_cumulative_planar_path_m"),
            "plant_semantics_valid": summary["plant_semantics_valid"]}), flush=True)
    if any(sha256(Path(p)) != h for p, h in inputs.items()) or any(sha256(ROOT/p) != h for p, h in frozen.items()):
        raise RuntimeError("source changed during evaluation")
    write_json(args.output / "summary.json", {"schema": SCHEMA, "episodes": summaries,
        "old_hfield_summary_descriptive_only": old,
        "plant_semantics_valid": all(s["plant_semantics_valid"] for s in summaries),
        "execution_evidence_valid": all(p["passed"] for p in pairs), "prefix_checks": pairs,
        "actual_control_transitions": sum(s["actual_transitions"] for s in summaries),
        "actual_physics_substeps": sum(s["actual_observed_physics_substeps"] for s in summaries),
        "carried_control_transitions": 1600 if carried else 0,
        "new_control_transitions": sum(s["actual_transitions"] for s in summaries if s["case"] not in carried),
        "new_physics_substeps": sum(s["actual_observed_physics_substeps"] for s in summaries if s["case"] not in carried),
        "all_original_gates_passed": all(s["gates"]["passed"] for s in summaries),
        "frozen77_unchanged": True, "default_changed": False, "full_driving_goal_complete": False})
    write_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
