"""One fixed post-stop leg-damping candidate on the original raw G1 commands."""
from __future__ import annotations

import argparse
import gzip
import json
from contextlib import ExitStack
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from scripts import probe_d1_heading_g1 as g1
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv
from scripts.d1_probe_archive import ProbeStateArchive, run_archived_g1_episode
from scripts.d1_stop_leg_damping import (
    STOP_LEG_DAMPING_NSPM,
    STOP_LEG_DAMPING_SCHEMA,
    StopLegDampingController,
)
from scripts.probe_d1_heading_release import bitwise_equal, paired_check
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest
from wheel_legged_control.d1.control_loop import D1WheelLegControllerAdapter

SCHEMA = "d1-heading-stop-leg-damping-probe-v1"
CONDITIONS = {"bypass": False, "stop_damping": True}


class StopDampingEnv(D1HeadingTrackingEnv):
    """Thin zero-only diagnostic environment with a distinct checkpoint identity."""

    task_schema = "d1-heading-stop-leg-damping-zero-task-v1"

    def __init__(self, *, damping_enabled, diagnostic_output=None, **kwargs):
        self._damping_enabled = damping_enabled
        super().__init__(**kwargs)
        self._controller = D1WheelLegControllerAdapter(StopLegDampingController(
            enabled=damping_enabled, **asdict(self.wheel_leg_control)
        ))
        self._diagnostic_output = diagnostic_output
        self._files = ExitStack()
        self._stream = None if diagnostic_output is None else self._files.enter_context(
            gzip.open(diagnostic_output / "damping_trace.jsonl.gz", "xt")  # noqa: SIM115 -- owned by env.close
        )
        self._count = 0
        self._error = None
        self._state_archive = ProbeStateArchive(self.plant, diagnostic_output)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._state_archive.capture("reset")
        return result

    def _build_heading_task_config(self):
        config = super()._build_heading_task_config()
        config["stop_leg_damping"] = {"schema": STOP_LEG_DAMPING_SCHEMA,
            "enabled": self._damping_enabled, "damping_nspm": STOP_LEG_DAMPING_NSPM,
            "zero_residual_only": True, "release_reference_shaping": False}
        return config

    def step(self, action):
        checked = np.asarray(action, dtype=np.float64)
        if checked.shape != (8,) or not np.isfinite(checked).all() or np.any(checked):
            raise ValueError("stop damping diagnostic requires eight exactly zero residuals")
        before = self.heading_decision
        state = self.decision.context.state
        try:
            transition = super().step(action)
            record = self._controller.controller.last_damping
            controller_result = self._controller.controller.last_result
            damping = {f.name: getattr(record, f.name).tolist()
                       if isinstance(getattr(record, f.name), np.ndarray) else getattr(record, f.name)
                       for f in fields(record)}
            traces = self.plant.last_control_interval_actuator_traces
            row = {"tick": before.tick, "endpoint_tick": before.tick+1,
                "raw_user_command": asdict(before.user_command),
                "servo_command": asdict(before.servo_command), "damping": damping,
                "body_forward_before_mps": float(state.base_linear_velocity_body[0]),
                "wheel_rolling_before_mps": float(np.mean(state.joint_velocity[[3, 7, 11, 15]])*.087),
                "requested_torque_nm": self.last_transition.requested_torque_nm.tolist(),
                "applied_torque_nm": [t.applied_nm.tolist() for t in traces],
                "wheel_integral_before_nm": controller_result.memory_before.wheel_integral_nm.tolist(),
                "wheel_integral_after_nm": controller_result.memory_after.wheel_integral_nm.tolist(),
                "torque_protected": controller_result.torque_limited.tolist()}
            if self._stream is not None:
                self._stream.write(json.dumps(row, allow_nan=False)+"\n")
            self._count += 1
            if transition[2] or transition[3]:
                self._files.close()
            return transition
        except BaseException as error:
            self._error = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            self._state_archive.capture("failed_step" if self._error else "completed_step")

    def close(self):
        self._files.close()
        self._state_archive.close()
        if self._diagnostic_output is not None:
            substeps = round(float(self.plant.data.time)/float(self.plant.model.opt.timestep))
            write_json(self._diagnostic_output / "damping_execution_receipt.json", {
                "actual_physics_substeps_from_clock": substeps,
                "actual_completed_control_intervals": substeps // self.plant.physics_steps,
                "partial_interval_physics_substeps": substeps % self.plant.physics_steps,
                "recorded_diagnostic_intervals": self._count, "error": self._error,
                "no_retry_or_padding": True,
            })
        return super().close()


def run_episode(output, case, condition, gates):
    factory = lambda **kwargs: StopDampingEnv(
        damping_enabled=CONDITIONS[condition], diagnostic_output=output, **kwargs
    )
    summary, rows = run_archived_g1_episode(output, case, condition, gates, factory)
    with gzip.open(output / "damping_trace.jsonl.gz", "rt") as stream:
        diagnostic = [json.loads(line) for line in stream]
    if len(diagnostic) != len(rows):
        raise RuntimeError("damping log T differs from physically scored T")
    with np.load(output / "states.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    for key in ("requested_torque_nm", "applied_torque_nm", "wheel_integral_before_nm", "wheel_integral_after_nm"):
        arrays[key] = np.asarray([r[key] for r in diagnostic])
    active = [r for r in diagnostic if r["damping"]["active"]]
    if any(r["damping"]["joint_power_w"] > 1e-12 for r in active):
        raise RuntimeError("sampled damping torque injected joint energy")
    summary["damping_diagnostics"] = {
        "active_control_intervals": len(active),
        "first_active_tick": active[0]["tick"] if active else None,
        "peak_abs_increment_nm": max(max(abs(v) for v in r["damping"]["delta_torque_nm"]) for r in diagnostic),
        "joint_power_max_w": max(r["damping"]["joint_power_w"] for r in diagnostic),
        "joint_power_min_w": min(r["damping"]["joint_power_w"] for r in diagnostic),
        "protected_joint_interval_count": sum(sum(r["torque_protected"]) for r in diagnostic),
    }
    write_json(output / "damping_summary.json", summary)
    write_json(output / "complete_manifest.json", {
        str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*")) if p.is_file()
    })
    return summary, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-protocol", type=Path, required=True)
    parser.add_argument("--old-episodes", type=Path, required=True)
    parser.add_argument("--gain-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256(args.g1_protocol) != g1.PROTOCOL_SHA256:
        raise ValueError("original G1 revision2 required")
    gain = json.loads(args.gain_provenance.read_text())["hypothetical_damper_gain_provenance"]
    if gain["proposed_extra_per_leg_damping_Nspm"] != STOP_LEG_DAMPING_NSPM:
        raise ValueError("gain differs from the preregistered nominal-mode derivation")
    protocol = json.loads(args.g1_protocol.read_text())
    cases = [protocol["cases"][i] for i in (0, 1, 4, 5)]
    frozen = json.loads((ROOT / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(sha256(ROOT / p) != v for p, v in frozen.items()):
        raise RuntimeError("frozen sources changed")
    inputs = source_hashes()
    for p in (Path(__file__), ROOT / "scripts/d1_stop_leg_damping.py",
              ROOT / "scripts/d1_probe_archive.py",
              ROOT / "scripts/probe_d1_heading_g1.py", ROOT / "scripts/probe_d1_heading_release.py",
              ROOT / "scripts/d1_release_velocity_governor.py", args.gain_provenance, args.g1_protocol):
        inputs[str(p.resolve())] = sha256(p)
    old = {}
    for case in cases:
        path = args.old_episodes / case["name"] / "zero/states.npz"
        inputs[str(path.resolve())] = sha256(path)
        with np.load(path) as data:
            old[case["name"]] = {k: data[k].copy() for k in data.files}
        if len(old[case["name"]]["qpos"]) != case["max_transitions"]+1:
            raise ValueError("old evidence incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"schema": SCHEMA, "cases": cases,
        "conditions": CONDITIONS, "fixed_damping_nspm": STOP_LEG_DAMPING_NSPM,
        "gain_provenance": gain, "input_sha256": inputs,
        "original_gates": protocol["proposed_gates"],
        "max_control_transitions": 8000, "max_physics_substeps": 40000,
        "raw_stop_tick": 400, "release_reference_shaping": False,
        "zero_only": True, "new_training_steps": 0, "no_parameter_sweep": True,
        "default_changed": False})
    summaries, pairs = [], []
    for case in cases:
        baseline = None
        for condition in CONDITIONS:
            summary, arrays = run_episode(args.output / case["name"] / condition,
                                          case, condition, protocol["proposed_gates"])
            summaries.append(summary)
            if baseline is None:
                checks = {k: bitwise_equal(arrays[k], v) for k, v in old[case["name"]].items()}
                pair = {"case": case["name"], "comparison": "old_zero_reproduction",
                        "checks": checks, "passed": all(checks.values())}
                baseline = arrays
            else:
                pair = paired_check(case, baseline, arrays)
                if case["command"]["stop_tick"] is not None:
                    end = case["command"]["stop_tick"] + 1
                    pair["checks"]["observation_through_release_state"] = bitwise_equal(
                        baseline["observations"][:end], arrays["observations"][:end])
                    pair["passed"] = all(pair["checks"].values())
                pair["comparison"] = "inactive_control_prefix_or_full_impulse"
            pairs.append(pair)
            write_json(args.output / f"pair_{case['name']}_{condition}.json", pair)
            if not pair["passed"]:
                raise RuntimeError("physical comparison invalid; remaining physics stopped")
            print(json.dumps({"case": case["name"], "condition": condition,
                "gates": summary["gates"], "velocity_rmse_mps": summary["velocity_rmse_mps"],
                "late_speed_mps": summary.get("late_stop_max_abs_body_vx_mps"),
                "late_path_m": summary.get("late_stop_cumulative_planar_path_m"),
                "damping": summary["damping_diagnostics"]}), flush=True)
    if any(sha256(ROOT / p) != v for p, v in inputs.items()) or any(sha256(ROOT / p) != v for p, v in frozen.items()):
        raise RuntimeError("source changed during evaluation")
    stops = [s for s in summaries if s["model"] == "stop_damping" and s["case"] in ("flat_forward_stop", "flat_reverse_stop")]
    write_json(args.output / "summary.json", {"schema": SCHEMA, "episodes": summaries,
        "pairs": pairs, "comparison_valid": all(p["passed"] for p in pairs),
        "actual_control_transitions": sum(s["actual_transitions"] for s in summaries),
        "actual_physics_substeps": sum(s["actual_observed_physics_substeps"] for s in summaries),
        "candidate_passed_both_stop_cases": len(stops) == 2 and all(s["gates"]["passed"] for s in stops),
        "frozen77_unchanged": True, "default_changed": False, "full_driving_goal_complete": False})
    write_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
