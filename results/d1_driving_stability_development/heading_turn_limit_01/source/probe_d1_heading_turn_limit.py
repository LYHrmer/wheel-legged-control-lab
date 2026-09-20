"""Fixed zero-policy yaw-limit causal probe, retaining original G1 scoring."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from scripts import probe_d1_heading_g1 as g1
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv
from scripts.d1_turn_contact_diagnostics import TURN_CONTACT_SCHEMA, sample_turn_contacts
from scripts.probe_d1_heading_release import bitwise_equal
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest

SCHEMA = "d1-heading-yaw-limit-causal-probe-v1"
CONDITIONS = {"limit0p6": .6, "limit1p0": 1.}


class TurnDiagnosticEnv(D1HeadingTrackingEnv):
    """Same env step, one fixed instance yaw cap, and read-only endpoint telemetry."""

    def __init__(self, *, diagnostic_output, yaw_limit_rps, **kwargs):
        super().__init__(**kwargs)
        self._turn_output = diagnostic_output
        self._turn_limit = yaw_limit_rps
        controller = self._controller.controller
        controller.yaw_request_limit_rps = yaw_limit_rps
        if yaw_limit_rps != .6:
            controller.control_schema = "d1-wheel-leg-yaw-limit1p0-diagnostic-v1"
        self._turn_files = ExitStack()
        self._turn_stream = self._turn_files.enter_context(
            gzip.open(diagnostic_output / "turn_diagnostics.jsonl.gz", "xt")  # noqa: SIM115 -- env.close owns ExitStack
        )
        self._turn_count = 0
        self._turn_error = None

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        config = {"schema": SCHEMA, "yaw_request_limit_rps": self._turn_limit,
                  "contact_schema": TURN_CONTACT_SCHEMA, "zero_only": True,
                  "original_raw_user_reference": True}
        self._episode_metadata["turn_probe_configuration"] = config
        info["episode_metadata"] = self.episode_metadata
        return observation, info

    def step(self, action):
        before = self.heading_decision
        state = self.decision.context.state
        yaw = state.base_rpy[2]
        lateral = state.foot_offset_world @ np.array([-np.sin(yaw), np.cos(yaw), 0.])
        rolling = state.joint_velocity[[3, 7, 11, 15]] * .087
        wheel_fit = np.linalg.lstsq(np.column_stack((np.ones(4), -lateral)), rolling, rcond=None)[0]
        try:
            result = super().step(action)
            control = self.loop.controller.controller.last_result
            saved_qpos = self.plant.data.qpos.copy()
            saved_qvel = self.plant.data.qvel.copy()
            saved_time = float(self.plant.data.time)
            contacts = sample_turn_contacts(self.plant)
            reference_wrench = self.plant.measure_wheel_contact_wrench()
            if not np.allclose(contacts["total_wrench_world_6"], reference_wrench.wrench_world,
                               rtol=0., atol=1e-10):
                raise RuntimeError("contact observer disagrees with original wrench convention")
            if (not bitwise_equal(saved_qpos, self.plant.data.qpos)
                    or not bitwise_equal(saved_qvel, self.plant.data.qvel)
                    or saved_time != float(self.plant.data.time)):
                raise RuntimeError("contact observer changed physical state")
            traces = self.plant.last_control_interval_actuator_traces
            servo = before.servo_command.yaw_rate_rps
            rate = float(state.base_angular_velocity_body[2])
            requested = servo + self.loop.controller.controller.yaw_feedback_gain * (servo-rate)
            row = {
                "tick": before.tick, "endpoint_tick": before.tick + 1,
                "body_yaw_rate_before_rps": rate,
                "body_yaw_rate_after_rps": float(self.last_transition.truth.base_angular_velocity_body[2]),
                "wheel_fit_forward_before_mps": float(wheel_fit[0]),
                "wheel_fit_yaw_before_rps": float(wheel_fit[1]),
                "lateral_wheel_offsets_before_m": lateral.tolist(),
                "raw_yaw_rps": before.user_command.yaw_rate_rps,
                "servo_yaw_rps": servo, "inner_unclipped_yaw_rps": requested,
                "effective_yaw_rps": control.effective_yaw_request_rps,
                "inner_yaw_clipped": abs(requested) > self._turn_limit,
                "nominal_wheel_speed_rad_s": control.nominal_wheel_speed_rad_s.tolist(),
                "wheel_integral_before_nm": control.memory_before.wheel_integral_nm.tolist(),
                "wheel_integral_after_nm": control.memory_after.wheel_integral_nm.tolist(),
                "requested_torque_nm": self.last_transition.requested_torque_nm.tolist(),
                "applied_torque_nm": [t.applied_nm.tolist() for t in traces],
                "contacts": contacts,
            }
            self._turn_stream.write(json.dumps(row, allow_nan=False) + "\n")
            self._turn_count += 1
            if result[2] or result[3]:
                self._turn_files.close()
            return result
        except BaseException as error:
            self._turn_error = {"type": type(error).__name__, "message": str(error)}
            raise

    def close(self):
        self._turn_files.close()
        substeps = round(float(self.plant.data.time) / float(self.plant.model.opt.timestep))
        write_json(self._turn_output / "turn_execution_receipt.json", {
            "physical_time_s": float(self.plant.data.time),
            "actual_physics_substeps_from_clock": substeps,
            "actual_completed_intervals_from_clock": substeps // self.plant.physics_steps,
            "partial_interval_physics_substeps": substeps % self.plant.physics_steps,
            "recorded_diagnostic_intervals": self._turn_count,
            "error": self._turn_error, "do_not_retry_or_pad": True,
        })
        return super().close()


def run_instrumented(output, case, condition, gates):
    # The original runner has no factory parameter. This isolated, sequential
    # probe replaces only its constructor binding and always restores it.
    original = g1.D1HeadingTrackingEnv
    g1.D1HeadingTrackingEnv = lambda **kwargs: TurnDiagnosticEnv(
        diagnostic_output=output, yaw_limit_rps=CONDITIONS[condition], **kwargs
    )
    try:
        summary, rows = g1.run_episode(output, case, {"kind": "zero_action", "label": condition}, gates)
    finally:
        g1.D1HeadingTrackingEnv = original
    with gzip.open(output / "turn_diagnostics.jsonl.gz", "rt") as stream:
        diagnostic = [json.loads(line) for line in stream]
    assert len(diagnostic) == len(rows)
    pulse = diagnostic[200:250]
    speeds = [c["tangential_relative_speed_mps"] for row in pulse for c in row["contacts"]["contacts"]]
    metrics = {
        "case": case["name"], "condition": condition, "gates": summary["gates"],
        "heading_peak_rad": summary["heading_peak_rad"],
        "pulse_body_yaw_mean_before_rps": float(np.mean([r["body_yaw_rate_before_rps"] for r in pulse])),
        "pulse_wheel_fit_yaw_mean_before_rps": float(np.mean([r["wheel_fit_yaw_before_rps"] for r in pulse])),
        "pulse_contact_tangent_speed_rms_mps": float(np.sqrt(np.mean(np.square(speeds)))) if speeds else None,
        "pulse_observed_contact_samples": len(speeds),
        "pulse_contact_yaw_moment_mean_nm": float(np.mean([r["contacts"]["total_wrench_world_6"][5] for r in pulse])),
        "actual_control_transitions": len(rows),
        "actual_physics_substeps": summary["actual_observed_physics_substeps"],
        "contact_sampling": "endpoint samples; not a continuous-slip integral",
    }
    write_json(output / "turn_summary.json", metrics)
    # g1.run_episode already wrote its exclusive manifest before env.close.
    # Keep it, then independently cover all now-closed diagnostic artifacts.
    write_json(output / "complete_manifest.json", {
        str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*")) if p.is_file()
    })
    return metrics, diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-protocol", type=Path, required=True)
    parser.add_argument("--old-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-left-baseline", type=Path,
                        help="Reuse one complete 800-step baseline after archive-only failure")
    args = parser.parse_args(argv)
    if sha256(args.g1_protocol) != g1.PROTOCOL_SHA256:
        raise ValueError("original G1 revision 2 protocol required")
    original = json.loads(args.g1_protocol.read_text())
    cases = [original["cases"][i] for i in (2, 3)]
    frozen = json.loads((ROOT / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(sha256(ROOT / p) != v for p, v in frozen.items()):
        raise RuntimeError("frozen inputs changed")
    inputs = source_hashes()
    for p in (Path(__file__), ROOT / "scripts/d1_turn_contact_diagnostics.py",
              ROOT / "scripts/probe_d1_heading_g1.py", ROOT / "scripts/probe_d1_heading_release.py",
              ROOT / "scripts/d1_release_velocity_governor.py",
              args.g1_protocol):
        inputs[str(p.resolve())] = sha256(p)
    old_states = {}
    for case in cases:
        p = args.old_episodes / case["name"] / "zero/states.npz"
        inputs[str(p.resolve())] = sha256(p)
        with np.load(p) as data:
            old_states[case["name"]] = {k: data[k].copy() for k in data.files}
        if old_states[case["name"]]["qpos"].shape != (801, 23):
            raise ValueError("old turning evidence is incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.completed_left_baseline is not None:
        p = args.completed_left_baseline
        carry = json.loads((p / "turn_summary.json").read_text())
        if (carry["case"] != cases[0]["name"] or carry["condition"] != "limit0p6"
                or carry["actual_control_transitions"] != 800
                or carry["actual_physics_substeps"] != 4000):
            raise ValueError("carry-forward accepts only a fully completed left baseline")
        with np.load(p / "states.npz") as data:
            if not all(bitwise_equal(data[k], old_states[cases[0]["name"]][k]) for k in data.files):
                raise ValueError("carried baseline does not reproduce original G1")
        for artifact in p.iterdir():
            if artifact.is_file():
                inputs[str(artifact.resolve())] = sha256(artifact)
    write_json(args.output / "protocol.json", {
        "schema": SCHEMA, "conditions": CONDITIONS, "cases": cases,
        "original_gates": original["proposed_gates"], "input_sha256": inputs,
        "g1_protocol_sha256": g1.PROTOCOL_SHA256,
        "max_control_transitions": 3200, "max_physics_substeps": 16000,
        "carried_completed_transitions": 800 if args.completed_left_baseline else 0,
        "max_new_control_transitions": 2400 if args.completed_left_baseline else 3200,
        "candidate_value_source": "existing heading outer-loop cap, fixed before observation",
        "no_training_or_ppo": True, "no_sweep": True, "default_changed": False,
    })
    summaries, pairs = [], []
    for case in cases:
        baseline = baseline_diag = None
        for condition in CONDITIONS:
            output = args.output / case["name"] / condition
            if args.completed_left_baseline is not None and case == cases[0] and condition == "limit0p6":
                shutil.copytree(args.completed_left_baseline, output)
                summary = json.loads((output / "turn_summary.json").read_text())
                with gzip.open(output / "turn_diagnostics.jsonl.gz", "rt") as stream:
                    diagnostic = [json.loads(line) for line in stream]
                if len(diagnostic) != 800:
                    raise ValueError("carried diagnostic trace is incomplete")
                write_json(output / "carry_forward_receipt.json", {
                    "original_directory": str(args.completed_left_baseline),
                    "actual_new_physics_steps": 0,
                    "original_archive_failure": "exclusive manifest already existed after complete episode",
                    "original_manifest_retained_as_failure_evidence": True,
                })
            else:
                summary, diagnostic = run_instrumented(output, case, condition, original["proposed_gates"])
            summaries.append(summary)
            with np.load(output / "states.npz") as data:
                arrays = {k: data[k].copy() for k in data.files}
            if baseline is None:
                checks = {k: bitwise_equal(v, old_states[case["name"]][k]) for k, v in arrays.items()}
                baseline, baseline_diag = arrays, diagnostic
                label = "old_zero_reproduction"
            else:
                checks = {k: bitwise_equal(v[:201 if k in ("qpos", "qvel", "truth_positions_world_m") else 200],
                                          baseline[k][:201 if k in ("qpos", "qvel", "truth_positions_world_m") else 200])
                          for k, v in arrays.items()}
                checks["diagnostic_prefix"] = baseline_diag[:200] == diagnostic[:200]
                label = "before_raw_yaw_pulse"
            pair = {"case": case["name"], "comparison": label, "checks": checks,
                    "passed": all(checks.values())}
            pairs.append(pair)
            write_json(args.output / f"pair_{case['name']}_{condition}.json", pair)
            if not pair["passed"]:
                raise RuntimeError("invalid comparison; remaining physics stopped")
            print(json.dumps(summary), flush=True)
    if any(sha256(ROOT / p) != value for p, value in inputs.items()):
        raise RuntimeError("experiment input changed")
    if any(sha256(ROOT / p) != v for p, v in frozen.items()):
        raise RuntimeError("frozen inputs changed")
    write_json(args.output / "summary.json", {
        "schema": SCHEMA, "episodes": summaries, "pairs": pairs,
        "actual_control_transitions": sum(s["actual_control_transitions"] for s in summaries),
        "actual_physics_substeps": sum(s["actual_physics_substeps"] for s in summaries),
        "actual_new_control_transitions": sum(s["actual_control_transitions"] for s in summaries) - (800 if args.completed_left_baseline else 0),
        "carried_completed_transitions": 800 if args.completed_left_baseline else 0,
        "candidate_passed_both_turn_cases": all(s["gates"]["passed"] for s in summaries if s["condition"] == "limit1p0"),
        "frozen77_unchanged": True, "default_changed": False, "stable_driving_complete": False,
    })
    write_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
