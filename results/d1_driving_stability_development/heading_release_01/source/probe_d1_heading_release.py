"""One fixed zero-policy release experiment; scores always use raw operator intent.

The heading environment consumes served references in this diagnostic adapter.
Its legacy user_command_before and reward therefore describe that reference,
not the raw user goal. This runner never loads a learned policy or changes a GUI.
"""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv
from scripts.d1_release_velocity_governor import (
    RELEASE_GOVERNOR_SCHEMA,
    ReleaseVelocityGovernor,
)
from scripts.probe_d1_heading_g1 import (
    DT,
    PROTOCOL_SHA256,
    NativeWrenchObserver,
    command_at_tick,
    command_source,
    gates_and_metrics,
    phase_name,
    prescribed_wrench,
    rmse,
)
from scripts.run_d1_heading_study import source_hashes
from scripts.run_d1_wheel_common_mean_study import ROOT, sha256, write_json, write_manifest
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig

SCHEMA = "d1-heading-release-zero-probe-v1"
CONDITIONS = {"bypass": None, "release_0p5": 0.5}
WHEELS = [3, 7, 11, 15]
OLD_ARRAY_KEYS = ("qpos", "qvel", "observations", "truth_positions_world_m", "applied_actions")


class PreparedReleaseSource:
    """One raw callback per sequential prepared tick, including terminal prepare.

    A new instance owns each episode. Duplicate calls reuse its record; an
    environment reset must explicitly reset this source or use a fresh instance.
    """

    def __init__(self, raw_source, deceleration_mps2):
        self.raw_source = raw_source
        self.governor = ReleaseVelocityGovernor(deceleration_mps2)
        self.records = {}

    def __call__(self, seconds):
        tick = round(float(seconds) / DT)
        if not np.isfinite(seconds) or abs(seconds - tick * DT) > 1e-12:
            raise ValueError("source requires the fixed prepared decision clock")
        if tick in self.records:
            if seconds != self.records[tick].simulation_time_s:
                raise ValueError("same tick with a different prepare time")
            return self.records[tick].served_command
        if tick != len(self.records):
            raise ValueError("prepare ticks must be sequential from zero")
        record = self.governor.advance(self.raw_source(seconds), seconds)
        self.records[tick] = record
        return record.served_command


def bitwise_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes() == right.tobytes())


def release_diagnostics(case, rows, records):
    release = case["command"]["stop_tick"]
    if release is None or len(rows) <= release:
        return None
    first = rows[release:release + 50]
    zero = next((k for k in range(release, len(rows))
                 if records[k].served_command.forward_velocity_mps == 0), None)
    return {
        "first_0p5s_recorded_intervals": len(first),
        "first_0p5s_body_wheel_mismatch_rms_mps": rmse([
            row["body_forward_before_mps"] - row["wheel_rolling_before_mps"] for row in first
        ]),
        "first_zero_served_decision_tick": zero,
        "slew_elapsed_from_last_nonzero_raw_s": (
            records[zero].simulation_time_s - records[release - 1].simulation_time_s
            if zero is not None else None
        ),
        "executed_nonzero_reference_after_raw_release_s": sum(
            row["heading_task"]["actual_dt_s"] for row in rows[release:]
            if row["served_reference_command"]["forward_velocity_mps"] != 0
        ),
        "body_speed_rms_5to6s_mps": rmse([
            row["body_forward_mps"] for row in rows if 500 <= row["endpoint_tick"] < 600
        ]) if len(rows) >= 599 else None,
    }


def run_episode(output, case, condition, thresholds):
    output.mkdir(parents=True, exist_ok=False)
    source = PreparedReleaseSource(command_source(case["command"]), CONDITIONS[condition])
    env = D1HeadingTrackingEnv(
        terrain=D1LocomotionTerrainConfig(**case["terrain"]),
        episode_seconds=case["episode_seconds"], command_source=source,
    )
    rows, qpos, qvel, observations, positions = [], [], [], [], []
    observer = NativeWrenchObserver(env.plant, case)
    try:
        observation, metadata = env.reset(seed=case["seed"])
        write_json(output / "episode_metadata.json", metadata)
        positions.append(env.plant.base_position.copy())
        qpos.append(env.plant.data.qpos.copy())
        qvel.append(env.plant.data.qvel.copy())
        observations.append(observation.copy())
        with observer, gzip.open(output / "trace.jsonl.gz", "xt") as stream:
            for tick in range(case["max_transitions"]):
                if round(float(env.plant.data.time) / DT) != tick:
                    raise RuntimeError("unexpected time or hidden reset")
                record = source.records[tick]
                raw = command_at_tick(case["command"], tick)
                if raw != record.raw_command:
                    raise RuntimeError("raw intent differs from G1")
                state_before = env.decision.context.state
                body_before = float(state_before.base_linear_velocity_body[0])
                wheel_before = float(np.mean(state_before.joint_velocity[WHEELS]) * .087)
                start = len(observer.samples)
                observation, reward, terminated, truncated, info = env.step(
                    np.zeros(8, dtype=np.float32)
                )
                samples = observer.samples[start:]
                traces = env.plant.last_control_interval_actuator_traces
                expected_wrench = prescribed_wrench(case, tick)
                if (len(samples) != env.plant.physics_steps or len(traces) != len(samples)
                        or any(not np.array_equal(s["wrench_world"], expected_wrench)
                               for s in samples)):
                    raise RuntimeError("native wrench/substep receipt mismatch")
                heading = dict(info["heading_task"])
                if heading["user_command_before"] != asdict(record.served_command):
                    raise RuntimeError("environment did not execute served reference")
                # Explicitly relabel the old environment's use of the word user.
                heading["user_command_semantics"] = "served_reference_not_raw_operator_intent"
                truth = env.last_transition.truth
                result = env.loop.controller.controller.last_result
                w, x, y, z = env.plant.data.qpos[3:7]
                attitude = [float(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
                            float(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))]
                if not np.allclose(attitude, truth.base_rpy[:2], atol=1e-12, rtol=0):
                    raise RuntimeError("truth/quaternion attitude mismatch")
                if np.any(info["applied_action"]):
                    raise RuntimeError("zero residual experiment received a nonzero action")
                body_after = float(truth.base_linear_velocity_body[0])
                row = {
                    "tick": tick, "endpoint_tick": tick + 1, "phase": phase_name(case, tick),
                    "raw_user_command": asdict(raw),
                    "served_reference_command": asdict(record.served_command),
                    "governor_record": asdict(record), "heading_task": heading,
                    "reward": reward, "reward_terms": info["reward_terms"],
                    "servo_reward": info["servo_reward"],
                    "reward_semantics": "served_reference_diagnostic_only",
                    "metrics": info["metrics"], "terrain_exposure": info["terrain_exposure"],
                    "body_forward_before_mps": body_before,
                    "wheel_rolling_before_mps": wheel_before,
                    "body_forward_mps": body_after,
                    "user_forward_error_mps": body_after - raw.forward_velocity_mps,
                    "served_forward_error_mps": body_after - record.served_command.forward_velocity_mps,
                    "heading_error_rad": heading["heading_error_after"],
                    "actual_roll_pitch_rad": attitude,
                    "truth_position_world_m": truth.base_position.tolist(),
                    "cross_track_after_m": float(truth.base_position[1] - positions[0][1]),
                    "applied_action": info["applied_action"].tolist(),
                    "physics_wrench_samples": samples,
                    "actual_signed_yaw_impulse_nms": sum(s["wrench_world"][5]*s["actual_dt_s"] for s in samples),
                    "nominal_wheel_speed_rad_s": result.nominal_wheel_speed_rad_s.tolist(),
                    "wheel_speed_target_rad_s": result.wheel_speed_target_rad_s.tolist(),
                    "wheel_integral_before_nm": result.memory_before.wheel_integral_nm.tolist(),
                    "wheel_integral_after_nm": result.memory_after.wheel_integral_nm.tolist(),
                    "controller_unlimited_torque_nm": result.requested_torque_nm.tolist(),
                    "requested_torque_nm": env.last_transition.requested_torque_nm.tolist(),
                    "actuator_requested_nm": [t.requested_nm.tolist() for t in traces],
                    "actuator_applied_nm": [t.applied_nm.tolist() for t in traces],
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "terminal_reason": info["terminal_reason"],
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                qpos.append(env.plant.data.qpos.copy())
                qvel.append(env.plant.data.qvel.copy())
                observations.append(observation.copy())
                positions.append(truth.base_position.copy())
                if terminated or truncated:
                    break
        arrays = {"qpos": np.asarray(qpos), "qvel": np.asarray(qvel),
                      "observations": np.asarray(observations),
                      "truth_positions_world_m": np.asarray(positions),
                      "applied_actions": np.asarray([row["applied_action"] for row in rows]),
                      "requested_torque_nm": np.asarray([row["requested_torque_nm"] for row in rows]),
                      "actuator_applied_nm": np.asarray([row["actuator_applied_nm"] for row in rows]),
                      "raw_commands": np.asarray([list(asdict(r.raw_command).values())
                                               for r in source.records.values()]),
                      "served_commands": np.asarray([list(asdict(r.served_command).values())
                                                  for r in source.records.values()]),
                      "servo_commands": np.asarray([list(row["heading_task"]["servo_command_before"].values())
                                                 for row in rows]),
                      "physics_wrenches": np.asarray([[s["wrench_world"] for s in row["physics_wrench_samples"]]
                                                   for row in rows])}
        np.savez_compressed(output / "states.npz", **arrays)
        prepared_last = env.heading_decision.tick
        write_json(output / "prepared_commands.json", [
            {"decision_tick": tick, "physically_executed": tick < len(rows),
             "prepare_succeeded": tick <= prepared_last, **asdict(record)}
            for tick, record in source.records.items()
        ])
        metrics, gates = gates_and_metrics(case, rows, arrays["truth_positions_world_m"], thresholds)
        impulse = sum(row["actual_signed_yaw_impulse_nms"] for row in rows)
        expected = sum(prescribed_wrench(case, k)[5] * DT for k in range(len(rows)))
        if abs(impulse - expected) > 1e-10:
            raise RuntimeError("actual yaw impulse differs from executed intervals")
        summary = {
            "case": case["name"], "condition": condition, **metrics, "gates": gates,
            "release_diagnostics": release_diagnostics(case, rows, source.records),
            "actual_signed_yaw_impulse_nms": impulse,
            "actual_observed_physics_substeps": len(observer.samples),
            "source_records": len(source.records),
            "prepared_decisions": prepared_last + 1,
            "executed_decisions": len(rows),
            "reward_comparable_raw_goal_score": False,
            "raw_goal_scoring": "endpoint COM body vx minus raw command of executed interval",
            "wrench_zero_after_episode": bool(np.all(env.plant.data.xfrc_applied == 0)),
        }
        write_json(output / "summary.json", summary)
        write_manifest(output)
        return summary, arrays, rows
    except BaseException as error:
        np.savez_compressed(output / "partial_states.npz", qpos=np.asarray(qpos),
                            qvel=np.asarray(qvel), observations=np.asarray(observations),
                            truth_positions_world_m=np.asarray(positions),
                            final_physical_qpos=env.plant.data.qpos.copy(),
                            final_physical_qvel=env.plant.data.qvel.copy())
        write_json(output / "partial_source_records.json", [
            {"decision_tick": k, **asdict(record)} for k, record in source.records.items()
        ])
        write_json(output / "partial_wrench_samples.json", observer.samples)
        write_json(output / "execution_failure.json", {
            "type": type(error).__name__, "message": str(error),
            "fully_recorded_transitions": len(rows),
            "physical_time_s": float(env.plant.data.time),
            "observed_physics_substeps": len(observer.samples),
            "completed_physical_intervals": len(observer.samples) // env.plant.physics_steps,
            "do_not_retry_or_reset_to_pad": True,
        })
        raise
    finally:
        env.plant.data.xfrc_applied[:] = 0
        env.close()


def paired_check(case, baseline, candidate):
    stop = case["command"]["stop_tick"]
    checks = {}
    for key in baseline:
        count = None if stop is None else stop + (key in ("qpos", "qvel", "truth_positions_world_m"))
        checks[key] = bitwise_equal(baseline[key][:count], candidate[key][:count])
    return {"case": case["name"], "checks": checks, "passed": all(checks.values())}


def both_stop_cases_pass(summaries):
    selected = [s for s in summaries if s["condition"] == "release_0p5"
                and s["case"] in ("flat_forward_stop", "flat_reverse_stop")]
    return (len(selected) == 2 and len({s["case"] for s in selected}) == 2
            and all(s["gates"]["passed"] for s in selected))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-protocol", type=Path, required=True)
    parser.add_argument("--old-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256(args.g1_protocol) != PROTOCOL_SHA256:
        raise ValueError("requires original reviewed G1 protocol revision 2")
    original = json.loads(args.g1_protocol.read_text())
    cases = [original["cases"][i] for i in (0, 1, 4, 5)]
    frozen = json.loads((ROOT / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(sha256(ROOT / p) != value for p, value in frozen.items()):
        raise RuntimeError("77 frozen inputs must be unchanged")
    inputs = source_hashes()
    for path in (Path(__file__), ROOT / "scripts/d1_release_velocity_governor.py",
                 ROOT / "scripts/probe_d1_heading_g1.py", args.g1_protocol):
        inputs[str(path.resolve())] = sha256(path)
    for case in cases:
        path = args.old_episodes / case["name"] / "zero/states.npz"
        inputs[str(path.resolve())] = sha256(path)
        with np.load(path) as old:
            count = case["max_transitions"]
            for key, width in zip(OLD_ARRAY_KEYS, (23, 22, 85, 3, 8)):
                expected_count = count if key == "applied_actions" else count + 1
                if old[key].shape != (expected_count, width) or not np.isfinite(old[key]).all():
                    raise ValueError(f"invalid old evidence shape/data: {path} {key}")
            if old["observations"].dtype != np.float32:
                raise ValueError("old observation schema must remain float32")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": SCHEMA, "governor_schema": RELEASE_GOVERNOR_SCHEMA,
        "conditions": CONDITIONS, "cases": cases, "gates": original["proposed_gates"],
        "g1_protocol_sha256": PROTOCOL_SHA256, "input_sha256": inputs,
        "max_actual_control_transitions": 8000, "max_actual_physics_substeps": 40000,
        "no_training_or_ppo_loading": True, "first_termination_ends_episode": True,
        "original_user_stop_tick": 400, "late_speed_endpoint_ticks": [500, 800],
        "path_position_ticks_inclusive": [500, 700],
        "reward_semantics": "served-reference diagnostics only; gates use raw intent",
        "no_parameter_sweep": True, "default_behavior_changed": False,
    }
    write_json(args.output / "protocol.json", protocol)
    summaries, pairs, reproduction = [], [], []
    for case in cases:
        baseline = None
        for condition in CONDITIONS:
            summary, arrays, rows = run_episode(
                args.output / case["name"] / condition, case, condition, protocol["gates"]
            )
            summaries.append(summary)
            if baseline is None:
                baseline = arrays
                with np.load(args.old_episodes / case["name"] / "zero/states.npz") as old:
                    checks = {key: bitwise_equal(arrays[key], old[key]) for key in OLD_ARRAY_KEYS}
                reproduction.append({"case": case["name"], "checks": checks,
                                     "passed": all(checks.values())})
                if not all(checks.values()):
                    write_json(args.output / "invalid_reproduction.json", reproduction)
                    raise RuntimeError("bypass failed old zero reproduction; remaining physics stopped")
            else:
                pairs.append(paired_check(case, baseline, arrays))
                if case["command"]["stop_tick"] is None:
                    pairs[-1]["raw_equals_served"] = all(
                        row["raw_user_command"] == row["served_reference_command"] for row in rows
                    )
                    pairs[-1]["passed"] &= pairs[-1]["raw_equals_served"]
                if not pairs[-1]["passed"]:
                    write_json(args.output / "invalid_pair.json", pairs)
                    raise RuntimeError("pair invariants failed; remaining physics stopped")
            print(json.dumps(summary, allow_nan=False), flush=True)
    unchanged = all(sha256(ROOT / p) == value for p, value in inputs.items())
    if not unchanged or any(sha256(ROOT / p) != value for p, value in frozen.items()):
        raise RuntimeError("experiment inputs changed")
    valid = all(item["passed"] for item in pairs + reproduction)
    stop_pass = both_stop_cases_pass(summaries)
    result = {
        "schema": SCHEMA, "episodes": summaries, "pairs": pairs,
        "old_zero_reproduction": reproduction, "comparison_valid": valid,
        "actual_control_transitions": sum(s["actual_transitions"] for s in summaries),
        "actual_physics_substeps": sum(s["actual_observed_physics_substeps"] for s in summaries),
        "input_unchanged": unchanged, "frozen77_unchanged": True,
        "candidate_passed_both_stop_cases": valid and stop_pass,
        "adopted_as_default": False, "full_driving_objective_complete": False,
    }
    write_json(args.output / "summary.json", result)
    write_manifest(args.output)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
