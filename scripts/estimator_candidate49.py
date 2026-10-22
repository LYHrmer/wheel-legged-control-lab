"""Preregistered, development-only latency compensation after sensor fusion.

This is separate from the frozen B22 training/runtime protocol. It imports the
unchanged measurement observer and bounded compensator; no controller, sensor
noise, contact predictor, or rejection threshold is retuned. Run each phase in
a fresh directory under the resource-limited serial launcher.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.env import encode_d1_observation
from wheel_legged_control.d1.hierarchical import D1LQRVMCController
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import (
    D1ProprioceptiveEstimator,
    D1SensorMeasurements,
    D1SensorStateSource,
)
from wheel_legged_control.d1.state_estimation import (
    D1MujocoTruthStateSource,
    prepare_d1_control_state,
)
from wheel_legged_control.d1.terrain_tracking_env import terrain_normal_to_rpy

# Reuse the original experiment's complete, fixed parameter definitions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_d1_sensor_estimation import INITIAL_POSITION, NOISE_PROFILES, TERRAINS

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs/experiments49/estimator_latency49_preregistered.json"
ARMS = ("none", "constant_velocity")
SEEDS = (17, 29, 43)
CONTACT_FIELDS = (
    "wheel_contact", "wheel_contact_point", "wheel_contact_normal",
    "wheel_contact_jacobian", "undesired_ground_contacts",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_sources(prereg: dict) -> None:
    for relative, expected in prereg["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"preregistered source changed: {relative}")


def rmse(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def experiment_cases(phase: str) -> list[dict]:
    terrains = ("flat",) if phase == "preflight" else tuple(TERRAINS)
    noises = ("noise", "bias") if phase == "preflight" else tuple(NOISE_PROFILES)
    delays = (0, 2) if phase == "full" else (2,)
    arms = ("offline_pair",) if phase == "offline" else ARMS
    return [
        {"terrain": terrain, "noise": noise, "seed": seed, "delay_steps": delay, "arm": arm}
        for terrain in terrains for noise in noises for seed in SEEDS
        for delay in delays for arm in arms
    ]


def case_name(case: dict) -> str:
    return (
        f"{case['terrain']}_{case['noise']}_delay{case['delay_steps']}_"
        f"seed{case['seed']}_{case['arm']}"
    )


def apply_candidate(raw, arm: str):
    result = prepare_d1_control_state(raw, latency_compensation=arm)
    control = result.control_state
    # Predictions retain measurement provenance and all unobserved contact geometry.
    assert control.age_s == raw.age_s
    assert control.measurement_time_s == raw.measurement_time_s
    assert control.control_time_s == raw.control_time_s
    for name in CONTACT_FIELDS:
        assert np.array_equal(getattr(control, name), getattr(raw, name)), name
    return result


def packet_arrays(packets: list) -> dict:
    return {field.name: np.asarray([getattr(p, field.name) for p in packets])
            for field in fields(D1SensorMeasurements)}


def packets_from_archive(path: Path) -> list:
    with np.load(path, allow_pickle=False) as archive:
        names = {field.name for field in fields(D1SensorMeasurements)}
        if set(archive.files) != names:
            raise ValueError("measurement archive contains missing or non-sensor fields")
        return [D1SensorMeasurements(**{
            name: (int(archive[name][i]) if name == "sequence"
                   else float(archive[name][i]) if name == "time_s" else archive[name][i])
            for name in names
        }) for i in range(len(archive["sequence"]))]


def replay_states(packets: list, delay: int):
    observer = D1ProprioceptiveEstimator()
    fused = observer.reset(packets[0], initial_position=INITIAL_POSITION)
    yield fused
    for index, current in enumerate(packets[1:], start=1):
        delayed = packets[max(0, index - delay)]
        if index > delay:
            fused = observer.update(delayed)
        yield replace(fused, sequence=current.sequence, control_time_s=current.time_s,
                      measurement_time_s=delayed.time_s)


def archived_raw_values(raw) -> dict:
    return {
        "estimate_x_m": float(raw.base_position[0]),
        "estimate_height_m": float(raw.base_position[2]),
        "estimate_velocity_body_x_mps": float(raw.base_linear_velocity_body[0]),
        "estimate_velocity_world_z_mps": float(raw.base_linear_velocity_world[2]),
        "estimate_pitch_rad": float(raw.base_rpy[1]),
        "measurement_time_s": raw.measurement_time_s,
        "measurement_age_s": raw.age_s,
    }


def offline_case(case: dict, archive: Path, output: Path, archived_summary: dict) -> dict:
    stem = (f"closed_loop_{case['terrain']}_{case['noise']}_"
            f"delay2_seed{case['seed']}")
    csv_path, packet_path = archive / f"{stem}.csv", archive / f"{stem}_measurements.npz"
    with csv_path.open() as handle:
        reference_rows = list(csv.DictReader(handle))
    packets = packets_from_archive(packet_path)
    states = replay_states(packets, 2)
    next(states)  # The archive CSV starts after the first control step.
    rows, counts, max_difference = [], Counter(), 0.0
    for reference, raw in zip(reference_rows, states, strict=True):
        result = apply_candidate(raw, "constant_velocity")
        counts[result.status] += 1
        max_difference = max(max_difference, max(
            abs(value - float(reference[name])) for name, value in archived_raw_values(raw).items()
        ), abs(raw.control_time_s - float(reference["time_s"])))
        truth_pitch = float(reference["truth_pitch_rad"])
        rows.append({
            "time_s": raw.control_time_s,
            "measurement_time_s": raw.measurement_time_s,
            "state_age_s": raw.age_s,
            "raw_pitch_error_rad": float((raw.base_rpy[1] - truth_pitch + np.pi) %
                                         (2 * np.pi) - np.pi),
            "candidate_pitch_error_rad": float((result.control_state.base_rpy[1] - truth_pitch +
                                               np.pi) % (2 * np.pi) - np.pi),
            "compensation_status": result.status,
            "compensation_horizon_s": result.applied_horizon_s,
        })
    if len(rows) != archived_summary["steps"]:
        raise ValueError("archive length differs from original summary")
    with (output / f"{case_name(case)}.csv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        **case, "source_csv_sha256": sha256(csv_path),
        "source_measurements_sha256": sha256(packet_path),
        "archive_termination_reason": archived_summary["termination_reason"],
        "archive_steps": len(rows), "archive_replay_max_abs_difference": max_difference,
        "raw_pitch_rmse_rad": rmse([r["raw_pitch_error_rad"] for r in rows]),
        "candidate_pitch_rmse_rad": rmse([r["candidate_pitch_error_rad"] for r in rows]),
        "compensation_status_counts": dict(counts), "provenance_invariants_pass": True,
        "new_physics_steps": 0,
    }


def run_case(case: dict, output: Path) -> dict:
    """Original four-second control task, varying only post-fusion treatment."""
    plant = D1Plant(training_terrain=TERRAINS[case["terrain"]])
    plant.reset(base_position=np.asarray(INITIAL_POSITION))
    controller = D1LQRVMCController(plant)
    truth_source = D1MujocoTruthStateSource(plant)
    source = D1SensorStateSource(
        plant, noise=NOISE_PROFILES[case["noise"]], initial_position=INITIAL_POSITION,
        delay_steps=case["delay_steps"],
    )
    truth, raw = truth_source.reset(), source.reset(seed=case["seed"])
    packets, raw_states, rows, counts = [source.latest_measurement], [raw], [], Counter()
    started, termination = time.perf_counter(), "time_limit"
    stem = output / case_name(case)
    with stem.with_suffix(".csv").open("x") as csv_file, \
            Path(str(stem) + "_measurements.jsonl").open("x") as packet_file:
        def save_packet(packet):
            packet_file.write(json.dumps({
                field.name: np.asarray(getattr(packet, field.name)).tolist()
                for field in fields(packet)
            }, allow_nan=False) + "\n")
            packet_file.flush()

        save_packet(packets[0])
        writer = None
        for step in range(400):
            result = apply_candidate(raw, case["arm"])
            state = result.control_state
            counts[result.status] += 1
            ground = source.ground_reference(state)
            roll, pitch = terrain_normal_to_rpy(
                -np.tan(ground.pitch_rad), np.tan(ground.roll_rad), state.base_rpy[2]
            )
            fraction = float(np.clip((step * plant.control_dt - 0.3) / 0.3, 0, 1))
            velocity_command = 0.35 * fraction**2 * (3 - 2 * fraction)
            command = D1Command(forward_velocity_mps=velocity_command,
                                base_height_m=ground.height_m + 0.455,
                                roll_rad=roll, pitch_rad=pitch)
            observation = np.concatenate((encode_d1_observation(
                state=state, command=command,
                baseline_longitudinal_force_n=controller.last_longitudinal_force_n,
                previous_applied_action=np.zeros(2),
            ), [pitch, roll]))
            assert observation.shape == (44,) and np.isfinite(observation).all()
            plant.step(controller.compute(command, state, residual_force_n=np.zeros(2)))
            # Truth is queried exclusively after control, for reporting/termination.
            truth, raw = truth_source.read(), source.read()
            packets.append(source.latest_measurement)
            raw_states.append(raw)
            save_packet(packets[-1])
            actual_ground = plant.training_ground_reference(*truth.base_position[:2])
            row = {
                "step": step + 1, "time_s": (step + 1) * plant.control_dt,
                "command_velocity_mps": velocity_command,
                "truth_velocity_body_x_mps": float(truth.base_linear_velocity_body[0]),
                "truth_pitch_rad": float(truth.base_rpy[1]),
                "truth_clearance_m": float(truth.base_position[2] - actual_ground.height_m),
                **archived_raw_values(raw),
                "applied_compensation_status": result.status,
                "applied_compensation_horizon_s": result.applied_horizon_s,
                "applied_measurement_age_s": state.age_s,
            }
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()  # Preserve executed prefixes if the resource monitor stops the unit.
            if row["truth_clearance_m"] < 0.22 or np.max(np.abs(truth.base_rpy[:2])) > 0.85:
                termination = "fallen"
                break
            if np.max(np.abs(truth.base_position[:2])) > 5.5:
                termination = "out_of_bounds"
                break
    np.savez_compressed(Path(str(stem) + "_measurements.npz"), **packet_arrays(packets))
    replay_exact = all(
        np.array_equal(getattr(replayed, field.name), getattr(actual, field.name))
        for replayed, actual in zip(replay_states(packets, case["delay_steps"]),
                                   raw_states, strict=True)
        for field in fields(actual)
    )
    return {
        **case, "steps": len(rows), "duration_s": len(rows) * plant.control_dt,
        "termination_reason": termination,
        "tracking_rmse_mps": rmse([r["truth_velocity_body_x_mps"] -
                                   r["command_velocity_mps"] for r in rows]),
        "velocity_estimation_rmse_mps": rmse([r["estimate_velocity_body_x_mps"] -
                                             r["truth_velocity_body_x_mps"] for r in rows]),
        "compensation_status_counts": dict(counts), "replay_bitwise_equal": replay_exact,
        "provenance_invariants_pass": True, "wall_time_s": time.perf_counter() - started,
        "native_steps": len(rows) * plant.physics_steps,
    }


def evaluate_gate(phase: str, cases: list[dict]) -> dict:
    expected = experiment_cases(phase)
    identities = [{key: row[key] for key in expected[0]} for row in cases]
    if identities != expected:
        return {"passed": False, "reason": "missing, duplicate, or reordered cases"}
    if phase == "offline":
        raw = float(np.mean([r["raw_pitch_rmse_rad"] for r in cases]))
        candidate = float(np.mean([r["candidate_pitch_rmse_rad"] for r in cases]))
        terrain_ratios = {
            terrain: float(np.mean([r["candidate_pitch_rmse_rad"] for r in cases
                                    if r["terrain"] == terrain]) /
                           np.mean([r["raw_pitch_rmse_rad"] for r in cases
                                    if r["terrain"] == terrain]))
            for terrain in TERRAINS
        }
        replay_pass = all(r["archive_replay_max_abs_difference"] <= 1e-12 and
                          r["provenance_invariants_pass"] for r in cases)
        return {
            "passed": bool(replay_pass and candidate <= 0.98 * raw and
                           max(terrain_ratios.values()) <= 1.05),
            "raw_mean_pitch_rmse_rad": raw, "candidate_mean_pitch_rmse_rad": candidate,
            "candidate_over_raw_pitch_ratio": candidate / raw,
            "terrain_pitch_ratios": terrain_ratios, "archive_replay_pass": replay_pass,
            "interpretation": "offline direction only; no closed-loop improvement claim",
        }
    pairs = list(zip(cases[::2], cases[1::2], strict=True))
    def complete(row):
        return row["steps"] == 400 and row["termination_reason"] == "time_limit"

    common = [(a, b) for a, b in pairs if complete(a) and complete(b)]
    raw_mean = float(np.mean([a["tracking_rmse_mps"] for a, _ in common])) if common else None
    candidate_mean = (float(np.mean([b["tracking_rmse_mps"] for _, b in common]))
                      if common else None)
    ratio = candidate_mean / raw_mean if raw_mean is not None and raw_mean > 0 else None
    ratios_by_delay = {}
    for delay in sorted({r["delay_steps"] for r in cases}):
        subset = [(a, b) for a, b in common if a["delay_steps"] == delay]
        numerator = float(np.mean([b["tracking_rmse_mps"] for _, b in subset])) if subset else None
        denominator = float(np.mean([a["tracking_rmse_mps"] for a, _ in subset])) if subset else None
        ratios_by_delay[str(delay)] = (numerator / denominator
                                       if denominator is not None and denominator > 0 else None)
    completions = {str(delay): {arm: sum(complete(r) for r in cases
                                        if r["delay_steps"] == delay and r["arm"] == arm)
                               for arm in ARMS}
                   for delay in sorted({r["delay_steps"] for r in cases})}
    delayed = completions["2"]
    gain = delayed["constant_velocity"] - delayed["none"]
    terminations = {arm: dict(Counter(r["termination_reason"] for r in cases if r["arm"] == arm))
                    for arm in ARMS}
    no_extra_falls = (terminations["constant_velocity"].get("fallen", 0) <=
                      terminations["none"].get("fallen", 0))
    integrity = all(r["replay_bitwise_equal"] and r["provenance_invariants_pass"] for r in cases)
    tracking_pass = all(value is not None and value <= 1.05
                        for value in ratios_by_delay.values())
    if phase == "preflight":
        passed = (integrity and no_extra_falls and gain >= 0 and tracking_pass and
                  (gain > 0 or ratio <= 0.98))
    else:
        passed = (integrity and gain >= 6 and tracking_pass and
                  all(value == 36 for value in completions["0"].values()))
    return {
        "passed": bool(passed), "completions": completions, "delay20_completion_gain": gain,
        "common_complete_pairs": len(common), "common_raw_tracking_rmse_mps": raw_mean,
        "common_candidate_tracking_rmse_mps": candidate_mean,
        "common_tracking_rmse_ratio": ratio, "replay_and_provenance_pass": integrity,
        "common_tracking_rmse_ratios_by_delay": ratios_by_delay,
        "termination_counts_by_arm": terminations, "no_extra_falls": no_extra_falls,
        "failed_cases": [case_name(r) for r in cases if not complete(r)],
        "interpretation": "development matrix; ideal-seed repeats are not independent samples",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("offline", "preflight", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path)
    args = parser.parse_args()
    prereg = json.loads(PREREG.read_text())
    prereg_hash, script_hash = sha256(PREREG), sha256(Path(__file__))
    verify_sources(prereg)
    if args.phase != "offline":
        if args.previous_summary is None:
            raise ValueError("a complete, passing previous-phase summary is required")
        previous = json.loads(args.previous_summary.read_text())
        expected_phase = "offline" if args.phase == "preflight" else "preflight"
        if not (previous["phase"] == expected_phase and previous["status"] == "complete" and
                previous["gate"]["passed"] and previous["preregistration_sha256"] == prereg_hash and
                previous["script_sha256"] == script_hash):
            raise ValueError("previous phase did not pass under the identical preregistration/code")
    args.output.mkdir(parents=True, exist_ok=False)
    attempt = {
        "phase": args.phase, "purpose": "development; no held-out or policy claims",
        "preregistration_sha256": prereg_hash, "script_sha256": script_hash,
        "preregistration": prereg, "planned_cases": experiment_cases(args.phase),
        "previous_summary_sha256": (sha256(args.previous_summary)
                                    if args.previous_summary else None),
    }
    write_json(args.output / "attempt.json", attempt)
    cases = []
    active_case = None

    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}; attempt is partial and cannot resume")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        archive = ROOT / prereg["archive"]
        original = json.loads((archive / "summary.json").read_text())
        for case in attempt["planned_cases"]:
            active_case = case
            if args.phase == "offline":
                archived = [r for r in original["cases"] if r["mode"] == "closed_loop" and
                            all(r[key] == case[key] for key in
                                ("terrain", "noise", "seed", "delay_steps"))]
                if len(archived) != 1:
                    raise ValueError("archive must contain exactly one matching case")
                result = offline_case(case, archive, args.output, archived[0])
            else:
                result = run_case(case, args.output)
            cases.append(result)
            active_case = None
            write_json(args.output / f"{case_name(case)}.json", result)
            print(case_name(case), "recorded", flush=True)
        gate = evaluate_gate(args.phase, cases)
        verify_sources(prereg)
        if sha256(PREREG) != prereg_hash or sha256(Path(__file__)) != script_hash:
            raise RuntimeError("registration or candidate code changed during this phase")
        write_json(args.output / "summary.json", {
            **attempt, "status": "complete", "cases": cases, "gate": gate,
            "source_closure_reverified_at_end": True,
            "executed_control_steps": sum(r.get("steps", 0) for r in cases),
            "executed_native_steps": sum(r.get("native_steps", 0) for r in cases),
        })
        print(json.dumps(gate, indent=2), flush=True)
    except BaseException:
        logged_prefix = 0
        if active_case is not None and args.phase != "offline":
            path = args.output / f"{case_name(active_case)}.csv"
            if path.exists():
                with path.open() as handle:
                    logged_prefix = sum(1 for _ in csv.DictReader(handle))
        completed_controls = sum(r.get("steps", 0) for r in cases)
        completed_native = sum(r.get("native_steps", 0) for r in cases)
        counts_known = active_case is None or args.phase == "offline"
        write_json(args.output / "failure.json", {
            **attempt, "status": "failed_or_partial", "finished_cases": cases,
            "active_case": active_case, "active_case_logged_control_prefix": logged_prefix,
            "completed_case_control_steps": completed_controls,
            "completed_case_native_steps": completed_native,
            "executed_control_steps": completed_controls if counts_known else None,
            "executed_native_steps": completed_native if counts_known else None,
            "control_step_lower_bound": completed_controls + logged_prefix,
            "counts_exact": counts_known,
            "traceback": traceback.format_exc(), "resume_permitted": False,
        })
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
