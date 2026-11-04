"""Candidate83: compare legacy/coherent sensor packets with unchanged legacy fusion.

Only the 12-case development preflight is authorized here. No full matrix,
training, policy call, archived rerun, resume or default-controller replacement.
Run under the separately supervised serial CPU/resource unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np

from wheel_legged_control.d1.coherent_sensor83 import (
    D1CoherentSensorStateSource83,
    control_state_times83,
)
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.env import encode_d1_observation
from wheel_legged_control.d1.hierarchical import D1LQRVMCController
from wheel_legged_control.d1.linear_model import identify_sagittal_model
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.terrain_tracking_env import terrain_normal_to_rpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import estimator_candidate49 as original49

SCHEMA = "d1-coherent-sampling83-preflight-v1"
ARMS = ("legacy_none", "coherent_none")
REFERENCE = ROOT / "results/estimator49_preflight_20261022_01"
OLD_PREREG = ROOT / "docs/experiments49/estimator_latency49_preregistered.json"
OLD_REPORT = ROOT / "reports/estimator49_results_20261022.json"
NEW_SOURCES = (
    "scripts/coherent_sensor83.py",
    "src/wheel_legged_control/d1/coherent_sensor83.py",
    "tests/test_coherent_sensor83.py",
    "tests/test_coherent_sensor83_runner.py",
)
OUTPUT_LIMIT = 32 * 1024**2
FINAL_RESERVE = 128 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_cases() -> list[dict]:
    return [
        {"terrain": "flat", "noise": noise, "seed": seed, "delay_steps": 2, "arm": arm}
        for noise in ("noise", "bias")
        for seed in (17, 29, 43)
        for arm in ARMS
    ]


def case_name(case: dict) -> str:
    return original49.case_name(case)


def protocol() -> dict:
    return {
        "schema": SCHEMA,
        "phase": "preflight",
        "full_authorized": False,
        "sole_variable": "sensor sampling phase: legacy_mixed versus private-copy post-step "
        "forward",
        "arms": list(ARMS),
        "planned_cases": experiment_cases(),
        "fixed": {
            "plant_sampling_mode": "legacy_mixed",
            "observer_wheel_center_jacobians": False,
            "postfusion_compensation": "none",
            "duration_s": 4.0,
            "control_dt_s": 0.01,
            "physics_steps_per_control": 5,
            "controller": "unmodified D1LQRVMCController",
            "noise_delay_sample_order_and_ground_reference": "original candidate49 run_case",
            "external_wrench": "zero; plant.step called without push arguments",
            "truth_metrics_and_termination": "unchanged original legacy truth source in both arms",
        },
        "limits": {
            "cpu_s": 120,
            "wall_s": 180,
            "output_bytes": OUTPUT_LIMIT,
            "task_control_steps": 4800,
            "identification_auxiliary_plant_steps": 511,
            "total_plant_steps": 5311,
            "task_explicit_integration_substeps": 24000,
            "identification_explicit_integration_substeps": 2555,
            "total_explicit_integration_substeps": 26555,
            "candidate_sampling_copy_forward_calls": 2406,
            "sensor_rne_post_constraint_calls": 4812,
            "task_plant_constructions": 12,
            "identification_plant_constructions": 1,
            "total_plant_constructions": 13,
            "online_observer_constructions": 12,
            "replay_observer_constructions": 12,
            "training_updates": 0,
            "policy_calls": 0,
        },
        "cold_initialization": "fresh process; identify_sagittal_model cache must have zero "
        "hits/misses/entries before the first case, one miss and eleven "
        "hits after twelve; 500 settle + 11 finite-difference transitions "
        "are auxiliary plant.step calls, not training/task samples",
        "count_boundary": "task integration plus cold identification integration are explicit; "
        "sensor refresh calls and observer FK use separate ledgers; compiler "
        "internals, other native calls and total native calls remain unknown",
        "gate": "original49.evaluate_gate preflight predicate with explicit arm-label mapping; "
        "all 12 terminal executions/replays/provenance valid; completion >= baseline, "
        "fallen <= baseline, >=1 common full pair, paired mean RMSE <=1.05 baseline, "
        "and completion gain >0 OR paired RMSE <=0.98 baseline; also all six original none "
        "CSV common fields/core metrics reproduce exactly and source-copy invariants pass",
        "offline_gate_scope": "49/50 offline pitch gate qualified a different postfusion CV "
        "variable; no archived pitch improvement is claimed or reused "
        "to admit this sampling fix",
        "full_status": "NOT_AUTHORIZED; passing preflight alone never authorizes a full run",
        "failure_handling": "retain fallen terminals and every partial; no overwrite/resume/retry; "
        "in-flight interrupted counts unknown with saved-prefix lower bounds",
        "resources": {
            "serial": True,
            "Nice": 10,
            "threads": 1,
            "MemoryHigh": "2G",
            "MemoryMax": "3G",
            "MemorySwapMax": 0,
            "start_available_gib": 4,
            "run_reserve_gib": 2,
        },
    }


def source_binding() -> dict[str, str]:
    """Verify the old closure; bind six archived baselines and all new code."""
    prereg = json.loads(OLD_PREREG.read_text())
    previous = json.loads((REFERENCE / "summary.json").read_text())
    if (
        previous["status"] != "complete"
        or previous["phase"] != "preflight"
        or previous["preregistration_sha256"] != sha256(OLD_PREREG)
        or previous["preregistration"] != prereg
        or previous["script_sha256"] != sha256(ROOT / "scripts/estimator_candidate49.py")
        or not previous["source_closure_reverified_at_end"]
    ):
        raise ValueError("original49 preflight/source binding is incomplete or changed")
    bound = dict(prereg["source_sha256"])
    report = json.loads(OLD_REPORT.read_text())
    archived = report["artifact_sha256"]
    baseline_paths = [REFERENCE / "summary.json"]
    for case in experiment_cases()[::2]:
        stem = case_name({**case, "arm": "none"})
        baseline_paths.extend((REFERENCE / (stem + ".csv"), REFERENCE / (stem + ".json")))
    for path in baseline_paths:
        relative = str(path.relative_to(ROOT))
        bound[relative] = archived[relative]
    for relative, expected in bound.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("frozen source/archive changed: " + relative)
    for path in (OLD_PREREG, OLD_REPORT, *(ROOT / name for name in NEW_SOURCES)):
        bound[str(path.relative_to(ROOT))] = sha256(path)
    return dict(sorted(bound.items()))


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(text)


def build_plan(output: Path) -> None:
    plan = {"protocol": protocol(), "source_sha256": source_binding()}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)


def verify_plan(plan: dict) -> None:
    if plan.get("protocol") != protocol():
        raise ValueError("plan differs from the fixed preflight protocol; full is not authorized")
    if plan.get("source_sha256") != source_binding():
        raise ValueError("plan source closure differs from current/executed sources")


class Budget:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.cpu_started = time.process_time()
        self.wall_started = time.monotonic()

    def size(self) -> int:
        return sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())

    def check(self, *, controls: int = 0, reserve_bytes: int = 65536) -> None:
        if (
            time.process_time() - self.cpu_started >= 120
            or time.monotonic() - self.wall_started >= 180
        ):
            raise TimeoutError("candidate83 fixed CPU/wall budget exhausted")
        if controls > 4800:
            raise RuntimeError("candidate83 control budget exceeded")
        if self.size() + reserve_bytes > OUTPUT_LIMIT - FINAL_RESERVE:
            raise RuntimeError("candidate83 output budget exhausted")

    def final_json(self, path: Path, value: object) -> None:
        encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        if self.size() + len(encoded) > OUTPUT_LIMIT:
            raise RuntimeError("final receipt would exceed candidate83 output budget")
        with path.open("xb") as handle:
            handle.write(encoded)


def source_factory(case: dict, plant):
    """Explicit source selection; no mutable global class replacement."""
    if case["arm"] not in ARMS:
        raise ValueError("unregistered source arm")
    source_type = D1SensorStateSource if case["arm"] == ARMS[0] else D1CoherentSensorStateSource83
    return source_type(
        plant,
        noise=original49.NOISE_PROFILES[case["noise"]],
        initial_position=original49.INITIAL_POSITION,
        delay_steps=case["delay_steps"],
    )


def identification_cache() -> dict:
    return dict(identify_sagittal_model.cache_info()._asdict())


def require_cold_identification_cache() -> dict:
    before = identification_cache()
    if any(before[name] != 0 for name in ("hits", "misses", "currsize")):
        raise ValueError(
            "candidate83 requires a fresh process with an untouched identification cache"
        )
    return before


def identification_accounting(before: dict, after: dict, *, first_case: bool) -> dict:
    expected_before = (
        {"hits": 0, "misses": 0, "currsize": 0}
        if first_case
        else {
            "hits": before["hits"],
            "misses": 1,
            "currsize": 1,
        }
    )
    expected_after = {"hits": before["hits"] + int(not first_case), "misses": 1, "currsize": 1}
    if any(before[name] != value for name, value in expected_before.items()) or any(
        after[name] != value for name, value in expected_after.items()
    ):
        raise ValueError("unexpected cold identification/cache path")
    return {
        "cache_before": before,
        "cache_after": after,
        "cold_identification_calls": int(first_case),
        "auxiliary_plant_steps": 511 * int(first_case),
        "auxiliary_explicit_integration_substeps": 2555 * int(first_case),
        "auxiliary_plant_constructions": int(first_case),
        "count_method": "frozen identify_sagittal_model: round(5/.01)=500 settle + 8 state "
        "probes +2 input probes +1 affine probe; verified cache miss/hit path",
    }


def observer_fk_accounting(packets: list, delay: int) -> dict:
    updates_each = max(0, len(packets) - 1 - delay)
    snapshots_each = 1 + 2 * updates_each
    contact_jacobians_each = int(np.count_nonzero(packets[0].wheel_contact)) + 2 * sum(
        int(np.count_nonzero(packet.wheel_contact)) for packet in packets[1 : updates_each + 1]
    )
    return {
        "observer_count": 2,
        "updates_each": updates_each,
        "snapshots_each": snapshots_each,
        "mj_kinematics": 2 * snapshots_each,
        "mj_comPos": 2 * snapshots_each,
        "mj_jacBodyCom": 8 * snapshots_each,
        "contact_mj_jac": 2 * contact_jacobians_each,
        "count_method": "derived from bound legacy observer source and all recorded packets; "
        "one reset snapshot, two per update, online plus full replay; "
        "not instrumentation of all native calls",
    }


def integration_signature(plant) -> dict:
    return {
        "time": plant.data.time,
        **{
            name: getattr(plant.data, name).copy()
            for name in (
                "qpos",
                "qvel",
                "act",
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
            )
        },
    }


def check_source_only_read(before: dict, plant) -> None:
    after = integration_signature(plant)
    if any(not np.array_equal(value, after[key]) for key, value in before.items()):
        raise ValueError("sensor read changed live integration state/input/warmstart")


def baseline_reproduction(case: dict, rows: list[dict], metrics: dict, reference=REFERENCE) -> dict:
    if case["arm"] != ARMS[0]:
        return {"applicable": False, "passed": True}
    stem = case_name({**case, "arm": "none"})
    with (reference / (stem + ".csv")).open() as handle:
        reader = csv.DictReader(handle)
        columns, archived_rows = reader.fieldnames, list(reader)
    archived_metrics = json.loads((reference / (stem + ".json")).read_text())
    equal = len(rows) == len(archived_rows)
    comparisons, maximum = 0, 0.0
    for current, old in zip(rows, archived_rows):
        for column in columns:
            if column not in current:
                equal = False
                continue
            comparisons += 1
            if column == "applied_compensation_status":
                equal = equal and current[column] == old[column]
            else:
                gap = abs(float(current[column]) - float(old[column]))
                maximum = max(maximum, gap)
                equal = equal and gap == 0.0
    keys = (
        "steps",
        "termination_reason",
        "tracking_rmse_mps",
        "velocity_estimation_rmse_mps",
        "compensation_status_counts",
        "native_steps",
    )
    equal = equal and all(metrics[key] == archived_metrics[key] for key in keys)
    return {
        "applicable": True,
        "passed": bool(equal),
        "rows": len(rows),
        "reference_rows": len(archived_rows),
        "common_columns": columns,
        "value_comparisons": comparisons,
        "maximum_absolute_numeric_difference": maximum,
        "reference_case": stem,
    }


def run_case(case: dict, output: Path, budget: Budget, completed_controls: int) -> dict:
    """Minimal candidate49 loop copy; only source_factory selects the new arm."""
    budget.check(controls=completed_controls)
    plant = D1Plant(training_terrain=original49.TERRAINS[case["terrain"]])
    if plant.physics_steps != 5 or plant.sampling_mode != "legacy_mixed":
        raise ValueError("original plant phase/physics-step contract changed")
    plant.reset(base_position=np.asarray(original49.INITIAL_POSITION))
    cache_before = identification_cache()
    controller = D1LQRVMCController(plant)
    initialization = identification_accounting(
        cache_before,
        identification_cache(),
        first_case=case == experiment_cases()[0],
    )
    truth_source = D1MujocoTruthStateSource(plant)
    source = source_factory(case, plant)
    if source.estimator.wheel_center_jacobians:
        raise ValueError("observer Jacobian branch changed")
    truth = truth_source.reset()
    before = integration_signature(plant)
    raw = source.reset(seed=case["seed"])
    check_source_only_read(before, plant)
    packets, raw_states, rows, counts = [source.latest_measurement], [raw], [], Counter()
    started, termination = time.monotonic(), "time_limit"
    stem = output / case_name(case)
    with (
        stem.with_suffix(".csv").open("x") as csv_file,
        Path(str(stem) + "_measurements.jsonl").open("x") as packet_file,
        Path(str(stem) + "_times.jsonl").open("x") as time_file,
    ):

        def save_packet(packet):
            packet_file.write(
                json.dumps(
                    {
                        field.name: np.asarray(getattr(packet, field.name)).tolist()
                        for field in fields(packet)
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            packet_file.flush()

        save_packet(packets[0])
        writer = None
        for step in range(400):
            budget.check(controls=completed_controls + len(rows))
            result = original49.apply_candidate(raw, "none")
            state = result.control_state
            timing = control_state_times83(
                raw,
                result,
                coherent_measurement_packet=case["arm"] == ARMS[1],
            )
            counts[result.status] += 1
            ground = source.ground_reference(state)
            roll, pitch = terrain_normal_to_rpy(
                -np.tan(ground.pitch_rad),
                np.tan(ground.roll_rad),
                state.base_rpy[2],
            )
            fraction = float(np.clip((step * plant.control_dt - 0.3) / 0.3, 0, 1))
            velocity_command = 0.35 * fraction**2 * (3 - 2 * fraction)
            command = D1Command(
                forward_velocity_mps=velocity_command,
                base_height_m=ground.height_m + 0.455,
                roll_rad=roll,
                pitch_rad=pitch,
            )
            observation = np.concatenate(
                (
                    encode_d1_observation(
                        state=state,
                        command=command,
                        baseline_longitudinal_force_n=controller.last_longitudinal_force_n,
                        previous_applied_action=np.zeros(2),
                    ),
                    [pitch, roll],
                )
            )
            if observation.shape != (44,) or not np.isfinite(observation).all():
                raise ValueError("original observation contract failed")
            plant.step(controller.compute(command, state, residual_force_n=np.zeros(2)))
            truth = truth_source.read()
            before = integration_signature(plant)
            raw = source.read()
            check_source_only_read(before, plant)
            packets.append(source.latest_measurement)
            raw_states.append(raw)
            save_packet(packets[-1])
            if case["arm"] == ARMS[1]:
                sample = source.measurements.sample_provenance
                if (
                    sample["sample_time_s"] != source.latest_measurement.time_s
                    or sample["sampling_forward_calls"] != len(packets)
                    or sample["sensor_rne_post_constraint_calls"] != len(packets)
                ):
                    raise ValueError("coherent sample clock/count invariant failed")
            time_file.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "control_input": timing,
                        "post_step_packet_time_s": source.latest_measurement.time_s,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            time_file.flush()
            actual_ground = plant.training_ground_reference(*truth.base_position[:2])
            row = {
                "step": step + 1,
                "time_s": (step + 1) * plant.control_dt,
                "command_velocity_mps": velocity_command,
                "truth_velocity_body_x_mps": float(truth.base_linear_velocity_body[0]),
                "truth_pitch_rad": float(truth.base_rpy[1]),
                "truth_clearance_m": float(truth.base_position[2] - actual_ground.height_m),
                **original49.archived_raw_values(raw),
                "applied_compensation_status": result.status,
                "applied_compensation_horizon_s": result.applied_horizon_s,
                "applied_measurement_age_s": state.age_s,
            }
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            if row["truth_clearance_m"] < 0.22 or np.max(np.abs(truth.base_rpy[:2])) > 0.85:
                termination = "fallen"
                break
            if np.max(np.abs(truth.base_position[:2])) > 5.5:
                termination = "out_of_bounds"
                break
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **original49.packet_arrays(packets))
    payload = buffer.getvalue()
    budget.check(controls=completed_controls + len(rows), reserve_bytes=len(payload) + 65536)
    with Path(str(stem) + "_measurements.npz").open("xb") as handle:
        handle.write(payload)
    replay_exact = True
    # Consume every saved packet even after an equality failure, so the complete
    # replay/FK ledger never silently describes only an all(...) short-circuit.
    for replayed, actual in zip(
        original49.replay_states(packets, case["delay_steps"]), raw_states, strict=True
    ):
        if any(
            not np.array_equal(getattr(replayed, field.name), getattr(actual, field.name))
            for field in fields(actual)
        ):
            replay_exact = False
    metrics = {
        **case,
        "execution_status": "COMPLETE_TERMINAL",
        "steps": len(rows),
        "duration_s": len(rows) * plant.control_dt,
        "termination_reason": termination,
        "full400": len(rows) == 400 and termination == "time_limit",
        "tracking_rmse_mps": original49.rmse(
            [r["truth_velocity_body_x_mps"] - r["command_velocity_mps"] for r in rows]
        ),
        "velocity_estimation_rmse_mps": original49.rmse(
            [r["estimate_velocity_body_x_mps"] - r["truth_velocity_body_x_mps"] for r in rows]
        ),
        "compensation_status_counts": dict(counts),
        "replay_bitwise_equal": replay_exact,
        "provenance_invariants_pass": True,
        "coherent_sample_invariants_pass": True,
        "wall_time_s": time.monotonic() - started,
        "native_steps": len(rows) * plant.physics_steps,
        "sampling_copy_forward_calls": (
            source.measurements.sampling_refresh_count if case["arm"] == ARMS[1] else 0
        ),
        "sensor_rne_post_constraint_calls": len(packets),
        "training_updates": 0,
        "policy_calls": 0,
        "object_constructions": {
            "plant": 1,
            "online_observer": 1,
            "replay_observer": 1,
            "extra_sampling_mjdata": int(case["arm"] == ARMS[1]),
        },
        "initialization_accounting": initialization,
        "observer_fk_accounting": observer_fk_accounting(packets, case["delay_steps"]),
        "observer_wheel_center_jacobians": source.estimator.wheel_center_jacobians,
    }
    metrics["baseline_reproduction"] = baseline_reproduction(case, rows, metrics)
    return metrics


def evaluate_gate(cases: list[dict]) -> dict:
    identities = [{name: row.get(name) for name in experiment_cases()[0]} for row in cases]
    if identities != experiment_cases():
        return {
            "passed": False,
            "original_gate_passed": False,
            "reason": "missing, duplicate, or reordered registered cases",
            "full_authorized": False,
        }
    # The frozen pure predicate uses CV as a label. This mapping does not run CV
    # or change any executable/source class; the report exposes the mapping.
    labels = {ARMS[0]: "none", ARMS[1]: "constant_velocity"}
    old = original49.evaluate_gate(
        "preflight",
        [{**row, "arm": labels[row["arm"]]} for row in cases],
    )
    reverse = {value: key for key, value in labels.items()}
    integrity = all(
        row["coherent_sample_invariants_pass"] and row["baseline_reproduction"]["passed"]
        for row in cases
    )
    passed = bool(old["passed"] and integrity)
    return {
        **old,
        "passed": passed,
        "original_gate_passed": old["passed"],
        "additional_source_and_baseline_integrity_pass": integrity,
        "completions": {
            delay: {reverse[arm]: count for arm, count in values.items()}
            for delay, values in old["completions"].items()
        },
        "termination_counts_by_arm": {
            reverse[arm]: values for arm, values in old["termination_counts_by_arm"].items()
        },
        "failed_cases": [
            case_name(row)
            for row in cases
            if row["steps"] != 400 or row["termination_reason"] != "time_limit"
        ],
        "common_complete_pair_ids": [
            {name: left[name] for name in ("terrain", "noise", "seed", "delay_steps")}
            for left, right in zip(cases[::2], cases[1::2], strict=True)
            if all(
                row["steps"] == 400 and row["termination_reason"] == "time_limit"
                for row in (left, right)
            )
        ],
        "gate_implementation": "frozen scripts/estimator_candidate49.py:evaluate_gate(preflight)",
        "gate_argument_label_mapping_only": labels,
        "decision": "PREFLIGHT_PASS_FULL_NOT_AUTHORIZED" if passed else "REJECTED",
        "performance_qualified": False,
        "full_authorized": False,
    }


def execute(plan_path: Path, output: Path) -> None:
    if not __debug__:
        raise RuntimeError("python -O is forbidden for the inherited assertion-based guards")
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(output)
    plan_hash = sha256(plan_path)
    plan = json.loads(plan_path.read_text())
    attempt = {
        "schema": SCHEMA,
        "phase": "preflight",
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan_hash,
        "planned_cases": experiment_cases(),
        "resume_permitted": False,
        "full_status": "NOT_AUTHORIZED",
    }
    write_json(output / "attempt.json", attempt)
    cases, active_case = [], None

    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}; no resume or retry")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        verify_plan(plan)
        budget.check()
        cold_cache = require_cold_identification_cache()
        for case in experiment_cases():
            active_case = case
            controls = sum(row["steps"] for row in cases)
            result = run_case(case, output, budget, controls)
            write_json(output / (case_name(case) + ".json"), result)
            cases.append(result)
            active_case = None
            print(case_name(case), result["termination_reason"], result["steps"], flush=True)
        gate = evaluate_gate(cases)
        verify_plan(plan)
        if sha256(plan_path) != plan_hash:
            raise ValueError("plan changed during execution")
        final_cache = identification_cache()
        if final_cache["hits"] != 11 or final_cache["misses"] != 1 or final_cache["currsize"] != 1:
            raise ValueError("final identification cache does not match the registered cold path")
        budget.check(controls=sum(row["steps"] for row in cases))
        artifacts = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
        budget.final_json(
            output / "summary.json",
            {
                **attempt,
                "status": "COMPLETE",
                "cases": cases,
                "gate": gate,
                "source_closure_reverified_at_end": True,
                "source_sha256": plan["source_sha256"],
                "artifact_sha256": artifacts,
                "executed_control_steps": sum(r["steps"] for r in cases),
                "task_explicit_integration_substeps": sum(r["native_steps"] for r in cases),
                "identification_auxiliary_plant_steps": 511,
                "identification_explicit_integration_substeps": 2555,
                "total_plant_steps": sum(r["steps"] for r in cases) + 511,
                "total_explicit_integration_substeps": sum(r["native_steps"] for r in cases) + 2555,
                "identification_cache_before": cold_cache,
                "identification_cache_after": final_cache,
                "object_constructions": {
                    "task_plants": 12,
                    "identification_plants": 1,
                    "online_observers": 12,
                    "replay_observers": 12,
                    "extra_sampling_mjdata": 6,
                },
                "sampling_copy_forward_calls": sum(r["sampling_copy_forward_calls"] for r in cases),
                "sensor_rne_post_constraint_calls": sum(
                    r["sensor_rne_post_constraint_calls"] for r in cases
                ),
                "observer_fk_calls_by_source_formula": {
                    name: sum(r["observer_fk_accounting"][name] for r in cases)
                    for name in ("mj_kinematics", "mj_comPos", "mj_jacBodyCom", "contact_mj_jac")
                },
                "compiler_internal_native_calls": None,
                "total_native_calls": None,
                "training_updates": 0,
                "policy_calls": 0,
                "partial_cases": 0,
                "fallen_terminal_cases": sum(r["termination_reason"] == "fallen" for r in cases),
                "full400_cases": sum(r["full400"] for r in cases),
                "cpu_s": time.process_time() - budget.cpu_started,
                "wall_s": time.monotonic() - budget.wall_started,
            },
        )
        print(json.dumps(gate, indent=2), flush=True)
    except BaseException:
        prefix = 0
        if active_case is not None:
            path = output / (case_name(active_case) + ".csv")
            if path.exists():
                with path.open() as handle:
                    prefix = sum(None not in row.values() for row in csv.DictReader(handle))
        exact = active_case is None
        finished = sum(r["steps"] for r in cases)
        budget.final_json(
            output / "failure.json",
            {
                **attempt,
                "status": "FAILED_OR_PARTIAL",
                "finished_cases": cases,
                "active_case": active_case,
                "active_case_logged_prefix": prefix,
                "executed_control_steps": finished if exact else None,
                "task_explicit_integration_substeps": 5 * finished if exact else None,
                "identification_cache_at_failure": identification_cache(),
                "auxiliary_plant_steps": (
                    sum(r["initialization_accounting"]["auxiliary_plant_steps"] for r in cases)
                    if exact
                    else None
                ),
                "total_plant_steps": None,
                "total_explicit_integration_substeps": None,
                "control_step_lower_bound": finished + prefix,
                "counts_exact": exact,
                "training_updates": 0,
                "policy_calls": 0,
                "traceback": traceback.format_exc(),
                "not_run_cases": [
                    case for case in experiment_cases()[len(cases) :] if case != active_case
                ],
            },
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("plan")
    build.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("preflight")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        build_plan(args.output)
    else:
        execute(args.plan, args.output)


if __name__ == "__main__":
    main()
