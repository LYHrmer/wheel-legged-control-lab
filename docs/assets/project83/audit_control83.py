"""Independent stdlib readback of the two registered control83 experiments.

No experiment module, NumPy, simulator, model, optimizer or policy is imported.
Numerical work reconstructs saved arrays/metrics only. Every output is exclusive.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import signal
import struct
import time
import zipfile
from collections import Counter
from pathlib import Path

REG82_SHA = "ea7847eafc8deca1b38879523708b963a3743b74cf04193d5ca916b0343f2b76"
MPC_SEEDS = (490101, 490102, 490103)
SENSOR_SEEDS = (17, 29, 43)
SCENARIOS = ("nominal", "mismatch_delay")
MPC_COLUMNS = (
    "time_s", "x", "pitch", "extension", "vx", "pitch_rate", "extension_rate",
    "wheel_force", "leg_force", "target_vx", "target_extension", "push", "reward",
)
MPC_GUARDS = (
    "velocity_rmse_mps", "pitch_rmse_deg", "max_abs_pitch_deg",
    "normalized_effort_rms", "wheel_force_rms_n", "leg_force_rms_n",
)
SENSOR_ARMS = ("legacy_none", "coherent_none")
REG49 = "reports/control_candidates49_preregistration.json"
REG82 = "docs/experiments82/delay_mpc82_preregistered.json"
OLD49 = "results/mpc49_preflight_20261022_01"
OLD82 = "results/delay_mpc82_20261103_01"
OLD_SENSOR = "results/estimator49_preflight_20261022_01"


class AuditError(ValueError):
    pass


class Audit:
    def __init__(self, root):
        self.root = root.resolve()
        self.inputs, self.gaps = {}, {}
        self.checks = 0
        self.cpu_start, self.wall_start = time.process_time(), time.monotonic()

    def limit(self):
        if time.process_time() - self.cpu_start > 65 or time.monotonic() - self.wall_start > 120:
            raise TimeoutError("audit CPU65s/wall120s limit")

    def check(self, value, message):
        self.limit()
        self.checks += 1
        if not value:
            raise AuditError(message)

    def close(self, actual, expected, label, absolute=2e-9, relative=2e-10):
        self.check(isinstance(actual, (int, float)) and isinstance(expected, (int, float))
                   and math.isfinite(actual) and math.isfinite(expected), label + " nonfinite")
        gap = abs(actual - expected)
        self.gaps[label] = max(self.gaps.get(label, 0.0), gap)
        self.check(gap <= absolute + relative * abs(expected), label + " numeric mismatch")

    def vector(self, actual, expected, label):
        self.check(len(actual) == len(expected), label + " vector length")
        for left, right in zip(actual, expected):
            self.close(left, right, label)

    def bind(self, path, expected=None):
        self.limit()
        path = Path(path).absolute()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024**2), b""):
                self.limit()
                digest.update(block)
        value = digest.hexdigest()
        previous = self.inputs.setdefault(str(path), value)
        self.check(previous == value and (expected is None or value == expected),
                   "changed or unbound input: " + str(path))
        return value

    def data(self, path):
        path = Path(path)
        self.check(path.stat().st_size <= 64 * 1024**2, "individual parse input exceeds64MiB")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        previous = self.inputs.setdefault(str(path.absolute()), digest)
        self.check(previous == digest, "input changed between reads: " + str(path))
        return data

    def js(self, path):
        return json.loads(self.data(path))

    def csv(self, path):
        reader = csv.DictReader(io.StringIO(self.data(path).decode()))
        self.check(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                   "missing/duplicate CSV fields")
        rows = list(reader)
        self.check(all(None not in row and None not in row.values() for row in rows), "partial CSV row")
        return reader.fieldnames, rows

    def json_lines(self, path):
        return [json.loads(line) for line in self.data(path).splitlines() if line.strip()]

    def subsection(self, callback):
        try:
            result = callback()
            return {"status": "COMPLETE", "audit_passed": True, **result}
        except Exception as exc:
            return {"status": "INCOMPLETE", "audit_passed": False,
                    "failure": type(exc).__name__ + ": " + str(exc),
                    "performance_qualified": False}

    def closure(self, mapping, base=None):
        self.check(isinstance(mapping, dict) and bool(mapping), "missing hash closure")
        for name, expected in mapping.items():
            relative = Path(name)
            self.check(not relative.is_absolute() and ".." not in relative.parts, "unsafe bound path")
            self.bind((base or self.root) / relative, expected)
        return len(mapping)


def average(values):
    return math.fsum(values) / len(values)


def rms(values):
    return math.sqrt(average([value * value for value in values]))


def ratio(numerator, denominator):
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    return numerator / denominator


def percentile95(values):
    values = sorted(values)
    position = (len(values) - 1) * .95
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def npz_stdlib(audit, path):
    """Decode ordinary numeric .npy members, never pickle or execute headers."""
    output = {}
    with zipfile.ZipFile(io.BytesIO(audit.data(path))) as archive:
        audit.check(len(archive.namelist()) == len(set(archive.namelist())), "duplicate NPZ member")
        for info in archive.infolist():
            audit.check(info.filename.endswith(".npy") and info.file_size <= 32 * 1024**2,
                        "unsupported/oversized NPZ member")
            data = archive.read(info.filename)
            audit.check(data[:6] == b"\x93NUMPY" and data[6] in (1, 2, 3), "NPY header")
            start = 10 if data[6] == 1 else 12
            length = int.from_bytes(data[8:start], "little")
            audit.check(length <= 16384, "oversized NPY header")
            metadata = ast.literal_eval(data[start:start + length].decode())
            audit.check(metadata["fortran_order"] is False, "Fortran NPY not supported")
            shape = metadata["shape"]
            audit.check(isinstance(shape, tuple) and all(type(n) is int and n >= 0 for n in shape),
                        "invalid NPY shape")
            dtype = metadata["descr"]
            codes = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "u8": "Q", "u4": "I",
                     "b1": "?", "u1": "B", "i1": "b"}
            audit.check(dtype[0] in "<>|=" and dtype[1:] in codes, "unsupported NPY dtype")
            endian = ">" if dtype[0] == ">" else "<"
            count = math.prod(shape)
            code = endian + str(count) + codes[dtype[1:]]
            payload = data[start + length:]
            audit.check(len(payload) == struct.calcsize(code), "NPY payload size")
            values = list(struct.unpack(code, payload))
            audit.check(all(not isinstance(x, float) or math.isfinite(x) for x in values),
                        "nonfinite NPY values")
            output[info.filename[:-4]] = {"shape": shape, "values": values, "dtype": dtype}
    return output


def matrix(array):
    rows, columns = array["shape"]
    return [array["values"][index * columns:(index + 1) * columns] for index in range(rows)]


def matvec(a, x):
    return [math.fsum(u * v for u, v in zip(row, x)) for row in a]


def numeric_rows(audit, rows, ignored=()):
    output = [{key: float(value) for key, value in row.items() if key not in ignored} for row in rows]
    audit.check(all(math.isfinite(value) for row in output for value in row.values()), "nonfinite CSV")
    return output


def mpc_metrics(rows, solves=None):
    wheel, leg = [r["wheel_force"] for r in rows], [r["leg_force"] for r in rows]
    result = {
        "height_rmse_mm": 1000 * rms([r["extension"] - r["target_extension"] for r in rows]),
        "velocity_rmse_mps": rms([r["vx"] - r["target_vx"] for r in rows]),
        "pitch_rmse_deg": rms([math.degrees(r["pitch"]) for r in rows]),
        "max_abs_pitch_deg": max(abs(math.degrees(r["pitch"])) for r in rows),
        "normalized_effort_rms": rms([v / 80 for v in wheel] + [v / 260 for v in leg]),
        "wheel_force_rms_n": rms(wheel), "leg_force_rms_n": rms(leg),
        "mean_reward": average([r["reward"] for r in rows]),
    }
    if solves is not None:
        result["solve_p95_ms"] = percentile95(solves)
    return result


def check_partial_mpc(audit, receipt):
    attempts, returned = receipt.get("compute_attempts", 0), receipt.get("compute_successful_returns", 0)
    audit.check(0 <= returned <= attempts, "partial compute counters")
    audit.check(receipt.get("solve_calls_completed_lower_bound", 0) >= returned,
                "partial solve lower bound")
    if attempts != returned or receipt.get("solve_counter_unresolved"):
        audit.check(receipt.get("solve_calls") is None, "unreturned compute counted as exact solve")
    if receipt.get("physics_counter_unresolved"):
        audit.check(receipt.get("physics_steps") is None, "unresolved native clock counted as exact")
    audit.check(receipt.get("physics_steps_lower_bound", 0) >= 10 * receipt.get("steps", 0),
                "partial integration prefix lost")
    audit.check(not receipt.get("completed"), "partial promoted to complete")
    return {"status": "INCOMPLETE", "execution_status": "INCOMPLETE",
            "partial_accounting_checks_passed": True,
            "saved_steps": receipt.get("steps"), "compute_attempts": attempts,
            "compute_successful_returns": returned, "solve_calls": receipt.get("solve_calls"),
            "physics_steps": receipt.get("physics_steps"),
            "physics_steps_lower_bound": receipt.get("physics_steps_lower_bound"),
            "solve_calls_completed_lower_bound": receipt.get("solve_calls_completed_lower_bound")}


def audit_mpc_episode(audit, directory, item, old82_report):
    seed, scenario, arm = item["seed"], item["scenario"], item["arm"]
    name = f"{seed}_{scenario}_benchmark_{arm}"
    folder = directory / name
    receipt = audit.js(folder / "metrics.json")
    audit.check(receipt == item, "per-episode receipt differs from parent report: " + name)
    if not receipt.get("completed"):
        return {"name": name, **check_partial_mpc(audit, receipt)}
    _, raw = audit.csv(folder / "trajectory.csv")
    rows = numeric_rows(audit, raw)
    audit.check(len(rows) == item["steps"] == 400 and item["physics_steps"] == 4000,
                "complete MPC episode length/native count")
    audit.check(item["compute_attempts"] == item["compute_successful_returns"] == item["solve_calls"] == 401,
                "complete MPC compute/solve counters")
    audit.check(item["solve_calls_completed_lower_bound"] == 401
                and item["physics_steps_lower_bound"] == 4000
                and item["initial_solve_calls"] == item["initial_solve_calls_completed_lower_bound"] == 1
                and item["control_attempted"] == item["queue_checks_completed"] == 400,
                "complete MPC prefix/attempt counters")
    audit.check(not item["solve_counter_unresolved"] and not item["physics_counter_unresolved"],
                "unresolved complete MPC counters")
    delay = 0 if scenario == "nominal" else 2
    for key, expected in {"physical_intervals_completed": 400, "newly_issued_controls": 400,
                          "newly_issued_controls_applied": 400 - delay,
                          "initialized_queue_controls_applied": delay,
                          "newly_issued_controls_still_pending": delay,
                          "terminal_unapplied_compute_calls": 1}.items():
        audit.check(item[key] == expected, "issued/applied/terminal boundary: " + key)
    solves = item["solve_samples_ms"]
    audit.check(len(solves) == 401 and all(math.isfinite(v) and v >= 0 for v in solves), "solve samples")
    metrics = mpc_metrics(rows, solves)
    for key, value in metrics.items():
        audit.close(value, item[key], "mpc_metric_" + key)
    snapshots = npz_stdlib(audit, folder / "nominal_prediction_model.npz")
    old_name = f"{seed}_{scenario}_benchmark_candidate/nominal_prediction_model.npz"
    old_model = audit.root / OLD82 / old_name
    audit.bind(old_model, old82_report["raw_artifacts_sha256"][old_name])
    reference = npz_stdlib(audit, old_model)
    for key in ("a", "b", "q", "r", "terminal_q", "sx", "su", "equilibrium"):
        audit.check(snapshots[key] == reference[key], "nominal model/cost/prediction differs from82: " + key)
    a, b = matrix(snapshots["a"]), matrix(snapshots["b"])
    eq = snapshots["equilibrium"]["values"]
    audit.check(snapshots["a"]["shape"] == (6, 6) and snapshots["b"]["shape"] == (6, 2)
                and len(eq) == 2 and b[5][1] > 0, "nominal observer matrix shape/gain")
    observers = audit.json_lines(folder / "observer.jsonl")
    terminal = audit.js(folder / "terminal_observer.json")
    audit.check(len(observers) == item["observer_rows"] == 400 and item["terminal_observer_recorded"],
                "observer prefix incomplete")
    audit.check(terminal["compute_index"] == 400 and terminal["issued"] is False
                and terminal["applied"] is False, "terminal compute boundary")
    observers.append(terminal)
    queue = [list(value) for value in item["initial_pending_controls"]]
    audit.check(queue == [eq.copy() for _ in range(delay)], "initial physical FIFO")
    previous_truth = item["initial_state"]
    for index, row in enumerate(rows):
        audit.close(row["time_s"], (index + 1) * .02, "mpc_time", absolute=1e-12)
        audit.close(row["next_solve_ms"], solves[index + 1], "mpc_solve_alignment")
        measured = [row[f"measured_{i}"] for i in range(6)]
        audit.vector(observers[index]["measured"], measured, "observer_csv_alignment")
        audit.vector([row[f"noise_{i}"] for i in range(6)],
                     [x - y for x, y in zip(measured, previous_truth)], "saved_measurement_noise")
        audit.check(observers[index]["interval"] == index, "observer interval index")
        expected_pending = queue if delay else [[0.0, 0.0], [0.0, 0.0]]
        for p in range(2):
            audit.vector([row[f"pending{p}_wheel"], row[f"pending{p}_leg"]],
                         expected_pending[p], "absolute_fifo_before")
        issued = [row["issued_wheel_force"], row["issued_leg_force"]]
        audit.check(-80 <= issued[0] <= 80 and 0 <= issued[1] <= 260, "absolute issued boxes")
        queue.append(issued)
        applied = queue.pop(0)
        audit.vector(applied, [row["wheel_force"], row["leg_force"]], "physical_fifo_applied")
        audit.check(abs(row["pitch"]) < .70, "unsafe complete MPC pitch")
        previous_truth = [row[k] for k in ("x", "pitch", "extension", "vx", "pitch_rate", "extension_rate")]
    disturbance, updates, frozen = 0.0, 0, 0
    alpha = 1.0 - math.exp(-.02 / .2)
    for index, observer in enumerate(observers):
        audit.vector(observer["nominal_eq"], eq, "fixed_nominal_equilibrium")
        if arm == "baseline":
            audit.close(observer["disturbance"], 0.0, "baseline_no_observer")
            audit.check(observer["updated"] is False and observer["reason"] == "baseline"
                        and observer["previous_applied"] is None and observer["raw_disturbance"] is None,
                        "baseline observer unexpectedly active")
        elif index == 0:
            audit.check(observer["previous_applied"] is None and observer["raw_disturbance"] is None
                        and observer["updated"] is False and observer["reason"] == "no_previous_transition",
                        "initial observation invented a transition")
        else:
            previous = observers[index - 1]["measured"]
            current = observer["measured"]
            applied = [rows[index - 1]["wheel_force"], rows[index - 1]["leg_force"]]
            audit.vector(observer["previous_applied"], applied, "observer_uses_previous_applied_force")
            pred_a, pred_b = matvec(a, previous), matvec(b, [x - y for x, y in zip(applied, eq)])
            residual = [x - y - z for x, y, z in zip(current, pred_a, pred_b)]
            raw_force = residual[5] / b[5][1]
            audit.close(observer["raw_disturbance"], raw_force, "raw_disturbance", absolute=2e-8)
            freeze = abs(previous[2]) >= .115 or abs(current[2]) >= .115
            if freeze:
                frozen += 1
            else:
                updates += 1
                disturbance = max(-60.0, min(60.0, (1 - alpha) * disturbance + alpha * raw_force))
            audit.check(observer["updated"] is (not freeze), "observer update/freeze switch")
            audit.check(observer["reason"] == ("measured_extension_near_limit" if freeze else "updated"),
                        "observer freeze reason")
        audit.close(observer["disturbance"], disturbance, "filtered_disturbance", absolute=2e-8)
        audit.vector(observer["estimated_eq"], [eq[0], eq[1] - disturbance], "consistent_estimated_equilibrium")
    audit.check(-80 <= terminal["unissued_control"][0] <= 80
                and 0 <= terminal["unissued_control"][1] <= 260, "terminal unissued control boxes")
    baseline_exact = None
    if arm == "baseline":
        _, old = audit.csv(audit.root / OLD82 / f"{seed}_{scenario}_benchmark_candidate/trajectory.csv")
        baseline_exact = len(old) == 400 and all(
            row[key] == float(saved[key]) for row, saved in zip(rows, old) for key in MPC_COLUMNS
        )
        audit.check(baseline_exact and item["baseline82_replay_exact"], "six baseline82 physical trajectories")
    return {"name": name, "seed": seed, "scenario": scenario, "arm": arm,
            "execution_status": "COMPLETE", "metrics": metrics, "rows": len(rows),
            "observer_records": len(observers), "observer_updates": updates, "observer_freezes": frozen,
            "final_disturbance_force_n": disturbance, "baseline82_exact": baseline_exact,
            "pending_issued_controls_at_end": queue,
            "baseline_exact_columns": list(MPC_COLUMNS) if baseline_exact is not None else None,
            "baseline_exact_excludes": "solve timing and extra measured/noise columns; not bytewise whole CSV"}


def original_mpc_gates(baselines, candidates):
    ratios, gates = {}, {}
    for scenario in SCENARIOS:
        left = [r["metrics"] for r in baselines if r["scenario"] == scenario]
        right = [r["metrics"] for r in candidates if r["scenario"] == scenario]
        for metric in MPC_GUARDS:
            key = f"{scenario}/{metric}"
            value = ratio(average([r[metric] for r in right]), average([r[metric] for r in left]))
            ratios[key + "_ratio"] = value
            gates[key + "_within_5_percent"] = value is not None and value <= 1.05
        if scenario == "mismatch_delay":
            value = ratio(average([r["height_rmse_mm"] for r in right]),
                          average([r["height_rmse_mm"] for r in left]))
            ratios["mismatch_height_ratio"] = value
            gates["height_improvement_at_least_10_percent"] = value is not None and value <= .90
    p95 = max(r["metrics"]["solve_p95_ms"] for r in candidates)
    ratios["candidate_max_episode_solve_p95_ms"] = p95
    gates["solve_p95_within_20_ms"] = p95 <= 20
    return {"accepted": all(gates.values()), "gates": gates, "details": ratios,
            "failed_gates": [key for key, passed in gates.items() if not passed]}


def audit_mpc(audit, directory, external_plan=None):
    report = audit.js(directory / "report.json")
    plan = audit.js(directory / "plan.json")
    if external_plan:
        audit.bind(external_plan, report["plan_sha256"])
        audit.check(audit.js(external_plan) == plan, "MPC plan copy differs")
    else:
        audit.bind(directory / "plan.json", report["plan_sha256"])
    source_count = audit.closure(plan["source_sha256"])
    audit.check(report["source_sha256"] == plan["source_sha256"], "MPC report/plan source closure")
    artifact_count = audit.closure(report["raw_artifacts_sha256"], directory)
    for old_registration in (REG49, REG82):
        old = audit.js(audit.root / old_registration)
        audit.check(all(plan["source_sha256"].get(name) == digest
                        for name, digest in old["source_sha256"].items()), "old source closure rebound")
    audit.bind(audit.root / REG82, REG82_SHA)
    audit.check(plan["preflight_seeds"] == list(MPC_SEEDS) and plan["scenarios"] == list(SCENARIOS)
                and plan["episode_control_steps"] == 400 and plan["expected_solve_calls"] == 4812
                and plan["control_steps_cap"] == 4800 and plan["integration_steps_cap"] == 48000
                and plan["full_phase_authorized"] is False, "MPC fixed matrix/budget")
    audit.check(report["full_status"] == "NOT_RUN" and report["training_steps"] == 0,
                "MPC unauthorized full/training")
    old82 = audit.js(audit.root / OLD82 / "report.json")
    episodes = [audit.subsection(lambda item=item: audit_mpc_episode(audit, directory, item, old82))
                for item in report.get("episodes", [])]
    expected = {(seed, scenario, arm) for seed in MPC_SEEDS for scenario in SCENARIOS
                for arm in ("baseline", "candidate")}
    identities = [(r["seed"], r["scenario"], r["arm"]) for r in report.get("episodes", [])]
    complete = (report["status"] == "COMPLETE" and len(identities) == 12
                and set(identities) == expected and all(r["status"] == "COMPLETE" for r in episodes))
    if not complete:
        audit.check(report.get("decision", {}).get("accepted") is not True,
                    "partial MPC report claims acceptance")
        return {"status": "INCOMPLETE", "source_files": source_count, "artifact_files": artifact_count,
                "episodes": episodes, "performance_qualified": False,
                "reason": "missing/partial/failed audit items; no complete cohort gate"}
    audit.check(report["verification_after"] == "COMPLETE", "MPC final source verification")
    for key, value in {"control_steps": 4800, "integration_steps": 48000, "solve_calls": 4812,
                       "initial_solve_calls": 12}.items():
        audit.check(report[key] == value, "MPC total " + key)
    baselines = [r for r in episodes if r["arm"] == "baseline"]
    candidates = [r for r in episodes if r["arm"] == "candidate"]
    old49_summary = audit.js(audit.root / OLD49 / "summary.json")
    old49_baselines = []
    for item in old49_summary["episodes"]:
        if item["arm"] != "baseline":
            continue
        path = audit.root / OLD49 / f"{item['seed']}_{item['scenario']}_benchmark_baseline/trajectory.csv"
        _, raw = audit.csv(path)
        rows = numeric_rows(audit, raw)
        audit.check(item["completed"] and len(rows) == 400, "absolute49 baseline completeness")
        metrics = mpc_metrics(rows)
        for key, value in metrics.items():
            audit.close(value, item[key], "absolute49_metric_" + key)
        old49_baselines.append({"seed": item["seed"], "scenario": item["scenario"], "metrics": metrics})
    audit.check(len(old49_baselines) == 6, "six absolute49 baselines")
    audit.check({(row["seed"], row["scenario"]) for row in old49_baselines}
                == {(seed, scenario) for seed in MPC_SEEDS for scenario in SCENARIOS},
                "absolute49 baseline matrix")
    comparisons = {"absolute_original49": original_mpc_gates(old49_baselines, candidates),
                   "incremental_Delay82": original_mpc_gates(baselines, candidates)}
    for label, reconstructed in comparisons.items():
        saved = report["decision"][label]
        audit.check(saved["gates"] == reconstructed["gates"]
                    and saved["accepted"] == reconstructed["accepted"], "MPC original gate mismatch")
        for key, value in reconstructed["details"].items():
            audit.close(value, saved["details"][key], "gate_" + key)
    accepted = all(value["accepted"] for value in comparisons.values())
    audit.check(report["decision"]["accepted"] == accepted, "MPC final acceptance mismatch")
    return {"source_files": source_count, "artifact_files": artifact_count, "episodes": episodes,
            "original_gate_references": [REG49 + ":mpc.preflight_rule", "scripts/control_candidates49.py:decision"],
            "recomputed_comparisons": comparisons, "accepted": accepted, "full_status": "NOT_RUN",
            "interpretation": "saved eligible/meaning text is ignored; only recomputed booleans count"}


def sensor_name(case):
    return f"flat_{case['noise']}_delay2_seed{case['seed']}_{case['arm']}"


def sensor_expected():
    return [{"terrain": "flat", "noise": noise, "seed": seed, "delay_steps": 2, "arm": arm}
            for noise in ("noise", "bias") for seed in SENSOR_SEEDS for arm in SENSOR_ARMS]


def audit_sensor_case(audit, directory, item, index):
    name = sensor_name(item)
    receipt = audit.js(directory / (name + ".json"))
    audit.check(receipt == item, "sensor per-case receipt/summary differ")
    columns, raw = audit.csv(directory / (name + ".csv"))
    rows = numeric_rows(audit, raw, ("applied_compensation_status",))
    count = len(rows)
    audit.check(0 < count == item["steps"] <= 400, "sensor task prefix length")
    audit.check(item["execution_status"] == "COMPLETE_TERMINAL"
                and item["termination_reason"] in ("time_limit", "fallen", "out_of_bounds"),
                "sensor missing terminal status")
    full = count == 400 and item["termination_reason"] == "time_limit"
    audit.check(item["full400"] == full, "fallen prefix promoted to complete400")
    audit.check(item["termination_reason"] != "time_limit" or count == 400, "short time-limit prefix")
    audit.check(item["native_steps"] == 5 * count and item["training_updates"] == item["policy_calls"] == 0,
                "sensor integration/training count")
    for key in ("replay_bitwise_equal", "provenance_invariants_pass", "coherent_sample_invariants_pass"):
        audit.check(item[key] is True, "bound online/replay invariant false: " + key)
    audit.check(item["observer_wheel_center_jacobians"] is False, "observer Jacobian branch changed")
    tracking = rms([r["truth_velocity_body_x_mps"] - r["command_velocity_mps"] for r in rows])
    velocity = rms([r["estimate_velocity_body_x_mps"] - r["truth_velocity_body_x_mps"] for r in rows])
    audit.close(tracking, item["tracking_rmse_mps"], "sensor_tracking_rmse")
    audit.close(velocity, item["velocity_estimation_rmse_mps"], "sensor_velocity_rmse")
    for step, (saved, row) in enumerate(zip(raw, rows), start=1):
        audit.check(row["step"] == step and saved["applied_compensation_status"] == "disabled",
                    "sensor row index/CV unexpectedly enabled")
        audit.close(row["time_s"], step * .01, "sensor_clock", absolute=1e-12)
        audit.close(row["measurement_time_s"], max(0, step - 2) * .01, "sensor_packet_delay", absolute=1e-12)
        audit.close(row["measurement_age_s"], min(step * .01, .02), "sensor_age", absolute=1e-12)
        audit.close(row["applied_measurement_age_s"], min((step - 1) * .01, .02), "sensor_input_age", absolute=1e-12)
        fraction = max(0.0, min(1.0, ((step - 1) * .01 - .3) / .3))
        audit.close(row["command_velocity_mps"], .35 * fraction**2 * (3 - 2 * fraction), "sensor_command")
    audit.check(item["compensation_status_counts"] == {"disabled": count}, "sensor compensation counts")
    packets = audit.json_lines(directory / (name + "_measurements.jsonl"))
    arrays = npz_stdlib(audit, directory / (name + "_measurements.npz"))
    audit.check(len(packets) == count + 1 and set(arrays) == set(packets[0]), "sensor packet archive closure")
    audit.check(set(arrays) == {"sequence", "time_s", "gyro_rad_s", "specific_force_m_s2",
                               "joint_position_rad", "joint_velocity_rad_s", "wheel_contact"},
                "sensor packet has non-measurement fields")
    for packet_index, packet in enumerate(packets):
        audit.check(packet["sequence"] == packet_index, "sensor sequence changed")
        audit.close(packet["time_s"], packet_index * .01, "sensor_sample_time", absolute=1e-12)
        for key, value in packet.items():
            array = arrays[key]
            audit.check(array["shape"][0] == len(packets), "sensor array sample dimension")
            width = math.prod(array["shape"][1:])
            expected = array["values"][packet_index * width:(packet_index + 1) * width]
            actual = value if isinstance(value, list) else [value]
            audit.check(actual == expected, "measurement JSONL/NPZ not bit-value equal: " + key)
    times = audit.json_lines(directory / (name + "_times.jsonl"))
    audit.check(len(times) == count, "control field-time records missing")
    coherent = item["arm"] == SENSOR_ARMS[1]
    for step, record in enumerate(times):
        timing = record["control_input"]
        control_time, measurement_time = step * .01, max(0, step - 2) * .01
        audit.check(record["step"] == step + 1 and timing["coherent_measurement_packet"] == coherent,
                    "sensor control-input metadata identity")
        audit.close(timing["control_time_s"], control_time, "metadata_control_time", absolute=1e-12)
        audit.close(timing["measurement_time_s"], measurement_time, "metadata_sample_time", absolute=1e-12)
        audit.check(timing["mixed_estimate_times"] is False
                    and timing["whole_state_current"] == (coherent and step == 0),
                    "delayed/legacy packet falsely current")
        audit.check(len(timing["field_times"]) == 15, "incomplete state field-time ledger")
        audit.close(record["post_step_packet_time_s"], (step + 1) * .01,
                    "metadata_post_sample_time", absolute=1e-12)
        for field in timing["field_times"].values():
            audit.check(field["operation"] == "retained_measurement", "sampling candidate predicts state")
            audit.close(field["estimate_time_s"], measurement_time, "field_estimate_time", absolute=1e-12)
            if coherent:
                audit.close(field["information_time_s"], measurement_time, "field_info_time", absolute=1e-12)
            else:
                audit.check(field["information_time_s"] is None, "legacy mixed phase falsely exact")
    expected_refresh = count + 1 if coherent else 0
    audit.check(item["sampling_copy_forward_calls"] == expected_refresh
                and item["sensor_rne_post_constraint_calls"] == count + 1, "sensor sample call counts")
    initialization = item["initialization_accounting"]
    audit.check(initialization["auxiliary_plant_steps"] == (511 if index == 0 else 0)
                and initialization["auxiliary_explicit_integration_substeps"] == (2555 if index == 0 else 0),
                "cold identification auxiliary budget")
    updates = max(0, count - 2)
    snapshots = 1 + 2 * updates
    contact_jac = 2 * (sum(packets[0]["wheel_contact"])
                       + 2 * sum(sum(p["wheel_contact"]) for p in packets[1:updates + 1]))
    for key, value in {"mj_kinematics": 2 * snapshots, "mj_comPos": 2 * snapshots,
                       "mj_jacBodyCom": 8 * snapshots, "contact_mj_jac": contact_jac}.items():
        audit.check(item["observer_fk_accounting"][key] == value, "observer FK source-formula ledger")
    baseline_exact = None
    if not coherent:
        old_name = sensor_name({**item, "arm": "none"})
        old_columns, old_rows = audit.csv(audit.root / OLD_SENSOR / (old_name + ".csv"))
        audit.check(len(old_rows) == count and set(old_columns) <= set(columns), "old sensor baseline length")
        baseline_exact = all(
            left[key] == right[key] if key == "applied_compensation_status"
            else float(left[key]) == float(right[key])
            for left, right in zip(raw, old_rows) for key in old_columns
        )
        audit.check(baseline_exact and item["baseline_reproduction"]["passed"], "six sensor baselines differ")
    return {"name": name, "noise": item["noise"], "seed": item["seed"], "arm": item["arm"],
            "execution_status": "COMPLETE_TERMINAL", "steps": count, "full400": full,
            "termination_reason": item["termination_reason"], "tracking_rmse_mps": tracking,
            "velocity_estimation_rmse_mps": velocity, "baseline49_exact": baseline_exact,
            "sampling_copy_forward_calls": expected_refresh,
            "online_replay_scope": "SHA-bound execution replay flag checked; no new fusion/FK/model execution",
            "termination_scope": "fallen terminal retained; CSV omits roll, so full safety predicate is not replayed"}


def audit_sensor(audit, directory):
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        failure = audit.js(directory / "failure.json")
        audit.check(failure["status"] == "FAILED_OR_PARTIAL" and failure["resume_permitted"] is False,
                    "invalid sensor failure status")
        if failure["active_case"] is not None:
            audit.check(failure["executed_control_steps"] is None
                        and failure["task_explicit_integration_substeps"] is None,
                        "in-flight sensor counts falsely exact")
        return {"status": "INCOMPLETE", "partial_receipt": failure,
                "performance_qualified": False, "full_authorized": False}
    summary = audit.js(summary_path)
    plan_path = Path(summary["plan_path"])
    audit.bind(plan_path, summary["plan_sha256"])
    plan = audit.js(plan_path)
    source_count = audit.closure(plan["source_sha256"])
    artifact_count = audit.closure(summary["artifact_sha256"], directory)
    original_registration = audit.js(audit.root / "docs/experiments49/estimator_latency49_preregistered.json")
    audit.check(all(plan["source_sha256"].get(name) == digest
                    for name, digest in original_registration["source_sha256"].items()),
                "sensor original49 source closure rebound")
    audit.check(summary["source_sha256"] == plan["source_sha256"]
                and summary["source_closure_reverified_at_end"] is True, "sensor source closure differs")
    audit.check(summary["status"] == "COMPLETE" and summary["planned_cases"] == sensor_expected()
                and plan["protocol"]["planned_cases"] == sensor_expected(), "sensor exact matrix")
    identities = [{key: row[key] for key in sensor_expected()[0]} for row in summary["cases"]]
    audit.check(identities == sensor_expected(), "sensor missing/duplicate/reordered cases")
    audit.check(summary["full_status"] == "NOT_AUTHORIZED"
                and plan["protocol"]["full_authorized"] is False, "sensor full unauthorized")
    cases = [audit.subsection(lambda item=item, index=index:
                            audit_sensor_case(audit, directory, item, index))
             for index, item in enumerate(summary["cases"])]
    if not all(case["status"] == "COMPLETE" for case in cases):
        return {"status": "INCOMPLETE", "cases": cases, "source_files": source_count,
                "artifact_files": artifact_count, "performance_qualified": False}
    controls = sum(case["steps"] for case in cases)
    full_count = sum(case["full400"] for case in cases)
    fallen_count = sum(case["termination_reason"] == "fallen" for case in cases)
    for key, expected in {"executed_control_steps": controls,
                          "task_explicit_integration_substeps": 5 * controls,
                          "identification_auxiliary_plant_steps": 511,
                          "identification_explicit_integration_substeps": 2555,
                          "total_plant_steps": controls + 511,
                          "total_explicit_integration_substeps": 5 * controls + 2555,
                          "partial_cases": 0, "full400_cases": full_count,
                          "fallen_terminal_cases": fallen_count, "training_updates": 0, "policy_calls": 0,
                          "sampling_copy_forward_calls": sum(c["sampling_copy_forward_calls"] for c in cases),
                          "sensor_rne_post_constraint_calls": controls + 12}.items():
        audit.check(summary[key] == expected, "sensor aggregate count: " + key)
    audit.check(summary["total_native_calls"] is None and summary["compiler_internal_native_calls"] is None,
                "unknown native calls represented as complete count")
    before, after = summary["identification_cache_before"], summary["identification_cache_after"]
    audit.check([before[k] for k in ("hits", "misses", "currsize")] == [0, 0, 0]
                and [after[k] for k in ("hits", "misses", "currsize")] == [11, 1, 1], "cold cache ledger")
    counts = {arm: sum(c["arm"] == arm and c["full400"] for c in cases) for arm in SENSOR_ARMS}
    falls = {arm: sum(c["arm"] == arm and c["termination_reason"] == "fallen" for c in cases)
             for arm in SENSOR_ARMS}
    common, pairs = [], []
    for left, right in zip(cases[::2], cases[1::2]):
        both_full = left["full400"] and right["full400"]
        pair = {"noise": left["noise"], "seed": left["seed"],
                "baseline_status": left["termination_reason"], "candidate_status": right["termination_reason"],
                "baseline_steps": left["steps"], "candidate_steps": right["steps"],
                "baseline_rmse_mps": left["tracking_rmse_mps"], "candidate_rmse_mps": right["tracking_rmse_mps"],
                "common_full400": both_full,
                "new_paired_fall": left["termination_reason"] != "fallen" and right["termination_reason"] == "fallen",
                "recovered_pair": not left["full400"] and right["full400"]}
        pairs.append(pair)
        if both_full:
            common.append(pair)
    baseline_mean = average([p["baseline_rmse_mps"] for p in common]) if common else None
    candidate_mean = average([p["candidate_rmse_mps"] for p in common]) if common else None
    rmse_ratio = candidate_mean / baseline_mean if common and baseline_mean > 0 else None
    gain = counts[SENSOR_ARMS[1]] - counts[SENSOR_ARMS[0]]
    no_extra_falls = falls[SENSOR_ARMS[1]] <= falls[SENSOR_ARMS[0]]
    passed = (no_extra_falls and gain >= 0 and rmse_ratio is not None and rmse_ratio <= 1.05
              and (gain > 0 or rmse_ratio <= .98))
    gate = summary["gate"]
    audit.check(gate["passed"] == gate["original_gate_passed"] == passed
                and gate["completions"]["2"] == counts and gate["no_extra_falls"] == no_extra_falls
                and gate["delay20_completion_gain"] == gain and gate["common_complete_pairs"] == len(common),
                "sensor original gate/count mismatch")
    terminations = {arm: dict(Counter(c["termination_reason"] for c in cases if c["arm"] == arm))
                    for arm in SENSOR_ARMS}
    audit.check(gate["termination_counts_by_arm"] == terminations
                and gate["replay_and_provenance_pass"] is True
                and gate["additional_source_and_baseline_integrity_pass"] is True,
                "sensor original gate integrity/termination counts")
    for key, expected_value in {
        "common_raw_tracking_rmse_mps": baseline_mean,
        "common_candidate_tracking_rmse_mps": candidate_mean,
        "common_tracking_rmse_ratio": rmse_ratio,
    }.items():
        if expected_value is None:
            audit.check(gate[key] is None, "empty common sample incorrectly numeric")
        else:
            audit.close(gate[key], expected_value, "sensor_" + key)
    audit.check(gate["full_authorized"] is False and gate["performance_qualified"] is False,
                "sensor preflight promoted to qualification")
    audit.check(gate["gate_implementation"] == "frozen scripts/estimator_candidate49.py:evaluate_gate(preflight)",
                "sensor original gate reference")
    return {"source_files": source_count, "artifact_files": artifact_count, "cases": cases,
            "task_controls": controls, "task_explicit_integrations": 5 * controls,
            "auxiliary_controls": 511, "auxiliary_explicit_integrations": 2555,
            "total_explicit_integrations": 5 * controls + 2555,
            "full400": full_count, "fallen_terminals": fallen_count, "partial": 0,
            "completion_counts": counts, "fallen_counts": falls, "all_six_pairs": pairs,
            "common_complete_pair_count": len(common), "common_baseline_mean_rmse_mps": baseline_mean,
            "common_candidate_mean_rmse_mps": candidate_mean, "common_rmse_ratio": rmse_ratio,
            "original_preflight_gate_passed": passed, "full_authorized": False,
            "new_paired_falls": [p for p in pairs if p["new_paired_fall"]],
            "recovered_pairs": [p for p in pairs if p["recovered_pair"]],
            "interpretation": "original no_extra_falls is an aggregate-count gate, not a no-new-failure-per-pair gate; "
                              "common-complete subset RMSE is not whole-cohort improvement"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/lyh/wheel-legged-control-lab"))
    parser.add_argument("--mpc-dir", type=Path, default=Path("results/offset_mpc83_20261104_01"))
    parser.add_argument("--sensor-dir", type=Path, default=Path("results/coherent_sensor83_preflight_20261104_01"))
    parser.add_argument("--mpc-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = Audit(args.root)
    mpc_dir = args.mpc_dir if args.mpc_dir.is_absolute() else audit.root / args.mpc_dir
    sensor_dir = args.sensor_dir if args.sensor_dir.is_absolute() else audit.root / args.sensor_dir
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(InterruptedError(str(signum))))
    result = {
        "schema": "independent-control83-stdlib-audit-v1",
        "script_sha256": audit.bind(Path(__file__)),
        "mpc": audit.subsection(lambda: audit_mpc(audit, mpc_dir, args.mpc_plan)),
        "sensor": audit.subsection(lambda: audit_sensor(audit, sensor_dir)),
        "new_model_calls": 0, "new_policy_calls": 0, "new_training_updates": 0,
        "new_simulation_steps": 0,
    }
    def closure_end():
        for name, digest in list(audit.inputs.items()):
            audit.bind(name, digest)
        return {"files": len(audit.inputs)}
    result["final_input_reverification"] = audit.subsection(closure_end)
    result["status"] = ("COMPLETE" if all(result[key]["status"] == "COMPLETE"
                                          for key in ("mpc", "sensor", "final_input_reverification"))
                        else "INCOMPLETE")
    result.update(checks=audit.checks, maximum_numeric_gaps=audit.gaps,
                  read_input_sha256=audit.inputs,
                  cpu_seconds=time.process_time() - audit.cpu_start,
                  wall_seconds=time.monotonic() - audit.wall_start)
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 2 * 1024**2:
        raise RuntimeError("audit result exceeds2MiB; no truncated success report written")
    with args.output.open("xb") as handle:
        handle.write(encoded)
    print(json.dumps({"status": result["status"], "checks": audit.checks,
                      "input_files": len(audit.inputs)}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
