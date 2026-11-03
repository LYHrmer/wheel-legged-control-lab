"""One registered development comparison of known-delay MPC prediction.

No full phase, search, retries, training, archive48 access, or runtime replacement.
Numerical imports are delayed until the root-owned bounded execution begins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (490101, 490102, 490103)
SCENARIOS = ("nominal", "mismatch_delay")
ARMS = ("baseline", "candidate")
REG49 = "reports/control_candidates49_preregistration.json"
OLD49 = "results/mpc49_preflight_20261022_01"
NEW_SOURCES = (
    "src/wheel_legged_control/delay_mpc82.py",
    "scripts/delay_mpc82.py",
    "tests/test_delay_mpc82.py",
    "tests/test_delay_mpc82_runner.py",
)
CORE_COLUMNS = (
    "time_s",
    "x",
    "pitch",
    "extension",
    "vx",
    "pitch_rate",
    "extension_rate",
    "wheel_force",
    "leg_force",
    "target_vx",
    "target_extension",
    "push",
    "reward",
    "next_solve_ms",
)
COMPARE_COLUMNS = CORE_COLUMNS[:-1]
GUARDED_METRICS = (
    "height_rmse_mm",
    "velocity_rmse_mps",
    "pitch_rmse_deg",
    "max_abs_pitch_deg",
    "normalized_effort_rms",
    "wheel_force_rms_n",
    "leg_force_rms_n",
    "mean_reward",
    "solve_p95_ms",
)
SETTINGS = {
    "schema": "wheel-delay-mpc82-preregistered-v1",
    "preflight_seeds": list(SEEDS),
    "scenarios": list(SCENARIOS),
    "profiles": ["benchmark"],
    "arms": list(ARMS),
    "episodes": 12,
    "episode_control_steps": 400,
    "control_steps_cap": 4800,
    "integration_steps_cap": 48000,
    "expected_solve_calls": 4812,
    "control_dt_s": 0.02,
    "integration_dt_s": 0.002,
    "horizon": 20,
    "q_height": 90.0,
    "candidate_delay_steps": {"nominal": 0, "mismatch_delay": 2},
    "cpu_seconds": 60,
    "wall_seconds": 120,
    "artifact_cap_bytes": 16 << 20,
    "training_steps": 0,
    "full_phase_authorized": False,
    "seed_role": "already observed49 development seeds; not a new blinded holdout",
}


def require(okay, message):
    if not okay:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_record(value):
    """Preserve nonfinite evidence explicitly; never reinterpret it as a value."""
    paths = []

    def visit(item, path):
        if isinstance(item, float) and (not math.isfinite(item)):
            paths.append(path)
            return "NaN" if math.isnan(item) else "+Infinity" if item > 0 else "-Infinity"
        if isinstance(item, dict):
            return {
                key: visit(child, path + "[" + json.dumps(str(key)) + "]")
                for (key, child) in item.items()
            }
        if isinstance(item, (tuple, list)):
            return [visit(child, f"{path}[{index}]") for (index, child) in enumerate(item)]
        return item

    result = visit(value, "$")
    if paths:
        require(isinstance(result, dict), "nonfinite evidence must have a record envelope")
        result["nonfinite_paths"] = paths
        result["nonfinite_encoding"] = (
            "explicit NaN/+Infinity/-Infinity strings; not finite measurements"
        )
        result["status"] = "INCOMPLETE"
        if "completed" in result:
            result["completed"] = False
        if not result.get("failure"):
            result["failure"] = "nonfinite numeric evidence retained"
        if "decision" in result:
            result["decision"] = {"accepted": False, "meaning": "INCOMPLETE: nonfinite evidence"}
    return result


def write_json(path, value):
    clean = safe_record(value)
    text = json.dumps(clean, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(text)
    return clean


def persist_episode_receipt(destination, receipt):
    """Also sanitize the in-memory object referenced by the parent report."""
    clean = safe_record(receipt)
    receipt.clear()
    receipt.update(clean)
    write_json(Path(destination) / "metrics.json", receipt)


def write_terminal_report(output, report, cap_bytes):
    """Exclusive terminal record with an explicit small failure fallback.

    The loop keeps1MiB free for finalization. If an external writer has already
    consumed even the minimal record's space, retain existing evidence and fail;
    do not exceed the cap or delete/truncate any raw artifact.
    """
    output = Path(output)
    target = output / "report.json"
    if target.exists():
        raise FileExistsError(target)
    used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    available = cap_bytes - used
    failure = None
    attempted_bytes = None
    try:
        clean = safe_record(report)
        text = json.dumps(clean, indent=2, allow_nan=False) + "\n"
        attempted_bytes = len(text.encode())
        if attempted_bytes > available:
            failure = "final_artifact_cap"
    except BaseException as exc:  # noqa: BLE001 - interruption must retain a failure record
        failure = "final_serialization: " + type(exc).__name__ + ": " + str(exc)
    if failure is not None:
        clean = {
            "schema": "wheel-delay-mpc82-result-v1",
            "status": "INCOMPLETE",
            "decision": {"accepted": False, "meaning": "INCOMPLETE: final report fallback"},
            "verification_after": report.get("verification_after", "NOT_RUN"),
            "failure": str(report.get("failure") or failure)[:2048],
            "finalization_failure": failure[:2048],
            "raw_bytes_before_report": used,
            "attempted_report_bytes": attempted_bytes,
            "artifact_cap_bytes": cap_bytes,
            "full_status": "NOT_RUN",
            "training_steps": 0,
            "retained_episode_artifacts": len(report.get("episodes", [])),
        }
        for key in (
            "plan_sha256",
            "control_steps",
            "integration_steps",
            "solve_calls",
            "initial_solve_calls",
        ):
            if key in report and isinstance(report[key], (str, int, float, type(None))):
                clean[key] = report[key]
        clean = safe_record(clean)
        text = json.dumps(clean, separators=(",", ":"), allow_nan=False) + "\n"
        if len(text.encode()) > available:
            clean = {"status": "INCOMPLETE", "failure": "final_report_budget_or_serialization"}
            text = json.dumps(clean, separators=(",", ":")) + "\n"
        if len(text.encode()) > available:
            clean = {"status": "INCOMPLETE"}
            text = json.dumps(clean, separators=(",", ":")) + "\n"
    require(
        len(text.encode()) <= available,
        "no space even for minimal failure record; existing prefix retained",
    )
    with target.open("x") as stream:
        stream.write(text)
    return clean


def required_inputs():
    old = json.loads((ROOT / REG49).read_text())
    paths = (
        set(old["source_sha256"])
        | set(NEW_SOURCES)
        | {
            REG49,
            OLD49 + "/summary.json",
            OLD49 + "/experiment.json",
            "scripts/run_bounded49.py",
            "scripts/resource_guard42.py",
        }
    )
    for seed in SEEDS:
        for scenario in SCENARIOS:
            paths.add(f"{OLD49}/{seed}_{scenario}_benchmark_baseline/trajectory.csv")
    return sorted(paths)


def build_plan():
    """Root may call this after all four new files are final; no numeric imports."""
    old = json.loads((ROOT / OLD49 / "experiment.json").read_text())
    return {
        **SETTINGS,
        "source_sha256": {name: sha(ROOT / name) for name in required_inputs()},
        "versions": old["versions"],
        "unique_variable": "internal known input-delay FIFO prediction only; physical cost and optimizer unchanged",
        "target_object": "original combined mass1.25,damping1.35,delay2,noise1,push-45N@4.2s/0.1s",
        "metric_alignment": "original post-transition state versus same transition command,including startup",
        "gate": "unchanged49 preflight height<=.90; each scenario tracking/pitch/effort/force<=1.05; solveP95<=20ms; all12 complete",
        "scope": "synthetic six-state two-force planar object; no D1/B22/hardware claim",
        "failed_partial_policy": "retain all new artifacts,never overwrite/resume/retry; no automatic full run",
    }


def verify_hashes(root, expected):
    for name, digest in expected.items():
        require(sha(Path(root) / name) == digest, "source/input changed: " + name)


def verify_plan(plan):
    require(all(plan.get(k) == v for (k, v) in SETTINGS.items()), "registered82 settings differ")
    require(set(required_inputs()) <= plan["source_sha256"].keys(), "source closure incomplete")
    old = json.loads((ROOT / REG49).read_text())
    require(
        all(plan["source_sha256"].get(k) == v for (k, v) in old["source_sha256"].items()),
        "frozen49 source closure changed",
    )
    require(
        plan["versions"] == json.loads((ROOT / OLD49 / "experiment.json").read_text())["versions"],
        "original49 numerical versions changed",
    )
    verify_hashes(ROOT, plan["source_sha256"])


def check_fifo_transition(before, issued, applied, after):
    """Pure scalar check of append then pop; zero delay also uses the same rule."""
    require(len(before) in (0, 2) and len(after) == len(before), "unsupported FIFO length")
    require(all(len(x) == 2 for x in [*before, issued, applied, *after]), "two-force FIFO required")
    expected = [*before, issued]
    require(applied == expected[0] and after == expected[1:], "actual FIFO transition differs")


def percentile95(values):
    values = sorted(values)
    require(bool(values), "solve samples missing")
    position = (len(values) - 1) * 0.95
    (lo, hi) = (math.floor(position), math.ceil(position))
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def metrics_from_csv(path, solves):
    """Independent stdlib equivalent of frozen compute_metrics,all saved rows."""
    with Path(path).open(newline="") as stream:
        rows = [{k: float(v) for (k, v) in row.items()} for row in csv.DictReader(stream)]
    require(bool(rows), "no completed trajectory rows")
    require(all(math.isfinite(v) for r in rows for v in r.values()), "nonfinite CSV")

    def rms(values):
        return math.sqrt(math.fsum(x * x for x in values) / len(values))

    wheel = [r["wheel_force"] for r in rows]
    leg = [r["leg_force"] for r in rows]
    return (
        {
            "height_rmse_mm": 1000 * rms([r["extension"] - r["target_extension"] for r in rows]),
            "velocity_rmse_mps": rms([r["vx"] - r["target_vx"] for r in rows]),
            "pitch_rmse_deg": rms([math.degrees(r["pitch"]) for r in rows]),
            "max_abs_pitch_deg": max(abs(math.degrees(r["pitch"])) for r in rows),
            "normalized_effort_rms": rms([x / 80 for x in wheel] + [x / 260 for x in leg]),
            "wheel_force_rms_n": rms(wheel),
            "leg_force_rms_n": rms(leg),
            "mean_reward": math.fsum(r["reward"] for r in rows) / len(rows),
            "solve_p95_ms": percentile95(solves),
        },
        rows,
    )


def exact_baseline_replay(new_rows, old_path):
    with Path(old_path).open(newline="") as stream:
        old = list(csv.DictReader(stream))
    return len(new_rows) == len(old) == 400 and all(
        row[key] == float(previous[key])
        for (row, previous) in zip(new_rows, old, strict=True)
        for key in COMPARE_COLUMNS
    )


def nominal_identity_checks(output):
    """Compare all physical/reward columns for each concurrent zero-delay pair."""
    output = Path(output)
    checks = []
    for seed in SEEDS:
        baseline = output / f"{seed}_nominal_benchmark_baseline/trajectory.csv"
        candidate = output / f"{seed}_nominal_benchmark_candidate/trajectory.csv"
        with candidate.open(newline="") as stream:
            rows = [
                {key: float(row[key]) for key in COMPARE_COLUMNS} for row in csv.DictReader(stream)
            ]
        checks.append(
            {
                "seed": seed,
                "exact": exact_baseline_replay(rows, baseline),
                "candidate_rows": len(rows),
                "baseline_path": str(baseline.relative_to(output)),
                "candidate_path": str(candidate.relative_to(output)),
                "baseline_sha256": sha(baseline),
                "candidate_sha256": sha(candidate),
            }
        )
    return {
        "by_seed": checks,
        "all_exact": all(row["exact"] for row in checks),
        "comparison_columns": list(COMPARE_COLUMNS),
        "excluded_core_columns": ["next_solve_ms"],
        "required_rows_per_episode": 400,
        "scope": "concurrent nominal candidate versus baseline; integrity only",
    }


def assess(rows, nominal_identity=None):
    from scripts.control_candidates49 import decision

    spec = json.loads((ROOT / REG49).read_text())["mpc"]
    old = decision(rows, "mpc", "preflight", spec)
    nominal_identity = nominal_identity or {}
    nominal_checks = nominal_identity.get("by_seed", [])
    nominal_exact = (
        nominal_identity.get("all_exact") is True
        and nominal_identity.get("comparison_columns") == list(COMPARE_COLUMNS)
        and nominal_identity.get("required_rows_per_episode") == 400
        and len(nominal_checks) == len(SEEDS)
        and {row.get("seed") for row in nominal_checks} == set(SEEDS)
        and all(row.get("exact") is True for row in nominal_checks)
    )
    integrity = {
        "all_metric_crosschecks": len(rows) == 12
        and all(r.get("metric_crosscheck", {}).get("passed") is True for r in rows),
        "six_original49_baselines_exact": sum(r["arm"] == "baseline" for r in rows) == 6
        and all(r.get("baseline49_replay_exact") is True for r in rows if r["arm"] == "baseline"),
        "all_three_nominal_candidate_baseline_exact": nominal_exact,
        "all_queue_checks": len(rows) == 12
        and all(r.get("queue_checks_completed") == 400 for r in rows),
        "all_original_model_snapshots_preserved": len(rows) == 12
        and all(r.get("nominal_model_preserved") is True for r in rows),
    }
    accepted = old["accepted"] and all(integrity.values())
    return {
        "accepted": bool(accepted),
        "all_episodes_complete": old["all_episodes_complete"],
        "cohort_valid": old["cohort_valid"],
        "performance_gates": old["gates"],
        "details": old["details"],
        "integrity_gates": integrity,
        "meaning": "passes development preflight only; no full run or deployment authorized"
        if accepted
        else "REJECTED: original performance or integrity gates did not all pass",
        "legacy_stage_meaning_unmodified": old.get("meaning"),
        "full_status": "NOT_RUN",
        "solver_scope": "unchanged frozen optimizer; finite controls and original per-episode P95 gate,not a new optimizer-optimality certificate",
    }


class RecordedController:
    """Observe inputs/outputs without extra sensor reads,random draws or solves."""

    def __init__(self, controller):
        self.controller = controller
        self.compute_calls = 0
        self.last_measured_state = None
        self.last_issued = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def reset(self, state=None):
        self.controller.reset(state)
        self.compute_calls = 0
        self.last_measured_state = None
        self.last_issued = None

    def compute(self, state, command):
        self.last_measured_state = state.copy()
        self.compute_calls += 1
        issued = self.controller.compute(state, command)
        self.last_issued = issued.copy()
        return issued


def run_episode(seed, scenario, arm, destination, receipt, limit):
    import numpy as np

    from wheel_legged_control.controllers import DEFAULT_Q, LinearMPCController
    from wheel_legged_control.delay_mpc82 import DelayMPC82
    from wheel_legged_control.env import WheelLeggedResidualEnv
    from wheel_legged_control.experiments import Rollout, compute_metrics, scenario_options

    (env, samples, solves, snapshot) = (None, [], [], None)
    receipt.update(
        completed=False,
        steps=0,
        control_attempted=0,
        physics_steps=0,
        queue_checks_completed=0,
        solve_calls=0,
        delay_steps=0 if scenario == "nominal" else 2,
    )
    write_json(destination / "started.json", receipt)
    try:
        limit()
        env = WheelLeggedResidualEnv(baseline="mpc", randomize=False, episode_seconds=8.0)
        q = DEFAULT_Q.copy()
        q[2, 2] = 90.0
        controller = (
            LinearMPCController(env.plant, q=q)
            if arm == "baseline"
            else DelayMPC82(env.plant, delay_steps=receipt["delay_steps"], q=q)
        )
        env.controller = RecordedController(controller)
        snapshot = {
            name: getattr(controller, name).copy() for name in ("a", "b", "q", "r", "terminal_q")
        }
        with (destination / "nominal_prediction_model.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                **snapshot,
                sx=controller._sx,
                su=controller._su,
                equilibrium=env.plant.equilibrium_control,
            )
        env.reset(seed=seed, options=scenario_options(scenario))
        require(env._delay_steps == receipt["delay_steps"], "actual delay changed")
        require(
            env.plant.physics_steps == 10
            and env.plant.control_dt == 0.02
            and (env.plant.model.opt.timestep == 0.002),
            "original integration schedule changed",
        )
        solves.append(float(env.controller.last_solve_ms))
        receipt["initial_state"] = env.plant.state().tolist()
        receipt["initial_pending_controls"] = [v.tolist() for v in env._delay_queue]
        receipt["initial_solve_calls"] = 1
        trace = destination / "trajectory.csv"
        extra = (
            "issued_wheel_force",
            "issued_leg_force",
            "pending0_wheel",
            "pending0_leg",
            "pending1_wheel",
            "pending1_leg",
            *[f"measured_{i}" for i in range(6)],
            *[f"noise_{i}" for i in range(6)],
            "next_iterations",
        )
        with trace.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*CORE_COLUMNS, *extra])
            writer.writeheader()
            terminated = truncated = False
            while not (terminated or truncated):
                limit()
                require(receipt["control_attempted"] < 400, "episode control cap")
                before = [v.tolist() for v in env._delay_queue]
                measured = env.controller.last_measured_state.copy()
                true_before = env.plant.state()
                issued = np.clip(
                    env._baseline_control + np.zeros(2),
                    env.plant.actuator_low,
                    env.plant.actuator_high,
                )
                if arm == "candidate":
                    require(
                        np.array_equal(
                            controller.last_pending_before,
                            np.asarray(before).reshape(receipt["delay_steps"], 2),
                        ),
                        "prediction FIFO before differs from actual",
                    )
                receipt["control_attempted"] += 1
                (_, reward, terminated, truncated, info) = env.step(np.zeros(2, dtype=np.float32))
                after = [v.tolist() for v in env._delay_queue]
                check_fifo_transition(before, issued.tolist(), info["control"].tolist(), after)
                if arm == "candidate":
                    require(
                        np.array_equal(
                            controller.last_pending_before,
                            np.asarray(after).reshape(receipt["delay_steps"], 2),
                        ),
                        "prediction FIFO next differs from actual",
                    )
                receipt["queue_checks_completed"] += 1
                command = [info["command_velocity_mps"], info["command_leg_extension_m"]]
                solves.append(float(env.controller.last_solve_ms))
                samples.append(
                    (info["state"], info["control"], command, info["push_force_n"], reward)
                )
                pending = before if before else [[0.0, 0.0], [0.0, 0.0]]
                values = [
                    len(samples) * 0.02,
                    *info["state"],
                    *info["control"],
                    *command,
                    info["push_force_n"],
                    reward,
                    solves[-1],
                    *issued,
                    *pending[0],
                    *pending[1],
                    *measured,
                    *measured - true_before,
                    controller.last_iterations,
                ]
                writer.writerow(dict(zip([*CORE_COLUMNS, *extra], values, strict=True)))
                stream.flush()
                receipt.update(
                    steps=len(samples),
                    physics_steps=round(env.plant.data.time / 0.002),
                    solve_calls=env.controller.compute_calls,
                )
                require(
                    np.isfinite(info["state"]).all()
                    and np.isfinite(info["control"]).all()
                    and np.isfinite(controller._solution).all(),
                    "nonfinite state/control/solver iterate",
                )
            require(
                truncated and (not terminated) and (len(samples) == 400),
                "unsafe or incomplete episode",
            )
        (states, controls, commands, pushes, rewards) = map(np.asarray, zip(*samples))
        rollout = Rollout(
            arm,
            scenario,
            np.arange(len(samples)) * 0.02,
            states,
            controls,
            commands,
            pushes,
            rewards,
            terminated,
            np.asarray(solves),
        )
        metrics = compute_metrics(rollout)
        metrics.update(
            wheel_force_rms_n=float(np.sqrt(np.mean(controls[:, 0] ** 2))),
            leg_force_rms_n=float(np.sqrt(np.mean(controls[:, 1] ** 2))),
        )
        (independently, raw) = metrics_from_csv(trace, solves)
        gaps = {key: abs(metrics[key] - independently[key]) for key in GUARDED_METRICS}
        passed = all(
            math.isfinite(metrics[k]) and gap <= 1e-10 + 1e-09 * abs(independently[k])
            for (k, gap) in gaps.items()
        )
        require(passed, "independent original metric crosscheck failed")
        receipt.update(
            metrics,
            metric_crosscheck={"passed": passed, "maximum_gaps": gaps},
            q_height=90.0,
            completed=True,
            solve_calls=len(solves),
            solve_samples_ms=solves,
            control_decisions_applied=400,
            post_transition_compute_calls=400,
            terminal_unapplied_compute_calls=1,
            nominal_model_preserved=all(
                np.array_equal(getattr(controller, k), v) for (k, v) in snapshot.items()
            ),
            baseline49_replay_exact=exact_baseline_replay(
                raw, ROOT / f"{OLD49}/{seed}_{scenario}_benchmark_baseline/trajectory.csv"
            )
            if arm == "baseline"
            else None,
            trace_sha256=sha(trace),
        )
        require(receipt["nominal_model_preserved"], "nominal controller model/cost changed")
        require(
            receipt["physics_steps"] == 4000 and receipt["solve_calls"] == 401,
            "per-episode count mismatch",
        )
        if arm == "baseline":
            require(
                receipt["baseline49_replay_exact"],
                "original49 baseline trajectory did not exactly reproduce",
            )
    except BaseException as exc:
        receipt.update(completed=False, failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup_errors = []
        if env is not None:
            try:
                sim_time = float(env.plant.data.time)
                receipt["last_simulation_time_s"] = sim_time
                if math.isfinite(sim_time):
                    receipt["physics_steps"] = round(sim_time / env.plant.model.opt.timestep)
                else:
                    receipt["physics_counter_unresolved"] = True
                    receipt["physics_counter_semantics"] = (
                        "last finite saved native-step prefix only"
                    )
                    receipt["completed"] = False
                receipt["solve_calls"] = getattr(env.controller, "compute_calls", 0)
                if not receipt["completed"]:
                    state = env.plant.state()
                    control = env.plant.data.ctrl.copy()
                    pending = np.asarray(list(env._delay_queue), dtype=np.float64).reshape(-1, 2)
                    receipt["partial_final_state"] = state.tolist()
                    receipt["partial_actual_control"] = control.tolist()
                    receipt["partial_pending_controls"] = pending.tolist()
                    receipt["partial_accounting_note"] = (
                        "native integration from MuJoCo time may exceed persisted CSV intervals if failure occurred during the next solve"
                    )
                    raw_path = destination / "partial_numeric_state.npz"
                    with raw_path.open("xb") as stream:
                        np.savez_compressed(
                            stream,
                            state=state,
                            control=control,
                            pending=pending,
                            simulation_time=np.asarray(sim_time),
                        )
                    receipt["partial_numeric_file"] = raw_path.name
                    receipt["partial_numeric_sha256"] = sha(raw_path)
            except BaseException as exc:  # noqa: BLE001 - retain the primary failure and CSV prefix
                cleanup_errors.append("partial capture: " + type(exc).__name__ + ": " + str(exc))
                receipt["completed"] = False
            try:
                env.close()
            except BaseException as exc:  # noqa: BLE001 - cleanup cannot discard prior evidence
                cleanup_errors.append("env.close: " + type(exc).__name__ + ": " + str(exc))
                receipt["completed"] = False
        receipt["steps"] = len(samples)
        receipt["solve_samples_ms"] = solves
        if cleanup_errors:
            receipt["cleanup_errors"] = cleanup_errors
            if not receipt.get("failure"):
                receipt["failure"] = "; ".join(cleanup_errors)
        persist_episode_receipt(destination, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "wheel-delay-mpc82-result-v1",
        "status": "INCOMPLETE",
        "verification_after": "NOT_RUN",
        "candidate": "input_delay_fifo",
        "phase": "development_preflight",
        "episodes": [],
        "decision": {"accepted": False, "meaning": "INCOMPLETE; no ranking"},
        "full_status": "NOT_RUN",
        "training_steps": 0,
        "archive48_invocations": 0,
        "runtime_v27_changed": False,
        "failure": None,
    }
    write_json(args.output / "initial_incomplete.json", report)
    (started_cpu, started_wall) = (time.process_time(), time.monotonic())

    def interrupted(signum, _frame):
        raise InterruptedError("82 signal " + str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU, signal.SIGALRM):
        signal.signal(sig, interrupted)
    resource.setrlimit(resource.RLIMIT_CPU, (59, 60))
    signal.alarm(120)
    plan = None

    def limit():
        require(time.process_time() - started_cpu < 52, "82 cooperative CPU deadline")
        require(time.monotonic() - started_wall < 110, "82 cooperative wall deadline")
        require(
            sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) < 15 << 20,
            "82 artifact reserve reached",
        )

    try:
        plan = json.loads(args.plan.read_text())
        report["plan_sha256"] = sha(args.plan)
        verify_plan(plan)
        require(
            os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
            "Nice10/one CPU required",
        )
        require(
            all(
                os.environ.get(k) == "1"
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "LP_NUM_THREADS",
                )
            ),
            "single thread required",
        )
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CPU only")
        for directory in (ROOT, ROOT / "src", ROOT / ".local-deps"):
            sys.path.insert(0, str(directory))
        report["versions"] = {name: version(name) for name in plan["versions"]}
        require(report["versions"] == plan["versions"], "original49 versions not reproduced")
        write_json(args.output / "plan.json", plan)
        report["source_sha256"] = plan["source_sha256"]
        for index, seed in enumerate(SEEDS):
            for scenario in SCENARIOS:
                for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
                    limit()
                    destination = args.output / f"{seed}_{scenario}_benchmark_{arm}"
                    destination.mkdir(exist_ok=False)
                    row = {
                        "seed": seed,
                        "scenario": scenario,
                        "profile": "benchmark",
                        "arm": arm,
                        "completed": False,
                    }
                    report["episodes"].append(row)
                    run_episode(seed, scenario, arm, destination, row, limit)
                    require(
                        sum(r.get("steps", 0) for r in report["episodes"]) <= 4800
                        and sum(r.get("physics_steps", 0) for r in report["episodes"]) <= 48000,
                        "total integration cap",
                    )
        limit()
        report["nominal_candidate_baseline_identity"] = nominal_identity_checks(args.output)
        report["decision"] = assess(
            report["episodes"], report["nominal_candidate_baseline_identity"]
        )
        require(report["decision"]["all_episodes_complete"], "incomplete cohort")
        report["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001 - terminal interruption is explicitly INCOMPLETE
        report.update(status="INCOMPLETE", failure=f"{type(exc).__name__}: {exc}")
        report["decision"] = {"accepted": False, "meaning": "INCOMPLETE; no performance ranking"}
    finally:
        report.update(
            control_steps=sum(r.get("steps", 0) for r in report["episodes"]),
            integration_steps=sum(r.get("physics_steps", 0) for r in report["episodes"]),
            solve_calls=sum(r.get("solve_calls", 0) for r in report["episodes"]),
            initial_solve_calls=sum(r.get("initial_solve_calls", 0) for r in report["episodes"]),
            cpu_seconds=time.process_time() - started_cpu,
            wall_seconds=time.monotonic() - started_wall,
        )
        try:
            require(
                plan is not None and sha(args.plan) == report["plan_sha256"], "plan absent/changed"
            )
            verify_plan(plan)
            require(
                {name: version(name) for name in plan["versions"]} == plan["versions"],
                "versions changed",
            )
            if report["status"] == "COMPLETE":
                require(
                    (
                        report["control_steps"],
                        report["integration_steps"],
                        report["solve_calls"],
                        report["initial_solve_calls"],
                    )
                    == (4800, 48000, 4812, 12),
                    "complete cohort counts differ",
                )
            report["verification_after"] = "COMPLETE"
        except BaseException as exc:  # noqa: BLE001 - preserve verification failure independently
            report.update(
                status="INCOMPLETE",
                verification_after="FAILED",
                verification_failure=f"{type(exc).__name__}: {exc}",
            )
            report["decision"] = {
                "accepted": False,
                "meaning": "INCOMPLETE: final source/contract verification failed",
            }
        report["raw_artifacts_sha256"] = {
            str(p.relative_to(args.output)): sha(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        }
        report["raw_bytes_before_report"] = sum(
            p.stat().st_size for p in args.output.rglob("*") if p.is_file()
        )
        report = write_terminal_report(args.output, report, 16 << 20)
        signal.alarm(0)
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "status",
                    "verification_after",
                    "control_steps",
                    "integration_steps",
                    "solve_calls",
                    "decision",
                )
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
