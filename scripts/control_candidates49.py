"""One-shot, preregistered MPC-height and identified-wheel-feedforward candidates.

Run under the serial resource-limited systemd launcher. No training, parameter
search, archived-run resumption, or edits to existing controller code are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REGISTRATION = ROOT / "reports/control_candidates49_preregistration.json"
ARMS = ("baseline", "candidate")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def interrupted(signum, _frame) -> None:
    raise InterruptedError(f"candidate received signal {signum}")


def verify_sources(registration: dict) -> None:
    changed = [name for name, digest in registration["source_sha256"].items()
               if sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f"preregistered sources changed: {changed}")


def scaled_feedforward_controller(config, parameters, scale: float):
    """Inject a total feedforward multiplier into the existing PI implementation.

    Scaling the supplied load inertia and all three continuous model parameters
    is algebraically exactly scale * feedforward. These are controller-only
    coefficients, never refitted parameters or altered physical plant values.
    The original compute method still applies anti-windup to the scaled total.
    """
    from wheel_legged_control.actuator_bench import WHEEL_INERTIA_KG_M2
    from wheel_legged_control.wheel_control import WheelVelocityController

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("feedforward scale must be finite and positive")
    coefficients = replace(
        parameters, armature=parameters.armature * scale,
        damping=parameters.damping * scale,
        coulomb_friction=parameters.coulomb_friction * scale,
    )
    return WheelVelocityController(config, coefficients, WHEEL_INERTIA_KG_M2 * scale)


def paired_ratio(rows: list[dict], metric: str) -> float:
    means = {arm: float(np.mean([r[metric] for r in rows if r["arm"] == arm]))
             for arm in ARMS}
    if means["baseline"] == 0:
        return 1.0 if means["candidate"] == 0 else float("inf")
    return means["candidate"] / means["baseline"]


def decision(rows: list[dict], candidate: str, phase: str, spec: dict) -> dict:
    """Reject incomplete cohorts before any performance gate is considered."""
    seeds = spec[phase + "_seeds"]
    profiles = ["stress"] if candidate == "pi" and phase == "preflight" else spec["profiles"]
    expected = {(seed, scenario, profile, arm) for seed in seeds
                for scenario in spec["scenarios"] for profile in profiles for arm in ARMS}
    keys = [(r["seed"], r["scenario"], r["profile"], r["arm"]) for r in rows]
    cohort_valid = len(keys) == len(expected) and set(keys) == expected
    complete = cohort_valid and all(r["completed"] for r in rows)
    result = {"cohort_valid": cohort_valid, "all_episodes_complete": complete,
              "accepted": False, "gates": {}, "details": {}}
    if not complete:
        result["reason"] = "missing, duplicate, failed or partial episodes; no candidate ranking"
        return result
    gates, details = result["gates"], result["details"]
    if candidate == "mpc":
        mismatch = [r for r in rows if r["scenario"] == "mismatch_delay"]
        details["mismatch_height_ratio"] = paired_ratio(mismatch, "height_rmse_mm")
        gates["height_improvement_at_least_10_percent"] = (
            details["mismatch_height_ratio"] <= 0.90
        )
        guards = ("velocity_rmse_mps", "pitch_rmse_deg", "max_abs_pitch_deg",
                  "normalized_effort_rms", "wheel_force_rms_n", "leg_force_rms_n")
        for scenario in spec["scenarios"]:
            selected = [r for r in rows if r["scenario"] == scenario]
            for metric in guards:
                ratio = paired_ratio(selected, metric)
                details[f"{scenario}/{metric}_ratio"] = ratio
                gates[f"{scenario}/{metric}_within_5_percent"] = ratio <= 1.05
        details["candidate_max_episode_solve_p95_ms"] = max(
            r["solve_p95_ms"] for r in rows if r["arm"] == "candidate"
        )
        gates["solve_p95_within_20_ms"] = details["candidate_max_episode_solve_p95_ms"] <= 20
        if phase == "full":
            from scipy.stats import t

            paired = {(r["seed"], r["arm"]): r["height_rmse_mm"] for r in mismatch}
            differences = np.array([paired[(s, "candidate")] - paired[(s, "baseline")]
                                    for s in seeds])
            half_width = float(t.ppf(0.975, len(seeds) - 1)
                               * differences.std(ddof=1) / np.sqrt(len(seeds)))
            interval = [float(differences.mean() - half_width),
                        float(differences.mean() + half_width)]
            details["mismatch_height_paired_t95_mm"] = interval
            gates["paired_height_interval_below_zero"] = interval[1] < 0
    else:
        for scenario in spec["scenarios"]:
            selected = [r for r in rows if r["scenario"] == scenario]
            stress = [r for r in selected if r["profile"] == "stress"]
            means = {arm: np.mean([r["saturation_fraction"] for r in stress
                                  if r["arm"] == arm]) for arm in ARMS}
            reduction = float(means["baseline"] - means["candidate"])
            details[f"{scenario}/stress_saturation_reduction_pp"] = reduction * 100
            gates[f"{scenario}/stress_saturation_reduced_5pp"] = reduction >= 0.05
            for profile in profiles:
                subset = [r for r in selected if r["profile"] == profile]
                ratio = paired_ratio(subset, "velocity_rmse_rad_s")
                details[f"{scenario}/{profile}_rmse_ratio"] = ratio
                gates[f"{scenario}/{profile}_rmse_within_5_percent"] = ratio <= 1.05
        gates["no_speed_guard_triggered"] = not any(r["speed_guard_triggered"] for r in rows)
    result["accepted"] = bool(all(gates.values()))
    result["meaning"] = ("eligible for independent full comparison" if phase == "preflight"
                         else "passes registered synthetic experiment gates")
    return result


def inspect_existing() -> dict:
    """Read archived torque components; this is not a closed-loop prediction."""
    output = {"physics_steps": 0, "training_steps": 0, "cases": []}
    for scenario in ("matched", "mismatch"):
        path = ROOT / "results/wheel_control" / scenario / "stress_identified_ff.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        original = [float(r["requested_torque_nm"]) for r in rows]
        frozen_feedback = [float(r["proportional_torque_nm"])
                           + float(r["integral_torque_nm"])
                           + 0.75 * float(r["feedforward_torque_nm"]) for r in rows]
        output["cases"].append({
            "scenario": scenario, "source": str(path.relative_to(ROOT)),
            "sha256": sha256(path), "samples": len(rows),
            "recorded_saturation_fraction": float(np.mean(np.abs(original) > 2.0)),
            "frozen_feedback_counterfactual_saturation_fraction":
                float(np.mean(np.abs(frozen_feedback) > 2.0)),
        })
    output["limitation"] = (
        "Counterfactual holds recorded P and I fixed; real feedback and anti-windup change. "
        "Only a directional screening calculation, not a closed-loop performance claim. "
        "Planar MPC mismatch_delay delays the whole control by 2*0.02s and also changes "
        "mass/damping, measurement noise and push; no isolated delay attribution. "
        "Nominal MPC seeds are deterministic repetitions, not independent statistical evidence."
    )
    return output


def run_mpc(seed: int, scenario: str, arm: str, destination: Path) -> dict:
    from wheel_legged_control.controllers import DEFAULT_Q, LinearMPCController
    from wheel_legged_control.env import WheelLeggedResidualEnv
    from wheel_legged_control.experiments import Rollout, compute_metrics, scenario_options

    env = WheelLeggedResidualEnv(baseline="mpc", randomize=False, episode_seconds=8.0)
    q = DEFAULT_Q.copy()
    q[2, 2] = 90.0 if arm == "baseline" else 180.0
    # Construct before reset/set_domain: both arms retain the same nominal model.
    env.controller = LinearMPCController(env.plant, q=q)
    samples = []
    solves = []
    try:
        env.reset(seed=seed, options=scenario_options(scenario))
        solves.append(env.controller.last_solve_ms)
        with (destination / "trajectory.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "x", "pitch", "extension", "vx", "pitch_rate",
                             "extension_rate", "wheel_force", "leg_force", "target_vx",
                             "target_extension", "push", "reward", "next_solve_ms"])
            terminated = truncated = False
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
                command = [info["command_velocity_mps"], info["command_leg_extension_m"]]
                solves.append(env.controller.last_solve_ms)
                samples.append((info["state"], info["control"], command,
                                info["push_force_n"], reward))
                writer.writerow([len(samples) * env.plant.control_dt, *info["state"],
                                 *info["control"], *command, info["push_force_n"], reward,
                                 solves[-1]])
                handle.flush()
        states, controls, commands, pushes, rewards = map(np.asarray, zip(*samples))
        rollout = Rollout(arm, scenario, np.arange(len(samples)) * 0.02, states,
                          controls, commands, pushes, rewards, terminated, np.asarray(solves))
        metric = compute_metrics(rollout)
        metric.update({"steps": len(samples), "physics_steps": len(samples) * env.plant.physics_steps,
                       "completed": bool(truncated and not terminated and len(samples) == 400),
                       "wheel_force_rms_n": float(np.sqrt(np.mean(controls[:, 0] ** 2))),
                       "leg_force_rms_n": float(np.sqrt(np.mean(controls[:, 1] ** 2))),
                       "solve_calls": len(solves), "q_height": q[2, 2]})
        return metric
    finally:
        env.close()


def stream_pi_rows(config, truth, controller, *, profile: str, duration: float,
                   seed: int, noise_scale: float, stribeck_friction_nm: float,
                   destination: Path) -> list[dict]:
    """Same time alignment/noise/control order as the frozen wheel evaluator.

    Persist each completed physical transition before requesting the next one;
    interruption or a bad state must not discard this episode's earlier data.
    """
    from scripts import evaluate_wheel_control as legacy

    plant = legacy.ActuatorBench(config, truth, stribeck_friction_nm=stribeck_friction_nm)
    plant.reset()
    controller.reset()
    time_s = np.arange(round(duration / config.dt)) * config.dt
    target_velocity, target_acceleration = legacy.reference(profile, time_s)
    rng = np.random.default_rng(np.random.SeedSequence((seed, 314)))
    noise = rng.normal(size=len(time_s)) * noise_scale * 0.002
    rows = []
    with (destination / "trajectory.csv").open("x", newline="") as handle:
        writer = None
        for index, time in enumerate(time_s):
            position, velocity = plant.state
            measured_velocity = velocity + noise[index]
            control = controller.compute(
                target_velocity[index], target_acceleration[index], measured_velocity,
            )
            next_position, next_velocity = plant.step(control.command_torque_nm)
            if not np.isfinite((next_position, next_velocity)).all():
                raise FloatingPointError("non-finite plant state; run has no completed summary")
            speed_guard_triggered = bool(abs(next_velocity) > legacy.SPEED_GUARD_RAD_S)
            row = {
                "time_s": time,
                "reference_velocity_rad_s": target_velocity[index],
                "reference_acceleration_rad_s2": target_acceleration[index],
                "position_rad": position, "actual_velocity_rad_s": velocity,
                "measured_velocity_rad_s": measured_velocity,
                "tracking_error_rad_s": target_velocity[index] - velocity,
                **asdict(control), "applied_torque_nm": plant.applied_torque_nm,
                "next_position_rad": next_position, "next_velocity_rad_s": next_velocity,
                "speed_guard_triggered": speed_guard_triggered,
            }
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if speed_guard_triggered:
                break
    return rows


def run_pi(seed: int, scenario: str, profile: str, arm: str, destination: Path) -> dict:
    from scripts.evaluate_wheel_control import compute_metrics
    from wheel_legged_control.actuator_bench import ActuatorParameters, BenchConfig
    from wheel_legged_control.wheel_control import WheelControlConfig

    fit = json.loads((ROOT / "results/wheel_control" / scenario / "fit.json").read_text())
    config = BenchConfig()
    controller = scaled_feedforward_controller(
        WheelControlConfig(), ActuatorParameters(**fit["parameters"]),
        1.0 if arm == "baseline" else 0.75,
    )
    rows = stream_pi_rows(
        config, ActuatorParameters(0.017, 0.105, 0.043, 2), controller,
        profile=profile, duration=6.0, seed=seed, noise_scale=1.0,
        stribeck_friction_nm=0.08 if scenario == "mismatch" else 0.0,
        destination=destination,
    )
    metric = compute_metrics(rows, config.dt)
    metric.update({"completed": len(rows) == 3000 and not metric["speed_guard_triggered"],
                   "physics_steps": len(rows)})
    return metric


def admit_full(preflight: Path | None, candidate: str, registration_sha: str) -> None:
    if preflight is None:
        raise ValueError("full phase requires a completed passing --preflight directory")
    summary = json.loads((preflight / "summary.json").read_text())
    manifest = json.loads((preflight / "manifest.json").read_text())
    if "summary.json" not in manifest:
        raise ValueError("preflight manifest does not bind its summary")
    for name, digest in manifest.items():
        if sha256(preflight / name) != digest:
            raise ValueError(f"preflight artifact changed: {name}")
    if (summary["candidate"] != candidate or summary["phase"] != "preflight"
            or summary["registration_sha256"] != registration_sha
            or summary["status"] != "complete" or not summary["decision"]["accepted"]):
        raise ValueError("preflight is incomplete, rejected, or from a different registration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("mpc", "pi"), required=True)
    parser.add_argument("--phase", choices=("inspect", "preflight", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    registration = json.loads(REGISTRATION.read_text())
    verify_sources(registration)
    registration_sha = sha256(REGISTRATION)
    if args.phase == "full":
        admit_full(args.preflight, args.candidate, registration_sha)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "registration.json", registration)
    started = perf_counter()
    experiment = {"candidate": args.candidate, "phase": args.phase,
                  "registration_sha256": registration_sha,
                  "versions": {name: version(name) for name in ("numpy", "scipy", "mujoco")},
                  "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "experiment.json", experiment)
    rows = []
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.phase == "inspect":
            summary = {**experiment, "status": "complete", "inspection": inspect_existing()}
        else:
            spec = registration[args.candidate]
            profiles = (["stress"] if args.candidate == "pi" and args.phase == "preflight"
                        else spec["profiles"])
            for index, seed in enumerate(spec[args.phase + "_seeds"]):
                for scenario in spec["scenarios"]:
                    for profile in profiles:
                        # Alternate order without altering pairings or sampling.
                        for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
                            label = f"{seed}_{scenario}_{profile}_{arm}"
                            destination = args.output / label
                            destination.mkdir()
                            write_json(destination / "started.json", {
                                "seed": seed, "scenario": scenario, "profile": profile, "arm": arm,
                            })
                            if args.candidate == "mpc":
                                metric = run_mpc(seed, scenario, arm, destination)
                            else:
                                metric = run_pi(seed, scenario, profile, arm, destination)
                            row = {**metric, "seed": seed, "scenario": scenario,
                                   "profile": profile, "arm": arm}
                            write_json(destination / "metrics.json", row)
                            rows.append(row)
                            print(json.dumps({"episode": label, "complete": row["completed"]}),
                                  flush=True)
            result = decision(rows, args.candidate, args.phase, spec)
            summary = {**experiment, "status": "complete" if result["all_episodes_complete"]
                       else "partial", "decision": result, "episodes": rows,
                       "physics_steps": sum(r["physics_steps"] for r in rows),
                       "training_steps": 0}
    except BaseException as exc:
        write_json(args.output / "failure.json", {**experiment, "status": "partial",
                   "error": repr(exc), "finished_episodes": rows,
                   "elapsed_seconds": perf_counter() - started})
        raise
    summary["elapsed_seconds"] = perf_counter() - started
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "manifest.json", {
        str(p.relative_to(args.output)): sha256(p) for p in sorted(args.output.rglob("*"))
        if p.is_file()
    })
    print(json.dumps({"status": summary["status"], "decision": summary.get("decision"),
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
