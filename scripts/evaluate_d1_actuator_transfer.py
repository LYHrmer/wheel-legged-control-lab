"""Synthetic torque-response identification and causal D1 transfer comparison.

This is NOT armature/damping/friction identification. Calibration assumes a
synthetic noisy joint-side torque measurement, not torque inferred from D1
encoders. Existing MuJoCo passive friction is unchanged. Only development roads
are evaluated, with zero RL residual and the same nominal low-level controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from wheel_legged_control.d1.actuator_channel import (
    ActuatorChannel,
    ActuatorChannelConfig,
    ActuatorTrace,
)
from wheel_legged_control.d1.model import D1_JOINT_NAMES, JOINT_TORQUE_LIMIT

ROOT = Path(__file__).resolve().parents[1]
PHYSICS_DT_S = 0.002
CONTROL_DT_S = 0.01
WHEELS = np.asarray([index for index, name in enumerate(D1_JOINT_NAMES) if "_foot_" in name])
LEGS = np.asarray([index for index in range(16) if index not in WHEELS])
LIMITS_NM = JOINT_TORQUE_LIMIT.copy()
CASES = ("ideal", "nonideal", "fitted_compensation")
SYNTHETIC_WHEELS = ActuatorChannelConfig(
    n_axes=4,
    physics_dt_s=PHYSICS_DT_S,
    delay_steps=2,
    gain=(0.80, 0.90, 0.85, 0.95),
    time_constant_s=(0.006, 0.008, 0.010, 0.012),
    torque_limit_nm=12.0,
)
QUALITY = {
    "velocity_rmse_mps": 0.08,
    "yaw_rmse_rps": 0.08,
    "height_rmse_m": 0.025,
    "attitude_rmse_rad": 0.15,
}


def _finite_matrix(value, name, columns=4):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite (N, {columns}) array")
    return array


@dataclass(frozen=True)
class TorqueLog:
    """command[k] -> response[k] after one 2 ms channel step; reset history zero."""

    commands_nm: np.ndarray
    measured_nm: np.ndarray
    split: str
    physics_dt_s: float = PHYSICS_DT_S

    def __post_init__(self):
        if self.split not in ("calibration", "validation") or self.physics_dt_s != PHYSICS_DT_S:
            raise ValueError("require calibration/validation and the known 2 ms channel clock")
        for name in ("commands_nm", "measured_nm"):
            array = _finite_matrix(getattr(self, name), name)
            object.__setattr__(
                self, name, np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)
            )
        if self.commands_nm.shape != self.measured_nm.shape or len(self.commands_nm) < 32:
            raise ValueError("require matching logs with at least 32 samples")


def excitation(split: str, *, seed: int, samples: int) -> np.ndarray:
    """Different waveform families, not just new noise seeds; never use global RNG."""
    if split not in ("calibration", "validation") or samples < 32 or seed < 0:
        raise ValueError("invalid excitation split, seed, or sample count")
    rng = np.random.default_rng(seed)
    result = np.zeros((samples, 4))
    if split == "calibration":
        for axis in range(4):
            start = 20
            while start < samples:
                end = min(samples, start + int(rng.integers(7, 45)))
                result[start:end, axis] = rng.uniform(-3.0, 3.0)
                start = end
    else:
        t = np.arange(samples) * PHYSICS_DT_S
        for axis in range(4):
            phase = rng.uniform(-np.pi, np.pi)
            result[:, axis] = 1.7 * np.sin(
                2 * np.pi * (0.7 * t + 1.8 * t**2) + phase
            ) + 0.9 * np.sin(2 * np.pi * (8.0 + axis) * t)
        result[:20] = 0.0
    return result


def channel_response(commands_nm, config: ActuatorChannelConfig) -> np.ndarray:
    commands = _finite_matrix(commands_nm, "commands_nm", config.n_axes)
    channel = ActuatorChannel(config)
    return np.asarray(
        [channel.step(command, np.zeros(config.n_axes)).applied_nm for command in commands]
    )


def record_synthetic_log(split, *, seed, samples, noise_std_nm=0.01):
    if not math.isfinite(noise_std_nm) or noise_std_nm < 0:
        raise ValueError("noise_std_nm must be finite and nonnegative")
    commands = excitation(split, seed=seed, samples=samples)
    truth = channel_response(commands, SYNTHETIC_WHEELS)
    noise = np.random.default_rng(seed + 1_000_000).normal(0, noise_std_nm, truth.shape)
    return TorqueLog(commands, truth + noise, split), truth


def _linear_response(commands, gain, tau, delay):
    shifted = np.zeros_like(commands)
    shifted[delay:] = commands[: len(commands) - delay] if delay else commands
    a = np.exp(-PHYSICS_DT_S / tau)
    return lfilter([gain * (1.0 - a)], [1.0, -a], shifted)


def fit_torque_channel(log: TorqueLog, *, max_delay_steps=4) -> dict:
    """Fit output-error trajectories; validation and hidden parameters are not inputs.

    Search bounds are declared teaching priors, not measured D1 capabilities.
    No saturation occurs for |command| <= 6 Nm and gain <= 1.5 at a 12 Nm limit.
    Delay candidates use only calibration error; all optimizer statuses remain.
    """
    if not isinstance(log, TorqueLog) or log.split != "calibration":
        raise ValueError("only a calibration TorqueLog can select parameters")
    if (
        isinstance(max_delay_steps, bool)
        or not isinstance(max_delay_steps, int)
        or not 0 <= max_delay_steps <= 10
    ):
        raise ValueError("max_delay_steps must be an integer in [0, 10]")
    if np.max(np.abs(log.commands_nm)) > 6.0 or np.any(np.ptp(log.commands_nm, axis=0) < 0.1):
        raise ValueError("fit needs varying, unsaturated excitation on every wheel")
    candidates = []
    for delay in range(max_delay_steps + 1):
        gains, taus, errors, statuses = [], [], [], []
        for axis in range(4):
            command, measured = log.commands_nm[:, axis], log.measured_nm[:, axis]

            def residual(parameters, command=command, measured=measured, delay=delay):
                return _linear_response(command, parameters[0], parameters[1], delay) - measured

            solution = least_squares(
                residual,
                (1.0, 0.01),
                bounds=((0.5, 0.001), (1.5, 0.03)),
                max_nfev=80,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            gains.append(float(solution.x[0]))
            taus.append(float(solution.x[1]))
            errors.append(float(np.mean(residual(solution.x) ** 2)))
            statuses.append(
                {
                    "success": bool(solution.success),
                    "nfev": int(solution.nfev),
                    "message": solution.message,
                }
            )
        candidates.append(
            {
                "delay_steps": delay,
                "gain": gains,
                "time_constant_s": taus,
                "calibration_mse_nm2": float(np.mean(errors)),
                "optimizers": statuses,
            }
        )
    eligible = [
        item for item in candidates if all(status["success"] for status in item["optimizers"])
    ]
    if not eligible:
        raise RuntimeError("no delay candidate completed continuous optimization")
    winner = min(eligible, key=lambda item: item["calibration_mse_nm2"])
    config = ActuatorChannelConfig(
        n_axes=4,
        physics_dt_s=log.physics_dt_s,
        gain=winner["gain"],
        time_constant_s=winner["time_constant_s"],
        delay_steps=winner["delay_steps"],
        torque_limit_nm=12.0,
    )
    return {
        "config": asdict(config),
        "candidates": candidates,
        "selection": "minimum calibration output-error MSE among converged candidates",
        "bounds": {
            "gain": [0.5, 1.5],
            "time_constant_s": [0.001, 0.03],
            "delay_steps": [0, max_delay_steps],
        },
        "observation": "synthetic noisy motor output torque [Nm], not encoder-only identification",
    }


class CausalTorqueInverse:
    """Feedforward inverse of fitted gain/lag; deliberately NOT an inverse of delay.

    v[k] = (r[k] - a_hat*r[k-1]) / (g_hat*(1-a_hat)). The physical channel
    still clips v and still delays it. No measured state, hidden torque, future
    command, or plant configuration is accessible through this Interface.
    """

    def __init__(self, fitted: ActuatorChannelConfig):
        if fitted.n_axes != 4 or fitted.physics_dt_s != PHYSICS_DT_S:
            raise ValueError("inverse requires four wheels at 2 ms")
        self.fitted = fitted
        self.gain = np.broadcast_to(fitted.gain, (4,)).copy()
        tau = np.broadcast_to(fitted.time_constant_s, (4,))
        if np.any(self.gain <= 0):
            raise ValueError("inverse needs positive fitted gains")
        self.a = np.zeros(4)
        mask = tau > 0
        self.a[mask] = np.exp(-PHYSICS_DT_S / tau[mask])
        self.reset()

    def reset(self):
        self.previous_desired_nm = np.zeros(4)

    def step(self, desired_nm):
        desired = np.asarray(desired_nm, dtype=float)
        if desired.shape != (4,) or not np.isfinite(desired).all():
            raise ValueError("desired torque must be a finite four-wheel vector")
        command = (desired - self.a * self.previous_desired_nm) / (self.gain * (1 - self.a))
        if not np.isfinite(command).all():
            raise ValueError("nonfinite inverse request")
        self.previous_desired_nm = desired.copy()
        return command


class WheelOnlyChannel(ActuatorChannel):
    """16-axis Adapter: actual four-wheel channel and independent 12-axis passthrough.

    config is only the 16-axis limit/clock contract required by D1Plant. Use
    topology_metadata for actual wheel-only gains, response and delay. Trace
    requested_nm is the drive request AFTER inverse feedforward; the original
    controller request is logged separately. No passive friction is added.
    """

    def __init__(self, wheel_config: ActuatorChannelConfig, fitted=None):
        if wheel_config.n_axes != 4 or wheel_config.physics_dt_s != PHYSICS_DT_S:
            raise ValueError("wheel channel requires four axes at 2 ms")
        if wheel_config.torque_limit_nm is None or not np.array_equal(
            np.broadcast_to(wheel_config.torque_limit_nm, (4,)), LIMITS_NM[WHEELS]
        ):
            raise ValueError("wheel channel limits must be the nominal 12 Nm")
        self.wheels = ActuatorChannel(wheel_config)
        self.legs = ActuatorChannel(
            ActuatorChannelConfig(n_axes=12, torque_limit_nm=tuple(LIMITS_NM[LEGS]))
        )
        self.inverse = None if fitted is None else CausalTorqueInverse(fitted)
        super().__init__(ActuatorChannelConfig(torque_limit_nm=tuple(LIMITS_NM)))

    def reset(self):
        self.wheels.reset()
        self.legs.reset()
        if self.inverse is not None:
            self.inverse.reset()
        self.physics_step_count = 0

    @property
    def topology_metadata(self):
        wheel = self.wheels.config
        gain, tau, delay = np.ones(16), np.zeros(16), np.zeros(16, dtype=int)
        gain[WHEELS], tau[WHEELS], delay[WHEELS] = (
            wheel.gain,
            wheel.time_constant_s,
            wheel.delay_steps,
        )
        return {
            "schema": "four-wheel-torque-response-twelve-leg-passthrough-v1",
            "wheel_indices": WHEELS.tolist(),
            "physics_dt_s": PHYSICS_DT_S,
            "actual_gain_per_axis": gain.tolist(),
            "actual_tau_s_per_axis": tau.tolist(),
            "actual_delay_physics_steps_per_axis": delay.tolist(),
            "actual_torque_limit_nm_per_axis": LIMITS_NM.tolist(),
            "fitted_inverse_config": None if self.inverse is None else asdict(self.inverse.fitted),
            "inverse_delay_compensation": "none; fitted delay is never advanced/cancelled",
            "friction": "existing MuJoCo passive friction unchanged; no additional friction",
            "trace_requested_nm": "drive request after optional fitted inverse",
            "installed": "after env.reset, before first env.step; no physics already consumed",
        }

    def step(self, torque_nm, joint_velocity_rps):
        torque, velocity = (
            np.asarray(torque_nm, dtype=float),
            np.asarray(joint_velocity_rps, dtype=float),
        )
        if (
            torque.shape != (16,)
            or velocity.shape != (16,)
            or not np.isfinite((torque, velocity)).all()
        ):
            raise ValueError("require finite 16-axis torque and velocity")
        request = torque[WHEELS] if self.inverse is None else self.inverse.step(torque[WHEELS])
        wheel = self.wheels.step(request, velocity[WHEELS])
        leg = self.legs.step(torque[LEGS], velocity[LEGS])
        fields = []
        for name in (
            "requested_nm",
            "limited_nm",
            "delayed_nm",
            "applied_nm",
            "joint_velocity_rps",
        ):
            value = np.empty(16)
            value[WHEELS], value[LEGS] = getattr(wheel, name), getattr(leg, name)
            fields.append(value)
        self.physics_step_count += 1
        return ActuatorTrace(*fields)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(_json_bytes(value))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot(output):
    paths = [Path(__file__), ROOT / "pyproject.toml", *sorted((ROOT / "src").rglob("*.py"))]
    paths += [
        p
        for p in sorted((ROOT / "src/wheel_legged_control/d1/assets").rglob("*"))
        if p.is_file() and p.suffix.lower() in (".stl", ".urdf", ".xml", ".txt")
    ]
    hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in paths}
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    _write_json(
        output / "source.json",
        {
            "sha256": hashes,
            "git_head": head,
            "archive_sha256": _sha256(output / "source.tar.gz"),
            "scope": "actual working-tree bytes, not a claim of clean HEAD",
        },
    )
    return hashes


def summarize(rows, terminal):
    if not rows or terminal.get("terminal_reason") is None:
        raise ValueError("need recorded rows and an explicit episode termination")
    result = {
        "steps": len(rows),
        "duration_s": rows[-1]["time_s"],
        "terminal_reason": terminal["terminal_reason"],
        "completed": terminal["terminal_reason"] == "time_limit",
    }
    for key, names in (
        ("velocity_rmse_mps", ("velocity_error_mps",)),
        ("yaw_rmse_rps", ("yaw_rate_error_rps",)),
        ("height_rmse_m", ("height_error_m",)),
        ("attitude_rmse_rad", ("roll_error_rad", "pitch_error_rad")),
    ):
        result[key] = float(
            np.sqrt(np.mean([sum(row[name] ** 2 for name in names) for row in rows]))
        )
    result["mechanical_activity_w"] = float(np.mean([row["mechanical_power_w"] for row in rows]))
    result["drive_input_clipped_fraction"] = float(
        np.mean([row["torque_input_clipped_fraction"] for row in rows])
    )
    result["quality_pass"] = result["completed"] and all(
        result[key] <= limit for key, limit in QUALITY.items()
    )
    result["terrain_exposure"] = terminal["terrain_exposure"]
    return result


def evaluate_case(directory, *, case, fitted_config, seed=17, duration_s=60.0, terrain_index=0):
    """One real serial D1 rollout. Truth-channel settings never come from the fit."""
    import mujoco

    from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
    from wheel_legged_control.d1.locomotion_terrain import locomotion_terrain_configs

    if case not in CASES or terrain_index not in (0, 1):
        raise ValueError("only declared cases and the two development roads are allowed")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("case output must be new")
    env = D1LocomotionEnv(
        episode_seconds=duration_s,
        command_mode="development",
        terrain=locomotion_terrain_configs("development")[terrain_index],
    )
    start = time.monotonic()
    try:
        env.reset(seed=seed)
        wheel_config = (
            ActuatorChannelConfig(n_axes=4, torque_limit_nm=12.0)
            if case == "ideal"
            else SYNTHETIC_WHEELS
        )
        channel = WheelOnlyChannel(
            wheel_config, fitted_config if case == "fitted_compensation" else None
        )
        assert env.plant.data.time == 0.0
        env.plant.actuator_channel = channel
        directory.mkdir(parents=True, exist_ok=False)
        mujoco.mj_saveModel(env.plant.model, str(directory / "model.mjb"), None)
        parameters = {
            "case": case,
            "seed": seed,
            "duration_s": duration_s,
            "terrain_index": terrain_index,
            "terrain_split": "development",
            "environment_reset_metadata": env.episode_metadata,
            "installed_channel": channel.topology_metadata,
            "reset_metadata_actuator_scope": "pre-install nominal contract; installed_channel is authoritative",
            "model_sha256": _sha256(directory / "model.mjb"),
        }
        _write_json(directory / "parameters.json", parameters)
        rows, requested, traces, states = (
            [],
            [],
            [],
            [np.r_[env.plant.data.qpos, env.plant.data.qvel].copy()],
        )
        while True:
            _, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
            row = dict(
                info["metrics"],
                reward=reward,
                **{f"command_{key}": value for key, value in info["command"].items()},
                nonflat_now=int(info["terrain_exposure"]["nonflat_now"]),
            )
            rows.append(row)
            interval = env.plant.last_control_interval_actuator_traces
            for trace in interval:
                requested.append(env.last_transition.requested_torque_nm)
                traces.append(
                    np.stack(
                        [
                            trace.requested_nm,
                            trace.limited_nm,
                            trace.delayed_nm,
                            trace.applied_nm,
                            trace.joint_velocity_rps,
                        ]
                    )
                )
            states.append(np.r_[env.plant.data.qpos, env.plant.data.qvel].copy())
            if terminated or truncated:
                break
        if channel.physics_step_count != 5 * len(rows):
            raise RuntimeError("mixed actuator was not advanced exactly five times per decision")
        with (directory / "metrics.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        values = np.asarray(traces)
        np.savez_compressed(
            directory / "physical_trace.npz",
            physics_dt_s=PHYSICS_DT_S,
            controller_requested_nm=np.asarray(requested),
            drive_requested_nm=values[:, 0],
            drive_limited_nm=values[:, 1],
            delayed_nm=values[:, 2],
            applied_nm=values[:, 3],
            joint_velocity_rad_s=values[:, 4],
            decision_qpos_qvel=np.asarray(states),
        )
        result = summarize(rows, info)
        result.update(
            {
                "case": case,
                "terrain_index": terrain_index,
                "seed": seed,
                "wall_s": time.monotonic() - start,
                "physics_steps": channel.physics_step_count,
                "parameters_sha256": _sha256(directory / "parameters.json"),
                "torque_tracking_rmse_nm": float(
                    np.sqrt(np.mean((values[:, 3, WHEELS] - np.asarray(requested)[:, WHEELS]) ** 2))
                ),
            }
        )
        _write_json(directory / "summary.json", result)
        return result, rows
    finally:
        env.close()


def run(output, *, seed=17, duration_s=60.0):
    output = Path(output)
    if output.exists():
        raise FileExistsError("output must be a new directory; failed runs are retained")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        not math.isfinite(duration_s)
        or duration_s < CONTROL_DT_S
        or not np.isclose(
            duration_s / CONTROL_DT_S, round(duration_s / CONTROL_DT_S), rtol=0, atol=1e-8
        )
    ):
        raise ValueError("duration must be a positive integer number of 10 ms ticks")
    output.mkdir(parents=True, exist_ok=False)
    source = _source_snapshot(output)
    protocol = {
        "schema": "synthetic-torque-identification-d1-transfer-v1",
        "seed": seed,
        "duration_s": duration_s,
        "full_60s_protocol": duration_s == 60.0,
        "calibration_samples": 2000,
        "validation_samples": 1500,
        "noise_std_nm": 0.01,
        "calibration_seed": seed + 10000,
        "validation_seed": seed + 20000,
        "known_synthetic_wheels": asdict(SYNTHETIC_WHEELS),
        "physics_dt_s": PHYSICS_DT_S,
        "control_dt_s": CONTROL_DT_S,
        "cases": CASES,
        "terrain_split": "development",
        "terrain_indices": [0, 1],
        "quality_thresholds": QUALITY,
        "causal_compensation": "fitted gain/lag inverse from current/previous request, no delay cancellation",
        "parameter_selection": "calibration only; no validation/terrain tuning or holdout evaluation",
        "limits": "synthetic torque measurement; no real D1 torque sensing or hardware identification claim",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    _write_json(output / "protocol.json", protocol)
    logs = {}
    for split, samples, offset in (("calibration", 2000, 10000), ("validation", 1500, 20000)):
        log, truth = record_synthetic_log(split, seed=seed + offset, samples=samples)
        logs[split] = log
        np.savez_compressed(
            output / f"{split}.npz",
            commands_nm=log.commands_nm,
            measured_nm=log.measured_nm,
            diagnostic_truth_nm=truth,
            physics_dt_s=PHYSICS_DT_S,
            split=split,
        )
    fit = fit_torque_channel(logs["calibration"])
    fitted_config = ActuatorChannelConfig(**fit["config"])
    for split, log in logs.items():
        prediction = channel_response(log.commands_nm, fitted_config)
        fit[f"{split}_prediction_rmse_nm"] = np.sqrt(
            np.mean((prediction - log.measured_nm) ** 2, axis=0)
        ).tolist()
        np.savez_compressed(output / f"{split}_prediction.npz", prediction_nm=prediction)
    _write_json(output / "fit.json", fit)
    cases, paired = [], []
    for road in (0, 1):
        by_case = {}
        for case in CASES:
            summary, rows = evaluate_case(
                output / f"road{road}_{case}",
                case=case,
                fitted_config=fitted_config,
                seed=seed,
                duration_s=duration_s,
                terrain_index=road,
            )
            cases.append(summary)
            by_case[case] = rows
            print(
                f"road{road} {case}: {summary['terminal_reason']}, {summary['duration_s']:.2f}s",
                flush=True,
            )
        common = min(len(value) for value in by_case.values())
        paired.append(
            {
                "terrain_index": road,
                "common_prefix_steps": common,
                "common_prefix_seconds": common * CONTROL_DT_S,
                "velocity_rmse_mps_common_prefix": {
                    case: float(
                        np.sqrt(np.mean([row["velocity_error_mps"] ** 2 for row in rows[:common]]))
                    )
                    for case, rows in by_case.items()
                },
                "scope": "common time prefix; geometric paths may differ, no survival advantage implied",
            }
        )
    changed = [name for name, digest in source.items() if _sha256(ROOT / name) != digest]
    _write_json(output / "source_consistency.json", {"unchanged": not changed, "changed": changed})
    if changed:
        raise RuntimeError("source changed during experiment; no completed report")
    report = {
        "cases": cases,
        "paired": paired,
        "fit_sha256": _sha256(output / "fit.json"),
        "protocol_sha256": _sha256(output / "protocol.json"),
        "source_archive_sha256": _sha256(output / "source.tar.gz"),
        "interpretation": "report each metric and termination; improved torque fit does not establish locomotion benefit",
    }
    _write_json(output / "report.json", report)
    artifacts = {
        str(path.relative_to(output)): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    _write_json(output / "artifact_manifest.json", {"status": "complete", "sha256": artifacts})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--duration", type=float, default=60.0, help="seconds; shorter runs are smoke tests"
    )
    args = parser.parse_args(argv)
    try:
        run(args.output, seed=args.seed, duration_s=args.duration)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Actuator transfer failed: {exc}; retain partial output for inspection.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
