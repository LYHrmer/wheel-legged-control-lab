"""Log-only, bounded actuator identification on the independent MuJoCo bench.

This is a small PACE-inspired exercise, not a port of its Isaac/CMA-ES pipeline.
Known load geometry, command units, sample period and torque limit are inputs.
Only calibration trajectories select parameters; other splits are prediction-only.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .actuator_bench import ActuatorBench, ActuatorParameters, BenchConfig

PARAMETER_NAMES = ("armature", "damping", "coulomb_friction")
PARAMETER_LOWER = np.asarray((0.002, 0.001, 0.0))
PARAMETER_UPPER = np.asarray((0.060, 0.350, 0.200))
POSITION_SCALE_RAD = 0.05
VELOCITY_SCALE_RAD_S = 0.5
CSV_COLUMNS = (
    "time_s",
    "command_torque_nm",
    "position_rad",
    "velocity_rad_s",
    "next_position_rad",
    "next_velocity_rad_s",
)


@dataclass(frozen=True)
class ActuatorLog:
    """T commands and T+1 encoder states; command[k] follows state[k].

    Every trajectory starts with zero pending motor commands. Velocity is an
    explicit synthetic encoder-velocity channel, not an assumed torque sensor.
    Positions are continuous angles, not angles wrapped to [-pi, pi].
    """

    config: BenchConfig
    commands: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    name: str = "trajectory"
    split: str = "calibration"

    def __post_init__(self) -> None:
        if not isinstance(self.config, BenchConfig):
            raise TypeError("config must be BenchConfig")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if self.split not in ("calibration", "validation", "holdout"):
            raise ValueError("split must be calibration, validation or holdout")
        for name in ("commands", "positions", "velocities"):
            array = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite one-dimensional array")
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=np.float64))
        if len(self.commands) < 2 or self.positions.shape != (len(self.commands) + 1,):
            raise ValueError("require T >= 2 commands and T+1 positions")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("require T+1 velocities")


def save_actuator_log(path: str | Path, log: ActuatorLog) -> None:
    """Write a self-describing CSV exclusively; never overwrite an experiment."""
    header = {
        "schema_version": 1,
        "config": asdict(log.config),
        "name": log.name,
        "split": log.split,
        "sampling": "state_then_command_then_next_state",
        "initial_command_history": "zero",
    }
    rows = np.column_stack(
        (
            np.arange(len(log.commands)) * log.config.dt,
            log.commands,
            log.positions[:-1],
            log.velocities[:-1],
            log.positions[1:],
            log.velocities[1:],
        )
    )
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        stream.write("# " + json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def load_actuator_log(path: str | Path) -> ActuatorLog:
    """Reject mismatched clocks or discontinuous state rows instead of resampling."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        first = stream.readline()
        if not first.startswith("# "):
            raise ValueError("missing actuator log metadata")
        try:
            metadata = json.loads(first[2:])
            if (
                metadata["schema_version"] != 1
                or metadata["sampling"] != "state_then_command_then_next_state"
                or metadata["initial_command_history"] != "zero"
            ):
                raise ValueError("unsupported actuator log schema or timing")
            config = BenchConfig(**metadata["config"])
            name, split = metadata["name"], metadata["split"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid actuator log metadata") from error
        reader = csv.reader(stream)
        if next(reader, None) != list(CSV_COLUMNS):
            raise ValueError("unexpected actuator log columns")
        try:
            rows = np.asarray(list(reader), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid numeric actuator log rows") from error
    if rows.ndim != 2 or rows.shape[1] != len(CSV_COLUMNS) or rows.shape[0] < 2:
        raise ValueError("actuator log needs at least two complete rows")
    if not np.all(np.isfinite(rows)):
        raise ValueError("actuator log rows must be finite")
    if not np.allclose(rows[:, 0], np.arange(len(rows)) * config.dt, rtol=0, atol=1e-10):
        raise ValueError("actuator log clock must start at zero and match config.dt")
    if not np.allclose(rows[:-1, 4:6], rows[1:, 2:4], rtol=0, atol=1e-12):
        raise ValueError("actuator log next states must match the following row")
    return ActuatorLog(
        config,
        rows[:, 1],
        np.r_[rows[:, 2], rows[-1, 4]],
        np.r_[rows[:, 3], rows[-1, 5]],
        name=name,
        split=split,
    )


def predict_actuator_log(log: ActuatorLog, parameters: ActuatorParameters) -> np.ndarray:
    """Free-running replay from the first measured state, without teacher forcing."""
    bench = ActuatorBench(log.config, parameters)
    bench.reset(log.positions[0], log.velocities[0])
    states = np.empty((len(log.commands) + 1, 2))
    states[0] = bench.state
    for index, command in enumerate(log.commands):
        states[index + 1] = bench.step(command)
    if not np.all(np.isfinite(states)):
        raise ValueError("non-finite actuator prediction")
    return states


def prediction_metrics(log: ActuatorLog, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(log.commands) + 1, 2) or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction must contain T+1 finite [position, velocity] rows")
    error = prediction[1:] - np.column_stack((log.positions[1:], log.velocities[1:]))
    rms = np.sqrt(np.mean(np.square(error), axis=0))
    return {"position_rmse_rad": float(rms[0]), "velocity_rmse_rad_s": float(rms[1])}


@dataclass(frozen=True)
class FitResult:
    parameters: ActuatorParameters
    candidates: tuple[dict, ...]
    jacobian_singular_values: tuple[float, ...]
    jacobian_rank: int


def fit_actuator_parameters(
    logs: Sequence[ActuatorLog],
    *,
    max_delay_steps: int = 4,
    max_nfev: int = 30,
) -> FitResult:
    """Enumerate integer delays and fit three bounded continuous parameters.

    The least-squares variables are normalized to [0, 1]. Fixed physical
    residual scales prevent mixing radians and rad/s without a declared weight.
    Report all candidate statuses: minimum training loss is not identifiability
    or held-out validation. Jacobian diagnostics cover only continuous variables
    locally, conditional on the selected integer delay.
    """
    logs = tuple(logs)
    if not logs or any(not isinstance(log, ActuatorLog) for log in logs):
        raise ValueError("provide at least one ActuatorLog")
    if any(log.split != "calibration" for log in logs):
        raise ValueError("only calibration logs may select parameters")
    if any(log.config != logs[0].config for log in logs):
        raise ValueError("all calibration logs must use the same known bench config")
    for name, value, minimum in (
        ("max_delay_steps", max_delay_steps, 0),
        ("max_nfev", max_nfev, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not any(np.ptp(log.commands) > 1e-6 for log in logs):
        raise ValueError("calibration needs varying excitation commands")
    span = PARAMETER_UPPER - PARAMETER_LOWER
    initial = (np.asarray((0.01, 0.08, 0.025)) - PARAMETER_LOWER) / span
    count = sum(len(log.commands) for log in logs)
    candidates, optimizations = [], []

    def unpack(normalized: np.ndarray, delay: int) -> ActuatorParameters:
        values = PARAMETER_LOWER + span * normalized
        return ActuatorParameters(*values, delay_steps=delay)

    for delay in range(int(max_delay_steps) + 1):

        def residual(normalized: np.ndarray, candidate_delay: int = delay) -> np.ndarray:
            parameters = unpack(normalized, candidate_delay)
            parts = []
            for log in logs:
                prediction = predict_actuator_log(log, parameters)
                parts.extend(
                    (
                        (prediction[1:, 0] - log.positions[1:]) / POSITION_SCALE_RAD,
                        (prediction[1:, 1] - log.velocities[1:]) / VELOCITY_SCALE_RAD_S,
                    )
                )
            return np.concatenate(parts) / np.sqrt(count)

        result = least_squares(
            residual,
            initial,
            bounds=(0.0, 1.0),
            max_nfev=int(max_nfev),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        parameters = unpack(result.x, delay)
        score = float(2.0 * result.cost)
        candidates.append(
            {
                "parameters": asdict(parameters),
                "calibration_loss": score,
                "optimizer_success": bool(result.success),
                "nfev": int(result.nfev),
                "message": str(result.message),
            }
        )
        optimizations.append(result)
    selected = int(np.argmin([candidate["calibration_loss"] for candidate in candidates]))
    singular = np.linalg.svd(optimizations[selected].jac, compute_uv=False)
    rank = int(np.sum(singular > (singular[0] * 1e-7 if len(singular) else 0.0)))
    return FitResult(
        ActuatorParameters(**candidates[selected]["parameters"]),
        tuple(candidates),
        tuple(float(value) for value in singular),
        rank,
    )
