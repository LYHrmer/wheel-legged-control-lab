"""Position-only encoder identification on the same single-axis MuJoCo bench.

Only requested torque commands, an encoder position sequence, the known bench
config and an explicit "started at rest" acquisition declaration are inputs.
There is no velocity channel here: rest is a declared property of how the log
was acquired, never something inferred from position differences. Measured
output torques, plant objects and generating parameters stay out of this module.
This is a small PACE-inspired exercise, not a port of its full pipeline.
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
from .actuator_identification import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    POSITION_SCALE_RAD,
    FitResult,
)

SCHEMA_NAME = "encoder_position_only"
SCHEMA_VERSION = 1
SAMPLING = "position_then_command_then_next_position"
INITIAL_COMMAND_HISTORY = "zero"
INITIAL_STATE = "known_rest"
CSV_COLUMNS = (
    "time_s",
    "command_torque_nm",
    "position_rad",
    "next_position_rad",
)
METADATA_KEYS = frozenset(
    (
        "schema",
        "schema_version",
        "config",
        "name",
        "split",
        "sampling",
        "initial_command_history",
        "initial_state",
        "known_initial_rest",
    )
)
CONFIG_KEYS = frozenset(("kind", "dt", "torque_limit_nm"))
CLOCK_TOLERANCE_S = 1e-10
CONTINUITY_TOLERANCE_RAD = 1e-12
INITIAL_PARAMETERS = (0.01, 0.08, 0.025)


@dataclass(frozen=True)
class EncoderLog:
    """T commands and T+1 encoder positions; command[k] follows position[k].

    ``known_initial_rest`` must be declared ``True`` by the acquisition code:
    the trajectory starts at zero velocity with an empty command queue. There is
    deliberately no velocity, applied-torque or truth field. Positions are
    continuous angles, not angles wrapped to [-pi, pi].
    """

    config: BenchConfig
    commands: np.ndarray
    positions: np.ndarray
    known_initial_rest: bool
    name: str = "trajectory"
    split: str = "calibration"

    def __post_init__(self) -> None:
        if not isinstance(self.config, BenchConfig):
            raise TypeError("config must be BenchConfig")
        if not isinstance(self.known_initial_rest, bool) or self.known_initial_rest is not True:
            raise ValueError("known_initial_rest must be declared exactly True")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if self.split not in ("calibration", "validation", "holdout"):
            raise ValueError("split must be calibration, validation or holdout")
        for name in ("commands", "positions"):
            if np.iscomplexobj(getattr(self, name)):
                raise ValueError(f"{name} must contain real values")
            array = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite one-dimensional array")
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=np.float64))
        if len(self.commands) < 2 or self.positions.shape != (len(self.commands) + 1,):
            raise ValueError("require T >= 2 commands and T+1 positions")


def save_encoder_log(path: str | Path, log: EncoderLog) -> None:
    """Write the position-only CSV exclusively; never overwrite an experiment."""
    if not isinstance(log, EncoderLog):
        raise TypeError("log must be EncoderLog")
    header = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "config": asdict(log.config),
        "name": log.name,
        "split": log.split,
        "sampling": SAMPLING,
        "initial_command_history": INITIAL_COMMAND_HISTORY,
        "initial_state": INITIAL_STATE,
        "known_initial_rest": log.known_initial_rest,
    }
    rows = np.column_stack(
        (
            np.arange(len(log.commands)) * log.config.dt,
            log.commands,
            log.positions[:-1],
            log.positions[1:],
        )
    )
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        stream.write("# " + json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate encoder log metadata keys")
    return dict(pairs)


def _reject_constant(name: str) -> float:
    raise ValueError(f"encoder log metadata must be finite, got {name}")


def _read_metadata(line: str) -> tuple[BenchConfig, str, str]:
    if not line.startswith("# "):
        raise ValueError("missing encoder log metadata")
    try:
        metadata = json.loads(
            line[2:], object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as error:
        raise ValueError("invalid encoder log metadata") from error
    if not isinstance(metadata, dict) or set(metadata) != METADATA_KEYS:
        raise ValueError("encoder log metadata keys must match the position-only schema")
    if (
        metadata["schema"] != SCHEMA_NAME
        or type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
        or metadata["sampling"] != SAMPLING
        or metadata["initial_command_history"] != INITIAL_COMMAND_HISTORY
        or metadata["initial_state"] != INITIAL_STATE
    ):
        raise ValueError("unsupported encoder log schema, timing or command history")
    rest = metadata["known_initial_rest"]
    if not isinstance(rest, bool) or rest is not True:
        raise ValueError("encoder log must declare known_initial_rest true")
    config_fields = metadata["config"]
    if not isinstance(config_fields, dict) or set(config_fields) != CONFIG_KEYS:
        raise ValueError("encoder log config keys must match the known bench config")
    try:
        config = BenchConfig(**config_fields)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid encoder log bench config") from error
    return config, metadata["name"], metadata["split"]


def load_encoder_log(path: str | Path) -> EncoderLog:
    """Reject mismatched clocks or discontinuous position rows instead of resampling."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        config, name, split = _read_metadata(stream.readline())
        reader = csv.reader(stream)
        if next(reader, None) != list(CSV_COLUMNS):
            raise ValueError("unexpected encoder log columns")
        try:
            rows = np.asarray(list(reader), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid numeric encoder log rows") from error
    if rows.ndim != 2 or rows.shape[1] != len(CSV_COLUMNS) or rows.shape[0] < 2:
        raise ValueError("encoder log needs at least two complete rows")
    if not np.all(np.isfinite(rows)):
        raise ValueError("encoder log rows must be finite")
    if not np.allclose(
        rows[:, 0], np.arange(len(rows)) * config.dt, rtol=0, atol=CLOCK_TOLERANCE_S
    ):
        raise ValueError("encoder log clock must start at zero and match config.dt")
    if not np.allclose(rows[:-1, 3], rows[1:, 2], rtol=0, atol=CONTINUITY_TOLERANCE_RAD):
        raise ValueError("encoder log next positions must match the following row")
    return EncoderLog(
        config,
        rows[:, 1],
        np.r_[rows[:, 2], rows[-1, 3]],
        known_initial_rest=True,
        name=name,
        split=split,
    )


def predict_encoder_log(log: EncoderLog, parameters: ActuatorParameters) -> np.ndarray:
    """Free-run positions from the first sample and the declared rest, no teacher forcing."""
    if not isinstance(log, EncoderLog):
        raise TypeError("log must be EncoderLog")
    bench = ActuatorBench(log.config, parameters)
    bench.reset(log.positions[0], 0.0)
    positions = np.empty(len(log.commands) + 1)
    positions[0] = bench.state[0]
    for index, command in enumerate(log.commands):
        positions[index + 1] = bench.step(command)[0]
    if not np.all(np.isfinite(positions)):
        raise ValueError("non-finite encoder prediction")
    return positions


def encoder_prediction_metrics(log: EncoderLog, prediction: np.ndarray) -> dict[str, float]:
    """Position RMSE over the predicted samples, excluding the known first sample."""
    if np.iscomplexobj(prediction):
        raise ValueError("prediction must contain real positions")
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(log.commands) + 1,) or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction must contain T+1 finite positions")
    error = prediction[1:] - log.positions[1:]
    return {"position_rmse_rad": float(np.sqrt(np.mean(np.square(error))))}


def fit_encoder_parameters(
    logs: Sequence[EncoderLog],
    *,
    max_delay_steps: int = 4,
    max_nfev: int = 30,
) -> FitResult:
    """Enumerate integer delays and fit the three bounded continuous parameters.

    Residuals are position-only, in units of ``POSITION_SCALE_RAD``; there is no
    velocity channel to weight. Bounds, names and the ``FitResult`` container are
    the ones already used by the q+v fitter. All candidate statuses are reported:
    minimum calibration loss is neither identifiability nor held-out validation.
    Jacobian diagnostics are local and conditional on the selected integer delay.
    """
    logs = tuple(logs)
    if not logs or any(not isinstance(log, EncoderLog) for log in logs):
        raise ValueError("provide at least one EncoderLog")
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
    initial = (np.asarray(INITIAL_PARAMETERS) - PARAMETER_LOWER) / span
    count = sum(len(log.commands) for log in logs)
    candidates, optimizations = [], []

    def unpack(normalized: np.ndarray, delay: int) -> ActuatorParameters:
        values = PARAMETER_LOWER + span * normalized
        named = dict(zip(PARAMETER_NAMES, values, strict=True))
        return ActuatorParameters(**named, delay_steps=delay)

    for delay in range(int(max_delay_steps) + 1):

        def residual(normalized: np.ndarray, candidate_delay: int = delay) -> np.ndarray:
            parameters = unpack(normalized, candidate_delay)
            parts = [
                (predict_encoder_log(log, parameters)[1:] - log.positions[1:]) / POSITION_SCALE_RAD
                for log in logs
            ]
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
        candidates.append(
            {
                "parameters": asdict(unpack(result.x, delay)),
                "calibration_loss": float(2.0 * result.cost),
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
