"""A two-tick leg-velocity predictor fitted to the archived round-85 action recordings."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import replace
from itertools import chain
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.terrain_axle86 import ARCHIVE_SHA256, SEEDS

DT_S = .01
RIDGE = .01
MIN_IMPROVEMENT = .10


def _joint_addresses(cp, name):
    encoded = cp[0]["source"]["fields"]["estimator"]["fields"][name]
    if encoded["shape"] != [16] or encoded["dtype"] != "<i4":
        raise ValueError(f"unexpected archived {name} joint-address format")
    all_axes = np.frombuffer(bytes.fromhex(encoded["hex"]), dtype="<i4")
    if len(all_axes) != 16 or len(set(all_axes)) != 16 or np.any(all_axes < 0):
        raise ValueError(f"invalid archived {name} joint addresses")
    return all_axes.reshape(4, 4)[:, :3]


def _rows(z, path, qadr, dadr):
    with z.open(path) as stream:
        for line in stream:
            row = json.loads(line)
            tick = row["step"]
            if abs(row["time_s"] - tick * DT_S) > 1e-9:
                raise ValueError("recording control tick and timestamp disagree")
            pos = np.asarray(row["qpos"], dtype=float)[qadr]
            vel = np.asarray(row["qvel"], dtype=float)[dadr]
            tau = np.asarray(row["applied_torque_nm"], dtype=float).reshape(4, 4)[:, :3]
            if not np.isfinite((pos, vel, tau)).all():
                raise ValueError("nonfinite joint data in damping source")
            yield tick, pos, vel, tau


def _transitions(rows, unique):
    history = deque(maxlen=2)
    count = 0
    for row in rows:
        if history and row[0] != history[-1][0] + 1:
            raise ValueError("nonconsecutive ticks in damping source")
        if len(history) == 2:
            old, previous = history
            sample = np.concatenate((old[1], old[2], previous[3], row[3], row[2]), axis=1)
            # Exact equality of all four legs' inputs and four current-velocity targets.
            key = sample.tobytes()
            unique.setdefault(key, sample)
            count += 1
        history.append(row)
    return tuple(history), count


def _read_groups(archive):
    groups = {seed: {} for seed in SEEDS}
    counts = []
    with Path(archive).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("input does not match the published round-85 recordings")
        with ZipFile(stream) as z:
            for seed, unique in groups.items():
                raw = 0
                for terrain in ("up", "down"):
                    start = f"trials/action/{terrain}_{seed}/"
                    cp = json.loads(z.read(start + "checkpoint_python.json"))
                    qadr, dadr = _joint_addresses(cp, "_qadr"), _joint_addresses(cp, "_dadr")
                    tail, n = _transitions(_rows(z, start + "prefix.jsonl", qadr, dadr), unique)
                    if len(tail) != 2:
                        raise ValueError("checkpoint lacks two consecutive prefix rows")
                    raw += n
                    for arm in ("negative", "zero", "positive"):
                        rows = _rows(z, start + arm + ".jsonl", qadr, dadr)
                        _, n = _transitions(chain(tail, rows), unique)
                        if n == 0:
                            raise ValueError("source branch contains no control ticks")
                        raw += n
                counts.append({"seed": seed, "raw": raw, "unique": len(unique),
                               "duplicates": raw - len(unique)})
    return groups, counts


def _fit(samples):
    # Each sample is (4 legs, 12 inputs + 3 current velocities), ordered FL/FR/RL/RR.
    data = np.stack(samples)
    x, current = data[:, :, :12], data[:, :, 12:]
    delta = current - x[:, :, 3:6]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    constant = np.ptp(x, axis=0) == 0
    # Use exact constants; summation round-off can give a constant column a tiny variance.
    mean[constant] = x[0][constant]
    scale[constant | (scale == 0)] = 1.
    standardized = (x - mean) / scale
    intercept = delta.mean(axis=0)
    coefficients = []
    for leg in range(4):
        design = standardized[:, leg, :]
        targets = delta[:, leg, :] - intercept[leg]
        gram = design.T @ design / len(x) + RIDGE * np.eye(12)
        coefficients.append(np.linalg.solve(gram, design.T @ targets / len(x)))
    return {
        "schema": "d1-leg-damping-ridge86-v1", "dt_s": DT_S, "ridge": RIDGE,
        "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": np.asarray(coefficients).tolist(), "delta_intercept": intercept.tolist(),
        "velocity_min": current.min(axis=0).tolist(), "velocity_max": current.max(axis=0).tolist(),
    }


def _model_arrays(model):
    if model["schema"] != "d1-leg-damping-ridge86-v1" or model["dt_s"] != DT_S:
        raise ValueError("incompatible damping model")
    shapes = {"mean": (4, 12), "scale": (4, 12), "weights": (4, 12, 3),
              "delta_intercept": (4, 3), "velocity_min": (4, 3), "velocity_max": (4, 3)}
    arrays = {key: np.asarray(model[key], dtype=float) for key in shapes}
    if any(arr.shape != shapes[key] or not np.isfinite(arr).all() for key, arr in arrays.items()):
        raise ValueError("invalid damping model dimensions or coefficients")
    if np.any(arrays["scale"] <= 0) or np.any(arrays["velocity_min"] > arrays["velocity_max"]):
        raise ValueError("invalid damping model scales or bounds")
    return arrays


def _predict(model, features):
    standardized = (features - model["mean"]) / model["scale"]
    delta = np.einsum("...li,lij->...lj", standardized, model["weights"]) + model["delta_intercept"]
    return np.clip(features[..., 3:6] + delta, model["velocity_min"], model["velocity_max"])


def fit_predictor(archive: Path) -> tuple[dict, dict | None]:
    """Validate once per held-out development seed, fit a final model only if all qualify."""
    groups, counts = _read_groups(archive)
    pooled = {k: row for group in groups.values() for k, row in group.items()}
    folds = []
    for seed, held in groups.items():
        train = [row for k, row in pooled.items() if k not in held]
        if not train or not held:
            raise ValueError("empty train or validation group after exact deduplication")
        data = np.stack(list(held.values()))
        truth, delayed = data[:, :, 12:], data[:, :, 3:6]
        prediction = _predict(_model_arrays(_fit(train)), data[:, :, :12])
        hold_rmse = float(np.sqrt(np.mean((truth - delayed) ** 2)))
        predictor_rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
        improvement = 1. - predictor_rmse / hold_rmse if hold_rmse > 0 else None
        folds.append({
            "seed": seed, "train_unique": len(train), "validation_unique": len(held),
            "hold_rmse_rad_s": hold_rmse, "predictor_rmse_rad_s": predictor_rmse,
            "relative_improvement": improvement,
            "passed": improvement is not None and improvement >= MIN_IMPROVEMENT,
        })
    passed = all(f["passed"] for f in folds)
    report = {
        "archive_sha256": ARCHIVE_SHA256, "samples_by_seed": counts,
        "unique_across_all_seeds": len(pooled), "folds": folds, "passed": passed,
        "minimum_relative_improvement": MIN_IMPROVEMENT, "ridge_in_mean_loss": RIDGE,
        "axes": "FL,FR,RL,RR; hip,thigh,calf per leg",
        "inputs": ["q(t-2)", "dq(t-2)", "applied_tau(t-2,t-1]", "applied_tau(t-1,t]"],
        "target": "dq(t) - dq(t-2), per leg",
        "offline_encoder_source": "Recorded native qpos/qvel; per-tick noisy encoders were not logged.",
        "runtime_encoder_source": "Delayed measured joint positions/velocities; two applied controls.",
    }
    return report, _fit(list(pooled.values())) if passed else None


def fit_damping(archive: Path, output: Path) -> dict:
    """Persist one fit with exclusive paths; failed or interrupted evidence stays in place."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report, model = fit_predictor(archive)
    if model is not None:
        with (output / "model.json").open("x") as stream:
            json.dump(model, stream, allow_nan=False, indent=2)
    with (output / "fit.json").open("x") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
    return report


class DampedLowLevel:
    """Replace leg damping inputs for the existing round-85 VMC allocation and clipping path."""

    def __init__(self, low_level, model):
        if getattr(low_level, "contact_allocator", None) is not None:
            raise ValueError("the round-85 legacy VMC support path is required")
        self.inner = low_level
        self.model = _model_arrays(model)
        self.history = deque(maxlen=2)
        self.first_control_s = None
        self.last_measurement_s = None
        self.prediction_count = 0
        self.last_prediction_used = False

    def __getattr__(self, name):
        # Deepcopy reconstructs objects before the instance fields exist.
        if name.startswith("__") or "inner" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.inner, name)

    @property
    def requested_transfer_n(self):
        return self.inner.requested_transfer_n

    @requested_transfer_n.setter
    def requested_transfer_n(self, value):
        self.inner.requested_transfer_n = value

    def reset(self):
        self.inner.reset()
        self.history.clear()
        self.first_control_s = self.last_measurement_s = None
        self.prediction_count = 0
        self.last_prediction_used = False

    def record_applied(self, torque, end_time_s):
        value = np.asarray(torque, dtype=float)
        if value.shape != (16,) or not np.isfinite(value).all() or not np.isfinite(end_time_s):
            raise ValueError("expected a timestamp and sixteen finite applied torques")
        if self.history and abs(end_time_s - self.history[-1][0] - DT_S) > 1e-9:
            raise ValueError("applied-torque history must advance by one tick")
        self.history.append((float(end_time_s), value.reshape(4, 4)[:, :3].copy()))

    def compute(self, command, state, **efforts):
        now, stamp = float(state.control_time_s), float(state.measurement_time_s)
        if self.first_control_s is None:
            self.first_control_s = now
        fresh = self.last_measurement_s is None or stamp > self.last_measurement_s + 1e-9
        self.last_measurement_s = stamp
        self.last_prediction_used = False
        if fresh and now - self.first_control_s >= 2 * DT_S - 1e-9:
            if abs(now - stamp - 2 * DT_S) > 1e-9:
                raise ValueError("damping predictor requires the measured 20 ms encoder delay")
            if (len(self.history) != 2 or abs(self.history[0][0] - now + DT_S) > 1e-9
                    or abs(self.history[1][0] - now) > 1e-9):
                raise ValueError("missing two applied-torque intervals at the current control time")
            pos = state.joint_position.reshape(4, 4)[:, :3]
            velocity = state.joint_velocity.reshape(4, 4).copy()
            features = np.concatenate((pos, velocity[:, :3], self.history[0][1], self.history[1][1]),
                                      axis=1)
            velocity[:, :3] = _predict(self.model, features)
            state = replace(state, joint_velocity=velocity.ravel())
            self.prediction_count += 1
            self.last_prediction_used = True
        return self.inner.compute(command, state, **efforts)


def attach_damping(ep, damping_model: dict) -> None:
    """Wrap either existing axle allocator, after creating the original outer controller."""
    if isinstance(ep.controller.low_level, DampedLowLevel):
        raise TypeError("a damping predictor is already attached")
    ep.controller.low_level = DampedLowLevel(ep.controller.low_level, damping_model)


def record_applied(ep, applied_torque_nm, end_time_s: float) -> None:
    """Post-step feedback from actuator command readback and control clock. No state truth."""
    low = ep.controller.low_level
    if isinstance(low, DampedLowLevel):
        low.record_applied(applied_torque_nm, end_time_s)
