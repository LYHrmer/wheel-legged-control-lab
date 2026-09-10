"""Audit the friction/delay study from saved arrays, without loading a simulator.

No fitted model is rerun here. The independent checks cover archived prediction
errors, causal row timing, paired sensor draws, PI/FF arithmetic, and file hashes.
Optimizer convergence and identifiability remain separate from control benefit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

DT = 0.002
POSITION_SCALE_RAD = 0.05
PROFILES = ("tracking", "reversal", "stress")
TRAJECTORIES = ("calibration_0", "calibration_1", "validation_2", "holdout_3")
NOMINAL = {"armature": 0.01, "damping": 0.08, "coulomb_friction": 0.025, "delay_steps": 0}
PARAMETER_BOUNDS = {
    "armature": (0.002, 0.060),
    "damping": (0.001, 0.350),
    "coulomb_friction": (0.0, 0.200),
}
SOURCE_FILES = {
    "scripts/evaluate_friction_delay.py",
    "scripts/evaluate_wheel_control.py",
    "scripts/run_actuator_identification.py",
    "src/wheel_legged_control/__init__.py",
    "src/wheel_legged_control/controllers.py",
    "src/wheel_legged_control/model.py",
    "src/wheel_legged_control/rewards.py",
    "src/wheel_legged_control/encoder_identification.py",
    "src/wheel_legged_control/actuator_identification.py",
    "src/wheel_legged_control/actuator_bench.py",
    "src/wheel_legged_control/wheel_control.py",
    "src/wheel_legged_control/provenance.py",
    "pyproject.toml",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(items):
    require(len({key for key, _ in items}) == len(items), "duplicate JSON key")
    return dict(items)


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_pairs, parse_constant=_constant)

    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), "non-finite JSON number")
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(root, name):
    require(isinstance(name, str), "artifact name must be a string")
    path = PurePosixPath(name)
    require(
        name
        and not path.is_absolute()
        and str(path) == name
        and ".." not in path.parts
        and "\\" not in name,
        f"unsafe artifact: {name}",
    )
    result = root / name
    require(
        not any(p.is_symlink() for p in (result, *result.parents) if p != root.parent),
        f"symlink artifact: {name}",
    )
    require(result.is_file() and result.resolve().is_relative_to(root), f"missing artifact: {name}")
    return result


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
            f"missing/duplicate CSV columns: {path}",
        )
        rows = list(reader)
    require(
        rows and all(None not in row and None not in row.values() for row in rows),
        f"empty or incomplete CSV: {path}",
    )
    return rows


def number(value):
    require(not isinstance(value, bool), "boolean in numeric field")
    result = float(value)
    require(math.isfinite(result), "non-finite numeric field")
    return result


def integer(value):
    result = number(value)
    require(result.is_integer(), "non-integer field")
    return int(result)


def boolean(value):
    require(type(value) is bool or value in ("True", "False"), "invalid boolean field")
    return value if type(value) is bool else value == "True"


def column(rows, name):
    return np.asarray([number(row[name]) for row in rows], dtype=float)


def same(actual, expected, label, *, atol=1e-11, rtol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.isfinite(expected).all()
        and np.allclose(actual, expected, atol=atol, rtol=rtol),
        f"mismatch: {label}",
    )


def indexed(rows, fields):
    result = {tuple(row[field] for field in fields): row for row in rows}
    require(len(result) == len(rows), f"duplicate matrix row: {fields}")
    return result


def position_metrics(measured, predicted):
    measured, predicted = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    require(
        measured.ndim == 1 and measured.shape == predicted.shape and len(measured) >= 3,
        "prediction requires equal T+1 position arrays",
    )
    require(np.isfinite(measured).all() and np.isfinite(predicted).all(), "non-finite prediction")
    errors = predicted[1:] - measured[1:]
    return {
        "samples": len(errors),
        "position_rmse_rad": math.sqrt(math.fsum(errors**2) / len(errors)),
        "position_mae_rad": math.fsum(abs(errors)) / len(errors),
        "peak_position_error_rad": float(max(abs(errors))),
    }


def calibration_loss(traces):
    """Each (measured, predicted) pair includes its initialization sample."""
    squared = []
    for measured, predicted in traces:
        position_metrics(measured, predicted)  # Shape/finite validation, not a production metric.
        squared.extend(
            ((np.asarray(predicted)[1:] - np.asarray(measured)[1:]) / POSITION_SCALE_RAD) ** 2
        )
    require(squared, "empty calibration residuals")
    return math.fsum(squared) / len(squared)


def audit_inventory(root):
    inventory = read_json(artifact(root, "manifest.json"))["sha256"]
    require(isinstance(inventory, dict) and inventory, "empty manifest")
    files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    extra = files - set(inventory) - {"manifest.json"}
    require(extra <= {"README.md"}, "unexpected files outside generation-time manifest")
    for name, digest in inventory.items():
        require(sha256(artifact(root, name)) == digest, f"artifact SHA mismatch: {name}")
    source = read_json(artifact(root, "source.json"))
    require(
        set(source["sha256"]) == SOURCE_FILES
        and source["root_relative_files"] == list(dict.fromkeys(source["root_relative_files"]))
        and set(source["root_relative_files"]) == SOURCE_FILES,
        "source closure differs",
    )
    with tarfile.open(artifact(root, "source.tar.gz"), "r:gz") as archive:
        members = archive.getmembers()
        require(
            len(members) == len(SOURCE_FILES) and {m.name for m in members} == SOURCE_FILES,
            "source archive members differ",
        )
        for member in members:
            require(member.isfile(), "source archive contains a non-regular member")
            payload = archive.extractfile(member).read()
            require(
                len(payload) == member.size
                and hashlib.sha256(payload).hexdigest() == source["sha256"][member.name],
                f"source SHA mismatch: {member.name}",
            )
    consistency = read_json(artifact(root, "source_consistency.json"))
    require(
        consistency["unchanged"] is True
        and consistency["sha256"] == source["sha256"]
        and consistency["pre_run_sha256"] == source["sha256"],
        "source changed during study",
    )
    return inventory, source["sha256"], sorted(extra)


def _excitation(times, duration, index, family):
    if index == 0:
        return (
            0.12 * np.sin(2 * np.pi * 0.35 * times)
            if family == "low_speed"
            else 0.40 * np.sin(2 * np.pi * (0.4 * times + 1.8 * times**2 / (2 * duration)))
        )
    if index == 1:
        return (
            0.10 * np.sin(2 * np.pi * 0.55 * times + 0.7) + 0.025 * np.sin(2 * np.pi * 1.8 * times)
            if family == "low_speed"
            else 0.30 * np.sin(2 * np.pi * 0.7 * times + 1.2)
            + 0.18 * np.sin(2 * np.pi * 2.7 * times)
        )
    if index == 2:
        return 0.35 * np.sin(2 * np.pi * (0.6 * times + 2.2 * times**2 / (2 * duration)) + 0.2)
    levels = np.array((0.0, 0.22, -0.48, 0.07, 0.38, -0.16, 0.0))
    return levels[np.minimum((times / duration * len(levels)).astype(int), len(levels) - 1)]


def audit_log(path, trajectory, args, family, seed):
    with path.open(encoding="utf-8", newline="") as stream:
        header = stream.readline()
        require(header.startswith("# "), "missing encoder metadata")
        meta = json.loads(header[2:], object_pairs_hook=_pairs, parse_constant=_constant)
        reader = csv.DictReader(stream)
        rows = list(reader)
    expected = {
        "schema": "encoder_position_only",
        "schema_version": 1,
        "config": {"kind": "wheel", "dt": DT, "torque_limit_nm": 2.0},
        "name": trajectory,
        "split": trajectory.rsplit("_", 1)[0],
        "sampling": "position_then_command_then_next_position",
        "initial_command_history": "zero",
        "initial_state": "known_rest",
        "known_initial_rest": True,
    }
    require(
        meta == expected
        and type(meta["schema_version"]) is int
        and meta["known_initial_rest"] is True,
        "encoder metadata/schema differs",
    )
    require(
        reader.fieldnames == ["time_s", "command_torque_nm", "position_rad", "next_position_rad"],
        "encoder columns differ",
    )
    steps = round(args["calibration_duration"] / DT)
    require(len(rows) == steps, "encoder sample count differs")
    same(column(rows, "time_s"), np.arange(steps) * DT, "encoder clock", rtol=0, atol=1e-10)
    q, next_q = column(rows, "position_rad"), column(rows, "next_position_rad")
    same(q[1:], next_q[:-1], "encoder continuity", atol=1e-12, rtol=0)
    index = TRAJECTORIES.index(trajectory)
    same(
        column(rows, "command_torque_nm"),
        _excitation(np.arange(steps) * DT, args["calibration_duration"], index, family),
        "excitation family",
        rtol=0,
        atol=1e-14,
    )
    first_noise = np.random.default_rng(np.random.SeedSequence((seed, index))).normal(
        size=steps + 1
    )[0]
    same(
        q[0],
        first_noise * args["noise_scale"] * 0.0002,
        "initial encoder noise",
        rtol=0,
        atol=1e-15,
    )
    return np.r_[q, next_q[-1]]


def _reference(profile, times):
    if profile == "reversal":
        phase = 2 * np.pi * 0.35 * times
        inner = 3 * np.sin(phase)
        return 2.5 * np.tanh(inner), 2.5 * 3 * 2 * np.pi * 0.35 * np.cos(phase) / np.cosh(
            inner
        ) ** 2
    frequency, amplitude = (0.45, 3.0) if profile == "tracking" else (1.0, 18.0)
    phase = 2 * np.pi * frequency * times
    return amplitude * np.sin(phase), amplitude * 2 * np.pi * frequency * np.cos(phase)


def audit_control(rows, metric, parameters, actual_delay, seed, args):
    n, full_n = len(rows), round(args["duration"] / DT)
    require(0 < n <= full_n, "control sample count differs")
    times = np.arange(n) * DT
    same(column(rows, "time_s"), times, "control clock", rtol=0, atol=1e-14)
    q, v = column(rows, "position_rad"), column(rows, "actual_velocity_rad_s")
    same(
        q[1:],
        column(rows, "next_position_rad")[:-1],
        "physical position continuity",
        rtol=0,
        atol=0,
    )
    same(
        v[1:],
        column(rows, "next_velocity_rad_s")[:-1],
        "physical velocity continuity",
        rtol=0,
        atol=0,
    )
    require(q[0] == 0 and v[0] == 0, "control did not reset at rest")
    measured = column(rows, "measured_velocity_rad_s")
    noise = np.random.default_rng(np.random.SeedSequence((seed, 314))).normal(size=full_n)[:n]
    same(
        measured,
        v + noise * args["noise_scale"] * 0.002,
        "paired control sensor noise",
        rtol=0,
        atol=4e-15,
    )
    reference, acceleration = _reference(metric["profile"], times)
    same(column(rows, "reference_velocity_rad_s"), reference, "paired reference")
    same(column(rows, "reference_acceleration_rad_s2"), acceleration, "paired reference derivative")
    integral = 0.0
    for row in rows:
        target, measured_v = (
            number(row["reference_velocity_rad_s"]),
            number(row["measured_velocity_rad_s"]),
        )
        error = target - measured_v
        proportional, increment = 0.12 * error, 0.6 * error * DT
        proposed_i = integral + increment
        ff = (
            (0.5 * 0.5 * 0.087**2 + parameters["armature"])
            * number(row["reference_acceleration_rad_s2"])
            + parameters["damping"] * target
            + parameters["coulomb_friction"] * math.tanh(target / 0.05)
        )
        proposed = proportional + proposed_i + ff
        frozen = (proposed > 2 and increment > 0) or (proposed < -2 and increment < 0)
        integral = integral if frozen else proposed_i
        requested = proportional + integral + ff
        for field, value in (
            ("error_rad_s", error),
            ("proportional_torque_nm", proportional),
            ("integral_torque_nm", integral),
            ("feedforward_torque_nm", ff),
            ("requested_torque_nm", requested),
            ("command_torque_nm", min(2.0, max(-2.0, requested))),
        ):
            require(
                math.isclose(number(row[field]), value, rel_tol=1e-10, abs_tol=1e-11),
                f"mismatch: shared PI/FF {field}",
            )
        require(
            boolean(row["integration_frozen"]) == frozen
            and boolean(row["saturated"]) == (abs(requested) > 2.0),
            "PI limiter flags differ",
        )
    command, requested, applied = (
        column(rows, field)
        for field in ("command_torque_nm", "requested_torque_nm", "applied_torque_nm")
    )
    same(
        applied, np.r_[np.zeros(actual_delay), command][:n], "actual actuator delay", rtol=0, atol=0
    )
    error = reference - v
    same(
        column(rows, "tracking_error_rad_s"),
        error,
        "current-state tracking error",
        rtol=0,
        atol=1e-14,
    )
    guard = abs(column(rows, "next_velocity_rad_s")) > 30.0
    require(not guard[:-1].any(), "control continued after speed guard")
    require(
        [boolean(row["speed_guard_triggered"]) for row in rows] == list(guard), "guard flags differ"
    )
    rms = lambda values: math.sqrt(math.fsum(values**2) / len(values))
    computed = {
        "steps": n,
        "duration_s": n * DT,
        "velocity_rmse_rad_s": rms(error),
        "velocity_mae_rad_s": math.fsum(abs(error)) / n,
        "peak_velocity_error_rad_s": float(max(abs(error))),
        "integrated_absolute_error_rad": math.fsum(abs(error)) * DT,
        "command_torque_rms_nm": rms(command),
        "applied_torque_rms_nm": rms(applied),
        "peak_requested_torque_nm": float(max(abs(requested))),
        "saturation_fraction": sum(abs(requested) > 2.0) / n,
        "integration_frozen_fraction": sum(boolean(row["integration_frozen"]) for row in rows) / n,
    }
    for field, value in computed.items():
        same(number(metric[field]), value, f"control metric {field}")
    computed.update(
        completed=n == full_n and not bool(guard.any()), speed_guard_triggered=bool(guard.any())
    )
    for field in ("completed", "speed_guard_triggered"):
        require(boolean(metric[field]) == computed[field], f"control status {field} differs")
    return computed


def _identity(row, identity):
    require(
        row["cell"] == identity["cell"] and row["family"] == identity["family"],
        "cell/family mismatch",
    )
    for field in ("stribeck_friction_nm", "actual_delay_steps", "seed"):
        require(number(row[field]) == identity[field], f"cell identity mismatch: {field}")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _benefit(pairs, nominal_field, fitted_field):
    ratios = [_ratio(p[fitted_field], p[nominal_field]) for p in pairs]
    valid = [value for value in ratios if value is not None]
    return {
        "pairs": len(pairs),
        "nominal_rmse_mean": float(np.mean([p[nominal_field] for p in pairs])),
        "fitted_rmse_mean": float(np.mean([p[fitted_field] for p in pairs])),
        "median_rmse_ratio": float(np.median(valid)) if valid else None,
        "undefined_ratio_count": len(ratios) - len(valid),
        "worse_count": sum(p[fitted_field] > p[nominal_field] for p in pairs),
    }


def analyze(study: Path) -> dict:
    root = Path(study).resolve()
    require(root.is_dir(), "study must be an existing directory")
    inventory, source, extra_docs = audit_inventory(root)
    manifest_sha = sha256(root / "manifest.json")
    protocol, summary = read_json(root / "protocol.json"), read_json(root / "summary.json")
    require(
        type(protocol["schema_version"]) is int
        and protocol["schema_version"] == 1
        and protocol["data_origin"] == "synthetic_only",
        "unsupported study protocol",
    )
    args, grids = protocol["arguments"], protocol["grids"]
    require(
        protocol["bench_config"] == {"kind": "wheel", "dt": DT, "torque_limit_nm": 2.0}
        and protocol["controller_config_all_conditions"]
        == {"kp": 0.12, "ki": 0.6, "dt": DT, "torque_limit_nm": 2.0},
        "bench/controller protocol differs",
    )
    require(
        protocol["nominal_parameters"] == NOMINAL
        and protocol["continuous_truth_not_available_to_fitter_or_controller"]
        == {"armature": 0.017, "damping": 0.105, "coulomb_friction": 0.043},
        "parameter protocol differs",
    )
    for argument, grid in (
        ("frictions", "extra_hidden_stribeck_friction_nm"),
        ("actual_delays", "hidden_actual_delay_steps"),
        ("families", "calibration_families"),
        ("seeds", "noise_seeds"),
    ):
        require(
            args[argument] == grids[grid]
            and args[argument]
            and len(set(args[argument])) == len(args[argument]),
            f"invalid grid: {argument}",
        )
    require(set(args["families"]) <= {"standard", "low_speed"}, "unknown calibration family")
    for field in ("max_delay_steps", "max_nfev"):
        require(integer(args[field]) >= (field == "max_nfev"), "invalid optimizer budget")
    require(
        args["calibration_duration"] >= 0.5
        and args["duration"] >= 1.0
        and args["noise_scale"] >= 0,
        "invalid duration/noise",
    )
    same(
        protocol["position_noise_std_rad"], args["noise_scale"] * 0.0002, "position noise protocol"
    )
    cells = []
    for friction, delay, seed, family in itertools.product(
        args["frictions"], args["actual_delays"], args["seeds"], args["families"]
    ):
        require(
            number(friction) >= 0 and integer(delay) >= 0 and integer(seed) >= 0, "negative grid"
        )
        token = repr(float(friction)) if friction else "0"
        cells.append(
            {
                "cell": f"friction{token}_delay{delay}_seed{seed}_{family}",
                "stribeck_friction_nm": friction,
                "actual_delay_steps": delay,
                "seed": seed,
                "family": family,
            }
        )
    counts = {
        "fits": len(cells),
        "candidates": len(cells) * (args["max_delay_steps"] + 1),
        "prediction_metric_rows": len(cells) * 8,
        "control_cases": len(cells) * 6,
    }
    require(
        protocol["expected_counts"] == counts
        and summary["observed_counts"] == {"cells": len(cells), **counts},
        "matrix counts differ",
    )
    tables = {
        name: read_rows(artifact(root, name + ".csv.gz"))
        for name in ("fits", "candidates", "prediction_metrics", "control_metrics")
    }
    fit_index = indexed(tables["fits"], ("cell",))
    candidate_index = indexed(tables["candidates"], ("cell", "candidate_delay_steps"))
    prediction_index = indexed(tables["prediction_metrics"], ("cell", "trajectory", "model"))
    control_index = indexed(tables["control_metrics"], ("cell", "profile", "controller"))
    require(set(fit_index) == {(c["cell"],) for c in cells}, "fit matrix differs")
    require(
        set(candidate_index)
        == {(c["cell"], str(d)) for c in cells for d in range(args["max_delay_steps"] + 1)},
        "candidate matrix differs",
    )
    require(
        set(prediction_index)
        == {
            (c["cell"], t, m) for c in cells for t in TRAJECTORIES for m in ("nominal", "selected")
        },
        "prediction matrix differs",
    )
    require(
        set(control_index)
        == {
            (c["cell"], p, m) for c in cells for p in PROFILES for m in ("nominal_ff", "fitted_ff")
        },
        "control matrix differs",
    )
    fits, candidates, predictions, controls, family_pairs = [], [], [], [], {}
    raw_counts = {"candidate_prediction_rows": 0, "prediction_rows": 0, "control_rows": 0}
    for identity in cells:
        name = identity["cell"]
        fit, fit_csv = read_json(artifact(root, name + "/fit.json")), fit_index[(name,)]
        _identity(fit_csv, identity)
        candidate_records = fit["candidates"]
        require(
            [r["parameters"]["delay_steps"] for r in candidate_records]
            == list(range(args["max_delay_steps"] + 1)),
            "fit candidate order differs",
        )
        best = min(candidate_records, key=lambda r: r["calibration_loss"])
        selected = integer(fit["parameters"]["delay_steps"])
        require(
            fit["parameters"] == best["parameters"], "selection is not minimum calibration loss"
        )
        flags = {
            "selected_optimizer_success": best["optimizer_success"],
            "delay_at_search_boundary": selected in (0, args["max_delay_steps"]),
            "actual_delay_out_of_search_range": identity["actual_delay_steps"]
            > args["max_delay_steps"],
        }
        for field, value in flags.items():
            require(
                type(value) is bool and fit[field] is value and boolean(fit_csv[field]) is value,
                f"fit status {field} differs",
            )
        require(
            fit["delay_compensated"] is False
            and fit["actual_delay_steps"] == identity["actual_delay_steps"],
            "fit actual delay differs",
        )
        delay_error = selected - identity["actual_delay_steps"]
        require(
            integer(fit_csv["fitted_delay_steps"]) == selected
            and integer(fit_csv["delay_error_steps"]) == fit["delay_error_steps"] == delay_error,
            "delay error differs",
        )
        require(
            fit["selected_nfev"] == best["nfev"] and fit["selected_message"] == best["message"],
            "selected optimizer record differs",
        )
        same(fit["selected_calibration_loss"], best["calibration_loss"], "selected loss")
        same(number(fit_csv["selected_calibration_loss"]), best["calibration_loss"], "fit CSV loss")
        for field, value in fit["parameters"].items():
            same(number(fit_csv["fitted_" + field]), value, "fitted parameter")
        singular = np.asarray(fit["jacobian_singular_values"], dtype=float)
        require(
            singular.shape == (3,)
            and np.isfinite(singular).all()
            and (singular >= 0).all()
            and (np.diff(singular) <= 0).all(),
            "invalid recorded Jacobian spectrum",
        )
        rank = int(np.sum(singular > singular[0] * 1e-7))
        require(
            integer(fit["jacobian_rank"]) == integer(fit_csv["jacobian_rank"]) == rank,
            "recorded Jacobian rank differs",
        )
        bound_parameters = [
            field
            for field, bounds in PARAMETER_BOUNDS.items()
            if any(np.isclose(fit["parameters"][field], bound, atol=1e-6) for bound in bounds)
        ]
        require(
            fit["continuous_parameters_at_bounds"] == bound_parameters,
            "parameter bound flags differ",
        )
        logs = {
            trajectory: audit_log(
                artifact(root, f"{name}/{trajectory}.csv"),
                trajectory,
                args,
                identity["family"],
                identity["seed"],
            )
            for trajectory in TRAJECTORIES
        }
        selected_traces = {}
        for candidate in candidate_records:
            delay = integer(candidate["parameters"]["delay_steps"])
            row = candidate_index[(name, str(delay))]
            _identity(row, identity)
            require(
                1 <= integer(candidate["nfev"]) <= args["max_nfev"]
                and integer(row["nfev"]) == candidate["nfev"]
                and row["message"] == candidate["message"],
                "candidate optimizer record differs",
            )
            require(
                boolean(row["optimizer_success"]) is candidate["optimizer_success"]
                and boolean(row["selected"]) == (delay == selected),
                "candidate status differs",
            )
            for field, value in candidate["parameters"].items():
                same(number(row["candidate_" + field]), value, "candidate parameter")
            require(
                set(candidate["parameters"]) == {*PARAMETER_BOUNDS, "delay_steps"}
                and all(
                    low <= number(candidate["parameters"][field]) <= high
                    for field, (low, high) in PARAMETER_BOUNDS.items()
                ),
                "candidate parameters outside declared bounds",
            )
            trace = read_rows(
                artifact(root, f"{name}/candidate_delay{delay}_calibration_predictions.csv.gz")
            )
            raw_counts["candidate_prediction_rows"] += len(trace)
            require(
                [r["trajectory"] for r in trace] == [t for t in TRAJECTORIES[:2] for _ in logs[t]],
                "candidate trajectory/sample count differs",
            )
            pieces = []
            for trajectory in TRAJECTORIES[:2]:
                part = [r for r in trace if r["trajectory"] == trajectory]
                measured, predicted = (
                    column(part, "measured_position_rad"),
                    column(part, "predicted_position_rad"),
                )
                same(
                    column(part, "time_s"),
                    np.arange(len(logs[trajectory])) * DT,
                    "candidate clock",
                    rtol=0,
                    atol=1e-14,
                )
                same(measured, logs[trajectory], "candidate measured positions", rtol=0, atol=0)
                require(predicted[0] == measured[0], "candidate initialization differs")
                pieces.append((measured, predicted))
                if delay == selected:
                    selected_traces[trajectory] = predicted
            loss = calibration_loss(pieces)
            for value in (
                candidate["calibration_loss"],
                number(row["reported_calibration_loss"]),
                number(row["recomputed_calibration_loss"]),
            ):
                same(value, loss, "candidate calibration loss", rtol=1e-9, atol=1e-12)
            candidates.append(
                {
                    **identity,
                    "candidate_delay_steps": delay,
                    "selected": delay == selected,
                    "recomputed_calibration_loss": loss,
                    "reported_calibration_loss": candidate["calibration_loss"],
                    "optimizer_success": candidate["optimizer_success"],
                    "nfev": candidate["nfev"],
                    "parameters": candidate["parameters"],
                }
            )
        for trajectory in TRAJECTORIES:
            trace = read_rows(artifact(root, f"{name}/{trajectory}_predictions.csv.gz"))
            raw_counts["prediction_rows"] += len(trace)
            same(
                column(trace, "time_s"),
                np.arange(len(logs[trajectory])) * DT,
                "prediction clock",
                rtol=0,
                atol=1e-14,
            )
            measured = column(trace, "measured_position_rad")
            same(measured, logs[trajectory], "prediction measurements", rtol=0, atol=0)
            for model in ("nominal", "selected"):
                row = prediction_index[(name, trajectory, model)]
                _identity(row, identity)
                require(row["split"] == trajectory.rsplit("_", 1)[0], "prediction split differs")
                predicted = column(trace, model + "_position_rad")
                require(predicted[0] == measured[0], "prediction initialization differs")
                if model == "selected" and trajectory in selected_traces:
                    same(
                        predicted,
                        selected_traces[trajectory],
                        "selected candidate/prediction consistency",
                        rtol=0,
                        atol=0,
                    )
                values = position_metrics(measured, predicted)
                for field, value in values.items():
                    same(number(row[field]), value, "prediction metric " + field)
                predictions.append(
                    {
                        **identity,
                        "trajectory": trajectory,
                        "split": row["split"],
                        "model": model,
                        **values,
                    }
                )
        for profile, model in itertools.product(PROFILES, ("nominal_ff", "fitted_ff")):
            row = control_index[(name, profile, model)]
            _identity(row, identity)
            trace = read_rows(artifact(root, f"{name}/{profile}_{model}_rows.csv.gz"))
            raw_counts["control_rows"] += len(trace)
            values = audit_control(
                trace,
                row,
                NOMINAL if model == "nominal_ff" else fit["parameters"],
                identity["actual_delay_steps"],
                identity["seed"],
                args,
            )
            controls.append({**identity, "profile": profile, "controller": model, **values})
        fits.append(
            {
                **identity,
                "parameters": fit["parameters"],
                "fitted_delay_steps": selected,
                "delay_error_steps": delay_error,
                "selected_nfev": fit["selected_nfev"],
                "selected_calibration_loss": fit["selected_calibration_loss"],
                "continuous_parameters_at_bounds": fit["continuous_parameters_at_bounds"],
                "jacobian_rank": fit["jacobian_rank"],
                "jacobian_singular_values": fit["jacobian_singular_values"],
                **flags,
            }
        )
        pair = tuple(
            identity[field] for field in ("stribeck_friction_nm", "actual_delay_steps", "seed")
        )
        paired_files = (
            "validation_2.csv",
            "holdout_3.csv",
            *(p + "_nominal_ff_rows.csv.gz" for p in PROFILES),
        )
        hashes = {filename: inventory[name + "/" + filename] for filename in paired_files}
        if pair in family_pairs:
            require(hashes == family_pairs[pair], "held-out/nominal rows differ across families")
        family_pairs[pair] = hashes
    status_counts = {
        "converged_selected_fits": sum(r["selected_optimizer_success"] for r in fits),
        "nonconverged_selected_fits": sum(not r["selected_optimizer_success"] for r in fits),
        "delay_at_search_boundary": sum(r["delay_at_search_boundary"] for r in fits),
        "actual_delay_out_of_search_range": sum(
            r["actual_delay_out_of_search_range"] for r in fits
        ),
        "incomplete_control_runs": sum(not r["completed"] for r in controls),
        "speed_guard_triggered": sum(r["speed_guard_triggered"] for r in controls),
    }
    require(summary["observed_status_counts"] == status_counts, "summary status counts differ")
    labels = {
        "families": sorted({r["family"] for r in fits}),
        "splits": sorted({r["split"] for r in predictions}),
        "profiles": sorted({r["profile"] for r in controls}),
        "controllers": sorted({r["controller"] for r in controls}),
        "prediction_models": sorted({r["model"] for r in predictions}),
        "fitted_delay_steps": sorted({r["fitted_delay_steps"] for r in fits}),
    }
    require(summary["observed_labels"] == labels, "summary matrix labels differ")
    require(
        summary["status"]
        == (
            "contains_incomplete_runs" if status_counts["incomplete_control_runs"] else "completed"
        ),
        "summary completion status differs",
    )
    groups = []
    for friction, delay, family in itertools.product(
        args["frictions"], args["actual_delays"], args["families"]
    ):
        selected_fits = [
            r
            for r in fits
            if (r["stribeck_friction_nm"], r["actual_delay_steps"], r["family"])
            == (friction, delay, family)
        ]
        group_cells = sorted({r["cell"] for r in selected_fits})
        holdout = [
            {
                m: next(
                    r["position_rmse_rad"]
                    for r in predictions
                    if r["cell"] == name and r["split"] == "holdout" and r["model"] == m
                )
                for m in ("nominal", "selected")
            }
            for name in group_cells
        ]
        by_profile = {}
        for profile in PROFILES:
            pairs = []
            for name in group_cells:
                pair = {
                    m: next(
                        r
                        for r in controls
                        if r["cell"] == name and r["profile"] == profile and r["controller"] == m
                    )
                    for m in ("nominal_ff", "fitted_ff")
                }
                pairs.append(pair)
            by_profile[profile] = {
                **_benefit(
                    [{m: p[m]["velocity_rmse_rad_s"] for m in p} for p in pairs],
                    "nominal_ff",
                    "fitted_ff",
                ),
                "nominal_saturation_mean": float(
                    np.mean([p["nominal_ff"]["saturation_fraction"] for p in pairs])
                ),
                "fitted_saturation_mean": float(
                    np.mean([p["fitted_ff"]["saturation_fraction"] for p in pairs])
                ),
                "fitted_saturation_max": max(p["fitted_ff"]["saturation_fraction"] for p in pairs),
                "incomplete_pairs": sum(
                    not (p["nominal_ff"]["completed"] and p["fitted_ff"]["completed"])
                    for p in pairs
                ),
            }
        groups.append(
            {
                "stribeck_friction_nm": friction,
                "actual_delay_steps": delay,
                "family": family,
                "fits": len(selected_fits),
                "selected_delay_steps": [r["fitted_delay_steps"] for r in selected_fits],
                "delay_wrong_count": sum(r["delay_error_steps"] != 0 for r in selected_fits),
                "mean_absolute_delay_error_steps": float(
                    np.mean([abs(r["delay_error_steps"]) for r in selected_fits])
                ),
                "nonconverged_fits": sum(
                    not r["selected_optimizer_success"] for r in selected_fits
                ),
                "holdout_prediction": _benefit(holdout, "nominal", "selected"),
                "control_by_profile": by_profile,
            }
        )
    # Check that the original immutable inputs did not change while being read.
    require(
        sha256(root / "manifest.json") == manifest_sha, "input manifest changed during analysis"
    )
    for name, digest in inventory.items():
        require(sha256(artifact(root, name)) == digest, f"input changed during analysis: {name}")
    default = (
        args["frictions"] == [0.0, 0.04, 0.08]
        and args["actual_delays"] == [0, 2, 4]
        and args["seeds"] == [53, 67, 79]
        and args["families"] == ["standard", "low_speed"]
        and args["calibration_duration"] == 3.0
        and args["duration"] == 6.0
        and args["max_delay_steps"] == 4
        and args["max_nfev"] == 30
        and args["noise_scale"] == 1.0
    )
    return {
        "schema": "friction-delay-analysis-v1",
        "study": str(root),
        "study_manifest_sha256": manifest_sha,
        "analysis_source_sha256": sha256(Path(__file__)),
        "verification": {
            "status": "verified",
            "counts": counts,
            **raw_counts,
            "source_members": len(source),
            "manifest_files": len(inventory),
            "default_formal_protocol": default,
            "post_experiment_docs_not_in_original_manifest": extra_docs,
            "paired_family_groups": len(family_pairs),
        },
        "observed_status_counts": status_counts,
        "fits": fits,
        "candidates": candidates,
        "predictions": predictions,
        "controls": controls,
        "groups": groups,
        "limitations": [
            "Recomputes recorded arrays, not hidden MuJoCo dynamics or optimizer evaluations.",
            "Optimizer convergence/local Jacobian rank is not parameter identifiability.",
            "Noise seeds are repeats for one object, not independent robots; nominal controls duplicated across families are not extra evidence.",
            "Control feedback retains an independent synthetic velocity sensor; no delay compensation, D1 or hardware claim.",
            "Ratios from incomplete runs compare different prefixes and are not a valid full-duration benefit claim.",
        ],
    }


def run(study: Path, output: Path) -> dict:
    require(not Path(output).is_symlink(), "output must not be a symlink")
    study, output = Path(study).resolve(), Path(output).resolve()
    require(not output.exists(), "output exists; choose a new analysis directory")
    require(
        not output.is_relative_to(study) and not study.is_relative_to(output),
        "analysis output overlaps study",
    )
    report = analyze(study)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with (output / "analysis_source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    with (output / "analysis_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "sha256": {
                    name: sha256(output / name) for name in ("analysis.json", "analysis_source.py")
                },
                "scope": "analysis creation files only; later plots are separate artifacts",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.study, args.output)
    except (ValueError, KeyError, OSError, TypeError, tarfile.TarError) as error:
        parser.exit(1, f"analysis failed: {error}\n")
    print(
        json.dumps(
            {
                "status": "verified",
                "output": str(args.output.resolve()),
                **report["verification"]["counts"],
            },
            allow_nan=False,
        )
    )
    return report


if __name__ == "__main__":
    main()
