"""C32 saved-event critic diagnostic. NumPy only; no simulation or policy calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


TARGETS = ("saved_gae", "mc_gamma1")
METHODS = ("P", "L0", "L1", "H1")
RIDGE = 0.01


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(values, shape, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {arr.shape}")
    return arr


def _resolve_input(spec: dict, contract_path: Path) -> Path:
    path = Path(spec["path"])
    if path.is_absolute():
        return path
    nearby = contract_path.parent / path
    if nearby.is_file():
        return nearby
    # The C32 contract lives in continuation32; archived inputs are in its
    # sibling continuation31. This also accepts paths relative to W.
    return contract_path.parent.parent.parent / path


def load_inputs(contract_path: Path):
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    specs = contract["inputs"]
    if len(specs) != 12:
        raise ValueError("contract must bind eight batches, final parameters, and three sources")
    paths, manifest = {}, []
    for spec in specs:
        path = _resolve_input(spec, contract_path)
        name = path.name
        if name in paths:
            raise ValueError(f"duplicate input basename: {name}")
        raw = path.read_bytes()
        digest = sha256(raw)
        if digest != spec["sha256"] or len(raw) != int(spec["bytes"]):
            raise ValueError(f"frozen input hash/length mismatch: {path}")
        paths[name] = json.loads(raw) if path.suffix == ".json" else raw
        manifest.append({"path": str(path.resolve()), "sha256": digest, "bytes": len(raw)})
    expected = {f"batch_{i:02d}.json" for i in range(8)} | {"final_parameters31.json"}
    expected |= {"learning31.py", "observation31.py", "final_review31.json"}
    if set(paths) != expected:
        raise ValueError(f"contract input names differ: {sorted(set(paths) ^ expected)}")
    if contract.get("schema") != "d1-c32-single-saved-value-diagnostic-contract-v1":
        raise ValueError("C32 contract schema differs")
    return contract, sha256(contract_bytes), paths, manifest


def recheck_inputs(manifest, contract_path, contract_hash):
    if sha256(contract_path.read_bytes()) != contract_hash:
        raise ValueError("contract changed during diagnostic")
    for item in manifest:
        raw = Path(item["path"]).read_bytes()
        if len(raw) != item["bytes"] or sha256(raw) != item["sha256"]:
            raise ValueError(f"input changed during diagnostic: {item['path']}")


def prepare(paths):
    rows = []
    for batch in range(8):
        prepared = paths[f"batch_{batch:02d}.json"]["prepared_batch"]
        if prepared["cycle_indices"] != list(range(8*batch, 8*batch+8)):
            raise ValueError(f"batch {batch}: cycle indices differ")
        part = prepared["rows"]
        if len(part) != 32 or prepared["macro_count"] != 32:
            raise ValueError(f"batch {batch}: expected 32 actual events")
        rows.extend(part)
    if [(r["cycle_index"], r["macro_index"]) for r in rows] != [
        (cycle, macro) for cycle in range(64) for macro in range(4)
    ]:
        raise ValueError("frozen events must be exactly 64 ordered four-macro cycles")
    x = _finite([r["observation54"] for r in rows], (256, 54), "observation54")
    saved = _finite([r["value_target"] for r in rows], (256,), "value_target")
    old = _finite([r["old_value"] for r in rows], (256,), "old_value")
    gae = _finite([r["gae"] for r in rows], (256,), "gae")
    reward = _finite([r["reward"] for r in rows], (256,), "reward")
    if not np.allclose(saved, old+gae, rtol=0, atol=1e-10):
        raise ValueError("saved value target differs from old value + GAE")
    direction = x[:, 0].astype(int)
    macro = np.array([r["macro_index"] for r in rows], dtype=int)
    if not np.array_equal(x[:, 0], direction) or not np.isin(direction, (-1, 1)).all():
        raise ValueError("observation direction must be exactly -1 or +1")
    if not np.allclose(x[:, 5], macro/3, rtol=0, atol=1e-12):
        raise ValueError("observation macro column differs")
    if (not np.all(np.isin(x[:, 1:5], (0., 1.))) or
            not np.allclose(x[:, 1:5].sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("four leg columns are not one-hot")
    mc = np.zeros(256)
    for cycle in range(64):
        running = 0.
        for macro_index in range(3, -1, -1):
            index = 4*cycle+macro_index
            if bool(rows[index]["terminal"]) != (macro_index == 3):
                raise ValueError("terminal flag differs from four-macro cycle")
            components = rows[index]["reward_components"]
            reconstructed_reward = (
                10*float(components["progress_potential_difference"])
                - float(components["elapsed_s"])
                - .05*float(components["backtrack_normalized"])
                - .02*float(components["longitudinal_normalized_square_integral_s"])
                - .02*float(components["lateral_goal_error_normalized_square_integral_s"])
                - .02*float(components["yaw_error_normalized_square_integral_s"])
                - .5*float(components["torque_normalized_square_integral_s"])
                + float(rows[index]["terminal_adjustment"]))
            if not math.isclose(reconstructed_reward, reward[index], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"saved reward differs at cycle {cycle} macro {macro_index}")
            next_value = 0. if macro_index == 3 else old[index+1]
            next_gae = 0. if macro_index == 3 else gae[index+1]
            delta = reward[index]+next_value-old[index]
            if (not math.isclose(delta, float(rows[index]["delta"]), rel_tol=0, abs_tol=1e-10)
                    or not math.isclose(float(rows[index]["gae_factor"]),
                                        .95**float(rows[index]["duration_s"]), rel_tol=0,
                                        abs_tol=1e-10)
                    or not math.isclose(gae[index],
                                        delta+float(rows[index]["gae_factor"])*next_gae,
                                        rel_tol=0, abs_tol=1e-10)):
                raise ValueError(f"saved GAE recursion differs at cycle {cycle} macro {macro_index}")
            running = float(reward[index]) + running  # gamma = 1
            mc[index] = running
    y = np.column_stack((saved, mc))
    cell = (direction == 1).astype(int)*4 + macro
    split = np.repeat(["train", "validation", "test"], [128, 64, 64])
    return rows, x, y, direction, macro, cell, split


def metrics(y, predicted):
    error = predicted-y
    variance = float(np.var(y))
    mse = float(np.mean(error*error))
    return {"n": len(y), "target_mean": float(np.mean(y)),
            "target_variance_ddof0": variance, "mse": mse,
            "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(error))),
            "mean_prediction_minus_target": float(np.mean(error)),
            "explained_variance": None if variance == 0 else
            float(1-np.var(error)/variance)}


def fit_lstsq(x, y, *, ridge=0., free=1):
    if ridge:
        penalty = np.eye(x.shape[1]); penalty[:free] = 0
        design = np.vstack((x, np.sqrt(len(x)*ridge)*penalty))
        outcome = np.vstack((y, np.zeros((x.shape[1], y.shape[1]))))
    else:
        design, outcome = x, y
    coefficients, residuals, rank, singular = np.linalg.lstsq(design, outcome, rcond=1e-12)
    return coefficients, {
        "raw_design_shape": list(x.shape),
        "solve_design_shape": list(design.shape), "solve_rank": int(rank),
        "solve_condition": None if singular[-1] == 0 else
        float(singular[0]/singular[-1]),
        "solve_singular_values": singular.tolist(),
        "lstsq_residual_sums": residuals.tolist(), "ridge": ridge, "rcond": 1e-12,
        "unpenalized_columns": free,
    }


def fit_predict(x, y, cell):
    train = slice(0, 128)
    onehot = np.eye(8)[cell]
    p = np.empty((8, 2))
    for c in range(8):
        selected = cell[train] == c
        if not selected.any():
            raise ValueError(f"P cell {c} absent from training cycles")
        p[c] = np.mean(y[train][selected], axis=0)
    predictions = {"P": p[cell]}
    lstsq_calls = 0
    coefficients = {"P": {"cell_means": p.tolist(),
                           "cell_order": [f"{d}:{m}" for d in (-1,1) for m in range(4)]}}
    # L0 and L1 use the archived normalized54 verbatim. The intercept is
    # explicit; their 54 observed features are not transformed again.
    linear = np.column_stack((np.ones(256), x))
    hybrid = np.column_stack((onehot, x[:, 6:54]))
    designs = {"L0": (linear, 0., 1), "L1": (linear, RIDGE, 1),
               "H1": (hybrid, RIDGE, 8)}
    for name, (design, ridge, free) in designs.items():
        coef, diagnostics = fit_lstsq(design[train], y[train], ridge=ridge, free=free)
        if not np.isfinite(coef).all():
            raise ValueError(f"{name} produced nonfinite coefficients")
        lstsq_calls += 1
        predictions[name] = design @ coef
        coefficients[name] = {"columns": (["bias"]+[f"observation54[{i}]" for i in range(54)])
            if name != "H1" else ([f"cell[{i}]" for i in range(8)]+
                                      [f"observation54[{i}]" for i in range(6,54)]),
            "target_order": list(TARGETS), "values": coef.tolist(), **diagnostics}
    counts = {"lstsq_calls": lstsq_calls, "lstsq_target_fits": 2*lstsq_calls,
              "cell_mean_target_fits": y.shape[1],
              "model_scalar_predictions": sum(v.size for v in predictions.values())}
    return predictions, coefficients, counts


def summarize(y, predictions, direction, macro, split):
    result = {}
    for name, values in predictions.items():
        result[name] = {}
        for j, target in enumerate(TARGETS):
            sections = {}
            for phase in ("train", "validation", "test"):
                mask = split == phase
                sections[phase] = {"all": metrics(y[mask, j], values[mask, j]),
                    "by_direction": {str(d): metrics(y[mask & (direction == d), j],
                                                    values[mask & (direction == d), j])
                                     for d in (-1, 1)},
                    "by_macro": {str(m): metrics(y[mask & (macro == m), j],
                                                values[mask & (macro == m), j])
                                 for m in range(4)}}
            result[name][target] = sections
    for target in TARGETS:
        for phase in ("train", "validation", "test"):
            for section in ("all", "by_direction", "by_macro"):
                baseline = result["P"][target][phase][section]
                groups = (None,) if section == "all" else tuple(baseline)
                for group in groups:
                    base = baseline["mse"] if group is None else baseline[group]["mse"]
                    for name in result:
                        item = result[name][target][phase][section]
                        item = item if group is None else item[group]
                        item["mse_over_P"] = None if base == 0 else item["mse"]/base
    return result


def run(contract_path: Path, output: Path):
    began = time.monotonic()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    contract, contract_hash, paths, manifest = load_inputs(contract_path)
    rows, x, y, direction, macro, cell, split = prepare(paths)
    predictions, coefficients, fitted_counts = fit_predict(x, y, cell)
    if any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("nonfinite fitted prediction")
    param = paths["final_parameters31.json"]
    weight = _finite(param["value.weight"], (1, 54), "value.weight")[0]
    bias = _finite(param["value.bias"], (1,), "value.bias")[0]
    final_value = (x @ weight + bias)[:, None]
    frozen_predictions = np.repeat(final_value, 2, axis=1)
    all_predictions = {**predictions, "frozen_value31": frozen_predictions}
    leg = np.argmax(x[:, 1:5], axis=1)
    report = {
        "schema": "c32-offline-critic-diagnostic-v1", "contract_sha256": contract_hash,
        "contract_path": str(contract_path.resolve()), "input_manifest": manifest,
        "targets": {"saved_gae": "archived value_target = old_value + GAE",
                    "mc_gamma1": "same-cycle reverse reward sum, gamma=1"},
        "splits": {"train": "cycles 0..31", "validation": "cycles 32..47",
                   "test": "cycles 48..63"},
        "counts": {"cycles": len(rows)//4, "events": len(rows),
                   "train_events": int(np.sum(split == "train")),
                   "validation_events": int(np.sum(split == "validation")),
                   "test_events": int(np.sum(split == "test")), **fitted_counts,
                   "frozen_value31_scalar_predictions": final_value.size,
                   "total_scalar_predictions": sum(v.size for v in predictions.values())+final_value.size,
                   "actor_calls": 0, "B22_calls": 0, "torch_calls": 0,
                   "physics_calls": 0, "native_integrations": 0,
                   "optimizer_steps": 0, "new_data_collection": 0,
                   "hyperparameter_sweeps": 0},
        "cell_leg_audit": {f"{d}:{m}": {str(l): int(np.sum((direction == d) &
                                                    (macro == m) & (leg == l)))
                                              for l in range(4)}
                            for d in (-1, 1) for m in range(4)},
        "saved_final_value_training_overlap": contract["saved_final_value"]["training_overlap"],
        "metrics": summarize(y, all_predictions, direction, macro, split),
        "target_difference": {phase: metrics(y[split == phase, 1], y[split == phase, 0])
                              for phase in ("train", "validation", "test")},
        "elapsed_seconds": time.monotonic()-began,
    }
    report["adequate_descriptive_fit"] = {
        name: {target: all(
            report["metrics"][name][target][phase]["all"]["explained_variance"] is not None
            and report["metrics"][name][target][phase]["all"]["explained_variance"] >= .8
            and report["metrics"][name][target][phase]["all"]["mse_over_P"] is not None
            and report["metrics"][name][target][phase]["all"]["mse_over_P"] <= .5
            for phase in ("validation", "test")) for target in TARGETS}
        for name in METHODS}
    prediction_rows = [{"cycle_index": int(r["cycle_index"]),
                        "macro_index": int(r["macro_index"]),
                        "split": str(split[i]), "direction": int(direction[i]),
                        "target": dict(zip(TARGETS, y[i].tolist())),
                        "prediction": {name: dict(zip(TARGETS, values[i].tolist()))
                                       for name, values in predictions.items()},
                        "frozen_value31_prediction": float(final_value[i, 0])}
                       for i, r in enumerate(rows)]
    if time.monotonic()-began > 300:
        raise TimeoutError("C32 elapsed wall time exceeded 300 seconds")
    if any(report["counts"][key] != contract["budgets"][contract_key] for key, contract_key in (
            ("lstsq_calls", "dual_target_lstsq_calls"),
            ("lstsq_target_fits", "target_lstsq_fits"),
            ("cell_mean_target_fits", "target_mean_fits"),
            ("model_scalar_predictions", "fitted_scalar_predictions"),
            ("frozen_value31_scalar_predictions", "saved_final_value_predictions"),
            ("total_scalar_predictions", "total_scalar_predictions"))):
        raise ValueError("actual compute counts differ from contract")
    recheck_inputs(manifest, contract_path, contract_hash)
    report["input_hashes_rechecked_after_fit"] = True
    output.mkdir(parents=True, exist_ok=False)
    for name, document in (("report.json", report), ("coefficients.json", coefficients),
                           ("predictions.json", prediction_rows)):
        (output/name).write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.contract, args.output)
    print(json.dumps({"output": str(args.output), "counts": report["counts"],
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
