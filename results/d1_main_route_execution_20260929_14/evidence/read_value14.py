"""Independent NumPy-only readback of one frozen value14 model result."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


GAMMA = .99
RHO = GAMMA ** 64
D64 = (1 - RHO) / (1 - GAMMA)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, *, atol=1e-8, rtol=1e-6):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False),
            f"{label} differs")


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def check_window_indices(sample):
    start = sample["start_indices"]
    end = sample["endpoint_indices"]
    q = sample["quartile"]
    tick = sample["episode_tick"]
    n = len(start)
    require(all(x.shape == (n,) for x in (end, q, tick, sample["episode_index"])),
            "window index shapes differ")
    require(np.issubdtype(start.dtype, np.integer) and
            np.issubdtype(end.dtype, np.integer), "window indices are not integers")
    require(np.array_equal(end, start + 64), "endpoint is not start+64")
    require(np.all(start >= 0) and np.all(end < 65536), "window outside saved controls")
    require(np.all(tick >= 0) and np.all(tick % 64 == 0), "start tick is not aligned")
    require(np.array_equal(q, start // 16384) and np.array_equal(q, end // 16384),
            "window crosses quartile")
    require(len(np.unique(start)) == n, "duplicate start")
    require(np.all(np.diff(start) > 0), "starts are not chronological")
    for group in np.unique(q):
        group_starts = start[q == group]
        require(np.all(np.diff(group_starts) >= 64), "reward windows overlap")


def check_values(sample, endpoints, values):
    n = len(sample["start_indices"])
    reward = sample["rewards"]
    require(reward.shape == (n, 64) and reward.dtype == np.float64,
            "reward shape or dtype differs")
    require(sample["start_obs"].shape == (n, 99) and
            sample["endpoint_obs"].shape == (n, 99), "observation shapes differ")
    require(np.array_equal(endpoints["endpoint_indices"], sample["endpoint_indices"]),
            "endpoint value indices differ")
    require(np.array_equal(values["start_indices"], sample["start_indices"]) and
            np.array_equal(values["quartile"], sample["quartile"]),
            "value row order differs")
    for key in ("start_values", "endpoint_values", "discounted_rewards",
                "targets_float64", "targets_used_float32", "residual"):
        require(values[key].shape == (n,) and np.isfinite(values[key]).all(),
                f"value array {key} invalid")
    require(endpoints["endpoint_values"].shape == (n,) and
            np.isfinite(endpoints["endpoint_values"]).all(), "endpoint array invalid")
    require(values["start_values"].dtype == np.float32 and
            values["endpoint_values"].dtype == np.float32 and
            values["targets_float64"].dtype == np.float64 and
            values["targets_used_float32"].dtype == np.float32,
            "value/target precision differs")
    require(np.array_equal(values["endpoint_values"], endpoints["endpoint_values"]),
            "endpoint values differ across files")
    g = reward @ (GAMMA ** np.arange(64, dtype=np.float64))
    close(sample["discounted_rewards"], g, "frozen reward discount", atol=1e-10, rtol=0)
    close(values["discounted_rewards"], g, "model reward discount", atol=1e-10, rtol=0)
    target = g + RHO * endpoints["endpoint_values"].astype(np.float64)
    close(values["targets_float64"], target, "float64 target", atol=1e-9, rtol=0)
    require(np.array_equal(values["targets_used_float32"], target.astype(np.float32)),
            "float32 loss target differs")
    delta = target - values["start_values"].astype(np.float64)
    close(values["residual"], delta, "saved-behavior residual", atol=1e-9, rtol=0)
    return g, target, delta


def check_counts(receipt):
    counts = receipt["counts"]
    required = {"ppo_load_attempted": 1, "ppo_load_returned": 1,
                "value_batches_attempted": 5, "value_batches_returned": 5,
                "value_states_attempted": 1024, "value_states_returned": 1024,
                "head_batches_attempted": 5, "head_batches_returned": 5,
                "head_states_attempted": 1024, "head_states_returned": 1024,
                "backward_attempted": 4, "backward_returned": 4}
    for key, expected in required.items():
        require(counts.get(key) == expected, f"call count differs: {key}")
    require(0 <= counts["torch_load_attempted"] == counts["torch_load_returned"] <= 3,
            "Torch load count differs")
    for key in ("actor_forward_attempted", "optimizer_step_attempted", "learn_attempted",
                "train_attempted", "save_attempted", "autograd_grad_attempted",
                "clip_attempted"):
        require(counts.get(key) == 0, f"forbidden operation count: {key}")
    require(receipt["physics_controls"] == receipt["native_steps"] == 0,
            "physics/native count differs")
    require(receipt["status"] == "passed" and receipt["source_postcheck_passed"] is True
            and receipt["cleanup_errors"] == [], "worker did not close cleanly")
    require(receipt["loaded_state"] == receipt["unchanged_state"] ==
            receipt["cleanup_state"], "policy/optimizer hash changed")
    require(receipt["shared_trainable_parameters"] == 0 and
            receipt["final_gradients_cleared"] is True and
            receipt["actor_forward_zero"] is True and
            receipt["physics_modules_and_libraries_absent"] is True and
            receipt["model_state_matches_recorded_final"] is True,
            "model/physics integrity differs")
    require(receipt["historical_GAE_reconstructed"] is False and
            receipt["on_policy_claim"] is False, "evidence boundary differs")


def check_gradient_batch(gradients, batch, parameters, starts, used_targets):
    expected_names = {k for k, v in parameters.items() if v["group"] == "critic"}
    actor_count = sum(v["group"] == "actor" for v in parameters.values())
    require(len(expected_names) == 6 and actor_count == 7 and
            set(gradients) == expected_names, "gradient parameter set differs")
    norms = {}
    for name, vector in gradients.items():
        require(list(vector.shape) == parameters[name]["shape"] and
                vector.dtype == np.float32 and np.isfinite(vector).all(),
                f"gradient invalid: {name}")
        norms[name] = float(np.linalg.norm(vector.astype(np.float64)))
        close(batch["per_parameter_l2"][name], norms[name], name + " norm")
        close(batch["per_parameter_grad_over_parameter_norm"][name],
              norms[name] / max(parameters[name]["l2_norm"], 1e-12),
              name + " relative norm")
    require(set(batch["per_parameter_l2"]) == expected_names and
            set(batch["per_parameter_grad_over_parameter_norm"]) == expected_names,
            "reported gradient names differ")
    total = math.sqrt(sum(value * value for value in norms.values()))
    close(batch["critic_gradient_l2"], total, "critic gradient norm")
    close(batch["critic_only_clip_factor"], min(1., .5 / (total + 1e-6)),
          "critic-only clip factor")
    expected_bias = float(np.mean(starts.astype(np.float64) -
                                  used_targets.astype(np.float64)))
    actual_bias = float(gradients["value_net.bias"].item())
    require(abs(actual_bias - expected_bias) <= 1e-5 + 1e-5 * abs(expected_bias),
            "raw head bias gradient differs from analytic result")
    close(batch["analytic_bias_gradient_expected"], expected_bias, "reported bias expectation")
    close(batch["actual_bias_gradient"], actual_bias, "reported raw bias gradient")
    loss = float(.5 * np.mean(np.square(starts.astype(np.float64) -
                                       used_targets.astype(np.float64))))
    close(batch["weighted_proxy_loss_float32"], loss, "weighted proxy loss", atol=1e-4)
    require(batch["actor_gradients_all_none"] is True and
            batch["actor_gradient_counts"] == {"none": actor_count, "zero": 0,
                                                 "nonzero": 0},
            "actor gradient counts differ")
    return total


def scale_gate(g, delta):
    scale = max(1., float(np.median(np.abs(g / D64))) / (1 - GAMMA))
    denom = (1 - RHO) * scale
    return scale, float(np.median(delta) / denom), float(np.sqrt(np.mean(delta**2)) / denom)


def distribution_stats(vector):
    return {"mean": float(np.mean(vector)), "median": float(np.median(vector)),
            "std": float(np.std(vector)), "min": float(np.min(vector)),
            "max": float(np.max(vector)), "p05": float(np.quantile(vector, .05)),
            "p95": float(np.quantile(vector, .95))}


def residual_stats(starts, targets):
    starts = np.asarray(starts, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    delta = targets - starts
    target_var = float(np.var(targets))
    return {"n": len(starts), "bias_target_minus_value": float(np.mean(delta)),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "median_error_target_minus_value": float(np.median(delta)),
            "p05_error": float(np.quantile(delta, .05)),
            "p95_error": float(np.quantile(delta, .95)),
            "value_mean": float(np.mean(starts)), "value_std": float(np.std(starts)),
            "target_mean": float(np.mean(targets)), "target_std": float(np.sqrt(target_var)),
            "explained_variance": None if target_var <= 1e-12 else
            float(1 - np.var(delta) / target_var)}


def check_tree(actual, expected, label):
    require(set(actual) == set(expected), f"{label} keys differ")
    for key, value in expected.items():
        got = actual[key]
        if value is None or isinstance(value, (int, bool)):
            require(got == value, f"{label}.{key} differs")
        else:
            close(got, value, f"{label}.{key}")


def check_sample_selection(sample, selection):
    check_window_indices(sample)
    n = len(sample["start_indices"])
    require(n == 512 and np.array_equal(sample["quartile"], np.repeat(np.arange(4), 128)),
            "sample batch order differs")
    require(selection["algorithm"] == "d1-value14-selection-v1" and
            selection["batch_boundaries"] == [[q * 128, (q + 1) * 128] for q in range(4)]
            and selection["selected_start_indices"] == sample["start_indices"].tolist(),
            "frozen selection differs")
    require(selection["counts"]["candidates_by_quartile"] == [245] * 4 and
            selection["counts"]["selected_by_quartile"] == [128] * 4,
            "candidate or selected counts differ")
    require(len(selection["sample_details"]) == n, "sample detail count differs")
    source_dir = Path(__file__).parent
    for key, name in (("plan_identity", "astra_plan_14.md"),
                      ("preparation_source_identity", "prepare_value14_data.py"),
                      ("math_source_identity", "value14_math.py"),
                      ("pure_test_source_identity", "test_value14_pure.py")):
        require(selection[key] == identity(source_dir / name),
                f"frozen selection source identity differs: {name}")
    for i, row in enumerate(selection["sample_details"]):
        require(row["start_index"] == sample["start_indices"][i] and
                row["endpoint_index"] == sample["endpoint_indices"][i] and
                row["quartile"] == sample["quartile"][i] and
                row["episode_index"] == sample["episode_index"][i] and
                row["episode_tick"] == sample["episode_tick"][i],
                f"sample detail row {i} index differs")
        require(row["terrain"] == sample["terrain_label"][i] and
                row["servo_yaw_active"] == bool(sample["window_has_yaw"][i]),
                f"sample detail row {i} grouping differs")
        close(row["window_mean_abs_servo_vx"],
              sample["window_mean_abs_servo_vx"][i], f"servo row {i}", atol=0, rtol=0)
        close(row["window_effective_action_nonzero_fraction"],
              sample["window_effective_action_nonzero_fraction"][i],
              f"effective row {i}", atol=0, rtol=0)
        for key, value in (("reward64_sha256", sample["rewards"][i]),
                           ("start_obs_sha256", sample["start_obs"][i]),
                           ("endpoint_obs_sha256", sample["endpoint_obs"][i])):
            require(row[key] == hashlib.sha256(value.tobytes()).hexdigest(),
                    f"sample detail row {i} {key} differs")


def group_masks(sample):
    q = sample["quartile"]
    masks = {"all": np.ones(len(q), dtype=bool)}
    masks.update({f"quartile:{i}": q == i for i in range(4)})
    masks.update({"terrain:" + name: sample["terrain_label"] == name
                  for name in sorted(set(sample["terrain_label"].tolist()))})
    f = sample["window_effective_action_nonzero_fraction"]
    masks.update({"effective:zero": f == 0, "effective:partial": (f > 0) & (f < 1),
                  "effective:all": f == 1})
    speed = sample["window_mean_abs_servo_vx"]
    masks.update({"speed:below_0p05": speed < .05,
                  "speed:0p05_to_0p8": (speed >= .05) & (speed < .8),
                  "speed:above_0p8": speed >= .8,
                  "yaw:present": sample["window_has_yaw"],
                  "yaw:absent": ~sample["window_has_yaw"]})
    return masks


def readback(run: Path, sample_dir: Path):
    selection_path = sample_dir / "selection.json"
    samples_path = sample_dir / "samples.npz"
    selection = json.loads(selection_path.read_text())
    require(identity(samples_path) == selection["samples_identity"],
            "frozen samples file identity differs")
    reservation = json.loads((run / "reservation.json").read_text())
    plan_path = Path(reservation["plan"])
    require(identity(plan_path) == reservation["plan_identity"] and
            reservation["retry_permitted"] is False, "reservation or plan identity differs")
    plan = json.loads(plan_path.read_text())
    require(Path(plan["output"]).resolve() == run.resolve() and
            Path(plan["samples"]).resolve() == samples_path.resolve(),
            "plan output/sample path differs")
    for path in (selection_path, samples_path, sample_dir.parent / "astra_plan_14.md"):
        require(plan["inputs"].get(str(path.resolve())) == identity(path),
                f"plan small input identity differs: {path.name}")
    receipt = json.loads((run / "receipt.json").read_text())
    require(receipt["plan_identity"] == reservation["plan_identity"],
            "worker plan identity differs")
    frozen_paths = {str(Path(name).resolve()) for name in plan["inputs"]}
    for key in ("compute_library_maps_before_load", "compute_library_maps_after"):
        paths = receipt.get(key)
        require(isinstance(paths, list), f"{key} missing")
        for name in paths:
            require(isinstance(name, str) and Path(name).is_absolute() and
                    str(Path(name).resolve()) == name and name in frozen_paths,
                    f"{key} contains an unfrozen compute library")
    check_counts(receipt)
    sample = npz(samples_path)
    check_sample_selection(sample, selection)
    endpoints = npz(run / "endpoint_values.npz")
    values = npz(run / "values.npz")
    require(identity(run / "values.npz") == receipt["arrays"],
            "worker values file identity differs")
    g, target, delta = check_values(sample, endpoints, values)
    parameters = receipt["parameters"]
    totals = []
    for q in range(4):
        sl = slice(q * 128, (q + 1) * 128)
        batch = json.loads((run / f"batch_{q}.json").read_text())
        batch_values = npz(run / f"batch_{q}_values.npz")
        require(batch == receipt["batches"][q] == receipt["completed_batches"][q],
                f"batch {q} receipt differs")
        require(batch["quartile"] == q and batch["rows"] == 128 and
                np.array_equal(batch_values["start_indices"], sample["start_indices"][sl]),
                f"batch {q} row order differs")
        for key in ("start_values", "endpoint_values", "targets_float64",
                    "targets_used_float32"):
            require(np.array_equal(batch_values[key], values[key][sl]),
                    f"batch {q} {key} differs")
        gradients = npz(run / f"batch_{q}_gradients.npz")
        totals.append(check_gradient_batch(gradients, batch, parameters,
                                           values["start_values"][sl],
                                           values["targets_used_float32"][sl]))
        s, b, e = scale_gate(g[sl], delta[sl])
        for key, expected in (("reward_scale_S", s), ("C_bias", b), ("C_rmse", e)):
            close(batch[key], expected, f"batch {q} {key}")
        check_tree(batch["stats"], residual_stats(values["start_values"][sl], target[sl]),
                   f"batch {q} stats")
    require(len(receipt["batches"]) == len(receipt["completed_batches"]) == 4,
            "batch count differs")
    masks = group_masks(sample)
    require(set(masks) == set(receipt["groups"]), "predefined group labels differ")
    group_result = {}
    for label, mask in masks.items():
        row = receipt["groups"][label]
        n = int(mask.sum())
        require(row["n"] == n and row["enough_for_group_decision"] == (n >= 32),
                f"group {label} size/gate differs")
        if not n:
            continue
        require(row["episodes"] == len(set(sample["episode_index"][mask].tolist())),
                f"group {label} episode count differs")
        s, b, e = scale_gate(g[mask], delta[mask])
        for key, expected in (("reward_scale_S", s), ("C_bias", b), ("C_rmse", e),
                              ("mae", float(np.mean(np.abs(delta[mask])))),
                              ("positive_residual_fraction", float(np.mean(delta[mask] > 0)))):
            close(row[key], expected, f"group {label} {key}")
        check_tree(row["metrics"], residual_stats(values["start_values"][mask], target[mask]),
                   f"group {label} metrics")
        distributions = {"discounted_rewards": g, "start_values": values["start_values"],
                         "endpoint_values": values["endpoint_values"],
                         "targets": target, "residual": delta}
        require(set(row["distributions"]) == set(distributions),
                f"group {label} distributions differ")
        for name, vector in distributions.items():
            check_tree(row["distributions"][name], distribution_stats(vector[mask]),
                       f"group {label} {name}")
        group_result[label] = {"n": n, "episodes": row["episodes"],
                               "S": s, "B": b, "E": e,
                               "decision_eligible": n >= 32}
    s, b, e = scale_gate(g, delta)
    for key, expected in (("S", s), ("C_bias", b), ("C_rmse", e),
                          ("positive_residual_fraction", float(np.mean(delta > 0))),
                          ("negative_residual_fraction", float(np.mean(delta < 0)))):
        close(receipt["normalization"][key], expected, f"global {key}")
    check_tree(receipt["stats"], residual_stats(values["start_values"], target),
               "global stats")
    require(receipt["normalization"]["gamma"] == GAMMA and
            receipt["normalization"]["horizon"] == 64 and
            math.isclose(receipt["normalization"]["D64"], D64),
            "global discount contract differs")
    quartile_b = [group_result[f"quartile:{q}"]["B"] for q in range(4)]
    under = b >= .1 and np.mean(delta > 0) >= .75 and sum(x >= .1 for x in quartile_b) >= 3 and quartile_b[3] >= .1
    over = b <= -.1 and np.mean(delta < 0) >= .75 and sum(x <= -.1 for x in quartile_b) >= 3 and quartile_b[3] <= -.1
    gates = receipt["gates"]
    require(gates["broad_underestimate"] == bool(under) and
            gates["broad_overestimate"] == bool(over) and
            gates["state_or_distribution_mismatch"] == bool(e >= .2 and not under and not over)
            and gates["critic_only_clip_pressure"] == bool(sum(x >= 5 for x in totals) >= 3 and totals[3] >= 5)
            and gates["all_batches_critic_norm_below_cap"] == bool(all(x <= .5 for x in totals))
            and gates["early_time_concentration"] == bool(abs(quartile_b[3]) < .1 and
                group_result["quartile:3"]["E"] < .2 and
                any(group_result[f"quartile:{q}"]["E"] >= .2 for q in (0, 1))),
            "global scientific gates differ")
    for family in ("terrain:", "effective:", "speed:", "yaw:"):
        eligible = {k: v for k, v in group_result.items()
                    if k.startswith(family) and v["decision_eligible"]}
        concentrated = len(eligible) >= 2 and any(
            value["E"] >= .2 and all(other["E"] < .1 for name, other in eligible.items()
                                       if name != key) for key, value in eligible.items())
        require(gates["state_group_concentration"][family] == bool(concentrated),
                f"state concentration gate differs: {family}")
    journal = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    events = Counter(row["event"] for row in journal)
    require(journal and journal[-1]["event"] == "worker_exit:passed" and
            journal[-1]["counts"] == receipt["counts"] and
            events["ppo_load_entry"] == 1 and events["value_entry"] == 5 and
            events["value_return"] == 5 and events["backward_entry"] == 4 and
            events["backward_return"] == 4 and events["batch_completed"] == 4,
            "journal accounting differs")
    output_files = ["receipt.json", "journal.jsonl", "reservation.json",
                    "endpoint_values.npz", "values.npz"] + [
                        f"batch_{q}{suffix}" for q in range(4) for suffix in
                        (".json", "_values.npz", "_gradients.npz")]
    return {"schema": "d1-value14-independent-readback-v1", "passed": True,
            "sample_identity": identity(samples_path),
            "selection_identity": identity(selection_path),
            "plan_identity": identity(plan_path),
            "verified_output_files": {name: identity(run / name) for name in output_files},
            "value_rows": 1024, "backward_calls": 4, "model_loads": 0,
            "physics_controls": 0, "optimizer_steps": 0,
            "gradient_l2_by_quartile": totals,
            "groups": group_result, "gates": gates,
            "boundary": "64-step saved-behavior bootstrapped target; not historical GAE, final on-policy return, or physics test"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = readback(args.run.resolve(), args.sample.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "model_loads": 0,
                      "verified_outputs": len(result["verified_output_files"])}))
