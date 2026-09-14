"""Independently replay saved yaw-ablation policy inputs and recompute comparisons."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, states, n):
    rows = rows[:n]
    q = states["qpos"][: n + 1]
    w, x, y, z = q[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    result = {}
    for field, key in (
        ("velocity_error_mps", "velocity_rmse_mps"),
        ("yaw_rate_error_rps", "yaw_rate_rmse_rps"),
        ("height_error_m", "height_rmse_m"),
        ("roll_error_rad", "roll_rmse_rad"),
        ("pitch_error_rad", "pitch_rmse_rad"),
    ):
        result[key] = float(np.sqrt(np.mean([r["info"]["metrics"][field] ** 2 for r in rows])))
    applied = np.asarray([r["info"]["applied_action"] for r in rows])
    common = applied[:, 4:].mean(1)
    result.update(
        steps=n,
        simulated_duration_s=float(states["time_s"][n] - states["time_s"][0]),
        heading_rmse_rad=float(np.sqrt(np.mean((yaw[1:] - yaw[0]) ** 2))),
        final_heading_change_rad=float(yaw[-1] - yaw[0]),
        lateral_rmse_m=float(np.sqrt(np.mean((q[1:, 1] - q[0, 1]) ** 2))),
        max_abs_lateral_displacement_m=float(np.abs(q[:, 1] - q[0, 1]).max()),
        final_y_m=float(q[-1, 1]),
        final_x_m=float(q[-1, 0]),
        wheel_common_mean=float(common.mean()),
        wheel_differential_rms_mean=float(
            np.sqrt(np.mean((applied[:, 4:] - common[:, None]) ** 2, axis=1)).mean()
        ),
        mean_executed_action_8=applied.mean(0).tolist(),
    )
    result["return"] = float(sum(r["reward"] for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("output must be new and outside the run directory")
    sys.path[:0] = [str(args.repo / ".local-deps"), str(args.repo / "src"), str(args.repo)]
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    failures, checks = [], []

    def check(label, condition):
        checks.append(label)
        if not bool(condition):
            failures.append(label)

    def close(label, a, b, atol=1e-8):
        a, b = np.asarray(a), np.asarray(b)
        check(
            label,
            a.shape == b.shape
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=1e-8),
        )

    protocol, summary = read(args.run / "protocol.json"), read(args.run / "summary.json")
    reference = read(args.reference)
    check("exact development case", protocol["case"] == reference["case"])
    check(
        "all 19 completed records",
        summary["n_episodes"] == 19 and summary["all_episodes_completed"],
    )
    check(
        "sources unchanged report",
        read(args.run / "integrity.json")["sources_models_protocol_and_runner_unchanged"],
    )
    for path in args.run.rglob("manifest.json"):
        manifest = read(path)
        for item in manifest["files"]:
            file = path.parent / item["path"]
            check(
                f"manifest:{file}",
                file.is_file()
                and file.stat().st_size == item["bytes"]
                and sha(file) == item["sha256"],
            )
        check(
            f"manifest inventory:{path}",
            {x["path"] for x in manifest["files"]}
            == {
                str(f.relative_to(path.parent))
                for f in path.parent.rglob("*")
                if f.is_file() and f != path
            },
        )
    for relative, expected in protocol["source_hashes"].items():
        check(f"source:{relative}", sha(args.repo / relative) == expected)
    frozen = read(Path(protocol["frozen_source_file"]["path"]))["sha256"]
    check(
        "77 frozen sources",
        len(frozen) == 77 and all(sha(args.repo / p) == h for p, h in frozen.items()),
    )
    loaded = {}
    for label, record in protocol["models"].items():
        for kind in ("model", "metadata"):
            check(
                f"model:{label}:{kind}",
                sha(Path(record[kind + "_path"])) == record[kind + "_sha256"],
            )
        loaded[label] = PPO.load(record["model_path"], device="cpu")
    first = None
    episodes, results = {}, []
    references = {f"seed{r['seed']}_{r['variant']}_full": r for r in reference["references"]}
    for label in protocol["episode_labels"]:
        folder = args.run / "episodes" / label
        with np.load(folder / "states.npz", allow_pickle=False) as archive:
            states = {k: archive[k] for k in archive.files}
        rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
        original, init = read(folder / "summary.json"), read(folder / "initial.json")
        n, mode = len(rows), original["intervention_mode"]
        check(
            f"{label}: complete state count",
            n == original["n_steps"] and all(len(a) == n + 1 for a in states.values()),
        )
        check(f"{label}: finite states", all(np.isfinite(a).all() for a in states.values()))
        if first is None:
            first = init
        check(f"{label}: identical initial record", init == first)
        for key in ("observation", "qpos", "qvel"):
            check(
                f"{label}: initial {key} bytes",
                hashlib.sha256(np.ascontiguousarray(states[key][0]).tobytes()).hexdigest()
                == init[key + "_sha256"],
            )
        predicted = np.asarray([r["predicted_action"] for r in rows])
        if mode == "zero":
            expected = np.zeros((n, 8))
        else:
            model = loaded[f"seed{original['policy_seed']}/{original['policy_variant']}"]
            expected = model.predict(states["observation"][:-1], deterministic=True)[0].astype(
                float
            )
        close(f"{label}: branch observation policy replay", predicted, expected, atol=2e-6)
        expected = predicted.copy()
        if mode == "zero_legs":
            expected[:, :4] = 0
        elif mode == "common_wheels":
            expected[:, 4:] = predicted[:, 4:].mean(1, keepdims=True)
        close(f"{label}: intervened action", [r["intervened_action"] for r in rows], expected)
        applied = np.asarray([r["info"]["applied_action"] for r in rows])
        close(f"{label}: actual applied action", applied, expected)
        valid = np.array(
            [
                r["info"].get("terminal_observation_source") != "last_executable_decision"
                for r in rows
            ]
        )
        close(
            f"{label}: next previous action",
            states["observation"][1:, 74:82][valid],
            applied.astype(np.float32)[valid],
        )
        close(f"{label}: dt", np.diff(states["time_s"]), np.full(n, 0.01))
        close(
            f"{label}: reward contributions",
            [r["reward"] for r in rows],
            [sum(r["info"]["reward_terms"].values()) for r in rows],
        )
        close(
            f"{label}: state position trace",
            states["qpos"][1:, :3],
            [[r["info"]["metrics"][k] for k in ("x_m", "y_m", "z_m")] for r in rows],
        )
        check(
            f"{label}: first terminal ended branch",
            not any(r["terminated"] or r["truncated"] for r in rows[:-1])
            and (rows[-1]["terminated"] or rows[-1]["truncated"]),
        )
        result = metrics(rows, states, n)
        for key, value in result.items():
            close(f"{label}: summary:{key}", original["metrics"][key], value)
        if label in references:
            old = Path(references[label]["old_full_directory"])
            with np.load(old / "states.npz", allow_pickle=False) as archive:
                check(
                    f"{label}: original full all state arrays bit-identical",
                    all(np.array_equal(states[k], archive[k]) for k in states),
                )
            check(
                f"{label}: original full checkpoint SHA",
                original["model_sha256"] == references[label]["model_sha256"],
            )
        result.update(
            label=label,
            mode=mode,
            terminal_reason=original["terminal_reason"],
            evaluation_completed=original["evaluation_completed"],
        )
        episodes[label] = rows, states
        results.append(result)
    comparisons = {}
    for seed in (48001, 48002, 48003):
        for variant in ("unbounded", "bounded"):
            key = f"seed{seed}/{variant}"
            labels = {
                m: ("zero" if m == "zero" else f"seed{seed}_{variant}_{m}")
                for m in ("zero", "full", "zero_legs", "common_wheels")
            }
            n = min(len(episodes[label][0]) for label in labels.values())
            prefix = {mode: metrics(*episodes[label], n) for mode, label in labels.items()}
            stored = summary["per_model_common_prefix_comparison"][key]
            check(f"{key}: common prefix count", stored["common_prefix_steps"] == n)
            for mode, values in prefix.items():
                for field, value in values.items():
                    close(
                        f"{key}:prefix:{mode}:{field}", stored["prefix_metrics"][mode][field], value
                    )
            comparisons[key] = {"common_prefix_steps": n, "metrics": prefix}
    result = {
        "schema": "d1-yaw-ablation-independent-audit-v1",
        "passed": not failures,
        "check_count": len(checks),
        "failures": failures,
        "physical_steps_executed_by_auditor": 0,
        "runner_source_sha256": protocol["script"]["sha256"],
        "auditor_sha256": sha(Path(__file__)),
        "total_recorded_physical_steps": sum(r["steps"] for r in results),
        "episodes": results,
        "per_model_common_prefix": comparisons,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "check_count": len(checks),
                "failures": failures,
                "output": str(args.output),
            }
        )
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
