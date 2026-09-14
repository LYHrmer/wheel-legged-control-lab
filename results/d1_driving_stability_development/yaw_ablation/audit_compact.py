"""Read-only numeric and checkpoint replay audit of this compact evidence bundle."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    repo = args.repo.resolve()
    if args.output.exists() or args.output.resolve().is_relative_to(bundle):
        raise ValueError("Output must be new and outside this evidence bundle.")
    sys.path[:0] = [str(repo / ".local-deps"), str(repo / "src"), str(repo), str(bundle)]
    import torch
    from audit_yaw_ablation import metrics
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    failures = []
    checks = 0

    def check(name, condition):
        nonlocal checks
        checks += 1
        if not bool(condition):
            failures.append(name)

    def close(name, a, b, atol=1e-8):
        a, b = np.asarray(a), np.asarray(b)
        check(
            name,
            a.shape == b.shape
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=1e-8),
        )

    protocol = read(bundle / "protocol.json")
    summary = read(bundle / "summary.json")
    schema = read(bundle / "compact_schema.json")
    for item in read(bundle / "compact_data_manifest.json")["files"]:
        file = bundle / item["path"]
        check(
            "compact data SHA:" + item["path"],
            sha(file) == item["sha256"] and file.stat().st_size == item["bytes"],
        )
    for path, expected in protocol["source_hashes"].items():
        check("source SHA:" + path, sha(repo / path) == expected)
    frozen = read(repo / protocol["frozen_source_file"]["path"])["sha256"]
    check(
        "frozen 77 source SHA",
        len(frozen) == 77 and all(sha(repo / p) == h for p, h in frozen.items()),
    )
    for key in ("evaluation_protocol_file", "frozen_source_file"):
        check(key + " SHA", sha(repo / protocol[key]["path"]) == protocol[key]["sha256"])
    check(
        "runner source SHA",
        sha(bundle / protocol["script"]["bundle_path"]) == protocol["script"]["sha256"],
    )
    models = {}
    for key, record in protocol["models"].items():
        for kind in ("model", "metadata"):
            check(
                key + ":" + kind + " SHA",
                sha(repo / record[kind + "_path"]) == record[kind + "_sha256"],
            )
        models[key] = PPO.load(repo / record["model_path"], device="cpu")
    proofs = {
        p["label"]: p for p in read(bundle / "full_rerun_bitwise_verification.json")["models"]
    }
    episodes = {}
    initial_reference = None
    results = []
    for label in protocol["episode_labels"]:
        folder = bundle / "episodes" / label
        initial = read(folder / "initial.json")
        saved = read(folder / "summary.json")
        with np.load(folder / "trace.npz", allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        n = len(data["reward"])
        mode = saved["intervention_mode"]
        check(label + ": count", n == saved["n_steps"] and len(data["qpos"]) == n + 1)
        if initial_reference is None:
            initial_reference = initial
        check(label + ": shared initial record", initial == initial_reference)
        for key in ("observation", "qpos", "qvel"):
            digest = hashlib.sha256(np.ascontiguousarray(data[key][0]).tobytes()).hexdigest()
            check(label + ": initial " + key, digest == initial[key + "_sha256"])
        if label in proofs:
            for key, proof in proofs[label]["arrays"].items():
                actual = {
                    "shape": list(data[key].shape),
                    "dtype": str(data[key].dtype),
                    "sha256": hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest(),
                }
                check(
                    label + ": original full " + key, actual == proof["original"] == proof["fresh"]
                )
        if mode == "zero":
            expected = np.zeros((n, 8))
        else:
            expected = models[f"seed{saved['policy_seed']}/{saved['policy_variant']}"].predict(
                data["observation"][:-1], deterministic=True
            )[0]
        close(
            label + ": current branch observation replay",
            data["predicted_action"],
            expected,
            atol=2e-6,
        )
        expected = data["predicted_action"].copy()
        if mode == "zero_legs":
            expected[:, :4] = 0
        elif mode == "common_wheels":
            expected[:, 4:] = expected[:, 4:].mean(1, keepdims=True)
        for key in ("intervened_action", "applied_action", "raw_action", "policy_action"):
            close(label + ": " + key, data[key], expected)
        valid = data["terminal_observation_source"] != "last_executable_decision"
        close(
            label + ": previous action feedback",
            data["observation"][1:, 74:82][valid],
            data["applied_action"][valid].astype(np.float32),
        )
        close(label + ": reward sum", data["reward"], data["reward_terms"].sum(1))
        close(label + ": control dt", np.diff(data["time_s"]), np.full(n, 0.01))
        check(
            label + ": first terminal",
            not np.any(data["terminated"][:-1] | data["truncated"][:-1])
            and (data["terminated"][-1] or data["truncated"][-1]),
        )
        rows = [
            {
                "reward": float(data["reward"][i]),
                "info": {
                    "metrics": dict(zip(schema["metrics_columns"], data["metrics"][i])),
                    "applied_action": data["applied_action"][i],
                },
            }
            for i in range(n)
        ]
        states = {key: data[key] for key in ("observation", "qpos", "qvel", "time_s")}
        computed = metrics(rows, states, n)
        for key, value in computed.items():
            close(label + ": summary " + key, value, saved["metrics"][key])
        episodes[label] = (rows, states)
        results.append({"label": label, "steps": n, "terminal_reason": saved["terminal_reason"]})
    for key, comparison in summary["per_model_common_prefix_comparison"].items():
        seed, variant = key.split("/")
        labels = {
            mode: ("zero" if mode == "zero" else seed + "_" + variant + "_" + mode)
            for mode in ("zero", "full", "zero_legs", "common_wheels")
        }
        n = min(len(episodes[label][0]) for label in labels.values())
        check(key + ": common prefix length", n == comparison["common_prefix_steps"])
        for mode, label in labels.items():
            for field, value in metrics(*episodes[label], n).items():
                close(
                    key + ": prefix " + mode + " " + field,
                    value,
                    comparison["prefix_metrics"][mode][field],
                )
    report = {
        "schema": "d1-yaw-compact-independent-audit-v1",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "new_physical_steps": 0,
        "auditor_sha256": sha(Path(__file__)),
        "recorded_episodes": results,
        "recorded_physical_steps": sum(r["steps"] for r in results),
        "models_replayed": len(models),
        "full_arrays_verified_against_original_digests": 6,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed": report["passed"], "checks": checks, "failures": failures}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
