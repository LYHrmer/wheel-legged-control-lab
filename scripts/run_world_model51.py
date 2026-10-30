"""Run the preregistered CPU-only, zero-simulation dynamics pilot once."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from world_model51 import (
    evaluate,
    extract,
    fit_baseline,
    gate,
    require,
    sha256,
    train_residual,
    write_json,
    write_npz,
)


def verify_identities(root, identities):
    for relative, expected in identities.items():
        path = root / relative
        require(
            sha256(path) == expected["sha256"] and path.stat().st_size == expected["bytes"],
            f"registered identity changed: {relative}",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "offline-world-model51-result-v1",
        "status": "INCOMPLETE",
        "failure": None,
        "new_physics_steps": 0,
        "policy_updates": 0,
        "simulator_imported": False,
        "protocol_sha256": sha256(args.protocol),
        "seeds": [],
        "candidate_passed": False,
        "decision": "DO_NOT_START_POLICY_TRAINING",
        "source_policy_seed": 201021,
        "model_initialization_seeds": protocol["seeds"],
        "interpretation": "recorded-torque-conditioned physical-state prediction only",
    }
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
        require(
            len(protocol["seeds"]) == len(set(protocol["seeds"])) == 3,
            "exactly three distinct model seeds required",
        )
        all_episodes = [e for group in protocol["splits"].values() for e in group]
        require(
            len(all_episodes) == len(set(all_episodes)) == 24
            and set(all_episodes) == set(protocol["episodes"]),
            "split overlap or omission",
        )
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10, "Nice10 required")
        require(len(os.sched_getaffinity(0)) == 1, "single CPU affinity required")
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            require(os.environ.get(name) == "1", f"{name}=1 required")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU visibility must be disabled")
        verify_identities(root, protocol["frozen_inputs"])
        require(
            len(protocol["implementation"]) >= 3, "implementation must be sealed before execution"
        )
        verify_identities(root, protocol["implementation"])
        report["execution_commit"] = subprocess.check_output(
            ["rtk", "proxy", "git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        data = extract(root, protocol, args.output)
        require(time.process_time() - cpu_started < 360, "data CPU budget exhausted")
        splits = protocol["splits"]
        train_mask = np.isin(data["episode"], splits["train"])
        baseline = fit_baseline(data, train_mask, protocol["fit"]["ridge"])
        write_npz(args.output / "baseline.npz", **baseline)
        baseline_scores = {
            name: {str(h): evaluate(baseline, data, episodes, h) for h in (1, 8)}
            for name, episodes in splits.items()
        }
        report["baseline"] = baseline_scores
        write_json(args.output / "baseline_scores.json", baseline_scores)
        for seed in protocol["seeds"]:
            output = args.output / f"seed_{seed}"
            output.mkdir(exist_ok=False)
            result = {"seed": seed, "status": "INCOMPLETE", "failure": None}
            try:
                model, training = train_residual(
                    baseline,
                    data,
                    train_mask,
                    seed,
                    protocol["fit"],
                    output,
                    global_cpu_deadline=cpu_started + 1800,
                )
                result["training"] = training
                result["scores"] = {
                    name: {str(h): evaluate(model, data, episodes, h) for h in (1, 8)}
                    for name, episodes in splits.items()
                }
                result["gate"] = gate(
                    baseline_scores["test"], result["scores"]["test"], protocol["gates"]
                )
                result["status"] = "COMPLETE"
            except Exception as exc:  # noqa: BLE001 — preserve any failed seed; never retry it.
                result["failure"] = f"{type(exc).__name__}: {exc}"
                with (output / "failure.txt").open("x") as stream:
                    traceback.print_exc(file=stream)
            write_json(output / "result.json", result)
            report["seeds"].append(result)
            print(
                json.dumps({"seed": seed, "status": result["status"], "gate": result.get("gate")}),
                flush=True,
            )
            require(time.process_time() - cpu_started < 1800, "total CPU budget exhausted")
        verify_identities(root, protocol["frozen_inputs"])
        verify_identities(root, protocol["implementation"])
        report["source_identities_unchanged"] = True
        complete = all(r["status"] == "COMPLETE" for r in report["seeds"])
        report["status"] = "COMPLETE" if complete else "INCOMPLETE"
        report["candidate_passed"] = complete and all(r["gate"]["passed"] for r in report["seeds"])
        report["decision"] = (
            "ELIGIBLE_FOR_A_SEPARATE_POLICY_STUDY_PROPOSAL"
            if report["candidate_passed"]
            else "DO_NOT_START_POLICY_TRAINING"
        )
        if complete:
            values = [r["scores"]["test"]["8"]["velocity_nrmse"] for r in report["seeds"]]
            report["test_h8_seed_summary"] = {
                "mean": float(np.mean(values)),
                "minimum": float(min(values)),
                "maximum": float(max(values)),
                "range": float(np.ptp(values)),
            }
        report["artifact_bytes"] = sum(
            p.stat().st_size for p in args.output.rglob("*") if p.is_file()
        )
        require(report["artifact_bytes"] <= 100 << 20, "artifact budget exceeded")
    except Exception as exc:  # noqa: BLE001 — record failure evidence before exiting nonzero.
        report["status"], report["failure"] = "INCOMPLETE", f"{type(exc).__name__}: {exc}"
        report["candidate_passed"] = False
        report["decision"] = "DO_NOT_START_POLICY_TRAINING"
        with (args.output / "failure.txt").open("x") as stream:
            traceback.print_exc(file=stream)
    report["wall_seconds"] = time.monotonic() - started
    report["cpu_seconds"] = time.process_time() - cpu_started
    write_json(args.output / "report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("baseline", "seeds")}), flush=True
    )
    return int(report["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
