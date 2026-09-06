"""Plot separate same-input latency and closed-loop D1 development experiments.

Inputs are benchmark JSON and an evaluation directory containing summary.json
and steps.csv.gz. No experiment is rerun. A short or failed rollout is labelled
as such even when its available drive/brake trace looks stable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_rollout(directory: Path) -> tuple[dict, dict, bool]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    steps_path = directory / "steps.csv.gz"
    expected_hash = summary.get("artifacts", {}).get("steps", {}).get("sha256")
    if expected_hash and hashlib.sha256(steps_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("steps.csv.gz does not match the summary artifact hash")
    groups = defaultdict(list)
    with gzip.open(steps_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[(row["scenario"], int(row["seed"]))].append(row)
    expected = {(scenario, seed) for scenario in ("stand", "drive_brake", "push")
                for seed in (21, 22, 23)}
    episodes = summary.get("episodes", [])
    complete = bool(
        bool(expected_hash) and summary.get("protocol_complete") is True
        and summary.get("validation_passed") is True
        and summary.get("validation_status") == "passed" and len(episodes) == 9
        and {(item["scenario"], item["seed"]) for item in episodes} == expected
        and set(groups) == expected
        and all(item.get("completed") is True and item.get("validation_passed") is True
                and item.get("applied_steps") == 600 and item.get("elapsed_s") == 6.0
                for item in episodes)
    )
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["step"]))
        complete = bool(
            complete and len(rows) == 600
            and [int(row["step"]) for row in rows] == list(range(600))
            and all(row["applied"].lower() == "true" and row["status"] == "solved" for row in rows)
            and np.isclose(float(rows[-1]["output_time_s"]), 6.0)
        )
    drive = {
        seed: [row for row in groups.get(("drive_brake", seed), [])
               if row["applied"].lower() == "true"]
        for seed in (21, 22, 23)
    }
    return summary, drive, complete


def _flag(value) -> str:
    return "unknown" if value is None else str(bool(value)).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", type=Path, required=True, help="benchmark JSON")
    parser.add_argument("--rollout", type=Path, required=True, help="evaluation output directory")
    parser.add_argument("--output", type=Path, required=True, help="new PNG; never overwritten")
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix.lower() != ".png":
        parser.error("--output must be a new .png file")
    latency = json.loads(args.latency.read_text(encoding="utf-8"))
    rollout, drive, complete = _read_rollout(args.rollout)
    latency_warm = _flag(latency.get("candidate_configuration", {}).get("warm_start"))
    rollout_warm = _flag(rollout.get("controller_configuration", {}).get("warm_start"))
    equivalence = _flag(latency.get("numerical_equivalence_passed"))
    protocol = rollout.get("protocol", {})
    source = protocol.get("state_source", "unknown state source")
    randomization = protocol.get("domain_randomization")
    randomization_label = ("no domain randomization" if randomization is False
                           else "domain randomization" if randomization is True
                           else "randomization unspecified")
    development = "development" if rollout.get("development_only") is True else "scope unspecified"

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "medium"})
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5.7))
    figure.subplots_adjust(left=.07, right=.98, bottom=.26, top=.78, wspace=.25)
    figure.suptitle("D1 inverse dynamics: timing and closed-loop development", y=.96, fontsize=15)
    figure.text(.5, .895, "Two separate experiments; faster computation does not establish equivalence.",
                ha="center", color="#536171", fontsize=10)
    max_time = 10.0
    for name, color, label in (
        ("reference", "#586b82", f"Reference {latency.get('reference_commit', 'unknown')[:7]}"),
        ("candidate", "#087f8c", f"Candidate (warm_start={latency_warm})"),
    ):
        timings = np.asarray([value for run in latency["results"][name]["passes"]
                              for value in run["compute_ms"]], dtype=float)
        if not len(timings) or not np.isfinite(timings).all() or np.any(timings <= 0):
            raise ValueError(f"{name} latency samples must be finite and positive")
        ordered = np.sort(timings)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        left.step(np.r_[0., ordered], np.r_[0., cumulative], where="post", color=color,
                  linewidth=2, label=f"{label}\nP50 {np.median(timings):.2f} / "
                  f"P99 {np.percentile(timings, 99):.2f} ms; n={len(timings)}")
        max_time = max(max_time, float(ordered[-1]))
    left.axvline(10., color="#b45f35", linestyle="--", linewidth=1.2, label="10 ms reference")
    left.set(xlabel="Whole compute call (ms)", ylabel="Empirical cumulative fraction",
             xlim=(0, 1.06 * max_time), ylim=(0, 1.02),
             title="Same-input latency | host measurement")
    left.legend(loc="lower right", frameon=False, fontsize=8.5)
    left.grid(alpha=.18)

    for (seed, rows), color, style in zip(
        drive.items(), ("#2366a8", "#087f8c", "#c78226"), ("-", "--", ":"), strict=True
    ):
        if rows:
            right.plot([float(row["output_time_s"]) for row in rows],
                       [float(row["output_origin_forward_velocity_mps"]) for row in rows],
                       color=color, linestyle=style, linewidth=1.7, label=f"Seed {seed}")
    command_rows = max(drive.values(), key=len)
    if command_rows:
        right.step([float(row["input_time_s"]) for row in command_rows]
                   + [float(command_rows[-1]["output_time_s"])],
                   [float(row["command_forward_velocity_mps"]) for row in command_rows]
                   + [float(command_rows[-1]["command_forward_velocity_mps"])],
                   where="post", color="#263342", linewidth=1.5, linestyle="--", label="Command")
        right.legend(loc="upper right", frameon=False, fontsize=9)
    else:
        right.text(.5, .5, "No applied drive/brake samples", transform=right.transAxes, ha="center")
    status = "FULL PROTOCOL PASSED" if complete else "PARTIAL / FAILED / UNVERIFIED"
    right.set(xlabel="Simulation time (s)", ylabel="Forward velocity (m/s)",
              xlim=(0, max(6., float(protocol.get("seconds", 6.)))),
              title=f"Drive + brake | {status}")
    right.grid(alpha=.18)
    performance = _flag(latency.get("performance_passed", latency.get("passed")))
    figure.text(.07, .075, f"LEFT  warm_start={latency_warm}; performance_passed={performance}\n"
                f"numerical_equivalence_passed={equivalence}\n"
                "Host measurement, not a real-time guarantee",
                fontsize=9, linespacing=1.5, color="#364555")
    shown = ", ".join(str(seed) for seed, rows in drive.items() if rows) or "none"
    figure.text(.56, .075, f"RIGHT  warm_start={rollout_warm}; {development} / {source}\n"
                f"{randomization_label}; seeds shown: {shown}\n"
                "Protocol: 9 episodes; this panel shows drive/brake only",
                fontsize=9, linespacing=1.5, color="#364555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        figure.savefig(handle, format="png", dpi=180, facecolor="white")
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "rollout_complete_and_validated": complete,
                      "numerical_equivalence_passed": latency.get("numerical_equivalence_passed")}))


if __name__ == "__main__":
    main()
