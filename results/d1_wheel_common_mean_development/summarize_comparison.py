#!/usr/bin/env python3
"""Summarize frozen evaluation records without loading policies or running physics.

Usage: python3 summarize_comparison.py --input EVALUATION_ROOT --output NEW_DIR
Full-episode means give each declared case equal weight despite unequal terminal
durations. Common-prefix diagnostics retain all recorded rewards, including any
termination penalty at the cutoff. Neither view establishes task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "d1-wheel-common-mean-comparison-v1"
VARIANTS = ("unbounded", "bounded")
ERROR_FIELDS = {
    "velocity_rmse_mps": "velocity_error_mps",
    "yaw_rate_rmse_rps": "yaw_rate_error_rps",
    "height_rmse_m": "height_error_m",
    "roll_rmse_rad": "roll_error_rad",
    "pitch_rmse_rad": "pitch_error_rad",
}
MEAN_FIELDS = (*ERROR_FIELDS, "cumulative_return", "signed_forward_progress_m")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("input contains a nonfinite number")
    return result


def reject_constant(value):
    raise ValueError(f"input contains nonstandard JSON constant {value}")


def decode(text):
    return json.loads(text, parse_float=finite_float, parse_constant=reject_constant)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"inconsistent {label}: {actual!r} != {expected!r}")


def trace_snapshots(path, summary, cutoffs):
    """Read each transition once; snapshots are inclusive of the kth transition."""
    squared = dict.fromkeys(ERROR_FIELDS, 0.0)
    total_return, terms, snapshots = 0.0, None, {}
    for_count = 0
    with path.open() as stream:
        for count, line in enumerate(stream, 1):
            row = decode(line)
            if row["transition"] != count:
                raise ValueError(f"nonconsecutive transition at {path}:{count}")
            metrics, command = row["metrics"], row["command"]
            close(metrics["time_s"], count * .01, f"{path}:{count} time")
            errors = {
                "velocity_rmse_mps": metrics["velocity_mps"] - command["forward_velocity_mps"],
                "yaw_rate_rmse_rps": metrics["yaw_rate_rps"] - command["yaw_rate_rps"],
                "height_rmse_m": metrics["clearance_m"] - command["clearance_m"],
                "roll_rmse_rad": metrics["roll_error_rad"],
                "pitch_rmse_rad": metrics["pitch_error_rad"],
            }
            for output_field, error in errors.items():
                close(error, metrics[ERROR_FIELDS[output_field]], f"{path}:{count} error")
                squared[output_field] += error * error
            close(row["reward"], sum(row["reward_terms"].values()), f"{path}:{count} reward")
            if terms is None:
                terms = dict.fromkeys(row["reward_terms"], 0.0)
            if set(terms) != set(row["reward_terms"]):
                raise ValueError(f"reward fields changed within {path}")
            for name in terms:
                terms[name] += row["reward_terms"][name]
            total_return += row["reward"]
            if (row["terminated"] or row["truncated"]) and count != summary["actual_transitions"]:
                raise ValueError(f"records continue after termination in {path}")
            if count in cutoffs:
                snapshots[count] = {
                    "actual_transitions": count,
                    "integrated_duration_s": metrics["time_s"],
                    **{name: math.sqrt(value / count) for name, value in squared.items()},
                    "initial_x_m": summary["initial_x_m"], "final_x_m": metrics["x_m"],
                    "signed_forward_progress_m": metrics["x_m"] - summary["initial_x_m"],
                    "cumulative_return": total_return, "cumulative_reward_terms": dict(terms),
                    "terminated_at_endpoint": row["terminated"],
                    "truncated_at_endpoint": row["truncated"],
                    "terminal_reason_at_endpoint": row["terminal_reason"],
                    "geometric_nonflat_exposure": row["terrain_exposure"],
                }
            for_count = count
    if for_count != summary["actual_transitions"] or set(snapshots) != set(cutoffs):
        raise ValueError(f"trace length or requested prefixes are incomplete: {path}")
    full = snapshots[for_count]
    for name in (*MEAN_FIELDS, "integrated_duration_s", "initial_x_m", "final_x_m"):
        close(full[name], summary[name], f"{path} full {name}")
    if set(full["cumulative_reward_terms"]) != set(summary["cumulative_reward_terms"]):
        raise ValueError(f"reward fields differ from full summary: {path}")
    for name, value in full["cumulative_reward_terms"].items():
        close(value, summary["cumulative_reward_terms"][name], f"{path} full {name}")
    for name in ("terminated", "truncated", "terminal_reason"):
        if full[f"{name}_at_endpoint"] != summary[name]:
            raise ValueError(f"terminal summary disagrees with trace: {path}")
    if not (full["terminated_at_endpoint"] or full["truncated_at_endpoint"]):
        raise ValueError(f"trace lacks a terminal transition: {path}")
    return snapshots


def equal_case_mean(records):
    return {name: sum(row[name] for row in records) / len(records) for name in MEAN_FIELDS}


def analyze(root):
    """Reject missing pairs or records; preserve every original full summary."""
    paths = sorted({root / "protocol.json", *root.rglob("summary.json"),
                    *root.rglob("trace.jsonl"), *root.rglob("initial.json")})
    before = {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
              for path in paths}
    protocol = decode((root / "protocol.json").read_text())
    source = decode((root / "summary.json").read_text())
    if source.get("mode") != "evaluate" or not source.get("all_runs_completed"):
        raise ValueError("input must contain all normally recorded evaluation runs")
    cases = protocol["selected_case_names"]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("selected case names must be nonempty and unique")
    specs = {case["name"]: case for case in protocol["evaluation_protocol"]["cases"]}
    if any(case not in specs for case in cases):
        raise ValueError("selected case is absent from the frozen protocol")
    planned = {case: round(specs[case]["episode_seconds"] / .01) for case in cases}
    index, seeds = {}, set()
    for record in source["results"]:
        label, case = record["policy"], record["case"]
        match = re.fullmatch(r"seed(0|[1-9][0-9]*)/(unbounded|bounded)", label)
        if label != "zero" and match is None:
            raise ValueError(f"unsupported policy label {label!r}")
        if match:
            seeds.add(int(match.group(1)))
        if case not in cases or (case, label) in index:
            raise ValueError("unknown case or duplicate case/policy summary")
        if type(record.get("actual_transitions")) is not int or not (
                0 < record["actual_transitions"] <= planned[case]):
            raise ValueError(f"invalid transition count in {case}/{label}")
        index[case, label] = record
    seeds = sorted(seeds)
    if not seeds or set(seeds) != set(protocol["seeds"]):
        raise ValueError("inferred seed labels disagree with the evaluation protocol")
    labels = ["zero", *(f"seed{seed}/{variant}" for seed in seeds for variant in VARIANTS)]
    if set(index) != {(case, label) for case in cases for label in labels}:
        raise ValueError("each declared case must contain zero and both variants for every seed")
    expected_files = {"summary.json", "protocol.json"}
    for case in cases:
        for label in labels:
            expected_files.update(f"{case}/{label}/{name}" for name in
                                  ("summary.json", "trace.jsonl", "initial.json"))
    if set(before) != expected_files:
        raise ValueError("missing or unexpected summary, trace or initial records")

    prefixes, full_checks = [], []
    for case in cases:
        lengths = {seed: min(index[case, label]["actual_transitions"] for label in
                            ("zero", *(f"seed{seed}/{variant}" for variant in VARIANTS)))
                   for seed in seeds}
        snapshots = {}
        initial = decode((root / case / "zero/initial.json").read_text())
        for label in labels:
            summary = index[case, label]
            directory = root / case / label
            if decode((directory / "summary.json").read_text()) != {
                    key: value for key, value in summary.items() if key != "case"}:
                raise ValueError(f"root and per-run summaries differ: {case}/{label}")
            if decode((directory / "initial.json").read_text()) != initial:
                raise ValueError(f"initial state/observation metadata differs: {case}/{label}")
            close(summary["initial_x_m"], index[case, "zero"]["initial_x_m"], "initial x")
            cutoffs = set(lengths.values()) if label == "zero" else {
                lengths[int(label.split("/")[0][4:])]}
            cutoffs.add(summary["actual_transitions"])
            snapshots[label] = trace_snapshots(directory / "trace.jsonl", summary, cutoffs)
            full_checks.append({"case": case, "policy": label, "recomputed_full_metrics":
                                snapshots[label][summary["actual_transitions"]]})
        for seed in seeds:
            count = lengths[seed]
            prefixes.append({
                "seed": seed, "case": case, "common_transitions": count,
                "planned_transitions": planned[case],
                "policies": {variant: {"policy": label, "original_actual_transitions":
                                       index[case, label]["actual_transitions"],
                                       **snapshots[label][count]}
                             for variant, label in (("zero", "zero"),
                                                    *((v, f"seed{seed}/{v}") for v in VARIANTS))},
            })
    aggregates = []
    for seed in seeds:
        for variant in ("zero", *VARIANTS):
            label = "zero" if variant == "zero" else f"seed{seed}/{variant}"
            records = [index[case, label] for case in cases]
            completions = [{"case": row["case"], "planned_transitions": planned[row["case"]],
                            **{key: row[key] for key in ("actual_transitions", "integrated_duration_s",
                               "terminated", "truncated", "terminal_reason")},
                            "completed": row["actual_transitions"] == planned[row["case"]]
                            and row["terminal_reason"] == "time_limit" and row["truncated"]
                            and not row["terminated"]} for row in records]
            prefix_records = [entry["policies"][variant] for entry in prefixes if entry["seed"] == seed]
            aggregates.append({"seed": seed, "variant": variant, "policy": label,
                               "case_count": len(cases),
                               "completed_count": sum(row["completed"] for row in completions),
                               "durations_and_termination": completions,
                               "equal_case_mean_full_episode": equal_case_mean(records),
                               "equal_case_mean_common_prefix": equal_case_mean(prefix_records)})
    after_paths = sorted({root / "protocol.json", *root.rglob("summary.json"),
                         *root.rglob("trace.jsonl"), *root.rglob("initial.json")})
    if paths != after_paths or any(sha256(root / name) != value["sha256"]
                                   for name, value in before.items()):
        raise RuntimeError("input records changed during analysis")
    report = {
        "schema": SCHEMA, "input_root": str(root), "selected_split": protocol["selected_split"],
        "cases": cases, "seeds": seeds, "case_weight": 1 / len(cases),
        "definitions": {
            "completed_count": "Runs reaching the planned horizon by time_limit truncation, without termination; not terrain traversal proof.",
            "equal_case_mean_full_episode": "Arithmetic mean of each case's full observed metric. Different terminal durations remain different; no pooling or padding.",
            "common_prefix": "For each seed/case, K=min(actual steps of zero, unbounded, bounded); recompute all three on recorded transitions 1..K inclusive, retaining endpoint termination penalties.",
            "common_prefix_limit": "Equal duration does not imply equal trajectory or terrain exposure. Cutoff uses observed termination; this is a diagnostic, not an independent success metric.",
            "signed_progress": "Endpoint world x minus initial world x, not path length or command-integrated distance.",
            "zero_reuse": "One zero run per case is reused in each seed comparison; copies are not independent replications.",
        },
        "unequal_full_episode_durations_present": any(
            len({index[case, label]["actual_transitions"] for label in labels}) > 1 for case in cases),
        "skill_improvement_established": False,
        "full_results": source["results"], "full_metrics_recomputed_from_traces": full_checks,
        "per_seed_equal_case_means": aggregates, "common_prefix_comparisons": prefixes,
        "input_records_unchanged": True,
    }
    return report, before


def markdown(report):
    lines = ["Full observed episodes; each case has equal weight. Unequal terminal durations remain visible.",
             "These averages do not establish successful completion or terrain traversal.", "",
             "| Seed | Policy | Completed/cases | vx RMSE | Yaw RMSE | Height RMSE | Return | Durations / reason |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in report["per_seed_equal_case_means"]:
        mean = row["equal_case_mean_full_episode"]
        durations = "; ".join(f"{d['case']}: {d['integrated_duration_s']:.2f}s/{d['terminal_reason']}"
                              for d in row["durations_and_termination"])
        lines.append(f"| {row['seed']} | {row['variant']} | {row['completed_count']}/{row['case_count']} "
                     f"| {mean['velocity_rmse_mps']:.6f} | {mean['yaw_rate_rmse_rps']:.6f} "
                     f"| {mean['height_rmse_m']:.6f} | {mean['cumulative_return']:.6f} | {durations} |")
    lines.extend(["", "Common prefix per seed/case: recorded transitions 1..K, without padding or continuation.",
                  "Any termination penalty on transition K remains included; equal time does not mean equal terrain exposure.", "",
                  "| Seed | Case | K / seconds | Policy | vx RMSE | Yaw RMSE | Height RMSE | Return | Signed x progress |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|"])
    for entry in report["common_prefix_comparisons"]:
        for variant, row in entry["policies"].items():
            lines.append(f"| {entry['seed']} | {entry['case']} | {entry['common_transitions']} / "
                         f"{row['integrated_duration_s']:.2f} | {variant} | {row['velocity_rmse_mps']:.6f} "
                         f"| {row['yaw_rate_rmse_rps']:.6f} | {row['height_rmse_m']:.6f} "
                         f"| {row['cumulative_return']:.6f} | {row['signed_forward_progress_m']:.6f} |")
    lines.extend(["", "Units: vx m/s; yaw rad/s; height and progress m. Zero is reused, not replicated.",
                  "JSON preserves all original full summaries, per-case reward terms, prefix termination flags, and input hashes.", ""])
    return "\n".join(lines)


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root, output = args.input.resolve(strict=True), args.output.absolute()
    if output.exists() or any(path.is_symlink() for path in (output, *output.parents)):
        raise FileExistsError("output must be a new directory without symlink ancestors")
    if output == root or root in output.parents:
        raise ValueError("output must be outside the read-only evaluation directory")
    report, inputs = analyze(root)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", report)
    write_json(output / "input_manifest.json", inputs)
    with (output / "comparison.md").open("x") as stream:
        stream.write(markdown(report))
    write_json(output / "manifest.json", {
        "helper_sha256": sha256(Path(__file__)),
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(output.iterdir()) if path.is_file()},
    })
    print(json.dumps({"output": str(output), "seeds": report["seeds"], "cases": report["cases"],
                      "all_full_metrics_recomputed": True}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
