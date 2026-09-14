#!/usr/bin/env python3
"""Read paired GUI records and verify the final brake adapter against baseline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(base, repo):
    report = {"forward": [], "side": [], "input_sha256": {}}
    for old_group, new_group, kind in (("forward_02", "forward_braking_01", "forward"),
                                      ("side_01", "side_braking_01", "side")):
        old_summary = json.loads((base / old_group / "summary.json").read_text())
        new_summary = json.loads((base / new_group / "summary.json").read_text())
        assert [row["case"] for row in old_summary] == [row["case"] for row in new_summary]
        for old, new in zip(old_summary, new_summary):
            name = old["case"]["name"]
            before, after = base / old_group / name, base / new_group / name
            for directory in (before, after):
                for filename in ("states.npz", "telemetry.csv", "protocol.json", "summary.json", "manifest.json"):
                    file = directory / filename
                    report["input_sha256"][str(file.relative_to(base))] = sha(file)
            protocol = json.loads((after / "protocol.json").read_text())
            assert protocol["schema"] == "d1-course-side-stepping-v4"
            assert protocol["braking_profile"] == "legacy_release_reference_edge_v1"
            changed = [name for name, expected in protocol["source_sha256"].items()
                       if sha(repo / name) != expected]
            assert not changed, changed
            assert sha(before / "model.mjb") == sha(after / "model.mjb")
            with (after / "telemetry.csv").open() as stream:
                telemetry = list(csv.DictReader(stream))
            events = [{key: row[key] for key in ("tick", "segment_before_time_s",
                       "brake_reference_previous_error_m", "brake_reference_new_error_m",
                       "brake_reference_event_count")} for row in telemetry
                      if row["brake_reference_reanchored"] == "True"]
            with np.load(before / "states.npz") as x, np.load(after / "states.npz") as y:
                if kind == "forward":
                    cut = round(old["release_time_s"]/.01)
                    checks = {key: bool(np.array_equal(x[key][:cut+1], y[key][:cut+1]))
                              for key in ("qpos", "qvel", "segment_time_s")}
                    checks["pre_release_torque"] = bool(np.array_equal(
                        x["applied_torque_nm"][:cut], y["applied_torque_nm"][:cut]))
                    assert len(events) == 1 and int(events[0]["tick"]) == cut+1
                    assert all(int(row["brake_reference_event_count"]) == int(i >= cut)
                               for i, row in enumerate(telemetry))
                else:
                    checks = {key: bool(np.array_equal(x[key], y[key])) for key in x.files}
                    assert events == []
                    assert all(row["brake_reference_event_count"] == "0" for row in telemetry)
                assert all(checks.values()), (name, checks)
                assert np.isfinite(y["qpos"]).all() and np.isfinite(y["qvel"]).all()
            report[kind].append({"case": name, "unchanged_physics_checks": checks,
                                 "compiled_model_identical": True, "current_source_files":
                                 len(protocol["source_sha256"]), "source_unchanged": True,
                                 "braking_events": events, "old_full_summary": old,
                                 "new_full_summary": new})
    formal = json.loads((base.parents[1] / "rl_improvement_20260914/frozen_source_before.json").read_text())
    changed = [name for name, expected in formal["sha256"].items() if sha(repo / name) != expected]
    assert not changed, changed
    report.update(formal_source_files_unchanged=len(formal["sha256"]), all_passed=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.input.resolve(), args.repo.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"all_passed": True, "forward_pairs": len(result["forward"]),
                      "side_pairs": len(result["side"]), "frozen_files": result["formal_source_files_unchanged"]}))


if __name__ == "__main__":
    main()
