"""Portable identity and saved-record verification for 06. No physics rescore.

Usage: python3 offline_verify_06.py --package COMPACT --archive-root EXTRACTED_RAW
Requires only Python stdlib and NumPy; no engine, policy, or older raw release.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.abc
import json
import math
import sys
from pathlib import Path, PurePosixPath

BLOCKED = {"mujoco", "torch", "stable_baselines3", "gymnasium"}


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in BLOCKED:
            raise RuntimeError("forbidden offline import: " + fullname)


if any(name.partition(".")[0] in BLOCKED for name in sys.modules):
    raise RuntimeError("engine or policy imported before offline verifier")
sys.meta_path.insert(0, NoEngine())
import numpy as np

CASES = ("speed200_plane_final_policy", "speed200_box_final_policy",
         "speed250_plane_final_policy", "speed250_box_final_policy")
PAIR = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


def identity(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object expected: {path}")
    return value


def rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def safe(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not part.parts or part.is_absolute() or ".." in part.parts or part.as_posix() != relative:
        raise ValueError(f"unsafe file name: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"missing, symlinked or escaping file: {path}")
    return path


def verify_tree(root: Path, manifest: dict, *, excluded: set[str]) -> int:
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("manifest has no file table")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in evidence tree: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected) | excluded:
        raise ValueError(f"file set differs: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected)-excluded)}")
    for name, expected_identity in expected.items():
        if identity(safe(root, name)) != expected_identity:
            raise ValueError(f"file identity differs: {name}")
    return len(expected)


def same(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def check_case(package: Path, raw: Path, name: str, source: dict) -> dict:
    folder = raw / source["source_dir"]
    if not folder.is_dir() or not folder.resolve().is_relative_to(raw):
        raise ValueError(f"bad source folder: {name}")
    for filename, expected in source["raw_file_hashes"].items():
        if identity(safe(folder, filename)) != expected:
            raise ValueError(f"case source identity differs: {name}/{filename}")
    score = read(folder / "score.json")
    saved_score = read(package / "cases" / name / "score.json")
    receipt = read(folder / "receipt.json")
    stage = read(folder / "body_stage_validation.json")
    trace, endpoints, native, entries = (rows(folder / file) for file in
        ("trace.jsonl.gz", "endpoints.jsonl.gz", "native.jsonl.gz", "native_entry_events.jsonl.gz"))
    count = source["completed_controls"]
    if (score != saved_score or score.get("record_valid") is not True
            or stage["stage"]["record_valid"] is not True
            or stage["precontrol_archive"]["record_valid"] is not True
            or receipt["completed_control_intervals"] != count
            or count != 1200 or len(trace) != count or len(endpoints) != count + 1
            or len(native) != 5 * count or len(entries) != len(native)
            or score["metrics"]["completed_control_intervals"] != count
            or score["task_passed"] is not source["task_passed"]):
        raise ValueError(f"case score/length/stage differs: {name}")
    with np.load(folder / "states.npz", allow_pickle=False) as saved, np.load(
            package / "paired_initial" / f"{name}.npz", allow_pickle=False) as old:
        if any(not same(saved[key][0], old[key]) for key in PAIR):
            raise ValueError(f"case initial state differs from published old-policy/05 pair: {name}")
        if any(saved[key].shape[0] != count + 1 for key in ("qpos", "qvel")):
            raise ValueError(f"case state lengths differ: {name}")
        for tick, row in enumerate(trace):
            if (row["tick"] != tick or row["endpoint_tick"] != tick + 1
                    or not same(row["controller_result"]["clipped_action"], row["applied_action"])
                    or not same(row["controller_result"]["torque_nm"], row["torque_nm"])
                    or not same(row["stop_record"]["safe_torque_nm"], row["torque_nm"])):
                raise ValueError(f"trace action/torque link differs: {name}/{tick}")
        for index, row in enumerate(native):
            tick = index // 5
            if (row["index"] != index or row["returned"] is not True or row["error"] is not None
                    or entries[index]["attempt"] != index
                    or not same(row["ctrl_nm"], trace[tick]["torque_nm"])):
                raise ValueError(f"native control link differs: {name}/{index}")
            if index % 5 == 0 and (not same(row["qpos_before"], saved["qpos"][tick])
                                    or not same(row["qvel_before"], saved["qvel"][tick])):
                raise ValueError(f"native initial substep differs: {name}/{index}")
            if index % 5 == 4 and (not same(row["qpos_returned"], saved["qpos"][tick + 1])
                                    or not same(row["qvel_returned"], saved["qvel"][tick + 1])):
                raise ValueError(f"native final substep differs: {name}/{index}")
            if index and (not same(native[index - 1]["qpos_returned"], row["qpos_before"])
                          or not same(native[index - 1]["qvel_returned"], row["qvel_before"])):
                raise ValueError(f"native continuity differs: {name}/{index}")
    if [row["tick"] for row in endpoints] != list(range(count + 1)):
        raise ValueError(f"endpoint ticks differ: {name}")
    speed = int(name[5:8]) / 1000
    measured = np.asarray([row["body_vx_mps"] for row in endpoints[275:876]], dtype=float)
    if measured.shape != (601,) or not np.isfinite(measured).all():
        raise ValueError(f"speed window malformed: {name}")
    mean = float(measured.mean())
    rms = float(np.sqrt(np.mean((measured - speed) ** 2)))
    metrics = score["metrics"]
    if (not math.isclose(mean, metrics["speed_window_mean_body_vx_mps"], rel_tol=0, abs_tol=1e-12)
            or not math.isclose(rms, metrics["speed_window_rms_command_error_mps"], rel_tol=0, abs_tol=1e-12)
            or score["speed_gate_passed"] is not (mean >= .9 * speed and rms <= .05)):
        raise ValueError(f"saved speed score differs: {name}")
    compact = json.loads((package / "trajectories" / f"{name}.json").read_text(encoding="utf-8"))
    if compact[0]["tick"] != 0 or compact[-1]["tick"] != 1200:
        raise ValueError(f"compact trajectory incomplete: {name}")
    for row in compact:
        endpoint = endpoints[row["tick"]]
        if any(row[key] != endpoint[key] for key in
               ("time_s", "base_position_m", "rpy_rad", "body_vx_mps", "com_vz_mps")):
            raise ValueError(f"compact trajectory link differs: {name}/{row['tick']}")
    return {"controls": count, "native_returns": len(native), "task_passed": score["task_passed"],
            "mean_body_vx_mps": mean, "speed_rms_mps": rms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    raw = args.archive_root.resolve(strict=True)
    publication = read(package / "publication_manifest.json")
    archive = read(raw / "archive_manifest.json")
    if (publication.get("schema") != "body-common-p-06-compact-publication-v1"
            or archive.get("schema") != "body-common-p-06-raw-archive-v1"
            or archive.get("closed_run_files") != 77
            or identity(raw / "archive_manifest.json") != publication["full_archive_manifest"]):
        raise ValueError("compact/raw publication schema or manifest identity differs")
    package_files = verify_tree(package, publication, excluded={"publication_manifest.json"})
    archive_files = verify_tree(raw, archive, excluded={"archive_manifest.json"})
    if identity(package / "evidence/full_archive_manifest.json") != identity(raw / "archive_manifest.json"):
        raise ValueError("compact and raw manifests differ")
    closed = read(package / "evidence/run_01_archive_manifest_01.json")["files"]
    if len(closed) != 77 or {name.removeprefix("run_01/"): row for name, row in archive["files"].items()
                            if name.startswith("run_01/")} != closed:
        raise ValueError("closed 77-file run differs from raw asset")
    receipt = read(package / "evidence/body_evaluation_receipt.json")
    readback = read(package / "evidence/root_run_01_readback_01.json")
    closure = read(package / "evidence/run_01_boundary_closure_01.json")
    source = read(package / "evidence/case_source_map.json")["cases"]
    if (tuple(row["case"] for row in receipt["cases"]) != CASES or set(source) != set(CASES)
            or receipt.get("execution_valid") is not True or readback.get("passed") is not True
            or closure.get("reservation_closed") is not True or closure.get("execution_valid") is not True):
        raise ValueError("closed four-case execution/readback differs")
    results = {name: check_case(package, raw, name, source[name]) for name in CASES}
    for row in receipt["cases"]:
        name = row["case"]
        saved_score = read(package / "cases" / name / "score.json")
        if (row["score"] != saved_score or readback["cases"][name]["new_score"] != saved_score
                or readback["cases"][name]["controls"] != results[name]["controls"]
                or readback["cases"][name]["task_passed"] is not results[name]["task_passed"]):
            raise ValueError(f"receipt/readback/case score differs: {name}")
    if (receipt["actual_completed_controls"] != 4800
            or receipt["actual_normal_native_returns"] != 24000
            or readback["actual_controls"] != 4800
            or readback["actual_normal_native"] != 24000
            or readback["actual_compiler_native"] != 5):
        raise ValueError("closed 06 native/control totals differ")
    deltas = {terrain: results[f"speed250_{terrain}_final_policy"]["mean_body_vx_mps"]
              - results[f"speed200_{terrain}_final_policy"]["mean_body_vx_mps"]
              for terrain in ("plane", "box")}
    qualified = all(row["task_passed"] for row in results.values()) and all(
        delta >= .03 for delta in deltas.values())
    if (receipt["candidate_qualified"] is not qualified
            or readback["candidate_qualified"] is not qualified
            or closure["candidate_qualified"] is not qualified
            or publication["candidate_qualified"] is not qualified
            or any(not math.isclose(deltas[k], receipt["policy_speed_mean_delta_mps"][k],
                                     rel_tol=0, abs_tol=1e-12) for k in deltas)):
        raise ValueError("candidate qualification or speed increment differs")
    print(json.dumps({"file_identity_and_saved_record_links_passed": True,
                      "package_files": package_files, "archive_files": archive_files,
                      "cases": results, "policy_speed_mean_delta_mps": deltas,
                      "candidate_qualified": qualified, "physical_geometry_rescored": False},
                     sort_keys=True))


if __name__ == "__main__":
    main()
