"""Verify compact publication and extracted full records without an engine import.

This checks file identity and saved record links, not physical contact geometry.
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
from typing import Any

BLOCKED = {"mujoco", "torch", "stable_baselines3", "gymnasium"}


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.partition(".")[0] in BLOCKED:
            raise RuntimeError(f"forbidden engine/policy import: {fullname}")


if any(name.partition(".")[0] in BLOCKED for name in sys.modules):
    raise RuntimeError("engine or policy imported before offline verification")
sys.meta_path.insert(0, NoEngine())

import numpy as np  # import barrier is deliberately installed first

PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
BASE_CASES = tuple(
    f"speed{round(speed * 1000):03d}_{terrain}_{actor}"
    for speed in (.2, .25) for terrain in ("plane", "box")
    for actor in ("zero", "final_policy")
)
DRIVE_CASES = tuple(name for name in BASE_CASES if name.endswith("final_policy"))


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object expected: {path}")
    return value


def digest(path: Path) -> dict[str, Any]:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return {"sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def safe_file(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if (not part.parts or part.is_absolute() or ".." in part.parts
            or part.as_posix() != relative):
        raise ValueError(f"unsafe manifest entry: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"missing, symlinked or escaping file: {path}")
    return path


def check_manifest(root: Path, manifest: dict[str, Any], *, excluded: set[str]) -> int:
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
        raise ValueError(f"file set differs: missing={sorted(set(expected)-actual)}, "
                         f"extra={sorted(actual-set(expected)-excluded)}")
    for relative, identity in expected.items():
        if digest(safe_file(root, relative)) != identity:
            raise ValueError(f"file hash/bytes mismatch: {relative}")
    return len(expected)


def same(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def check_case(package: Path, archive: Path, name: str, source: dict[str, Any],
               *, drive: bool, baseline_initial: dict[str, Any] | None,
               paired: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    prefix = "evidence/drive" if drive else "evidence/rl"
    folder = archive / source["source_dir"]
    if not folder.resolve().is_relative_to(archive) or not folder.is_dir():
        raise ValueError(f"case source directory invalid: {name}")
    for filename, identity in source["raw_file_hashes"].items():
        if digest(folder / filename) != identity:
            raise ValueError(f"case source hash differs: {name}/{filename}")
    score = read(folder / "score.json")
    saved_score = read(package / prefix / "completed_cases" / name / "score.json")
    if score != saved_score or score["record_valid"] is not True:
        raise ValueError(f"case score differs or record is invalid: {name}")
    receipt = read(folder / "receipt.json")
    trace, endpoints, native = (rows(folder / filename) for filename in
                                ("trace.jsonl.gz", "endpoints.jsonl.gz", "native.jsonl.gz"))
    count = receipt["completed_control_intervals"]
    if (count != source["completed_controls"] or len(trace) != count
            or len(endpoints) != count + 1 or len(native) != 5 * count
            or score["metrics"]["completed_control_intervals"] != count):
        raise ValueError(f"case control/endpoint/native lengths differ: {name}")
    if drive:
        entries = rows(folder / "native_entry_events.jsonl.gz")
        if len(entries) != len(native) or any(row["attempt"] != index
                                              for index, row in enumerate(entries)):
            raise ValueError(f"drive native entry count/order differs: {name}")
    with np.load(folder / "states.npz", allow_pickle=False) as saved:
        initial = {field: saved[field][0].copy() for field in PAIR_FIELDS}
        if any(saved[field].shape[0] != count + 1 for field in ("qpos", "qvel")):
            raise ValueError(f"case state archive length differs: {name}")
        for index, row in enumerate(trace):
            if row["tick"] != index or row["endpoint_tick"] != index + 1:
                raise ValueError(f"case trace tick differs: {name}/{index}")
            controller = row["controller_result"]
            if (not same(controller["clipped_action"], row["applied_action"])
                    or not same(controller["torque_nm"], row["torque_nm"])
                    or not same(row["stop_record"]["safe_torque_nm"], row["torque_nm"])):
                raise ValueError(f"case controller/action/torque link differs: {name}/{index}")
        for index, row in enumerate(native):
            tick = index // 5
            if (row["index"] != index or row["returned"] is not True
                    or row["error"] is not None
                    or not same(row["ctrl_nm"], trace[tick]["torque_nm"])):
                raise ValueError(f"native return/torque link differs: {name}/{index}")
            if index % 5 == 0 and (not same(row["qpos_before"], saved["qpos"][tick])
                                    or not same(row["qvel_before"], saved["qvel"][tick])):
                raise ValueError(f"native pre-state differs: {name}/{index}")
            if index % 5 == 4 and (not same(row["qpos_returned"], saved["qpos"][tick + 1])
                                    or not same(row["qvel_returned"], saved["qvel"][tick + 1])):
                raise ValueError(f"native post-state differs: {name}/{index}")
            if index and (not same(native[index - 1]["qpos_returned"], row["qpos_before"])
                          or not same(native[index - 1]["qvel_returned"], row["qvel_before"])):
                raise ValueError(f"native continuity differs: {name}/{index}")
    speed_label, terrain, actor = name.split("_", 2)
    actor = actor.removeprefix("plane_").removeprefix("box_")
    speed = int(speed_label.removeprefix("speed")) / 1000
    pair_key = (speed_label, terrain)
    if drive:
        if baseline_initial is None or any(not same(initial[key], baseline_initial[key])
                                           for key in PAIR_FIELDS):
            raise ValueError(f"drive case initial state differs from old policy: {name}")
    elif actor == "zero":
        paired[pair_key] = initial
    elif pair_key not in paired or any(not same(initial[key], paired[pair_key][key])
                                       for key in PAIR_FIELDS):
        raise ValueError(f"zero/policy initial pair differs: {name}")
    if count >= 875:
        samples = np.asarray([row["body_vx_mps"] for row in endpoints[275:876]], dtype=float)
        if samples.shape != (601,) or not np.isfinite(samples).all():
            raise ValueError(f"speed window invalid: {name}")
        mean = float(samples.mean())
        rms = float(np.sqrt(np.mean((samples - speed) ** 2)))
        if (not math.isclose(mean, score["metrics"]["speed_window_mean_body_vx_mps"],
                             rel_tol=0, abs_tol=1e-12)
                or not math.isclose(rms, score["metrics"]["speed_window_rms_command_error_mps"],
                                    rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"speed window differs from saved score: {name}")
    compact = json.loads((package / prefix / "compact_trajectories" / f"{name}.json")
                         .read_text(encoding="utf-8"))
    if compact[0]["tick"] != 0 or compact[-1]["tick"] > count:
        raise ValueError(f"compact trajectory bounds differ: {name}")
    for row in compact:
        saved = endpoints[row["tick"]]
        if any(row[key] != saved[key] for key in
               ("time_s", "base_position_m", "rpy_rad", "body_vx_mps", "com_vz_mps")):
            raise ValueError(f"compact trajectory source differs: {name}/{row['tick']}")
    return {"controls": count, "native_returns": len(native),
            "task_passed": score["task_passed"], "speed_window_checked": count >= 875}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    archive = args.archive_root.resolve(strict=True)
    publication = read(package / "publication_manifest.json")
    full = read(archive / "archive_manifest.json")
    if (publication.get("schema") != "rolling-isolated-engine-drive-compact-publication-v3"
            or full.get("schema") != "closed-rolling-full-evidence-archive-v1"
            or digest(archive / "archive_manifest.json")
            != publication["complete_raw_archive_manifest"]):
        raise ValueError("publication/raw archive schemas or identities differ")
    package_files = check_manifest(package, publication, excluded={"publication_manifest.json"})
    archive_files = check_manifest(archive, full, excluded={"archive_manifest.json"})
    if digest(package / "evidence/full_archive_manifest.json") != digest(archive / "archive_manifest.json"):
        raise ValueError("compact and full archive manifests differ")
    for label, path in (("run_02", "evidence/rl/run_02_interrupted_archive_manifest_01.json"),
                        ("run_03", "evidence/rl/run_03_archive_manifest_01.json"),
                        ("drive_failed_04", "evidence/drive/failed_04/run_01_failed_archive_manifest_01.json"),
                        ("drive_run_01", "evidence/drive/run_01_archive_manifest_01.json")):
        manifest = read(package / path)["files"]
        actual = {name.removeprefix(label + "/"): identity for name, identity in full["files"].items()
                  if name.startswith(label + "/")}
        if actual != manifest:
            raise ValueError(f"run manifest differs from full archive: {label}")
    old_map = read(package / "evidence/rl/case_source_map.json")["cases"]
    drive_map = read(package / "evidence/drive/case_source_map.json")["cases"]
    if set(old_map) != set(BASE_CASES) or set(drive_map) != set(DRIVE_CASES):
        raise ValueError("12 completed case names differ")
    failed = read(package / "evidence/drive/failed_04/run_01_boundary_closure_01.json")
    drive_receipt = read(package / "evidence/drive/run_01/drive_evaluation_receipt.json")
    drive_readback = read(package / "evidence/drive/root_run_01_readback_01.json")
    if (failed.get("reservation_closed") is not True
            or failed.get("execution_valid") is not False
            or failed.get("actual_control_returns") != 0
            or failed.get("actual_normal_native_returns") != 0
            or drive_receipt.get("execution_valid") is not True
            or drive_readback.get("passed") is not True
            or drive_receipt.get("candidate_qualified")
            != drive_readback.get("candidate_qualified")
            or drive_receipt.get("candidate_qualified")
            != publication.get("corrected_05_candidate_qualified")):
        raise ValueError("failed04/corrected05 status differs across publication evidence")
    paired: dict[tuple[str, str], dict[str, Any]] = {}
    results = {}
    for name in BASE_CASES:
        results[name] = check_case(package, archive, name, old_map[name], drive=False,
                                   baseline_initial=None, paired=paired)
    for name in DRIVE_CASES:
        old_path = archive / old_map[name]["source_dir"] / "states.npz"
        with np.load(old_path, allow_pickle=False) as baseline:
            initial = {field: baseline[field][0].copy() for field in PAIR_FIELDS}
        results["drive/" + name] = check_case(package, archive, name, drive_map[name],
                                               drive=True, baseline_initial=initial,
                                               paired=paired)
    checkpoint = package / "checkpoint"
    sidecar = read(checkpoint / "model.metadata.json")
    training = read(package / "evidence/rl/run_02/training/training_receipt.json")
    updates_path = package / "evidence/rl/run_02/training/completed_updates.jsonl"
    updates = [json.loads(line) for line in updates_path.read_text(encoding="utf-8").splitlines()]
    if (training.get("physical_control_transitions") != 65536
            or training.get("completed_train_calls") != 512
            or training.get("optimization_epochs") != 2048
            or len(updates) != 512):
        raise ValueError("training control/update totals differ")
    for index, row in enumerate(updates, start=1):
        if (row["train_call"], row["completed_control"], row["optimization_epochs"]) != (
            index, 128 * index, 4 * index
        ):
            raise ValueError(f"training update sequence differs at {index}")
    if (updates[-1]["policy_sha256"] != training["final_parameter_sha256"]
            or updates[-1]["actor_sha256"] != training["final_actor_sha256"]):
        raise ValueError("final saved update differs from training receipt")
    if (digest(checkpoint / "model.zip")["sha256"] != sidecar["model_sha256"]
            or digest(checkpoint / "model.zip")["sha256"] != training["model_sha256"]
            or digest(checkpoint / "model.metadata.json")["sha256"]
            != training["checkpoint_metadata_sha256"]
            or digest(checkpoint / "reload_probe.npz")["sha256"]
            != sidecar["reload_probe_sha256"]):
        raise ValueError("checkpoint model/metadata/probe identity differs")
    with np.load(checkpoint / "reload_probe.npz", allow_pickle=False) as probe:
        if (probe["observations"].shape != (2, 85)
                or probe["actions"].shape != (2, 1)
                or not np.isfinite(probe["observations"]).all()
                or not np.isfinite(probe["actions"]).all()):
            raise ValueError("saved reload probe is malformed")
    print(json.dumps({"file_identity_and_saved_record_links_passed": True,
                      "package_files": package_files, "archive_files": archive_files,
                      "case_records": results, "physical_geometry_rescored": False},
                     sort_keys=True))


if __name__ == "__main__":
    main()
