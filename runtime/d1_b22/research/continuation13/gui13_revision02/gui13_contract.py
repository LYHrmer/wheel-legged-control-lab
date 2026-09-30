"""Pure, fixed GUI13 development profile and evidence preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

W = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCHEMA = "d1-world-upright-flat0p6-gui13-v1"
PLAN_SCHEMA = SCHEMA + "-plan"
SOURCE_SEED = 88701
SEED = 88813
HORIZON = 600
NORMAL_NATIVE = 3000
MODEL_SHA = "6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70"
PAIR = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
READBACK = W / "rl11/partial_readback_20260929_05.json"
PARTIAL_CLOSED = W / "rl11/partial_closed_manifest_20260929_05.json"
TRAIN = W / "rl11/training_run_01"
TRAIN_HOST = TRAIN / "launcher_receipt.json"
MANIFEST = TRAIN / "final_checkpoint_manifest.json"
CHECKPOINT = TRAIN / "final_checkpoint"


def digest(path: Path) -> dict:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return {"sha256": hasher.hexdigest(), "bytes": size}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def command_speed(tick: int) -> float:
    if type(tick) is not int or not 0 <= tick < HORIZON:
        raise ValueError("GUI13 command tick outside the 600-control profile")
    return 0.6 if 175 <= tick < 425 else 0.0


def verify_evidence() -> dict:
    """Bind the real final bytes and the one scored actor/task; no all-12 gate."""
    readback = read_json(READBACK)
    if (readback.get("schema") != "d1-world-upright-short-rl16-partial-readback-11-v1"
            or readback.get("run") != str(TRAIN)
            or readback.get("complete_case_count", 0) < 2
            or readback.get("saved_records_rescored") is not True
            or readback.get("training_complete") is not True
            or readback.get("budget_reserved_and_closed_without_retry") is not True):
        raise RuntimeError("GUI13 partial05 identity/complete-case count differs")
    host = read_json(TRAIN_HOST)
    if (host.get("source_hash_mismatches") != []
            or host.get("worker_exited") is not True
            or host.get("owned_worker_cleanup", {}).get("no_orphans") is not True):
        raise RuntimeError("GUI13 11-S source/worker closure differs")
    closed = read_json(PARTIAL_CLOSED)
    if (closed.get("schema") != "d1-world-upright-short-rl16-closed-manifest-11-v1"
            or closed.get("run") != str(TRAIN)
            or not isinstance(closed.get("files"), dict)
            or closed.get("file_count") != len(closed["files"])):
        raise RuntimeError("GUI13 11-S partial closed manifest differs")
    rows = readback.get("individual_complete_case_scores_not_matched_pair_aggregate")
    chosen = [row for row in rows if row.get("case_id") == "flat_0p6"
              and row.get("actor") == "final_policy"] if isinstance(rows, list) else []
    if (len(chosen) != 1 or chosen[0].get("task_passed") is not True
            or chosen[0].get("terrain") != "flat"
            or chosen[0].get("seed") != SOURCE_SEED
            or chosen[0].get("target_speed_mps") != 0.6
            or chosen[0].get("record", {}).get("record_valid") is not True):
        raise RuntimeError("GUI13 flat0p6 final_policy task pass is absent")
    full = readback.get("complete_full_native_heldout_cases_verified")
    matches = [row for row in full if row.get("case_id") == "flat_0p6"
               and row.get("actor") == "final_policy"] if isinstance(full, list) else []
    if (len(matches) != 1 or matches[0].get("physical_record_integrity_passed") is not True
            or matches[0].get("full_horizon_controls") is not True
            or matches[0].get("legitimate_task_failure") is not False
            or matches[0].get("seed") != SOURCE_SEED
            or matches[0].get("completed_controls") != 1600
            or matches[0].get("native_returned") != 8000):
        raise RuntimeError("GUI13 cited flat0p6 full native record is absent")
    manifest = read_json(MANIFEST)
    if (manifest.get("loader_schema") != "d1-course-rl16-final-checkpoint-v2-budgeted"
            or manifest.get("files", {}).get("final_model.zip", {}).get("sha256") != MODEL_SHA
            or Path(manifest.get("folder", "")) != CHECKPOINT):
        raise RuntimeError("GUI13 final checkpoint manifest differs")
    for name, expected in manifest["files"].items():
        if digest(CHECKPOINT / name) != expected:
            raise RuntimeError("GUI13 final checkpoint payload differs: " + name)
    if digest(CHECKPOINT / "final_metadata.json")["sha256"] != manifest["metadata_sha256"]:
        raise RuntimeError("GUI13 final metadata differs")
    return {"readback": str(READBACK), "readback_identity": digest(READBACK),
            "partial_closed_manifest": str(PARTIAL_CLOSED),
            "partial_closed_manifest_identity": digest(PARTIAL_CLOSED),
            "training_host": str(TRAIN_HOST), "training_host_identity": digest(TRAIN_HOST),
            "manifest": str(MANIFEST), "manifest_identity": digest(MANIFEST),
            "checkpoint_model_sha256": MODEL_SHA, "source_case_horizon": 1600,
            "source_case_seed": SOURCE_SEED, "development_seed": SEED,
            "development_profile_horizon": HORIZON,
            "original_1600_task_replayed": False,
            "flat0p6_final_policy_task_passed": True}


def check_plan(plan: dict) -> None:
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("decision") != "GO"
            or set(plan.get("arms", {})) != {"headless", "gui"}
            or plan.get("evidence") != verify_evidence()
            or plan.get("retry_permitted") is not False):
        raise RuntimeError("GUI13 plan/evidence GO differs")
    for arm in ("headless", "gui"):
        item = plan["arms"][arm]
        if (item.get("arm") != arm or item.get("seed") != SEED
                or item.get("control_limit") != HORIZON
                or item.get("normal_native_limit") != NORMAL_NATIVE
                or item.get("compiler_native_limit") != 2
                or item.get("wallclock_hard_s") != 120
                or item.get("wallclock_soft_s") != 95
                or item.get("preflight_wall_s") != 240
                or item.get("postcheck_wall_s") != 180
                or item.get("outer_host_wall_s") != 600
                or item.get("cleanup_grace_s") != 5
                or item.get("archive_grace_s") != 20
                or item.get("actor") != "final_policy"
                or item.get("terrain") != "flat"
                or item.get("retry_permitted") is not False
                or not isinstance(item.get("output_directory"), str)
                or not isinstance(item.get("reservation_path"), str)):
            raise RuntimeError("GUI13 arm exceeds the fixed 600/3000+2/120s profile")
        output = Path(item["output_directory"])
        reservation = Path(item["reservation_path"])
        if (not output.is_absolute() or output.parent != HERE or
                not reservation.is_absolute() or reservation.parent != HERE):
            raise RuntimeError("GUI13 output/reservation must be local and exclusive")


__all__ = ["CHECKPOINT", "HERE", "HORIZON", "MANIFEST", "NORMAL_NATIVE", "PAIR",
           "PARTIAL_CLOSED", "PLAN_SCHEMA", "READBACK", "SCHEMA", "SEED", "SOURCE_SEED",
           "TRAIN", "TRAIN_HOST", "W", "check_plan", "command_speed", "digest",
           "read_json", "verify_evidence"]
