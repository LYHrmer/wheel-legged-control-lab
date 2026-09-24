"""Build a compact, source-bound package after both physical runs close.

The original process is interrupted. The five-case continuation is a separate
reservation. This offline script never loads MuJoCo, the policy, or a model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

OLD_CASES = ("speed200_plane_zero", "speed200_plane_final_policy", "speed200_box_zero")
NEW_CASES = ("speed200_box_final_policy", "speed250_plane_zero",
             "speed250_plane_final_policy", "speed250_box_zero",
             "speed250_box_final_policy")
DSO_SHA256 = "3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c"
MODEL_SHA256 = "4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e"


def digest(path: Path) -> dict[str, Any]:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return {"sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def copy_one(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, destination.open("xb") as out:
        shutil.copyfileobj(inp, out, 1024 * 1024)
    if digest(destination) != digest(source):
        raise RuntimeError(f"copied artifact differs: {source}")


def verify_freeze(path: Path) -> dict[str, Any]:
    freeze = load_json(path)
    files = freeze.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"freeze lacks its input map: {path}")
    for name, expected in files.items():
        source = Path(name)
        if not source.is_absolute() or not source.is_file() or digest(source) != expected:
            raise RuntimeError(f"frozen input changed after execution: {name}")
    return freeze


def verify_archive(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"raw archive manifest is empty: {manifest_path}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files):
        raise RuntimeError(f"raw archive file set differs from {manifest_path}")
    for relative, expected in files.items():
        source = root / relative
        if (not source.resolve().is_relative_to(root.resolve())
                or not source.is_file() or digest(source) != expected):
            raise RuntimeError(f"raw archive file differs: {source}")
    return manifest


def compact_endpoint_rows(path: Path) -> list[dict[str, Any]]:
    special = {0, 175, 275, 875, 975, 1101, 1200}
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            tick = row["tick"]
            if tick % 10 == 0 or tick in special:
                rows.append({
                    "tick": tick, "time_s": row["time_s"],
                    "base_position_m": row["base_position_m"],
                    "rpy_rad": row["rpy_rad"],
                    "body_vx_mps": row["body_vx_mps"],
                    "com_vz_mps": row["com_vz_mps"],
                })
    return rows


def case_sources(original: Path, continuation: Path,
                 stitched: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = stitched.get("cases")
    if not isinstance(rows, list) or tuple(row["case"] for row in rows) != (*OLD_CASES, *NEW_CASES):
        raise ValueError("stitched result lacks the prescribed old-three/new-five order")
    sources: dict[str, Any] = {}
    compact: dict[str, Any] = {}
    for row in rows:
        name, process = row["case"], row["source_process"]
        if name in OLD_CASES and process == "interrupted_run_02_complete_record":
            run = original
            label = "run_02"
        elif name in NEW_CASES and process == "new_run_03_continuation":
            run = continuation
            label = "run_03"
        else:
            raise ValueError(f"case source process differs: {name}, {process}")
        folder = run / "evaluation" / name
        score_path = folder / "score.json"
        if (load_json(score_path) != row["score"]
                or row["score"]["record_valid"] is not True):
            raise ValueError(f"saved case score differs or is invalid: {name}")
        receipt = load_json(folder / "receipt.json")
        completed = receipt["completed_control_intervals"]
        if completed != row["score"]["metrics"]["completed_control_intervals"]:
            raise ValueError(f"case control count differs: {name}")
        raw_files = {
            filename: digest(folder / filename)
            for filename in ("endpoints.jsonl.gz", "trace.jsonl.gz", "native.jsonl.gz",
                             "states.npz", "score.json", "receipt.json")
        }
        sources[name] = {
            "source_process": process, "archive_run": label,
            "source_dir": f"{label}/evaluation/{name}",
            "source_dir_is_archive_relative": True,
            "relative_case_dir": f"{label}/evaluation/{name}",
            "completed_controls": completed, "task_passed": row["score"]["task_passed"],
            "raw_file_hashes": raw_files,
        }
        compact[name] = compact_endpoint_rows(folder / "endpoints.jsonl.gz")
        if not compact[name] or compact[name][0]["tick"] != 0:
            raise ValueError(f"case trajectory is empty: {name}")
    return sources, compact


def verify_figures(figures: Path, original: Path, continuation: Path,
                   stitched_path: Path, sources: dict[str, Any]) -> dict[str, Any]:
    receipt = load_json(figures / "figure_receipt.json")
    if (receipt.get("schema") != "saved-rolling-two-process-trajectory-figures-v2"
            or receipt.get("original_run") != str(original)
            or receipt.get("continuation_run") != str(continuation)
            or receipt.get("eight_case_stitch") != digest(stitched_path)
            or receipt.get("no_engine_or_model_calls") is not True
            or set(receipt.get("sources", {})) != set(sources)):
        raise ValueError("figure receipt refers to another study or omits a completed case")
    for name, source in sources.items():
        recorded = receipt["sources"][name]
        folder = (original if source["archive_run"] == "run_02" else continuation)
        folder = folder / "evaluation" / name
        if (recorded.get("source_process") != source["source_process"]
                or recorded.get("source_dir") != str(folder.resolve())
                or recorded.get("completed_control") != source["completed_controls"]):
            raise ValueError(f"figure source mapping differs: {name}")
        for filename in ("endpoints.jsonl.gz", "trace.jsonl.gz", "score.json"):
            if recorded["files"].get(filename) != source["raw_file_hashes"][filename]:
                raise ValueError(f"figure source hash differs: {name}/{filename}")
    for name in ("speed_tracking.png", "box_pitch.png", "shared_leg_residual.png"):
        if receipt["figures"].get(name) != digest(figures / name):
            raise ValueError(f"rendered figure hash differs: {name}")
    return receipt


def static_package_map(repo: Path, engine: Path, epa: Path, rl: Path,
                       original: Path, continuation: Path, independent: Path,
                       figures: Path, *, frozen_dso: bool,
                       additional: list[tuple[str, Path]]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    names = (
        "d1_rolling_residual_task.py", "d1_rolling_residual_env.py",
        "d1_rolling_residual_checkpoint.py", "d1_rolling_residual_scoring.py",
        "d1_rolling_engine_runtime.py", "d1_rolling_native_monitor.py",
        "train_d1_rolling_residual_ppo.py", "evaluate_d1_rolling_residual.py",
        "run_d1_rolling_residual_study.py", "bootstrap_d1_rolling_residual_study.py",
    )
    mapping.update({f"rl_source/scripts/{name}": repo / "scripts" / name for name in names})
    mapping.update({
        "rl_source/tests/test_d1_rolling_residual_task.py": repo / "tests/test_d1_rolling_residual_task.py",
        "rl_source/tests/test_d1_rolling_residual_records.py": repo / "tests/test_d1_rolling_residual_records.py",
        "rl_source/tests/fixtures/d1_rolling_historical_zero_prefix.json":
            repo / "tests/fixtures/d1_rolling_historical_zero_prefix.json",
        "rl_source/continuation/run_evaluation_continuation_03.py":
            rl / "run_evaluation_continuation_03.py",
        "rl_source/continuation/freeze_continuation_03.py": rl / "freeze_continuation_03.py",
        "rl_source/continuation/launch_continuation_once_03.py":
            rl / "launch_continuation_once_03.py",
    })
    mapping.update({
        f"engine_patch/src/{name}": engine / name
        for name in ("engine_local_gjk.c", "engine_local_support.c",
                     "engine_interpose.c", "engine_interpose.h")
    })
    upstream = ("engine_collision_gjk.c", "engine_collision_gjk.h",
                "engine_collision_convex.c", "engine_collision_convex.h",
                "engine_inline.h", "engine_util_blas.h", "engine_util_errmem.h")
    mapping.update({f"engine_patch/upstream/engine/{name}": epa / "sources/engine" / name
                    for name in upstream})
    mapping.update({
        "engine_patch/BUILD.md": rl / "publication_staging/engine_patch_build_01.md",
        "engine_patch/LICENSE": repo / "results/d1_driving_stability_development/contact_geometry_forensics_20260923_01/sources/MUJOCO_LICENSE",
        "engine_patch/LICENSES_THIRD_PARTY.md": repo / "results/d1_driving_stability_development/contact_geometry_forensics_20260923_01/sources/MUJOCO_LICENSES_THIRD_PARTY.md",
        "engine_patch/provenance/local_kernel_vs_original.diff":
            engine / "build/local_kernel_vs_original.diff",
        "engine_patch/provenance/build_dependencies.json":
            engine / "build/build_dependencies.json",
        "engine_patch/provenance/official_source_manifest.json": epa / "sources/manifest.json",
        "engine_patch/provenance/engine_binding.py": engine / "engine_binding.py",
        "engine_patch/provenance/nm_defined.txt": engine / "build/nm_defined.txt",
        "engine_patch/provenance/nm_undefined.txt": engine / "build/nm_undefined.txt",
        "engine_patch/provenance/readelf_dynamic.txt": engine / "build/readelf_dynamic.txt",
        "engine_patch/provenance/sol_engine_preparation_receipt_01.md":
            engine / "sol_engine_preparation_receipt_01.md",
    })
    for name in ("deps_engine_local_gjk.d", "deps_engine_local_support.d",
                 "deps_engine_interpose.d"):
        mapping[f"engine_patch/provenance/{name}"] = engine / "build" / name
    mapping.update({
        "evidence/epa/a_baseline_reconciliation_01.json": epa / "a_baseline_reconciliation_01.json",
        "evidence/epa/b_regression_reconciliation_01.json": epa / "b_regression_reconciliation_01.json",
        "evidence/epa/epa_contract_closure_01.json": epa / "epa_contract_closure_01.json",
        "evidence/epa/astra_epa_mechanism_memo_01.md": epa / "astra_epa_mechanism_memo_01.md",
        "evidence/epa/astra_epa_b_result_review_01.md": epa / "astra_epa_b_result_review_01.md",
        "evidence/engine/run_d/readiness_receipt.json": engine / "run_d/readiness_receipt.json",
        "evidence/engine/c_stage_closure_01.json": engine / "c_stage_closure_01.json",
        "evidence/engine/run_c/root_launcher_receipt.json": engine / "run_c/root_launcher_receipt.json",
        "evidence/engine/run_c/binding_proof.json": engine / "run_c/binding_proof.json",
        "evidence/engine/run_c1/load_receipt.json": engine / "run_c1/load_receipt.json",
        "evidence/engine/run_c1/binding_proof.json": engine / "run_c1/binding_proof.json",
        "evidence/engine/run_c1/root_launcher_receipt.json":
            engine / "run_c1/root_launcher_receipt.json",
        "evidence/engine/root_d_readback_01.json": engine / "root_d_readback_01.json",
        "evidence/engine/astra_engine_d_result_review_01.md":
            engine / "astra_engine_d_result_review_01.md",
        "evidence/rl/original_contract.md":
            engine / "next_rolling_rl_speed_plan_01/next_contract.md",
        "evidence/rl/environment_contract.md":
            rl / "next_rl_environment_plan_02/next_contract.md",
        "evidence/rl/continuation_contract.md":
            rl / "next_eval_continuation_plan_03/next_contract.md",
        "evidence/rl/root_execution_freeze_02.json": rl / "root_execution_freeze_02.json",
        "evidence/rl/root_execution_freeze_03.json": rl / "root_execution_freeze_03.json",
        "evidence/rl/run_02_boundary_closure_01.json": rl / "run_02_boundary_closure_01.json",
        "evidence/rl/run_02_interruption_observation_01.json":
            rl / "run_02_interruption_observation_01.json",
        "evidence/rl/run_02_interrupted_archive_manifest_01.json":
            rl / "run_02_interrupted_archive_manifest_01.json",
        "evidence/rl/root_run_02_interrupted_readback_01.json":
            rl / "root_run_02_interrupted_readback_01.json",
        "evidence/rl/run_03_archive_manifest_01.json":
            rl / "run_03_archive_manifest_01.json",
        "evidence/rl/root_run_03_readback_01.json": independent,
        "evidence/rl/run_03/continuation_receipt.json":
            continuation / "continuation_receipt.json",
        "evidence/rl/run_03/eight_case_stitch.json":
            continuation / "eight_case_stitch.json",
        "evidence/rl/run_03/root_launcher_receipt.json":
            continuation / "root_launcher_receipt.json",
        "evidence/rl/run_03/runtime_initial.json":
            continuation / "runtime_initial.json",
        "evidence/rl/run_03/checkpoint_reload_receipt.json":
            continuation / "checkpoint_reload_receipt.json",
        "evidence/rl/run_03/interrupted_prefix_comparison.json":
            continuation / "interrupted_prefix_comparison.json",
        "evidence/rl/run_02/training/training_receipt.json":
            original / "training/training_receipt.json",
        "evidence/rl/run_02/training/completed_updates.jsonl":
            original / "training/completed_updates.jsonl",
        "evidence/rl/run_02/protocol.json": original / "protocol.json",
        "evidence/rl/run_01_boundary_closure.json": rl / "run_01_boundary_closure.json",
        "evidence/rl/prior_closed_attempt_run_01/study_receipt.json":
            rl / "run_01/study_receipt.json",
        "evidence/rl/prior_closed_attempt_run_01/root_launcher_receipt.json":
            rl / "run_01/root_launcher_receipt.json",
        "evidence/rl/pure_test_round_1_receipt.json": rl / "pure_test_round_1_receipt.json",
        "evidence/rl/pure_test_round_2_receipt.json": rl / "pure_test_round_2_receipt.json",
        "evidence/rl/astra_rl_execution_review_01.md":
            rl / "astra_rl_execution_review_01.md",
        "evidence/rl/astra_rl_execution_review_inputs_01.json":
            rl / "astra_rl_execution_review_inputs_01.json",
        "evidence/rl/astra_rl_environment_execution_review_02.md":
            rl / "astra_rl_environment_execution_review_02.md",
        "evidence/rl/astra_rl_environment_execution_review_inputs_02.json":
            rl / "astra_rl_environment_execution_review_inputs_02.json",
        "provenance/root_interrupted_readback_01.py":
            rl / "root_interrupted_readback_01.py",
        "provenance/root_readback_03.py": rl / "root_readback_03.py",
        "provenance/claude_opus_publication_review_01.md":
            rl / "publication_staging/claude_opus_publication_review_01.md",
        "provenance/root_claude_publication_review_disposition_01.md":
            rl / "root_claude_publication_review_disposition_01.md",
        "provenance/engine_source_package_draft_01.md":
            rl / "publication_staging/engine_source_package_draft_01.md",
        "provenance/build_compact_publication_02.py":
            rl / "publication_staging/build_compact_publication_02.py",
        "provenance/plot_saved_rolling_trajectories_02.py":
            rl / "publication_staging/plot_saved_rolling_trajectories_02.py",
    })
    for name in ("speed_tracking.png", "box_pitch.png", "shared_leg_residual.png",
                 "figure_receipt.json"):
        mapping[f"plots/{name}"] = figures / name
    if frozen_dso:
        mapping["engine_patch/artifact/libepa01_engine.so"] = engine / "build/libepa01_engine.so"
    for label, path in additional:
        target = PurePosixPath(label)
        if (not target.parts or target.parts[0] != "evidence"
                or ".." in target.parts or label in mapping
                or label == "evidence/rl/case_source_map.json"
                or label.startswith("evidence/rl/compact_trajectories/")):
            raise ValueError(f"invalid additional evidence destination: {label}")
        mapping[label] = path
    return mapping


def result_readme(*, stitched: dict[str, Any], old_readback: dict[str, Any],
                  new_readback: dict[str, Any], execution_valid: bool,
                  policy_gates: bool, archive_url: str) -> str:
    lines = [
        "# Isolated CCD patch and rolling shared-leg RL trial",
        "",
        "The 65,536-control PPO training and its single final checkpoint are preserved.",
        "Three original evaluation cases completed in run_02. That process then",
        "disappeared during the fourth case for an unestablished reason. Its final",
        "native C counters and exit status are unavailable; the entire original",
        "reservation is closed. The partial fourth case is diagnosis only.",
        "A separately budgeted run_03 evaluated precisely the five remaining cases,",
        "restarting the fourth from the prescribed initial state with the same checkpoint.",
        "",
        "- Original run_02 study qualified: **False**; final C counts: **unknown**.",
        f"- New five-case execution valid: **{execution_valid}**; independent readback: **{new_readback['passed']}**.",
        f"- Four final-policy task and speed gates across the fixed eight cases: **{policy_gates}**.",
        "- Uninterrupted global accounting: **False**. These are two process budgets.",
        "- Prior run_01 closed at its environment bootstrap with zero training controls;",
        "  it did not contribute physics to the training or eight cases.",
        "",
        "## Eight fixed cases",
        "",
        "| Speed | Terrain | Actor | Process | Record valid | Task passed | Mean body vx (m/s) |",
        "| ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in stitched["cases"]:
        score = row["score"]
        mean = score["metrics"]["speed_window_mean_body_vx_mps"]
        lines.append(f"| {row['speed_mps']} | {row['terrain']} | {row['actor']} "
                     f"| {row['source_process']} | {score['record_valid']} "
                     f"| {score['task_passed']} | {mean} |")
    lines += [
        "",
        f"Policy mean-speed deltas (.25 minus .20 m/s command): `{stitched['policy_speed_mean_delta_mps']}`.",
        "A zero-policy task failure remains a failure; it is not recast as a policy result.",
        "The actor/parameter hash change alone does not show a causal benefit.",
        "These eight deterministic simulator episodes do not establish statistical",
        "reliability or hardware performance.",
        "",
        "## Reproduce and audit",
        "",
        "`engine_patch/BUILD.md` records the exact isolated DSO build and pinned ABI.",
        "`engine_patch/provenance/engine_binding.py` is the executed fixed-offset",
        "binding bridge; its absolute paths are specific to the original environment.",
        "The installed MuJoCo library was never overwritten; the optional DSO artifact",
        "is for hash comparison only and is not automatically loaded or installed.",
        "The compact package includes source, checkpoint, receipts, scores, source",
        "hashes, decimated trajectories and plots. It omits the full native/endpoint/",
        "trace/state streams. Full paired-initial, torque and contact-record checks",
        "require the two raw run directories and the pinned independent readbacks.",
        "The complete release archive contains raw states and native records from",
        "both processes, including the interrupted prefix kept outside the eight",
        f"completed-case result: {archive_url}",
        "`evidence/rl/case_source_map.json` binds each compact case to its raw archive",
        "directory and file hashes. The compact package alone cannot independently",
        "recompute every contact-geometry or force gate.",
        "No original score or frozen input was rewritten.",
        "",
        f"Original completed-case readback passed: `{old_readback['passed_for_completed_records_only']}`.",
        f"New five-case readback passed: `{new_readback['passed']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--engine-work", type=Path, required=True)
    parser.add_argument("--epa-work", type=Path, required=True)
    parser.add_argument("--rl-work", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--continuation-run", type=Path, required=True)
    parser.add_argument("--independent-readback", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--full-archive-url", required=True)
    parser.add_argument("--include-frozen-dso", action="store_true")
    parser.add_argument("--additional-evidence", nargs=2, action="append", default=[],
                        metavar=("PACKAGE_PATH", "SOURCE_PATH"))
    args = parser.parse_args()
    repo, engine, epa, rl, original, continuation, independent, figures = (
        path.resolve(strict=True) for path in (
            args.repo, args.engine_work, args.epa_work, args.rl_work,
            args.original_run, args.continuation_run, args.independent_readback,
            args.figures,
        )
    )
    if (original != rl / "run_02" or continuation != rl / "run_03"
            or independent != rl / "root_run_03_readback_01.json"
            or not args.full_archive_url.startswith("https://")):
        raise ValueError("reviewed run locations, readback or complete archive URL differ")
    if args.destination.exists():
        raise FileExistsError("publication destination must be new and exclusive")

    old_freeze = verify_freeze(rl / "root_execution_freeze_02.json")
    new_freeze = verify_freeze(rl / "root_execution_freeze_03.json")
    if (old_freeze.get("output_directory") != str(original)
            or new_freeze.get("output_directory") != str(continuation)):
        raise ValueError("run directories differ from the actual source freezes")
    old_archive = verify_archive(original, rl / "run_02_interrupted_archive_manifest_01.json")
    verify_archive(continuation, rl / "run_03_archive_manifest_01.json")
    if len(old_archive["files"]) != 55:
        raise ValueError("original retained archive no longer has 55 files")
    boundary = load_json(rl / "run_02_boundary_closure_01.json")
    old_readback = load_json(rl / "root_run_02_interrupted_readback_01.json")
    launcher = load_json(continuation / "root_launcher_receipt.json")
    receipt = load_json(continuation / "continuation_receipt.json")
    stitched_path = continuation / "eight_case_stitch.json"
    stitched = load_json(stitched_path)
    new_readback = load_json(independent)
    if (boundary.get("final_native_C_counts") is not None
            or boundary.get("retry_same_run_permitted") is not False
            or old_readback.get("passed_for_completed_records_only") is not True
            or old_readback.get("study_qualified") is not False
            or len(old_readback.get("completed_cases", {})) != 3):
        raise ValueError("interrupted original run boundary/readback differs")
    execution_valid = bool(
        launcher.get("process_attempts") == 1 and launcher.get("exit_code") == 0
        and launcher.get("error") is None and not launcher.get("post_execution_hash_mismatches")
        and receipt.get("new_five_case_execution_valid") is True
        and receipt.get("actual_counts_consistent") is True
        and new_readback.get("passed") is True
    )
    if not execution_valid:
        raise ValueError("new five-case execution or independent readback is not valid")
    if (stitched.get("new_five_case_execution_valid") is not True
            or stitched.get("original_run_02_study_qualified") is not False
            or stitched.get("original_final_C_counts") is not None
            or stitched.get("uninterrupted_global_accounting") is not False
            or receipt.get("original_run_02_study_qualified") is not False
            or receipt.get("original_final_C_counts") is not None
            or new_readback.get("original_run_02_qualified") is not False
            or new_readback.get("original_final_C_counts") is not None
            or receipt.get("actual_new_evaluation_controls") != new_readback.get("actual_new_controls")):
        raise ValueError("interrupted/new-process accounting or readback differs")
    policy_gates = stitched.get("aggregate_four_policy_task_and_speed_gates_passed")
    if (type(policy_gates) is not bool
            or policy_gates != receipt.get("aggregate_four_policy_task_and_speed_gates_passed")
            or policy_gates != new_readback.get("aggregate_policy_task_and_speed_gates_passed")):
        raise ValueError("aggregate policy qualification differs across independent receipts")

    checkpoint = original / "training/final_checkpoint"
    sidecar = load_json(checkpoint / "model.metadata.json")
    training = load_json(original / "training/training_receipt.json")
    model_hash = digest(checkpoint / "model.zip")["sha256"]
    metadata_hash = digest(checkpoint / "model.metadata.json")["sha256"]
    probe_hash = digest(checkpoint / "reload_probe.npz")["sha256"]
    if (training.get("passed") is not True
            or training.get("physical_control_transitions") != 65536
            or training.get("completed_train_calls") != 512
            or model_hash != MODEL_SHA256 or sidecar.get("model_sha256") != model_hash
            or training.get("model_sha256") != model_hash
            or training.get("checkpoint_metadata_sha256") != metadata_hash
            or sidecar.get("reload_probe_sha256") != probe_hash
            or sidecar.get("num_timesteps") != 65536
            or sidecar.get("completed_train_calls") != 512
            or sidecar.get("optimization_epochs") != 2048):
        raise ValueError("checkpoint model, metadata, reload probe or training count differs")
    sources, compact = case_sources(original, continuation, stitched)
    verify_figures(figures, original, continuation, stitched_path, sources)

    extra = [(label, Path(source).resolve(strict=True))
             for label, source in args.additional_evidence]
    mapping = static_package_map(
        repo, engine, epa, rl, original, continuation, independent, figures,
        frozen_dso=args.include_frozen_dso, additional=extra,
    )
    for name in (*OLD_CASES, *NEW_CASES):
        run = original if name in OLD_CASES else continuation
        folder = run / "evaluation" / name
        destination_prefix = f"evidence/rl/completed_cases/{name}"
        for filename in ("score.json", "receipt.json", "episode_metadata.json",
                         "geometry_manifest.json"):
            mapping[f"{destination_prefix}/{filename}"] = folder / filename
        mapping[f"{destination_prefix}/case.json"] = run / "evaluation" / f"{name}.case.json"
    for name in NEW_CASES:
        for stage in ("before", "after"):
            mapping[f"evidence/rl/case_boundaries/{name}.{stage}.json"] = (
                continuation / "evaluation" / f"{name}.{stage}.json"
            )
    for filename in ("model.zip", "model.metadata.json", "reload_probe.npz"):
        mapping[f"checkpoint/{filename}"] = checkpoint / filename
    if args.include_frozen_dso and digest(engine / "build/libepa01_engine.so")["sha256"] != DSO_SHA256:
        raise ValueError("optional DSO differs from the executed frozen artifact")

    # Resolve all inputs and trajectory decimation before creating a destination.
    missing = [f"{relative}: {source}" for relative, source in mapping.items()
               if not source.is_file()]
    if missing:
        raise FileNotFoundError("publication inputs absent:\n" + "\n".join(missing))
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for relative, source in mapping.items():
        copy_one(source, destination / relative)
    for name, rows in compact.items():
        path = destination / "evidence/rl/compact_trajectories" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, rows)
    write_json(destination / "evidence/rl/case_source_map.json", {
        "schema": "two-process-rolling-case-sources-v1",
        "source_runs": {"run_02": "interrupted original", "run_03": "bounded continuation"},
        "old_partial_fourth_case_excluded_from_results": True,
        "cases": sources,
    })
    readme = result_readme(stitched=stitched, old_readback=old_readback,
                           new_readback=new_readback, execution_valid=execution_valid,
                           policy_gates=policy_gates, archive_url=args.full_archive_url)
    with (destination / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(readme)
    listed = {
        str(path.relative_to(destination)): digest(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "publication_manifest.json"
    }
    write_json(destination / "publication_manifest.json", {
        "schema": "rolling-isolated-engine-two-process-compact-publication-v2",
        "original_run_02_study_qualified": False,
        "original_final_C_counts": None,
        "new_five_case_execution_valid": execution_valid,
        "aggregate_four_policy_task_and_speed_gates_passed": policy_gates,
        "uninterrupted_global_accounting": False,
        "complete_raw_archive_url": args.full_archive_url,
        "run_02_raw_archive_manifest": digest(rl / "run_02_interrupted_archive_manifest_01.json"),
        "run_03_raw_archive_manifest": digest(rl / "run_03_archive_manifest_01.json"),
        "independent_readback": digest(independent),
        "figure_receipt": digest(figures / "figure_receipt.json"),
        "full_native_archives_copied": False,
        "installed_mujoco_library_copied": False,
        "frozen_dso_artifact_copied_for_audit_only": bool(args.include_frozen_dso),
        "files": listed,
    })
    print(json.dumps({"destination": str(destination),
                      "new_five_case_execution_valid": execution_valid,
                      "aggregate_policy_gates": policy_gates,
                      "published_files": len(listed) + 1}))


if __name__ == "__main__":
    main()
