"""Prepare a compact 07 evidence package and separate complete raw run asset.

Draft only: run after both independent root readbacks and closure receipts exist.
Uses stdlib only, never imports the engine, policy, or a physical verifier.
Existing 05/06 release assets are linked by identity, not copied.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

R = Path("/home/lyh/wheel-legged-control-lab")
B = Path("/home/lyh/wheel-legged-control-lab-work/recovery-20260912")
W = B / "stability_20260924_rl_gui01"
OLD_W = B / "stability_20260924_body_speed01"
EXPECTED_HEAD = "9a4079d72ccc3e25d4788256784cd9a78aa815e5"
SCOPES = (
    ("frozen77", B / "rl_improvement_20260914/frozen_source_before.json", "sha256", R),
    ("formal215", B / "stability_20260920/shared_heave_formal_preflight_01.json", "input_sha256", R),
    ("physical193", B / "stability_20260922/single_step_readiness_01/protocol.json", "input_sha256", R),
    ("runtime767", OLD_W / "root_execution_freeze_01.json", "files", R),
    ("closed77", OLD_W / "run_01_archive_manifest_01.json", "files", OLD_W / "run_01"),
    ("published06", R / "results/d1_body_speed_feedback_20260924/publication_manifest.json",
     "files", R / "results/d1_body_speed_feedback_20260924"),
    ("published05", R / "results/d1_rolling_residual_speed_20260924/publication_manifest.json",
     "files", R / "results/d1_rolling_residual_speed_20260924"),
)
EXPECTED_COUNTS = {"frozen77": 77, "formal215": 215, "physical193": 193,
                   "runtime767": 767, "closed77": 77, "published06": 61,
                   "published05": 245}
RUNTIME_FILES = (
    "scripts/play_d1_latest_rl.py", "scripts/run_d1_latest_rl.py",
    "scripts/d1_latest_rl_controls.py", "scripts/d1_latest_rl_viewer.py",
    "scripts/d1_latest_rl_validation.py", "docs/latest_rl_gui_contract_07.md",
    "docs/latest_rl_gui.md", "docs/codex_handoff_20260924_rl_gui.md",
    "results/d1_latest_rl_sessions/.gitignore",
    "tests/test_d1_latest_rl_controls_opus.py",
    "tests/test_d1_latest_rl_render_opus.py", "tests/test_d1_latest_rl_integration_pure.py",
)
WORK_FILES = (
    "new_contract.md", "verify_gui_session_07.py",
    "verify_gui_session_07_readback02.py", "launch_gui_validation_01.py",
    "pure_test_round_1_receipt.json", "pure_test_round_2_receipt.json",
    "actual_claude_contribution_01.json", "actual_claude_contribution_final_07.json",
    "gui_policy_box_readback_attempt_01.json", "astra_gui_review_inputs_07.json",
    "astra_gui_execution_review_07.md", "astra_gui_a_readback_addendum_07.md",
    "astra_gui_prefix_readback_07.json", "manual_static_readiness_07.md",
    "astra_gui_final_acceptance_07.md", "astra_gui_final_acceptance_checks_07.json",
    "baseline_hash_audit_01.json", "build_gui_publication_07.py",
    "close_execution_07.py",
    "final_execution_audit_07.json", "delivery_readiness_07.json",
    "continuation_state_07.json",
)
RUN_SUMMARY_FILES = (
    "session.json", "child_pid.json", "worker_preflight.json",
    "environment_mutation_receipt.json", "runtime_initial.json",
    "construction_receipt.json", "strict_original_checkpoint_reload.json",
    "controller_law_transfer.json", "module_origins.json",
    "validation_driver_receipt.json", "worker_receipt.json", "launcher_receipt.json",
    "native_monitor_receipt.json",
    "segment_00/boundary_before.json", "segment_00/boundary_after.json",
    "segment_00/reset_pair_receipt.json", "segment_00/episode_metadata.json",
    "segment_00/geometry_manifest.json", "segment_01/boundary_before.json",
    "segment_01/boundary_after.json", "segment_01/reset_pair_receipt.json",
    "segment_01/episode_metadata.json", "segment_01/geometry_manifest.json",
)
BANNED_NAMES = ("xauthority", "cookie", "credential", "secret", "token", ".env", "claude")


def fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    fail(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def identity(path: Path) -> dict:
    fail(path.is_file() and not path.is_symlink(), f"missing/symlinked input: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def historical_audit() -> dict:
    seen: dict[str, dict] = {}
    counts = {}
    for label, manifest, key, base in SCOPES:
        table = read(manifest)[key]
        fail(len(table) == EXPECTED_COUNTS[label], f"historical {label} count differs")
        for name, expected in table.items():
            path = base / name
            actual = seen.setdefault(str(path), identity(path))
            digest = expected if isinstance(expected, str) else expected["sha256"]
            fail(actual["sha256"] == digest
                 and (not isinstance(expected, dict) or
                      actual["bytes"] == expected.get("bytes", actual["bytes"])),
                 f"historical frozen input changed: {path}")
        counts[label] = len(table)
    fail(len(seen) == 1183, "historical unique input count differs")
    return {"scope_counts": counts, "unique_files_rehashed": len(seen),
            "hash_mismatches": [], "historical_head_at_06_release": EXPECTED_HEAD}


def run_tree(folder: Path, profile: str) -> dict[str, dict]:
    fail(folder.is_absolute() and folder.is_dir() and not folder.is_symlink(),
         f"run directory missing: {folder}")
    session = read(folder / "session.json")
    worker = read(folder / "worker_receipt.json")
    launcher = read(folder / "launcher_receipt.json")
    fail(session["validation_profile"] == profile and session["mode"] == "validation"
         and session["output_directory"] == str(folder)
         and worker["execution_complete"] is True and worker["failure"] is None
         and worker["completed_controls"] == 1200
         and launcher["exit_code"] == 0 and launcher["reason"] is None
         and launcher["source_hash_mismatches"] == [],
         f"closed validation run not clean: {profile}")
    for name in RUN_SUMMARY_FILES:
        identity(folder / name)
    if profile == "gui_policy_box":
        identity(folder / "render_receipt.json")
        for name in ("segment_00/drive_200.png", "segment_00/drive_650.png",
                     "segment_01/after_reset.png"):
            fail(identity(folder / name)["bytes"] > 0, f"GUI image empty: {name}")
    files = {}
    for path in folder.rglob("*"):
        fail(not path.is_symlink(), f"symlink in raw run: {path}")
        if path.is_file():
            relative = path.relative_to(folder).as_posix()
            fail(not any(word in relative.lower() for word in BANNED_NAMES)
                 and (relative.endswith((".json", ".jsonl.gz", ".npz", ".png", ".log"))),
                 f"raw run file requires manual review: {relative}")
            files[relative] = identity(path)
    return files


def copy_checked(source: Path, target: Path) -> dict:
    prior = identity(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, target.open("xb") as out:
        shutil.copyfileobj(inp, out)
    fail(identity(target) == prior, f"copy changed bytes: {source}")
    return prior


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_archive(path: Path, rows: dict[str, dict], extras: dict[str, tuple[Path, dict]],
                   manifest: dict) -> None:
    expected = {f"runs/{profile}/{name}": row
                for profile, files in rows.items() for name, row in files.items()}
    expected.update({name: row for name, (_path, row) in extras.items()})
    expected["archive_manifest.json"] = {
        "sha256": hashlib.sha256((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()).hexdigest(),
        "bytes": len((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()),
    }
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        fail({m.name for m in members} == set(expected)
             and all(m.isfile() for m in members), "raw asset member set differs")
        for member in members:
            stream = archive.extractfile(member)
            fail(stream is not None, f"raw asset member unreadable: {member.name}")
            h, size = hashlib.sha256(), 0
            while block := stream.read(1024 * 1024):
                h.update(block)
                size += len(block)
            fail({"sha256": h.hexdigest(), "bytes": size} == expected[member.name],
                 f"raw asset member hash differs: {member.name}")


def prepare(args: argparse.Namespace) -> dict:
    fail(args.package.is_absolute() and args.archive.is_absolute(),
         "--package and --archive must be absolute new paths")
    fail(args.package == R / "results/d1_latest_rl_gui_20260924",
         "compact destination must be the new 07 repo result package")
    fail(not args.package.exists() and not args.archive.exists(),
         "publication destination already exists; no overwrite")
    fail((R / "results/my_course_drive_01").is_dir(),
         "protected user drive directory missing; stop before publication")
    old = historical_audit()
    go = read(W / "astra_gui_review_inputs_07.json")
    pure = W / "pure_test_round_2_receipt.json"
    fail(go.get("decision") == "GO"
         and go.get("pure_test_receipt_sha256") == identity(pure)["sha256"]
         and read(pure).get("all_passed") is True,
         "frozen GO/pure-test receipt differs")
    for name, expected in go["inputs"].items():
        fail(identity(Path(name)) == expected, f"GO input changed: {name}")
    profiles = {"gui_policy_box": args.gui_run, "headless_zero_plane": args.headless_run}
    readbacks = {"gui_policy_box": args.gui_readback,
                 "headless_zero_plane": args.headless_readback}
    closed_manifests = {"gui_policy_box": args.gui_closed_manifest,
                        "headless_zero_plane": args.headless_closed_manifest}
    final_audit = read(W / "final_execution_audit_07.json")
    fail(final_audit.get("schema") == "d1-latest-rl-gui-closed-execution-v1"
         and final_audit.get("passed") is True
         and final_audit.get("old_unique_files_rehashed") == 1183
         and final_audit.get("go_inputs_unchanged") == 51
         and final_audit.get("closed_run_files_unchanged") == 98
         and final_audit.get("physical_contract_07_closed") is True
         and final_audit.get("additional_physics_under_07_permitted") is False
         and final_audit.get("new_control_steps") == 2400
         and final_audit.get("new_normal_native_steps") == 12000
         and final_audit.get("new_compiler_native_steps") == 6
         and final_audit.get("new_training_steps") == 0
         and final_audit.get("hash_mismatches") == [],
         "root final execution audit is absent or incomplete")
    source_rows = {}
    for profile, folder in profiles.items():
        source_rows[profile] = run_tree(folder, profile)
        outer = read(W / f"{profile}_root_exit_01.json")
        reservation = read(W / f"{profile}_root_reservation_01.json")
        fail(outer.get("exit_code") == 0 and outer.get("error") is None
             and outer.get("retry_permitted") is False
             and reservation.get("automatic_retry") is False
             and reservation.get("output") == str(folder),
             f"root bounded process receipt differs: {profile}")
        closed = read(closed_manifests[profile])
        fail(closed.get("schema") == "closed-07-physical-run-files-v1"
             and {str(folder / relative): row for relative, row in source_rows[profile].items()}
             == closed.get("files")
             and closed.get("control_steps") == 1200
             and closed.get("normal_native_steps") == 6000
             and closed.get("compiler_native_steps") == 3,
             f"sealed raw run file set differs: {profile}")
        result = read(readbacks[profile])
        audited = final_audit["profiles"][profile]
        fail(result.get("passed") is True and result.get("profile") == profile
             and result.get("controls") == 1200 and result.get("normal_native") == 6000
             and result.get("compiler_native") == 3
             and audited["readback_identity"] == identity(readbacks[profile])
             and audited["run_files"] == len(source_rows[profile]),
             f"independent readback or boundary closure incomplete: {profile}")
    raw_extra_paths = {
        **{f"source/repo/{name}": R / name for name in RUNTIME_FILES},
        **{f"source/work/{name}": W / name for name in (
            "verify_gui_session_07.py", "verify_gui_session_07_readback02.py",
            "launch_gui_validation_01.py", "run_gui_pure_tests_01.py",
            "run_gui_pure_tests_02.py", "new_contract.md", "close_execution_07.py",
            "astra_gui_review_inputs_07.json")},
        **{f"root/{profile}/{suffix}.json": W / f"{profile}_root_{suffix}_01.json"
           for profile in profiles for suffix in ("reservation", "exit")},
    }
    extras = {name: (path, identity(path)) for name, path in raw_extra_paths.items()}
    # All inputs above are checked before either exclusive output is created.
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest = {"schema": "d1-latest-rl-gui-07-raw-manifest-v1",
                        "profiles": {name: {"files": rows, "file_count": len(rows)}
                                     for name, rows in source_rows.items()},
                        "raw_run_file_count": sum(map(len, source_rows.values())),
                        "extras": {name: row for name, (_path, row) in extras.items()},
                        "historical_audit": old,
                        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "excluded_private_x11_authentication": True,
                        "excluded_unreviewed_claude_transcripts": True}
    with tarfile.open(args.archive, "x:gz") as archive:
        for profile, folder in profiles.items():
            for relative in source_rows[profile]:
                archive.add(folder / relative, arcname=f"runs/{profile}/{relative}", recursive=False)
        for name, (path, _row) in extras.items():
            archive.add(path, arcname=name, recursive=False)
        payload = (json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n").encode()
        import io
        info = tarfile.TarInfo("archive_manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    verify_archive(args.archive, source_rows, extras, archive_manifest)
    asset = identity(args.archive)
    args.package.mkdir(parents=True, exist_ok=False)
    for relative in RUNTIME_FILES:
        copy_checked(R / relative, args.package / "source" / relative)
    for name in WORK_FILES:
        copy_checked(W / name, args.package / "evidence" / name)
    for profile, folder in profiles.items():
        for name in (f"{profile}_root_reservation_01.json", f"{profile}_root_exit_01.json"):
            copy_checked(W / name, args.package / "evidence" / profile / name)
        for name in RUN_SUMMARY_FILES:
            copy_checked(folder / name, args.package / "evidence" / profile / name)
        if profile == "gui_policy_box":
            copy_checked(folder / "render_receipt.json",
                         args.package / "evidence" / profile / "render_receipt.json")
            for name in ("segment_00/drive_200.png", "segment_00/drive_650.png",
                         "segment_01/after_reset.png"):
                copy_checked(folder / name, args.package / "images" / name)
        copy_checked(readbacks[profile], args.package / "evidence" / profile / "readback.json")
        copy_checked(closed_manifests[profile],
                     args.package / "evidence" / profile / "closed_raw_manifest.json")
    write_json(args.package / "evidence/raw_archive_manifest.json", archive_manifest)
    write_json(args.package / "evidence/historical_hash_audit.json", old)
    summary = {"schema": "d1-latest-rl-gui-07-publication-summary-v1",
               "status": "both_independent_validation_readbacks_passed",
               "profiles": {p: read(readbacks[p]) for p in profiles},
               "training_controls_added": 0,
               "interactive_validation_not_a_causal_rl_comparison": True,
               "hardware_claimed": False, "broad_terrain_claimed": False,
               "raw_asset": {"filename": args.archive.name, **asset},
               "historical_hash_audit": old}
    write_json(args.package / "summary.json", summary)
    gui_speed = read(readbacks["gui_policy_box"])["segments"][0]["interval_descriptive_only"]["0.25"]
    lines = ["# D1 latest RL interactive GUI — 07 evidence", "",
             "On the validated machine, `rtk proxy d1-rl` opens a fresh default policy/box window;",
             "from the repository root, `rtk proxy python3 -B scripts/play_d1_latest_rl.py`",
             "is the source entry. Append `--check` for a read-only source preflight.", "",
             "This package records two bounded validation sessions of the published final RL checkpoint",
             "and the 06 body-speed controller. It documents the actual saved results; it does not",
             "establish an independent RL benefit, hardware behavior, or broad terrain reliability.", "",
             "The GUI box run used the saved final policy; the headless plane run used exact zero",
             "policy residual with the same baseline controller. Different terrains make the two",
             "profiles an interface check, not a causal policy comparison.", "",
             "| Profile | Controls | Normal native | Compiler | Readback |", "|---|---:|---:|---:|---|",]
    for profile in profiles:
        row = read(readbacks[profile])
        lines.append(f"| {profile} | {row['controls']} | {row['normal_native']} | "
                     f"{row['compiler_native']} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines += ["", (f"The GUI box run's switched-command 0.25 m/s interval had mean observed body speed "
                   f"{gui_speed['mean_body_vx_mps']:.8f} m/s over {gui_speed['controls']} controls."),
              "This interval includes transients and is descriptive, not the 06 fixed-speed",
              "qualification window. The 12-second GUI simulation took 130.25 seconds of wall",
              "time in validation; real-time playback and human keyboard usability are unproven.", "",
              "The full raw asset is intended for the `latest-rl-gui-20260924` release:",
              "https://github.com/LYHrmer/wheel-legged-control-lab/releases/tag/latest-rl-gui-20260924",
              "Confirm the actual published URL, asset digest and CI in the final publication receipt.", "",
              "`summary.json` links the compact evidence to the separate full raw asset;",
              "`publication_manifest.json` lists every compact file hash. The archive manifest",
              "lists the complete closed A/B run files, including native and trace records.", "",
              "Earlier frozen packages remain available at the existing releases:",
              "https://github.com/LYHrmer/wheel-legged-control-lab/releases/tag/rolling-rl-damping-20260924",
              "https://github.com/LYHrmer/wheel-legged-control-lab/releases/tag/body-speed-qualified-20260924", ""]
    (args.package / "README.md").write_text("\n".join(lines), encoding="utf-8")
    files = {path.relative_to(args.package).as_posix(): identity(path)
             for path in args.package.rglob("*") if path.is_file()}
    write_json(args.package / "publication_manifest.json", {
        "schema": "d1-latest-rl-gui-07-compact-publication-v1", "files": files,
        "raw_asset": asset, "raw_archive_manifest":
            identity(args.package / "evidence/raw_archive_manifest.json"),
        "profiles": list(profiles), "historical_hash_audit": old})
    fail(historical_audit() == old, "historical frozen inputs changed during packaging")
    for profile, folder in profiles.items():
        fail(run_tree(folder, profile) == source_rows[profile],
             f"closed run changed during packaging: {profile}")
    for name, expected in go["inputs"].items():
        fail(identity(Path(name)) == expected, f"GO input changed during packaging: {name}")
    for name, (path, expected) in extras.items():
        fail(identity(path) == expected, f"raw source changed during packaging: {name}")
    return {"package": str(args.package), "package_files": len(files),
            "raw_asset": {"path": str(args.archive), **asset}, "historical_files": 1183}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui-run", type=Path, required=True)
    parser.add_argument("--headless-run", type=Path, required=True)
    parser.add_argument("--gui-readback", type=Path, required=True)
    parser.add_argument("--headless-readback", type=Path, required=True)
    parser.add_argument("--gui-closed-manifest", type=Path, required=True)
    parser.add_argument("--headless-closed-manifest", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
