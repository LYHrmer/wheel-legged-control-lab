"""Offline 08-T failed-run readback; never imports an engine or policy.

The failed control's returned native interval is evidence, not a completed
control or a Gaussian/compact training row. No heldout or model qualification.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
from verify_course_e_08_03 import close, identity, require
from verify_rl16_training_08 import (
    SCHEMA,
    _closed_manifest,
    _curriculum_events,
    _file_matches,
    _json,
    _load_npz,
    _source,
    _training_schedule,
)


def _compact_prefix(run: Path, completed: int) -> dict:
    folder = run / "training"
    manifest = _json(folder / "training_blocks_manifest.json")
    require(manifest["complete"] is False
            and manifest["completed_controls"] == manifest["gaussian_records"] == completed
            and manifest["runtime_control_completed"] == completed
            and manifest["pending_gaussian_control_index"] is None
            and manifest["reason"] == "train_or_checkpoint_failure",
            "failed T compact manifest count/seal differs")
    blocks, gaussian = manifest["numeric_blocks"], manifest["gaussian_blocks"]
    require(len(blocks) == len(gaussian) == math.ceil(completed / 1024),
            "failed T compact/Gaussian block count differs")
    seen = nonzero = clipped = 0
    episode_ticks: dict[int, int] = {}
    episode_starts: dict[int, int] = {}
    episode_last: dict[int, dict] = {}
    last_control = None
    for block_index, (left, right) in enumerate(zip(blocks, gaussian)):
        rows = min(1024, completed - seen)
        require(left["file"] == f"controls_{block_index:04d}.npz"
                and right["file"] == f"gaussian_{block_index:04d}.npz"
                and left["rows"] == right["rows"] == rows
                and left["first_control_index"] == right["first_control_index"] == seen
                and left["last_control_index"] == right["last_control_index"]
                == seen + rows - 1,
                "failed T compact/Gaussian block indices differ")
        _file_matches(folder / "training_numeric_blocks" / left["file"], left)
        _file_matches(folder / "training_gaussian_blocks" / right["file"], right)
        a = _load_npz(folder / "training_numeric_blocks" / left["file"])
        g = _load_npz(folder / "training_gaussian_blocks" / right["file"])
        require(all(values.shape[0] == rows and np.isfinite(values).all()
                    for values in (*a.values(), *g.values()))
                and a["input_observation99"].shape == (rows, 99)
                and a["effective_action16"].shape == (rows, 16)
                and a["actuator_applied5x16"].shape == (rows, 5, 16)
                and np.array_equal(a["control_index"], np.arange(seen, seen + rows))
                and np.array_equal(g["control_index"], a["control_index"])
                and np.array_equal(g["episode_index"], a["episode_index"])
                and np.array_equal(g["episode_tick"], a["episode_tick"])
                and np.array_equal(g["clipped_action16"], a["policy_env_input16"])
                and np.array_equal(a["policy_clipped16"],
                                   np.clip(a["policy_env_input16"], -1, 1))
                and np.array_equal(g["effective_action16"], a["effective_action16"])
                and np.array_equal(np.clip(g["raw_gaussian_action16"], -1, 1),
                                   g["clipped_action16"])
                and np.array_equal(a["actuator_delayed5x16"],
                                   a["actuator_applied5x16"])
                and np.array_equal(a["actuator_applied5x16"],
                                   np.broadcast_to(a["stage_safe_torque_nm"][:, None, :],
                                                   (rows, 5, 16)))
                and np.allclose(a["reward_terms9"].sum(axis=1), a["reward"],
                                rtol=0, atol=1e-9),
                "failed T completed compact control/action/reward chain differs")
        nonzero += int(np.count_nonzero(np.any(a["effective_action16"] != 0, axis=1)))
        clipped += int(np.count_nonzero(np.any(
            g["raw_gaussian_action16"] != g["clipped_action16"], axis=1)))
        for offset in range(rows):
            episode = int(a["episode_index"][offset])
            tick = int(a["episode_tick"][offset])
            if episode not in episode_ticks:
                schedule = _training_schedule(folder, episode)
                require(episode == len(episode_ticks) and tick == 0,
                        "failed T episode reset order differs")
                episode_ticks[episode] = 0
                episode_starts[episode] = seen + offset
                episode_last[episode] = {
                    "schedule": schedule, "servo_vx": 0.0, "servo_yaw": 0.0,
                    "count": 0, "end": seen + offset,
                    "drive_sse": 0.0, "drive_count": 0, "reward_sum": 0.0,
                    "terminated": False, "truncated": False,
                }
            entry = episode_last[episode]
            require(tick == episode_ticks[episode]
                    and entry["schedule"]["start_completed_global_controls"]
                    == episode_starts[episode]
                    and not entry["terminated"] and not entry["truncated"],
                    "failed T completed episode tick/start differs")
            raw = entry["schedule"]["raw_commands"][tick]
            values = [raw[key] for key in (
                "forward_velocity_mps", "lateral_velocity_mps",
                "yaw_rate_rps", "clearance_m")]
            entry["servo_vx"] += max(-.005, min(.005,
                                               values[0] - entry["servo_vx"]))
            entry["servo_yaw"] += max(-.006, min(.006,
                                                values[2] - entry["servo_yaw"]))
            require(close(a["raw_command_vx_vy_yaw_clearance"][offset], values)
                    and close(a["servo_command_vx_vy_yaw_clearance"][offset],
                              [entry["servo_vx"], 0.0, entry["servo_yaw"], .455])
                    and not (a["terminated"][offset] and a["truncated"][offset]),
                    "failed T completed raw/servo/terminal trace differs")
            if values[0] == values[2] == 0.0:
                require(not np.any(a["effective_action16"][offset]),
                        "failed T completed zero-request residual is nonzero")
            else:
                require(np.array_equal(a["effective_action16"][offset],
                                       a["policy_clipped16"][offset]),
                        "failed T completed applied action differs")
            if entry["servo_vx"] > 0.0:
                diff = float(a["body_com_vx_mps"][offset]) - entry["servo_vx"]
                entry["drive_sse"] += diff * diff
                entry["drive_count"] += 1
            entry["reward_sum"] += float(a["reward"][offset])
            entry["terminated"] = bool(a["terminated"][offset])
            entry["truncated"] = bool(a["truncated"][offset])
            entry["count"] += 1
            entry["end"] += 1
            episode_ticks[episode] += 1
            last_control = {"global_index": seen + offset,
                            "episode_index": episode, "episode_tick": tick,
                            "reward": float(a["reward"][offset])}
        seen += rows
    require(seen == completed and nonzero > 0,
            "failed T compact completed prefix is missing or all zero residual")
    curriculum = _curriculum_events(folder, episode_last, False)
    require(curriculum["closed_episodes"] == manifest["closed_episodes"]
            and curriculum["level_reached"] == manifest["level_reached"],
            "failed T curriculum events differ from compact completed prefix")
    partial = _json(folder / "training_partial_episode.json")
    require(manifest["partial_episode"] == partial
            and partial["closed"] is False
            and partial["episode_index"] == max(episode_ticks)
            and partial["completed_controls"] == episode_ticks[partial["episode_index"]]
            and partial["start_control_index"] == episode_starts[partial["episode_index"]]
            and partial["end_completed_global_controls"] == completed
            and close(partial["drive_sse"],
                      episode_last[partial["episode_index"]]["drive_sse"], atol=1e-7)
            and partial["drive_count"]
            == episode_last[partial["episode_index"]]["drive_count"]
            and close(partial["reward_sum"],
                      episode_last[partial["episode_index"]]["reward_sum"], atol=1e-7),
            "failed T last partial episode differs from compact prefix")
    return {"completed_controls_verified": seen, "numeric_blocks": len(blocks),
            "gaussian_blocks": len(gaussian), "nonzero_effective_action_controls": nonzero,
            "Gaussian_clipped_controls": clipped, "episode_count_seen": len(episode_ticks),
            "last_completed_control": last_control,
            "partial_episode": partial, "curriculum_trace": curriculum,
            "full_native_contact_force_qualification_available": False}


def _native_prefix(run: Path, completed: int, native_returned: int,
                   segment: dict) -> dict:
    folder = run / "training"
    files, arrays = segment["native_files"], segment["train_array_files"]
    require(len(files) == len(arrays) == math.ceil(native_returned / 5120),
            "failed T native block count differs")
    seen = max_roll = max_pitch = nonwheel = 0
    final_interval: list[dict] = []
    for index, (name, array_name) in enumerate(zip(files, arrays)):
        require(name == f"native_block_{index:04d}.jsonl.gz"
                and array_name == f"native_arrays_{index:04d}.npz",
                "failed T native block order differs")
        path = folder / name
        require(path.is_file() and not path.is_symlink(),
                "failed T compact native summary absent")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        a = _load_npz(folder / array_name)
        count = min(5120, native_returned - seen)
        require(len(rows) == count and all(value.shape[0] == count
                                           and np.isfinite(value).all()
                                           for value in a.values()),
                "failed T native compact row/array length or finite differs")
        for offset, row in enumerate(rows):
            current = seen + offset
            require(row["native_index"] == current
                    and close(row["start_time_s"], a["start_time_s"][offset], atol=1e-12)
                    and close(row["end_time_s"], a["end_time_s"][offset], atol=1e-12)
                    and close(row["end_time_s"] - row["start_time_s"], .002, atol=1e-12)
                    and row["nonwheel_contact_count"] >= 0,
                    "failed T native index/clock/array differs")
            max_roll = max(max_roll, abs(row["roll_deg"]))
            max_pitch = max(max_pitch, abs(row["pitch_deg"]))
            nonwheel += row["nonwheel_contact_count"]
            if current >= 5 * completed:
                final_interval.append({
                    "native_index": current,
                    "start_time_s": row["start_time_s"],
                    "end_time_s": row["end_time_s"],
                    "nonwheel_contact_count": row["nonwheel_contact_count"],
                    "qpos_after_sha256": _array_digest(a["qpos_after"][offset]),
                    "qvel_after_sha256": _array_digest(a["qvel_after"][offset]),
                    "ctrl_after_sha256": _array_digest(a["ctrl_after"][offset]),
                })
        require(np.allclose(a["end_time_s"] - a["start_time_s"], .002,
                            rtol=0, atol=1e-12),
                "failed T native array clock is not one .002 step")
        seen += count
    require(seen == native_returned and nonwheel == 0
            and len(final_interval) == native_returned - 5 * completed,
            "failed T native returned/extra interval/nonwheel differs")
    return {"native_rows_verified": seen,
            "native_completed_control_prefix": 5 * completed,
            "returned_native_after_last_completed_control": final_interval,
            "last_native_interval_is_completed_control": False,
            "nonwheel_contact_candidates": nonwheel,
            "max_abs_roll_deg": max_roll, "max_abs_pitch_deg": max_pitch,
            "full_contact_force_qualification_available": False}


def _array_digest(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify(run: Path) -> tuple[dict, dict]:
    require(run.is_absolute() and run.is_dir(), "T partial --run must be absolute")
    session, source = _source(run)
    worker = _json(run / "worker_receipt.json")
    launcher = _json(run / "launcher_receipt.json")
    require(worker["schema"] == launcher["schema"] == SCHEMA
            and worker["execution_complete"] is False
            and worker["failure"] is not None
            and launcher["exit_code"] != 0 and launcher["failure"] is None
            and launcher["source_hash_mismatches"] == []
            and launcher["worker_exited"] is True
            and launcher["owned_worker_cleanup"]["no_orphans"] is True
            and launcher["fully_reserved_budget_closed"] is True
            and launcher["retry_permitted"] is worker["retry_permitted"] is False
            and worker["heldout_cases"] == []
            and worker["final_checkpoint_manifest"] is None,
            "T is not an honestly closed failed run without heldout")
    segment = _json(run / "training/training_segment_receipt.json")
    before = _json(run / "training/boundary_before.json")
    after = segment["boundary_after"]
    guard = segment["native_segment"]
    c, p, g = worker["C_final"], worker["python"], worker["native_guard"]
    n = p["control_completed"]
    attempted = p["control_attempted"]
    native = p["native_returned"]
    require(0 < n < 262144 and attempted == n + 1
            and native == p["native_attempted"] == p["clock_advanced_substeps"]
            == c["control_attempts"] == c["control_returns"]
            == g["native_returned"] == g["native_attempted"] == g["native_checked"]
            == guard["native_returned"] == guard["native_attempted"]
            == 5 * attempted
            and p["native_failed"] == p["forbidden_entries"] == c["violations"] == 0
            and p["fatal_latched"] is True and c["phase"] == c["target_model"]
            == c["target_data"] == 0 and c["construction_attempts"]
            == c["construction_returns"] == 2
            and c["ccd_attempts"] == c["ccd_returns"]
            and c["native_ccd_caller_verified"] is True
            and c["native_construction_caller_verified"] is True
            and worker["control_step_caller_verified_against_closed_E_callsite"] is True
            and g["failure"] is None and g["nonwheel_ground_contacts"] == 0
            and segment["completed_controls"] == segment["curriculum_completed_controls"]
            == segment["gaussian_records"] == n
            and segment["numeric_training_complete"] is False
            and segment["failure"] == worker["failure"],
            "failed T C/Python/guard actual completed/attempted boundary differs")
    for family in ("C_state", "python"):
        require(after[family] == worker["C_final" if family == "C_state" else "python"]
                and after[family]["control_returns" if family == "C_state" else
                                  "control_completed"] >= 0,
                "failed T segment/final C or Python snapshot differs")
    require(after["C_state"]["control_returns"]
            - before["C_state"]["control_returns"] == native
            and after["python"]["control_completed"]
            - before["python"]["control_completed"] == n
            and after["python"]["control_attempted"]
            - before["python"]["control_attempted"] == attempted,
            "failed T segment boundary does not expose extra attempted control")
    compact = _compact_prefix(run, n)
    native_proof = _native_prefix(run, n, native, guard)
    require(close(native_proof["max_abs_roll_deg"], g["max_abs_roll_deg"], atol=1e-9)
            and close(native_proof["max_abs_pitch_deg"], g["max_abs_pitch_deg"], atol=1e-9)
            and max(native_proof["max_abs_roll_deg"],
                    native_proof["max_abs_pitch_deg"]) <= 45,
            "failed T native posture summary differs from saved returns")
    failed = _json(run / "training/learning_failure_receipt.json")
    manifest = _json(run / "training/failure_checkpoint_manifest.json")
    metadata_path = run / "failure_checkpoint/failure_receipt.json"
    metadata = _json(metadata_path)
    require(failed["status"] == "incomplete"
            and failed["qualified_for_final_checkpoint"] is False
            and failed["actual"]["num_timesteps"] == n
            and failed["actual"]["transitions"] == n
            and failed["actual"]["train_calls"] == len(failed["updates"])
            == len(failed["rollouts"])
            and failed["actual"]["epochs"] == 4 * len(failed["updates"])
            and failed["actual"]["optimizer_steps"] == 16 * len(failed["updates"])
            and manifest["qualified_for_heldout"] is False
            and metadata["qualified_for_heldout"] is False
            and metadata["is_final_model"] is False
            and metadata["training_receipt"] == failed
            and identity(metadata_path)["sha256"] == manifest["metadata_sha256"]
            and manifest["folder"] == str(run / "failure_checkpoint"),
            "failed T learning receipt or failure checkpoint was promoted")
    for name, expected in manifest["files"].items():
        _file_matches(run / "failure_checkpoint" / name, expected)
    require(not (run / "final_checkpoint").exists()
            and not (run / "heldout").exists(),
            "failed T has unexpected final checkpoint or heldout output")
    result = {
        "schema": "d1-course-t-independent-partial-readback-v1",
        "run": str(run), "source_closure": source,
        "offline_reader_source": {"path": str(Path(__file__).resolve()),
                                  **identity(Path(__file__).resolve())},
        "actual_go_sha256": session["go_sha256"],
        "execution_complete": False, "contract_failed": True,
        "failure": worker["failure"],
        "launcher_exit_code": launcher["exit_code"],
        "fully_reserved_budget_closed": True,
        "completed_control_prefix": compact,
        "failed_attempt": {
            "control_index": n,
            "python_attempted_controls": attempted,
            "python_completed_controls": n,
            "C_native_attempted_and_returned": native,
            "extra_native_returns_after_completed_prefix": native - 5 * n,
            "native_evidence": native_proof,
            "missing_completed_control_trace": [
                "no compact numeric/reward row for failed control index",
                "no Gaussian completed-control row for failed control index",
                "no post-control Env transition or PPO rollout return for failed control index",
            ],
        },
        "learning_prefix": {"actual": failed["actual"],
                            "qualified_for_final_checkpoint": False,
                            "failure_checkpoint_qualified_for_heldout": False},
        "heldout_cases": [], "six_task_qualification_available": False,
        "RL_contribution_available": False,
        "training_compact_full_contact_force_qualification_available": False,
        "engine_or_policy_imported": False,
        "physics_replayed_or_rescored": False,
    }
    closed = _closed_manifest(run)
    closed["schema"] = "d1-course-t-failed-run-closed-file-manifest-v1"
    return result, closed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    require(args.run.is_absolute(), "T partial --run must be absolute")
    result, manifest = verify(args.run.resolve(strict=True))
    for target in (args.output, args.manifest_output):
        if target is not None:
            require(target.is_absolute() and not target.exists(),
                    "T partial readback output must be new and exclusive")
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    if args.manifest_output is not None:
        with args.manifest_output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
