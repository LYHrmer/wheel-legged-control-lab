"""Offline readback of the closed, timed-out 11-S run; no engine or model import.

Verifies complete training and the first nine full heldout cases. The tenth
case's interrupted files are retained by digest only; no missing rows or
unstarted ramp episodes are projected into success or RL contribution.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import verify_short_rl16_training_11 as full

HERE = Path(__file__).resolve().parent
FULL_READER_SHA = "cec7b847f99cb1eaf8150e03b6498731e1b431cec9026f3f7f8518b2f0fe685d"
PREFIX_CASES = tuple((spec, actor) for spec in full.CASES[:4]
                     for actor in ("zero", "final_policy")) + (
                         (full.CASES[4], "zero"),
                     )


def source(run: Path) -> tuple[dict, dict, dict, dict]:
    full.require(full.identity(HERE / "verify_short_rl16_training_11.py")["sha256"]
                 == FULL_READER_SHA, "frozen pure full-reader helper changed")
    session = full.document(run / "session.json")
    host = full.document(run / "launcher_receipt.json")
    worker = full.document(run / "worker_receipt.json")
    full.require(session["schema"] == host["schema"] == worker["schema"] == full.SCHEMA
                 and session["phase"] == "T11"
                 and session["output_directory"] == str(run)
                 and session["contract_sha256"] == full.CONTRACT_SHA
                 and session["control_limit"] == 85136
                 and session["normal_native_limit"] == 425680
                 and session["compiler_native_limit"] == 2
                 and session["wallclock_limit_s"] == 1800
                 and session["training_control_limit"] == 65536
                 and session["heldout_control_limit"] == 19600
                 and session["segments"] == [65536] + [1600] * 10 + [1800] * 2
                 and session["ppo_seed"] == 88621
                 and session["command_seed"] == 88622
                 and session["measurement_seed"] == 88623
                 and session["retry_permitted"] is False,
                 "11-S partial session/contract/budget identity differs")
    frozen = session["source_hashes"]
    full.require(isinstance(frozen, dict) and len(frozen) >= 100,
                 "11-S actual input freeze absent")
    for name, expected in frozen.items():
        full.require(Path(name).is_absolute(), "11-S frozen input path not absolute")
        full.match_file(Path(name), expected)
    full.require(full.identity(Path(session["contract_path"]))["sha256"]
                 == full.CONTRACT_SHA, "11-S contract source changed")
    go_path = HERE / "astra_short_rl_go_11.json"
    go = full.document(go_path)
    full.require(full.identity(go_path)["sha256"] == session["go_sha256"]
                 and go["decision"] == "GO"
                 and go["contract_sha256"] == full.CONTRACT_SHA,
                 "11-S actual signed GO changed")
    for name, expected in go["inputs"].items():
        full.require(frozen.get(name) == expected,
                     "11-S GO input missing from actual source freeze")
        full.match_file(Path(name), expected)
    full.require(host["exit_code"] == -15
                 and host["failure"] == {"type": "TimeoutError",
                                        "message": "host 1800s T watchdog expired"}
                 and host["source_hash_mismatches"] == []
                 and host["worker_exited"] is True
                 and host["owned_worker_cleanup"]["no_orphans"] is True
                 and host["fully_reserved_budget_closed"] is True
                 and host["retry_permitted"] is False
                 and worker["execution_complete"] is False
                 and worker["failure"]["type"] == "FileExistsError"
                 and worker["warnings"] == []
                 and worker["retry_permitted"] is False,
                 "11-S host/worker did not close as the recorded timeout failure")
    origins = full.document(run / "module_origins.json")
    full.require(origins["engine_binding"] == session["binding_module"]
                 and all(path in frozen for path in origins.values()),
                 "11-S actual imported course/binding source not frozen")
    for name in ("training/learning_dependency_origins_before_physics.json",
                 "learning_dependency_origins_before_heldout.json"):
        dependencies = full.document(run / name)
        full.require(all(path in frozen for path in
                         dependencies["module_origins"].values())
                     and set(dependencies["synthetic_module_aliases"])
                     == {"torch.ops", "torch.classes"}
                     and all(alias["synthetic_module_alias"] is True
                             and alias["identity_verified"] is True
                             and alias["defining_source"] in frozen
                             and alias["defining_source"]
                             == dependencies["module_origins"][name]
                             for name, alias in
                             dependencies["synthetic_module_aliases"].items()),
                     "11-S actual learning dependency origin not frozen")
    return session, host, worker, {
        "frozen_source_count": len(frozen), "go_input_count": len(go["inputs"]),
        "signed_go_sha256": session["go_sha256"], "full_pure_helper_sha256": FULL_READER_SHA,
    }


def completed_prefix(run: Path, worker: dict, construction: dict) -> tuple[list[dict], list[dict]]:
    binding = full.check_binding(construction["actual_geometry_binding"])
    summaries, cases = [], []
    for spec, actor in PREFIX_CASES:
        row, canonical = full.heldout_case(
            run, spec, actor, binding, construction["compiled_geometry"])
        summaries.append(row)
        cases.append(full.score_case(canonical))
    full.require(len(summaries) == len(worker["heldout_cases"]) == 9
                 and worker["heldout_cases"] == [
                     {key: full.document(run / "heldout"
                           / f"{row['case_id']}_{row['actor']}"
                           / "case_receipt.json")[key]
                      for key in ("case_id", "actor", "seed", "terrain", "control_cap",
                                  "completed_controls", "policy_prediction_attempts",
                                  "policy_predictions", "record_valid", "terminated",
                                  "truncated", "stop_reason")}
                     for row in summaries],
                 "11-S worker completed-prefix summary differs from nine raw cases")
    for index in range(4):
        zero, policy = summaries[2*index:2*index+2]
        spec = full.CASES[index]
        full.require(zero["case_id"] == policy["case_id"] == spec[0]
                     and (zero["actor"], policy["actor"]) == ("zero", "final_policy")
                     and zero["actual_model_address"] == policy["actual_model_address"]
                     and zero["actual_data_address"] == policy["actual_data_address"],
                     "11-S four completed pair identities differ")
        first, second = (run / "heldout" / f"{spec[0]}_{actor}"
                         for actor in ("zero", "final_policy"))
        a = full.arrays(first / "initial_state.npz")
        b = full.arrays(second / "initial_state.npz")
        full.require(all(a[key].dtype == b[key].dtype
                         and a[key].shape == b[key].shape
                         and a[key].tobytes() == b[key].tobytes()
                         for key in full.PAIR_FIELDS),
                     "11-S completed pair initial state differs bitwise")
    return summaries, cases


def damaged_tenth(run: Path, worker: dict) -> dict:
    folder = run / "heldout/rough_0p35_final_policy"
    full.require(folder.is_dir() and not (folder / "case_receipt.json").exists()
                 and not (folder / "states.npz").exists()
                 and not (run / "heldout/ramp_0p45_complete_zero").exists()
                 and not (run / "heldout/ramp_0p45_complete_final_policy").exists(),
                 "11-S partial or unstarted heldout boundary differs")
    segment = worker["python"]["segments"][-1]
    full.require(segment["name"] == "heldout_rough_0p35_final_policy"
                 and segment["limit"] == 1600
                 and segment["attempted"] == 1024
                 and segment["completed"] == 1023
                 and segment["native_attempted"] == segment["native_returned"] == 5120,
                 "11-S tenth segment physical prefix/accounting differs")
    actual_files = {}
    for path in sorted(folder.iterdir()):
        full.require(path.is_file() and not path.is_symlink(),
                     "11-S tenth partial record contains nonregular path")
        actual_files[path.name] = full.identity(path)
    full.require("native_block_0000.jsonl.gz" in actual_files
                 and "initial_state.npz" in actual_files
                 and "reset_receipt.json" in actual_files,
                 "11-S damaged tenth provenance files absent")
    readable_rows = 0
    damaged_or_unsealed: list[str] = []
    for name in sorted(actual_files):
        if name.startswith("control_records_") and name.endswith(".jsonl.gz"):
            try:
                with gzip.open(folder / name, "rt", encoding="utf-8") as stream:
                    readable_rows += sum(1 for line in stream if json.loads(line))
            except (OSError, EOFError, UnicodeError, ValueError, json.JSONDecodeError):
                damaged_or_unsealed.append(name)
    return {
        "case_id": "rough_0p35", "actor": "final_policy",
        "record_valid": False, "qualification_possible": False,
        "actual_control_attempted": 1024, "actual_control_completed": 1023,
        "actual_native_returned": 5120,
        "readable_control_rows_without_endpoint_or_case_receipt": readable_rows,
        "damaged_or_unsealed_control_blocks": damaged_or_unsealed,
        "raw_files_preserved_unmodified": actual_files,
        "native_block_is_hashed_not_repaired_or_qualified": True,
        "unstarted_ramp_actors": ["zero", "final_policy"],
    }


def verify(run: Path) -> tuple[dict, dict]:
    full.require(run.is_absolute() and run.is_dir(),
                 "11-S partial --run must name absolute saved run")
    session, host, worker, provenance = source(run)
    initial = full.document(run / "runtime_initial.json")
    construction = full.document(run / "construction_receipt.json")
    full.require(initial["proof"]["passed"] is True
                 and initial["proof"]["dso_path"] == session["library"]
                 and initial["proof"]["dso_sha256"]
                 == session["source_hashes"][session["library"]]["sha256"]
                 and len(initial["proof"]["jump_slots"]) == 4
                 and all(slot["passed"] is True for slot in initial["proof"]["jump_slots"])
                 and construction["C_state"]["construction_attempts"]
                 == construction["C_state"]["construction_returns"] == 2
                 and construction["nominal_cache"]["misses"] == 1,
                 "11-S partial actual DSO/GOT/cold2 provenance differs")
    training = full.training(run)
    full.require(training["controls"] == 65536
                 and worker["training_guard_segment"] == full.document(
                     run / "training/training_segment_receipt.json")["native_segment"],
                 "11-S complete training source/guard differs")
    learning = full.document(run / "training/learning_receipt.json")
    checkpoint = full.checkpoint(run, learning, session)
    mode = full.document(run / "heldout_mode_transition.json")
    pair = [construction["course_plant_model_address"],
            construction["course_plant_data_address"]]
    full.require(mode["from"] == "train" and mode["to"] == "eval"
                 and mode["same_compiled_model_data_pair"] == pair
                 and mode["numerical_controller_unchanged"] is True,
                 "11-S partial heldout used another plant/controller")
    prefix, individual_scores = completed_prefix(run, worker, construction)
    full.require(all([row["actual_model_address"], row["actual_data_address"]]
                     == pair for row in prefix),
                 "11-S completed cases switched actual model/data")
    tenth = damaged_tenth(run, worker)
    c, py, guard = worker["C_final"], worker["python"], worker["native_guard"]
    completed = 65536 + 9 * 1600 + 1023
    attempted = completed + 1
    returned = 5 * attempted
    full.require(py["control_attempted"] == attempted
                 and py["control_completed"] == completed
                 and py["native_attempted"] == py["native_returned"] == returned
                 and py["native_failed"] == 0
                 and py["clock_advanced_substeps"] == returned
                 and py["fatal_latched"] is True
                 and c["control_attempts"] == c["control_returns"] == returned
                 and c["construction_attempts"] == c["construction_returns"] == 2
                 and c["ccd_attempts"] == c["ccd_returns"]
                 and c["native_ccd_caller_verified"] is True
                 and c["native_construction_caller_verified"] is True
                 and c["phase"] == c["target_model"] == c["target_data"] == 0
                 and c["violations"] == py["forbidden_entries"] == 0
                 and guard["native_attempted"] == guard["native_returned"]
                 == guard["native_checked"] == returned
                 and guard["failure"] is None
                 and worker["control_step_caller_verified_against_closed_E_callsite"]
                 is True,
                 "11-S closed partial C/Python/guard counts or caller differ")
    full.require(worker["actual_model_calls"]["learn"] == 1
                 and worker["actual_model_calls"]["save"] == 1
                 and worker["actual_model_calls"]["load"] == 2
                 and worker["actual_model_calls"]["predict"] == 7428
                 and worker["actual_prediction_calls_by_phase"]
                 ["training_and_finalization"] == 4
                 and worker["actual_prediction_calls_by_phase"]
                 ["heldout:rough_0p35:final_policy"] == 1024
                 and len(worker["actual_final_probe_batches"]) == 4
                 and all(row["shape"] == [32, 99]
                         and row["deterministic"] is True
                         for row in worker["actual_final_probe_batches"])
                 and all(worker["actual_prediction_calls_by_phase"].get(
                     "heldout:" + row["case_id"] + ":" + row["actor"], 0)
                     == row["predict_calls"] for row in prefix),
                 "11-S actual final/probe/partial policy call count differs")
    result = {
        "schema": "d1-world-upright-short-rl16-partial-readback-11-v1",
        "run": str(run), "source_closure": provenance,
        "offline_reader_source": {"path": str(Path(__file__).resolve()),
                                  **full.identity(Path(__file__).resolve())},
        "host_exit_code": host["exit_code"], "host_failure": host["failure"],
        "worker_failure": worker["failure"],
        "budget_reserved_and_closed_without_retry": True,
        "training_complete": True, "training": training,
        "unique_final_checkpoint_reload_saved_evidence": checkpoint,
        "complete_full_native_heldout_cases_verified": prefix,
        "individual_complete_case_scores_not_matched_pair_aggregate": individual_scores,
        "complete_case_count": 9, "bitwise_initial_pairs_verified": 4,
        "partial_tenth_case": tenth,
        "ramp_heldouts_started": False,
        "all_twelve_heldout_verified": False,
        "all_six_policy_task_qualified": False,
        "RL_contribution_passed": False,
        "RL_contribution_evaluation_complete": False,
        "qualified_for_default_GUI": False,
        "normal_native_returned_including_unreturned_outer_control": returned,
        "compiler_native_returned": 2,
        "engine_model_policy_or_learner_imported": False,
        "physics_replayed_or_rescored": False,
        "training_compact_full_contact_force_qualified": False,
    }
    return result, full.manifest(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    full.require(args.run.is_absolute(), "11-S partial --run must be absolute")
    result, closed = verify(args.run.resolve(strict=True))
    for path in (args.output, args.manifest_output):
        if path is not None:
            full.require(path.is_absolute() and not path.exists(),
                         "11-S partial readback output must be new absolute path")
    if args.output is None:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    if args.manifest_output is not None:
        with args.manifest_output.open("x", encoding="utf-8") as stream:
            json.dump(closed, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
