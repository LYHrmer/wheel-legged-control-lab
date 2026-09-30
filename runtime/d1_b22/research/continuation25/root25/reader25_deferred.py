"""Saved-only C25 readback with mandatory deferred-to-durable archive closure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import reader25 as base


def check_deferred_metadata(receipt, result, controls, episodes, cap):
    need = base.require
    need(receipt["schema"] == "d1-c25-deferred-archive-v1"
         and receipt["failure"] is None and receipt["pending_count"] == 0,
         "deferred archive has not closed successfully")
    need(receipt["owner_thread_ident"] == result["owner_thread_ident"]
         and receipt["active_start_ns"] == result["active_start_ns"]
         and receipt["active_end_ns"] == result["active_end_ns"],
         "deferred archive owner or actual active clock differs")
    need(0 < receipt["defer_begin_ns"] <= receipt["flush_start_ns"]
         and receipt["active_start_ns"] < receipt["active_end_ns"]
         <= receipt["flush_start_ns"] <= receipt["flush_end_ns"],
         "archive flush precedes actual active end")
    queued, flushed = receipt["queued_jobs"], receipt["flushed_jobs"]
    need(receipt["max_jobs"] == 64 and 1 <= len(queued) == receipt["peak_jobs"] <= 64
         and len(flushed) == len(queued), "finite deferred FIFO job ledger differs")
    ids = [r["id"] for r in queued]
    need(all(type(i) is int for i in ids) and len(set(ids)) == len(ids)
         and ids == sorted(ids) and [r["id"] for r in flushed] == ids,
         "deferred jobs were duplicated or flushed out of FIFO order")
    need(receipt["control_rows"] == controls <= cap <= 6000
         and receipt["native_rows"] == 5*controls <= 5*cap
         and receipt["episode_count"] == episodes and 1 <= episodes <= 2,
         "deferred finite control/native/episode row accounting differs")
    last_enqueue = receipt["defer_begin_ns"]
    last_flush = receipt["flush_start_ns"]
    for queued_job, flushed_job in zip(queued,flushed):
        need(queued_job["archive_status"] == "pending"
             and isinstance(queued_job["kind"],str) and queued_job["kind"]
             and last_enqueue <= queued_job["enqueued_ns"] <= receipt["flush_start_ns"],
             "queued archive claimed durability or has invalid ordering")
        need(flushed_job["manifest_name"] == queued_job["manifest_name"]
             and last_flush <= flushed_job["completed_ns"] <= receipt["flush_end_ns"],
             "actual durable completion differs from queued transaction")
        last_enqueue = queued_job["enqueued_ns"]
        last_flush = flushed_job["completed_ns"]
    for section, fields in (("C_state",("control_attempts","control_returns",
                                           "construction_attempts","construction_returns")),
                            ("python",("control_attempted","control_completed",
                                        "native_attempted","native_returned"))):
        before,after = receipt["boundary_before_flush"][section],receipt["boundary_after_flush"][section]
        need(all(before[k] == after[k] for k in fields), "deferred flush performed extra physical work")
    c = receipt["boundary_after_flush"]["C_state"]
    py = receipt["boundary_after_flush"]["python"]
    need(c["control_attempts"] == c["control_returns"] == py["native_attempted"]
         == py["native_returned"] == 5*controls
         and c["construction_attempts"] == c["construction_returns"] == 2
         and py["control_attempted"] == py["control_completed"] == controls,
         "deferred flush boundary does not close the same actual physical ledger")


def check_c25_input_semantics(contract_id, driver, raw_rows, generic):
    need=base.require
    need(contract_id == "C25_B22_GUI_reset_archive_v1",
         "new OS bookkeeping semantics cannot relabel an old C23 contract")
    sent=[e for e in driver["events"] if e["operation"] != "initial_focus"]
    ticks=[100,180,200,230,250,280,350,400,405,430,450,470,500,510,520,590,
           610,650,660,720,725,760,800,850,855,860,1100]
    need(len(sent)==27 and [e["tick"] for e in sent] == ticks and driver["all_sent"] is True,
         "C25 predeclared27 OS injection plan differs")
    cleanup=sent[13]
    need(cleanup["operation"] == "key_up" and cleanup.get("key") == "w",
         "only the exact510 W release may be OS bookkeeping")
    callbacks=[]
    for row in raw_rows:
        for event in row["poll"]["events"]:
            if event["type"] == "key" and event["action"] in (0,1):
                callbacks.append({"operation":"key_down" if event["action"] == 1 else "key_up",
                                  "key":base.GLFW_KEYS.get(event["key"],"unsupported"),
                                  "wall_ns":event["wall_ns"]})
            elif event["type"] == "focus":
                callbacks.append({"operation":"focus_return" if event["focused"] else "focus_lost",
                                  "wall_ns":event["wall_ns"]})
        callbacks.extend({"operation":"button_click","button":"stop" if e["key"] == "space" else "reset",
                          "wall_ns":e["wall_ns"]} for e in row["raw_button_events"])
    meaningful=[e for i,e in enumerate(sent) if i != 13]
    matching=base.ordered_callback_matches(meaningful,callbacks)
    by_tick={r["planned_tick"]:r for r in matching["matches"]}
    anchors=all(t in by_tick for t in (450,470,500,520))
    auto=[]
    release_snapshot=False
    if anchors:
        lost=by_tick[470]["callback_wall_ns"]
        gained=by_tick[500]["callback_wall_ns"]
        next_press=by_tick[520]["callback_wall_ns"]
        auto=[e for e in callbacks if e["operation"] == "key_up" and e.get("key") == "w"
              and lost <= e["wall_ns"] <= gained]
        release_snapshot=any(gained <= r["snapshot"]["timestamp_ns"] <= next_press
                             and r["snapshot"]["focused"] is True
                             and "w" not in r["snapshot"]["held"] for r in raw_rows)
    checks={"all26_meaningful_callbacks":matching["passed"],
            "all_original_input_paths":all(generic["path_checks"].values()),
            "focus_and_press_anchors_observed":anchors,
            "focus_loss_auto_W_release_observed":bool(auto),
            "fresh_focused_released_snapshot_before_new_press":release_snapshot,
            "OS_cleanup_between_focus_return_request_and_new_press_request":
                sent[12]["wall_ns"] <= cleanup["wall_ns"] <= sent[14]["wall_ns"],
            "W_was_observed_pressed_before_focus_loss":bool(anchors and
                by_tick[450]["callback_wall_ns"] < by_tick[470]["callback_wall_ns"]
                < by_tick[500]["callback_wall_ns"] < by_tick[520]["callback_wall_ns"])}
    return {"passed":all(checks.values()),"preregistered_OS_injections":27,
            "meaningful_callbacks_required":26,"actual_callbacks_matched":len(matching["matches"]),
            "OS_bookkeeping_injections":1,"matching":matching,"checks":checks,
            "path_checks":generic["path_checks"],"decision_reasons":generic["decision_reasons"],
            "bookkeeping":{"event":cleanup,"duplicate_GLFW_release_required":False,
                           "actual_focus_loss_auto_releases":auto,
                           "evidence_kind":"frozen_successful_XTest_plus_XSync_record_and_actual_GLFW_rearm_chain",
                           "direct_Xserver_keymap_query_performed":False},
            "physical_human_keyboard_tested":False,
            "old_C23_all_packets_callback_contract_relabelled":False}


def verify_deferred(actual, run):
    session,report = actual["session"],actual["report"]
    base.require(session["execution_contract_id"] == "C25_B22_GUI_reset_archive_v1"
                 and session["arm"] == "keyboard_gui", "C25 wrapper is only its new keyboard contract")
    for source in (Path(__file__).resolve(),Path(base.__file__).resolve()):
        base.require(session["source_hashes"].get(str(source)) == base.identity(source),
                     "actual base/wrapper reader source is not frozen in new execution closure")
    worker = base.saved_document(run/"worker_receipt.json")
    receipt_path = run/"deferred_archive_receipt.json"
    receipt = base.saved_document(receipt_path)
    check_deferred_metadata(receipt,worker["result"],report["completed_controls"],
                            report["actual_episode_count"],session["control_limit"])
    all_payloads = set()
    for queued,flushed in zip(receipt["queued_jobs"],receipt["flushed_jobs"]):
        paths = [Path(p) for p in queued["payload_paths"]]
        base.require(paths and len(set(paths)) == len(paths)
                     and all(p.is_absolute() and p.is_relative_to(run) and not p.is_symlink() for p in paths)
                     and all(p.parent == paths[0].parent for p in paths),
                     "queued transaction payload paths are unsafe or ambiguous")
        manifest = paths[0].with_name(paths[0].name+".manifest.json")
        base.require(queued["manifest_name"] == manifest.name
                     and base.identity(manifest) == flushed["actual_manifest_identity"],
                     "actual durable manifest is not the queued transaction identity")
        record = base.committed(paths[0])
        base.require([r["file"] for r in record["payloads"]] == [p.name for p in paths],
                     "durable transaction payload membership differs")
        base.require(not (all_payloads & set(paths)), "a deferred payload was committed twice")
        all_payloads.update(paths)
        base.require(all(str(p) in worker["writer_committed"] for p in (*paths,manifest)),
                     "durable transaction absent from final writer committed ledger")
    required = set()
    for index,episode in enumerate(actual["episodes"]):
        folder=run/f"episode_{index}"
        required.update(folder/name for name in ("controls.jsonl.gz","states.npz","segment_receipt.json"))
        segment=base.saved_document(folder/"segment_receipt.json")
        required.update(folder/name for name in segment["native_segment"]["native_files"])
        if index:
            required.update(folder/name for name in ("reset.json","initial_state.npz"))
    base.require(required <= all_payloads, "an active or final episode archive bypassed the deferred FIFO")
    report["deferred_archive"]={"passed":True,"receipt_identity":base.identity(receipt_path),
                                "all_jobs_durable":True,"pending_count":0,
                                "jobs":len(receipt["queued_jobs"]),"payloads":len(all_payloads),
                                "flush_wall_s":(receipt["flush_end_ns"]-receipt["flush_start_ns"])/1e9,
                                "flush_started_after_actual_active_end":True,
                                "unchanged_physical_boundary_verified":True}
    generic_events=report["real_events"]
    semantic=check_c25_input_semantics(session["execution_contract_id"],
                base.document(run/"x11_events_receipt.json"),
                list(base.saved_rows(run/"input_snapshots.jsonl.gz")),generic_events)
    report["all_packets_callback_diagnostic"]=generic_events
    report["real_events"]=semantic
    passed=bool(report["physical_safety_passed"] and report["actual_nonzero_B22_residual"]
                and report["render"]["passed"] and report["last100_stop"]["passed"]
                and semantic["passed"])
    report["arm_qualification_passed"]=passed
    report["execution_prerequisite_passed"]=passed
    report["original_v1_arm_qualification_is_current_C25_gate"]=False
    report["schema"]="d1-c25-deferred-independent-readback-v1"
    report["base_reader_source_identity"]=report["reader_source_identity"]
    report["reader_source_identity"]=base.identity(Path(__file__).resolve())
    report["old_C23_script_pair_inherited_not_rerun"]=True
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    actual=base.read_run_details(run)
    report=verify_deferred(actual,run)
    with args.output.open("x") as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    return 0 if report["execution_prerequisite_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
