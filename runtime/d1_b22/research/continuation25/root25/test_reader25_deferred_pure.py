"""Reader-side rejection of pending, early or physically active archive drains."""
import copy
import pytest
from reader25_deferred import check_deferred_metadata


def fixture():
    boundary={"C_state":{"control_attempts":15,"control_returns":15,
                         "construction_attempts":2,"construction_returns":2},
              "python":{"control_attempted":3,"control_completed":3,
                        "native_attempted":15,"native_returned":15}}
    result={"owner_thread_ident":7,"active_start_ns":100,"active_end_ns":200}
    receipt={"schema":"d1-c25-deferred-archive-v1","failure":None,"pending_count":0,
             "owner_thread_ident":7,"active_start_ns":100,"active_end_ns":200,
             "defer_begin_ns":99,"flush_start_ns":210,"flush_end_ns":300,
             "max_jobs":64,"peak_jobs":2,"control_rows":3,"native_rows":15,"episode_count":1,
             "queued_jobs":[{"id":0,"kind":"native","manifest_name":"a.manifest.json",
                             "enqueued_ns":150,"archive_status":"pending"},
                            {"id":1,"kind":"controls","manifest_name":"b.manifest.json",
                             "enqueued_ns":205,"archive_status":"pending"}],
             "flushed_jobs":[{"id":0,"manifest_name":"a.manifest.json","completed_ns":240},
                             {"id":1,"manifest_name":"b.manifest.json","completed_ns":290}],
             "boundary_before_flush":copy.deepcopy(boundary),"boundary_after_flush":copy.deepcopy(boundary)}
    return receipt,result


def test_complete_finite_fifo_metadata_is_accepted():
    receipt,result=fixture()
    check_deferred_metadata(receipt,result,3,1,1200)


@pytest.mark.parametrize("fault",["pending","before_active_end","physical_step","out_of_order","false_durable","row_loss"])
def test_unclosed_or_misrepresented_archive_is_rejected(fault):
    receipt,result=fixture()
    if fault=="pending":receipt["pending_count"]=1
    elif fault=="before_active_end":receipt["flush_start_ns"]=190
    elif fault=="physical_step":receipt["boundary_after_flush"]["python"]["native_returned"]=16
    elif fault=="out_of_order":receipt["flushed_jobs"].reverse()
    elif fault=="false_durable":receipt["queued_jobs"][0]["archive_status"]="durable"
    elif fault=="row_loss":receipt["native_rows"]=10
    with pytest.raises(ValueError):
        check_deferred_metadata(receipt,result,3,1,1200)


def semantic_fixture():
    # Synthetic new-contract packet labels over saved callback shapes only.
    # Full execution/source verification separately rejects the old C23 run.
    import gzip
    import json
    from pathlib import Path
    run=Path(__file__).resolve().parents[2]/"continuation23/keyboard_gui_01"
    driver=json.loads((run/"x11_events_receipt.json").read_text())
    driver["events"][-1]["tick"]=1100
    with gzip.open(run/"input_snapshots.jsonl.gz","rt") as stream:
        rows=[json.loads(line) for line in stream]
    generic={"path_checks":{"independent_paths":True},"decision_reasons":["released","focus_lost"]}
    return driver,rows,generic


def test_semantic_contract_counts26_callbacks_and_one_bookkeeping_request():
    from reader25_deferred import check_c25_input_semantics
    driver,rows,generic=semantic_fixture()
    result=check_c25_input_semantics("C25_B22_GUI_reset_archive_v1",driver,rows,generic)
    assert result["passed"] is True
    assert result["actual_callbacks_matched"]==26
    assert result["OS_bookkeeping_injections"]==1
    assert result["bookkeeping"]["event"]["tick"]==510


def test_new_semantics_cannot_relabel_an_old_execution_contract():
    from reader25_deferred import check_c25_input_semantics
    with pytest.raises(ValueError,match="cannot relabel"):
        check_c25_input_semantics("C23_B22_GUI_v1",None,None,None)


@pytest.mark.parametrize("fault",["missing_auto_release","no_focused_release","failed_control_path","added505"])
def test_bookkeeping_exception_requires_the_entire_exact_focus_rearm_chain(fault):
    from reader25_deferred import check_c25_input_semantics
    driver,rows,generic=semantic_fixture()
    if fault=="missing_auto_release":
        for row in rows:
            row["poll"]["events"]=[e for e in row["poll"]["events"]
                if e.get("wall_ns")!=95377275608873]
    elif fault=="no_focused_release":
        for row in rows:
            if row["snapshot"]["focused"] and "w" not in row["snapshot"]["held"]:
                row["snapshot"]["held"]=["w"]
    elif fault=="failed_control_path":generic["path_checks"]["independent_paths"]=False
    else:
        driver["events"].insert(14,{"tick":505,"operation":"key_down","key":"w","wall_ns":0})
    if fault=="added505":
        with pytest.raises(ValueError):
            check_c25_input_semantics("C25_B22_GUI_reset_archive_v1",driver,rows,generic)
    else:
        result=check_c25_input_semantics("C25_B22_GUI_reset_archive_v1",driver,rows,generic)
        assert result["passed"] is False
