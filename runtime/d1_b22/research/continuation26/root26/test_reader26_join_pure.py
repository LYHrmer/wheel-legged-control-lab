"""Reject extended deadlines or evidence produced before the actual owner ended."""
import pytest
from reader26_deferred import check_owner_join_metadata


def fixture():
    return ({"deadline_monotonic_s":215.,"join_start_monotonic_s":120.,
             "join_end_monotonic_s":131.,"owner_alive_after":False},
            {"monotonic_s":100.},{"close_s":115,"hard_s":120},
            {"flush_end_ns":130_900_000_000})


def test_closed_owner_uses_remaining_original_deadline():
    result=check_owner_join_metadata(*fixture())
    assert result["passed"] and result["join_wall_s"]==11.


@pytest.mark.parametrize("fault",["extended_from_join","alive","deadline_expired","before_flush","nonfinite"])
def test_concurrent_or_late_evidence_is_rejected(fault):
    join,ready,session,deferred=fixture()
    if fault=="extended_from_join":join["deadline_monotonic_s"]=235.
    elif fault=="alive":join["owner_alive_after"]=True
    elif fault=="deadline_expired":join["join_end_monotonic_s"]=216.
    elif fault=="before_flush":deferred["flush_end_ns"]=132_000_000_000
    else:join["join_end_monotonic_s"]=float("nan")
    with pytest.raises(ValueError):
        check_owner_join_metadata(join,ready,session,deferred)
