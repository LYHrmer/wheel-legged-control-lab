"""Saved-evidence fault checks; no renderer, model, servo, or physics execution."""
from copy import deepcopy
import pytest
from reader23 import input_replay, check_input_publication_links, stop_window


def sample(seq, now, held=(), events=(), focused=True):
    return dict(sequence=seq,timestamp_ns=now,held=list(held),focused=focused,
                closed=False,events=list(events),events_truncated=False)


def test_reader_repeated_fresh_input_then_expiry_requires_real_release():
    a=sample(0,1_000_000_000,('w',))
    b=sample(1,1_400_000_000,('w',))
    c=sample(2,1_410_000_000)
    d=sample(3,1_420_000_000,('w','q'))
    rows=[dict(snapshot=s,now_ns=t) for s,t in (
        (a,1_010_000_000),(a,1_020_000_000),(a,1_260_000_001),
        (b,1_400_000_001),(c,1_410_000_001),(d,1_420_000_001))]
    actual=input_replay('yaw_1p2',rows)
    assert [r['reason'] for r in actual]==['held_command','held_command','stale_snapshot',
        'release_w_to_rearm','released','held_command']
    assert actual[1]['reused_snapshot'] and actual[-1]['yaw_rps']==.3
    assert all(r['forward_mps']==0. for r in actual[2:5])


def test_reader_stop_precedes_reset_and_reset_edge_cannot_repeat():
    events=[{'kind':'press','key':'r'},{'kind':'press','key':'space'}]
    first=sample(0,100,('r','space'),events)
    second=sample(1,200,('r',),[{'kind':'press','key':'r'}])
    actual=input_replay('yaw_1p2',[dict(snapshot=first,now_ns=101),
        dict(snapshot=second,now_ns=201),dict(snapshot=second,now_ns=202)])
    assert actual[0]['reason']=='operator_stop' and not actual[0]['reset_requested']
    assert actual[1]['reset_requested'] and not actual[2]['reset_requested']


def test_reader_exit_has_authority_even_if_clock_is_old():
    s=sample(0,10,('escape',))
    assert input_replay('yaw_1p2',[dict(snapshot=s,now_ns=999_999_999)])[0]['exit_requested']


def test_reader_pending_short_stop_cannot_be_lost_between_polls():
    a=sample(0,100,(),[{'kind':'press','key':'space'}])
    b=sample(1,200)
    polls=[dict(snapshot=s,published=True,publish_begin_ns=s['timestamp_ns'],
                publish_wall_ns=s['timestamp_ns']+1) for s in (a,b)]
    delivered=deepcopy(b)
    delivered['events']=a['events']
    row=dict(snapshot=delivered,now_ns=201)
    assert check_input_publication_links(polls,[row])['pending_edges_preserved']
    bad=deepcopy(row)
    bad['snapshot']['events']=[]
    with pytest.raises(ValueError,match='lost or invented'):
        check_input_publication_links(polls,[bad])


def test_reader_stop_claim_needs_full_actual_zero_command_window():
    row=dict(raw_command=dict(forward_velocity_mps=0.,yaw_rate_rps=0.),
        info=dict(consumed_command=dict(forward_velocity_mps=0.,yaw_rate_rps=0.),
                  metrics=dict(body_com_vx_mps=.01,body_yaw_rate_rps=.01,clearance_m=.455)))
    assert stop_window([deepcopy(row) for _ in range(100)])['passed']
    assert not stop_window([deepcopy(row) for _ in range(99)])['passed']
    bad=[deepcopy(row) for _ in range(100)]
    bad[0]['info']['consumed_command']['forward_velocity_mps']=.005
    assert not stop_window(bad)['passed']
