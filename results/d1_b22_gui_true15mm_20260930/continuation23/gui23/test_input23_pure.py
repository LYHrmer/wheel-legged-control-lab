"""Pure GUI23 authority fixtures; no GLFW, model or physics import."""
from dataclasses import replace

import pytest

from input23 import Event23, InputController23, LatestInput23, Snapshot23


def snap(seq, ms, keys=(), *, focused=True, closed=False, events=(), truncated=False):
    return Snapshot23(seq,int(ms*1e6),frozenset(keys),focused,closed,
                      tuple(events),truncated)


def test_owner_100hz_reuses_60hz_poll_without_false_stale():
    control = InputController23('flat_0p6',ttl_s=.25)
    for owner_tick in range(100):
        poll = owner_tick*60//100
        sample = snap(poll,poll*1000/60,('w',))
        output = control.update(sample,now_ns=owner_tick*10_000_000)
        assert output.forward_mps==.6
        assert output.reset_requested is False
        assert output.reused_snapshot is (owner_tick*60//100==
                                          (owner_tick-1)*60//100 if owner_tick else False)


def test_ttl_expiry_stops_and_new_w_release_rearms():
    control = InputController23('yaw_1p2')
    initial = snap(0,0,('w','q'))
    assert control.update(initial,now_ns=0).yaw_rps==.3
    assert control.update(initial,now_ns=249_000_000).forward_mps==1.2
    assert control.update(initial,now_ns=251_000_000).reason=='stale_snapshot'
    assert control.update(snap(1,260,('w',)),now_ns=260_000_000).reason=='release_w_to_rearm'
    assert control.update(snap(2,270,()),now_ns=270_000_000).reason=='released'
    assert control.update(snap(3,280,('w','e')),now_ns=280_000_000).yaw_rps==-.3


def test_short_reset_edge_is_one_shot_across_repeated_poll_and_reset():
    control = InputController23('flat_0p6')
    edge = snap(0,0,('w',),events=(Event23('press','r'),Event23('release','r')))
    first = control.update(edge,now_ns=0)
    assert first.reset_requested and first.forward_mps==0.
    assert not control.update(edge,now_ns=10_000_000).reset_requested
    assert control.update(snap(1,20,('w',)),now_ns=20_000_000).forward_mps==0.
    control.update(snap(2,30,()),now_ns=30_000_000)
    assert control.update(snap(3,40,('w',)),now_ns=40_000_000).forward_mps==.6
    second = control.update(snap(4,50,('w','r'),events=(Event23('press','r'),)),
                            now_ns=50_000_000)
    assert second.reason=='reset_limit' and not second.reset_requested


def test_focus_loss_short_stop_unsupported_and_exit_latch():
    control = InputController23('flat_1p6')
    assert control.update(snap(0,0,('w',)),now_ns=0).forward_mps==1.6
    blur = snap(1,10,('w',),events=(Event23('focus_lost'),Event23('focus_gained')))
    assert control.update(blur,now_ns=10_000_000).reason=='focus_lost'
    assert control.update(snap(2,20,('w',)),now_ns=20_000_000).reason=='release_w_to_rearm'
    control.update(snap(3,30,()),now_ns=30_000_000)
    assert control.update(snap(4,40,('w',)),now_ns=40_000_000).forward_mps==1.6
    assert control.update(snap(5,50,('w',),events=(Event23('press','space'),)),
                          now_ns=50_000_000).reason=='operator_stop'
    assert control.update(snap(6,60,('w','a')),now_ns=60_000_000).reason=='unsupported_key'
    assert control.update(snap(5,50,('w',),closed=True),now_ns=70_000_000).exit_requested
    assert control.update(snap(7,80,('w',)),now_ns=80_000_000).reason=='exit_latched'


def test_same_sequence_mutation_and_input_mailbox_order_rejected():
    control = InputController23('flat_0p6')
    first = snap(0,0,('w',))
    assert control.update(first,now_ns=0).forward_mps==.6
    assert control.update(replace(first,held=frozenset()),now_ns=10_000_000).reason=='same_sequence_changed'
    mailbox = LatestInput23()
    mailbox.publish(first)
    assert mailbox.latest()==first
    with pytest.raises(ValueError):
        mailbox.publish(first)
    mailbox.publish(snap(1,20,()))
    assert mailbox.latest().sequence==1


def test_short_stop_reset_and_blur_survive_multiple_polls_before_owner_read():
    mailbox = LatestInput23()
    control = InputController23('flat_0p6')
    mailbox.publish(snap(0,0,('w',)))
    assert control.update(mailbox.latest(),now_ns=0).forward_mps==.6
    mailbox.publish(snap(1,10,('w',),events=(Event23('press','space'),)))
    mailbox.publish(snap(2,15,('w',),events=(Event23('release','space'),)))
    mailbox.publish(snap(3,20,('w',)))
    delivered = mailbox.latest()
    assert [event.kind for event in delivered.events]==['press','release']
    assert control.update(delivered,now_ns=20_000_000).reason=='operator_stop'
    assert control.update(mailbox.latest(),now_ns=30_000_000).reason=='release_w_to_rearm'
    mailbox.publish(snap(4,40,()))
    control.update(mailbox.latest(),now_ns=40_000_000)
    mailbox.publish(snap(5,50,('w',),events=(Event23('focus_lost'),)))
    mailbox.publish(snap(6,55,('w',),events=(Event23('focus_gained'),)))
    assert control.update(mailbox.latest(),now_ns=60_000_000).reason=='focus_lost'
    mailbox.publish(snap(7,70,()))
    control.update(mailbox.latest(),now_ns=70_000_000)
    mailbox.publish(snap(8,80,('w',),events=(Event23('press','r'),)))
    mailbox.publish(snap(9,85,('w',),events=(Event23('release','r'),)))
    mailbox.publish(snap(10,90,('w',)))
    assert control.update(mailbox.latest(),now_ns=90_000_000).reset_requested is True
    assert control.update(mailbox.latest(),now_ns=100_000_000).reset_requested is False
