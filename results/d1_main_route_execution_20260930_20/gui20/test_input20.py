import pytest
from input20 import InputController20, Snapshot20


def feed(c, seq, keys='', **kw):
    return c.update(Snapshot20(seq, seq*.01, frozenset(keys.split()), **kw), now=seq*.01)


def test_forward_yaw_release_and_conflicting_yaw():
    c = InputController20('yaw_1p2')
    assert feed(c,0,'w q').yaw_rps == .3
    assert feed(c,1,'w e').yaw_rps == -.3
    assert feed(c,2,'w q e').yaw_rps == 0.
    assert feed(c,3,'q').forward_mps == feed(c,4).yaw_rps == 0.
    assert feed(c,5,'w').forward_mps == 1.2


def test_stop_and_focus_require_actual_release():
    c = InputController20('flat_1p6')
    assert feed(c,0,'w').forward_mps == 1.6
    assert feed(c,1,'w space').forward_mps == 0.
    assert feed(c,2,'w').reason == 'release_w_to_rearm'
    feed(c,3)
    assert feed(c,4,'w').forward_mps == 1.6
    feed(c,5,'w',focused=False)
    assert feed(c,6,'w').forward_mps == 0.
    feed(c,7)
    assert feed(c,8,'w').forward_mps == 1.6


def test_reset_is_globally_one_shot_even_with_many_polls():
    c = InputController20('flat_0p6')
    assert feed(c,0,'r w').reset_requested
    assert not feed(c,1,'r w').reset_requested
    feed(c,2)
    assert not feed(c,3,'r').reset_requested
    assert feed(c,4,'w').forward_mps == 0.


@pytest.mark.parametrize('stamp,now,reason',[(float('nan'),1.,'invalid_clock'),
    (2.,1.,'future_snapshot'),(0.,1.,'stale_snapshot')])
def test_bad_input_never_enables_motion(stamp,now,reason):
    c=InputController20('flat_1p6')
    r=c.update(Snapshot20(0,stamp,frozenset({'w'})),now=now)
    assert r.forward_mps == 0. and r.reason == reason and r.consumed_sequence is None


def test_polled_freshness_and_exit_latch():
    c=InputController20('flat_1p6')
    assert feed(c,0,'w q a d').yaw_rps == 0.
    r=c.update(Snapshot20(0,0.,frozenset({'w'})),now=.1)
    assert r.forward_mps == 0.
    feed(c,11)
    assert feed(c,12,'w').forward_mps == 1.6
    assert feed(c,13,'escape w').exit_requested
    assert feed(c,14,'w').forward_mps == 0.


def test_immutable_public_profile():
    c=InputController20('flat_0p6')
    with pytest.raises(AttributeError):
        c.profile='flat_1p6'


def test_stale_close_still_removes_all_motion_authority():
    c=InputController20('flat_1p6')
    r=c.update(Snapshot20(0,0.,frozenset({'w'}),closed=True),now=1.)
    assert r.exit_requested and r.forward_mps == 0.
    assert feed(c,101).forward_mps == 0.
    assert feed(c,102,'w').forward_mps == 0.
