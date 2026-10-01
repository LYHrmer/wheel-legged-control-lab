"""Narrow saved ownership regressions, including non-adjacent wrapper/worker PIDs."""
import copy
from pathlib import Path

import pytest
import read31  # noqa: F401 -- establish existing cold reader import guard
from source_read31_v2 import readiness31, HOST_IDENTITY


def fixture():
    run = Path('/tmp/c31_saved_fixture')
    go = dict(worker='/tmp/frozen_worker31.py', host='/tmp/frozen_host31.py')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', go['worker'],
        '--session', str(run/'session.json'), '--output', str(run)]
    session = dict(argv=command[4:], host_started_monotonic=100.,
                   source_hashes={go['host']: HOST_IDENTITY})
    ready = dict(pid=91, monotonic_s=103., session_sha256='bound_session')
    child = dict(pid=10, argv=command)
    host = dict(exit_code=0, failure=None, source_mismatches=[], owned_no_orphans=True,
        reservation_closed=True, elapsed_s=40., execution_started_monotonic=103.,
        owned_cleanup=dict(method='linux_subreaper_ancestry_birth_tick', remaining={}))
    return [run, session, ready, child, host, go, {'sha256': 'bound_session'}, HOST_IDENTITY]


def test_real_wrapper_descendant_need_not_share_pid_or_be_adjacent():
    values = fixture()
    proof = readiness31(*values)
    assert proof['passed'] and proof['rtk_wrapper_pid'] == 10 and proof['actual_worker_ready_pid'] == 91
    assert not proof['direct_saved_PID_PPID_tree_available']
    assert not proof['PID_equality_or_adjacency_used']
    values[2]['pid'] = 10
    assert readiness31(*values)['passed']


@pytest.mark.parametrize('kind', ['session', 'argv', 'time', 'host_failure', 'source'])
def test_unbound_or_failed_readiness_is_rejected(kind):
    values = copy.deepcopy(fixture())
    if kind == 'session':
        values[2]['session_sha256'] = 'different'
    elif kind == 'argv':
        values[3]['argv'][4] = '/tmp/other_worker.py'
    elif kind == 'time':
        values[2]['monotonic_s'] = 99.
        values[4]['execution_started_monotonic'] = 99.
    elif kind == 'host_failure':
        values[4]['failure'] = {'message': 'readiness mismatch'}
    else:
        values[7]['sha256'] = 'other_host'
    with pytest.raises(ValueError):
        readiness31(*values)
