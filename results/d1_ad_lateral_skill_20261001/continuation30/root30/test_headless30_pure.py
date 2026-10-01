"""Saved-data/pure checks only; do not import an engine or a learning runtime."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from runtime30 import fence_copy_data30, NativeWindow30
from macro30 import components30, build_macros30

W = Path(__file__).resolve().parents[2]


def test_headless_copy_permissions_reject_reverse_and_foreign_thread():
    plant = SimpleNamespace(data=object(), measurement_data=object(), model=object())
    calls, counts = [], []
    mj = SimpleNamespace(mj_copyData=lambda *args: calls.append(args))
    access = SimpleNamespace(scratch=object(), count=lambda kind: counts.append(kind))
    stop = threading.Event()
    _, edges = fence_copy_data30(mj, plant=plant, side_access=access, stop_event=stop)
    mj.mj_copyData(plant.measurement_data, plant.model, plant.data)
    mj.mj_copyData(access.scratch, plant.model, plant.measurement_data)
    with pytest.raises(RuntimeError, match='owner/data'):
        mj.mj_copyData(plant.data, plant.model, access.scratch)
    errors = []
    def foreign():
        try:
            mj.mj_copyData(plant.measurement_data, plant.model, plant.data)
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert len(calls) == 2 and counts == ['copy'] and len(errors) == 1
    assert edges == dict(live_to_measurement=1, measurement_to_side_scratch=1, rejected=2)
    assert stop.is_set()


def test_actual_saved_five_native_rows_close_without_engine():
    run = W/'continuation29/profile_left_01'
    construction = json.loads((run/'construction_receipt.json').read_text())
    guard = NativeWindow30(construction['side_kinematic_binding'])
    guard.begin(0)
    with gzip.open(run/'episode_0/native_block_0000.jsonl.gz', 'rt') as stream:
        rows = [json.loads(next(stream)) for _ in range(5)]
    for row in rows:
        guard.sink(copy.deepcopy(row))
    receipt = guard.complete()
    assert guard.pure_fk_calls == 5 and receipt['control_index'] == 0
    assert [r['native_index'] for r in receipt['native_rows']] == list(range(5))
    assert all(len(r['whole_wheel_min_z_m']) == 4 for r in receipt['native_rows'])
    # The saved cold reset first touches at native 4; do not invent contact
    # during its four preceding integration returns.
    assert [r['active_wheel_terrain_contact'] for r in receipt['native_rows']] == (
        [[False]*4 for _ in range(4)] + [[True]*4])
    with pytest.raises(RuntimeError, match='index'):
        guard.sink(copy.deepcopy(rows[-1]))


def test_event_partition_counts_controls_once_and_keeps_terminal_tail():
    q = np.zeros((7, 23))
    q[:, 3] = 1.
    q[:, 1] = [0., -.006, 0., .012, .036, .033, .03]
    states = dict(qpos=q, qvel=np.zeros((7, 22)), time=np.arange(7)*.01)
    rows = [dict(native_actuator_traces=[dict(applied_nm=[1.]*16) for _ in range(5)]) for _ in range(6)]
    latches = [dict(control_index=i, proposed_action=[0., 0., .5], projected_action=[0., 0., .5],
                    consumed_action=[0., 0., .5], reason='fixture') for i in (0, 3)]
    macros = build_macros30(states, rows, latches, torque_limits=np.ones(16)*2,
                           origin_xy=[0., 0.], yaw0=0., direction=1,
                           end_reason='success_after_retention')
    whole = components30(q, np.ones((6, 5, 16)), np.ones(16)*2,
                         origin_xy=[0., 0.], yaw0=0., direction=1)
    assert len(macros) == 2 and sum(m['elapsed_controls'] for m in macros) == 6
    assert macros[0]['terminal_reason'] == 'next_leg'
    assert macros[-1]['terminal_reason'] == 'success_after_retention'
    for key in whole:
        assert sum(m['reward_components'][key] for m in macros) == pytest.approx(whole[key])
    assert whole['progress'] == pytest.approx(1.)
    assert whole['actual_torque_square_normalized_s'] == pytest.approx(.015)
    assert whole['backtrack'] == pytest.approx(.4)
    assert all(m['weighted_reward'] is None for m in macros)
