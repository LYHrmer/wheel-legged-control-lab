"""No engine imports/construction: real bookkeeping against narrow fake endpoints."""
import math
from types import SimpleNamespace

import pytest

import runtime31


def test_budget_keeps_cumulative_counts_and_forbids_reset_after_incomplete():
    budget = runtime31.EpisodeBudget31(cycles=2, controls=4400, macros=8)
    assert budget.begin(controls=0, native=0)['global_control_begin'] == 0
    first = budget.finish(controls=1600, native=8000, macros=4, complete=True)
    assert first['actual_normal_native'] == 8000
    with pytest.raises(RuntimeError):
        budget.begin(controls=0, native=0)
    second = budget.begin(controls=1600, native=8000)
    assert second['cycle_index'] == 1 and second['control_limit'] == 2200
    budget.finish(controls=2000, native=10000, macros=1, complete=False)
    with pytest.raises(RuntimeError):
        budget.begin(controls=2000, native=10000)
    assert budget.macros == 5
    stopped = runtime31.EpisodeBudget31(cycles=2, controls=4400, macros=8)
    stopped.begin(controls=0, native=0)
    stopped.finish(controls=50, native=250, macros=0, complete=False)
    with pytest.raises(RuntimeError):
        stopped.begin(controls=50, native=250)


def test_yaw_is_forwarded_through_one_original_reset_before_prepare():
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        return 'reset_result'
    env = SimpleNamespace(loop=SimpleNamespace(reset=original))
    state = runtime31.install_reset_yaw31(env)
    assert env.loop.reset(seed=7, base_position=(1, 2, 3)) == 'reset_result'
    assert calls[0] == dict(seed=7, base_position=(1, 2, 3))
    state['initial_yaw_rad'] = .04
    env.loop.reset(seed=7, base_position=(1, 2, 3))
    assert calls[1]['base_quaternion'] == (math.cos(.02), 0., 0., math.sin(.02))
    assert state['calls'] == 2
    state['initial_yaw_rad'] = .05
    with pytest.raises(ValueError):
        env.loop.reset(seed=7)
    assert len(calls) == 2


def test_native_windows_preserve_global_indices_and_reject_unclosed_overwrite(monkeypatch):
    class FakeWindow:
        def __init__(self, binding):
            self.rows = []
        def begin(self, index):
            self.index = index
            self.rows = []
        def sink(self, row):
            assert row['native_index'] == 5*self.index+len(self.rows)
            row['skill30_geometry'] = {}
            self.rows.append(row)
        def complete(self):
            if len(self.rows) != 5:
                raise RuntimeError('not five native rows')
            return dict(native_rows=self.rows)
    monkeypatch.setattr(runtime31, 'NativeWindow30', FakeWindow)
    window = runtime31.NativeWindow31({})
    window.begin_case31(0, 0)
    window.begin(0)
    with pytest.raises(RuntimeError):
        window.begin(1)
    with pytest.raises(RuntimeError):
        window.begin_case31(1, 0)
    for i in range(5):
        window.sink(dict(native_index=i))
    window.complete()
    with pytest.raises(RuntimeError):
        window.complete()
    with pytest.raises(RuntimeError):
        window.begin_case31(1, 0)
    window.begin_case31(1, 1)
    window.begin(0)
    for i in range(5, 10):
        window.sink(dict(native_index=i))
    result = window.complete()
    assert [r['native_index'] for r in result['native_rows']] == list(range(5, 10))
    assert window.native_rows31[0]['skill30_geometry']['local_native_index31'] == 0
    assert window.pure_fk_calls == 10
