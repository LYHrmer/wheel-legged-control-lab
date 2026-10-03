"""Reject a changed physical prefix or action before using paired speed claims."""
import gzip
import json

import numpy as np
import pytest

from pair_prefix33 import ARRAYS, identity, pair_prefix33


def test_entry_bridge_rejects_state_and_action_mutations(tmp_path):
    old = tmp_path / 'old' / 'episode_0'
    new = tmp_path / 'new'
    old.mkdir(parents=True)
    new.mkdir()
    initial = {key: np.zeros(2) for key in ARRAYS}
    states = {key: np.zeros((201, 2)) for key in ARRAYS}
    row = dict(actor='B', policy_predict_called=True, skill30=None,
               macro_events30=[], input_observation99=[0.]*99,
               raw_action16=[0.]*16, action16=[0.]*16,
               policy_input_action=[0.]*16, raw_command={},
               controller_handoff_receipt=None, side_start_receipt=None,
               info=dict(controller_record={}, raw_operator_command={},
                         consumed_command={}, servo_receipt={}))

    def controls(folder, changed=False):
        with gzip.open(folder / 'controls.jsonl.gz', 'wt') as out:
            for index in range(200):
                payload = dict(row)
                if changed and index == 42:
                    payload['raw_action16'] = [1.]+[0.]*15
                out.write(json.dumps(payload)+'\n')

    for folder in (old, new):
        np.savez(folder / 'initial_state.npz', **initial)
        np.savez(folder / 'states.npz', **states)
        controls(folder)
    report = tmp_path / 'baseline.json'
    report.write_text(json.dumps({'run': str(old.parent)}))
    bound = [report, old/'initial_state.npz', old/'states.npz', old/'controls.jsonl.gz']
    sources = {str(path): identity(path) for path in bound}
    assert pair_prefix33(report, new, sources)['passed']
    states['qvel'][42, 0] = .001
    np.savez(new/'states.npz', **states)
    with pytest.raises(RuntimeError, match='actual reset and 200-control entry differ'):
        pair_prefix33(report, new, sources)
    states['qvel'][42, 0] = 0.
    np.savez(new/'states.npz', **states)
    controls(new, changed=True)
    with pytest.raises(RuntimeError, match='actual reset and 200-control entry differ'):
        pair_prefix33(report, new, sources)
