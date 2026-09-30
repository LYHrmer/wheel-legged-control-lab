"""Saved C18 data exercises the new training recorder without a plant or policy."""
from dataclasses import make_dataclass
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from curriculum20 import Curriculum20
from recipes20 import select_episode
from runtime_support20 import atomic_training_io, ModelCounter


def test_new_training_recorder_preserves_saved_c18_chain():
    c18 = Path(__file__).resolve().parent.parent/'continuation18'
    folder = c18/'qualification_primary_01/heldout/flat_0p6_grouped_continue'
    receipt = json.loads((folder/'case_receipt.json').read_text())
    with gzip.open(folder/receipt['controller_record_blocks'][0]['file'], 'rt') as stream:
        saved = json.loads(next(stream))
    with np.load(folder/'states.npz', allow_pickle=False) as data:
        states = {k:data[k][:2] for k in data.files}
    traces = []
    for raw in saved['native_actuator_traces']:
        Trace = make_dataclass('SavedTrace', list(raw))
        traces.append(Trace(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in raw.items()}))
    plant = SimpleNamespace(data=SimpleNamespace(**{k:v[1] for k,v in states.items() if k!='observation'}),
                            last_control_interval_actuator_traces=traces)
    env = SimpleNamespace(plant=plant, _last_observation=states['observation'][1])
    env.unwrapped = env
    recorder = object.__new__(Curriculum20)
    recorder.env = env
    recorder._episode = {'episode_index':0,'completed_controls':0}
    recorder._active_schedule = select_episode(0,'A')
    recorder.completed_controls = recorder.source_offset = 0
    recorder._full_rows = []
    recorder._pre_state20 = {k:v[0] for k,v in states.items()}
    row = recorder._numeric_row(states['observation'][0], saved['reward'],
        saved['terminated'], saved['truncated'], saved['info'])
    full = recorder._full_rows[0]
    assert full['info'] == saved['info']
    assert full['pre_state']['qpos'] == states['qpos'][0].tolist()
    assert full['post_state']['qpos'] == states['qpos'][1].tolist()
    assert row['actuator_applied5x16'].shape == (5,16)
    assert full['info']['controller_record']['calculation']['wheel_common_reference_z_before_rad_s'] == 0.


def test_new_atomic_io_does_not_change_frozen_curriculum(tmp_path):
    import curriculum20
    import short_curriculum_11
    from archive13.atomic_archive_13 import ArchiveWriter, verify_manifest
    prior = curriculum20._NumericBlocks
    frozen_prior = short_curriculum_11._NumericBlocks
    writer = ArchiveWriter()
    with atomic_training_io(writer):
        assert short_curriculum_11._NumericBlocks is frozen_prior
        blocks = curriculum20._NumericBlocks(tmp_path/'blocks','controls')
        blocks.append({'control_index':0,'x':np.ones(2)})
        blocks.flush()
        assert verify_manifest(tmp_path/'blocks/controls_0000.npz.manifest.json')
        (tmp_path/'blocks/controls_0001.npz').write_bytes(b'prior')
        blocks.append({'control_index':1,'x':np.ones(2)})
        with pytest.raises(FileExistsError):
            blocks.flush()
        assert writer.failed and len(blocks.files) == 1
    assert curriculum20._NumericBlocks is prior


def test_api_ceiling_refuses_extra_attempt_and_restores():
    class API:
        def call(self):
            return 7
    original = API.call
    counter = ModelCounter(True, {'train':1})
    counter._patch(API, 'call', 'train')
    try:
        assert API().call() == 7
        with pytest.raises(RuntimeError, match='limit exhausted'):
            API().call()
        assert counter.counts['train']['attempted'] == counter.counts['train']['returned'] == 1
    finally:
        counter.__exit__()
    assert API.call is original
