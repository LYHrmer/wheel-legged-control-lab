"""Real API wrapper restoration and transactional I/O, without a robot model."""
from types import SimpleNamespace
import numpy as np
import pytest
from runtime_support15 import ModelCounter, atomic_training_io


def test_real_library_wrappers_count_and_restore_without_policy_load():
    import torch
    from stable_baselines3.common.base_class import BaseAlgorithm
    original = BaseAlgorithm.__dict__['load']
    backward = torch.Tensor.backward
    fake = SimpleNamespace(policy=SimpleNamespace(predict=lambda *a, **k: (np.zeros((32,16),np.float32), None)))
    counter = ModelCounter(True)
    with counter:
        counter.phase = 'probe'
        BaseAlgorithm.predict(fake, np.zeros((32,99),np.float32), deterministic=True)
        torch.ones((),requires_grad=True).square().backward()
        assert counter.counts['predict']['rows_returned'] == 32
        assert counter.counts['backward']['returned'] == 1
    assert BaseAlgorithm.__dict__['load'] is original
    assert torch.Tensor.backward is backward


def test_api_exception_is_attempted_only_and_restores():
    class Dummy:
        def run(self):
            raise ValueError('synthetic call failure')
    original = Dummy.run
    counter = ModelCounter(True)
    counter._patch(Dummy, 'run', 'load')
    try:
        with pytest.raises(ValueError):
            Dummy().run()
        assert counter.counts['load']['attempted'] == 1
        assert counter.counts['load']['returned'] == 0
    finally:
        counter.__exit__()
    assert Dummy.run is original


def test_atomic_numeric_failure_keeps_prior_manifest_and_never_overwrites(tmp_path):
    from archive13.atomic_archive_13 import ArchiveWriter, verify_manifest
    import short_curriculum_11 as module
    original = module._NumericBlocks
    writer = ArchiveWriter()
    with atomic_training_io(writer):
        block = module._NumericBlocks(tmp_path/'blocks','controls')
        block.append({'control_index':0,'x':np.array([1.,2.])})
        block.flush()
        first = tmp_path/'blocks/controls_0000.npz'
        original_bytes = first.read_bytes()
        assert verify_manifest(first.with_name(first.name+'.manifest.json'))
        (tmp_path/'blocks/controls_0001.npz').write_bytes(b'existing')
        block.append({'control_index':1,'x':np.array([3.,4.])})
        with pytest.raises(FileExistsError):block.flush()
        assert writer.failed and len(block.files)==1 and block.count==1
        with pytest.raises(FileExistsError):block.flush()
        assert first.read_bytes()==original_bytes
        assert (tmp_path/'blocks/controls_0001.npz').read_bytes()==b'existing'
    assert module._NumericBlocks is original
