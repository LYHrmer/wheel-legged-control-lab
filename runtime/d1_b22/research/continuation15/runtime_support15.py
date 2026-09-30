"""Count model APIs and bridge frozen training I/O to archive13 transactions."""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
import hashlib
from pathlib import Path
import numpy as np


class ModelCounter:
    def __init__(self, training):
        self.training = training
        self.phase = 'setup'
        self.counts = {}
        self.restore = []
        self.limits = ({'load': 2, 'torch_load': 6, 'save': 2, 'learn': 1,
                        'train': 16, 'forward': 16384, 'evaluate_actions': 256,
                        'predict_values': 32, 'predict': 3, 'backward': 256}
                       if training else {'load': 2, 'torch_load': 6, 'save': 0,
                        'learn': 0, 'train': 0, 'forward': 0, 'evaluate_actions': 0,
                        'predict_values': 0, 'predict': 20802, 'backward': 0})

    def _patch(self, owner, name, key, row_argument=None, class_method=False):
        original_descriptor = owner.__dict__[name]
        original = original_descriptor.__func__ if class_method else getattr(owner, name)
        counter = self
        @wraps(original)
        def call(*args, **kwargs):
            record = counter.counts.setdefault(key, {'attempted': 0, 'returned': 0,
                                                      'rows_attempted': 0, 'rows_returned': 0,
                                                      'phases': {}})
            if record['attempted'] >= counter.limits[key]:
                raise RuntimeError('model API limit exhausted: ' + key)
            rows = 0
            if row_argument is not None:
                value = args[row_argument] if len(args) > row_argument else kwargs.get('observation', kwargs.get('obs'))
                rows = 1 if len(value.shape) == 1 else int(value.shape[0])
                expected = 256 if key == 'evaluate_actions' else (32 if key == 'predict' and counter.phase == 'probe' else 1)
                if rows != expected:
                    raise RuntimeError('model batch differs from fixed budget: ' + key)
            record['attempted'] += 1
            record['rows_attempted'] += rows
            phase = record['phases'].setdefault(counter.phase, {'attempted': 0, 'returned': 0, 'rows': 0})
            phase['attempted'] += 1
            value = original(*args, **kwargs)
            record['returned'] += 1
            record['rows_returned'] += rows
            phase['returned'] += 1
            phase['rows'] += rows
            return value
        setattr(owner, name, classmethod(call) if class_method else call)
        self.restore.append((owner, name, original_descriptor))

    def __enter__(self):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3.common.policies import ActorCriticPolicy
        self._patch(BaseAlgorithm, 'load', 'load', class_method=True)
        self._patch(BaseAlgorithm, 'save', 'save')
        self._patch(BaseAlgorithm, 'predict', 'predict', 1)
        self._patch(PPO, 'learn', 'learn')
        self._patch(PPO, 'train', 'train')
        self._patch(ActorCriticPolicy, 'forward', 'forward', 1)
        self._patch(ActorCriticPolicy, 'evaluate_actions', 'evaluate_actions', 1)
        self._patch(ActorCriticPolicy, 'predict_values', 'predict_values', 1)
        self._patch(torch, 'load', 'torch_load')
        self._patch(torch.Tensor, 'backward', 'backward')
        return self

    def __exit__(self, *_):
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        self.restore.clear()

    def summary(self):
        return {'schema': 'd1-model-api-ledger-15-v1', 'limits': self.limits,
                'counts': self.counts,
                'actor_rows': sum(self.counts.get(k, {}).get('rows_returned', 0)
                                  for k in ('forward', 'evaluate_actions', 'predict')),
                'critic_rows': sum(self.counts.get(k, {}).get('rows_returned', 0)
                                   for k in ('forward', 'evaluate_actions', 'predict_values'))}


@contextmanager
def atomic_training_io(writer):
    import short_curriculum_11 as module
    from rl16_curriculum_08 import _NumericBlocks
    class NumericBlocks(_NumericBlocks):
        def flush(self):
            if not self.count:
                return
            path = self.directory / f'{self.stem}_{len(self.files):04d}.npz'
            arrays = {key: np.stack(values) for key, values in self.pending.items()}
            writer.commit_npz(path, arrays)
            self.files.append({'file': path.name, 'rows': self.count,
                               'first_control_index': int(arrays['control_index'][0]),
                               'last_control_index': int(arrays['control_index'][-1]),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'bytes': path.stat().st_size})
            self.pending = {}
            self.count = 0
    originals = {key: getattr(module, key) for key in (
        '_NumericBlocks', '_write_exclusive_json', '_write_exclusive_npz')}
    module._NumericBlocks = NumericBlocks
    module._write_exclusive_json = writer.commit_json
    module._write_exclusive_npz = lambda path, **arrays: writer.commit_npz(path, arrays)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(module, key, value)
