"""Decode only the four CPU float64 tensors of the C31 Torch ZIP, without Torch.

The pickle is restricted to storage references, tensor reconstruction and an
empty OrderedDict. Unknown globals, views, devices, dtypes and shapes fail.
"""
from __future__ import annotations

from collections import OrderedDict
import io
from pathlib import Path
import pickle
import zipfile

import numpy as np

from math_read31 import PARAMETER_SHAPES, need, checked_parameters


def _tensor(storage, offset, size, stride, requires_grad, backward_hooks, metadata=None):
    need(isinstance(storage, np.ndarray) and offset == 0 and requires_grad is False
         and not backward_hooks and metadata is None, 'unsupported tensor view/metadata')
    shape = tuple(size)
    need(shape in PARAMETER_SHAPES.values() and storage.size == int(np.prod(shape)), 'unexpected tensor storage/shape')
    expected_stride = tuple(int(np.prod(shape[i+1:])) for i in range(len(shape)))
    need(tuple(stride) == expected_stride, 'noncontiguous tensor view')
    return storage.reshape(shape).copy()


class TensorOnlyUnpickler(pickle.Unpickler):
    def __init__(self, data, archive, prefix):
        super().__init__(io.BytesIO(data))
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ('torch', 'DoubleStorage'):
            return 'CPU_FLOAT64_STORAGE'
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return _tensor
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        raise ValueError('forbidden checkpoint pickle global: '+module+'.'+name)

    def persistent_load(self, identifier):
        need(isinstance(identifier, tuple) and len(identifier) == 5, 'unknown persistent pickle object')
        tag, dtype, key, device, size = identifier
        need(tag == 'storage' and dtype == 'CPU_FLOAT64_STORAGE' and device == 'cpu'
             and isinstance(key, str) and key.isdigit() and type(size) is int and 0 < size <= 162,
             'unregistered checkpoint storage/device/dtype')
        payload = self.archive.read(self.prefix+'data/'+key)
        need(len(payload) == 8*size, 'tensor storage byte count differs')
        return np.frombuffer(payload, dtype='<f8').copy()


def checkpoint_parameters31(path):
    path = Path(path)
    need(path.stat().st_size <= 1 << 20, 'C31 four linear tensors exceed bounded archive size')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) and all(not n.startswith('/') and '..' not in n.split('/') for n in names),
             'duplicate/unsafe checkpoint ZIP members')
        entries = [name for name in names if name.endswith('/data.pkl')]
        need(len(entries) == 1 and sum(i.file_size for i in archive.infolist()) <= 1 << 20,
             'checkpoint pickle/member sizes differ')
        prefix = entries[0][:-len('data.pkl')]
        need(archive.read(prefix+'byteorder') == b'little', 'unsupported tensor storage byte order')
        result = TensorOnlyUnpickler(archive.read(entries[0]), archive, prefix).load()
    need(isinstance(result, dict), 'checkpoint root must be four tensors')
    return checked_parameters(result)
