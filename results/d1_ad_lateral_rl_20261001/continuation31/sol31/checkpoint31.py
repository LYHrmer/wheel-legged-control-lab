"""One exclusive C31 final checkpoint, independent reload, and 16-row probe."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from learning31 import LinearEventPolicy31, _matrix54


CHECKPOINT_SCHEMA31 = 'd1-c31-final-lateral-linear-v1'


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exclusive(path: Path, payload: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _probe(policy: LinearEventPolicy31, rows) -> dict:
    x = _matrix54(rows)
    if x.shape != (16,54):
        raise ValueError('C31 final probe is exactly 16 frozen observations')
    with torch.no_grad():
        mean = policy.actor(x).detach().cpu().numpy()
        value = policy.value(x).detach().cpu().numpy()
    if mean.shape != (16,3) or value.shape != (16,1) or not (
            np.isfinite(mean).all() and np.isfinite(value).all()):
        raise RuntimeError('C31 final actor/value probe is invalid')
    return {'observation54':x.tolist(),'actor_mean_z3':mean.tolist(),
            'value':value[:,0].tolist()}


def save_and_reload_final31(policy: LinearEventPolicy31, folder,
                            *, metadata: dict, training_receipt: dict,
                            probe_observations54):
    """Save once; verify bytes and both heads on a separate reloaded object.

    Returns (independent_policy, manifest). The caller owns the overall
    model/load/row counters and must use only this reloaded model for eval.
    """
    if (not isinstance(policy,LinearEventPolicy31) or
            training_receipt.get('status') != 'complete' or
            training_receipt.get('actual_batches') != 8 or
            not 8 <= training_receipt.get('actual_optimizer_steps',-1) <= 64 or
            training_receipt.get('actor_nonzero_gradient_and_parameter_change') is not True or
            not isinstance(metadata,dict) or not metadata.get('source_hashes') or
            not metadata.get('execution_contract_id')):
        raise ValueError('C31 only a complete audited final run may be saved')
    destination = Path(folder)
    if destination.exists():
        raise FileExistsError(destination)
    before = _probe(policy,probe_observations54)
    destination.mkdir(mode=0o755,parents=False,exist_ok=False)
    state = {key:t.detach().cpu().clone() for key,t in policy.state_dict().items()}
    model_buffer = io.BytesIO()
    torch.save(state,model_buffer)
    model_bytes = model_buffer.getvalue()
    _exclusive(destination/'lateral_final.pt',model_bytes)
    probe_bytes = json.dumps(before,sort_keys=True,allow_nan=False).encode('utf-8')
    _exclusive(destination/'probe_before.json',probe_bytes)
    loaded = torch.load(destination/'lateral_final.pt',map_location='cpu',weights_only=True)
    independent = LinearEventPolicy31()
    independent.load_state_dict(loaded,strict=True)
    after = _probe(independent,probe_observations54)
    if before != after or any(
            not torch.equal(state[key],independent.state_dict()[key]) for key in state):
        raise RuntimeError('C31 independent checkpoint reload/probe differs')
    after_bytes = json.dumps(after,sort_keys=True,allow_nan=False).encode('utf-8')
    _exclusive(destination/'probe_after.json',after_bytes)
    manifest = {'schema':CHECKPOINT_SCHEMA31,
                'source_metadata':metadata,
                'training_receipt':training_receipt,
                'files':{
                    'lateral_final.pt':{'sha256':_sha(model_bytes),'bytes':len(model_bytes)},
                    'probe_before.json':{'sha256':_sha(probe_bytes),'bytes':len(probe_bytes)},
                    'probe_after.json':{'sha256':_sha(after_bytes),'bytes':len(after_bytes)}},
                'independent_reload':True,
                'probe_observation_rows':16,'probe_actor_rows':32,
                'probe_value_rows':32,'checkpoint_save_calls':1,
                'checkpoint_reload_calls':1}
    manifest_bytes=json.dumps(manifest,sort_keys=True,indent=2,
                              allow_nan=False).encode('utf-8')
    _exclusive(destination/'manifest.json',manifest_bytes)
    directory_fd=os.open(destination,os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    manifest['manifest_sha256']=_sha(manifest_bytes)
    return independent,manifest


def load_final31(folder, expected_manifest: dict):
    """One restricted weights-only load for a separate evaluation worker."""
    root=Path(folder)
    if expected_manifest.get('schema')!=CHECKPOINT_SCHEMA31:
        raise ValueError('C31 final manifest identity differs')
    if _sha((root/'manifest.json').read_bytes()) != expected_manifest.get('manifest_sha256'):
        raise ValueError('C31 final manifest bytes differ')
    for name,record in expected_manifest['files'].items():
        data=(root/name).read_bytes()
        if len(data)!=record['bytes'] or _sha(data)!=record['sha256']:
            raise ValueError(f'C31 final checkpoint file differs: {name}')
    state=torch.load(root/'lateral_final.pt',map_location='cpu',weights_only=True)
    policy=LinearEventPolicy31()
    policy.load_state_dict(state,strict=True)
    probe=json.loads((root/'probe_after.json').read_text())
    if _probe(policy,probe['observation54'])!=probe:
        raise ValueError('C31 loaded actor/value differs from saved probe')
    return policy
