"""C38 NumPy actor, validated36 provenance and a NumPy-only request ledger.

This consumes the unchanged prepared99 observation. It returns the original
clipped16 action; the unchanged C34 gain seam alone applies float32(.5).
Importing this module is stdlib-only and never loads exported weights.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'C38_B22_numpy_half_gain_v1'
EXPORT = ROOT/'results/b22_numpy36_20261018_01'
EXPORT_REPORT_SHA = '38ab697abaf231513490baa6794b44de6dff1a0a5a065edd182b78efdce019dd'
EXPORT_RELEASE_SHA = 'd1b55ae68be02b10bf47e75d1535141da8e0812dab4f5f89fda1b6c65742ff3f'
EXPORT_MANIFEST_SHA = '8bcaf616026b22585e4f698da7cedae35ab48c75d1d3621e8d98f1f618c5f583'
HELPER = ROOT/'scripts/export_b22_numpy36.py'
HELPER_SHA = '8a89d8d1b82e59996bb09283050a40351a563fa19dbae37a4e17302893f466ce'
FORBIDDEN = {'torch','stable_baselines3','cv2','jax','tensorflow'}
API_LIMITS = {'backward':0,'evaluate_actions':0,'forward':0,'learn':0,'load':0,'predict':1601,
              'predict_values':0,'save':0,'torch_load':0,'train':0}
_LOCAL = threading.local()


def require(value,message):
    if not value:
        raise ValueError(message)


class DenyFrameworks38(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.partition('.')[0] in FORBIDDEN:
            raise ImportError('C38 forbidden framework/import: '+fullname)


def frameworks_absent38():
    return not any(name.partition('.')[0] in FORBIDDEN for name in sys.modules)


def deny_frameworks38():
    require(frameworks_absent38(),'C38 was not a cold Torch/SB3/OpenCV boundary')
    if not any(type(guard) is DenyFrameworks38 for guard in sys.meta_path):
        sys.meta_path.insert(0,DenyFrameworks38())


def helper36():
    import hashlib
    require(hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA,'C36 numerical helper changed')
    spec = importlib.util.spec_from_file_location('b22_numpy36_bound38',HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_identities38():
    """Read only small provenance/artifact files, never deserialize real weights."""
    h = helper36()
    require(h.identity(EXPORT/'report.json')['sha256'] == EXPORT_REPORT_SHA
            and h.identity(EXPORT/'release.json')['sha256'] == EXPORT_RELEASE_SHA
            and h.identity(EXPORT/'actor_manifest.json')['sha256'] == EXPORT_MANIFEST_SHA,
            'C38 fixed successful36 export identity differs')
    report,release,bundle = (h.read_json(EXPORT/name) for name in ('report.json','release.json','actor_manifest.json'))
    require(report['status'] == 'COMPLETE' and report['cleanup']['remaining'] == {}
            and report['release_identity'] == h.identity(EXPORT/'release.json')
            and bundle['release_identity'] == report['release_identity']
            and bundle['model_sha256'] == h.MODEL_SHA and bundle['metadata_sha256'] == h.METADATA_SHA
            and bundle['policy_state_sha256'] == h.POLICY_STATE_SHA
            and bundle['source'] == release['source'] and bundle['fixture_identity'] == h.FIXTURE,
            'C38 completed36 policy/source/input proof differs')
    names = ('report.json','release.json','actor_manifest.json','actor.npz','torch_report.json','numpy_report.json',
             'torch_outputs.npz','numpy_outputs.npz')
    identities = {str(EXPORT/name):h.identity(EXPORT/name) for name in names}
    for stage in ('torch','numpy'):
        child = h.read_json(EXPORT/(stage+'_report.json'))
        require(report['stages'][stage]['status'] == 'COMPLETE'
                and report['stages'][stage]['report_identity'] == identities[str(EXPORT/(stage+'_report.json'))],
                'C38 export36 stage identity differs')
        h.verify_worker(release,EXPORT/'release.json',stage,child)
    require(bundle['weights'] == identities[str(EXPORT/'actor.npz')],'C38 actor arrays changed')
    identities[str(HELPER)] = h.identity(HELPER)
    return identities


def audited_origins38(session):
    require(frameworks_absent38(),'C38 forbidden framework loaded')
    origins = helper36().dependency_origins(session['source_hashes'],{'numpy','gymnasium','mujoco'})
    return {'schema':'b22-numpy38-runtime-origins-v1','module_origins':origins,
            'forbidden_modules_absent':True,'forbidden_prefixes':sorted(FORBIDDEN),
            'interpreter_backend':'NumPy','ppo_load_calls':0,'torch_load_calls':0}


class NumpyCounter38:
    """Counts actual NumPy actor calls/rows; it does not pretend to hook SB3."""
    def __init__(self,training,limits):
        require(training is False and limits == API_LIMITS and all(type(v) is int for v in limits.values()),
                'C38 fixed no-training NumPy request budget required')
        self.limits,self.phase,self.counts,self.bundle_loads = dict(limits),'setup',{},0

    def __enter__(self):
        require(frameworks_absent38() and getattr(_LOCAL,'counter',None) is None,'C38 active/nested framework counter')
        _LOCAL.counter = self
        return self

    def __exit__(self,*args):
        _LOCAL.counter = None

    def call(self,rows,function):
        require(getattr(_LOCAL,'counter',None) is self and frameworks_absent38(),'C38 unowned or framework prediction')
        require((self.phase == 'probe' and rows == 32) or (self.phase == 'control' and rows == 1),
                'C38 only one32 probe and batch1 control predictions allowed')
        record = self.counts.setdefault('predict',{'attempted':0,'returned':0,'rows_attempted':0,'rows_returned':0,'phases':{}})
        phase = record['phases'].setdefault(self.phase,{'attempted':0,'returned':0,'rows':0})
        require(record['attempted'] < 1601 and record['rows_attempted']+rows <= 1632
                and phase['attempted'] < (1 if self.phase == 'probe' else 1600), 'C38 NumPy request cap exhausted')
        record['attempted'] += 1;record['rows_attempted'] += rows;phase['attempted'] += 1
        value = function()
        record['returned'] += 1;record['rows_returned'] += rows;phase['returned'] += 1;phase['rows'] += rows
        return value

    def summary(self):
        return {'schema':'b22-numpy38-model-api-ledger-v1','backend':'NumPy','limits':dict(self.limits),
            'counts':json.loads(json.dumps(self.counts)),
            'actor_rows':self.counts.get('predict',{}).get('rows_returned',0),'critic_rows':0,
            'numpy_actor_bundle_loads':self.bundle_loads,'forbidden_modules_absent':frameworks_absent38(),
            'ppo_load_calls':0,'torch_load_calls':0,'instrumentation_scope':'owned NumPy predictor only; forbidden framework import gate'}


class NumpyActor38:
    def __init__(self,weights,np,counter):
        helper36().validate_weights(weights,np)
        require(type(counter) is NumpyCounter38,'C38 actual owned NumPy counter required')
        self.weights,self.np,self.counter = weights,np,counter

    def predict(self,observation,deterministic=True):
        np = self.np
        require(deterministic is True,'C38 only deterministic mean actor')
        value = np.asarray(observation)
        require(value.dtype == np.float32 and value.shape in ((99,),(32,99))
                and bool(np.isfinite(value).all()) and bool((abs(value) <= 5.).all()),'C38 finite prepared99 float32 required')
        rows = 1 if value.ndim == 1 else 32
        original = value.tobytes()
        def forward():
            w = self.weights
            hidden = np.tanh(value.reshape(rows,99)@w['w1'].T+w['b1'])
            hidden = np.tanh(hidden@w['w2'].T+w['b2'])
            clipped = np.clip(hidden@w['w3'].T+w['b3'],np.float32(-1.),np.float32(1.)).astype(np.float32,copy=False)
            require(bool(np.isfinite(clipped).all()) and value.tobytes() == original,'C38 invalid output or mutated input')
            return (clipped.reshape(16) if rows == 1 else clipped),None
        return self.counter.call(rows,forward)


def load_and_verify_numpy38(folder,expected_manifest,*,return_model=False):
    """A new numerical load proof, never a claim of byte-exact SB3 reload."""
    require(type(return_model) is bool and frameworks_absent38(),'C38 cold NumPy load required')
    counter = getattr(_LOCAL,'counter',None)
    require(type(counter) is NumpyCounter38 and counter.phase == 'probe' and counter.bundle_loads == 0,
            'C38 exactly one owned bundle admission required')
    identities = export_identities38()
    h = helper36()
    bundle = h.read_json(EXPORT/'actor_manifest.json')
    require(expected_manifest['files'] == bundle['original_probe_files']
            and expected_manifest['metadata_sha256'] == h.METADATA_SHA
            and str(Path(folder).resolve()) == str((ROOT/'runtime/d1_b22/checkpoint').resolve()),
            'C38 original B22 checkpoint/probe identity differs')
    counter.bundle_loads += 1
    import numpy as np
    require(np.__version__ == bundle['versions']['numpy'],'C38 NumPy version differs from verified36')
    with np.load(EXPORT/'actor.npz',allow_pickle=False) as archive:
        weights = {name:archive[name] for name in archive.files}
    model = NumpyActor38(weights,np,counter)
    observations = Path(folder)/'final_probe_observations_f32.bin'
    actions = Path(folder)/'final_probe_actions_f32.bin'
    require(all(h.identity(Path(folder)/name) == entry for name,entry in expected_manifest['files'].items()),
            'C38 original probe/ZIP changed')
    obs = np.frombuffer(observations.read_bytes(),dtype='<f4').reshape(32,99)
    reference = np.frombuffer(actions.read_bytes(),dtype='<f4').reshape(32,16)
    actual,state = model.predict(obs,deterministic=True)
    require(state is None and bool(np.isfinite(reference).all()),'C38 invalid original32 probe')
    raw = float(np.max(abs(actual.astype(np.float64)-reference.astype(np.float64))))
    half = float(np.max(abs((actual*np.float32(.5)).astype(np.float64)-(reference*np.float32(.5)).astype(np.float64))))
    require(raw <= 1e-6 and half <= 1e-6,'C38 NumPy32 probe exceeds original clipped16/half tolerance')
    require(all(h.identity(name) == entry for name,entry in identities.items()),'C38 exported actor changed during load')
    proof = {'schema':'b22-numpy38-load-proof-v1','execution_contract_id':CONTRACT,
        'backend':'NumPy','model_sha256':h.MODEL_SHA,'source_policy_state_sha256':h.POLICY_STATE_SHA,
        'metadata_sha256':h.METADATA_SHA,'export36_artifacts':identities,'probe_rows':32,
        'probe_prediction_calls':1,'probe_actions_byte_exact':actual.tobytes() == reference.tobytes(),
        'probe_actions_within_tolerance':True,'absolute_tolerance':1e-6,
        'clipped16_max_absolute_error':raw,'half_gain_max_absolute_error':half,
        'output_contract':'clipped16; worker applies float32 half gain once',
        'ppo_load_calls':0,'torch_load_calls':0,'verified_without_engine_or_reset':True,
        'control_qualification_granted':False}
    return (model,proof) if return_model else proof
