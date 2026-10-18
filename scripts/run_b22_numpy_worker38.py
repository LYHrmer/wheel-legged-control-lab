"""C38 cold binding proof or explicit NumPy-only derivative of the C34 owner.

No environment or model is created by import/default plan. The original
prepared99 builder, controller, native owner and archive code remain pinned.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
_SPEC = importlib.util.spec_from_file_location('b22_numpy_backend38',Path(__file__).with_name('b22_numpy_backend38.py'))
backend = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = backend
_SPEC.loader.exec_module(backend)

CONTRACT = backend.CONTRACT
ORIGINAL = ROOT/'scripts/run_b22_worker34.py'
ORIGINAL_SHA = '2f71895b709ec1abbd947c96673ec969de3a483bb4d145843315754513c3eb5d'
ZERO_FIELDS = ('construction_attempts','construction_returns','control_attempts','control_returns',
               'ccd_attempts','ccd_returns','violations','construction_limit','control_limit',
               'target_model','target_data','armed','phase','first_ccd_caller',
               'first_construction_step_caller','first_control_step_caller')


def require(value,message):
    backend.require(value,message)


def identity(path):
    digest,size = hashlib.sha256(),0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk);size += len(chunk)
    return {'bytes':size,'sha256':digest.hexdigest()}


def write_json(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def zero_state38(state):
    require(isinstance(state,dict) and all(type(state.get(key)) is int and state[key] == 0 for key in ZERO_FIELDS),
            'C38 bootstrap changed native counters/arm/phase/target/limits')


def bootstrap_components38():
    # Retain explicit extension order, including the unused rollout GOT slot.
    # isort: off
    import mujoco
    import mujoco._functions
    import mujoco._rollout
    import engine_binding
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    # isort: on
    return mujoco,engine_binding,_nominal_geometry


def bootstrap38(session,output):
    """Load fixed extensions and inspect binding/state, never arm or construct."""
    session_path = Path(output)/'session.json'
    require(not session_path.is_symlink() and json.loads(session_path.read_text()) == session,
            'C38 bootstrap session file differs from the admitted request')
    session_identity = identity(session_path)
    backend.deny_frameworks38()
    require(sys.version_info[:2] == (3,10) and 'LD_LIBRARY_PATH' not in os.environ
            and os.environ.get('LD_PRELOAD') == session['library'] and os.environ.get('LD_BIND_NOW') == '1'
            and all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),
            'C38 cold isolated loader environment differs')
    for key,value in session['runtime_environment'].items():
        require(os.environ.get(key) == value,'C38 loader environment changed: '+key)
    mujoco,binding,geometry = bootstrap_components38()
    require(str(Path(binding.__file__).resolve()) == session['binding_module']
            and identity(binding.__file__) == session['source_hashes'][session['binding_module']],
            'C38 engine binding source origin differs')
    cold = geometry.cache_info()._asdict()
    require(cold == {'hits':0,'misses':0,'maxsize':1,'currsize':0},'C38 nominal geometry cache was not cold')
    guard = binding.EngineGuard(session['library'])
    before = guard.snapshot();zero_state38(before)
    proof = guard.binding_proof()
    after = guard.snapshot();zero_state38(after)
    zero_state38(proof.get('counters'))
    require(proof.get('passed') is True and proof.get('dso_path') == session['library']
            and proof.get('dso_sha256') == session['source_hashes'][session['library']]['sha256']
            and len(proof.get('jump_slots',())) == 4
            and all(row.get('passed') is True and row.get('target') == row.get('expected_wrapper')
                    and row.get('elf_sha256') == session['source_hashes'][row['elf']]['sha256']
                    for row in proof['jump_slots'])
            and all(session['source_hashes'][name]['sha256'] == digest for name,digest in proof['elf_hashes'].items()),
            'C38 actual ELF/GOT binding proof failed')
    require(geometry.cache_info()._asdict() == cold and backend.frameworks_absent38(),'C38 bootstrap built geometry or loaded forbidden framework')
    origins = backend.helper36().dependency_origins(session['source_hashes'],{'numpy','mujoco'})
    value = {'schema':'b22-numpy38-bootstrap-v1','execution_contract_id':CONTRACT,'status':'COMPLETE',
        'session_identity':session_identity,
        'binding_proof':proof,'state_before':before,'state_after':after,'nominal_cache':cold,
        'forbidden_modules_absent':True,'forbidden_prefixes':sorted(backend.FORBIDDEN),'module_origins':origins,
        'actual_counts':{'controls':0,'normal_native':0,'compiler_native':0},
        'actual_nn_counts':{'predict_calls':0,'predict_rows':0,'ppo_load':0,'torch_load':0},
        'environment_constructions':0,'numpy_actor_bundle_loads':0,'qualification_granted':False,
        'mujoco_version':mujoco.__version__,
        'scope':'new no-Torch binding bootstrap; no old OpenCV mutation receipt; original_step is a resolved pointer, not a native-call counter'}
    write_json(Path(output)/'bootstrap38_report.json',value)
    return value


def replace_once(text,old,new):
    require(text.count(old) == 1,'C38 frozen derivation seam changed: '+old)
    return text.replace(old,new)


def derived_source38():
    text = ORIGINAL.read_text()
    require(not ORIGINAL.is_symlink() and hashlib.sha256(text.encode()).hexdigest() == ORIGINAL_SHA,
            'C38 pinned C34 owner changed')
    text = text.replace('gain34','gain38')
    edits = (
        ("'C34_B22_residual_gain_probe_v1'",repr(CONTRACT)),
        ("session['gain38'] not in (1.0,.5)","session['gain38'] != .5"),
        ('common._load_import_repair(session,output)','_bootstrap38(session,output)'),
        ('_control_step_caller_verified,_audited_learning_origins)',
         '_control_step_caller_verified)\n    from b22_numpy_backend38 import audited_origins38 as _audited_learning_origins'),
        ('from runtime_support22 import ModelCounter','from b22_numpy_backend38 import NumpyCounter38 as ModelCounter'),
        ('from rl16_learning_11 import load_and_verify_final','from b22_numpy_backend38 import load_and_verify_numpy38'),
        ('policy,load_report=load_and_verify_final(','policy,load_report=load_and_verify_numpy38('),
        ("'probe_actions_byte_exact'):","'probe_actions_within_tolerance'):"),
        ("'recipes18','run_rl16_training_08','rl16_learning_11','state21')",
         "'recipes18','run_rl16_training_08','b22_numpy_backend38','state21')"),
        ("(1 if policy is not None else 0)\n                                or counts.get('torch_load'", "0\n                                or counts.get('torch_load'"),
        ("(3 if policy is not None else 0)\n                                or any(counts.get(key", "0\n                                or any(counts.get(key"),
    )
    for old,new in edits:
        text = replace_once(text,old,new)
    receipt = "'schema':'d1-c23-gui-worker-receipt-v1'"
    require(text.count(receipt) == 2,'C38 worker receipt version seam changed')
    text = text.replace(receipt,"'schema':'b22-numpy38-worker-receipt-v1'")
    # This inner owner remains a derivative, not an old C34 runtime claim.
    return text.replace('C34 fixed two-profile residual gain probe','C38 NumPy-only fixed two-profile half-gain probe')


def derivation_proof38(source):
    return {'schema':'b22-numpy38-source-derivation-v1','execution_contract_id':CONTRACT,
        'original_source':str(ORIGINAL),'original_sha256':ORIGINAL_SHA,
        'derived_ast_sha256':hashlib.sha256(ast.dump(ast.parse(source),include_attributes=False).encode()).hexdigest(),
        'changed_seams':['C38 admission/gain field/receipt','no-Torch bootstrap','NumPy counter/loader/origins',
                         'zero PPO/Torch load counts; preserve32 probe and every control prediction'],
        'unchanged_scope':'prepared99 builder, controller/PI/z/gain arithmetic, stop logic, native owner, archive and original task thresholds'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--bootstrap',action='store_true')
    args = parser.parse_args(argv)
    if args.session is None and args.output is None:
        print(json.dumps({'status':'PLAN_ONLY','execution_contract_id':CONTRACT,'execution_started':False,
            'bootstrap_native':0,'script_controls':3200,'normal_native':16000,'compiler_native':4,
            'numpy_predict_calls':3202,'numpy_predict_rows':3264,'ppo_load':0,'torch_load':0,'qualification_granted':False}))
        return 0
    require(args.session is not None and args.output is not None,'C38 explicit supervised session/output required')
    session_path,output = args.session.resolve(strict=True),args.output.resolve(strict=True)
    backend.deny_frameworks38()
    if args.bootstrap:
        raw = json.loads(session_path.read_text())
        runner = ROOT/'scripts/run_b22_numpy38.py'
        require(identity(runner) == raw['source_hashes'].get(str(runner)),'C38 bootstrap supervisor source not bound')
        spec = importlib.util.spec_from_file_location('b22_numpy38_supervisor_bound',runner)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        session = module.validate_bootstrap_launch38(session_path,output)
        require(session.get('schema') == 'b22-numpy38-bootstrap-session-v1'
                and session.get('execution_contract_id') == CONTRACT,'C38 cold bootstrap session identity required')
        for path in (Path(__file__).resolve(),Path(backend.__file__).resolve(),Path(session['binding_module']),Path(session['library'])):
            require(identity(path) == session['source_hashes'].get(str(path)),'C38 bootstrap critical input not bound: '+str(path))
        bootstrap38(session,output)
        return 0
    source = derived_source38()
    write_json(output/'derived_sources38.json',derivation_proof38(source))
    namespace = {'__name__':'__main__','__file__':str(Path(__file__).resolve()),'_bootstrap38':bootstrap38}
    exec(compile(source,namespace['__file__'],'exec'),namespace)  # noqa: S102 -- explicit SHA-pinned C34 source derivative
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
