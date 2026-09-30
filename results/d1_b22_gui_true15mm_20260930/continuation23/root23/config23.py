"""Build the relocatable research-code closure for this pinned local engine stack."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

R=Path('/home/lyh/wheel-legged-control-lab')
W=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01')
B=R/'runtime/d1_b22'
SHA='7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'


def identity(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    return {'bytes':Path(path).stat().st_size,'sha256':digest.hexdigest()}


def configuration():
    source=json.loads((W/'continuation22/final_1/session.json').read_text())
    inputs={}
    for name,expected in source['source_hashes'].items():
        path=Path(name)
        if path.is_relative_to(W):
            target=B/'research'/path.relative_to(W)
            if not name.endswith('.py') or not target.exists():
                continue
            if identity(target)!=expected:
                raise ValueError('bundle legacy source differs from C22: '+str(target))
            inputs[str(target)]=expected
        elif path.is_relative_to(Path('/home/lyh/wheel-legged-control-lab-work')):
            # Old experimental evidence is provenance, never a runtime dependency.
            continue
        else:
            if identity(path)!=expected:
                raise ValueError('pinned engine/dependency/source changed: '+name)
            inputs[name]=expected
    for path in B.rglob('*'):
        if path.is_file() and path.name!='runtime_manifest.json' and '__pycache__' not in path.parts:
            inputs[str(path)]=identity(path)
    for path in (R/'scripts/run_d1_b22.py',):
        inputs[str(path)]=identity(path)
    # C22 never rendered; freeze newly used GUI and PNG implementation bytes.
    site=Path('/home/lyh/.local/lib/python3.10/site-packages')
    for folder in (site/'glfw',site/'PIL',site/'pillow.libs'):
        if not folder.is_dir():
            raise ValueError('missing current GUI dependency: '+str(folder))
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                actual=path.resolve(strict=True)
                inputs[str(actual)]=identity(actual)
    for name in ('libX11.so.6','libXtst.so.6'):
        actual=(Path('/usr/lib/x86_64-linux-gnu')/name).resolve(strict=True)
        inputs[str(actual)]=identity(actual)
    runtime_env=dict(source['runtime_environment'])
    runtime_env['PYTHONPATH']=':'.join(
        str(B/'research'/Path(part).relative_to(W)) if Path(part).is_relative_to(W) else part
        for part in runtime_env['PYTHONPATH'].split(':'))
    runtime_env['PYTHONPATH']+=':'+':'.join(str(B/'research'/relative) for relative in (
        'gui12','continuation13/gui13_revision02','continuation23/gui23',
        'continuation23/root23','continuation24'))
    manifest=json.loads((B/'checkpoint_manifest.json').read_text())
    original_folder=manifest['folder']
    manifest={**manifest,'folder':str(B/'checkpoint')}
    if identity(B/'checkpoint/final_model.zip')['sha256']!=SHA:
        raise ValueError('actual bundled B22 checkpoint differs')
    shared={'library':source['library'],'binding_module':source['binding_module'],
        'continuation_import_repair':source['continuation_import_repair'],
        'controller_variant':'combined','checkpoint_folder':str(B/'checkpoint'),
        'checkpoint_manifest':manifest,'checkpoint_sha256':SHA,
        'reference_construction_path':str(B/'reference_C22_construction.json'),
        'checkpoint_relocation':{'original_folder':original_folder,
            'current_folder':str(B/'checkpoint'),
            'original_manifest_path':str(B/'checkpoint_manifest.json'),
            'original_manifest_identity':identity(B/'checkpoint_manifest.json'),
            'changed_fields':['folder'],'payload_bytes_unchanged':True},
        'normal_native_per_control':5,'compiler_native_limit':2}
    return {'schema':'d1-b22-current-runtime-bundle-v1',
        'worker':str(B/'research/continuation23/gui23/run_gui23.py'),
        'host':str(B/'research/continuation23/root23/host23.py'),
        'source_hashes':inputs,'runtime_environment':runtime_env,'shared_session':shared,
        'model_limit_keys':['backward','evaluate_actions','forward','learn','load','predict',
                            'predict_values','save','torch_load','train'],
        'profile_seeds':dict(flat_0p6=181201,flat_1p6=181202,yaw_1p2=181203,
                             bumps_0p4=181204,rough_0p35=181205,ramp_0p45_complete=181206),
        'mirror_seed':181403,
        'qualification_scope':'see C23 and C24 saved independent readers; no RL superiority claim',
        'platform':'pinned local Linux/Python3.10/MuJoCo engine and numerical dependencies',
        'external_research_work_directory_required':False}
