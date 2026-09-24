"""Pure immutable input binding for the five predetermined continuation cases."""
import argparse
import hashlib
import json
from pathlib import Path

W=Path(__file__).resolve().parent
R=Path('/home/lyh/wheel-legged-control-lab')
E=W.parent/'stability_20260923_engine01'
CONTRACT_SHA='6be5ebc4557f124db74c7f6441ae624e70eb066f6efc94923a4269f0160dc194'
MODEL_SHA='4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e'
def record(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def load(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);a=p.parse_args()
    contract=W/'next_eval_continuation_plan_03/next_contract.md'
    assert record(contract)['sha256']==CONTRACT_SHA
    assert 'GO' in a.review.read_text()
    old=W/'root_execution_freeze_02.json';source=load(old);files=dict(source['files'])
    for name,row in files.items():assert record(name)==row,name
    archive=W/'run_02_interrupted_archive_manifest_01.json';archived=load(archive)['files']
    assert len(files)==457 and len(archived)==55
    for relative,row in archived.items():
        source_path=W/'run_02'/relative;assert record(source_path)==row,relative;files[str(source_path)]=row
    snapshot=load(a.snapshot)
    for name,row in snapshot['files'].items():assert record(name)==row,name;files[name]=row
    checkpoint=W/'run_02/training/final_checkpoint/model.zip'
    assert record(checkpoint)['sha256']==MODEL_SHA
    output=W/'run_03';budget=W/'budget_activation_03.json';freeze=W/'root_execution_freeze_03.json'
    worker=W/'run_evaluation_continuation_03.py';library=E/'build/libepa01_engine.so'
    assert not output.exists() and not budget.exists() and not freeze.exists()
    python_argv=[str(worker),'--library',str(library),'--original-run',str(W/'run_02'),
                 '--output',str(output),'--contract',str(contract),'--freeze',str(freeze)]
    argv=['rtk','proxy','env','-u','LD_LIBRARY_PATH',f'LD_PRELOAD={library}','LD_BIND_NOW=1',
          'PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
          f'PYTHONPATH=/home/lyh/.local/lib/python3.10/site-packages:{R}/.local-deps:{R}/src:{R}:{E}',
          'python3','-B',*python_argv]
    for source_path in [Path(__file__),worker,W/'launch_continuation_once_03.py',contract,a.review,a.snapshot,old,
                        archive,W/'run_02_boundary_closure_01.json',W/'run_02_interruption_observation_01.json',
                        W/'root_run_02_interrupted_readback_01.json',W/'root_interrupted_readback_01.py']:
        files[str(source_path.resolve())]=record(source_path)
    value={'schema':'rolling-evaluation-continuation-freeze-v1','astra_decision':'GO',
           'process_limit':1,'training_control_limit':0,'evaluation_control_limit':6000,
           'normal_native_limit':30000,'compiler_native_limit':5,'wallclock_limit_s':600,
           'contract_path':str(contract),'contract_sha256':CONTRACT_SHA,
           'prior_boundary_closure':str(W/'run_02_boundary_closure_01.json'),
           'original_run':str(W/'run_02'),'output_directory':str(output),'budget_path':str(budget),
           'original_freeze_path':str(old),'original_freeze_sha256':record(old)['sha256'],
           'original_archive_manifest_path':str(archive),'original_archive_manifest_sha256':record(archive)['sha256'],
           'original_training_protocol_sha256':'494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733',
           'checkpoint_sha256':MODEL_SHA,'python_argv':python_argv,'execution_argv':argv,
           'files':files,'old_original_run_02_qualification':False,'old_final_native_C_counts':None}
    with freeze.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps({'files':len(files),'freeze_sha256':record(freeze)['sha256']}))

if __name__=='__main__':main()
