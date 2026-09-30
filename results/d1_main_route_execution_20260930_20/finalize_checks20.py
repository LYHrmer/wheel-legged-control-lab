"""Root-only final necessary checks and command sealing, zero robot calls."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from worker20 import identity, write

C=Path(__file__).resolve().parent
W=C.parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision',type=int,required=True)
    args=parser.parse_args()
    files=sorted(C.glob('*.py'))
    for path in files:
        ast.parse(path.read_text(),filename=str(path))
    frozen={str(p):identity(p)['sha256'] for p in files}
    env=dict(os.environ)
    for k,v in json.loads((W/'rl11/training_run_01/session.json').read_text())['runtime_environment'].items():
        if v is None: env.pop(k,None)
        else: env[k]=v
    for k in ('LD_PRELOAD','LD_LIBRARY_PATH'):env.pop(k,None)
    env['PYTHONPATH'] += ':'+':'.join(map(str,(C,W/'continuation18',W/'continuation15',W/'continuation13')))
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1')
    existing=[]
    # Each relevant passing source is retained; changed reader source is checked
    # independently below, while unchanged scorer tests need no repetition.
    for name,names in (
        ('pure_learning_recipes_01.json',('recipes20.py','test_recipes20_pure.py')),
        ('pure_recording_decision_02.json',('curriculum20.py','runtime_support20.py','state20.py','test_recording20_pure.py','decision20.py','test_decision20_pure.py')),
        ('pure_eval_input_01.json',('learning20.py','checkpoint20.py','test_learning20_pure.py',
            )),
        ('pure_score_02.json',('score20.py','test_eval20_pure.py'))):
        receipt=json.loads((C/name).read_text())
        if receipt['exit_code']!=0:
            raise ValueError('required earlier check failed: '+name)
        for source in names:
            if receipt['source_sha256'][str(C/source)]!=identity(C/source)['sha256']:
                raise ValueError('necessary tested source changed: '+source)
        existing.append({'receipt':str(C/name),'identity':identity(C/name),'sources_validated':list(names)})
    commands=[
        ['rtk','proxy','/usr/bin/python3','-B','-m','pytest','-q','-p','no:cacheprovider',str(C/'test_training_reader20_pure.py')],
        ['rtk','proxy','/usr/bin/python3','-B',str(C/'offline_floor20.py'),'--probe-import','--output',str(C/f'pure_reader_import_20_{args.revision}.json')],
        ['rtk','proxy','/usr/bin/python3','-B','-c',
         'import sys; from offline_floor20 import _ForbidModelPhysicsImports, _assert_clean; '
         '_assert_clean("before"); sys.meta_path.insert(0,_ForbidModelPhysicsImports()); '
         'import read_training20; _assert_clean("after"); print("training reader clean import passed")'],
        ['rtk','proxy','/usr/bin/python3','-B','-m','ruff','check','--select','E9,F63,F7,F82',*map(str,files)],
    ]
    results=[]
    for i,cmd in enumerate(commands):
        t=time.monotonic();error=None;code=None
        logpath=C/f'final_check_{args.revision}_{i}.log'
        try:
            with logpath.open('xb') as log:
                p=subprocess.run(cmd,env=env,cwd=C,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=60)
                code=p.returncode
        except Exception as exc:error=repr(exc)
        result={'argv':cmd,'exit_code':code,'failure':error,'log':str(logpath),'elapsed_s':time.monotonic()-t}
        results.append(result)
        print(json.dumps(result),flush=True)
        if code!=0:
            print(logpath.read_text(),flush=True)
    changed=[path for path,digest in frozen.items() if identity(path)['sha256']!=digest]
    passed=not changed and all(x['exit_code']==0 and x['failure'] is None for x in results)
    result={'source_sha256':frozen,'checks':results,'earlier_checks':existing,'changed_sources':changed,
            'passed':passed,'robot_model_calls':0,'controls':0,'normal_native':0,'compiler_native':0}
    write(C/f'final_checks_receipt_{args.revision}.json',result)
    if passed:
        write(C/'pure_tests_receipt_20.json',result)
        command=['rtk','proxy','/usr/bin/python3','-B','-c',
            'from recipes20 import serialized_tables; from worker20 import write; '
            'from pathlib import Path; write(Path("sealed_command_tables_20.json"),serialized_tables())']
        subprocess.run(command,env=env,cwd=C,check=True,timeout=60)
        print(json.dumps({'passed':True,'sealed_commands':identity(C/'sealed_command_tables_20.json')}))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
