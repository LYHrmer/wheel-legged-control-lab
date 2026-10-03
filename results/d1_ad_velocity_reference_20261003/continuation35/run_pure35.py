"""Root-only finite pure checks, independent process deadlines, immutable logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

C=Path(__file__).resolve().parent
R=Path('/home/lyh/wheel-legged-control-lab')


def identity(p):
    raw=Path(p).read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--group',choices=['all','root35','sol35','read35','origins','cli'],default='all')
    p.add_argument('--pattern',default='test_*pure.py')
    p.add_argument('--method')
    p.add_argument('--origins-output', type=Path)
    a=p.parse_args()
    if '/' in a.name or not a.name.startswith('pure_'):
        raise ValueError('exclusive pure receipt name required')
    previous=list(C.glob('pure_*/receipt.json'))
    prior=sum(json.loads(x.read_text())['elapsed_s'] for x in previous)
    if prior >= 90:
        raise RuntimeError('C35 cumulative pure budget exhausted')
    out=C/a.name
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    source_dirs = ('root35','sol35') if a.group in ('root35','sol35','cli') else ('root35','sol35','read35')
    source=list(C.glob('*.py'))+[p for d in source_dirs for p in (C/d).glob('*.py')]
    before={str(p):identity(p) for p in source}
    env=os.environ.copy()
    for k in ('LD_PRELOAD','LD_LIBRARY_PATH','DISPLAY','XAUTHORITY'):
        env.pop(k,None)
    parent=json.loads((C.parent/'continuation34/source_go34.json').read_text())
    old=parent['runtime_environment']['PYTHONPATH'].split(':')
    env.update(PYTHONPATH=':'.join(dict.fromkeys([old[0],*[str(C/d) for d in ('root35','sol35','read35')],*old[1:]])),
        PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[]
    if a.group not in ('origins','cli'):
        groups=('root35','sol35','read35') if a.group=='all' else (a.group,)
        if a.method:
            commands.append(['rtk','proxy','/usr/bin/python3','-B','-m','unittest',a.method,'-v'])
        else:
            for d in groups:
                if d == 'read35':
                    files = sorted((C/d).glob(a.pattern))
                    if not files:
                        raise RuntimeError('No C35 pure reader test files found')
                    commands.append(['rtk','proxy','/usr/bin/python3','-B','-m','pytest',
                                     '-q','-p','no:cacheprovider',*map(str,files)])
                else:
                    commands.append(['rtk','proxy','/usr/bin/python3','-B','-m','unittest','discover',
                                     '-s',str(C/d),'-p',a.pattern,'-v'])
        commands.append(['rtk','proxy',str(R/'.local-deps/bin/ruff'),'check','--no-cache',
                         '--select','E9,F63,F7,F82',*map(str,source)])
    if a.group in ('all','origins'):
        commands.append(['rtk','proxy','/usr/bin/python3','-B',str(C/'reader_origins35.py')]
                        + (['--output',str(a.origins_output)] if a.origins_output else []))
    if a.group == 'cli':
        commands.extend([['rtk','proxy','/usr/bin/python3','-B',str(C/'root35'/name),'--help']
                         for name in ('worker35.py','host35.py','build_request35.py')])
    outcomes=[]
    failure=None
    try:
        for i,cmd in enumerate(commands):
            began=time.monotonic()
            result=subprocess.run(cmd,cwd=C,env=env,stdin=subprocess.DEVNULL,capture_output=True,
                                  timeout=max(.1,90-prior-(began-started)))
            (out/f'{i:02d}.stdout').write_bytes(result.stdout)
            (out/f'{i:02d}.stderr').write_bytes(result.stderr)
            outcomes.append(dict(command=cmd,exit_code=result.returncode,elapsed_s=time.monotonic()-began))
            if result.returncode:
                raise RuntimeError(f'check {i} returned {result.returncode}')
    except BaseException as error:
        failure=repr(error)
    after={str(p):identity(p) for p in source}
    elapsed=time.monotonic()-started
    receipt=dict(schema='d1-c35-root-pure-checks-v1',prior_elapsed_s=prior,elapsed_s=elapsed,
        cumulative_elapsed_s=prior+elapsed,limit_s=90,source_before=before,source_after=after,
        source_unchanged=before==after,results=outcomes,failure=failure,
        new_model_calls=0,new_physics_steps=0,fits=0,
        passed=failure is None and before==after and prior+elapsed<=90)
    with (out/'receipt.json').open('x') as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:receipt[k] for k in ('passed','failure','elapsed_s','cumulative_elapsed_s','source_unchanged')}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
