"""Bind frozen inputs without importing model or experiment code."""
import argparse
import hashlib
import json
import resource
import signal
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    resource.setrlimit(resource.RLIMIT_CPU,(29,30));signal.alarm(60)
    start=time.process_time();groups=[];failures=[]
    for root,relative in [('/home/lyh/ddt_ros2_ws','docs/experiments81/objective81_preregistered.json'),
                          ('/home/lyh/wheel-legged-control-lab','reports/control_candidates49_preregistration.json')]:
        base=Path(root);plan=base/relative;data=plan.read_bytes();manifest=json.loads(data)['source_sha256']
        for name,expected in manifest.items():
            actual=hashlib.sha256((base/name).read_bytes()).hexdigest()
            if actual!=expected:failures.append({'root':root,'path':name,'expected':expected,'actual':actual})
        groups.append({'root':root,'plan':relative,'plan_sha256':hashlib.sha256(data).hexdigest(),'files':len(manifest)})
    path=Path('/home/lyh/wheel-legged-control-lab/runtime/d1_b22/runtime_manifest_v27.json')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    expected='d3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09'
    if actual!=expected:failures.append({'path':str(path),'expected':expected,'actual':actual})
    result={'status':'PASS' if not failures else 'FAIL','groups':groups,'v27_sha256':actual,
            'failures':failures,'cpu_seconds':time.process_time()-start,'new_physics_steps':0,
            'new_model_calls':0,'new_parameter_updates':0}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result));return bool(failures)


if __name__=='__main__':raise SystemExit(main())
