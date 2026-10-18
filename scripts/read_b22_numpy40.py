"""Independent, saved-only C40 adjudication of the closed C39 flat episode."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'results/b22_numpy39_20261018_01'
OUTPUT = ROOT/'results/b22_numpy40_readback_20261018_01'
STAGE = 'flat_1p6_B'
CONTRACT = 'C40_B22_saved_numpy39_flat_readback_v1'
AUTHORED = ('scripts/read_b22_numpy40.py','tests/test_b22_numpy40.py','docs/b22_numpy40.md')
SOURCE39 = {
    'scripts/run_b22_numpy39.py':'d946e7a28ba7f1da44bc34d9668f9a1d27c08a5294e7a6aa20f48b9fca7ed4a5',
    'tests/test_b22_numpy39.py':'9c1ecd52748b3b770d4cf8d63e82c1eaa22cf5e9327cbbffde3e713f275f6f2b',
    'docs/b22_numpy39.md':'ba6b851f769e85726bc3f54c7a2dcd0875e48d61a5b1a5598e0f98137a958aed',
    'scripts/b22_runtime_packages39.py':'7d5111bb4b6ecddc2ecf67ee6dc83197c68045eca00baae283e7229405b72868',
    'tests/test_b22_runtime_packages39.py':'1f13e63d93752804a765c90a5aef79c3613dba7d5f3bd0535c9b5c83590b0d5c',
}
RECORD39 = {
    'release.json':'550e31df211516ec476e6c9769e9620363690e02361d4664077713d1bed326b2',
    'report.json':'c1b175e5b986e5097bcfb1c08aa45b91da5f6fa513d9ca0a188aaa03ebb3571f',
    'flat_1p6_B/worker_receipt.json':'4977565ecd0e7acb384407fa73b3882e17376cf0c140606830341c53f1311553',
    'bootstrap/bootstrap38_report.json':'2d15f02d41f74a644fe66a2b1766a5142bc40592468f0d436302322041cd096b',
}
SOURCE_COMMIT39 = 'bb23ef1d08435108e8bef70ca81e277175984671'
EXPECTED_COUNTS = {'controls':1600,'normal_native':8000,'compiler_native':2}
ZERO = {'controls':0,'normal_native':0,'compiler_native':0,'predict_calls':0,'predict_rows':0,
        'ppo_load':0,'torch_load':0}
FORBIDDEN = {'torch','stable_baselines3','mujoco','dm_control','cv2','onnxruntime','jax','tensorflow'}
LIMITS = {'wall_s':600,'start_effective_bytes':6 << 30,'reserve_effective_bytes':4 << 30,
          'required_nice':10,'resource_poll_s':1.,'physics_calls':0,'nn_forward_calls':0}
MAP_FILES = ('mapped_libraries_before_load.json','mapped_libraries_final.json')


def require(value,message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.is_file(), 'C40 nonregular input: '+str(path))
    digest,size = hashlib.sha256(),0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block);size += len(block)
    return {'bytes':size,'sha256':digest.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def tree_identity(folder):
    folder = Path(folder)
    require(folder.is_absolute() and folder.resolve(strict=True) == folder and not folder.is_symlink(),
            'C40 canonical original output required')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(),'C40 original output symlink: '+str(path))
        if path.is_dir():
            continue
        result[str(path)] = identity(path)
    require(result,'C40 original output is empty')
    return result


def source_identity():
    subprocess.run(['git','ls-files','--error-unmatch',*AUTHORED],cwd=ROOT,check=True,
                   stdout=subprocess.DEVNULL,timeout=5)
    subprocess.run(['git','diff','--quiet','HEAD','--',*AUTHORED],cwd=ROOT,check=True,timeout=5)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    return {'git_commit':commit,'authored':{str(ROOT/name):identity(ROOT/name) for name in AUTHORED}}


def load_graph39():
    for name,digest in SOURCE39.items():
        require(identity(ROOT/name)['sha256'] == digest,'C40 frozen C39 source changed: '+name)
    path = ROOT/'scripts/run_b22_numpy39.py'
    spec = importlib.util.spec_from_file_location('c40_pinned39_driver',path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    # This also pins all eight original C38 sources before compiling anything.
    return driver,driver.build_scope()


def adapt_maps40(text):
    """Change one actual script_sources loop; preserve every other AST node."""
    original = ast.parse(text)
    tree = copy.deepcopy(original)
    loops = [node for node in ast.walk(tree) if isinstance(node,ast.For)
             and isinstance(node.iter,ast.Tuple)
             and [getattr(item,'value',None) for item in node.iter.elts] == list(MAP_FILES)]
    require(len(loops) == 1,'C40 exact original Torch-map loop missing')
    loop = loops[0]
    expected = ast.parse("for filename in ('mapped_libraries_before_load.json','mapped_libraries_final.json'):\n"
        "    mapped = base.saved_document(run/filename)['paths']\n"
        "    c.require(mapped and all(path in sources for path in mapped), 'actual DSO left closure')\n").body[0]
    def dump(node):
        return ast.dump(node,include_attributes=False)
    require(dump(loop) == dump(expected),'C40 original Torch-map guard changed')
    old = copy.deepcopy(loop.body)
    loop.body = ast.parse("document40 = base.saved_document(run/filename)\n"
        "c.require(type(document40) is dict and document40 == {'paths':[]}, "
        "'C40 NumPy execution must have exactly empty Torch/Nvidia mappings')").body
    result = ast.fix_missing_locations(copy.deepcopy(tree))
    loop.body = old
    require(dump(tree) == dump(original),'C40 changed another metadata guard')
    return ast.unparse(result)+'\n'


def install_adapter40(driver,graph):
    tree,_ = driver.derive('reader')
    original = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'metadata_checker')
    node = copy.deepcopy(original)
    indexes = [i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id == 'namespace' for t in n.targets)]
    require(len(indexes) == 1,'C40 C39 metadata namespace seam changed')
    node.body.insert(indexes[0],ast.parse('text = _adapt_maps40(text)').body[0])
    namespace = dict(graph.reader.__dict__,_adapt_maps40=adapt_maps40)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(Path(__file__).resolve()),'exec'),namespace)  # noqa: S102 -- pinned C39 function, one inverse-checked metadata seam
    graph.reader.metadata_checker = namespace['metadata_checker']
    emitted = copy.deepcopy(node)
    node.body.pop(indexes[0])
    require(ast.dump(node,include_attributes=False) == ast.dump(original,include_attributes=False),
            'C40 metadata adapter changed another function statement')
    return {'schema':'b22-numpy40-reader-derivation-v1','source39_sha256':SOURCE39['scripts/run_b22_numpy39.py'],
        'metadata39_ast_sha256':hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
        'adapter40_ast_sha256':hashlib.sha256(ast.dump(emitted,include_attributes=False).encode()).hexdigest(),
        'inverse_ast_exact':True,'only_semantic_seam':'two Torch/Nvidia map documents require exact empty paths',
        'unchanged':'all C39 source/origin/GOT/probe/model/native/controller/complete-window/scoring checks'}


def check_original39():
    for name,digest in RECORD39.items():
        require(identity(ORIGINAL/name)['sha256'] == digest,'C40 original C39 receipt changed: '+name)
    release,parent = read_json(ORIGINAL/'release.json'),read_json(ORIGINAL/'report.json')
    require(release['schema'] == 'b22-numpy39-release-v1' and release['source']['git_commit'] == SOURCE_COMMIT39
            and parent['schema'] == 'b22-numpy39-supervision-v1' and parent['status'] == 'INCOMPLETE'
            and parent['actual_counts'] is None and parent['actual_nn_counts'] is None
            and parent['cleanup']['remaining'] == {} and not (ORIGINAL/'yaw_1p2_B').exists(),
            'C40 requires the fixed interrupted C39 attempt, not a relabelled run')
    return release,parent


class DenyRuntime40(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.partition('.')[0] in FORBIDDEN:
            raise ImportError('C40 saved-only reader forbidden import: '+fullname)


def deny_runtime40():
    require(not any(name.partition('.')[0] in FORBIDDEN for name in sys.modules),
            'C40 requires a cold saved-only process')
    sys.meta_path.insert(0,DenyRuntime40())


def plan():
    return {'schema':'b22-numpy40-plan-v1','status':'PLAN_ONLY','execution_contract_id':CONTRACT,
        'original_output':str(ORIGINAL),'output':str(OUTPUT),'stage':STAGE,'limits':dict(LIMITS),
        'execution_started':False,'new_actual_counts':dict(ZERO),'retry_permitted':False,
        'qualification_granted':False,'scope':'saved-only reanalysis of C39 flat; no yaw, model, or physics'}


def execute(*, graph_loader=load_graph39, source_reader=source_identity, clock=time.monotonic):
    started = clock()
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(),'C40 exclusive new output required; no retry')
    source = source_reader()
    deny_runtime40()
    require(all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),
            'C40 single-thread numeric environment required')
    driver,graph = graph_loader()
    require(os.getpriority(os.PRIO_PROCESS,0) >= LIMITS['required_nice'],'C40 requires external Nice10')
    available = graph.runner.supervision.memory()
    require(type(available) is int and available >= LIMITS['start_effective_bytes'],'C40 start RAM below6GiB/unknown')
    OUTPUT.mkdir(exist_ok=False)
    report = {'schema':'b22-numpy40-saved-readback-v1','status':'RUNNING','execution_contract_id':CONTRACT,
        'analyzed_execution_contract_id':'C39_B22_numpy_half_gain_v1','source40':source,'new_actual_counts':dict(ZERO),
        'original39_parent_status':'INCOMPLETE','original39_actual_counts':None,'original39_actual_nn_counts':None,
        'verified_saved_execution_counts':None,'verified_saved_nn_counts':None,'qualification_granted':False,
        'yaw_evaluated':False,'retry_permitted':False,'failure':None,
        'cleanup':{'training_or_physics_children_spawned':0}}
    write_json(OUTPUT/'report.json',report)
    prior = {sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
    def sample(signum=None,frame=None):
        elapsed,ram = clock()-started,graph.runner.supervision.memory()
        require(type(ram) is int and ram >= 0,'C40 effective RAM unknown')
        report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes',ram),ram)
        with (OUTPUT/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed_s':elapsed,'effective_available_bytes':ram})+'\n')
        require(ram >= LIMITS['reserve_effective_bytes'],'C40 runtime RAM below4GiB')
        require(elapsed < LIMITS['wall_s'],'C40 readback wall limit exhausted')
    def stop(signum,frame):
        raise InterruptedError('C40 interrupted: '+signal.Signals(signum).name)
    for sig in prior:signal.signal(sig,sample if sig == signal.SIGALRM else stop)
    signal.setitimer(signal.ITIMER_REAL,LIMITS['resource_poll_s'],LIMITS['resource_poll_s'])
    try:
        sample()
        release,_ = check_original39()
        frozen = tree_identity(ORIGINAL)
        proof = install_adapter40(driver,graph)
        inputs = {'schema':'b22-numpy40-inputs-v1','source40':source,'source39':release['source'],
            'frozen39_payloads':frozen,'original39_bound_inputs':release['inputs'],'derivation40':proof}
        write_json(OUTPUT/'inputs.json',inputs)
        report['inputs_identity'] = identity(OUTPUT/'inputs.json')
        # No new checkpoint, evaluator, model, bootstrap, or physical session.
        saved = graph.reader.read_stage(ORIGINAL/'release.json',STAGE)
        require(saved['actual_counts'] == EXPECTED_COUNTS and saved['numpy_predict_calls'] == 1601
                and saved['numpy_predict_rows'] == 1632 and saved['ppo_load'] == saved['torch_load'] == 0,
                'C40 saved complete execution counters differ')
        require(saved['reader_physics_calls'] == saved['reader_nn_forward_calls'] == 0,
                'C40 inherited reader is not saved-only')
        require(tree_identity(ORIGINAL) == frozen and source_reader() == source,'C40 source/payload changed during reanalysis')
        graph.runner.validate_release(ORIGINAL/'release.json')
        check_original39()
        sample()
        report.update(status='COMPLETE',derivation40=proof,stage=STAGE,cases=saved['cases'],
            pairing_identity=saved['pairing_identity'],verified_saved_execution_counts=dict(EXPECTED_COUNTS),
            verified_saved_nn_counts={'predict_calls':1601,'predict_rows':1632,'ppo_load':0,'torch_load':0},
            original39_numerical_reader_identity=saved['reader_identity'],artifacts=saved['artifacts'],
            scope='new C40 saved adjudication; numerical execution remains C39; no original C39 success artifact is created')
    except BaseException as error:  # noqa: BLE001 -- preserve interruption in this new output only
        report.update(status='INCOMPLETE',failure=type(error).__name__+': '+str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        report['elapsed_s'] = clock()-started
        if report['elapsed_s'] >= LIMITS['wall_s']:
            report.update(status='INCOMPLETE',failure='C40 final wall limit exhausted')
        try:
            write_json(OUTPUT/'report.json',report)
        finally:
            for sig,handler in prior.items():signal.signal(sig,handler)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    value = execute() if args.execute else plan()
    print(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    return 0 if value['status'] in ('PLAN_ONLY','COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
