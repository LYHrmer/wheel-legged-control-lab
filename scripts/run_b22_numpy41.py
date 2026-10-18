"""Independent41 yaw-only execution, with C40's saved C39 flat as old evidence."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'scripts/run_b22_numpy41.py'
OUTPUT=ROOT/'results/b22_numpy41_yaw_20261018_01'
CONTRACT='C41_B22_numpy_yaw_completion_v1'
AUTHORED=('scripts/run_b22_numpy41.py','tests/test_b22_numpy41.py','docs/b22_numpy41.md')
SOURCE39={
 'scripts/run_b22_numpy39.py':'d946e7a28ba7f1da44bc34d9668f9a1d27c08a5294e7a6aa20f48b9fca7ed4a5',
 'tests/test_b22_numpy39.py':'9c1ecd52748b3b770d4cf8d63e82c1eaa22cf5e9327cbbffde3e713f275f6f2b',
 'docs/b22_numpy39.md':'ba6b851f769e85726bc3f54c7a2dcd0875e48d61a5b1a5598e0f98137a958aed',
 'scripts/b22_runtime_packages39.py':'7d5111bb4b6ecddc2ecf67ee6dc83197c68045eca00baae283e7229405b72868',
 'tests/test_b22_runtime_packages39.py':'1f13e63d93752804a765c90a5aef79c3613dba7d5f3bd0535c9b5c83590b0d5c'}
SOURCE40={
 'scripts/read_b22_numpy40.py':'ba70475118e499abb7e383d1b1dc3e42c5a2461c4a90602a9582c5e517757bd4',
 'tests/test_b22_numpy40.py':'c35eee7ec71b8b7ce4d06a962d7b12ed1b9a78cdb3526187c68a9a3ca03d9de9',
 'docs/b22_numpy40.md':'4f626c45ad4ab4c5cb4af642609d52ac08efd01a04f8e5d5b03caac0cfa4530b'}
REUSED=ROOT/'results/b22_numpy40_readback_20261018_01'
REUSED_SHA={'report.json':'c08b9f05c72755289a6d97041ddaf4a2bd517799ffcd9f45e823eb9a8e0d6426',
            'inputs.json':'e9b566b8747b413c7222931c161c8081363a730e2f111470f6a5eb01f1458913'}
SOURCE_COMMIT40='635c2a8b525055301a2f9471f9bf375d5151d26d'
PHYSICAL={'controls':1600,'normal_native':8000,'compiler_native':2}
NN={'predict_calls':1601,'predict_rows':1632,'ppo_load':0,'torch_load':0}
REGISTRY='_b22_numpy41_verified_scope'


def require(value,message):
    if not value:raise ValueError(message)


def identity(path):
    path=Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),'C41 regular absolute input required')
    digest,size=hashlib.sha256(),0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block);size+=len(block)
    return {'bytes':size,'sha256':digest.hexdigest()}


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def pinned_sources():
    values={}
    for name,digest in {**SOURCE39,**SOURCE40}.items():
        p=ROOT/name;entry=identity(p)
        require(entry['sha256']==digest,'C41 frozen source changed: '+name);values[str(p)]=entry
    return values


def load40():
    pinned_sources()
    return load_file('b22_numpy41_saved40',ROOT/'scripts/read_b22_numpy40.py')


def reused40():
    """Bind successful40 and every original39/raw payload without new scoring/NN."""
    reader=load40();files=pinned_sources()
    for name,digest in REUSED_SHA.items():
        p=REUSED/name;entry=identity(p)
        require(entry['sha256']==digest,'C41 fixed C40 evidence changed: '+name);files[str(p)]=entry
    report,inputs=(reader.read_json(REUSED/name) for name in ('report.json','inputs.json'))
    require(report.get('schema')=='b22-numpy40-saved-readback-v1' and report.get('status')=='COMPLETE'
        and report.get('failure') is None and report.get('execution_contract_id')==reader.CONTRACT
        and report.get('analyzed_execution_contract_id')=='C39_B22_numpy_half_gain_v1'
        and report.get('source40')==inputs.get('source40')
        and report['source40'].get('git_commit')==SOURCE_COMMIT40
        and report.get('inputs_identity')==identity(REUSED/'inputs.json')
        and report.get('stage')=='flat_1p6_B' and report.get('original39_parent_status')=='INCOMPLETE'
        and report.get('original39_actual_counts') is None and report.get('original39_actual_nn_counts') is None
        and report.get('yaw_evaluated') is False and report.get('qualification_granted') is False
        and report.get('new_actual_counts')==reader.ZERO
        and all(type(v) is int and v==0 for v in report['new_actual_counts'].values())
        and report.get('verified_saved_execution_counts')==PHYSICAL
        and report.get('verified_saved_nn_counts')==NN
        and all(type(v) is int for group in (report['verified_saved_execution_counts'],report['verified_saved_nn_counts']) for v in group.values()),
        'C41 requires fixed complete40 saved-only flat evidence')
    require(report['source40'].get('authored')=={str(ROOT/name):identity(ROOT/name) for name in SOURCE40},
            'C41 C40 actual source identity differs')
    release,_=reader.check_original39()
    require(inputs.get('schema')=='b22-numpy40-inputs-v1' and inputs.get('source39')==release['source']
        and inputs.get('original39_bound_inputs')==release['inputs']
        and inputs.get('frozen39_payloads')==reader.tree_identity(reader.ORIGINAL),
        'C41 C40 original payload/source manifest differs')
    old_driver,old_graph=reader.load_graph39()
    require(report.get('derivation40')==inputs.get('derivation40')==reader.install_adapter40(old_driver,old_graph),
            'C41 C40 exact map-adapter proof differs')
    require(len(report.get('cases',[]))==1 and report['cases'][0].get('seed')==181202
            and report['cases'][0].get('completed_controls')==1600,'C41 C40 flat case is incomplete')
    for name,entry in inputs['original39_bound_inputs'].items():
        require(identity(name)==entry,'C41 old39 bound input changed: '+name);files[name]=entry
    files.update(inputs['frozen39_payloads'])
    for name,entry in reader.tree_identity(REUSED).items():files[name]=entry
    for name,entry in report['artifacts'].items():
        p=Path(name)
        require(not p.is_absolute() and '..' not in p.parts and identity(reader.ORIGINAL/'flat_1p6_B'/p)==entry,
                'C41 old flat artifact identity differs')
    return {'report_identity':identity(REUSED/'report.json'),'inputs_identity':identity(REUSED/'inputs.json'),
        'source40':report['source40'],'execution_contract_id':'C39_B22_numpy_half_gain_v1',
        'adjudication_contract_id':reader.CONTRACT,'stage':'flat_1p6_B',
        'reused_validated_case_counts':dict(PHYSICAL),'reused_validated_nn_counts':dict(NN),
        'original39_actual_counts':None,'original39_actual_nn_counts':None,'historical_total_resources_known':False,
        'inputs':files,'report':report}


def factory41():
    """Inverse-check identity-only adaptation of39's already audited factory."""
    pinned_sources()
    text=(ROOT/'scripts/run_b22_numpy39.py').read_text();original=ast.parse(text);tree=copy.deepcopy(original);edits=[]
    def replace(parent,field,index,new,label):
        old=getattr(parent,field)[index] if index is not None else getattr(parent,field)
        edits.append((parent,field,index,old,label))
        if index is None:setattr(parent,field,new)
        else:getattr(parent,field)[index]=new
    exact={'C39_B22_numpy_half_gain_v1':CONTRACT,'results/b22_numpy39_20261018_01':'results/b22_numpy41_yaw_20261018_01',
           'run_b22_numpy39.py':'run_b22_numpy41.py','scripts/run_b22_numpy39.py':'scripts/run_b22_numpy41.py',
           "'scripts/run_b22_numpy39.py'":"'scripts/run_b22_numpy41.py'",
           'b22_numpy_backend39':'b22_numpy_backend41','b22_numpy39_':'b22_numpy41_',
           '_b22_numpy39_verified_scope':REGISTRY,'derivation39':'derivation41'}
    for parent in list(ast.walk(tree)):
        for field,value in ast.iter_fields(parent):
            children=list(enumerate(value)) if isinstance(value,list) else [(None,value)]
            for index,child in children:
                if isinstance(child,ast.Constant) and isinstance(child.value,str):
                    new=exact.get(child.value,child.value)
                    if new.startswith('b22-numpy39-'):new='b22-numpy41-'+new[len('b22-numpy39-'):]
                    if new!=child.value:replace(parent,field,index,ast.Constant(new),'exact41 identity/path')
    # Authored files are new41 only. Original39/40 stay separate pinned legacy
    # inputs and retain their own historical identities.
    for index,node in enumerate(tree.body):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='AUTHORED':
            replace(node,'value',None,ast.parse(repr(AUTHORED),mode='eval').body,'new41 authored source set')
        if isinstance(node,ast.If) and ast.unparse(node.test)=="__name__ == '__main__'":
            replace(tree,'body',index,ast.Pass(),'new41 CLI owns entry')
    result=copy.deepcopy(tree)
    for parent,field,index,old,_ in reversed(edits):
        if index is None:setattr(parent,field,old)
        else:getattr(parent,field)[index]=old
    require(ast.dump(tree,include_attributes=False)==ast.dump(original,include_attributes=False),'C41 undeclared factory change')
    old_sha=[n.value for n in ast.walk(original) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)==64]
    new_sha=[n.value for n in ast.walk(result) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)==64]
    require(old_sha==new_sha,'C41 changed an embedded frozen SHA')
    module=ModuleType('b22_numpy41_factory');module.__file__=str(DRIVER)
    exec(compile(ast.fix_missing_locations(result),str(DRIVER),'exec'),module.__dict__)  # noqa: S102 -- pinned39 inverse-checked factory identity seams
    module.LEGACY={**module.LEGACY,**SOURCE39,**SOURCE40}
    proof={'schema':'b22-numpy41-factory-derivation-v1','source39_sha256':SOURCE39['scripts/run_b22_numpy39.py'],
        'derived_ast_sha256':hashlib.sha256(ast.dump(result,include_attributes=False).encode()).hexdigest(),
        'inverse_ast_exact':True,'all_sha_constants_unchanged':True,'seams':[x[4] for x in edits]}
    return module,proof


def request_walls41(factory,runner):
    """Change only three source-GO wall literals, with the inverse AST checked."""
    tree,_=factory.derive('runner')
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='script_request'))
    original=copy.deepcopy(node);changed=[]
    for item in ast.walk(node):
        if not isinstance(item,ast.Dict):continue
        for key,value in zip(item.keys,item.values):
            if isinstance(key,ast.Constant) and key.value in ('soft_s','close_s','hard_s'):
                old,new={'soft_s':(180,300),'close_s':(235,355),'hard_s':(240,360)}[key.value]
                require(isinstance(value,ast.Constant) and type(value.value) is int and value.value==old,
                        'C41 request wall source seam changed')
                changed.append((value,old));value.value=new
    require(len(changed)==3,'C41 exact three request wall seams required')
    result=copy.deepcopy(node)
    for value,old in changed:value.value=old
    require(ast.dump(node,include_attributes=False)==ast.dump(original,include_attributes=False),
            'C41 request changed beyond wall literals')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[result],type_ignores=[])),str(DRIVER),'exec'),runner.__dict__)  # noqa: S102 -- inverse-checked frozen request with three wall literals only
    return {'schema':'b22-numpy41-request-wall-seam-v1','inverse_ast_exact':True,
        'changed_literals':{'soft_s':[180,300],'close_s':[235,355],'hard_s':[240,360]},
        'derived_ast_sha256':hashlib.sha256(ast.dump(result,include_attributes=False).encode()).hexdigest()}


def build_scope():
    factory,proof=factory41();graph=factory.build_scope();r=graph.runner
    maps=load40().install_adapter40(factory,graph)
    full_stages=r.stages
    r.stages=lambda:[stage for stage in full_stages() if stage['name']=='yaw_1p2_B']
    r.LIMITS={**r.LIMITS,'controls':1600,'normal_native':8000,'compiler_native':2,
              'numpy_predict_calls':1601,'numpy_predict_rows':1632,'worker_sessions':2,
              'bootstrap_wall_s':120,'stage_wall_s':360,'reader_wall_s':240}
    walls=request_walls41(factory,r)
    # Only one validation/compare call may share a byte snapshot. Every separate
    # call, including each new stage's admission, starts from fresh hashes.
    depth=0;reuse_value=None
    @contextmanager
    def reuse_snapshot():
        nonlocal depth,reuse_value
        depth+=1
        try:yield
        finally:
            depth-=1
            if depth==0:reuse_value=None
    def current_reuse():
        nonlocal reuse_value
        if depth==0:return reused40()
        if reuse_value is None:reuse_value=reused40()
        return copy.deepcopy(reuse_value)
    original_plan,original_closure=r.plan,r.closure
    def plan(output=None):
        value=original_plan(output)
        value.update(reused40=current_reuse(),factory41=proof,maps40=maps,request_wall41=walls,
            scope='new41 yaw-only; fixed C39 flat adjudicated by40 is reused evidence, never new41 resource spending')
        return value
    r.plan=plan
    r.closure=lambda source,refs,exports,config:factory.merge_inputs(original_closure(source,refs,exports,config),current_reuse()['inputs'])
    original_validate=r.validate_release
    def validate_release(path):
        with reuse_snapshot():return original_validate(path)
    r.validate_release=validate_release
    r.execute.__kwdefaults__['validator']=validate_release
    # Preserve both historical35 anchors while executing only the yaw stage.
    reader40=load40();_,reference_graph=reader40.load_graph39()
    r.reference35=reference_graph.runner.reference35
    r.execute.__kwdefaults__['references_reader']=r.reference35
    original_compare=graph.reader.compare_reports
    compare_namespace=dict(original_compare.__globals__,r=SimpleNamespace(stages=full_stages))
    from types import FunctionType
    compare_math=FunctionType(original_compare.__code__,compare_namespace,original_compare.__name__)
    def compare_inner(release_path):
        release=r.validate_release(release_path);reuse=current_reuse();stage=r.stages()[0]
        path=Path(release['plan']['output'])/(stage['name']+'_readback.json');saved=r.c.read_json(path)
        r.stage_counts(stage,saved,r.c.identity(release_path),r.c.identity(DRIVER))
        for name,entry in saved['artifacts'].items():
            p=Path(name)
            require(not p.is_absolute() and '..' not in p.parts and r.c.identity(path.parent/stage['name']/p)==entry,
                    'C41 new yaw raw changed')
        result=compare_math([reuse['report'],saved],r.reference35()['reports'])
        return {'schema':'b22-numpy41-comparison-v1','status':'COMPLETE','release_identity':r.c.identity(release_path),
            'stage_reports':{path.name:r.c.identity(path)},'new_actual_counts':dict(saved['actual_counts']),
            'reused_validated_case_counts':dict(PHYSICAL),'reused_validated_nn_counts':dict(NN),
            'new_actual_nn_counts':{'predict_calls':saved['numpy_predict_calls'],'predict_rows':saved['numpy_predict_rows'],'ppo_load':0,'torch_load':0},
            'combined_validated_nn_counts':{'predict_calls':1601+saved['numpy_predict_calls'],'predict_rows':1632+saved['numpy_predict_rows'],'ppo_load':0,'torch_load':0},
            'combined_validated_case_counts':{k:v+saved['actual_counts'][k] for k,v in PHYSICAL.items()},
            'reused40_report_identity':reuse['report_identity'],'new_execution_contract_id':CONTRACT,
            'reused_execution_contract_id':'C39_B22_numpy_half_gain_v1','reused_adjudication_contract_id':reader40.CONTRACT,
            'original39_actual_counts':None,'original39_actual_nn_counts':None,'historical_total_resources_known':False,**result}
    def compare(release_path):
        with reuse_snapshot():return compare_inner(release_path)
    graph.reader.compare=compare
    graph.factory41_proof=proof
    graph.request_wall41_proof=walls
    graph.reuse_snapshot41=reuse_snapshot
    graph.current_reuse41=current_reuse
    return graph


def scope():
    entry=sys.modules.get(REGISTRY)
    if entry is None:
        entry=ModuleType(REGISTRY);entry.__file__=str(DRIVER);entry.driver_identity=identity(DRIVER)
        sys.modules[REGISTRY]=entry
        try:entry.graph=build_scope()
        except BaseException:sys.modules.pop(REGISTRY,None);raise
    require(entry.driver_identity==identity(DRIVER),'C41 source changed within process')
    return entry.graph


def validate_bootstrap_launch38(session_path,output):
    return scope().runner.validate_bootstrap_launch38(session_path,output)


def describe():
    return {'schema':'b22-numpy41-plan-v1','status':'PLAN_ONLY','execution_contract_id':CONTRACT,'output':str(OUTPUT),
        'stage_order':['bootstrap','yaw_1p2_B','saved_readback','comparison'],'new_actual_budget':dict(PHYSICAL),
        'new_nn_budget':dict(NN),'reused_validated_case_counts':dict(PHYSICAL),
        'combined_validated_case_counts':{k:2*v for k,v in PHYSICAL.items()},'historical_total_resources_known':False,
        'execution_started':False,'qualification_granted':False,'retry_permitted':False,
        'start_effective_bytes':6<<30,'reserve_effective_bytes':4<<30,'wall_s':900,'required_nice':10}


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if '--session' in args:
        require('--execute' not in args and '--release' not in args,'C41 mixed worker command')
        return scope().worker.main(args)
    if '--release' in args:
        require('--execute' not in args and '--bootstrap' not in args,'C41 mixed reader command')
        return scope().reader.main(args)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parsed=parser.parse_args(args)
    if not parsed.execute:print(json.dumps(describe(),ensure_ascii=False,indent=2));return 0
    started=time.monotonic();graph=scope();value=graph.runner.execute()
    value.update(original39_actual_counts=None,original39_actual_nn_counts=None,historical_total_resources_known=False)
    if (OUTPUT/'release.json').is_file():
        value['reused_validated_case_counts']=dict(PHYSICAL)
        value['reused_validated_nn_counts']=dict(NN)
    if value['status']=='COMPLETE':
        value['combined_validated_case_counts']={k:v+value['actual_counts'][k] for k,v in PHYSICAL.items()}
        value['combined_validated_nn_counts']={k:v+value['actual_nn_counts'][k] for k,v in NN.items()}
    value['elapsed_including_scope_s']=time.monotonic()-started
    if value['elapsed_including_scope_s']>=900:
        value.update(status='INCOMPLETE',failure='C41 total wall including scope exceeded')
    if OUTPUT.is_dir():graph.runner.atomic_json(OUTPUT/'report.json',value)
    print(json.dumps(value,ensure_ascii=False,indent=2));return 0 if value['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
