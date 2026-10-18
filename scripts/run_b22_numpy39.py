"""Independent C39 dependency-closure repair; derive frozen C38, never retry it.

Import/default planning use only the standard library. Actual execution requires
committed sources, a new output, the expanded package closure and a new cold
bootstrap. This file is also the exact host worker and saved-reader CLI owner.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT/'scripts/run_b22_numpy39.py'
OUTPUT = ROOT/'results/b22_numpy39_20261018_01'
CONTRACT = 'C39_B22_numpy_half_gain_v1'
AUTHORED = ('scripts/run_b22_numpy39.py','tests/test_b22_numpy39.py','docs/b22_numpy39.md',
            'scripts/b22_runtime_packages39.py','tests/test_b22_runtime_packages39.py')
LEGACY = {
 'scripts/run_b22_numpy38.py':'0675501fe193407d33aa1b4e97cc8875a4bcbf2e1391e1b426a2b1c2612bb431',
 'scripts/read_b22_numpy38.py':'06a31268a23dc389d846d4e86853ee129727f4eb97b39735a477b468ed1ab847',
 'scripts/b22_numpy_backend38.py':'183053fe4f81bdfff1388587b143cd0890d35b7739ac3fced4425e0b4dbcbd62',
 'scripts/run_b22_numpy_worker38.py':'67ce51a4e5c2f06dc357cf813fd7ad787985533b0a7fe060f0bded409e71b48e',
 'tests/test_b22_numpy38.py':'ebebcc9e930d10997b0fefcc1ec1cf98ce5e234bf9e237dfdf36fcd976a87374',
 'tests/test_b22_numpy_backend38.py':'f6a372523bd63f7ff95c1e2a603ee83118af145242055db0ee5aa7a4eb26b6a7',
 'tests/test_b22_numpy_worker38.py':'a76a188806e545a45f0b77c3b2b4b15e9210df1798e2da1cb569d35a08eb6e4e',
 'docs/b22_numpy38.md':'6298d923a964668950d52eec1d9e2edb608203bd3efd9435516a6b5b2331bf94'}
MODULES = {'backend':'scripts/b22_numpy_backend38.py','runner':'scripts/run_b22_numpy38.py',
           'reader':'scripts/read_b22_numpy38.py','worker':'scripts/run_b22_numpy_worker38.py'}
REGISTRY = '_b22_numpy39_verified_scope'


def require(value,message):
    if not value:
        raise ValueError(message)


def identity(path):
    digest,size = hashlib.sha256(),0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk);size += len(chunk)
    return {'bytes':size,'sha256':digest.hexdigest()}


def legacy_inputs():
    result = {}
    for name,digest in LEGACY.items():
        path = ROOT/name
        require(not path.is_symlink(),'C39 legacy source symlink: '+name)
        entry = identity(path)
        require(entry['sha256'] == digest,'C39 frozen C38 source changed: '+name)
        result[str(path)] = entry
    return result


def package_helper():
    path = ROOT/'scripts/b22_runtime_packages39.py'
    spec = importlib.util.spec_from_file_location('b22_runtime_packages39',path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def merge_inputs(*groups):
    result = {}
    for group in groups:
        for name,entry in group.items():
            require(name not in result or result[name] == entry,'C39 overlapping input identity differs: '+name)
            result[name] = entry
    return result


# Only identity/path strings are changed. Gain/math fields and every 64-hex
# constant remain unchanged, including historical C35/C36 references.
def identity_literal(value):
    exact = {'C38_B22_numpy_half_gain_v1':CONTRACT,
             'results/b22_numpy38_20261018_01':'results/b22_numpy39_20261018_01',
             'run_b22_numpy38.py':'run_b22_numpy39.py',
             'scripts/run_b22_numpy38.py':'scripts/run_b22_numpy39.py',
             'scripts/run_b22_numpy_worker38.py':'scripts/run_b22_numpy39.py',
             'scripts/read_b22_numpy38.py':'scripts/run_b22_numpy39.py',
             'scripts/b22_numpy_backend38.py':'scripts/run_b22_numpy39.py'}
    if value in exact:
        return exact[value]
    value = value.replace('b22-numpy38-','b22-numpy39-')
    # These are explicit import/module-origin source fragments, not arbitrary
    # substitutions of version digits or substrings inside a SHA.
    for old,new in (("'scripts/run_b22_numpy_worker38.py'","'scripts/run_b22_numpy39.py'"),
                    ("'scripts/b22_numpy_backend38.py'","'scripts/run_b22_numpy39.py'")):
        value = value.replace(old,new)
    return value.replace('b22_numpy_backend38','b22_numpy_backend39')


def _target(node,name):
    return isinstance(node,ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id == name


def _canonical(node):
    return ast.dump(node,include_attributes=False)


def derive(kind):
    """Return an inverse-checked AST and a closed list of identity-only seams."""
    name = MODULES[kind]
    text = (ROOT/name).read_text()
    require(hashlib.sha256(text.encode()).hexdigest() == LEGACY[name],'C39 source seam is not frozen')
    original = ast.parse(text)
    tree = copy.deepcopy(original)
    edits = []
    # Every replacement stores its original subtree, making inverse equality an
    # executable gate rather than merely a declaration in the report.
    def replace(parent,field,index,new,label):
        old = getattr(parent,field)[index] if index is not None else getattr(parent,field)
        edits.append((parent,field,index,old,label))
        if index is None:setattr(parent,field,new)
        else:getattr(parent,field)[index] = new
    for i,node in enumerate(list(tree.body)):
        if isinstance(node,ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            replace(tree,'body',i,ast.Pass(),'entry dispatched by new39 CLI')
        elif kind == 'runner' and _target(node,'backend'):
            replace(tree,'body',i,ast.parse('backend = _scope_backend').body[0],'runner backend injection')
        elif kind == 'runner' and _target(node,'AUTHORED'):
            replace(node,'value',None,ast.parse(repr(AUTHORED),mode='eval').body,'new authored source set')
        elif kind == 'reader' and _target(node,'_spec'):
            replace(tree,'body',i,ast.Pass(),'reader original spec omitted')
        elif kind == 'reader' and _target(node,'r'):
            replace(tree,'body',i,ast.parse('r = _scope_runner').body[0],'reader runner injection')
        elif kind == 'reader' and isinstance(node,ast.Expr) and ast.unparse(node.value) == '_spec.loader.exec_module(r)':
            replace(tree,'body',i,ast.Pass(),'reader original loader omitted')
        elif kind == 'worker' and (_target(node,'_SPEC') or ast.unparse(node).startswith('sys.modules[_SPEC.name] =')
            or isinstance(node,ast.Expr) and ast.unparse(node.value) == '_SPEC.loader.exec_module(backend)'):
            replace(tree,'body',i,ast.Pass(),'worker original loader omitted')
        elif kind == 'worker' and _target(node,'backend'):
            replace(tree,'body',i,ast.parse('backend = _scope_backend').body[0],'worker backend injection')
    for parent in list(ast.walk(tree)):
        for field,value in list(ast.iter_fields(parent)):
            children = list(enumerate(value)) if isinstance(value,list) else [(None,value)]
            for index,child in children:
                if isinstance(child,ast.Constant) and isinstance(child.value,str):
                    new = identity_literal(child.value)
                    if new != child.value:
                        replace(parent,field,index,ast.Constant(new),'exact identity/path literal')
                elif isinstance(child,ast.Call) and ((isinstance(child.func,ast.Attribute) and child.func.attr == 'load_file')
                    or (isinstance(child.func,ast.Name) and child.func.id == 'load_file')):
                    if len(child.args) == 2 and isinstance(child.args[1],ast.BinOp) and isinstance(child.args[1].right,ast.Constant):
                        target = child.args[1].right.value
                        if target in ('scripts/read_b22_numpy38.py','scripts/run_b22_numpy_worker38.py'):
                            function = '_scope_reader' if target == 'scripts/read_b22_numpy38.py' else '_scope_worker'
                            replace(parent,field,index,ast.Call(ast.Name(function,ast.Load()),[],[]),'derived module injection')
    # New facts are persisted in the original release construction; both are
    # validated against fresh package/source proofs before every new admission.
    if kind == 'runner':
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'execute')
        assignment = next(n for n in ast.walk(function) if _target(n,'release') and isinstance(n.value,ast.Dict))
        new = copy.deepcopy(assignment.value)
        new.keys.extend([ast.Constant('runtime_package_inventory'),ast.Constant('derivation39')])
        new.values.extend([ast.Call(ast.Name('_inventory39',ast.Load()),[],[]),ast.Call(ast.Name('_proof39',ast.Load()),[],[])])
        replace(assignment,'value',None,new,'expanded package/derivation release facts')
    result = copy.deepcopy(tree)
    for parent,field,index,old,label in reversed(edits):
        if index is None:setattr(parent,field,old)
        else:getattr(parent,field)[index] = old
    require(_canonical(tree) == _canonical(original),'C39 undeclared AST change')
    original_hashes = sorted(n.value for n in ast.walk(original) if isinstance(n,ast.Constant) and isinstance(n.value,str) and re.fullmatch('[0-9a-f]{64}',n.value))
    new_hashes = sorted(n.value for n in ast.walk(result) if isinstance(n,ast.Constant) and isinstance(n.value,str) and re.fullmatch('[0-9a-f]{64}',n.value))
    require(original_hashes == new_hashes,'C39 changed a frozen SHA constant')
    proof = {'original_source':str(ROOT/name),'original_sha256':LEGACY[name],
        'derived_ast_sha256':hashlib.sha256(_canonical(result).encode()).hexdigest(),
        'inverse_ast_exact':True,'all_sha_constants_unchanged':True,
        'seams':[entry[4] for entry in edits]}
    return ast.fix_missing_locations(result),proof


def build_scope():
    """Construct four stdlib-only derived modules with a single backend object."""
    pinned = legacy_inputs()
    package = package_helper()
    trees,proofs = {},{}
    for kind in MODULES:
        trees[kind],proofs[kind] = derive(kind)
    graph = SimpleNamespace()
    proof = {'schema':'b22-numpy39-derivation-v1','legacy_inputs':pinned,'modules':proofs,
        'unchanged_numeric_scope':'C38 control/native/NN count/metric/supervision code; exact identity injections plus dependency closure only'}
    def compile_module(kind,namespace):
        name = 'b22_numpy_backend39' if kind == 'backend' else 'b22_numpy39_'+kind
        module = ModuleType(name)
        module.__file__ = str(DRIVER)
        module.__dict__.update(namespace)
        sys.modules[name] = module
        exec(compile(trees[kind],str(DRIVER),'exec'),module.__dict__)  # noqa: S102 -- inverse-checked identity seams on pinned source
        return module
    graph.backend = compile_module('backend',{})
    injected = {'_scope_backend':graph.backend,'_scope_reader':lambda:graph.reader,
        '_scope_worker':lambda:graph.worker,'_inventory39':package.inventory39,'_proof39':lambda:proof}
    graph.runner = compile_module('runner',injected)
    graph.reader = compile_module('reader',{'_scope_runner':graph.runner,'_scope_worker':lambda:graph.worker})
    graph.worker = compile_module('worker',{'_scope_backend':graph.backend})
    graph.proof,graph.package = proof,package
    runner = graph.runner
    original_closure,original_validate = runner.closure,runner.validate_release
    def expanded_closure(source,references,exports,config):
        return merge_inputs(original_closure(source,references,exports,config),pinned,package.inventory39()['inputs'])
    def validate_release(path):
        saved = runner.c.read_json(path)
        require(set(saved) == {'schema','plan','source','references35','export36','inputs','v27_readiness',
                              'runtime_package_inventory','derivation39'},'C39 exact release fields differ')
        package.validate_inventory39(saved['runtime_package_inventory'])
        require(saved['derivation39'] == proof,'C39 source derivation differs')
        return original_validate(path)
    runner.closure,runner.validate_release = expanded_closure,validate_release
    runner.execute.__kwdefaults__['validator'] = validate_release
    def session_inventory(session):
        path = runner.OUTPUT/'release.json'
        saved = runner.c.read_json(path)
        require(session['source_hashes'].get(str(path)) == identity(path),
                'C39 worker package proof is not bound to this release')
        package.validate_inventory39(saved['runtime_package_inventory'])
        require(all(session['source_hashes'].get(name) == value
                    for name,value in saved['runtime_package_inventory']['inputs'].items()),
                'C39 worker omitted package input')
        return saved['runtime_package_inventory']
    original_bootstrap = graph.worker.bootstrap38
    def bootstrap(session,output):
        inventory = session_inventory(session)
        # Validate actual package __path__/module origins after import, before
        # the original dependency_origins call and before any actor/env load.
        original_components = graph.worker.bootstrap_components38
        def components():
            result = original_components()
            package.verify_loaded_roots39(inventory)
            return result
        graph.worker.bootstrap_components38 = components
        try:
            return original_bootstrap(session,output)
        finally:
            graph.worker.bootstrap_components38 = original_components
    graph.worker.bootstrap38 = bootstrap
    original_origins = graph.backend.audited_origins38
    def audited_origins(session):
        package.verify_loaded_roots39(session_inventory(session))
        return original_origins(session)
    graph.backend.audited_origins38 = audited_origins
    return graph


def scope():
    entry = sys.modules.get(REGISTRY)
    if entry is None:
        entry = ModuleType(REGISTRY)
        entry.__file__ = str(DRIVER)
        entry.driver_identity = identity(DRIVER)
        sys.modules[REGISTRY] = entry
        try:entry.graph = build_scope()
        except BaseException:
            sys.modules.pop(REGISTRY,None)
            raise
    require(entry.driver_identity == identity(DRIVER),'C39 driver changed during its process')
    return entry.graph


def validate_bootstrap_launch38(session_path,output):
    """Inherited function name only; checks C39 contract/output/nonce/source."""
    return scope().runner.validate_bootstrap_launch38(session_path,output)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    graph = scope()
    # The frozen host's exact worker argv has no --worker flag.
    if '--session' in args:
        require('--execute' not in args and '--release' not in args,'C39 mixed parent/worker/reader command')
        return graph.worker.main(args)
    if '--release' in args:
        require('--execute' not in args and '--bootstrap' not in args,'C39 mixed reader command')
        return graph.reader.main(args)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    parsed = parser.parse_args(args)
    if not parsed.execute:
        value = graph.runner.plan(parsed.output)
        value['derivation39'] = graph.proof
        value['runtime_package_inventory'] = graph.package.inventory39()
        print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    value = graph.runner.execute(parsed.output or OUTPUT)
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    return 0 if value['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
