"""C37 script owner: C35 numerical source, only new contract admission/derivation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'C37_B22_half_gain_missing_cases_v1'
ORIGINAL = 'scripts/run_b22_worker35.py'
ORIGINAL_SHA = '07be70b151c383a4a88a4b2b2f7c0242cb02919c9bc9561ce577c2321d6900cb'


def derived_source(root=ROOT):
    path = root/ORIGINAL
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError('C37 original C35 owner changed')
    spec = importlib.util.spec_from_file_location('completion37_original_derivation35', path)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    original = prior.derived_sources(root)['script']
    old = repr('C35_B22_half_gain_fixed_suite_v1')
    if original.count(old) != 1:
        raise ValueError('C37 exact contract-only AST seam changed')
    source = original.replace(old, repr(CONTRACT))
    normalized = source.replace(repr(CONTRACT), old)
    if ast.dump(ast.parse(normalized)) != ast.dump(ast.parse(original)):
        raise ValueError('C37 changed numerical/control/archive source')
    proof = {'schema': 'b22-worker37-source-derivation-v1', 'execution_contract_id': CONTRACT,
             'original_worker35_sha256': ORIGINAL_SHA,
             'original35_script_ast_sha256': hashlib.sha256(ast.dump(ast.parse(original)).encode()).hexdigest(),
             'derived_script_ast_sha256': hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest(),
             'scope': 'only C37 admission literal; original half gain, control, reset, counters and archive unchanged'}
    return source, proof


def validate_session(session, output):
    name = session.get('arm')
    expected = {
        'ramp_0p45_complete_zero': ('ramp_0p45_complete', 'zero', 181206, 1800, False),
        'ramp_0p45_complete_B': ('ramp_0p45_complete', 'B', 181206, 1800, False),
        'yaw_1p2_mirror_zero': ('yaw_1p2', 'zero', 181403, 1600, True),
        'yaw_1p2_mirror_B': ('yaw_1p2', 'B', 181403, 1600, True)}
    if (name not in expected or session.get('execution_contract_id') != CONTRACT
            or session.get('mode') != 'script' or session.get('render') is not False
            or session.get('qualification') is not False or type(session.get('gain35')) is not float
            or session['gain35'] != .5 or session.get('output_directory') != str(output)
            or type(session.get('seed')) is not int or type(session.get('control_limit')) is not int
            or type(session.get('mirror')) is not bool
            or tuple(session.get(k) for k in ('profile', 'actor', 'seed', 'control_limit', 'mirror')) != expected[name]):
        raise ValueError('C37 only four preregistered script sessions allowed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    session = json.loads(args.session.read_text())
    validate_session(session, args.output)
    source, proof = derived_source()
    with (args.output/'derived_sources37.json').open('x') as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write('\n')
    namespace = {'__name__': '__main__', '__file__': str(Path(__file__).resolve())}
    exec(compile(source, namespace['__file__'], 'exec'), namespace)  # noqa: S102 -- pinned contract-only AST derivative


if __name__ == '__main__':
    main()
