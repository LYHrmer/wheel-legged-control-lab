"""Explicit C35 derivatives of pinned C34/C24 owners; no live imports on import.

Only admission metadata, float32 residual gain and its extra ledger change.
The derived source/AST identities are saved before executing either owner.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'C35_B22_half_gain_fixed_suite_v1'
SCRIPT = 'scripts/run_b22_worker34.py'
STEP = 'runtime/d1_b22/research/continuation24/'
PINS = {
    SCRIPT: '2f71895b709ec1abbd947c96673ec969de3a483bb4d145843315754513c3eb5d',
    STEP+'worker24.py': '638f518e2cba500f60bee1ea1cc8a6e8e59d0fd4452ca345a6bed090bc85bb2b',
    STEP+'eval24.py': '927e373aa41bd6687ea6d06b36c0b7b8bee6722c3e698a44c171a2f3f465d073',
    STEP+'heldout24.py': '0b40b73c834dba65f4cc45a8ea18267a48c32b46d69799a31ec7846f3509f92f',
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('C35 pinned derivation seam changed: '+old)
    return source.replace(old, new)


def derived_sources(root=ROOT):
    """Produce exactly reviewed derivatives; never execute their imports here."""
    original = {}
    for name, expected in PINS.items():
        path = root/name
        data = path.read_bytes()
        if path.is_symlink() or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('C35 original source changed: '+name)
        original[name] = data.decode('utf-8')
    script = original[SCRIPT].replace('gain34', 'gain35')
    for old, new in (
        ("'C34_B22_residual_gain_probe_v1'", repr(CONTRACT)),
        ("or session.get('actor') != 'B'", "or session.get('actor') not in ('B','zero')"),
        ("or session.get('profile') not in ('flat_1p6','yaw_1p2')", "or session.get('profile') not in PROFILE_CASE"),
        ("session['gain35'] not in (1.0,.5)", "session['gain35'] != .5"),
        ("session['control_limit'] != 1600", "session['control_limit'] != (1800 if session['profile']=='ramp_0p45_complete' else 1600)"),
    ):
        script = replace_once(script, old, new)
    worker = original[STEP+'worker24.py']
    if worker.count('C24_true_15mm_B22_v1') != 2:
        raise ValueError('C35 step contract seam changed')
    worker = worker.replace('C24_true_15mm_B22_v1', CONTRACT)
    worker = replace_once(worker, 'd1-true15mm-worker-24-v1', 'd1-true15mm-worker-35-v1')
    worker = replace_once(worker, "or session.get('mode') != 'eval'", "or type(session.get('gain35')) is not float or session['gain35'] != .5\n                or session.get('mode') != 'eval'")
    worker = replace_once(worker, 'from eval24 import evaluate', 'evaluate = _evaluate35')
    evaluation = replace_once(original[STEP+'eval24.py'], 'from heldout24 import record_case', 'record_case = _record_case35')
    evaluation = replace_once(evaluation, 'execution_contract_id=CONTRACT_ID', 'execution_contract_id='+repr(CONTRACT))
    heldout = replace_once(original[STEP+'heldout24.py'],
        'action, predicted = _numeric_actor_action(score_actor, policy, observation)',
        'network_output16, predicted = _numeric_actor_action(score_actor, policy, observation)\n'
        '            network_output16 = np.asarray(network_output16, dtype=np.float32).copy()\n'
        '            action = network_output16*np.float32(.5)')
    heldout = replace_once(heldout, '"policy_input_action": action,',
        '"policy_input_action": action, "raw_action16": action, "action16": action,\n'
        '                "network_output16": network_output16, "gain35": .5,')
    return {'script': script, 'step_worker': worker, 'step_eval': evaluation, 'step_record': heldout}


def derivation_proof(sources):
    return {'schema': 'b22-worker35-source-derivation-v1', 'execution_contract_id': CONTRACT,
            'original_sha256': PINS,
            'derived_ast_sha256': {name: hashlib.sha256(ast.dump(ast.parse(text), include_attributes=False).encode()).hexdigest()
                                   for name, text in sources.items()},
            'scope': 'explicit C35 admission; fixed float32 half gain and ledger; all other owner/control/archive source unchanged'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    session = json.loads(args.session.read_text())
    if (session.get('execution_contract_id') != CONTRACT or type(session.get('gain35')) is not float
            or session['gain35'] != .5 or session.get('mode') not in ('script', 'eval')
            or session.get('output_directory') != str(args.output)):
        raise ValueError('explicit C35 owner/session required')
    sources = derived_sources()
    with (args.output/'derived_sources35.json').open('x') as stream:
        stream.write(json.dumps(derivation_proof(sources), indent=2, allow_nan=False)+'\n')
    # All executing code belongs to this new, GO-bound entry; original files
    # remain pinned derivation inputs, never masquerade as the executing owner.
    namespace = {'__name__': '__main__', '__file__': str(Path(__file__).resolve())}
    def evaluate35(*args):
        # Preserve C24's import-repair/ready/engine ordering: the old worker
        # asks for its evaluator only after constructing the counted owner.
        record = {'__name__': 'b22_step_record35', '__file__': namespace['__file__']}
        exec(compile(sources['step_record'], namespace['__file__'], 'exec'), record)  # noqa: S102 -- exact SHA-pinned source derivative
        evaluation = {'__name__': 'b22_step_eval35', '__file__': namespace['__file__'], '_record_case35': record['record_case']}
        exec(compile(sources['step_eval'], namespace['__file__'], 'exec'), evaluation)  # noqa: S102 -- exact SHA-pinned source derivative
        return evaluation['evaluate'](*args)
    if session['mode'] == 'eval':
        namespace['_evaluate35'] = evaluate35
    exec(compile(sources['script' if session['mode'] == 'script' else 'step_worker'], namespace['__file__'], 'exec'), namespace)  # noqa: S102 -- exact SHA-pinned source derivative


if __name__ == '__main__':
    main()
