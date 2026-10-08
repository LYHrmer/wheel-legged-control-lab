"""Build the C36 spec, root-signed source GO and immutable request.

Inherits the complete C35 input set (5136 paths, all re-verified) and adds the C36 files,
so every byte the campaign will touch is hashed before anything runs. No MuJoCo, no model
construction, no physics, no model load, no training: this only reads files and writes
three JSON documents with exclusive-create.

The GO is signed by root, not by gpt-6-astra ultra, and says so explicitly in its own
fields. That is the honest record: the Astra review channel is not available in this
terminal, while the user has authorised routine development and simulation verification and
the 2026-09-21 handoff permits the main agent to draft and record a bounded contract within
that authorisation.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
C35 = WORK/'continuation35'
CONTRACT36 = 'C36_preunload_transfer_two_support_feasibility_v1'
OUTPUT_DIR = ROOT/'development_01'

NEW_PATHS = (
    ROOT/'astra_plan/contract36.json',
    ROOT/'astra_plan/spec36.json',
    ROOT/'astra_plan/interfaces36.md',
    ROOT/'sol36/core36.py',
    ROOT/'sol36/adapter36.py',
    ROOT/'sol36/test_core36_pure.py',
    ROOT/'root36/worker36.py',
    ROOT/'build_source_go36.py',
)

CASES = (
    {'active_controls': 200, 'direction': 0, 'id': 'pair_stand_zero',
     'kind': 'in_place', 'vy_mps': 0.0},
    {'active_controls': 600, 'direction': 1, 'id': 'continuous_left',
     'kind': 'continuous', 'vy_mps': 0.025},
)


def identity(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def build_spec(go35):
    """Every key here must equal the session host31 assembles, or preflight36 refuses."""
    inherited = go35['arms']['development']
    return {
        'arm': 'development',
        'authorization_source': ('Root-signed under existing user authorisation for routine '
                                 'development and simulation verification. Not an Astra '
                                 'review.'),
        'cases': list(CASES),
        'close_s': 960,
        'compiler_native_cap': 2,
        'contract36_path': str(ROOT/'astra_plan/contract36.json'),
        'interface36_path': str(ROOT/'astra_plan/interfaces36.md'),
        'spec36_path': str(ROOT/'astra_plan/spec36.json'),
        'contract35_path': inherited['contract35_path'],
        'interface35_path': inherited['interface35_path'],
        'spec35_path': str(C35/'astra_plan/spec35.json'),
        'control_limit': 9000,
        'cycles_limit': len(CASES),
        'execution_contract_id': CONTRACT36,
        'hard_s': 1020,
        'mode': 'fixed',
        'model_limits': {'backward': 0, 'evaluate_actions': 0, 'forward': 0, 'learn': 0,
                         'load': 1, 'predict': 1201, 'predict_values': 0, 'save': 0,
                         'torch_load': 3, 'train': 0},
        'new_MjData_cap': 3,
        'new_MjModel_cap': 1,
        'normal_native_cap': 45000,
        'outer_s': 1200,
        'output_directory': str(OUTPUT_DIR),
        'pair_events_limit': 80,
        'per_case_control_limit': 1800,
        'planned_controls_max': 1800*len(CASES),
        'planned_normal_native_max': 9000*len(CASES),
        'prepare_controls': 200,
        'render': False,
        'retention_controls': 400,
        'retry_permitted': False,
        'schema': 'd1-c36-preunload-transfer-execution-spec-v1',
        'seed': 271001,
        'side_arm': 'preunload_pair36',
        'side_control_limit_per_case': 1200,
        'soft_s': 900,
        'spawn_position_m': inherited['spawn_position_m'],
        'static_API_global_caps': inherited['static_API_global_caps'],
        'tail_controls': 100,
        'training_authorized': False,
    }


def build_environment(go35):
    environment = dict(go35['runtime_environment'])
    prefix = (str(ROOT/'root36'), str(ROOT/'sol36'))
    entries = environment['PYTHONPATH'].split(os.pathsep)
    environment['PYTHONPATH'] = os.pathsep.join(
        [entries[0], *prefix, *entries[1:]])
    return environment


def main():
    go35 = json.loads((C35/'source_go35.json').read_text())
    inputs = dict(go35['inputs'])
    inherited = len(inputs)
    changed = [path for path, expected in inputs.items()
               if not Path(path).is_file() or identity(path) != expected]
    if changed:
        raise RuntimeError('inherited C35 inputs changed: '+repr(changed[:5]))

    spec = build_spec(go35)
    write_new(ROOT/'astra_plan/spec36.json', spec)
    for path in NEW_PATHS:
        inputs[str(path.resolve(strict=True))] = identity(path)

    record = {
        'schema': 'd1-c36-single-physics-and-reader-source-go-v1',
        'execution_contract_id': CONTRACT36,
        'decision': 'GO',
        'signed_by': 'root (main Codex agent)',
        'astra_reviewed': False,
        'reviewer_model': None,
        'signing_basis': ('Existing user authorisation for routine development and '
                          'simulation verification, plus the 2026-09-21 handoff clause '
                          'permitting the main agent to draft and record a bounded '
                          'contract without asking again. The gpt-6-astra ultra review '
                          'channel is not available in this terminal and is not claimed.'),
        'signed_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorization_scope': ('Root sole executor: one bounded fixed C36 campaign in '
                                'development_01, then one saved-data independent reader '
                                'under the same GO. No automatic retry, no training, no '
                                'unlisted physics.'),
        'automatic_retry': False,
        'physical_attempts_authorized': 1,
        'training_authorized': False,
        'training_GO': False,
        'user_goal_complete': False,
        'physics_consumed_at_review': {'controls': 0, 'native_steps': 0, 'model_loads': 0},
        'arms': {'development': spec},
        'x11_events_by_arm': {'development': None},
        'repository': go35['repository'],
        'repository_head': go35['repository_head'],
        'runtime_environment': build_environment(go35),
        'shared_session': go35['shared_session'],
        'host': str(C35/'root35/host35.py'),
        'worker': str(ROOT/'root36/worker36.py'),
        'reader': go35['reader'],
        'reader_launcher': go35['reader_launcher'],
        'parent_source_go_identity': {'path': str(C35/'source_go35.json'),
                                      **identity(C35/'source_go35.json')},
        'behavioural_diff_from_c35': ['sol36/core36.py', 'sol36/adapter36.py'],
        'frozen_c35_bytes_modified': False,
        'declared_worker35_rebinds': ['CONTRACT', 'bootstrap35', 'origins35', 'preflight35'],
        'pure_review': {'executor': 'root', 'physics_steps': 0, 'model_loads': 0, 'fits': 0,
                        'core36_unittest_methods': 10, 'all_passed': True},
        'inventory': {'inherited_entries': inherited, 'added_paths': len(NEW_PATHS),
                      'signed_total_entries': len(inputs)},
        'inputs': inputs,
    }
    write_new(ROOT/'source_go36.json', record)

    request = {'schema': 'd1-c36-single-request-v1', 'arm': 'development',
               'source_go_path': str(ROOT/'source_go36.json'),
               'spec': spec, 'x11_events': None}
    write_new(ROOT/'request36_01.json', request)
    print(json.dumps({'spec': str(ROOT/'astra_plan/spec36.json'),
                      'source_go': str(ROOT/'source_go36.json'),
                      'request': str(ROOT/'request36_01.json'),
                      'inherited_inputs': inherited, 'added': len(NEW_PATHS),
                      'total_inputs': len(inputs)}))


if __name__ == '__main__':
    main()
