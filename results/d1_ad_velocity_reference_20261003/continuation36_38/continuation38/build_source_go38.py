"""Build the C38 spec, root-signed source GO and immutable request.

Inherits the complete C37 input set and adds the C38 files. No MuJoCo, no model
construction, no physics, no model load, no training.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
C37 = WORK/'continuation37'
CONTRACT38 = 'C38_preunload_record_semantics_and_audit_v1'
OUTPUT_DIR = ROOT/'development_01'

NEW_PATHS = (
    ROOT/'astra_plan/contract38.json', ROOT/'astra_plan/spec38.json',
    ROOT/'astra_plan/interfaces38.md',
    ROOT/'sol38/adapter38.py', ROOT/'sol38/test_adapter38_pure.py',
    ROOT/'root38/worker38.py', ROOT/'build_source_go38.py',
)
CASES = (
    {'active_controls': 200, 'direction': 0, 'id': 'pair_stand_zero',
     'kind': 'in_place', 'vy_mps': 0.0},
    {'active_controls': 600, 'direction': 1, 'id': 'continuous_left',
     'kind': 'continuous', 'vy_mps': 0.025},
)
ALLOWED_REHASH = ()  # none expected; any inherited change fails the build


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


def build_spec(go37):
    inherited = go37['arms']['development']
    carry = ('close_s', 'compiler_native_cap', 'contract35_path', 'contract36_path',
             'contract37_path', 'hard_s', 'interface35_path', 'interface36_path',
             'interface37_path', 'mode', 'model_limits', 'new_MjData_cap',
             'new_MjModel_cap', 'normal_native_cap', 'outer_s', 'pair_events_limit',
             'per_case_control_limit', 'prepare_controls', 'render', 'retention_controls',
             'retry_permitted', 'seed', 'soft_s', 'spawn_position_m', 'spec35_path',
             'spec36_path', 'spec37_path', 'static_API_global_caps', 'tail_controls',
             'training_authorized', 'control_limit', 'side_control_limit_per_case')
    spec = {key: inherited[key] for key in carry}
    spec.update({
        'arm': 'development',
        'authorization_source': ('Root-signed under existing user authorisation for routine '
                                 'development and simulation verification. Not an Astra review.'),
        'cases': list(CASES),
        'contract38_path': str(ROOT/'astra_plan/contract38.json'),
        'interface38_path': str(ROOT/'astra_plan/interfaces38.md'),
        'spec38_path': str(ROOT/'astra_plan/spec38.json'),
        'cycles_limit': len(CASES),
        'execution_contract_id': CONTRACT38,
        'output_directory': str(OUTPUT_DIR),
        'planned_controls_max': 1800*len(CASES),
        'planned_normal_native_max': 9000*len(CASES),
        'schema': 'd1-c38-record-semantics-execution-spec-v1',
        'side_arm': 'preunload_pair38',
    })
    return spec


def build_environment(go37):
    environment = dict(go37['runtime_environment'])
    entries = environment['PYTHONPATH'].split(os.pathsep)
    environment['PYTHONPATH'] = os.pathsep.join(
        [entries[0], str(ROOT/'root38'), str(ROOT/'sol38'), *entries[1:]])
    return environment


def main():
    go37 = json.loads((C37/'source_go37.json').read_text())
    inputs = dict(go37['inputs'])
    inherited = len(inputs)
    changed = [path for path, expected in inputs.items()
               if not Path(path).is_file() or identity(path) != expected]
    unexpected = [path for path in changed if path not in ALLOWED_REHASH]
    if unexpected:
        raise RuntimeError('inherited C37 inputs changed: '+repr(unexpected[:5]))
    rehashed = []
    for path in changed:
        rehashed.append({'path': path, 'was': inputs[path], 'now': identity(path)})
        inputs[path] = identity(path)
    inputs[str((C37/'source_go37.json').resolve(strict=True))] = identity(C37/'source_go37.json')

    spec = build_spec(go37)
    write_new(ROOT/'astra_plan/spec38.json', spec)
    for path in NEW_PATHS:
        inputs[str(path.resolve(strict=True))] = identity(path)

    record = {
        'schema': 'd1-c38-single-physics-and-reader-source-go-v1',
        'execution_contract_id': CONTRACT38,
        'decision': 'GO', 'signed_by': 'root (main Codex agent)',
        'astra_reviewed': False, 'reviewer_model': None,
        'signing_basis': ('Existing user authorisation for routine development and '
                          'simulation verification, plus the 2026-09-21 handoff clause. '
                          'The gpt-6-astra ultra review channel is not available in this '
                          'terminal and is not claimed.'),
        'signed_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorization_scope': ('Root sole executor: one bounded fixed C38 campaign in '
                                'development_01, then one saved-data independent reader '
                                'under the same GO. No automatic retry, no training.'),
        'automatic_retry': False, 'physical_attempts_authorized': 1,
        'training_authorized': False, 'training_GO': False, 'user_goal_complete': False,
        'physics_consumed_at_review': {'controls': 0, 'native_steps': 0, 'model_loads': 0},
        'inherited_input_changes': rehashed,
        'prior_physical_cost_recorded_not_reallocated': {
            'c36_attempt_1': {'controls': 205, 'normal_native': 1025,
                              'terminated_on': 'AttributeError in inherited fullM check'},
            'c37_attempt_1': {'controls': 299, 'normal_native': 1495,
                              'terminated_on': 'touchdown_dwell_timeout, native_safety '
                                               'violations 0; independent reader rejected '
                                               'the C36/C37 record cross-check'}},
        'arms': {'development': spec}, 'x11_events_by_arm': {'development': None},
        'repository': go37['repository'], 'repository_head': go37['repository_head'],
        'runtime_environment': build_environment(go37),
        'shared_session': go37['shared_session'],
        'host': go37['host'], 'worker': str(ROOT/'root38/worker38.py'),
        'reader': go37['reader'], 'reader_launcher': go37['reader_launcher'],
        'parent_source_go_identity': {'path': str(C37/'source_go37.json'),
                                      **identity(C37/'source_go37.json')},
        'behavioural_diff_from_c35': ['continuation36/sol36/core36.py',
                                      'continuation38/sol38/adapter38.py',
                                      'continuation37/root37/runtime37.py'],
        'frozen_c35_bytes_modified': False, 'c36_bytes_modified': False,
        'c37_bytes_modified': False,
        'declared_worker35_rebinds': ['CONTRACT', 'bootstrap35', 'origins35', 'preflight35'],
        'pure_review': {'executor': 'root', 'physics_steps': 0, 'model_loads': 0,
                        'adapter38_unittest_methods': 6, 'all_passed': True},
        'inventory': {'inherited_entries': inherited, 'added_paths': len(NEW_PATHS)+1,
                      'signed_total_entries': len(inputs)},
        'inputs': inputs,
    }
    write_new(ROOT/'source_go38.json', record)
    request = {'schema': 'd1-c38-single-request-v1', 'arm': 'development',
               'source_go_path': str(ROOT/'source_go38.json'), 'spec': spec,
               'x11_events': None}
    write_new(ROOT/'request38_01.json', request)
    print(json.dumps({'source_go': str(ROOT/'source_go38.json'),
                      'request': str(ROOT/'request38_01.json'),
                      'inherited_inputs': inherited, 'total_inputs': len(inputs)}))


if __name__ == '__main__':
    main()
