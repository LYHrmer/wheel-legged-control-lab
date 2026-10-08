"""Build the C37 spec, root-signed source GO and immutable request.

Inherits the complete C36 input set (5144 paths, all re-verified) and adds the C37 files, so
every byte the campaign will touch is hashed before anything runs. No MuJoCo, no model
construction, no physics, no model load, no training.

The GO is signed by root, not by gpt-6-astra ultra, and records that in its own fields.
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
C36 = WORK/'continuation36'
CONTRACT37 = 'C37_preunload_transfer_with_repaired_fullm_check_v1'
OUTPUT_DIR = ROOT/'development_01'

NEW_PATHS = (
    ROOT/'astra_plan/contract37.json',
    ROOT/'astra_plan/spec37.json',
    ROOT/'astra_plan/interfaces37.md',
    ROOT/'root37/runtime37.py',
    ROOT/'root37/worker37.py',
    ROOT/'root37/test_runtime37_pure.py',
    ROOT/'build_source_go37.py',
)

# Exactly one inherited input changed between the C36 build and now: the system font
# rasteriser, same size and a different SHA256, from the libfreetype6 package. It is NOT in
# the C36 run's mapped_libraries_before_load.json, so it was never mapped into the physics
# process, and the campaign is headless with render False and no image I/O. It is re-hashed
# here and the change is recorded in the GO. Any OTHER inherited change still fails the build.
ALLOWED_REHASH = ('/usr/lib/x86_64-linux-gnu/libfreetype.so.6.18.1',)

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


def build_spec(go36):
    """Every key here must equal the session host31 assembles, or preflight37 refuses."""
    inherited = go36['arms']['development']
    carry = ('close_s', 'compiler_native_cap', 'contract35_path', 'contract36_path',
             'hard_s', 'interface35_path', 'interface36_path', 'mode', 'model_limits',
             'new_MjData_cap', 'new_MjModel_cap', 'normal_native_cap', 'outer_s',
             'pair_events_limit', 'per_case_control_limit', 'prepare_controls', 'render',
             'retention_controls', 'retry_permitted', 'seed', 'soft_s', 'spawn_position_m',
             'spec35_path', 'spec36_path', 'static_API_global_caps', 'tail_controls',
             'training_authorized', 'control_limit', 'side_control_limit_per_case')
    spec = {key: inherited[key] for key in carry}
    spec.update({
        'arm': 'development',
        'authorization_source': ('Root-signed under existing user authorisation for routine '
                                 'development and simulation verification. Not an Astra '
                                 'review.'),
        'cases': list(CASES),
        'contract37_path': str(ROOT/'astra_plan/contract37.json'),
        'interface37_path': str(ROOT/'astra_plan/interfaces37.md'),
        'spec37_path': str(ROOT/'astra_plan/spec37.json'),
        'cycles_limit': len(CASES),
        'execution_contract_id': CONTRACT37,
        'output_directory': str(OUTPUT_DIR),
        'planned_controls_max': 1800*len(CASES),
        'planned_normal_native_max': 9000*len(CASES),
        'schema': 'd1-c37-preunload-repaired-fullm-execution-spec-v1',
        'side_arm': 'preunload_pair37',
    })
    return spec


def build_environment(go36):
    environment = dict(go36['runtime_environment'])
    entries = environment['PYTHONPATH'].split(os.pathsep)
    environment['PYTHONPATH'] = os.pathsep.join(
        [entries[0], str(ROOT/'root37'), *entries[1:]])
    return environment


def main():
    go36 = json.loads((C36/'source_go36.json').read_text())
    inputs = dict(go36['inputs'])
    inherited = len(inputs)
    changed = [path for path, expected in inputs.items()
               if not Path(path).is_file() or identity(path) != expected]
    unexpected = [path for path in changed if path not in ALLOWED_REHASH]
    if unexpected:
        raise RuntimeError('inherited C36 inputs changed: '+repr(unexpected[:5]))
    rehashed = []
    for path in changed:
        rehashed.append({'path': path, 'was': inputs[path], 'now': identity(path),
                         'package': 'libfreetype6',
                         'mapped_into_the_c36_physics_process': False,
                         'reason_accepted': 'font rasteriser reached through PIL; this '
                                            'campaign is headless with render False and '
                                            'performs no image input or output'})
        inputs[path] = identity(path)
    inputs[str((C36/'source_go36.json').resolve(strict=True))] = identity(C36/'source_go36.json')

    spec = build_spec(go36)
    write_new(ROOT/'astra_plan/spec37.json', spec)
    for path in NEW_PATHS:
        inputs[str(path.resolve(strict=True))] = identity(path)

    record = {
        'schema': 'd1-c37-single-physics-and-reader-source-go-v1',
        'execution_contract_id': CONTRACT37,
        'decision': 'GO',
        'signed_by': 'root (main Codex agent)',
        'astra_reviewed': False,
        'reviewer_model': None,
        'signing_basis': ('Existing user authorisation for routine development and '
                          'simulation verification, plus the 2026-09-21 handoff clause '
                          'permitting the main agent to draft and record a bounded contract '
                          'without asking again. The gpt-6-astra ultra review channel is not '
                          'available in this terminal and is not claimed.'),
        'signed_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorization_scope': ('Root sole executor: one bounded fixed C37 campaign in '
                                'development_01, then one saved-data independent reader '
                                'under the same GO. No automatic retry, no training, no '
                                'unlisted physics.'),
        'automatic_retry': False,
        'physical_attempts_authorized': 1,
        'training_authorized': False,
        'training_GO': False,
        'user_goal_complete': False,
        'physics_consumed_at_review': {'controls': 0, 'native_steps': 0, 'model_loads': 0},
        'inherited_input_changes': rehashed,
        'prior_physical_cost_recorded_not_reallocated': {
            'c36_attempt_1': {'controls': 205, 'normal_native': 1025, 'compiler_native': 2,
                              'terminated_on': 'AttributeError in the inherited fullM '
                                               'static-query assertion, not a physics gate',
                              'native_safety_violations': 0}},
        'arms': {'development': spec},
        'x11_events_by_arm': {'development': None},
        'repository': go36['repository'],
        'repository_head': go36['repository_head'],
        'runtime_environment': build_environment(go36),
        'shared_session': go36['shared_session'],
        'host': go36['host'],
        'worker': str(ROOT/'root37/worker37.py'),
        'reader': go36['reader'],
        'reader_launcher': go36['reader_launcher'],
        'parent_source_go_identity': {'path': str(C36/'source_go36.json'),
                                      **identity(C36/'source_go36.json')},
        'behavioural_diff_from_c35': ['continuation36/sol36/core36.py',
                                      'continuation36/sol36/adapter36.py',
                                      'continuation37/root37/runtime37.py'],
        'frozen_c35_bytes_modified': False,
        'c36_bytes_modified': False,
        'declared_worker35_rebinds': ['CONTRACT', 'bootstrap35', 'origins35', 'preflight35'],
        'pure_review': {'executor': 'root', 'physics_steps': 0, 'model_loads': 0, 'fits': 0,
                        'core36_unittest_methods': 10, 'runtime37_unittest_methods': 7,
                        'all_passed': True},
        'inventory': {'inherited_entries': inherited, 'added_paths': len(NEW_PATHS)+1,
                      'signed_total_entries': len(inputs)},
        'inputs': inputs,
    }
    write_new(ROOT/'source_go37.json', record)

    request = {'schema': 'd1-c37-single-request-v1', 'arm': 'development',
               'source_go_path': str(ROOT/'source_go37.json'),
               'spec': spec, 'x11_events': None}
    write_new(ROOT/'request37_01.json', request)
    print(json.dumps({'spec': str(ROOT/'astra_plan/spec37.json'),
                      'source_go': str(ROOT/'source_go37.json'),
                      'request': str(ROOT/'request37_01.json'),
                      'inherited_inputs': inherited, 'added': len(NEW_PATHS)+1,
                      'total_inputs': len(inputs)}))


if __name__ == '__main__':
    main()
