"""C31 v2 source closure: distinguish the RTK child from its worker descendant.

The runtime closure stays frozen. New reader orchestration is separately sealed
before readback and cannot rewrite the preregistered mathematics or task gates.
"""
from __future__ import annotations

from pathlib import Path
import sys

from reader26 import require, identity, document, saved_document, MODEL_SHA

W = Path(__file__).resolve().parents[2]
P = W / 'continuation31'
R = Path('/home/lyh/wheel-legged-control-lab')
BUNDLE = R / 'runtime/d1_b22/research'
CONTRACT = 'C31_event_lateral_RL_pilot_v1'
RUNTIME_MODULES = {
    'worker30', 'runtime30', 'record30', 'macro30', 'side_skill30',
    'worker31', 'runtime31', 'record31', 'macro31', 'task31', 'ledger31',
    'observation31', 'side_skill31', 'learning31', 'checkpoint31',
    'hybrid_adapter27', 'side_runtime27', 'd1_fast_side_step', 'd1_side_step',
    'engine_binding', 'controller18', 'kinematics24', 'world_upright_course_11',
}
HOST_IDENTITY = {'bytes': 12875,
    'sha256': 'f02ff33cd232b4789b394483b4e10b66070d58e8a5a95929bd1d44ee204c6105'}


def readiness31(run, session, ready, child, host, go, session_identity, host_identity):
    """Use the executed host's online ancestry check, not an invented PID tree.

    This exact frozen host checks ready.pid in descendants(host_pid), including
    the RTK worker child, before accepting session SHA and monotonic time. Its
    successful receipt records the accepted timestamp. The historical files do
    not preserve a PID/PPID tree; no PID adjacency or equality inference is used.
    """
    run = Path(run).resolve()
    expected = ['rtk', 'proxy', '/usr/bin/python3', '-B', go['worker'],
                '--session', str(run/'session.json'), '--output', str(run)]
    require(host_identity == HOST_IDENTITY
            and session['source_hashes'].get(go['host']) == HOST_IDENTITY,
            'readiness ancestry proof requires the exact frozen descendant-checking host')
    require(type(ready['pid']) is int and ready['pid'] > 0
            and type(child['pid']) is int and child['pid'] > 0
            and child['argv'] == expected and session['argv'] == expected[4:]
            and ready['session_sha256'] == session_identity['sha256'],
            'C31 actual wrapper/worker command or readiness session differs')
    require(host['exit_code'] == 0 and host['failure'] is None
            and host['source_mismatches'] == [] and host['owned_no_orphans'] is True
            and host['reservation_closed'] is True
            and host['owned_cleanup']['method'] == 'linux_subreaper_ancestry_birth_tick'
            and host['owned_cleanup']['remaining'] == {}
            and host['execution_started_monotonic'] == ready['monotonic_s']
            and session['host_started_monotonic'] <= ready['monotonic_s']
                <= session['host_started_monotonic']+host['elapsed_s'],
            'C31 host did not successfully accept and close this worker readiness')
    return dict(passed=True, rtk_wrapper_pid=child['pid'], actual_worker_ready_pid=ready['pid'],
        accepted_ready_monotonic_s=ready['monotonic_s'], host_source_identity=host_identity,
        proof_kind='source_derived_from_successful_frozen_host_online_descendant_admission',
        direct_saved_PID_PPID_tree_available=False, PID_equality_or_adjacency_used=False,
        checks='exact wrapper and worker argv; session SHA; accepted monotonic timestamp; frozen host online ancestry check; successful subreaper closure')


def _reader_sources31(sources, review_path):
    review_path = Path(review_path).resolve(strict=True)
    review = document(review_path)
    require(review.get('decision') == 'GO' and isinstance(review.get('inputs'), dict),
            'C31 independent reader lacks its separate reviewed source closure')
    reader_sources = review['inputs']
    checked = {}
    roots = (W, R / 'runtime/d1_b22/research')
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        actual = Path(filename).resolve()
        # Only project readback modules are joined here. Python/NumPy and the
        # remaining historical dependency inventory are frozen by the host.
        if (actual.suffix != '.py' or not any(actual.is_relative_to(r) for r in roots)
                or not any(part in ('read31', 'read30', 'read27', 'root26',
                                    'root25', 'root23', 'verify18', 'verify22',
                                    'verify21', 'verify17', 'verify16')
                           for part in actual.parts)):
            continue
        actual_id = identity(actual)
        expected = sources.get(str(actual))
        if expected is None and actual.is_relative_to(W):
            expected = sources.get(str(BUNDLE / actual.relative_to(W)))
        if expected is None:
            expected = reader_sources.get(str(actual))
        require(expected == actual_id, 'C31 actual reader module is not sealed: ' + name)
        # A separate review may not override a runtime-frozen helper identity.
        if str(actual) in sources and str(actual) in reader_sources:
            require(reader_sources[str(actual)] == sources[str(actual)],
                    'C31 reader review tries to replace a frozen helper')
        checked[name] = dict(path=str(actual), **actual_id)
    own = Path(__file__).resolve()
    require(reader_sources.get(str(own)) == identity(own)
            and any(row['path'] == str(own) for row in checked.values()),
            'C31 source reader itself is not in the new readback closure')
    entry = Path(sys.argv[0]).resolve()
    require(entry.is_relative_to(P / 'read31')
            and reader_sources.get(str(entry)) == identity(entry),
            'C31 actual readback CLI is not independently sealed')
    return dict(review_path=str(review_path), review_identity=identity(review_path),
                actual_reader_sources=checked)


def sources31(run, session, worker, *, reader_review_path=None):
    run = Path(run).resolve(strict=True)
    train = session.get('arm') == 'train'
    expected = ((140800, 64, 256, 4200, 4500, 4560, 4800) if train else
                (22000, 10, 40, 900, 960, 1020, 1500))
    keys = ('control_limit', 'cycles_limit', 'macros_limit', 'soft_s',
            'close_s', 'hard_s', 'outer_s')
    require(session.get('arm') in ('train', 'evaluation')
            and session['execution_contract_id'] == CONTRACT
            and tuple(session[k] for k in keys) == expected
            and session['render'] is False and session['retry_permitted'] is False
            and session['side_arm'] == 'teacher'
            and session['spawn_position_m'] == [-8., -4.7, .455]
            and session['seed'] == 271001 and session['policy_seed'] == 310031
            and Path(session['output_directory']).resolve() == run,
            'C31 finite headless session differs')
    go_path = Path(session['source_go_path']).resolve(strict=True)
    go = document(go_path)
    require(identity(go_path) == session['source_go_identity']
            and go['decision'] == 'GO' and go['execution_contract_id'] == CONTRACT
            and all(session.get(k) == v for k, v in go['shared_session'].items())
            and all(session.get(k) == v for k, v in go['arms'][session['arm']].items()),
            'C31 session is not its actual reviewed arm')
    require(worker['schema'] == 'd1-c31-headless-worker-v1'
            and worker['execution_contract_id'] == CONTRACT
            and worker['arm'] == session['arm']
            and worker['session_identity'] == identity(run / 'session.json')
            and worker['failure'] is None and worker['cleanup_errors'] == []
            and worker['warnings'] == [] and worker['archive_failed'] is False
            and worker['execution_complete'] is True,
            'C31 failed worker/source/archive cannot become a valid training run')
    host = document(run / 'host_receipt.json')
    supervisor = document(run / 'supervisor_receipt.json')
    require(host['exit_code'] == 0 and host['failure'] is None
            and host['source_mismatches'] == [] and host['owned_no_orphans'] is True
            and host['reservation_closed'] is True and host['retry_permitted'] is False
            and host['reserved_controls'] == expected[0]
            and supervisor['exit_code'] == 0 and supervisor['failure'] is None
            and supervisor['cleanup']['remaining'] == {}
            and supervisor['outer_limit_s'] == expected[-1]
            and supervisor['elapsed_s'] <= expected[-1],
            'C31 actual host/supervisor/source check did not close')
    ready = document(run / 'preflight_ready.json')
    child = document(run / 'host_child.json')
    ownership = readiness31(run, session, ready, child, host, go,
                            identity(run/'session.json'), identity(go['host']))
    sources = session['source_hashes']
    require(all(sources.get(path) == wanted for path, wanted in go['inputs'].items())
            and sources.get(str(go_path)) == identity(go_path),
            'C31 session source closure differs from source GO')
    require(sources.get(go['worker']) == identity(Path(go['worker']))
            and sources.get(go['host']) == identity(Path(go['host'])),
            'C31 actual worker/host source identity differs')
    before = document(run / 'runtime31_module_origins_before.json')
    final = saved_document(run / 'runtime31_module_origins.json')
    require(before == final and RUNTIME_MODULES <= set(final),
            'C31 early/final real production import chain differs')
    checked = {}
    for name, value in final.items():
        path = Path(value['path'])
        wanted = {key: value[key] for key in ('bytes', 'sha256')}
        require(sources.get(str(path)) == identity(path) == wanted,
                'C31 actual runtime source changed: ' + name)
        checked[name] = wanted
    loaded = saved_document(run / 'loaded_origins_final.json')
    require(all(path in sources for path in loaded['module_origins'].values())
            and all(row['identity_verified'] is True and row['defining_source'] in sources
                    for row in loaded['synthetic_module_aliases'].values()),
            'C31 numerical import/alias identity is unbound')
    for name in ('mapped_libraries_before_load.json', 'mapped_libraries_final.json'):
        paths = saved_document(run / name)['paths']
        require(paths and all(path in sources for path in paths),
                'C31 actual mapped numerical DSO lacks frozen identity')
    relocation, manifest = session['checkpoint_relocation'], session['checkpoint_manifest']
    original_path = Path(relocation['original_manifest_path'])
    original = document(original_path)
    require(identity(original_path) == relocation['original_manifest_identity']
            == sources[str(original_path)]
            and relocation['changed_fields'] == ['folder']
            and original['folder'] == relocation['original_folder']
            and relocation['current_folder'] == session['checkpoint_folder']
            and manifest == {**original, 'folder': session['checkpoint_folder']},
            'C31 B22 relocation changed numeric identity')
    for name, wanted in manifest['files'].items():
        require(identity(Path(session['checkpoint_folder']) / name) == wanted,
                'C31 B22 checkpoint payload differs')
    load = saved_document(run / 'strict_load.json')
    require(session['checkpoint_sha256'] == manifest['files']['final_model.zip']['sha256']
            == MODEL_SHA and load['probe_rows'] == 32
            and load['probe_actions_byte_exact'] is True
            and load['matches_training_final_policy_state'] is True
            and load['verified_without_engine_or_reset'] is True
            and load['observation_space']['shape'] == [99]
            and load['action_space']['shape'] == [16],
            'C31 strict frozen B22 99D/16D load/probe differs')
    reader = _reader_sources31(sources, reader_review_path or
                              P / 'reader_source_review_31_v2.json')
    return dict(passed=True, actual_runtime_identities=checked, **reader,
                worker_readiness_ownership=ownership,
                source_GO_identity=identity(go_path),
                full_source_pre_and_postcheck_evidence='successful saved host with source_mismatches=[]',
                full_historical_inventory_rehashed_by_reader=False,
                lateral_checkpoint_and_learning_closure_checked_separately=True)
