"""C34 saved worker/host and independently sealed cold reader closure."""
from __future__ import annotations

from pathlib import Path
import sys

from reader26 import require, identity, document, saved_document, MODEL_SHA

W = Path(__file__).resolve().parents[2]
R = Path('/home/lyh/wheel-legged-control-lab')
CONTRACT = 'C34_fixed_velocity_reference_and_40mm_v1'
RUNTIME = {'worker34', 'body_curve33', 'side_timing33', 'side_velocity34',
           'pair_prefix33',
           'task34', 'macro34', 'record34', 'runtime31',
           'record31', 'macro31', 'task31', 'observation31', 'side_skill31',
           'hybrid_adapter27', 'd1_fast_side_step', 'd1_side_step',
           'engine_binding', 'controller18', 'kinematics24',
           'world_upright_course_11'}


def sources34(run, session, worker, *, reader_review_path):
    run = Path(run).resolve(strict=True)
    go_path = Path(session['source_go_path']).resolve(strict=True)
    go = document(go_path)
    wanted_limits = (22000, 10, 40, 900, 960, 1020, 1200)
    keys = ('control_limit', 'cycles_limit', 'macros_limit', 'soft_s',
            'close_s', 'hard_s', 'outer_s')
    require(tuple(session[k] for k in keys) == wanted_limits
            and session['arm'] == 'development' and session['mode'] == 'fixed'
            and session['execution_contract_id'] == CONTRACT
            and session['render'] is False and session['retry_permitted'] is False
            and session['seed'] == 271001 and session['side_arm'] == 'teacher'
            and session['alpha'] == 1.
            and session['allowed_beta34'] == [0., .5, 1.]
            and session['allowed_distance34_m'] == [.03, .04]
            and session['spawn_position_m'] == [-8., -4.7, .455]
            and Path(session['output_directory']).resolve() == run,
            'C34 finite saved session differs from frozen execution bounds')
    require(identity(go_path) == session['source_go_identity']
            and go['decision'] == 'GO' and go['execution_contract_id'] == CONTRACT
            and all(session.get(k) == value for k, value in go['shared_session'].items())
            and all(session.get(k) == value for k, value in go['arms']['development'].items())
            and all(session['source_hashes'].get(path) == value
                    for path, value in go['inputs'].items()),
            'C34 actual session/source GO differs from reviewed frozen sources')
    for name in ('contract34.json', 'spec34.json'):
        path = W/'continuation34/astra_plan'/name
        require(session['source_hashes'].get(str(path)) == go['inputs'].get(str(path))
                == identity(path), 'C34 final contract/spec is absent from source GO: '+name)
    require(worker['schema'] == 'd1-c34-headless-worker-v1'
            and worker['execution_contract_id'] == CONTRACT
            and worker['arm'] == 'development'
            and worker['session_identity'] == identity(run/'session.json')
            and worker['failure'] is None and worker['cleanup_errors'] == []
            and worker['warnings'] == [] and worker['archive_failed'] is False
            and worker['execution_complete'] is True
            and worker['retry_permitted'] is False,
            'C34 failed worker/archive cannot qualify')
    host, supervisor = document(run/'host_receipt.json'), document(run/'supervisor_receipt.json')
    require(host['exit_code'] == supervisor['exit_code'] == 0
            and host['failure'] is supervisor['failure'] is None
            and host['source_mismatches'] == []
            and host['owned_no_orphans'] is host['reservation_closed'] is True
            and host['retry_permitted'] is False
            and host['reserved_controls'] == 22000
            and supervisor['cleanup']['remaining'] == {}
            and supervisor['outer_limit_s'] == 1200
            and supervisor['elapsed_s'] <= 1200,
            'C34 owned host, source or deadline did not close')
    child, ready = document(run/'host_child.json'), document(run/'preflight_ready.json')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', go['worker'],
               '--session', str(run/'session.json'), '--output', str(run)]
    require(child['argv'] == command and session['argv'] == command[4:]
            and type(child['pid']) is int and child['pid'] > 0
            and type(ready['pid']) is int and ready['pid'] > 0
            and ready['session_sha256'] == identity(run/'session.json')['sha256']
            and host['execution_started_monotonic'] == ready['monotonic_s']
            and session['host_started_monotonic'] <= ready['monotonic_s']
                <= session['host_started_monotonic']+host['elapsed_s'],
            'C34 actual host-accepted worker readiness differs')
    sources = session['source_hashes']
    require(sources.get(go['worker']) == identity(go['worker'])
            and sources.get(go['host']) == identity(go['host'])
            and sources.get(str(go_path)) == identity(go_path),
            'C34 worker/host/GO identities changed')
    early = document(run/'runtime34_module_origins_before.json')
    final = saved_document(run/'runtime34_module_origins.json')
    require(early == final and RUNTIME <= set(final),
            'C34 runtime module chain changed after bootstrap')
    for name, value in final.items():
        path = Path(value['path'])
        require(sources.get(str(path)) == identity(path)
                == {key: value[key] for key in ('bytes', 'sha256')},
                'C34 loaded runtime source differs: '+name)
    loaded = saved_document(run/'loaded_origins_final.json')
    require(all(path in sources for path in loaded['module_origins'].values())
            and all(row['identity_verified'] is True and row['defining_source'] in sources
                    for row in loaded['synthetic_module_aliases'].values()),
            'C34 numerical loaded-source closure differs')
    for name in ('mapped_libraries_before_load.json', 'mapped_libraries_final.json'):
        paths = saved_document(run/name)['paths']
        require(paths and all(path in sources for path in paths),
                'C34 mapped numerical library lacks a frozen identity')
    manifest, relocation = session['checkpoint_manifest'], session['checkpoint_relocation']
    original_path = Path(relocation['original_manifest_path'])
    original = document(original_path)
    require(identity(original_path) == relocation['original_manifest_identity']
            == sources[str(original_path)]
            and relocation['changed_fields'] == ['folder']
            and original['folder'] == relocation['original_folder']
            and relocation['current_folder'] == session['checkpoint_folder']
            and manifest == {**original, 'folder': session['checkpoint_folder']},
            'C34 B22 checkpoint relocation changed numerical identity')
    for name, value in manifest['files'].items():
        require(identity(Path(session['checkpoint_folder'])/name) == value,
                'C34 B22 checkpoint payload differs: '+name)
    load = saved_document(run/'strict_load.json')
    require(session['checkpoint_sha256'] == MODEL_SHA
            == manifest['files']['final_model.zip']['sha256']
            and load['probe_rows'] == 32
            and load['probe_actions_byte_exact'] is True
            and load['matches_training_final_policy_state'] is True
            and load['verified_without_engine_or_reset'] is True
            and load['observation_space']['shape'] == [99]
            and load['action_space']['shape'] == [16],
            'C34 strict frozen B22 load/probe differs')
    review_path = Path(reader_review_path).resolve(strict=True)
    review = document(review_path)
    require(review['decision'] == 'GO' and isinstance(review['inputs'], dict),
            'C34 reader lacks an independently frozen source closure')
    roots = (W, R/'runtime/d1_b22/research')
    checked = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != '.py' or not any(path.is_relative_to(root) for root in roots):
            continue
        expected = sources.get(str(path), review['inputs'].get(str(path)))
        require(expected == identity(path), 'C34 loaded reader source is unsealed: '+name)
        if str(path) in sources and str(path) in review['inputs']:
            require(review['inputs'][str(path)] == sources[str(path)],
                    'C34 reader review attempted to override runtime source')
        checked[name] = dict(path=str(path), **identity(path))
    entry = Path(sys.argv[0]).resolve()
    require(entry.is_relative_to(W/'continuation34/read34')
            and review['inputs'].get(str(entry)) == identity(entry),
            'C34 actual reader CLI is not independently sealed')
    return dict(passed=True, actual_runtime_identities=final,
                reader_review_identity=identity(review_path),
                actual_reader_sources=checked,
                source_GO_identity=identity(go_path),
                host_online_descendant_admission_accepted=True,
                direct_saved_PID_PPID_tree_available=False,
                full_source_pre_and_postcheck_evidence='saved host source_mismatches=[]')
