"""Close the new GUI contract by reading actual files; never import physics."""
import hashlib
import json
from pathlib import Path

W = Path(__file__).resolve().parent
B = W.parent
R = Path('/home/lyh/wheel-legged-control-lab')
OLD = B / 'stability_20260924_body_speed01'


def read(path):
    return json.loads(path.read_text())


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def save(name, value):
    with (W / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    scopes = (
        ('frozen77', B / 'rl_improvement_20260914/frozen_source_before.json', 'sha256', R),
        ('formal215', B / 'stability_20260920/shared_heave_formal_preflight_01.json', 'input_sha256', R),
        ('physical193', B / 'stability_20260922/single_step_readiness_01/protocol.json', 'input_sha256', R),
        ('runtime767', OLD / 'root_execution_freeze_01.json', 'files', R),
        ('closed77', OLD / 'run_01_archive_manifest_01.json', 'files', OLD / 'run_01'),
        ('published06', R / 'results/d1_body_speed_feedback_20260924/publication_manifest.json', 'files',
         R / 'results/d1_body_speed_feedback_20260924'),
        ('published05', R / 'results/d1_rolling_residual_speed_20260924/publication_manifest.json', 'files',
         R / 'results/d1_rolling_residual_speed_20260924'),
    )
    checked, counts = {}, {}
    for label, manifest, key, base in scopes:
        rows = read(manifest)[key]
        for name, expected in rows.items():
            path = base / name
            if str(path) not in checked:
                checked[str(path)] = identity(path)
            actual = checked[str(path)]
            assert actual['sha256'] == (expected if isinstance(expected, str) else expected['sha256']), path
            if isinstance(expected, dict) and 'bytes' in expected:
                assert actual['bytes'] == expected['bytes'], path
        counts[label] = len(rows)
        if label in ('closed77', 'published05', 'published06'):
            actual_names = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
            expected_names = set(rows) | ({'publication_manifest.json'} if label.startswith('published') else set())
            assert actual_names == expected_names, label
    go = read(W / 'astra_gui_review_inputs_07.json')
    for name, expected in go['inputs'].items():
        assert identity(Path(name)) == expected, name
    assert go['decision'] == 'GO'
    profiles = {}
    ccd = 0
    for profile, readback_name in (
        ('gui_policy_box', 'gui_policy_box_readback_02.json'),
        ('headless_zero_plane', 'headless_zero_plane_readback_01.json'),
    ):
        folder = W / 'validation_runs' / (profile + '_01')
        manifest = read(W / (profile + '_closed_manifest_01.json'))
        actual_files = {str(p) for p in folder.rglob('*') if p.is_file()}
        assert actual_files == set(manifest['files'])
        for name, expected in manifest['files'].items():
            assert identity(Path(name)) == expected, name
        worker, launcher = read(folder / 'worker_receipt.json'), read(folder / 'launcher_receipt.json')
        result = read(W / readback_name)
        outer = read(W / (profile + '_root_exit_01.json'))
        assert worker['execution_complete'] and worker['failure'] is None
        assert launcher['exit_code'] == 0 and launcher['source_hash_mismatches'] == []
        assert result['passed'] and result['controls'] == 1200 and result['normal_native'] == 6000
        assert result['compiler_native'] == 3 and not result['engine_or_policy_imported']
        assert outer['exit_code'] == 0 and outer['error'] is None
        assert not outer['cleanup']['owned_group_alive_after_cleanup']
        assert not outer['cleanup']['signals_sent']
        ccd += worker['C_final']['ccd_returns']
        profiles[profile] = {'readback': readback_name, 'readback_identity': identity(W / readback_name),
                             'run_files': len(actual_files), 'controls': result['controls'],
                             'normal_native': result['normal_native'], 'compiler_native': result['compiler_native'],
                             'ccd': worker['C_final']['ccd_returns'],
                             'live_policy_calls': worker['live_policy_predict_calls']}
    for i in (1, 2):
        receipt = read(W / f'pure_test_round_{i}_receipt.json')
        assert receipt['all_passed'] and receipt['collected'] == 12 and receipt['exit_code'] == 0
    save('final_execution_audit_07.json', {
        'schema': 'd1-latest-rl-gui-closed-execution-v1', 'passed': True,
        'old_scope_counts': counts, 'old_unique_files_rehashed': len(checked), 'hash_mismatches': [],
        'go_inputs_unchanged': len(go['inputs']), 'closed_run_files_unchanged': 98,
        'profiles': profiles, 'new_control_steps': 2400, 'new_normal_native_steps': 12000,
        'new_compiler_native_steps': 6, 'new_ccd_returns': ccd, 'new_training_steps': 0,
        'physical_contract_07_closed': True, 'additional_physics_under_07_permitted': False,
        'pure_rounds_completed': 2, 'pure_case_executions_passed': 24,
        'readback_implementation_repair': '02 only fixes import and actual JSON field names; original preserved',
        'readback_first_attempt_failed_before_verification': True,
        'user_manual_results_touched': False, 'root_saw_all_three_actual_screenshots': True,
        'human_keyboard_usability_assessed': False, 'independent_rl_benefit_established': False,
        'R_is_simulation_reset': True, 'physical_self_righting': False,
    })
    print(json.dumps({'closed': True, 'old_unique_files': len(checked), 'profiles': profiles}))


if __name__ == '__main__':
    main()
