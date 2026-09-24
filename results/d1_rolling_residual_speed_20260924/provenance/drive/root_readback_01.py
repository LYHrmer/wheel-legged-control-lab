"""Reconcile full saved intervention records without engine or policy imports."""
import gzip
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'stable_baselines3', 'gymnasium'}:
            raise RuntimeError('forbidden offline import: ' + fullname)


sys.meta_path.insert(0, NoEngine())
import numpy as np
from drive_damping_validator_04 import validate_drive_trace, validate_precontrol_archive
from scripts.d1_rolling_residual_scoring import score_rolling_episode
from scripts.d1_rolling_residual_task import RollingEpisodeSpec

D = Path(__file__).resolve().parent
O = D / 'run_01'
L = D.parent / 'stability_20260923_rl01'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def same(left, right):
    return np.array_equal(np.asarray(left), np.asarray(right))


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    receipt = read(O / 'drive_evaluation_receipt.json')
    outer = read(O / 'root_launcher_receipt.json')
    assert outer['exit_code'] == 0 and outer['error'] is None
    assert not outer['post_execution_hash_mismatches']
    assert receipt['execution_valid'] and receipt['counts_consistent']
    assert receipt['training_controls'] == 0 and receipt['old_run_02_final_C_counts'] is None
    frozen = read(D / 'root_execution_freeze_01.json')
    for name, expected in frozen['files'].items():
        path = Path(name)
        assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], name
    initial = read(O / 'runtime_initial.json')
    assert initial['binding_proof']['passed']
    assert all(initial['initial_C_state_after_arm'][key] == 0 for key in (
        'control_attempts', 'control_returns', 'construction_attempts', 'construction_returns',
        'ccd_attempts', 'ccd_returns', 'violations'))
    baseline = {row['case']: row for row in read(L / 'complete_cases_map_03.json')['cases']}
    result = {}
    previous_after = None
    for case in receipt['cases']:
        name = case['case']
        folder = O / 'evaluation' / name
        spec = RollingEpisodeSpec(case['speed_mps'], 175, 975, case['terrain'] == 'box')
        trace, endpoints, native = (rows(folder / filename) for filename in (
            'trace.jsonl.gz', 'endpoints.jsonl.gz', 'native.jsonl.gz'))
        saved = np.load(folder / 'states.npz', allow_pickle=False)
        old_folder = Path(baseline[name]['source_dir'])
        old = np.load(old_folder / 'states.npz', allow_pickle=False)
        for field in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'observation'):
            assert same(saved[field][0], old[field][0]), (name, field)
        count = len(trace)
        score = read(folder / 'score.json')
        assert score == case['score'] and score['record_valid']
        raw_receipt = read(folder / 'receipt.json')
        rescored = score_rolling_episode({
            'obstacle_enabled': spec.obstacle_enabled, 'spec': spec.as_dict(),
            'completed_control_intervals': count, 'endpoints': endpoints,
            'trace': trace, 'native': native, 'error': raw_receipt['error'],
            'terminated': raw_receipt['terminated'], 'truncated': raw_receipt['truncated'],
            'source_identity_valid': True,
            'geometry_manifest': read(folder / 'geometry_manifest.json'),
        }, spec)
        assert rescored == score, name
        before_states = read(folder / 'drive_precontrol_states.json')
        stage_saved = read(folder / 'drive_stage_validation.json')
        stage = validate_drive_trace(trace, expected_raw_schedule=[
            case['speed_mps'] if 175 <= tick < 975 else 0.0 for tick in range(count)],
            pre_control_states=before_states)
        assert stage == stage_saved['stage']
        assert validate_precontrol_archive(
            before_states, saved_qpos=saved['qpos'], saved_qvel=saved['qvel'],
            joint_qpos_addresses=stage_saved['actual_compiled_joint_qpos_addresses'],
            joint_dof_addresses=stage_saved['actual_compiled_joint_dof_addresses'],
        ) == stage_saved['precontrol_archive']
        assert len(native) == 5 * count and len(endpoints) == count + 1
        for index, item in enumerate(native):
            tick = index // 5
            assert item['index'] == index and item['returned'] and item['error'] is None
            assert abs(item['actual_dt_s'] - .002) < 1e-12
            assert same(item['ctrl_nm'], trace[tick]['torque_nm'])
            if index:
                assert same(native[index - 1]['qpos_returned'], item['qpos_before'])
                assert same(native[index - 1]['qvel_returned'], item['qvel_before'])
            if index % 5 == 0:
                assert same(saved['qpos'][tick], item['qpos_before'])
                assert same(saved['qvel'][tick], item['qvel_before'])
            if index % 5 == 4:
                assert same(saved['qpos'][tick + 1], item['qpos_returned'])
                assert same(saved['qvel'][tick + 1], item['qvel_returned'])
        events = rows(folder / 'native_entry_events.jsonl.gz')
        assert len(events) == len(native) and all(row['attempt'] == i for i, row in enumerate(events))
        before = read(O / 'evaluation' / f'{name}.before.json')
        after = read(O / 'evaluation' / f'{name}.after.json')
        assert before['error'] is None and after['error'] is None
        proof_sha = digest(O / 'runtime_initial.json')
        assert before['binding_proof_sha256'] == after['binding_proof_sha256'] == proof_sha
        for boundary in (before, after):
            assert boundary['C_state']['construction_attempts'] == boundary['C_state']['construction_returns'] == 5
            assert boundary['C_state']['ccd_attempts'] == boundary['C_state']['ccd_returns']
        assert after['C_state']['ccd_returns'] >= before['C_state']['ccd_returns']
        for key in ('native_monitor_checked', 'native_monitor_passed'):
            assert after[key] - before[key] == 5 * count
        for key in ('control_attempts', 'control_returns'):
            assert after['C_state'][key] - before['C_state'][key] == 5 * count
            if previous_after is not None:
                assert before['C_state'][key] == previous_after['C_state'][key]
        for key in ('control_attempted', 'control_completed'):
            assert after['python_ledger'][key] - before['python_ledger'][key] == count
        previous_after = after
        result[name] = {'controls': count, 'normal_native': len(native),
                        'record_valid': True, 'task_passed': score['task_passed'],
                        'initial_state_exact_to_old_policy': True,
                        'old_score': read(old_folder / 'score.json'), 'new_score': score,
                        'stage_validation': stage}
    controls = sum(row['controls'] for row in result.values())
    assert controls <= 4800 and receipt['actual_completed_controls'] == controls
    assert receipt['C_state']['control_attempts'] == receipt['C_state']['control_returns'] == 5 * controls
    assert receipt['C_state']['construction_attempts'] == receipt['C_state']['construction_returns'] == 5
    speed_deltas = {}
    for terrain in ('plane', 'box'):
        means = {case['speed_mps']: result[case['case']]['new_score']['metrics']['speed_window_mean_body_vx_mps']
                 for case in receipt['cases'] if case['terrain'] == terrain}
        speed_deltas[terrain] = None if means[.2] is None or means[.25] is None else means[.25] - means[.2]
    qualified = (len(result) == 4 and all(row['controls'] == 1200 and row['task_passed']
                                         for row in result.values())
                 and all(value is not None and value >= .03 for value in speed_deltas.values()))
    assert qualified == receipt['candidate_qualified']
    assert speed_deltas == receipt['policy_speed_mean_delta_mps']
    manifest = {str(path.relative_to(O)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
                for path in sorted(O.rglob('*')) if path.is_file()}
    write(D / 'run_01_archive_manifest_01.json', {'scope': 'all closed drive intervention files', 'files': manifest})
    final = {'passed': True, 'scope': 'offline full native chain, same frozen scoring, new stage, pair and C-boundary reconciliation',
             'frozen_inputs': len(frozen['files']), 'archive_files': len(manifest),
             'actual_controls': controls, 'actual_normal_native': 5 * controls,
             'actual_compiler_native': 5, 'candidate_qualified': receipt['candidate_qualified'],
             'cases': result, 'old_run02_final_C_counts': None,
             'independent_rl_benefit_established': False, 'new_physics_or_policy_calls': 0}
    write(D / 'root_run_01_readback_01.json', final)
    print(json.dumps({key: value for key, value in final.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
