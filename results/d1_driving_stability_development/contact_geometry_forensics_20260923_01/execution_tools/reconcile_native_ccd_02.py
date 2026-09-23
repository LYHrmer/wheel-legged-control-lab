"""Pure root reconciliation of the one recorded native-CCD execution."""
from pathlib import Path
import hashlib
import json

import numpy as np

W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923')
R = Path('/home/lyh/wheel-legged-control-lab')


def bits(a, b):
    return np.asarray(a, dtype=np.float64).tobytes() == np.asarray(b, dtype=np.float64).tobytes()


def main():
    run = W / 'native_ccd_run_02'
    events = [json.loads(line) for line in (run / 'ccd_events.ndjson').read_text().splitlines()]
    attempts = [x for x in events if x['event'] == 'ccd_attempt']
    returns = [x for x in events if x['event'] == 'ccd_return']
    cases = [x for x in events if x['event'] == 'case_complete']
    summary = next(x for x in events if x['event'] == 'summary')
    assert [x['attempt_seq'] for x in attempts] == list(range(1, 17))
    assert [x['attempt_seq'] for x in returns] == list(range(1, 17))
    expected = [(1, 3485, s) for s in range(5)] + [(2, 3486, s) for s in range(5)]
    expected += [(3, 3487, s) for s in range(5)] + [(4, 3486, 0)]
    assert [(x['case_no'], x['native_index'], x['slot']) for x in attempts] == expected
    for a, b in zip(attempts, returns):
        assert all(a[k] == b[k] for k in ('attempt_seq', 'case_no', 'native_index', 'slot'))
        assert events.index(a) < events.index(b)
        assert b['bounds_valid'] and b['status_finite'] and b['witness_valid'] and not b['warning_seen']
    initialized = [x for x in events if x['event'] == 'descriptors_initialized']
    assert [x['init_calls_total'] for x in initialized] == [2, 4, 6, 8]
    assert summary['ccd_attempts'] == summary['ccd_returns'] == 16
    assert summary['init_calls'] == 8 and summary['ccd_size_calls'] == 1
    assert summary['version_calls'] == summary['version_string_calls'] == 1
    assert summary['workspace_malloc_calls'] == summary['workspace_free_calls'] == 1
    assert summary['math_total'] == sum(summary['math_by_name'].values()) == 171
    assert all(summary[k] == 0 for k in ('new_control', 'new_native', 'model_compiles', 'model_allocations', 'data_allocations'))
    assert not (run / 'stderr.log').read_bytes()
    fixture = json.loads((R / 'tests/fixtures/d1_single_step_native_normal_anomaly.json').read_text())
    rows = {x['index']: x for x in fixture['native_samples']}
    prepared = json.loads((W / 'kernel_preparation_02/d1_contact_native_ccd_input.json').read_text())
    poses = {x['native_index']: x for x in prepared['poses']}
    comparison = []
    bad = []
    for case in cases:
        refs = {x['index']: x for x in rows[case['native_index']]['contacts']['contacts']}
        pose = poses[case['native_index']]
        assert case['input_unchanged'] and case['all_finite']
        init = initialized[case['case_no'] - 1]
        for obj, prefix in zip(init['descriptors'], ('wheel', 'box')):
            assert bits(obj['pos'], pose[prefix + '_center_world_m'])
            assert bits(obj['mat'], pose[prefix + '_rotation_world'])
        for c in case['final_candidates']:
            ref = refs[c['old_contact_index']]
            match = bits(c['pos'], ref['position_world_m']) and bits(c['dist'], ref['distance_m'])
            match &= bits(c['frame'], ref['frame_geom1_to_geom2'])
            assert match
            normal = -np.asarray(c['frame'][:3])
            center = np.asarray(pose['box_center_world_m'])
            half = np.array([.18, .62, .0075])
            tol = abs(c['dist']) + .001 + 1e-7
            point = np.asarray(c['pos'])
            axes = (np.abs(normal) > 1e-6) & (np.abs(point - center - np.copysign(half, normal)) <= tol)
            residual = float(np.linalg.norm(normal[~axes]))
            valid = bool(axes.any() and np.all(np.abs(point - center) <= half + tol) and residual <= .0021)
            assert valid == c['support_valid']
            entry = {'case_no': case['case_no'], 'native_index': case['native_index'],
                     'old_contact_index': c['old_contact_index'], 'slot': c['slot'],
                     'position_distance_frame_bitwise_equal': bool(match),
                     'support_valid': valid, 'cone_residual': residual}
            comparison.append(entry)
            if not valid:
                bad.append(entry)
    assert len(bad) == 1 and bad[0]['case_no'] == 2 and bad[0]['old_contact_index'] == 5
    primary_same = all(bits(cases[1]['final_candidates'][0][k], cases[3]['final_candidates'][0][k])
                       for k in ('pos', 'dist', 'frame', 'normal'))
    assert primary_same and cases[3]['all_support_valid'] and cases[3]['count'] == 1
    queries = []
    for a, ret in zip(attempts, returns):
        s = ret['status']
        x1, x2 = np.asarray(s['raw_status_x1']), np.asarray(s['raw_status_x2'])
        wheel, box = a['descriptor_before']
        c = np.asarray(wheel['mat']).reshape(3, 3).T @ (x1 - wheel['pos'])
        b = np.asarray(box['mat']).reshape(3, 3).T @ (x2 - box['pos'])
        delta = np.array([np.linalg.norm(c[:2]) - wheel['size'][0], abs(c[2]) - wheel['size'][1]])
        wheel_gap = float(np.linalg.norm(np.maximum(delta, 0)) + min(float(max(delta)), 0) - wheel['margin']/2)
        delta = np.abs(b) - box['size']
        box_gap = float(np.linalg.norm(np.maximum(delta, 0)) + min(float(max(delta)), 0) - box['margin']/2)
        queries.append({'attempt_seq': a['attempt_seq'], 'native_index': a['native_index'], 'slot': a['slot_name'],
                        'gjk_iterations': s['gjk_iterations'], 'epa_iterations': s['epa_iterations'],
                        'epa_status': s['epa_status'], 'return_distance_m': ret['return_distance'],
                        'witness_x1_minus_x2': (x1-x2).tolist(),
                        'x1_inflated_cylinder_signed_gap_m': wheel_gap,
                        'x2_inflated_box_signed_gap_m': box_gap})
    result = {'status': 'completed_and_reconciled', 'source_freeze_sha256': hashlib.sha256((W/'root_native_execution_freeze_02.json').read_bytes()).hexdigest(),
              'event_log_sha256': hashlib.sha256((run/'ccd_events.ndjson').read_bytes()).hexdigest(),
              'process_attempts': 1, 'ccd_attempts': 16, 'ccd_returns': 16, 'ccd_remaining': 0,
              'descriptor_initializations': 8, 'pure_math_explicit_calls': 171,
              'model_compiles': 0, 'engine_model_data_allocations': 0, 'new_control': 0, 'new_native_integration': 0,
              'comparison': comparison, 'bad_candidate': bad[0], 'primary_only_matches_original_primary_bitwise': primary_same,
              'query_observations': queries, 'classification': 'bitwise_reproduction_of_three_saved_pairs',
              'epa_status_zero_not_proof_of_convergence': True, 'internal_numerical_root_cause_confirmed': False,
              'qualification_granted': False, 'original_score_overridden': False, 'rl_gate_open': False,
              'note': 'Pure analysis of saved queries; signed primitive gaps are observations, not a solver-fix proof.'}
    with (W/'native_ccd_reconciliation_02.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparison', 'query_observations')}, indent=2))


if __name__ == '__main__':
    main()
