"""Read-only evaluator review with synthetic record arithmetic, zero new physics."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import tempfile

from scripts.evaluate_d1_heading_study import common_prefix

WORK = Path(__file__).resolve().parent
REPO = Path('/home/lyh/wheel-legged-control-lab')


def main():
    checks = []
    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})

    with tempfile.TemporaryDirectory(prefix='heading_prefix_review_') as directory:
        output = Path(directory)
        values = {'zero': [0., 0., 0., 10000.],
                  'seed49001_step16384': [1., 2.],
                  'seed49001_step65536': [4., 5., 6.]}
        for label, samples in values.items():
            (output / label).mkdir()
            with gzip.open(output / label / 'trace.jsonl.gz', 'wt') as stream:
                for index, value in enumerate(samples):
                    row = {'metrics': {'time_s': (index + 1) * .01,
                                       'velocity_error_mps': value, 'height_error_m': value * 2},
                           'heading_task': {'heading_error_after': value * 3},
                           'cross_track_after_m': -value * 4}
                    stream.write(json.dumps(row) + '\n')
        result = common_prefix(output, list(values))
        check('zero_reference_and_16k_vs_65k_pairs_present', len(result['comparisons']) == 3)
        for pair in result['comparisons']:
            a, b = pair['first'], pair['second']
            expected_n = min(len(values[a]), len(values[b]))
            check(f'{a}:{b}:actual_common_steps', pair['actual_common_transitions'] == expected_n)
            check(f'{a}:{b}:actual_common_time', pair['time_s'] == expected_n * .01)
            for label in (a, b):
                expected = math.sqrt(sum(x*x for x in values[label][:expected_n]) / expected_n)
                metrics = pair['metrics'][label]
                for key, scale in [('velocity_rmse_mps', 1), ('height_rmse_m', 2),
                                   ('heading_rmse_rad', 3), ('cross_track_rmse_m', 4)]:
                    check(f'{a}:{b}:{label}:{key}', math.isclose(metrics[key], scale*expected,
                                                              rel_tol=1e-15, abs_tol=1e-15))
                check(f'{a}:{b}:{label}:last_shared_sample',
                      metrics['cross_track_final_m'] == -4 * values[label][expected_n-1])

    files = ['scripts/evaluate_d1_heading_study.py', 'scripts/d1_heading_tracking_env.py',
             'scripts/run_d1_heading_study.py', 'tests/test_d1_heading_study_evaluation.py']
    report = {
        'schema': 'd1-heading-evaluator-independent-review-v1',
        'read_only_source_review': True, 'additional_physical_steps': 0,
        'source_sha256': {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                          for name in files},
        'blockers': [],
        'reviewed_behavior': [
            'Each branch predicts deterministically from its own newly returned 85-value observation.',
            'Original applied actions, qpos/qvel and all observations are retained; first terminated/truncated immediately ends the case.',
            'Velocity and height use unchanged user target channels; user-yaw-rate error is explicitly separated from the servo reward.',
            'Cross track is endpoint world y minus initial world y; the only accepted development case has zero user yaw and the base spawn heading is zero.',
            'Heading RMSE uses wrapped endpoint reference error provided by the separately tested task environment.',
            'Terminal summary separates actual world roll/pitch, clearance, contact count and x/y boundary predicates.',
            'Protocol binds source, opened development protocol, training protocol, every checkpoint and sidecar SHA before evaluation; final check verifies unchanged inputs.',
            'Common-prefix comparisons use the shorter actual trace without padding, including zero-vs-model and within-seed 16k-vs-65k pairs.',
        ],
        'limitations': [
            'This reviews the implementation and independently verifies synthetic prefix arithmetic; it does not audit completed formal evaluation records.',
            'The road is an already opened development case; these metrics do not establish turning, braking, G1 or held-out performance.',
            'If execution raises an exception instead of returning an environment termination, the gzip trace prefix is retained but final NPZ and episode summary are not written; an incomplete directory cannot be treated as a complete case.',
            'The evaluator hashes generated input checkpoints but does not itself reconstruct training provenance or validate their budget counters; use the separately recorded training checkpoint metadata and counts for that audit.',
        ],
        'checks': checks, 'check_count': len(checks),
        'failures': sum(not item['passed'] for item in checks),
    }
    target = WORK / 'heading_evaluator_review.json'
    with target.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(target), 'checks': len(checks), 'failures': report['failures']}))
    assert report['failures'] == 0


if __name__ == '__main__':
    main()
