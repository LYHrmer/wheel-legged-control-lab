"""Saved-only C34 gain/scaled-input checks and unchanged C31 numeric readback."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('gain34_contract', Path(__file__).with_name('run_b22_gain34.py'))
c34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c34)
c = c34.c


def metadata_checker(prior):
    """Derive only explicit C34 source metadata; never edit a saved C31 identity."""
    path = c34.ROOT/'scripts/read_b22_suite31.py'
    c.require(c.identity(path)['sha256'] == c34.PINS['scripts/read_b22_suite31.py'], 'C31 reader source changed')
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'script_sources']
    c.require(len(functions) == 1, 'ambiguous pinned source-check function')
    node = copy.deepcopy(functions[0])
    replacements = {'C31_B22_fixed_suite_v1': c34.CONTRACT,
                    'scripts/run_b22_worker31.py': 'scripts/run_b22_worker34.py'}
    found = {key: 0 for key in replacements}
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value in replacements:
            original = item.value
            item.value = replacements[original]
            found[original] += 1
    c.require(all(count == 1 for count in found.values()), 'C34 metadata seam changed')
    node.name = 'script_sources34'
    namespace = dict(prior.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)  # noqa: S102 -- two literal metadata substitutions in pinned pure function
    return namespace['script_sources34']


def check_gain_row(row, gain):
    import numpy as np
    network = np.asarray(row.get('network_output16'), dtype=np.float32)
    action = np.asarray(row.get('action16'))
    expected = network*np.float32(gain)
    c.require(type(row.get('gain34')) is float and row['gain34'] == gain
              and network.shape == action.shape == (16,) and np.isfinite(network).all()
              and np.isfinite(action).all() and (abs(network) <= 1.).all()
              and np.array_equal(action, expected.astype(np.float64))
              and row['action16'] == row['raw_action16'] == row['policy_input_action']
              and row['policy_predict_called'] is True,
              'C34 network/gain/actual controller-input ledger differs')
    return 1


def read_stage(release_path, name):
    release = c34.validate_release(release_path)
    stage = next((item for item in c34.stages() if item['name'] == name), None)
    c.require(stage is not None, 'unknown fixed C34 stage')
    run = Path(release['plan']['output'])/name
    prior = c34.pinned('gain34_saved31', c34.ROOT/'scripts/read_b22_suite31.py')
    base, step, scorer = prior.helpers(release)
    session, worker, construction, count = metadata_checker(prior)(run, stage, release, base)
    c.require(type(session.get('gain34')) is float and session['gain34'] == stage['gain34']
              and worker['result'].get('gain34') == stage['gain34'], 'C34 session/result gain differs')
    rows = 0
    for row in base.saved_rows(run/'episode_0/controls.jsonl.gz'):
        c.require(row['control_index'] == rows, 'C34 gain ledger is not the whole ordered control stream')
        rows += check_gain_row(row, stage['gain34'])
    c.require(rows == count, 'C34 gain ledger count differs')
    canonical, pairing = prior.canonical_script(run, stage, session, worker, construction, count, base, step)
    scored = scorer.score_case(canonical, expected_seed=stage['seed'], mirror=False)
    return {'schema': 'b22-gain34-stage-readback-v1', 'status': 'COMPLETE', 'stage': name,
            'execution_contract_id': c34.CONTRACT, 'gain34': stage['gain34'],
            'release_identity': c.identity(release_path), 'reader_identity': c.identity(Path(__file__)),
            'actual_counts': {'controls': count, 'normal_native': 5*count, 'compiler_native': 2},
            'gain_ledger_rows': rows, 'pairing_identity': pairing, 'cases': [scored],
            'artifacts': prior.artifact_identities(run, base), 'reader_model_calls': 0, 'reader_physics_calls': 0}


def aggregate(reports):
    stages = c34.stages()
    c.require([item['stage'] for item in reports] == [stage['name'] for stage in stages],
              'C34 matrix missing, duplicated or reordered')
    for item, stage in zip(reports, stages):
        count = item['actual_counts']['controls']
        scored = item['cases'][0]
        c.require(item['status'] == 'COMPLETE' and item['execution_contract_id'] == c34.CONTRACT
                  and item['gain34'] == stage['gain34'] and type(count) is int and 0 < count <= 1600
                  and item['actual_counts']['normal_native'] == 5*count
                  and item['actual_counts']['compiler_native'] == 2 and item['gain_ledger_rows'] == count,
                  'C34 stage identity/actual budget differs')
        fixed = {'control_cap': 1600, 'hold': stage['hold'], 'release_tick_t0': stage['release_tick'],
                 'drive': [175, stage['release_tick']], 'final': [1500, 1600]}
        c.require(scored['case_id'] == stage['case_id'] and scored['seed'] == stage['seed']
                  and scored['actor'] == 'final_policy' and scored['mirror'] is False
                  and scored['completed_controls'] == count
                  and all(scored['windows'][key] == value for key, value in fixed.items()),
                  'C34 scored profile/actor/fixed windows differ')
        c.require(type(scored['record']['full_horizon']) is bool
                  and (not scored['record']['full_horizon'] or count == 1600),
                  'C34 full-horizon claim conflicts with actual controls')
    pairs = []
    all_full = all(item['cases'][0]['record']['full_horizon'] for item in reports)
    all_tasks = all(item['cases'][0]['task_passed'] for item in reports)
    for first, second in zip(reports[::2], reports[1::2]):
        a, b = first['cases'][0], second['cases'][0]
        c.require(first['pairing_identity'] == second['pairing_identity']
                  and all(a[key] == b[key] for key in ('case_id', 'seed', 'mirror')),
                  'new gain1/gain0.5 initial complete state or fixed window differs')
        sse = b['rl_terms']['sse_total']/a['rl_terms']['sse_total'] if all_full and a['rl_terms']['sse_total'] > 0 else None
        cost = b['rl_terms']['torque_cost_mean']/a['rl_terms']['torque_cost_mean'] if all_full and a['rl_terms']['torque_cost_mean'] > 0 else None
        pairs.append({'profile': a['case_id'], 'gain1': a, 'gain0p5': b,
                      'SSE_gain0p5_over_gain1': sse, 'cost_gain0p5_over_gain1': cost})
    return {'status': 'COMPLETE', 'all_four_full_horizons': all_full, 'all_four_tasks_passed': all_tasks,
            'full_window_ratios_available': all_full and all(
                item['SSE_gain0p5_over_gain1'] is not None and item['cost_gain0p5_over_gain1'] is not None for item in pairs),
            'pairs': pairs,
            'all_two_profile_SSE_and_cost_lower': bool(all_full and all_tasks and all(
                item['SSE_gain0p5_over_gain1'] is not None and item['cost_gain0p5_over_gain1'] is not None
                and item['SSE_gain0p5_over_gain1'] < 1. and item['cost_gain0p5_over_gain1'] < 1. for item in pairs)),
            'adoption_decision': 'NOT_AUTOMATIC', 'RL_contribution_claimed': False,
            'scope': 'four fresh sessions on two profiles; old zero is a diagnostic trigger only; no six-task contribution/generalization inference'}


def compare(release_path):
    release = c34.validate_release(release_path)
    root = Path(release['plan']['output'])
    reports, identities = [], {}
    for stage in c34.stages():
        path = root/(stage['name']+'_readback.json')
        item = c.read_json(path)
        c.require(item['release_identity'] == c.identity(release_path)
                  and item['reader_identity'] == c.identity(Path(__file__)), 'C34 reader/release identity differs')
        for name, expected in item['artifacts'].items():
            c.require(not Path(name).is_absolute() and '..' not in Path(name).parts
                      and c.identity(root/stage['name']/name) == expected, 'C34 raw artifact changed: '+name)
        identities[path.name] = c.identity(path)
        reports.append(item)
    return {'schema': 'b22-gain34-comparison-v1', 'release_identity': c.identity(release_path),
            'stage_reports': identities, **aggregate(reports)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stage')
    group.add_argument('--compare', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.release) if args.compare else read_stage(args.release, args.stage)
    c.write_json(args.report, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
