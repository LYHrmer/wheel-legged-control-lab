"""C35 saved-only gain/identity checks, unchanged full native and task scoring."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('gain_suite35_contract', Path(__file__).with_name('run_b22_gain_suite35.py'))
c35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c35)
c = c35.c


def metadata_checker(prior):
    path = c35.ROOT/'scripts/read_b22_suite31.py'
    c.require(c.identity(path)['sha256'] == c35.PINS['scripts/read_b22_suite31.py'], 'C35 prior reader source changed')
    node = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                              if isinstance(n, ast.FunctionDef) and n.name == 'script_sources'))
    replacements = {'C31_B22_fixed_suite_v1': c35.CONTRACT,
                    'scripts/run_b22_worker31.py': 'scripts/run_b22_worker35.py'}
    found = {key: 0 for key in replacements}
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value in replacements:
            old = item.value
            item.value = replacements[old]
            found[old] += 1
    c.require(all(n == 1 for n in found.values()), 'C35 source metadata seam changed')
    node.name = 'script_sources35'
    namespace = dict(prior.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)  # noqa: S102 -- pinned pure identity function, two explicit metadata substitutions
    return namespace['script_sources35']


def step_checker(step):
    path = c35.ROOT/'runtime/d1_b22/research/continuation24/read24.py'
    c.require(c.identity(path)['sha256'] == c35.PINS[str(path.relative_to(c35.ROOT))], 'C35 step reader source changed')
    node = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                              if isinstance(n, ast.FunctionDef) and n.name == 'read_run'))
    found = 0
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and item.value == 'd1-true15mm-worker-24-v1':
            item.value = 'd1-true15mm-worker-35-v1'
            found += 1
    c.require(found == 1, 'C35 step readback schema seam changed')
    namespace = dict(step.__dict__, CONTRACT_ID=c35.CONTRACT, READ_SCHEMA='d1-true15mm-independent-readback-35-v1')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)  # noqa: S102 -- pinned full C24 check, explicit version identity only
    return namespace['read_run']


def check_gain_row(row, actor):
    import numpy as np
    network = np.asarray(row.get('network_output16'), dtype=np.float32)
    action = np.asarray(row.get('policy_input_action'))
    c.require(type(row.get('gain35')) is float and row['gain35'] == .5
              and network.shape == action.shape == (16,) and np.isfinite(network).all()
              and np.isfinite(action).all() and (abs(network) <= 1.).all()
              and np.array_equal(action, (network*np.float32(.5)).astype(np.float64))
              and row.get('action16') == row.get('raw_action16') == row['policy_input_action']
              and row.get('policy_predict_called') is (actor == 'B')
              and (actor == 'B' or (network == 0.).all()), 'C35 network/gain/actual controller-input ledger differs')
    return 1


def check_derivation(run, release):
    path = c35.ROOT/'scripts/run_b22_worker35.py'
    worker = c.load_pinned(path, release['inputs'][str(path)], 'gain_suite35_derivation')
    c.require(c.read_json(run/'derived_sources35.json') == worker.derivation_proof(worker.derived_sources()),
              'C35 executed source derivation differs')


def read_stage(release_path, name):
    release = c35.validate_release(release_path)
    stage = next((item for item in c35.stages() if item['name'] == name), None)
    c.require(stage is not None, 'unknown C35 fixed stage')
    run = Path(release['plan']['output'])/name
    check_derivation(run, release)
    prior = c35.pinned('gain_suite35_numeric31', c35.ROOT/'scripts/read_b22_suite31.py')
    base, step, scorer = prior.helpers(release)
    common = {'schema': 'b22-gain-suite35-stage-readback-v1', 'status': 'COMPLETE', 'stage': name,
              'execution_contract_id': c35.CONTRACT, 'gain35': .5,
              'release_identity': c.identity(release_path), 'reader_identity': c.identity(Path(__file__)),
              'reader_model_calls': 0, 'reader_physics_calls': 0, 'retry_permitted': False}
    rows = 0
    if stage['kind'] == 'step':
        session = c.read_json(run/'session.json')
        worker = c.read_json(run/'worker_receipt.json')
        c35.case_completion(stage, worker)
        config = c.read_json(c35.ROOT/c.MANIFEST)
        expected_worker = str(c35.ROOT/'scripts/run_b22_worker35.py')
        c.require(session['output_directory'] == str(run) and session.get('gain35') == .5
                  and session['argv'] == [expected_worker, '--session', str(run/'session.json'), '--output', str(run)]
                  and all(session['source_hashes'].get(k) == v for k, v in release['inputs'].items())
                  and all(session.get(k) == v for k, v in config['shared_session'].items()), 'C35 step owner/source differs')
        original = step_checker(step)(run)
        for summary in original['heldout_summaries']:
            folder = run/'heldout'/(summary['case_id']+'_'+summary['actor'])
            receipt = step.document(folder/'case_receipt.json')
            count = 0
            for row in step.control_rows(folder, receipt['controller_record_blocks']):
                c.require(row['tick'] == count, 'C35 step gain ledger order differs')
                count += check_gain_row(row, summary['actor'])
            c.require(count == summary['completed_controls'], 'C35 step whole gain ledger missing')
            rows += count
        common.update(actual_counts={'controls': original['actual_completed_controls'],
                                     'normal_native': original['actual_normal_native_returns'],
                                     'compiler_native': original['actual_compiler_native_returns']},
                      cases=original['numeric_scores'], original_C24_numerical_readback=original)
    else:
        session, worker, construction, count = metadata_checker(prior)(run, stage, release, base)
        c.require(type(session.get('gain35')) is float and session['gain35'] == .5
                  and worker['result'].get('gain35') == .5, 'C35 session/result gain differs')
        if stage['terrain'] == 'ramp':
            evidence = c.read_json(run.parent/'step_pair_readback.json')
            c.require(evidence['status'] == 'COMPLETE' and evidence['execution_contract_id'] == c35.CONTRACT
                      and evidence['release_identity'] == c.identity(release_path), 'fresh C35 step kinematic proof absent')
            for filename, expected in evidence['artifacts'].items():
                c.require(c.identity(run.parent/'step_pair'/filename) == expected, 'fresh C35 step artifact changed')
        for row in base.saved_rows(run/'episode_0/controls.jsonl.gz'):
            c.require(row['control_index'] == rows, 'C35 script gain ledger order differs')
            rows += check_gain_row(row, stage['actor'])
        canonical, pairing = prior.canonical_script(run, stage, session, worker, construction, count, base, step)
        common.update(actual_counts={'controls': count, 'normal_native': 5*count, 'compiler_native': 2},
                      pairing_identity=pairing, cases=[scorer.score_case(canonical, expected_seed=stage['seed'], mirror=stage['mirror'])])
    c.require(rows == common['actual_counts']['controls'], 'C35 whole-stage gain ledger incomplete')
    return {**common, 'gain_ledger_rows': rows, 'artifacts': prior.artifact_identities(run, base)}


def aggregate(reports, pair_scorer, baseline):
    c.require(len(reports) == len(c35.stages()), 'C35 matrix incomplete')
    for item, stage in zip(reports, c35.stages()):
        c35.stage_counts(stage, item, item.get('release_identity'), item.get('reader_identity'))
        c.require(item['stage'] == stage['name'], 'C35 matrix reordered')
        if stage['kind'] == 'script':
            score = item['cases'][0]
            c.require(score['case_id'] == stage['case_id'] and score['seed'] == stage['seed']
                      and score['mirror'] is stage['mirror'] and score['actor'] == ('zero' if stage['actor'] == 'zero' else 'final_policy')
                      and score['completed_controls'] == item['actual_counts']['controls']
                      and score['windows']['control_cap'] == stage['controls']
                      and score['windows']['hold'] == stage['hold']
                      and score['windows']['release_tick_t0'] == stage['release_tick']
                      and score['windows']['drive'] == [175, stage['release_tick']]
                      and score['windows']['final'] == [stage['controls']-100, stage['controls']]
                      and (not score['record']['full_horizon'] or score['completed_controls'] == stage['controls']),
                      'C35 score profile/window/full horizon differs')
    prior = c35.pinned('gain_suite35_aggregate31', c35.ROOT/'scripts/read_b22_suite31.py')
    result = prior.aggregate(reports, pair_scorer)
    old = {(row['case_id'], row['seed'], row['mirror'], row['actor']): row for row in baseline['script_cases']}
    regression = []
    for row in result['script_cases']:
        key = row['case_id'], row['seed'], row['mirror'], row['actor']
        c.require(key in old, 'C35 historical task anchor missing')
        if old[key]['task_passed'] and not row['task_passed']:
            regression.append({'case_id': row['case_id'], 'seed': row['seed'], 'mirror': row['mirror'], 'actor': row['actor']})
    c.require(len(result['real_15mm_scores_separate']) == len(baseline['real_15mm_scores_separate']) == 2,
              'C35 true-step task anchors incomplete')
    for before, after in zip(baseline['real_15mm_scores_separate'], result['real_15mm_scores_separate']):
        c.require(before['actor'] == after['actor'], 'C35 step actor order differs')
        if before['task_qualified'] and not after['task_qualified']:
            regression.append({'case_id': 'real_15mm', 'actor': after['actor']})
    return {**result, 'no_task_regression': not regression, 'task_regressions': regression,
            'gain35': .5, 'adoption_decision': 'NOT_AUTOMATIC', 'default_controller_or_model_changed': False,
            'scope': 'fresh finite zero/half-gain seven-script and true-step pairs; original six contribution gate; historical31 only task anchors, not fresh paired arms'}


def compare(release_path):
    release = c35.validate_release(release_path)
    root = Path(release['plan']['output'])
    reports, identities = [], {}
    for stage in c35.stages():
        path = root/(stage['name']+'_readback.json')
        item = c.read_json(path)
        c.require(item['release_identity'] == c.identity(release_path)
                  and item['reader_identity'] == c.identity(Path(__file__)), 'C35 reader/release identity differs')
        for name, expected in item['artifacts'].items():
            c.require(not Path(name).is_absolute() and '..' not in Path(name).parts
                      and c.identity(root/stage['name']/name) == expected, 'C35 raw artifact changed: '+name)
        identities[path.name] = c.identity(path)
        reports.append(item)
    prior = c35.pinned('gain_suite35_scoring31', c35.ROOT/'scripts/read_b22_suite31.py')
    _, _, scorer = prior.helpers(release)
    baseline = c.read_json(c35.ROOT/'results/b22_suite31_20261018_01/comparison.json')
    return {'schema': 'b22-gain-suite35-comparison-v1', 'release_identity': c.identity(release_path),
            'execution_contract_id': c35.CONTRACT, 'stage_reports': identities,
            **aggregate(reports, scorer.score_pairs, baseline)}


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
