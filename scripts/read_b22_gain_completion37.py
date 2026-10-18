"""C37 saved-only identity/numerical reader; retain C35 report identities unchanged."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('completion37_contract', Path(__file__).with_name('run_b22_gain_completion37.py'))
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)
c = r.c


def functions(prior, old_step):
    """Only versioned owner identity and the explicitly reused geometry path differ."""
    path = r.ROOT/'scripts/read_b22_suite31.py'
    tree = ast.parse(path.read_text())
    checks = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'script_sources'))
    replace = {'C31_B22_fixed_suite_v1': r.CONTRACT, 'scripts/run_b22_worker31.py': 'scripts/run_b22_worker37.py'}
    found = {key: 0 for key in replace}
    for node in ast.walk(checks):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replace:
            before = node.value
            node.value = replace[before]
            found[before] += 1
    c.require(all(v == 1 for v in found.values()), 'C37 metadata adaptation seam changed')
    canonical = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'canonical_script'))
    count = 0
    for node in ast.walk(canonical):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'step_run':
            c.require(ast.dump(node.value) == ast.dump(ast.parse("run.parent/'step_pair'", mode='eval').body), 'C37 geometry reference seam changed')
            node.value = ast.parse('Path('+repr(str(old_step))+')', mode='eval').body
            count += 1
    c.require(count == 1, 'C37 historical step reference seam absent')
    namespace = dict(prior.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[checks, canonical], type_ignores=[])), str(path), 'exec'), namespace)  # noqa: S102 -- pinned complete numerical checks, identity and geometry-path seams only
    return namespace['script_sources'], namespace['canonical_script']


def read_stage(release_path, name):
    release = r.validate_release(release_path)
    stage = next((s for s in r.stages() if s['name'] == name), None)
    c.require(stage is not None, 'C37 only missing four stages allowed')
    run = Path(release['plan']['output'])/name
    worker = c.load_pinned(r.ROOT/'scripts/run_b22_worker37.py', release['inputs'][str(r.ROOT/'scripts/run_b22_worker37.py')], 'completion37_derivation')
    c.require(c.read_json(run/'derived_sources37.json') == worker.derived_source()[1], 'C37 executed source derivation differs')
    prior = r.old35.pinned('completion37_numeric31', r.ROOT/'scripts/read_b22_suite31.py')
    gain = r.load_old('completion37_gain35', 'scripts/read_b22_gain_suite35.py')
    base, step, scorer = prior.helpers(release)
    source_check, canonical = functions(prior, r.OLD_ROOT/'step_pair')
    session, receipt, construction, count = source_check(run, stage, release, base)
    r.case_completion(stage, receipt)
    c.require(type(session.get('gain35')) is float and session['gain35'] == .5 and receipt['result'].get('gain35') == .5,
              'C37 actual session/result gain differs')
    rows = 0
    for row in base.saved_rows(run/'episode_0/controls.jsonl.gz'):
        c.require(row['control_index'] == rows, 'C37 gain ledger order differs')
        rows += gain.check_gain_row(row, stage['actor'])
    c.require(rows == count, 'C37 whole gain ledger missing')
    arrays, pairing = canonical(run, stage, session, receipt, construction, count, base, step)
    scored = scorer.score_case(arrays, expected_seed=stage['seed'], mirror=stage['mirror'])
    return {'schema': 'b22-gain-completion37-stage-readback-v1', 'status': 'COMPLETE', 'stage': name,
            'execution_contract_id': r.CONTRACT, 'gain35': .5, 'release_identity': c.identity(release_path),
            'reader_identity': c.identity(Path(__file__)), 'reader_model_calls': 0, 'reader_physics_calls': 0,
            'retry_permitted': False, 'gain_ledger_rows': rows,
            'actual_counts': {'controls': count, 'normal_native': 5*count, 'compiler_native': 2},
            'pairing_identity': pairing, 'cases': [scored], 'artifacts': {**prior.artifact_identities(run, base), 'derived_sources37.json': c.identity(run/'derived_sources37.json')},
            'reused_step_geometry': {'case_root': str(r.OLD_ROOT/'step_pair'),
                                     'closed_readback_identity': r.OLD_RECEIPTS['step_pair_readback.json']}}


def no_regression(result, baseline):
    before = {(x['case_id'], x['seed'], x['mirror'], x['actor']): x for x in baseline['script_cases']}
    changes = []
    for row in result['script_cases']:
        key = row['case_id'], row['seed'], row['mirror'], row['actor']
        c.require(key in before, 'C37 historical task anchor absent')
        if before[key]['task_passed'] and not row['task_passed']:
            changes.append(dict(zip(('case_id','seed','mirror','actor'), key)))
    c.require(len(result['real_15mm_scores_separate']) == len(baseline['real_15mm_scores_separate']) == 2, 'C37 step anchors differ')
    for before, after in zip(baseline['real_15mm_scores_separate'], result['real_15mm_scores_separate']):
        c.require(before['actor'] == after['actor'], 'C37 step actor order differs')
        if before['task_qualified'] and not after['task_qualified']:
            changes.append({'case_id': 'real_15mm', 'actor': after['actor']})
    return {'no_task_regression': not changes, 'task_regressions': changes}


def compare(release_path):
    release = r.validate_release(release_path)
    reused = r.load_reused35()
    root = Path(release['plan']['output'])
    reports = dict(reused['reports'])
    identities, total = {}, {key: 0 for key in r.REUSED_COUNTS}
    for stage in r.stages():
        path = root/(stage['name']+'_readback.json')
        item = c.read_json(path)
        row = r.counts(stage, item, c.identity(release_path), release['inputs'][str(r.ROOT/'scripts/read_b22_gain_completion37.py')])
        for name, expected in item['artifacts'].items():
            relative = Path(name)
            c.require(not relative.is_absolute() and '..' not in relative.parts
                      and c.identity(root/stage['name']/name) == expected, 'C37 new raw payload changed')
        reports[stage['name']], identities[path.name] = item, c.identity(path)
        total = {key: total[key]+row[key] for key in total}
    prior = r.old35.pinned('completion37_aggregate31', r.ROOT/'scripts/read_b22_suite31.py')
    _, _, scorer = prior.helpers(release)
    # Preserve original identities: the pure scorer does not require common version labels.
    result = prior.aggregate([reports[s['name']] for s in r.old35.stages()], scorer.score_pairs)
    baseline = c.read_json(r.ROOT/'results/b22_suite31_20261018_01/comparison.json')
    return {'schema': 'b22-gain-completion37-comparison-v1', 'execution_contract_id': r.CONTRACT,
            'release_identity': c.identity(release_path), 'new_stage_reports': identities,
            'reused_stage_reports': {name: expected for name, expected in r.OLD_RECEIPTS.items() if name.endswith('_readback.json')},
            'new_actual_counts': total, 'reused_validated_case_counts': dict(r.REUSED_COUNTS),
            'combined_validated_case_counts': {key: total[key]+r.REUSED_COUNTS[key] for key in total},
            'old35_outer_actual_counts': None, 'historical_total_resources_known': False,
            **result, **no_regression(result, baseline), 'gain35': .5, 'adoption_decision': 'NOT_AUTOMATIC',
            'default_controller_or_model_changed': False,
            'scope': 'mixed provenance: eleven closed original C35 sessions and four independent C37 sessions; analysis-case sum is not historical resource spending'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage')
    mode.add_argument('--compare', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    value = compare(args.release) if args.compare else read_stage(args.release, args.stage)
    c.write_json(args.report, value)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
