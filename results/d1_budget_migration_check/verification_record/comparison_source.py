"""Compare relocated recomputation with exact path-only normalization."""
from datetime import datetime, timezone
import hashlib
import json

from archive_course_evidence import ROOT, WORK, digest, inventory, write_json


def compare(before, after, allowed, path=()):
    differences = []
    if type(before) is not type(after):
        raise AssertionError(f'type mismatch at {path}')
    if isinstance(before, dict):
        assert before.keys() == after.keys(), f'keys differ at {path}'
        for key in before:
            differences.extend(compare(before[key], after[key], allowed, path + (key,)))
    elif isinstance(before, list):
        assert len(before) == len(after), f'length differs at {path}'
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            differences.extend(compare(left, right, allowed, path + (index,)))
    elif before != after:
        assert path in allowed, f'non-allowed difference at {path}: {before!r} / {after!r}'
        assert (before, after) == allowed[path], f'path substitution not exact: {path}'
        differences.append({'field': list(path), 'before': before, 'after': after})
    return differences


def main():
    target = WORK / 'migration_check'
    study = target / 'relocated_study'
    reference = target / 'reference'
    original = ROOT / 'results/d1_budget_study'
    analysis_old = json.loads((reference / 'd1_budget_study_analysis/analysis.json').read_text())
    analysis_new = json.loads((target / 'recomputed_analysis/analysis.json').read_text())
    weights_old = json.loads((reference / 'd1_budget_study_weights/report.json').read_text())
    weights_new = json.loads((target / 'recomputed_weights/report.json').read_text())
    allowed_analysis = {('study',): (str(original), str(study))}
    for name in analysis_old['training']:
        allowed_analysis[('training', name, 'ppo_math', 'directory')] = (
            str(original / name / 'updates'), str(study / name / 'updates'))
    allowed_weights = {('study',): (str(original), str(study))}
    analysis_differences = compare(analysis_old, analysis_new, allowed_analysis)
    weights_differences = compare(weights_old, weights_new, allowed_weights)
    assert len(analysis_differences) == len(allowed_analysis) == 7
    assert len(weights_differences) == 1
    assert analysis_new['audited_train_calls'] == 3072
    assert analysis_new['evaluation_summary']['cases'] == 200
    assert analysis_new['evaluation_summary']['completed'] == 198
    assert analysis_new['evaluation_summary']['failed'] == 2
    assert analysis_new['evaluation_summary']['quality_passes'] == 192
    assert weights_new['total_zip_files_checked'] == 30
    assert weights_new['checkpoints_checked'] == 24
    assert weights_new['root_finals_checked'] == 6
    verified_output_files = []
    for directory in [target / 'recomputed_analysis', target / 'recomputed_weights']:
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, value in manifest['sha256'].items():
            assert digest(directory / name)['sha256'] == value
            verified_output_files.append(str((directory / name).relative_to(target)))
    before = json.loads((target / 'input_sha256.json').read_text())['files']
    assert inventory(original) == before == inventory(study)
    guards = {name: json.loads((target / f'{name}_read_guard.json').read_text()) for name in ['analysis', 'weights']}
    for record in guards.values():
        assert record['status'] == 'passed' and record['blocked_accesses'] == []
        assert record['study_access_counts']['open'] > 0
    normalized = json.loads(json.dumps(analysis_new))
    for path, (old_value, new_value) in allowed_analysis.items():
        node = normalized
        for key in path[:-1]:
            node = node[key]
        assert node[path[-1]] == new_value
        node[path[-1]] = old_value
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert canonical(analysis_old) == canonical(normalized)
    report = {
        'status': 'migration_recomputation_passed',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_comparison': {'numeric_and_nonpath_fields_exactly_equal': True,
            'allowed_path_changes': analysis_differences,
            'normalized_canonical_sha256': hashlib.sha256(canonical(normalized)).hexdigest(),
            'original_report_sha256': digest(reference / 'd1_budget_study_analysis/analysis.json')['sha256'],
            'migrated_report_sha256': digest(target / 'recomputed_analysis/analysis.json')['sha256']},
        'weight_comparison': {'all_tensor_and_file_hash_chains_exactly_equal': True,
            'allowed_path_changes': weights_differences,
            'original_report_sha256': digest(reference / 'd1_budget_study_weights/report.json')['sha256'],
            'migrated_report_sha256': digest(target / 'recomputed_weights/report.json')['sha256']},
        'counts': {'commands': 56, 'source_files': 77, 'train_calls': 3072, 'evaluation_cases': 200,
            'completed': 198, 'failed_map_boundary': 2, 'quality_passes': 192, 'checkpoint_zips': 24, 'root_final_zips': 6},
        'study_input_files': len(before), 'study_input_bytes': sum(v['bytes'] for v in before.values()),
        'original_and_migrated_inputs_unchanged': True,
        'input_sha256_manifest_sha256': digest(target / 'input_sha256.json')['sha256'],
        'reader_guards': guards, 'verified_output_files': verified_output_files,
        'limitations': ['Same machine and installed dependencies; no dependency-installation or cross-platform test.',
            'This run used physical copytree-equivalent file copies, not a GitHub download or result bundle.',
            'Archived analyzers were executed unchanged; no training, simulation evaluation, optimizer replay or remote write.'],
    }
    write_json(target / 'report.json', report)
    print(json.dumps({'status': report['status'], 'analysis_path_fields_changed': 7, 'weights_path_fields_changed': 1,
                      'all_other_fields_exactly_equal': True, 'input_files': len(before),
                      'report_sha256': digest(target / 'report.json')['sha256']}))


if __name__ == '__main__':
    main()
