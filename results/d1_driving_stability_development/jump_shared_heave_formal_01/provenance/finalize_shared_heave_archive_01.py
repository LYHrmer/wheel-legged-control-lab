"""Seal new artifacts only; no imports of controller, simulator, or policy code."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

R = Path('/home/lyh/wheel-legged-control-lab')
W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
OUT = R / 'results/d1_driving_stability_development/jump_shared_heave_formal_01'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


assert not (OUT / 'publication_manifest.json').exists()
checks = json.loads((W / 'shared_heave_formal_preflight_01.json').read_text())['input_sha256']
assert len(checks) == 215
assert all(digest(Path(path)) == expected for path, expected in checks.items())
frozen = json.loads((R / 'results/d1_budget_study/protocol.json').read_text())['source_sha256']
assert len(frozen) == 77
assert all(digest(R / path) == expected for path, expected in frozen.items())

for filename in [
    'audit_shared_heave_eval_01.py',
    'shared_heave_evaluation_root_audit_01.json',
    'shared_heave_formal_independent_audit_01.json',
    'shared_heave_formal_audit_helper_adaptation_01.json',
    'plot_shared_heave_results_01.py',
]:
    # This new archive has not yet been sealed or committed. Refresh its draft copy.
    shutil.copy2(W / filename, OUT / 'provenance' / filename)
shutil.copytree(W / 'shared_heave_figures_01', OUT / 'figures')
shutil.copytree(W / 'next_terminal_handoff_plan_01', OUT / 'next_terminal_plan')

for dirname, source in [('training', W / 'shared_heave_formal_01'),
                        ('evaluation', W / 'shared_heave_evaluation_01')]:
    for path in source.rglob('*'):
        if not path.is_file() or path.name == 'training_trace.jsonl.gz':
            continue
        public = OUT / dirname / path.relative_to(source)
        assert public.is_file(), public
        assert digest(public) == digest(path), public

original = W / 'shared_heave_formal_01/training_trace.jsonl.gz'
rebuilt = W / 'reconstructed_heave_training_trace_01.jsonl.gz'
trace_sha = 'f4dce015bf995d32544b3acd46a856af734bb8827f1ce936b0622f39f9aaa266'
assert original.stat().st_size == rebuilt.stat().st_size == 82852678
assert digest(original) == digest(rebuilt) == trace_sha
write_new(OUT / 'provenance/training_trace_reconstruction_01.json', {
    'passed': True, 'original_bytes': 82852678,
    'original_and_reconstructed_sha256': trace_sha,
    'reconstructed_file': str(rebuilt),
    'method': 'actual public byte parts reconstructed by archived script; hashes rechecked while sealing',
    'new_physics': 0,
})

audit = json.loads((W / 'shared_heave_evaluation_root_audit_01.json').read_text())
assert audit['passed'] and audit['controls_audited'] == 6000 and audit['native_audited'] == 30000
assert all(audit['pairs'].values())
assert sum(c['passed'] for c in audit['cases'] if c['condition'] == 'final_policy_131072') == 1

running = []
targets = ('run_d1_jump', 'evaluate_d1_jump', 'probe_d1_heading', 'qualify_shared_heave')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\0')
        if args and Path(args[0].decode(errors='replace')).name.startswith('python'):
            names = [Path(arg.decode(errors='replace')).name for arg in args[1:] if arg]
            if any(name.startswith(targets) and name.endswith('.py') for name in names):
                running.append({'pid': int(proc.name), 'scripts': names})
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        continue
assert not running, running

report = {
    'schema': 'd1-shared-heave-final-archive-seal-v1',
    'checked_utc': datetime.now(timezone.utc).isoformat(),
    'passed': True, 'new_physics': 0, 'source215_unchanged': True,
    'frozen77_unchanged': True, 'raw_training_and_evaluation_copy_hashes_match': True,
    'trace_reconstruction_verified': True, 'running_physics_processes': running,
    'final_learned_passed': 1, 'final_learned_total': 5, 'only_hold_passed': True,
    'complete_objective': False, 'new_total_controls': 315424, 'new_total_native': 1577120,
    'next_contract_executed': False,
    'next_contract_sha256': digest(OUT / 'next_terminal_plan/next_contract.md'),
    'publication_status_at_seal': 'not_yet_committed_or_uploaded',
}
write_new(W / 'final_archive_seal_01.json', report)
write_new(OUT / 'provenance/final_archive_seal_01.json', report)
shutil.copy2(Path(__file__), OUT / 'provenance' / Path(__file__).name)

files = {
    str(path.relative_to(OUT)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
    for path in sorted(OUT.rglob('*')) if path.is_file()
}
write_new(OUT / 'publication_manifest.json', {
    'schema': 'published-jump-shared-heave-formal-archive-v1',
    'archive_status': 'valid_records_capability_failed', 'complete_objective': False,
    'files': files,
})
print(json.dumps({'passed': True, 'leaf_files': len(files),
                  'leaf_bytes': sum(v['bytes'] for v in files.values()),
                  'source_inputs': len(checks), 'frozen_inputs': len(frozen),
                  'running_physics': running, 'next_contract_executed': False}, ensure_ascii=False))
