"""Recheck recorded bytes and independent kinematic limits before publishing summaries."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np

ROOT = Path('/home/lyh/wheel-legged-control-lab')
WORK = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912')
OUTPUT = ROOT/'results/d1_course_speed_gears'
OUTPUT.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, content):
    (OUTPUT/name).write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False)+'\n')


def copy(source, name):
    shutil.copyfile(source, OUTPUT/name)


artifacts = []
audit = []
for label, dirname in [('candidate', 'course_gears_01'), ('final', 'course_gears_02')]:
    directory = WORK/dirname
    copy(directory/'summary.json', label+'_summary.json')
    copy(directory/'protocol.json', label+'_protocol.json')
    summaries = json.loads((directory/'summary.json').read_text())
    for name, expected in summaries.items():
        trial = directory/name
        manifest = json.loads((trial/'manifest.json').read_text())
        checked = []
        for filename, digest in manifest.items():
            path = trial/filename
            assert sha(path) == digest
            artifacts.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
            checked.append(filename)
        protocol = json.loads((trial/'protocol.json').read_text())
        summary = json.loads((trial/'summary.json').read_text())
        assert summary['source_unchanged'] and summary['segments'] == 1
        assert sha(trial/'manifest.json') == expected['raw_manifest_sha256']
        with np.load(trial/'states.npz', allow_pickle=False) as states:
            qpos = states['qpos']
            times = states['segment_time_s']
            assert np.isfinite(qpos).all() and np.isfinite(states['qvel']).all()
            assert np.isfinite(states['applied_torque_nm']).all()
            assert np.allclose(np.diff(times), .01, atol=2e-12, rtol=0)
            assert np.all(states['segment_id'] == 0)
        quaternion = qpos[:, 3:7]
        assert np.allclose(np.linalg.norm(quaternion, axis=1), 1., atol=1e-10)
        w, x, y, z = quaternion.T
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
        actual_falls = (qpos[:, 2] < .22) | (np.abs(roll) > .85) | (np.abs(pitch) > .85)
        assert not np.any(actual_falls)
        release = 10. if label and name.startswith('flat_') else 14.
        mask = (times[:-1] >= release-2-1e-8) & (times[:-1] < release-1e-8)
        measured = float(np.diff(qpos[:, 0])[mask].sum()/np.diff(times)[mask].sum())
        assert abs(measured-expected['steady_last_2s_world_vx_mps']) < 1e-10
        current_mismatches = [path for path, digest in protocol['source_sha256'].items()
                              if sha(ROOT/path) != digest]
        if label == 'final':
            assert not current_mismatches, current_mismatches
        audit.append({'case': label+'/'+name, 'raw_files_verified': checked,
                      'states': len(qpos), 'transitions': len(qpos)-1,
                      'minimum_base_height_m': float(qpos[:, 2].min()),
                      'fall_threshold': 'z < .22m or abs(roll/pitch) > .85rad; same as D1StateEstimate',
                      'fallen_states': int(np.count_nonzero(actual_falls)),
                      'source_unchanged_during_trial': True,
                      'current_source_mismatches': current_mismatches,
                      'independent_world_vx_matches': True,
                      'compiled_model_sha256': protocol['compiled_model_sha256']})
    copy(directory/'flat_start_forward_gear1'/'protocol.json', label+'_runner_protocol.json')

manual = ROOT/'results/d1_side_step_manual_trial'
manual_manifest = json.loads((manual/'files.json').read_text())
assert all((manual/name).stat().st_size == item['bytes']
           and sha(manual/name) == item['sha256'] for name, item in manual_manifest.items())
save('artifact_audit.json', {'status': 'passed', 'cases': audit,
                           'manual_trial_small_files_verified': len(manual_manifest),
                           'physics_steps_performed_by_audit': 0,
                           'notes': ['Original probe no_fall check used stricter .70rad attitude bound but a stale safety-mode string; this independent audit applies exact .22m/.85rad D1StateEstimate thresholds to every recorded pose.',
                                     'No fall result remains unchanged in all 30 cases.']})
save('large_artifacts.json', {'local_only': True, 'files': artifacts})
copy(WORK/'probe_course_gears.py', 'probe_course_gears.py')
copy(WORK/'probe_gear_glfw.py', 'probe_gear_glfw.py')
copy(WORK/'archive_course_gears.py', 'archive_course_gears.py')
copy(WORK/'gear_glfw_02/report.json', 'glfw_report.json')
copy(WORK/'gear_glfw_02.x11/launcher_receipt.json', 'glfw_launcher_receipt.json')
copy(WORK/'gear_glfw_01.x11/launcher_receipt.json', 'glfw_sandbox_failure.json')
copy(WORK/'gear_glfw_01.x11/xvfb.log', 'glfw_sandbox_failure.log')
print(json.dumps({'cases': len(audit), 'raw_files_verified': len(artifacts),
                  'raw_bytes': sum(item['bytes'] for item in artifacts),
                  'output': str(OUTPUT)}, indent=2))
