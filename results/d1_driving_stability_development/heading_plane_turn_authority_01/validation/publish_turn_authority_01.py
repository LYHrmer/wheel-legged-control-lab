"""Publish fixed restored-feedback evidence without modifying earlier records."""
import hashlib
import json
from pathlib import Path
import shutil

R = Path('/home/lyh/wheel-legged-control-lab')
W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
D = R/'results/d1_driving_stability_development/heading_plane_turn_authority_01'


def copy(source, target):
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)


def main():
    audit = json.loads((W/'turn_authority_independent_audit.json').read_text())
    assert audit['passed'] and audit['candidate_passed_both_turn_cases']
    D.mkdir(exist_ok=False)
    for source, target in [('plane_turn_authority_01', 'episodes'), ('opus_turn_authority_01', 'opus'),
                           ('turn_actuation_plan_01', 'mechanics'), ('turn_authority_review_01', 'review')]:
        copy(W/source, D/target)
    for name in ['turn_authority_execution_contract_20260920.md', 'turn_authority_root_preflight.json',
                 'turn_authority_preflight_tests.xml', 'turn_authority_shadow_preflight.json',
                 'audit_turn_authority_preflight.py', 'audit_turn_authority_records.py',
                 'turn_authority_independent_audit.json', 'turn_authority_prelaunch_path_error.json',
                 'publish_turn_authority_01.py']:
        copy(W/name, D/'validation'/name)
    copy(W/'plane_turn_center_audit_01/audit_candidate.py', D/'validation/shared_audit_candidate.py')
    for name in ['d1_turn_yaw_authority.py', 'probe_d1_heading_turn_yaw_authority.py',
                 'd1_turn_leg_damping.py', 'd1_stop_leg_damping.py', 'd1_flat_plane_env.py',
                 'd1_probe_archive.py', 'd1_native_contact_diagnostics.py', 'd1_turn_contact_diagnostics.py',
                 'probe_d1_heading_flat_plane.py', 'probe_d1_heading_plane_stop_damping.py', 'probe_d1_heading_g1.py']:
        copy(R/'scripts'/name, D/'source/scripts'/name)
    copy(R/'tests/test_d1_turn_yaw_authority.py', D/'source/tests/test_d1_turn_yaw_authority.py')
    (D/'README.md').write_text('''# Restored original yaw feedback passes the two original turn cases

The independent native-plane, oracle-state, zero8 candidate passed every original G1 gate for both stationary turns. This is a finite development result. Stop/turn composition and robust driving are not qualified; default entry points remain unchanged.

| Case | Original plane heading peak | Candidate peak | Original gate result |
|---|---:|---:|---|
| Left stationary turn | .234134 rad | .063933 rad (3.663 deg) | all pass |
| Right stationary turn | .234213 rad | .063963 rad (3.665 deg) | all pass |
| Forward stop | original trajectory | complete bitwise no-op | three original failures retained |
| Reverse stop | original trajectory | complete bitwise no-op | three original failures retained |

One new four-case batch completed3200 control intervals /16000 native substeps. Corresponding `flat_plane_02` baselines were read, never rerun. Original raw commands, heading reference and all original gates, including the5-degree peak limit, were retained. A first launch used a nonexistent protocol path and failed during input hashing, before output creation, environment construction or physics; the zero-physics failure and corrected original path are recorded.

The sole intervention is to restore the original `servo_yaw + 4*(servo_yaw-body_yaw_rate)` during raw forward==0 and raw yaw!=0, without the inner ±.6 clamp. There is no new numeric cap, dynamic headroom limiter, leg damper, wheel-center compensation or latch. The original final wheel target ±30rad/s, wheel PI2.2/3, integral±4Nm, antiwindup, rated/outward protection, original leg PD/support and outer heading law remain. Original frozen77 files are unchanged.

The original baseline's inner .6 clamp masked the existing body-rate feedback during all50 pulse intervals. The restored formula changes the feedback response, but successful finite turns do not establish that this was the sole failure mechanism or remove contact/geometry limitations. The old1.0-cap experiment failed on the old hfield plant and kept clipping; it is preserved separately.

Actual wheel targets peaked at11.85298rad/s; no target clipping occurred. Unlimited wheel request peaked at23.09777Nm, with actual12Nm protection. Each turn had8 protected joint-intervals. Actual torque saturation is distinguished from target clipping and from removed inner-cap occupancy. Native mean pulse yaw moments were +3.17557/−3.17612Nm. Individual contact friction utilization reaches1; aggregate loading is not a guarantee against local sliding.

gpt-6-astra / ultra supplied the fixed method,208-pose qualification helper and independent source review. Root ran the helper and104 independent saved-state core calls, whose eight compared quantities had zero error; these restart original PI memory and are not a new trajectory.29 nonintegrating tests passed, followed only by an import-order correction and repository Ruff. Sampled rigid-leg damping proxies and headroom intervals were documented as diagnostics, not loaded closed-loop stability proofs.

An actual bounded Claude Opus call (provider `claude-opus-5`, $0.34730375) wrote the core. Original request/response/source/receipt and root integration diff are preserved. Root corrected documentation, diagnostic tolerance and export order; the control formula was unchanged. Root authored the environment/runner/tests, executed physics and reconstructed all3204 saved states, original/restored target formula, PI, protections, native and synchronized endpoint evidence, and original raw scores. The independent audit passed3,046,809 checks with no new integration, contact solve or controller calls. The archived base leg PD/support request is not independently reimplemented.

Before the turn pulse, execution0..199, physical states0..200, observations0..199 and native entries0..999 are bitwise equal to baseline. Observation200 is excluded because its preview target changes. Both no-op stops preserve every original800/801 record and all original summary fields except model, including all three failed gates. They do not contain the independently passing post-stop damper. Four initial native substeps without wheel support lie within the unchanged settling prefix in each episode.

The next task is a separately specified composition of the two qualified mechanisms, with transition cases and original scoring. This package does not establish GUI/manual driving, reliable jumping, obstacles, speed or physical self-righting. The earlier new-GUI generation timed out with no code; R remains simulation reset. No training or full24G1 replay was performed; older experiments were not edited.
''')
    entries = {str(p.relative_to(D)): {'bytes': p.stat().st_size,
               'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(D.rglob('*')) if p.is_file()}
    with (D/'manifest.json').open('x') as f:
        json.dump(entries, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps({'files': len(entries), 'bytes': sum(v['bytes'] for v in entries.values()), 'destination': str(D)}))


if __name__ == '__main__':
    main()
