"""Publish the one fixed failed raw-turn damper without editing old experiments."""
import hashlib
import json
from pathlib import Path
import shutil

R = Path('/home/lyh/wheel-legged-control-lab')
W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
D = R/'results/d1_driving_stability_development/heading_plane_turn_damping_01'


def copy(source, target):
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)


def main():
    audit = json.loads((W/'turn_damper_independent_audit.json').read_text())
    assert audit['passed'] and not audit['candidate_passed_both_turn_cases']
    D.mkdir(exist_ok=False)
    for source, target in [('plane_turn_damping_01', 'episodes'), ('opus_turn_damper_01', 'opus'),
                           ('turn_next_mechanics_plan_01', 'mechanics'), ('turn_damper_review_01', 'review')]:
        copy(W/source, D/target)
    for name in ['turn_damper_execution_contract_20260920.md', 'turn_damper_root_preflight.json',
                 'turn_damper_preflight_tests.xml', 'turn_damper_shadow_preflight.json',
                 'audit_turn_damper_preflight.py', 'audit_turn_damper_records.py',
                 'turn_damper_independent_audit.json', 'turn_damper_outcome_diagnosis.json',
                 'publish_turn_damping_01.py']:
        copy(W/name, D/'validation'/name)
    copy(W/'plane_turn_center_audit_01/audit_candidate.py', D/'validation/shared_audit_candidate.py')
    for name in ['d1_turn_leg_damping.py', 'probe_d1_heading_turn_damping.py', 'd1_stop_leg_damping.py',
                 'd1_flat_plane_env.py', 'd1_probe_archive.py', 'd1_native_contact_diagnostics.py',
                 'd1_turn_contact_diagnostics.py', 'probe_d1_heading_flat_plane.py',
                 'probe_d1_heading_plane_stop_damping.py', 'probe_d1_heading_g1.py']:
        copy(R/'scripts'/name, D/'source/scripts'/name)
    copy(R/'tests/test_d1_turn_leg_damping.py', D/'source/tests/test_d1_turn_leg_damping.py')
    (D/'README.md').write_text('''# Fixed turn leg damper: finite improvement, original gate failure

The fixed raw-turn-only leg damper failed the original 5-degree heading peak gate in both directions. It is not adopted. Default control remains unchanged.

| Case | Original plane peak | Damper peak | Original result |
|---|---:|---:|---|
| Left stationary turn | .234134 rad | .216487 rad (12.404 deg) | heading_peak fails |
| Right stationary turn | .234213 rad | .216581 rad (12.409 deg) | heading_peak fails |
| Forward stop | original trajectory | full bitwise no-op | original three failed gates retained |
| Reverse stop | original trajectory | full bitwise no-op | original three failed gates retained |

This is independent of the separately passing post-stop damper and the failed wheel-center target correction. Neither is combined. Four new candidate episodes completed once: 3200 control intervals / 16000 native substeps. Four retained `flat_plane_02` baselines were read, not rerun. All original raw commands/references and gates were used, with zero8 residuals.

An actual Claude Opus call (`claude-opus-5`, $0.3976225, original request/response/receipt retained) wrote the bounded core. Root read and integrated the source unchanged, authored the runner/tests, and ran the physical comparison. gpt-6-astra / ultra supplied the fixed 209-pose method and independent source review. The core adds `-126.4374005337902 J_bodyx.T (J_bodyx qdot_leg)` only when raw forward is exactly zero and raw yaw nonzero. There is no latch. The original wheel targets, PI/antiwindup, yaw cap and all torque protections are retained. The coefficient comes from the previously recorded nominal longitudinal mode, not a yaw fit.

Before physics, 26 nonintegrating tests and 104 independent saved-state controller calls passed. The root-executed 209-pose mechanical report verified finite requests, original wheel-spin damping and contact kinematics. The sampled free-inertia spectrum and nominal mode projection were qualified as approximations, not stability proofs. An all-point no-slip condition was not claimed: representative rolling solutions leave finite contact-patch residuals and require different wheel speeds from the original controller.

Root then reconstructed all 3204 saved states, added torques, original wheel PI, protections, native contact records, synchronized endpoint velocities and original raw scores. The independent audit passed 2,891,747 checks with zero new physics/controller calls. It preserves the distinction between valid evidence and failed task gates; the original leg PD/support request is archived rather than independently reimplemented. Turn execution0..199 and states AND observations0..200 match the baseline bitwise, including complete native entries. Stop traces match all800/801 and every original summary field except model.

The left-turn pulse leg-center velocity RMS fell from .05680 to .04441 m/s (pre-step samples), while actual body heading increment rose from .06587 to .08352 rad. This supports an effect on leg motion, but it did not meet the task. Native moments resolved in fixed world x/y changed from +6.386/−5.500 to +8.511/−7.635 Nm; net pulse mean barely changed (.8863→.8758 Nm), and the last25 net moment became more opposing. These are force components in fixed world axes, not a causal or energy allocation. No native force is multiplied by endpoint velocity as though synchronous.

Actual peak added leg torque was3.981 Nm, with no torque protection. The two no-op stops do not contain post-stop damping. Four initial native substeps without active wheel contact occur in each episode's unchanged settling prefix; they are not evidence of a new turning loss of support. Individual saved contact friction utilization can reach1; aggregate normal-load ratios must not be used to claim every contact remains unsaturated.

The independent diagnostic also confirms the original inner .6 yaw cap was occupied for all50 pulse intervals, masking the existing body-rate feedback. This is a distinct unresolved control-authority hypothesis, not proof that contact geometry is irrelevant. This damper and its coefficient are frozen; no tuning sweep, post-pulse latch or combined controller was tried. Any subsequent mechanism requires its own fixed contract and original scoring.

The full straight-driving → obstacles → speed objective remains incomplete. Turning robustness, stop/turn composition, new heading GUI, actual manual acceptance, jump clearance/landing, steps/slopes/rubble, speed and physical self-righting remain unfinished. The earlier RL GUI generation timed out without code; R remains simulator reset. No three65k training or full24G1 rerun; frozen77 and older experiments remain unchanged.
''')
    entries = {str(p.relative_to(D)): {'bytes': p.stat().st_size,
               'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(D.rglob('*')) if p.is_file()}
    with (D/'manifest.json').open('x') as f:
        json.dump(entries, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps({'files': len(entries), 'bytes': sum(v['bytes'] for v in entries.values()), 'destination': str(D)}))


if __name__ == '__main__':
    main()
