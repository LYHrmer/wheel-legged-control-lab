"""Publish a fixed failed turn candidate without altering prior studies."""
import hashlib
import json
from pathlib import Path
import shutil

R=Path('/home/lyh/wheel-legged-control-lab')
W=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
D=R/'results/d1_driving_stability_development/heading_plane_turn_center_01'


def copy(a,b):
    if a.is_dir():
        shutil.copytree(a,b,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:
        b.parent.mkdir(parents=True,exist_ok=True)
        assert not b.exists()
        shutil.copyfile(a,b)


def main():
    assert json.loads((W/'turn_center_root_raw_score_audit.json').read_text())['passed']
    audit=json.loads((W/'turn_center_independent_audit.json').read_text())
    assert audit['passed']
    assert json.loads((W/'turn_center_claims_positions_audit.json').read_text())['passed']
    assert not json.loads((W/'plane_turn_center_01/summary.json').read_text())['candidate_passed_both_turn_cases']
    D.mkdir(exist_ok=False)
    for src,dst in [('plane_turn_center_01','episodes'),('plane_turn_diagnosis_01','diagnosis'),
                    ('opus_turn_center_01','opus_local_assembly_failure'),('opus_turn_center_02','opus'),
                    ('plane_turn_center_audit_01','independent_audit_source')]:
        copy(W/src,D/dst)
    for name in ('turn_center_execution_contract_20260920.md','turn_center_root_preflight.json',
                 'turn_center_shadow_preflight.json','turn_center_preflight_tests.xml',
                 'turn_center_preflight_final_tests.xml','audit_turn_center_preflight.py',
                 'audit_turn_center_preflight_attempt01.py','turn_center_preflight_attempt01_failure.json',
                 'audit_turn_center_raw_scores.py','turn_center_root_raw_score_audit.json',
                 'turn_center_independent_audit.json','audit_turn_center_claims_positions.py',
                 'turn_center_claims_positions_audit.json','publish_turn_center_01.py'):
        copy(W/name,D/'validation'/name)
    for name in ('d1_turn_center_compensation.py','probe_d1_heading_turn_center.py',
                 'd1_flat_plane_env.py','d1_probe_archive.py','d1_native_contact_diagnostics.py',
                 'd1_turn_contact_diagnostics.py','probe_d1_heading_flat_plane.py',
                 'probe_d1_heading_plane_stop_damping.py','probe_d1_heading_g1.py'):
        copy(R/'scripts'/name,D/'source/scripts'/name)
    copy(R/'tests/test_d1_turn_center_compensation.py',D/'source/tests/test_d1_turn_center_compensation.py')
    (D/'README.md').write_text('''# Fixed wheel-center turn compensation: original peak-heading gate still fails

The fixed coefficient-1 candidate improved the two original turn peaks but failed the original 5-degree limit. It is not adopted. The default controller, original raw references and gates are unchanged.

| Case | Plane-zero peak | Candidate peak | Original failed gates |
|---|---:|---:|---|
| Left turn | .234134 rad | .179587 rad (10.290 degrees) | heading_peak |
| Right turn | .234213 rad | .179692 rad (10.296 degrees) | heading_peak |
| Forward stop, no-op | original trajectory | byte-identical | original late speed, late path, raw velocity RMS |
| Reverse stop, no-op | original trajectory | byte-identical | original late speed, late path, raw velocity RMS |

These no-op stop cases do not contain the separately qualified stop-leg damper. Their failures remain explicit; the passing damper records are in `../heading_plane_stop_damping_01`.

An actual Claude Opus call returned the bounded controller core (`claude-opus-5`, provider receipt and original source preserved). Root integrated it and wrote the thin raw-callback environment and fixed runner. gpt-6-astra / ultra supplied the hypothesis, fixed contract and independent reviews. The initial local prompt-assembly attempt failed before any provider call; it is preserved. No new GUI code was generated.

During the original 50-interval pure-turn pulse only, the candidate adds measured leg-center longitudinal speed / .087 to each unclipped wheel target, using the horizontal heading frame. The coefficient is exactly one, with no latch, filter, gain sweep, mean removal, contact weighting, target shaping, PI reset or stop damping. The effective body-yaw request remains limited to .6 rad/s; the compensated wheel differential can imply a larger geometric yaw. Its peak was 1.756 rad/s, and the peak actual wheel target was 4.436 rad/s. Neither is measured body yaw. No target clip or torque protection activated.

Exactly four new cases completed: 3200 control intervals / 16000 native substeps. The four plane-zero baselines were reused, never rerun. Turn execution intervals 0..199 and physical states 0..200 are byte-identical; observations are compared through index199 because observation200 previews the changed target. Each no-op stop matches all 800 intervals and 801 states/observations, complete native records, torque/PI/action/raw/servo values and the original summary except its model label. The raw turning reference remains +/-.300 rad after the unchanged pulse.

Before physics, 26 tests prohibited every integration entry point. Root independently reproduced the original-turn diagnosis byte-for-byte (SHA256 25cfa109082efe578e3ff7ab8c3a9d10e3f667e9412f43b2f9656de558ad1450) and checked 100 independent saved-state target/PI calculations. Maximum target and wheel-request differences were 4.45e-16 rad/s and 1.78e-15 N m. The first saved-state preflight failed on a tuple/NumPy indexing mistake before candidate compute or integration; both source and failure are retained. This algebra check did not predict performance: the candidate's actual target excursions exceeded those calculated independently on old states.

Root separately reconstructed the original raw-reference scores and ran the full independent saved-state/native evidence audit (3,032,135 checks, zero new physics). A supplementary audit matched aggregate pairing claims to the per-case evidence, compared trace positions bitwise with saved physical states, and independently reconstructed cross-track scores. Physics representation, execution evidence and isolation checks are separate from the failed turn task gates. Native force logs and synchronized endpoint velocities retain their phases; no mixed-phase work or passivity claim is made. The audit reconstructs the wheel PI and all final torque protections, but does not independently derive the original leg PD/support request.

The center correction is a rolling approximation. Original data attribute roughly 76.66% of the wheel/body velocity gap to its center-translation term, 20.89% to leg-driven carrier rotation and 2.44% to slip in the pulse window. These are terms in a kinematic identity, not causal fractions or predicted improvement. Later lateral friction and slip remain relevant. This failed candidate is frozen; any next mechanism needs a separate evidence-based contract, rather than a gain/cap sweep or unvalidated combination with the stop damper.

The full straight-driving, obstacle and higher-speed objective remains incomplete. Manual driving, meaningful jump clearance/landing, steps/slopes/rubble and speed progression remain unverified. R is still simulation reset; physical self-righting is not implemented. The previously timed-out RL GUI still has no generated implementation. Frozen77, all older experiments and the three 65k trainings/full24 G1 are unchanged.
''')
    entries={str(p.relative_to(D)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(D.rglob('*')) if p.is_file()}
    with (D/'manifest.json').open('x') as f:
        json.dump(entries,f,indent=2,sort_keys=True);f.write('\n')
    assert all(hashlib.sha256((D/p).read_bytes()).hexdigest()==v['sha256'] for p,v in entries.items())
    print(json.dumps({'files':len(entries),'bytes':sum(v['bytes'] for v in entries.values()),'destination':str(D)}))


if __name__=='__main__':
    main()
