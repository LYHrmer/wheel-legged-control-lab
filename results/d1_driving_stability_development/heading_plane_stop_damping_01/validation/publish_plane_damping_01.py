"""Publish a new candidate study; never modify a previous study."""
from pathlib import Path
import hashlib
import json
import shutil

R=Path('/home/lyh/wheel-legged-control-lab')
W=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
D=R/'results/d1_driving_stability_development/heading_plane_stop_damping_01'


def copy(a,b):
    if a.is_dir():
        shutil.copytree(a,b,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:
        b.parent.mkdir(parents=True,exist_ok=True)
        assert not b.exists()
        shutil.copyfile(a,b)


def main():
    audit=json.loads((W/'plane_stop_damping_independent_audit.json').read_text())
    assert audit['passed'] and audit['all_four_original_case_gates_passed']
    D.mkdir(exist_ok=False)
    copy(W/'plane_stop_damping_01',D/'episodes')
    copy(W/'plane_stop_diagnosis_01',D/'diagnosis')
    copy(W/'opus_stop_damper_01',D/'opus_controller')
    for name in ('plane_damping_root_preflight.json','plane_damping_execution_contract_20260920.md',
                 'plane_damping_preflight_tests.xml','plane_damping_preflight_final_tests.xml',
                 'plane_stop_damping_independent_audit.json','audit_plane_damping_records.py',
                 'audit_flat_plane_records.py','publish_plane_damping_01.py'):
        copy(W/name,D/'validation'/name)
    for name in ('d1_stop_leg_damping.py','probe_d1_heading_stop_damping.py',
                 'probe_d1_heading_plane_stop_damping.py','d1_flat_plane_env.py',
                 'd1_probe_archive.py','d1_native_contact_diagnostics.py','d1_turn_contact_diagnostics.py',
                 'probe_d1_heading_flat_plane.py','probe_d1_heading_g1.py'):
        copy(R/'scripts'/name,D/'source/scripts'/name)
    copy(R/'tests/test_d1_heading_plane_stop_damping.py',D/'source/tests/test_d1_heading_plane_stop_damping.py')
    (D/'README.md').write_text('''# Fixed stop leg damping on the native plane

All four candidate cases pass the original raw-command G1 gates. This is a finite zero-residual development result, not full driving or real-robot qualification. The default controller is unchanged.

The sole controller change is the fixed longitudinal leg damper, b=126.4374005337902 N s/m, implemented by an actual Claude Opus call and integrated by root following gpt-6-astra / ultra planning. It activates only on an executed nonzero-to-zero forward-command transition. There is no release ramp, wheel PI reset, yaw cap increase, new training or gain sweep. Root composed the separately reviewed native-plane environment and stop adapter before first reset.

| Original raw stop metric | Forward baseline | Forward candidate | Reverse baseline | Reverse candidate | Original limit |
|---|---:|---:|---:|---:|---:|
| Full raw velocity RMS, m/s | .059043 | .044766 | .058776 | .044838 | .05 |
| Late peak speed, m/s | .084356 | .005823 | .084236 | .006336 | .03 |
| Late cumulative planar path, m | .062993 | .002591 | .062819 | .002736 | .05 |

The remaining original stop gates also pass. No torque, position or speed protection activated. The maximum added leg torque is 7.640 / 7.622 N m. Recorded pre-protection and same-state protected incremental powers are nonpositive at the control sample; this is not an integrated-work or discrete closed-loop passivity proof.

Only four candidates were executed: two stops of 800 control intervals and two yaw impulses of 1200, totaling 4000 intervals / 20000 native substeps. Their four baselines were read from `../heading_flat_plane_01/episodes`, not rerun. Stop execution intervals 0..399 and states/observations 0..400 are byte-identical to the baseline. Both no-stop impulse trajectories are byte-identical for all 1200 intervals and 1201 states/observations, including raw/servo commands, torques, PI memory and full native entries. Actual external yaw impulses are +.1 and -.1 N m s. No same-case signed-zero exception is used.

Root's independent audit reconstructed the fixed damper from 4000 saved poses without integration, with maximum torque-equation discrepancy 1.33e-15 N m. It independently recomputed raw stopping metrics, compared complete native records, checked timing, applied control, force/wrench decompositions, torque protection and T/T+1 archives. Plane contact normals remained vertical. All 77 frozen source hashes and baseline files are unchanged.

The original fixed-gain derivation remains in `../heading_stop_kinematics_01`. The new plane-specific diagnosis was independently reproduced byte-for-byte by root (SHA256 0ccbdfdc4cc7234ae1395585785b28b343667d0b52da614f7959bb916d30edc7); its helper/report are preserved here. Ten additional nonintegrating tests cover the plane/damper composition, old checkpoint rejection, invalid actions and close failure. The preflight receipt explains the import-order-only difference between the live plane test file and its preserved original.

The previous .5 m/s² release reference failed and remains rejected. Original plane turn cases still fail peak heading error; turning, jumping, step traversal, higher speed, manual driving and physical self-righting are not established by this result. R remains simulator reset. The timed-out RL GUI generation produced no code. The three old 65k trainings and full old 24-case G1 suite were not repeated.
''')
    entries={str(p.relative_to(D)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(D.rglob('*')) if p.is_file()}
    with (D/'manifest.json').open('x') as f:
        json.dump(entries,f,indent=2,sort_keys=True);f.write('\n')
    assert all(hashlib.sha256((D/p).read_bytes()).hexdigest()==v['sha256'] for p,v in entries.items())
    print(json.dumps({'files':len(entries),'bytes':sum(v['bytes'] for v in entries.values()),'destination':str(D)}))


if __name__=='__main__':
    main()
