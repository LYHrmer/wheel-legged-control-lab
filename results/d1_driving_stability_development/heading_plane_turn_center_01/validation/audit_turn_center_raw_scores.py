"""Root file-only reconstruction of raw reference scores and fixed experiment counts."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

R=Path('/home/lyh/wheel-legged-control-lab')
W=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
C=W/'plane_turn_center_01'


def main():
    protocol=json.loads((C/'protocol.json').read_text())
    aggregate=json.loads((C/'summary.json').read_text())
    assert aggregate['comparison_valid'] and aggregate['actual_new_control_transitions']==3200
    assert aggregate['actual_new_physics_substeps']==16000 and aggregate['no_op_stop_failures_preserved']
    result=[]
    for case in protocol['cases']:
        folder=C/case['name']
        summary=json.loads((folder/'candidate_summary.json').read_text())
        with gzip.open(folder/'trace.jsonl.gz','rt') as f:
            rows=[json.loads(line) for line in f]
        assert len(rows)==800 and rows[-1]['truncated'] and not rows[-1]['terminated']
        with np.load(folder/'states.npz') as z:
            positions=z['truth_positions_world_m'].copy()
            assert z['qpos'].shape[0]==z['qvel'].shape[0]==z['observations'].shape[0]==801
        psi=rows[0]['heading_task']['reference_heading_before']
        errors=[]
        for k,row in enumerate(rows):
            c=case['command']
            raw=c['forward_target_mps']*np.clip((k-c['settle_ticks'])/c['ramp_ticks'],0.,1.)
            if c['stop_tick'] is not None and k>=c['stop_tick']:
                raw=0.
            pulse=c['yaw_pulse']
            yaw=pulse['user_yaw_rate_rps'] if pulse and pulse['start_tick']<=k<pulse['end_tick_exclusive'] else 0.
            assert raw==row['heading_task']['user_command_before']['forward_velocity_mps']
            assert yaw==row['heading_task']['user_command_before']['yaw_rate_rps']
            psi+=yaw*.01
            error=np.arctan2(np.sin(row['heading_task']['truth_heading_after']-psi),np.cos(row['heading_task']['truth_heading_after']-psi))
            assert abs(error-row['heading_error_rad'])<1e-10
            errors.append(error)
        peak=float(np.max(np.abs(errors)))
        assert abs(peak-summary['heading_peak_rad'])<1e-10
        if case['command']['yaw_pulse']:
            assert summary['gates']['failed']==['heading_peak'] and peak>np.deg2rad(5)
            assert max(abs(rows[k]['heading_error_rad']) for k in range(449,799))<=np.deg2rad(3)
            assert max(abs(rows[k]['body_forward_mps']) for k in range(449,799))<=.03
            assert np.max(np.linalg.norm(positions[:,:2]-positions[0,:2],axis=1))<=.1
        else:
            old=W/'flat_plane_02'/case['name']
            original=json.loads((old/'summary.json').read_text())
            assert all(summary[k]==v for k,v in original.items() if k!='model')
            with np.load(old/'states.npz') as a,np.load(folder/'states.npz') as b:
                assert all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a.files)
        result.append({'case':case['name'],'original_failed_gates':summary['gates']['failed'],
            'independent_raw_reference_peak_rad':peak,'heading_peak_degrees':float(np.rad2deg(peak)),
            'preserved_original_reference_delta_rad':psi-rows[0]['heading_task']['reference_heading_before']})
    frozen=json.loads((R/'results/d1_budget_study/protocol.json').read_text())['source_sha256']
    assert len(frozen)==77 and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==v for p,v in frozen.items())
    report={'passed':True,'new_physics_steps':0,'original_scores_preserved':True,'candidate_passed_both_turns':False,
        'actual_control_transitions':3200,'actual_native_substeps':16000,'frozen77_unchanged':True,'cases':result}
    with (W/'turn_center_root_raw_score_audit.json').open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
