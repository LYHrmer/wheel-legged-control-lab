"""File-only supplement for aggregate pair claims and cross-track provenance."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

W=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
C=W/'plane_turn_center_01'


def main():
    protocol=json.loads((C/'protocol.json').read_text())
    summary=json.loads((C/'summary.json').read_text())
    reports=[]
    for case,pair in zip(protocol['cases'],summary['pairs'],strict=True):
        name=case['name']
        expected=200 if case['command']['yaw_pulse'] else 800
        leaf=json.loads((C/f'pair_{name}.json').read_text())
        assert pair==leaf and pair['comparison']==name and pair['transitions']==expected
        assert pair['passed'] is True and pair['checks'] and all(value is True for value in pair['checks'].values())
        folder=C/name
        with np.load(folder/'states.npz') as z:
            positions=z['truth_positions_world_m'].copy()
        meta=json.loads((folder/'episode_metadata.json').read_text())
        yaw=meta['heading_reference_initial']['heading_rad']
        lateral=np.array([-np.sin(yaw),np.cos(yaw),0.])
        with gzip.open(folder/'trace.jsonl.gz','rt') as f:
            rows=[json.loads(line) for line in f]
        crosses=[]
        for k,row in enumerate(rows):
            actual=np.asarray(row['truth_position_world_m'])
            assert actual.dtype==positions.dtype and actual.tobytes()==positions[k+1].tobytes()
            cross=float((positions[k+1]-positions[0]) @ lateral)
            assert abs(cross-row['cross_track_after_m'])<=1e-12
            crosses.append(cross)
        leaf_summary=json.loads((folder/'summary.json').read_text())
        for key,value in [('cross_track_peak_m',max(abs(v) for v in crosses)),
                          ('cross_track_final_m',crosses[-1]),
                          ('cross_track_rmse_m',float(np.sqrt(np.mean(np.square(crosses)))) )]:
            assert abs(leaf_summary[key]-value)<=1e-12
        reports.append({'case':name,'pair_claims_match':True,'compared_execution_prefix':expected,
            'trace_positions_match_saved_states_bitwise':True,'cross_track_reconstructed':True})
    report={'passed':True,'new_physics_steps':0,'cases':reports,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (W/'turn_center_claims_positions_audit.json').open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
