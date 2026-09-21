"""Read-only native-chain and independent action/window reduction for new evaluation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path,'rt') as f:
        return [json.loads(line) for line in f]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    w=args.work;folder=w/'shared_heave_evaluation_01'
    protocol=json.loads((w/'shared_heave_rl_plan_01/protocol.json').read_text())['evaluation']
    summary=json.loads((folder/'summary.json').read_text());assert summary['integrity_passed']
    inputs={};results=[];pairs={};controls=native_count=0
    torque_limit=np.tile([80.,80.,80.,12.],4)
    for case in protocol['cases']:
        first=None
        for condition in protocol['conditions']:
            name=case['name']+'__'+condition;d=folder/name
            native=rows(d/'native.jsonl.gz');trace=rows(d/'trace.jsonl.gz')
            record=json.loads((d/'records.json').read_text());states=np.load(d/'states.npz')
            for f in ('native.jsonl.gz','trace.jsonl.gz','records.json','states.npz','physical_score.json','episode_metadata.json'):
                inputs[str(d/f)]=sha(d/f)
            assert len(trace)==600 and len(native)==3000 and len(states['qpos'])==601
            assert len(states['qvel'])==len(states['observation'])==601
            initial={key:states[key][0].tobytes() for key in ('qpos','qvel','observation')}
            if condition=='zero':first=initial
            else:pairs[case['name']]=first==initial;assert pairs[case['name']]
            start=case['request_tick']
            for k,row in enumerate(trace):
                assert row['tick']==k
                a=np.asarray(row['action']);assert a.shape==(1,) and np.isfinite(a).all()
                active=start is not None and start<=k<start+120
                expected=np.r_[np.full(4,np.clip(a[0],-1,1)),np.zeros(4)] if active else np.zeros(8)
                np.testing.assert_array_equal(row['info']['applied_action'],expected)
                np.testing.assert_array_equal(row['controller']['clipped_action'],expected)
                assert row['info']['heave_action']['executed_tick']==k
                assert row['info']['heave_action']['active']==active
                assert row['raw_command']['forward_velocity_mps']==row['raw_command']['yaw_rate_rps']==0
            for i,row in enumerate(native):
                assert row['index']==i and row['returned'] and row['error'] is None
                assert row['contact_sample_tag']=='native_step_solved_cache_not_synchronized_endpoint'
                assert abs(row['actual_dt_s']-.002)<1e-10
                assert abs(row['end_time_s']-row['start_time_s']-.002)<1e-10
                assert not np.any(row['xfrc_applied']) and not np.any(row['qfrc_applied'])
                for key in ('qpos_before','qvel_before','qpos_returned','qvel_returned','ctrl_nm'):
                    assert np.isfinite(row[key]).all()
                assert np.all(np.abs(row['ctrl_nm'])<=torque_limit)
                np.testing.assert_array_equal(row['ctrl_nm'],trace[i//5]['controller']['torque_nm'])
                if i:
                    assert row['start_time_s']==native[i-1]['end_time_s']
                    np.testing.assert_array_equal(row['qpos_before'],native[i-1]['qpos_returned'])
                    np.testing.assert_array_equal(row['qvel_before'],native[i-1]['qvel_returned'])
                if i%5==0:
                    for key in ('qpos','qvel'):np.testing.assert_array_equal(row[key+'_before'],states[key][i//5])
                if i%5==4:
                    for key in ('qpos','qvel'):np.testing.assert_array_equal(row[key+'_returned'],states[key][i//5+1])
            intervals=record['intervals'];assert len(intervals)==3000
            lo=1000 if start is None else start*5;hi=3000 if start is None else (start+120)*5
            gaps=np.array([r['endpoint_min_gap_m'] for r in intervals]);margins=np.array([r['contact_margin_m'] for r in intervals])
            valid=np.array([all(c==0 for c in r['active_wheel_contacts']) and all(f==0 for f in r['wheel_normal_load_n']) and r['endpoint_min_gap_m']>r['contact_margin_m'] for r in intervals])
            runs=[];j=lo
            while j<hi:
                if not valid[j]:j+=1;continue
                first_run=j
                while j<hi and valid[j]:j+=1
                runs.append({'first_native':first_run,'samples':j-first_run,'duration_ms':2*(j-first_run),
                    'start_com_vz_mps':intervals[first_run]['start_com_vz_mps'],
                    'net_gap_peak_m':float(np.max(gaps[first_run:j]-margins[first_run:j]))})
            old_equivalence=None
            if condition=='zero':
                old=w/'rl_jump_evaluation_01'/name
                original=np.load(old/'states.npz')
                old_trace=rows(old/'trace.jsonl.gz')
                initial_match=all(states[k][0].tobytes()==original[k][0].tobytes() for k in ('qpos','qvel'))
                command_match=all(a['raw_command']==b['raw_command'] for a,b in zip(trace,old_trace,strict=True))
                specs=[json.loads((path/'episode_metadata.json').read_text())['episode_metadata']['jump_episode'] for path in (d,old)]
                friction_match=all(s['friction_scale']==case['friction_scale'] for s in specs)
                comparable=initial_match and command_match and friction_match
                old_equivalence={'compared':comparable,'initial_match':initial_match,
                    'raw_command_match':command_match,'friction_match':friction_match}
                if comparable:old_equivalence['arrays']={key:states[key].tobytes()==original[key].tobytes() for key in states.files}
                inputs[str(old/'states.npz')]=sha(old/'states.npz')
                inputs[str(old/'episode_metadata.json')]=sha(old/'episode_metadata.json')
            score=json.loads((d/'physical_score.json').read_text())
            results.append({'case':case['name'],'condition':condition,'passed':score['passed'],
                'window_max_net_gap_mm':1000*float(np.maximum(0.,gaps[lo:hi]-margins[lo:hi]).max()),
                'unloaded_runs':runs,'maximum_planar_displacement_m':max(r['planar_displacement_m'] for r in record['endpoints']),
                'old_zero_physical_equivalence':old_equivalence})
            controls+=600;native_count+=3000
    assert controls==6000 and native_count==30000
    report={'passed':True,'new_physics':0,'controls_audited':controls,'native_audited':native_count,
        'pairs':pairs,'cases':results,'input_sha256':inputs,
        'scope':'Full raw native clock/state chain/held torque/zero external force/original rated limits; independent scalar gate mapping and request-window unloaded-run reduction from recorded geometry/load. No reconstructed geometry or force solve.'}
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','controls_audited','native_audited','cases')}))


if __name__=='__main__':main()
