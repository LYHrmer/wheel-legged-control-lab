"""Independent compact-log accounting; no simulator or policy imports."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    receipt=json.loads((args.folder/'receipt.json').read_text())
    updates=[json.loads(x) for x in (args.folder/'updates.jsonl').read_text().splitlines()]
    with gzip.open(args.folder/'episodes.jsonl.gz','rt') as f: episodes=[json.loads(x) for x in f]
    expected=receipt['successful_training_transitions'];by_episode={};count=0;native=0
    max_dt_error=0.;max_reward_error=0.;raw_zero=True;finite=True;stage_max={};bonus_sums={}
    with gzip.open(args.folder/'training_trace.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);count+=1;native+=row['native_calls'];e=row['episode_index']
            assert row['total_transition']==count
            info=by_episode.setdefault(e,{'count':0,'return':0.,'max_progress':0.,'flight_seen':False})
            assert row['executed_tick']==info['count'];info['count']+=1;info['return']+=row['reward']
            assert row['native_calls']==5 and row['endpoint']['physics_sample_count']==5
            raw_zero &= row['raw_command']['forward_velocity_mps']==0. and row['raw_command']['yaw_rate_rps']==0.
            max_dt_error=max(max_dt_error,abs(row['actual_dt_s']-.01))
            max_reward_error=max(max_reward_error,abs(row['reward']-sum(row['reward_terms'].values())))
            values=[*row['action'],row['reward'],*row['reward_terms'].values(),row['progress']['progress']]
            finite &= all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
            p=row['progress']['progress'];before=row['progress_event']['before']['progress']
            assert 0.<=before<=p<=1.
            assert abs(row['reward_terms']['clearance_progress']-2*(p-before))<1e-10
            assert row['progress_event']['in_bonus_window'] or p==before
            assert row['progress_event']['fully_unloaded'] or p==before
            if p>before: assert row['progress']['credited_run_active']
            info['max_progress']=max(info['max_progress'],p)
            info['flight_seen'] |= row['progress']['flight_seen']
            bonus_sums[e]=bonus_sums.get(e,0.)+sum(row['reward_terms'][k] for k in ('clearance_progress','flight_once','landing_once'))
            assert bonus_sums[e]<=5.+1e-10
    assert count==expected==receipt['episode_transition_sum']==receipt['ppo_timesteps']
    assert native==5*count==receipt['native']['returned_native_calls']==receipt['native']['attempted_native_calls']
    assert len(updates)==count//128==receipt['train_calls']
    for k,row in enumerate(updates,1):
        assert row['transitions']==k*128 and row['train_calls']==k
        assert row['optimization_epochs']==k*(1 if receipt['mode']=='smoke' else 4)
    for episode in episodes:
        observed=by_episode[episode['episode_index']]
        assert episode['transitions']==observed['count']
        assert abs(episode['return']-observed['return'])<1e-8
        assert abs(episode['progress']['progress']-observed['max_progress'])<1e-10
        key=str(episode['spec']['net_clearance_m'])
        bucket=stage_max.setdefault(key,{'episodes':0,'flight_episodes':0,'full_progress_episodes':0,'terminated':0,'max_progress':0.})
        bucket['episodes']+=1;bucket['flight_episodes']+=int(observed['flight_seen'])
        bucket['full_progress_episodes']+=int(observed['max_progress']==1.)
        bucket['terminated']+=int(episode['terminated']);bucket['max_progress']=max(bucket['max_progress'],observed['max_progress'])
    output={'passed':True,'mode':receipt['mode'],'control_intervals':count,'native_calls':native,
            'train_calls':len(updates),'epochs':receipt['optimization_epochs'],
            'finite_logged_rewards_actions_progress':finite,'all_raw_stationary':raw_zero,
            'max_dt_error_s':max_dt_error,'max_reward_sum_error':max_reward_error,
            'maximum_positive_task_bonus_per_episode':max(bonus_sums.values()),
            'completed_episode_stages':stage_max,'new_physics':0,
            'scope':'compact training records only; physical qualification belongs to independent final evaluation',
            'input_sha256':{name:hashlib.sha256((args.folder/name).read_bytes()).hexdigest() for name in
                            ('receipt.json','episodes.jsonl.gz','training_trace.jsonl.gz','updates.jsonl')}}
    assert finite and raw_zero and max_dt_error<1e-10 and max_reward_error<1e-10
    args.output.open('x').write(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
