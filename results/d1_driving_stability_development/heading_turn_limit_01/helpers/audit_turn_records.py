"""Read-only turn score, contact kinematics, carry-forward and budget audit."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {'new_physics_steps': 0, 'episodes': []}
    overall = json.loads((args.input/'summary.json').read_text())
    assert overall['actual_control_transitions'] == 3200
    assert overall['actual_physics_substeps'] == 16000
    assert overall['actual_new_control_transitions'] == 2400
    assert overall['carried_completed_transitions'] == 800
    assert all(p['passed'] and all(p['checks'].values()) for p in overall['pairs'])
    assert not overall['candidate_passed_both_turn_cases']
    for case, sign in [('stationary_turn_left_hold', 1), ('stationary_turn_right_hold', -1)]:
        for mode in ('limit0p6', 'limit1p0'):
            p=args.input/case/mode
            trace=rows(p/'trace.jsonl.gz'); diag=rows(p/'turn_diagnostics.jsonl.gz')
            assert len(trace)==len(diag)==800
            with np.load(p/'states.npz') as data:
                assert data['qpos'].shape==(801,23)
                assert data['qvel'].shape==(801,22)
                assert np.all(data['applied_actions']==0)
            reference=0.; errors=[]; residual=0.; force_error=0.; samples=0
            for k,(a,b) in enumerate(zip(trace,diag)):
                raw=sign*.6 if 200<=k<250 else 0.
                assert a['heading_task']['user_command_before']['yaw_rate_rps']==raw
                assert a['heading_task']['user_command_before']['forward_velocity_mps']==0.
                reference+=raw*a['heading_task']['actual_dt_s']
                difference=a['heading_task']['truth_heading_after']-reference
                error=float(np.arctan2(np.sin(difference),np.cos(difference)))
                assert abs(error-a['heading_error_rad'])<1e-12
                errors.append(error)
                assert b['tick']==k and b['endpoint_tick']==k+1
                assert len(b['applied_torque_nm'])==5
                c=b['contacts']; moment=np.zeros(3); force=np.zeros(3); counts=[0]*4
                assert abs(c['measurement_time_s']-(k+1)*.01)<1e-10
                for item in c['contacts']:
                    n=np.array(item['normal_world']); v=np.array(item['relative_velocity_world_mps'])
                    robot=np.array(item['robot_point_velocity_world_mps'])
                    terrain=np.array(item['terrain_point_velocity_world_mps'])
                    assert np.max(np.abs(robot-terrain-v))<1e-14
                    total=sum(np.array(item[key]) for key in ('leg_dof_velocity_world_mps','wheel_dof_velocity_world_mps','free_base_velocity_world_mps'))
                    residual=max(residual,float(np.linalg.norm(total-robot)))
                    assert abs(np.linalg.norm(v-np.dot(v,n)*n)-item['tangential_relative_speed_mps'])<1e-14
                    f=np.array(item['force_world_n']); force+=f
                    moment+=np.cross(np.array(item['pos_world_m'])-c['reference_world_m'],f)+item['torque_world_nm']
                    counts[item['wheel_index']]+=1
                assert counts==c['contact_count_by_wheel']
                force_error=max(force_error,float(np.max(np.abs(np.r_[force,moment]-c['total_wrench_world_6']))))
                samples+=len(c['contacts'])
            assert abs(reference-sign*.3)<1e-12
            peak=max(abs(e) for e in errors)
            s=json.loads((p/'turn_summary.json').read_text())
            assert abs(peak-s['heading_peak_rad'])<1e-12
            assert s['gates']['failed']==['heading_peak']
            assert residual<1e-12 and force_error<1e-10
            report['episodes'].append({'case':case,'condition':mode,'raw_heading_peak_rad':peak,
                'raw_heading_peak_deg':float(np.degrees(peak)),
                'contact_velocity_decomposition_max_error_mps':residual,
                'contact_wrench_resummation_max_error':force_error,
                'observed_endpoint_contacts':samples,'original_heading_peak_gate_passed':False})
    report.update(audit_checks_passed=True,actual_control_transitions_including_carry=3200,
        actual_new_control_transitions_in_resumed_run=2400,actual_physics_substeps=16000,
        both_turn_candidates_passed=False,
        normal_semantics='normal_world is raw geom1-to-geom2 contact-frame normal; force_world acts on robot. Tangent projection is invariant to normal sign.',
        contact_sampling='endpoint, not control-period average or continuous friction-work integral')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__=='__main__':
    main()
