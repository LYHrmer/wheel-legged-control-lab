"""Pure/synthetic C31 checks; root owns execution and its call accounting."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint31 import load_final31, save_and_reload_final31
from learning31 import (LinearEventPolicy31, LatchSampler31, STD31,
                        prepare_batch31, reward31, update_batch31)
from observation31 import observe31


def _components(duration):
    return {
        'progress_potential_difference': .1,
        'elapsed_s': duration,
        'backtrack_normalized': .2,
        'longitudinal_normalized_square_integral_s': .3,
        'lateral_goal_error_normalized_square_integral_s': .4,
        'yaw_error_normalized_square_integral_s': .5,
        'torque_normalized_square_integral_s': .6,
    }


def _event(cycle, j, start, end):
    observation = [0.]*54
    observation[0] = (-1.)**cycle
    observation[5] = j/3
    observation[6] = cycle/10
    latent = [.03*(cycle+1),-.02*(cycle+1),.01*(cycle+1)]
    old_logp = -sum((v/STD31)**2 for v in latent)/2 - 3*math.log(
        STD31*math.sqrt(2*math.pi))
    return {
        'control_range':[start,end], 'elapsed_controls':end-start,
        **_components((end-start)*.01),
        'skill31_latch':{'mode':'train','cycle_index':cycle,
            'local_control_index':start,
            'observation31':{'macro_index':j,'normalized54':observation},
            'latent_z3':latent,'old_mean_z3':[0.]*3,
            'old_log_prob':old_logp,'old_value':0.},
    }


def _cycles(first_two=False):
    cycles=[]
    for i in range(8):
        if first_two and i == 0:
            macros=[_event(i,0,200,300),_event(i,1,300,1000)]
        else:
            macros=[_event(i,0,200,1000)]
        cycles.append({'cycle_index':i,'terminal_kind':
            'controlled_failure' if i == 1 else 'success',
            'actual_seconds':10.,'archive_valid':True,'native_valid':True,
            'safe_handoff':True,'retention_complete':True,'macros':macros})
    return cycles


def test_observation_uses_pre_state_and_actual_joint_addresses():
    q=np.zeros(23)
    q[:3]=[1.,2.,.455]
    q[3]=1.
    q[7:]=np.linspace(-.2,.2,16)
    v=np.zeros(22)
    v[0]=.1
    mapping={'leg_qpos_addresses':[7,8,9,11,12,13,15,16,17,19,20,21],
             'joint_dof_addresses':list(range(6,22)),
             'leg_range_mid':[0.]*12,'leg_range_half':[1.]*12}
    row=observe31(qpos=q,qvel=v,initial_xy_m=[1.,2.],
        initial_yaw_rad=0.,direction=1,leg_index=2,macro_index=1,
        teacher_body_from_m=[0.,0.,.45],teacher_body_to_m=[0.,.03,.45],
        teacher_shift_time_s=1.,plan_com_height_m=.4,
        previous_consumed_action=[0.,0.,.5],completed_side_controls=200,
        joint_map=mapping)
    assert row['raw54'][3] == 1.
    assert row['normalized54'][5] == pytest.approx(1/3)
    assert row['raw54'][18:30] == pytest.approx(q[mapping['leg_qpos_addresses']])
    assert row['raw54'][30:46] == pytest.approx(v[6:22])
    assert row['raw54'][53] == 2.
    assert row['normalized54'][53] == pytest.approx(2/15)
    assert not any(row['clip_mask54'])


def test_smdp_gae_stays_within_cycle_and_failure_cost_is_real():
    batch=prepare_batch31(_cycles(first_two=True))
    rows=batch['rows']
    first,second=rows[:2]
    assert second['duration_s'] == 7.
    assert second['terminal_adjustment'] == 5.
    assert first['terminal_adjustment'] == 0.
    assert first['gae'] == pytest.approx(first['reward']+(.95**1.)*second['gae'])
    failed=next(r for r in rows if r['cycle_index']==1)
    assert failed['terminal_adjustment'] == -62.
    assert failed['gae'] == pytest.approx(failed['reward'])
    assert sum(r['normalized_advantage'] for r in rows) == pytest.approx(0.,abs=1e-12)
    assert reward31(_components(2.)) == pytest.approx(
        1.-2.-.05*.2-.02*.3-.02*.4-.02*.5-.5*.6)
    broken=_cycles()
    broken[0]['safe_handoff']=False
    with pytest.raises(ValueError):
        prepare_batch31(broken)


def test_actual_grouped_adam_and_pre_kl_rejection_are_distinct():
    batch=prepare_batch31(_cycles())
    policy=LinearEventPolicy31()
    optimizer=policy.optimizer31()
    ledger=update_batch31(policy,optimizer,batch,batch_index=0,
        step_count_before=0,permutation_seed=310031)
    assert ledger['hard_stop'] is None
    assert 1 <= ledger['actual_optimizer_steps'] <= 2
    stepped=next(a for a in ledger['attempts'] if a['optimizer_step'])
    assert stepped['backward'] is True
    assert stepped['parameters_after']['actor']['weight']['adam_step'] == 1
    assert stepped['parameters_after']['value']['weight']['adam_step'] == 1
    for name in ('actor','value'):
        gradients=stepped['gradients'][name]
        assert gradients['factor'] == pytest.approx(
            min(1.,.5/(gradients['raw_norm']+1e-6)))
        assert gradients['clipped_norm'] <= .5+1e-12
    rejected=LinearEventPolicy31()
    with np.errstate(all='raise'):
        rejected.actor.bias.data.fill_(.5)
    receipt=update_batch31(rejected,rejected.optimizer31(),batch,batch_index=0,
        step_count_before=0,permutation_seed=310031)
    assert receipt['normal_kl_stop'] is True
    assert receipt['actual_evaluated_minibatches'] == 1
    assert receipt['actual_optimizer_steps'] == 0
    assert receipt['attempts'][0]['value_forward_rows'] == 0
    assert receipt['attempts'][0]['backward'] is False


def test_sampling_train_vs_eval_and_one_final_reload(tmp_path):
    policy=LinearEventPolicy31()
    observation={'normalized54':[0.]*54}
    train=LatchSampler31(policy,mode='train')
    first=train(observation,{},0,200)
    assert first['old_value'] == 0.
    assert first['policy_receipt']['rng_samples_cumulative'] == 1
    evaluate=LatchSampler31(policy,mode='learned')
    deterministic=evaluate(observation,{},0,200)
    assert deterministic['latent_z3'] == deterministic['old_mean_z3'] == [0.]*3
    assert deterministic['old_value'] is None
    independent,manifest=save_and_reload_final31(policy,tmp_path/'final',
        metadata={'source_hashes':{'learning31.py':'sha'},
                  'execution_contract_id':'C31_event_lateral_RL_pilot_v1'},
        training_receipt={'status':'complete','actual_batches':8,
            'actual_optimizer_steps':8,
            'actor_nonzero_gradient_and_parameter_change':True},
        probe_observations54=[[0.]*54 for _ in range(16)])
    assert independent is not policy
    assert manifest['independent_reload'] is True
    reloaded=load_final31(tmp_path/'final',manifest)
    assert reloaded.actor.bias.detach().tolist() == [0.]*3
    with pytest.raises(FileExistsError):
        save_and_reload_final31(policy,tmp_path/'final',
            metadata={'source_hashes':{'learning31.py':'sha'},
                      'execution_contract_id':'C31_event_lateral_RL_pilot_v1'},
            training_receipt={'status':'complete','actual_batches':8,
                'actual_optimizer_steps':8,
                'actor_nonzero_gradient_and_parameter_change':True},
            probe_observations54=[[0.]*54 for _ in range(16)])
