from copy import deepcopy
from decision20 import decide, under
import pytest


def ratio(n=1.,d=1.):
    return {'numerator':n,'denominator':d}


def reports():
    valid={'training_valid':True,'coverage_valid':True}
    pool={'complete':True,'actors':{'B':{'all_tasks_passed':True}},
          'comparisons':{'B/A':{'error':ratio(.85),'cost':ratio()},
                         'B/old':{'error':ratio(.90),'cost':ratio()},
                         'B/zero':{'error':ratio(),'cost':ratio(1.1)}},
          'family_B_over_A':{f:{m:ratio() for m in ('drive_error_mean','hold_error_mean')}
                            for f in ('yaw','bumps')}}
    dev={'equal_case_pool':pool,'original_six_pool':deepcopy(pool),
         'source_closure_verified':True,'regression_reuse':{'complete':True},
         'numeric_scores':[{'experiment_actor':'B','case_id':n,'task_passed':True} for n in (
             'flat_0p6','flat_1p6','flat_1p2_yaw','bumps_0p4','rough_0p35','ramp_0p45_complete','flat_1p2_yaw_mirror')]}
    return valid,deepcopy(valid),dev,{'passed':True}


def test_requires_all_gates_including_original_cost_and_hold():
    args=reports()
    assert decide(*args)['candidate_stage']==2
    args[2]['original_six_pool']['comparisons']['B/A']['cost']=ratio(1.101)
    assert not decide(*args)['continue_training']
    args=reports()
    args[2]['equal_case_pool']['family_B_over_A']['yaw']['hold_error_mean']=ratio(1.021)
    assert not decide(*args)['continue_training']
    args=reports()
    args[1]['coverage_valid']=False
    assert decide(*args)['interpretation']=='inconclusive_no_extension'


def test_zero_denominator_never_manufactures_improvement():
    assert not under(ratio(0.,0.),.9)
    assert under(ratio(0.,0.),1.02)
    assert not under(ratio(1.,0.),1.2)


@pytest.mark.parametrize('change_initial',[False,True])
def test_pair_rejects_initial_state_mismatch_even_when_rollout_arrays_match(tmp_path,monkeypatch,change_initial):
    import numpy as np
    import decision20
    from archive13.atomic_archive_13 import ArchiveWriter
    monkeypatch.setattr(decision20,'C',tmp_path)
    writer=ArchiveWriter()
    for arm in ('A','B'):
        folder=tmp_path/f'train_{arm}_1/training'
        folder.mkdir(parents=True)
        state={key:np.zeros(3) for key in ('qpos','qvel','ctrl','qacc_warmstart','observation')}
        if arm=='B' and change_initial:
            state['qacc_warmstart'][0]=1.
        writer.commit_npz(folder/'training_episode_000000_initial_state.npz',state)
        writer.commit_json(folder/'training_episode_000000_control_reset.json',{'test_fixture':True,'z':0.})
        for name,stem in (('numeric','controls'),('gaussian','gaussian')):
            blocks=folder/('training_'+name+'_blocks')
            blocks.mkdir()
            writer.commit_npz(blocks/(stem+'_0000.npz'),{'control_index':np.arange(1024),'x':np.zeros((1024,2))})
    result=decision20.first_pair()
    assert result['passed'] is (not change_initial)
