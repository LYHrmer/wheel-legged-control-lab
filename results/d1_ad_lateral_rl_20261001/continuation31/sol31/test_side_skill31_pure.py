"""No-engine latch/reset checks for the one-instance C31 side extension."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import side_skill31


class FakeTeacher:
    def __init__(self, plant):
        self.plant=plant
        self.phase='idle'
        self.failure=None
        self.index=0
        self.direction=1
        self.yaw=0.
        self.quat=np.array([1.,0.,0.,0.])
        self.initial_pose=np.array([0.,0.,.455])
        self.com_height=.45
        self.leg=None
        self.feet=np.zeros((4,3))
        self.scratch=types.SimpleNamespace(qpos=np.zeros(23))
        self.prepare_calls=0
        self.ik_calls=0
        self.reset_calls=0

    def reset(self):
        self.reset_calls+=1

    def _prepare_leg(self):
        self.prepare_calls+=1
        self.phase='shift'
        self.leg=0
        self.body_from=np.array([0.,0.,.455])
        self.body_to=np.array([0.,.03,.455])
        self.shift_time=.5

    def _ik(self,*args,**kwargs):
        self.ik_calls+=1
        assert kwargs == {'outer':2,'inner':6}
        return np.zeros(16),0.,np.zeros((4,3)),np.array([.05,0.,.455])

    def _move_time(self,span):
        assert np.linalg.norm(span[:2]) == pytest.approx(.022)
        return .4


def _side(monkeypatch,mode, callback=None):
    qpos=np.zeros(23)
    qpos[3]=1.
    plant=types.SimpleNamespace(measurement_data=types.SimpleNamespace(
        qpos=qpos,qvel=np.zeros(22)))
    teacher=FakeTeacher(plant)
    created=[]
    def factory(p,arm):
        created.append((p,arm))
        return teacher
    monkeypatch.setitem(sys.modules,'hybrid_adapter27',
                        types.SimpleNamespace(make_fast_side27=factory))
    monkeypatch.setitem(sys.modules,'d1_fast_side_step',
                        types.SimpleNamespace(_edges=lambda _: [
                            (np.zeros(2),np.array([1.,0.]))]))
    monkeypatch.setitem(sys.modules,'wheel_legged_control.d1.model',
                        types.SimpleNamespace(JOINT_POSITION_LOW=np.full(16,-1.),
                                              JOINT_POSITION_HIGH=np.full(16,1.)))
    mapping={'leg_qpos_addresses':[7,8,9,11,12,13,15,16,17,19,20,21],
             'joint_dof_addresses':list(range(6,22)),
             'leg_range_mid':[0.]*12,'leg_range_half':[1.]*12}
    monkeypatch.setattr(side_skill31,'compiled_joint_map31',lambda p,b: mapping)
    clock={'control':0,'cycle':0}
    side=side_skill31.make_side31(plant,mode=mode,
        geometry_binding={'kinematics24':{},'wheel_index_by_body_id':
                          {10:0,11:1,12:2,13:3}},
        control_index_provider=lambda:clock['control'],
        cycle_index_provider=lambda:clock['cycle'],sample_latch31=callback)
    assert side is teacher and created == [(plant,'teacher')]
    return side,clock


def test_fixed_and_latent_zero_consume_identical_old_reference(monkeypatch):
    fixed,_=_side(monkeypatch,'fixed')
    fixed.begin_episode31(cycle_index=0,mode='fixed',direction=1)
    fixed._prepare_leg()
    fixed_event,=fixed.drain_macro_events()
    assert fixed.ik_calls==1
    assert fixed_event['consumed_action']==pytest.approx((0.,-.008,.5))
    assert fixed_event['skill31_latch']['latent_z3'] is None
    calls=[]
    def zero_sample(obs,plan,cycle,index):
        calls.append((obs,plan,cycle,index))
        return {'latent_z3':[0.,0.,0.],'old_mean_z3':[0.,0.,0.],
                'latent_std_z3':[.35]*3,
                'old_log_prob':-3*np.log(.35*np.sqrt(2*np.pi)),
                'old_value':0.}
    train,_=_side(monkeypatch,'train',zero_sample)
    train.begin_episode31(cycle_index=0,mode='train',direction=1)
    train._prepare_leg()
    train_event,=train.drain_macro_events()
    assert len(calls)==1 and calls[0][2:]==(0,0)
    assert train.ik_calls==1
    assert train_event['consumed_action']==fixed_event['consumed_action']
    assert train_event['consumed_body_to_m']==fixed_event['consumed_body_to_m']
    assert train_event['consumed_shift_time_s']==fixed_event['consumed_shift_time_s']
    assert train_event['skill31_latch']['observation31']['side_elapsed_s']==0.


def test_reset_clears_old_native_evidence_without_second_controller(monkeypatch):
    side,clock=_side(monkeypatch,'zero')
    side.begin_episode31(cycle_index=0,mode='zero',direction=1)
    side.set_interval_evidence(control_index=199,
        contact_free_5_by_wheel=(True,)*4,
        native_wheel_min_z_5x4=np.ones((5,4)))
    assert side._skill30_interval_evidence['control_index']==199
    clock['cycle']=1
    side.begin_episode31(cycle_index=1,mode='fixed',direction=1)
    assert side._skill30_interval_evidence is None
    assert side._skill31_previous_consumed.tolist()==[0.,0.,0.]
    side._prepare_leg()
    event,=side.drain_macro_events()
    assert event['skill31_latch']['cycle_index']==1
    assert side.reset_calls==1  # factory only; no second scratch or controller
