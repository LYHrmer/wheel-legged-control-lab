"""One analytical proxy check; root executes separately from completed tests."""
import numpy as np
import pytest
from score24 import diagnostics24


def test_exact_native_normalized_cost_and_drive_reference():
    torque=np.repeat((.5*np.asarray([80.,80.,80.,12.]*4))[None,:,None],1200,axis=0)
    torque=np.repeat(torque,5,axis=2)
    c=dict(completed_controls=1200,com_vx_mps=[.2]*1200,
           applied_servo_vx_mps=[.1]*1200,motor_torque_nm=torque)
    d=diagnostics24(c)['windows']
    assert d['drive']['completed_control_count']==800
    assert d['drive']['actual_native_torque_samples']==4000
    assert d['drive']['raw_forward_sse_m2ps2']==0.
    assert d['drive']['servo_forward_rms_mps']==pytest.approx(.1)
    assert d['drive']['torque_cost_mean']==pytest.approx(.25)
    assert d['drive']['torque_cost_sum']==pytest.approx(200.)
    assert d['full']['integral_normalized_motor_torque_squared_s']==pytest.approx(48.)
    c['motor_torque_nm']=torque[:,:,0]
    with pytest.raises(ValueError):diagnostics24(c)
