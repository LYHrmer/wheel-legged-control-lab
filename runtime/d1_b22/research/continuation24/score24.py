"""Preregistered true 15mm qualification; no dynamics or policy imports."""
from __future__ import annotations
import math
import numpy as np
from geometry24 import CONTROLS, FAR


def diagnostics24(c):
    """Observed raw/servo errors and actual 5x16 torque proxy, never a new gate."""
    n=c['completed_controls']
    velocity=np.asarray(c['com_vx_mps'],dtype=float)
    servo=np.asarray(c['applied_servo_vx_mps'],dtype=float)
    torque=np.asarray(c['motor_torque_nm'],dtype=float)
    if velocity.shape!=(n,) or servo.shape!=(n,) or torque.shape!=(n,16,5):
        raise ValueError('C24 diagnostics require actual five native torques per 16-axis control')
    limits=np.asarray([80.,80.,80.,12.]*4)
    cost=np.mean((torque/limits[None,:,None])**2,axis=(1,2))
    raw=np.asarray([.2 if 200<=tick<1000 else 0. for tick in range(n)])
    windows={}
    for name,begin,end in (('drive',200,1000),('full',0,1200)):
        stop=min(n,end)
        count=max(0,stop-begin)
        selection=slice(begin,stop)
        raw_sse=float(np.sum((velocity[selection]-raw[selection])**2))
        servo_sse=float(np.sum((velocity[selection]-servo[selection])**2))
        windows[name]=dict(requested_control_window_half_open=[begin,end],
            completed_control_count=count,actual_native_torque_samples=5*count,
            full_requested_window_recorded=n>=end,
            raw_forward_sse_m2ps2=raw_sse,servo_forward_sse_m2ps2=servo_sse,
            raw_forward_rms_mps=math.sqrt(raw_sse/count) if count else None,
            servo_forward_rms_mps=math.sqrt(servo_sse/count) if count else None,
            torque_cost_mean=float(cost[selection].mean()) if count else None,
            torque_cost_sum=float(cost[selection].sum()),
            integral_normalized_motor_torque_squared_s=float(.16*cost[selection].sum()))
    return dict(windows=windows,torque_limits_nm=limits.tolist(),
        torque_sample_source='actual 500Hz native actuator traces; 5 traces x 16 motors per control',
        cost_definition='C18 c_t = mean over 5 native samples and 16 motors of (tau/tau_limit)^2',
        integral_definition='.002 * sum over actual native samples and all 16 motors of (tau/tau_limit)^2',
        cost_is_energy=False,new_qualification_gate=False)


def score_case24(c):
    n = c['completed_controls']
    full = n == CONTROLS and c['truncated'] is True and c['terminated'] is False
    def values(key):
        return np.asarray(c[key],dtype=float)
    def maximum_abs(key):
        a=values(key)
        return float(np.max(np.abs(a))) if a.size else None
    speed = values('com_vx_mps')[274:875]
    speed_complete = len(speed)==601
    mean_speed=float(speed.mean()) if speed.size else None
    rms_speed=float(np.sqrt(np.mean((speed-.2)**2))) if speed.size else None
    stop_vx=values('com_vx_mps')[-100:]
    stop_vz=values('whole_system_com_vz_mps')[-100:]
    stop_z=values('base_world_z_m')[-100:]
    stop_yaw=values('body_yaw_rate_rps')[-100:]
    stop_clear=values('clearance_m')[-100:]
    load=values('native_positive_load_by_wheel')[-500:]
    fractions=load.mean(axis=0).tolist() if len(load) else [0.]*4
    bounds=c['final_collision_bounds']
    all_geoms_past = bool(bounds) and all(b['minimum_world_m'][0] > FAR+b['margin_m'] for b in bounds)
    wheels={b['wheel_index'] for b in bounds if b['wheel_index'] is not None}
    clearance=min([c['initial_clearance_m'],*c['clearance_m'],*c['native_clearance_m']])
    roll=max(abs(c['initial_roll_deg']),maximum_abs('native_roll_deg'))
    pitch=max(abs(c['initial_pitch_deg']),maximum_abs('native_pitch_deg'))
    yaw_deg=math.degrees(maximum_abs('native_heading_rad'))
    lateral=maximum_abs('native_lateral_offset_m')
    metrics=dict(speed_endpoint_window_inclusive=[275,875],speed_sample_count=len(speed),
        speed_mean_mps=mean_speed,speed_rms_error_mps=rms_speed,
        max_abs_world_roll_deg=roll,max_abs_world_pitch_deg=pitch,
        max_abs_heading_change_deg=yaw_deg,max_abs_lateral_displacement_m=lateral,
        min_initial_and_native_clearance_m=clearance,
        final100_max_abs_body_vx_mps=float(np.max(np.abs(stop_vx))),
        final100_max_abs_whole_system_com_vz_mps=float(np.max(np.abs(stop_vz))),
        final100_world_base_z_rmse_m=float(np.sqrt(np.mean((stop_z-.455)**2))),
        final100_mean_abs_body_yaw_rate_rps=float(np.mean(np.abs(stop_yaw))),
        final100_clearance_std_m=float(np.std(stop_clear)),
        final500_native_positive_load_fraction_by_wheel=fractions,
        positive_box_loaded_native_intervals=c['positive_wheel_load_native_counts_by_family'].get('step',0),
        positive_box_contact_features=c['box_feature_positive_counts'],
        final_all_robot_collision_shapes_past_far_edge=all_geoms_past,
        final_collision_bounds=bounds)
    gates=dict(full_1200_controls=full,
        real_box_positive_wheel_native_load=metrics['positive_box_loaded_native_intervals']>0,
        no_nonwheel_contact=sum(c['native_nonwheel_ground_candidate_count'])==0,
        world_roll_and_pitch_le_10deg=roll<=10 and pitch<=10,
        heading_change_le_5deg=yaw_deg<=5,
        lateral_displacement_le_0p1m=lateral<=.1,
        clearance_ge_0p28m=clearance>=.28,
        speed_mean_ge_0p18_and_rms_le_0p05=speed_complete and mean_speed>=.18 and rms_speed<=.05,
        every_robot_collision_shape_past_far_edge_with_own_margin=all_geoms_past and wheels==set(range(4)),
        final100_max_abs_body_vx_le_0p03=full and metrics['final100_max_abs_body_vx_mps']<=.03,
        final100_max_abs_whole_system_com_vz_le_0p03=full and metrics['final100_max_abs_whole_system_com_vz_mps']<=.03,
        final100_world_base_z_rmse_le_0p015=full and metrics['final100_world_base_z_rmse_m']<=.015,
        final500_native_each_wheel_load_fraction_ge_0p95=full and len(load)==500 and min(fractions)>=.95,
        final100_mean_abs_yaw_le_0p05=full and metrics['final100_mean_abs_body_yaw_rate_rps']<=.05,
        final100_clearance_std_le_0p02=full and metrics['final100_clearance_std_m']<=.02,
        no_map_escape_or_fall=c['map_escape_count']==0 and c['fall_count']==0)
    return dict(actor=c['experiment_actor'],checkpoint_sha256=c['checkpoint_sha256'],
        completed_controls=n,metrics=metrics,gates=gates,
        diagnostics=diagnostics24(c),
        task_qualified=all(bool(value) for value in gates.values()),
        semantics='true native plane plus 15mm box; C18 world-upright local-clearance command',
        trained_step_skill_or_rl_contribution_claim=False)
