#!/usr/bin/env python3
"""File-only independent shared-heave log audit; imports only Python stdlib.

Run only after receipt/source_after and closed gzip streams exist. Never imports
an environment, controller, policy or simulator; never loads a checkpoint model.
Rebuilds task event accounting from recorded contact summaries, not native states.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(x):
    if isinstance(x, float):
        assert math.isfinite(x), 'nonfinite number'
    elif isinstance(x, dict):
        assert '__nonfinite__' not in x, 'tagged nonfinite'
        for v in x.values():
            finite(v)
    elif isinstance(x, list):
        for v in x:
            finite(v)


def equal(a, b, path=''):
    if isinstance(b, dict):
        assert isinstance(a, dict) and a.keys() == b.keys(), ('keys', path)
        for k in b:
            equal(a[k], b[k], path + '/' + k)
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), ('shape', path)
        for i, (u, v) in enumerate(zip(a, b)):
            equal(u, v, path + '/' + str(i))
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        assert type(a) is type(b) and a == b, (path, a, b)
    elif isinstance(b, int):
        assert type(a) is int and a == b, (path, a, b)
    else:
        assert abs(a - b) <= 1e-12, (path, a, b)


def positive_zero(xs):
    return all(x == 0 and math.copysign(1., x) == 1 for x in xs)


def episode_spec(mode, index, completed):
    if mode == 'smoke':
        request, goal = (200, .005) if index == 0 else (None, 0.)
    elif index % 5 == 4:
        request, goal = None, 0.
    else:
        request = (150, 200, 250)[(index - index // 5) % 3]
        goal = .005 if completed < 32768 else .01 if completed < 65536 else .02
    return dict(request_tick=request, net_clearance_m=goal, friction_scale=1.,
                condition='hold' if request is None else 'profile', is_hold=request is None,
                bonus_window_ticks=None if request is None else [request, request+120],
                height_mask_window_ticks=None if request is None else [request+25, request+120])


def height(tick, request):
    offset = -1 if request is None else tick-request
    if 0 <= offset < 25:
        return .405
    if 25 <= offset < 40:
        return .5
    if 75 <= offset < 120:
        return .435
    return .455


def initial_progress():
    return dict(schema='d1-jump-clearance-progress-v1', progress=0., flight_seen=False,
                consecutive_unloaded_intervals=0, unloaded_native_samples=0,
                airborne_progress=0., run_upward_onset=False, run_maximum_net_gap_m=0.,
                certified_run_maximum_net_gap_m=0., completed_intervals=0,
                credited_run_active=False)


def rebuild_progress(before, endpoint, active, goal):
    p = dict(before)
    unloaded = (all(v == 0 for v in endpoint['active_sample_fraction_by_wheel'])
                and all(v == 0 for force in endpoint['wheel_force_world_n'] for v in force)
                and endpoint['simultaneous_minimum_gap_m'] > endpoint['contact_margin_m'])
    gap = max(0., endpoint['simultaneous_minimum_gap_m'] - endpoint['contact_margin_m'])
    onset = False
    if unloaded and active:
        onset = p['consecutive_unloaded_intervals'] == 0
        p['run_maximum_net_gap_m'] = gap if onset else max(p['run_maximum_net_gap_m'], gap)
        if onset:
            p['run_upward_onset'] = endpoint['com_velocity_mps'][2] > 0
        p['consecutive_unloaded_intervals'] += 1
        p['unloaded_native_samples'] += 5
    else:
        p.update(credited_run_active=False, consecutive_unloaded_intervals=0,
                 run_maximum_net_gap_m=0., run_upward_onset=False)
        if not p['flight_seen']:
            p['unloaded_native_samples'] = 0
    newly = (unloaded and active and not p['flight_seen'] and p['run_upward_onset']
             and p['consecutive_unloaded_intervals'] >= 2)
    credited = 0.
    if newly:
        p.update(flight_seen=True, credited_run_active=True,
                 certified_run_maximum_net_gap_m=p['run_maximum_net_gap_m'])
        credited = p['run_maximum_net_gap_m']
    elif unloaded and active and p['credited_run_active']:
        credited = gap
    if credited > 0:
        p['progress'] = max(p['progress'], min(1., credited/goal))
    p['completed_intervals'] += 1
    p['airborne_progress'] = 1. if p['flight_seen'] else min(1., p['unloaded_native_samples']/10)
    return p, unloaded, gap, onset, newly, credited


def audit(folder, mode, preflight):
    inputs = {}
    def read(path):
        path = Path(path)
        inputs[str(path)] = sha(path)
        value = json.loads(path.read_text())
        finite(value)
        return value

    pre = read(preflight)
    assert pre['passed'] and pre['interface_gate_passed'] and pre['root_execution_authorized']
    pinned = pre['input_sha256']
    assert len(pinned) == 214, ('source hash count', len(pinned))
    mismatches = [path for path, wanted in pinned.items() if sha(path) != wanted]
    assert not mismatches, ('source mismatch', mismatches)
    protocol = read(pre['evaluation_protocol'])
    assert protocol['action_schema'] == 'd1-jump-shared-heave-window-v1'
    expected = protocol[mode]
    receipt, init = read(folder/'receipt.json'), read(folder/'initial_training.json')
    assert receipt['passed'] is True and receipt['error'] is None
    source_after = read(folder/'source_after.json')
    assert all(source_after[k] is True for k in ['passed','source_unchanged','frozen77_unchanged'])
    assert init['seed'] == expected['policy_seed'] and init['mode'] == mode
    assert init['budget'] == expected['controls'] and init['optimizer_state_entries'] == 0
    hp = init['hyperparameters']
    epochs = 1 if mode == 'smoke' else 4
    for key in ['learning_rate','n_steps','batch_size','gamma','gae_lambda','clip_range','ent_coef','device']:
        equal(hp[key], protocol['ppo'][key], 'hyperparameters/'+key)
    assert hp['n_epochs'] == epochs and hp['num_envs'] == 1
    assert hp['policy_kwargs']['net_arch'] == [64,64] and hp['policy_kwargs']['log_std_init'] == -2.

    resets = {}
    for file in sorted((folder/'boundaries').glob('*_reset.json')):
        reset = read(file)
        index = reset['episode_index']
        assert index == len(resets) and reset['actual_seed'] == expected['episode_seed_base']+index
        assert reset['incoming_seed'] == (expected['policy_seed'] if index == 0 else None)
        assert reset['time_s'] == 0 and reset['episode_transitions'] == 0 and not reset['episode_finished']
        assert reset['episode_reward'] == 0 and reset['last_info'] is None
        spec = episode_spec(mode,index,reset['total_transitions'])
        equal(reset['spec'], spec)
        equal(reset['progress'], initial_progress())
        meta = reset['metadata']
        equal(meta['jump_episode'], spec)
        assert meta['task_schema'] == protocol['outer_task_schema']
        assert meta['action_schema'] == protocol['action_schema']
        assert meta['observation_schema'] == protocol['observation_schema']
        assert (meta['policy_action_size'],meta['physical_action_size']) == (1,8)
        matrix = meta['action_mapping']['active_embedding_matrix']
        assert matrix == [[1.],[1.],[1.],[1.],[0.],[0.],[0.],[0.]]
        assert meta['action_mapping']['gate']['window_ticks'] == 120
        assert meta['action_mapping']['gate']['clears_controller_or_action_memory'] is False
        assert meta['action_mapping']['leg_action_scale_m'] == .04
        assert meta['action_mapping']['wheel_action_scale_rad_s'] == 4.
        assert meta['domain']['friction_scale'] == 1.
        assert meta['collision_terrain']['geom_friction'] == [.9,.005,.0001]
        assert len(reset['observation']) == 95 and reset['observation'][85:87] == [-1.,0.]
        resets[index] = reset
    first = read(folder/'first_episode_metadata.json')
    equal(first,resets[0]['metadata'])

    stats = {i:dict(count=0,reward=0.,bonus=0.,last=None,progress=initial_progress(),
                    active_rows=0,flight_events=0,landing_events=0,peak_net_gap_m=0.) for i in resets}
    active_rows = masked_rows = protected_rows = leg_limited_rows = count = 0
    max_wheel_action = max_requested = max_protected = max_scalar = 0.
    terminal_indices = []
    trace_path = folder/'training_trace.jsonl.gz'
    with gzip.open(trace_path,'rt') as stream:
        for line in stream:
            row = json.loads(line); finite(row)
            count += 1
            assert row['total_transition'] == count
            index, tick = row['episode_index'], row['executed_tick']
            stat, reset = stats[index], resets[index]
            assert tick == stat['count'] and 0 <= tick < 600
            assert count == reset['total_transitions'] + tick + 1
            assert stat['last'] is None or not(stat['last']['terminated'] or stat['last']['truncated'])
            request = reset['spec']['request_tick']
            active = request is not None and request <= tick < request+120
            a = row['action']; assert len(a) == 1
            c = min(1.,max(-1.,a[0])); mapped = [c]*4+[0.]*4 if active else [0.]*8
            heave = row['heave_action']
            assert heave['schema'] == protocol['action_schema']
            assert heave['executed_tick'] == tick and heave['request_tick'] == request
            assert heave['active'] is active
            assert heave['policy_action_1'] == a and heave['clipped_policy_action_1'] == [c]
            assert heave['applied_physical_action_8'] == mapped and row['physical_action'] == mapped
            assert positive_zero(row['physical_action'][4:])
            assert active or positive_zero(row['physical_action'])
            assert heave['parent_reward_uses'] == 'applied_and_previous_physical_action_8'
            raw = row['raw_command']
            assert raw['forward_velocity_mps'] == raw['yaw_rate_rps'] == 0
            assert raw['clearance_m'] == height(tick,request)
            assert row['servo_command']['forward_velocity_mps'] == 0
            assert row['servo_command']['clearance_m'] == raw['clearance_m']
            assert abs(row['actual_dt_s']-.01) < 1e-10 and row['native_calls'] == 5
            end, geo, event = row['endpoint'],row['geometry'],row['progress_event']
            assert end['physics_sample_count'] == 5
            assert len(end['active_sample_fraction_by_wheel']) == 4
            assert all(0 <= f <= 1 and abs(f*5-round(f*5)) < 1e-12 for f in end['active_sample_fraction_by_wheel'])
            assert len(end['wheel_force_world_n']) == 4 and all(len(f)==3 for f in end['wheel_force_world_n'])
            equal(end['wheel_normal_force_n'],[f[2] for f in end['wheel_force_world_n']])
            loaded = all(v>0 for v in end['wheel_normal_force_n'])
            assert end['all_wheels_positively_loaded'] is loaded
            assert end['contact_margin_m'] == geo['contact_margin_m'] == .001
            assert end['wheel_bottom_gap_m'] == geo['wheel_bottom_gap_m']
            assert end['simultaneous_minimum_gap_m'] == min(end['wheel_bottom_gap_m'])
            assert end['simultaneous_minimum_gap_m'] == geo['simultaneous_minimum_gap_m']
            equal(end['com_vertical_velocity_mps'],end['com_velocity_mps'][2])
            displacement = math.hypot(*(u-v for u,v in zip(end['base_position_m'][:2],end['initial_origin_xy_m'])))
            equal(end['planar_displacement_m'],displacement)
            after,unloaded,gap,onset,newly,credited = rebuild_progress(stat['progress'],end,active,reset['spec']['net_clearance_m'])
            assert end['fully_unloaded'] is unloaded
            equal(end['net_gap_m'],gap)
            equal(event['before'],stat['progress']); equal(event['after'],after); equal(row['progress'],after)
            for key,value in dict(executed_tick=tick,in_bonus_window=active,fully_unloaded=unloaded,
                                  net_gap_m=gap,com_vertical_velocity_mps=end['com_velocity_mps'][2],
                                  run_onset=onset,newly_certified_flight=newly,credited_net_gap_m=credited,
                                  terminated=row['terminated'],truncated=row['truncated']).items():
                equal(event[key],value,'event/'+key)
            equal(event['dt_s'],row['actual_dt_s'])
            delta = 2*(after['progress']-stat['progress']['progress'])
            flight = 1. if newly else 0.
            equal(event['clearance_progress'],delta);equal(event['flight_once'],flight)
            landing = (tick == 599 and not row['terminated'] and row['truncated'] and request is not None
                       and after['flight_seen'] and after['progress'] >= 1.-1e-12 and loaded
                       and abs(end['base_position_m'][2]-.455)<=.015
                       and abs(end['body_forward_velocity_mps'])<=.03
                       and abs(end['com_velocity_mps'][2])<=.03
                       and max(map(abs,end['base_rpy_rad'][:2]))<=math.radians(10)
                       and abs(end['heading_error_rad'])<=math.radians(5) and displacement<=.1)
            terms = dict(row['parent_reward_terms'])
            masked = request is not None and request+25 <= tick < request+120
            if masked: terms['height'] = 0.
            terms.update(clearance_progress=delta,flight_once=flight,
                         stationary_offset=-row['actual_dt_s']*.5*min((displacement/.1)**2,4.),
                         landing_once=2. if landing else 0.)
            equal(row['reward_terms'],terms); equal(row['reward'],math.fsum(terms.values()))
            stat['count'] += 1; stat['reward'] += row['reward'];stat['bonus'] += delta+flight+terms['landing_once']
            assert -1e-12 <= stat['bonus'] <= 5.+1e-12
            stat['active_rows'] += int(active);stat['flight_events'] += int(newly);stat['landing_events'] += int(landing)
            stat['peak_net_gap_m'] = max(stat['peak_net_gap_m'],gap)
            stat['progress'],stat['last'] = after,row
            active_rows += int(active);masked_rows += int(masked)
            protected_rows += int(row['torque_protected_axes']>0);leg_limited_rows += int(row['leg_rate_limited_axes']>0)
            assert 0 <= row['leg_rate_limited_axes'] <= 12 and row['leg_rate_limited_axes'] <= row['rate_limited_axes'] <= 16
            assert 0 <= row['torque_protected_axes'] <= 16
            assert row['peak_abs_protected_torque_nm'] <= 80.+1e-12
            max_scalar = max(max_scalar,abs(a[0]));max_wheel_action = max(max_wheel_action,*map(abs,mapped[4:]))
            max_requested = max(max_requested,row['peak_abs_requested_torque_nm'])
            max_protected = max(max_protected,row['peak_abs_protected_torque_nm'])
            if row['terminated'] or row['truncated']:
                assert not row['truncated'] or tick == 599
                terminal_indices.append(index)
    inputs[str(trace_path)] = sha(trace_path)

    completed_path = folder/'episodes.jsonl.gz'
    with gzip.open(completed_path,'rt') as stream:
        completed = [json.loads(line) for line in stream]
    finite(completed);inputs[str(completed_path)] = sha(completed_path)
    assert [e['episode_index'] for e in completed] == terminal_indices
    def snapshot_check(snap,index):
        stat = stats[index]
        assert snap['episode_index'] == index and snap['episode_transitions'] == stat['count']
        equal(snap['spec'],resets[index]['spec']);equal(snap['progress'],stat['progress'])
        equal(snap['episode_reward'],stat['reward'])
        assert abs(snap['time_s']-stat['count']*.01) < 1e-9
        obs = snap['observation'];assert len(obs) == 95
        p = stat['progress']['progress']; airborne = stat['progress']['airborne_progress']
        f32 = lambda v: struct.unpack('f',struct.pack('f',v))[0]
        assert obs[91] == f32(p) and obs[92] == f32(airborne)
        if stat['last'] is not None:
            for field,infokey in [('physical_action','applied_action'),('progress','jump_progress'),
                                 ('progress_event','jump_progress_event'),('endpoint','jump_endpoint_metrics'),
                                 ('reward_terms','reward_terms'),('heave_action','heave_action')]:
                equal(snap['last_info'][infokey],stat['last'][field])
    for record in completed:
        i = record['episode_index'];stat = stats[i]
        assert record['transitions'] == stat['count'] and record['end_total_transitions'] == stat['last']['total_transition']
        equal(record['return'],stat['reward']);equal(record['spec'],resets[i]['spec']);equal(record['progress'],stat['progress'])
        for flag in ['terminated','truncated','terminal_reason']:
            assert record[flag] == stat['last'][flag]
        snap = read(folder/'boundaries'/f'episode_{i:05d}_end.json')
        snapshot_check(snap,i);assert snap['episode_finished'] is True
        if i+1 in resets: assert resets[i+1]['total_transitions'] == record['end_total_transitions']
    final = read(folder/'final_snapshot.json');snapshot_check(final,max(stats))
    assert final['total_transitions'] == count
    assert final['episode_finished'] is (max(stats) in terminal_indices)

    update_path = folder/'updates.jsonl'
    updates = [json.loads(line) for line in update_path.read_text().splitlines()]
    finite(updates);inputs[str(update_path)] = sha(update_path)
    for j,update in enumerate(updates,1):
        assert update['transitions'] == j*128 and update['train_calls'] == j
        assert update['optimization_epochs'] == j*epochs and update['logger']['train/n_updates'] == j*epochs
    assert count == expected['controls'] == receipt['successful_training_transitions'] == receipt['episode_transition_sum'] == receipt['ppo_timesteps']
    assert len(updates) == expected['train_calls'] == receipt['train_calls']
    assert len(updates)*epochs == expected['optimization_epochs'] == receipt['optimization_epochs']
    native = receipt['native']
    assert native['attempted_native_calls'] == native['returned_native_calls'] == count*5 == expected['native_substeps']
    assert native['maximum_native_calls'] == count*5 and native['failed_native_calls'] == 0
    assert abs(native['actual_integrated_seconds']-count*.01) < 1e-7
    assert native['full_native_training_archive'] is False
    assert len(completed) == receipt['completed_episodes']
    if mode == 'smoke':
        assert [s['count'] for s in stats.values()] == [600,40]
        assert receipt['smoke_weights_discarded'] is True and receipt['checkpoint_records'] == []
    else:
        assert [r['steps'] for r in receipt['checkpoint_records']] == [32768,65536,131072]
        for record in receipt['checkpoint_records']:
            cp = folder/record['directory']
            verification = read(cp/'reload_verification.json')
            assert verification == {'passed':True,'additional_physics':0}
            equal(read(cp/'receipt.json'),record)
            metadata = read(cp/'model.metadata.json')
            # Model archive hash and sidecar schema verified as data only; no deserialization.
            assert metadata['schema'] == protocol['checkpoint_schema']
            assert metadata['model_sha256'] == sha(cp/'model.zip')
            inputs[str(cp/'model.zip')] = metadata['model_sha256']
        smoke_init = read(folder.parent/'shared_heave_smoke_01/initial_training.json')
        assert init['initial_parameter_sha256'] != smoke_init['initial_parameter_sha256']
    return dict(passed=True,mode=mode,source_hash_count=len(pinned),source_mismatches=mismatches,
                seed=init['seed'],episode_seed_base=expected['episode_seed_base'],
                controls=count,native_calls=count*5,completed_episodes=len(completed),
                episode_transitions=[s['count'] for s in stats.values()],
                train_calls=len(updates),optimization_epochs=len(updates)*epochs,
                active_mapping_rows=active_rows,height_mask_rows=masked_rows,max_abs_policy_scalar=max_scalar,
                max_abs_wheel_residual=max_wheel_action,torque_protection_rows=protected_rows,
                leg_rate_limited_rows=leg_limited_rows,max_abs_requested_torque_nm=max_requested,
                max_abs_protected_torque_nm=max_protected,
                episodes=[dict(index=i,count=s['count'],return_sum=s['reward'],event_bonus_sum=s['bonus'],
                               flight_events=s['flight_events'],landing_events=s['landing_events'],
                               progress=s['progress']['progress'],peak_endpoint_net_gap_m=s['peak_net_gap_m']) for i,s in stats.items()],
                scope=['every recorded transition/action/raw schedule/reward and complete progress state rebuilt',
                       'every reset seed/curriculum, end boundary, final actor progress, PPO update and receipt checked',
                       'all 214 pinned inputs checked against current bytes; archived pre/post source claims retained'],
                limitations=['compact training logs lack all native qpos/contact entries and every-step observations',
                             '5-native-per-control established by recorded counter and sample summaries, not independent native replay',
                             'no independent geometry/COM reconstruction or checkpoint inference; no policy-success claim',
                             'global torque maximum bounds 80 Nm but compact logs cannot independently prove each wheel stayed within 12 Nm'],
                simulator_calls=0,controller_calls=0,model_calls=0,input_sha256=inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','formal'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = audit(args.directory,args.mode,args.preflight)
    result['audit_helper_sha256'] = sha(__file__)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','episodes','scope','limitations']},sort_keys=True))


if __name__ == '__main__':
    main()
