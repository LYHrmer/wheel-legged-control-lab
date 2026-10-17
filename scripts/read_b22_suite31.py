"""Read saved suite31 records only. Never import a model, GUI or simulator."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location('suite31_contract', Path(__file__).with_name('b22_suite31_contract.py'))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)


class ForwardRows:
    """Single-pass sequence for the frozen reader's ascending five-row slices.

    Only the current slice and last record are retained, never an entire native
    archive. Length is checked against the real iterator, including extra rows.
    """
    def __init__(self, factory, length):
        c.require(type(length) is int and length > 0, 'invalid streaming row count')
        self.factory, self.length = factory, length
        self.iterator = None
        self.position, self.last = 0, None

    def __len__(self):
        return self.length

    def _next(self):
        if self.iterator is None:
            self.iterator = iter(self.factory())
        try:
            value = next(self.iterator)
        except StopIteration as error:
            raise ValueError('stream has fewer rows than its actual ledger') from error
        self.position += 1
        self.last = value
        if self.position == self.length:
            c.require(next(self.iterator, None) is None, 'stream has extra rows beyond its ledger')
        return value

    def __iter__(self):
        c.require(self.position == 0, 'stream cannot be iterated twice')
        for _ in range(self.length):
            yield self._next()

    def __getitem__(self, key):
        if key == -1:
            c.require(self.position == self.length, 'last row requested before the stream was checked')
            return self.last
        c.require(isinstance(key, slice) and key.step in (None, 1)
                  and key.start == self.position and self.position < key.stop <= self.length,
                  'native stream requires consecutive ascending slices')
        return [self._next() for _ in range(key.stop-self.position)]


def json_rows(paths, expected_counts=None):
    for index, path in enumerate(paths):
        count = 0
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                c.require(isinstance(row, dict), 'nonobject saved row')
                count += 1
                yield row
        if expected_counts is not None:
            c.require(count == expected_counts[index], 'archive block row count differs')


def helpers(release):
    forbidden = ('torch', 'mujoco', 'onnxruntime', 'stable_baselines3', 'glfw', 'gymnasium')
    c.require(not any(name in sys.modules for name in forbidden), 'saved reader was not launched cold')
    root = c.ROOT/'runtime/d1_b22/research'
    path = root/'continuation26/root26/reader26.py'
    base = c.load_pinned(path, release['inputs'][str(path)], 'suite31_frozen_reader26')
    for directory in (root/'continuation24', root/'continuation18', root/'course_impl08'):
        sys.path.insert(0, str(directory))
    step_path = root/'continuation24/read24.py'
    step = c.load_pinned(step_path, release['inputs'][str(step_path)], 'suite31_frozen_read24')
    score_path = root/'continuation18/score18.py'
    score = c.load_pinned(score_path, release['inputs'][str(score_path)], 'suite31_frozen_score18')
    # The source remains frozen. Only its file-reading seam is streamed; every
    # C24 numerical, geometry, contact, pairing and qualification check is retained.
    def controls(folder, blocks):
        paths, counts = [], []
        for index, block in enumerate(blocks):
            name = f'control_records_{index:04d}.jsonl.gz'
            c.require(block['file'] == name and 0 < block['rows'] <= 200
                      and block['archive_manifest'] == name+'.manifest.json', 'C24 control block order differs')
            path = folder/name
            step._committed(path)
            c.require(c.identity(path) == {k: block[k] for k in ('bytes', 'sha256')}, 'C24 control SHA differs')
            paths.append(path)
            counts.append(block['rows'])
        return ForwardRows(lambda: json_rows(paths, counts), sum(counts))
    def native(folder, guard):
        names = guard['native_files']
        c.require(len(names) == len(set(names)) and guard['archive_block_manifests']
                  == [name+'.manifest.json' for name in names], 'C24 native manifest order differs')
        paths = []
        for name in names:
            c.require(Path(name).name == name, 'unsafe native path')
            path = folder/name
            step._committed(path)
            paths.append(path)
        return ForwardRows(lambda: json_rows(paths), guard['native_returned'])
    step.control_rows, step.native_rows = controls, native
    return base, step, score


def numeric_identity(arrays, reset):
    fields = {}
    for name, value in arrays.items():
        fields[name] = {'dtype': value.dtype.str, 'shape': list(value.shape),
                        'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    encoded = json.dumps(reset, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'initial_arrays': fields, 'complete_control_reset_sha256': hashlib.sha256(encoded).hexdigest()}


def artifact_identities(run, base):
    payloads = {}
    for path in sorted(run.rglob('*.manifest.json')):
        manifest = base.committed(path.with_name(path.name[:-len('.manifest.json')]))
        for item in manifest['payloads']:
            payloads[str((path.parent/item['file']).relative_to(run))] = {k: item[k] for k in ('bytes', 'sha256')}
    for name in ('session.json', 'host_receipt.json', 'supervisor_receipt.json',
                 'preflight_ready.json', 'worker_receipt.json'):
        payloads[name] = c.identity(run/name)
    return payloads


def script_sources(run, stage, release, base):
    session = c.read_json(run/'session.json')
    worker = base.saved_document(run/'worker_receipt.json')
    count = c.case_completion(stage, worker)
    spec = {'profile': stage['profile'], 'actor': stage['actor'], 'seed': stage['seed'],
            'mirror': stage['mirror'], 'control_limit': stage['controls'], 'mode': 'script',
            'render': False, 'qualification': False, 'retry_permitted': False,
            'execution_contract_id': 'C31_B22_fixed_suite_v1', 'controller_variant': 'combined',
            'arm': stage['name'], 'output_directory': str(run)}
    c.require(all(session.get(k) == v for k, v in spec.items())
              and worker['arm'] == session['arm']
              and worker['execution_contract_id'] == session['execution_contract_id']
              and worker['session_identity'] == c.identity(run/'session.json'), 'script session identity differs')
    sources = session['source_hashes']
    config = c.read_json(c.ROOT/c.MANIFEST)
    expected_worker = str(c.ROOT/'scripts/run_b22_worker31.py')
    c.require(session['model_limits'] == c.model_limits(stage['actor'], stage['controls'], config['model_limit_keys']),
              'script planned model limits differ')
    c.require(session['argv'] == [expected_worker, '--session', str(run/'session.json'),
                                 '--output', str(run)], 'script worker argv/output differs')
    c.require(all(sources.get(k) == v for k, v in release['inputs'].items())
              and sources.get(str(c.ROOT/c.MANIFEST)) == c.identity(c.ROOT/c.MANIFEST), 'script v27 closure missing')
    for name, expected in sources.items():
        c.require(c.identity(name) == expected, 'script actual input changed: '+name)
    go = c.read_json(session['source_go_path'])
    c.require(c.identity(session['source_go_path']) == session['source_go_identity']
              and go['decision'] == 'GO' and go['worker'] == expected_worker
              and go['repository'] == str(c.ROOT) and go['shared_session'] == config['shared_session']
              and all(session.get(k) == v for k, v in go['arms'][session['arm']].items())
              and all(session.get(k) == v for k, v in go['shared_session'].items()), 'script GO binding differs')
    host, outer = c.read_json(run/'host_receipt.json'), c.read_json(run/'supervisor_receipt.json')
    ready = c.read_json(run/'preflight_ready.json')
    c.require(host['failure'] is None and host['exit_code'] == 0 and host['source_mismatches'] == []
              and host['arm'] == session['arm']
              and host['owned_no_orphans'] is True and host['reservation_closed'] is True
              and host['reserved_controls'] == stage['controls']
              and outer['failure'] is None and outer['exit_code'] == 0 and outer['cleanup']['remaining'] == {}
              and ready['session_sha256'] == c.identity(run/'session.json')['sha256']
              and ready['full_source_sha256_checked'] is True
              and ready['monotonic_s'] == host['execution_started_monotonic'], 'script host/readiness closure differs')
    result = worker['result']
    base.check_ledgers(result, session, count)
    c.require(result['control_step_caller_verified'] is True and worker['writer_partial'] == []
              and len(result['segment_receipts']) == 1, 'script runtime/archive incomplete')
    construction = base.saved_document(run/'construction_receipt.json')
    proof = construction['proof']
    c.require(proof['passed'] is True and proof['dso_path'] == session['library']
              and proof['dso_sha256'] == sources[session['library']]['sha256']
              and len(proof['jump_slots']) == 4 and all(item['passed'] is True for item in proof['jump_slots'])
              and construction['nominal_cache']['misses'] == 1, 'script native engine construction proof differs')
    reference = c.read_json(session['reference_construction_path'])
    c.require(construction['geometry_binding'] == reference['geometry_binding']
              and construction['compiled_geometry'] == reference['compiled_geometry'], 'script robot/course changed')
    origins = base.saved_document(run/'loaded_origins_final.json')
    c.require(all(path in sources for path in origins['module_origins'].values())
              and all(row['identity_verified'] is True and row['defining_source'] in sources
                      for row in origins['synthetic_module_aliases'].values()), 'actual module origin left closure')
    relative = {'input23': 'continuation23/gui23/input23.py', 'controller18': 'continuation18/controller18.py',
                'residual18': 'continuation18/residual18.py', 'world_upright_course_11': 'upright11/world_upright_course_11.py',
                'full_drive_env_08': 'course_impl08/full_drive_env_08.py',
                'gui13_bridge': 'continuation13/gui13_revision02/gui13_bridge.py',
                'async_course_renderer_12': 'gui12/async_course_renderer_12.py',
                'latest_frame_mailbox_12': 'gui12/latest_frame_mailbox_12.py',
                'recipes18': 'continuation18/recipes18.py', 'rl16_learning_11': 'rl11/rl16_learning_11.py',
                'run_rl16_training_08': 'run_rl16_training_08.py', 'state21': 'continuation21/state21.py',
                'worker22': 'continuation22/worker22.py'}
    expected_origins = {name: str(c.ROOT/'runtime/d1_b22/research'/path) for name, path in relative.items()}
    expected_origins['run_gui23'] = expected_worker
    expected_origins['engine_binding'] = session['binding_module']
    c.require(base.saved_document(run/'runtime_module_origins.json') == expected_origins,
              'actual controller/helper import path differs')
    for filename in ('mapped_libraries_before_load.json', 'mapped_libraries_final.json'):
        mapped = base.saved_document(run/filename)['paths']
        c.require(mapped and all(path in sources for path in mapped), 'actual DSO left closure')
    return session, worker, construction, count


def canonical_script(run, stage, session, worker, construction, count, base, step):
    import numpy as np
    physics = sys.modules['verify_course_e_08_03']
    binding = base.check_binding(construction['geometry_binding'])
    geometry = base.geometry_map(construction['compiled_geometry'], binding)
    folder = run/'episode_0'
    reset = base.saved_document(folder/'reset.json')
    initial, states = base.saved_arrays(folder/'initial_state.npz'), base.saved_arrays(folder/'states.npz')
    segment = base.saved_document(folder/'segment_receipt.json')
    base.check_reset(reset['control_reset_state'], initial)
    c.require(reset['seed'] == stage['seed'] and segment['completed_controls'] == count
              and segment['native_segment'] == worker['result']['segment_receipts'][0]
              and set(states) == set(base.STATE_KEYS)
              and all(states[k].shape == (count+1, *initial[k].shape)
                      and base.byte_equal(states[k][0], initial[k]) for k in base.STATE_KEYS)
              and base.close(states['time'], np.arange(count+1)*.01), 'script reset/complete clock differs')
    for section, keys in (('C_state', ('control_attempts', 'control_returns', 'construction_attempts', 'construction_returns')),
                          ('python', ('control_attempted', 'control_completed', 'native_attempted', 'native_returned'))):
        c.require(all(reset['before'][section][key] == reset['after'][section][key] for key in keys),
                  'script reset refunded native budget')
    c.require(reset['before']['python']['control_completed'] == 0
              and reset['before']['python']['native_returned'] == 0, 'script reset is not its first control')
    guard = segment['native_segment']
    c.require(guard['mode'] == 'heldout' and guard['record_valid'] is True
              and guard['native_attempted'] == guard['native_returned'] == 5*count
              and guard['failure'] is None and guard['archive_failure'] is None
              and guard['partial_native_interval'] is False
              and guard['force_sampling_performed'] is True and guard['full_contact_qualification_recorded'] is True,
              'script complete native sampling proof differs')
    paths = []
    c.require(len(guard['native_files']) == len(set(guard['native_files']))
              and guard['archive_block_manifests'] == [name+'.manifest.json' for name in guard['native_files']],
              'script native file order differs')
    for name in guard['native_files']:
        c.require(Path(name).name == name, 'unsafe native file')
        base.committed(folder/name)
        paths.append(folder/name)
    native = ForwardRows(lambda: json_rows(paths), 5*count)
    base.committed(folder/'controls.jsonl.gz')
    rows = ForwardRows(lambda: json_rows([folder/'controls.jsonl.gz']), count)
    q0 = initial['qpos']
    rotation = physics._rotation(q0[3:7])
    yaw0 = math.atan2(rotation[1, 0], rotation[0, 0])
    forward, lateral = np.asarray((math.cos(yaw0), math.sin(yaw0))), np.asarray((-math.sin(yaw0), math.cos(yaw0)))
    height0, normal0, _ = base._ground_hit(geometry, q0[0], q0[1])
    roll0, pitch0 = base._roll_pitch_deg(q0)
    canonical = {'case_id': stage['case_id'], 'actor': 'zero' if stage['actor'] == 'zero' else 'final_policy',
                 'seed': stage['seed'], 'mirror': stage['mirror'], 'terrain': stage['terrain'],
                 'task_schema': 'd1-course-world-upright-rl16-task-v1',
                 'reference_schema': 'd1-course-world-upright-roll-pitch-zero-v1',
                 'reward_schema': 'd1-course-world-upright-bodycom-yaw-terrain-action-torque-v1',
                 'completed_controls': count, 'terminated': worker['result']['terminated'],
                 'truncated': worker['result']['truncated'], 'stop_reason': worker['result']['stop_reason'],
                 'tick_index': np.arange(count), 'native_index': np.arange(5*count),
                 'initial_base_x_m': float(q0[0]), 'initial_base_y_m': float(q0[1]),
                 'initial_clearance_m': float(q0[2]-height0), 'initial_roll_deg': roll0, 'initial_pitch_deg': pitch0,
                 'initial_terrain_relative_tilt_deg': math.degrees(math.acos(float(np.clip(rotation[:, 2]@normal0, -1., 1.)))),
                 'warning_count': 0, 'geometry_invalid_count': 0, 'map_escape_count': 0,
                 'fall_count': int(worker['result']['stop_reason'] == 'fall_or_low_clearance')}
    array_names = ('applied_servo_vx_mps', 'applied_servo_yaw_rps', 'com_vx_mps', 'body_yaw_rate_rps',
                   'heading_rad', 'clearance_m', 'lateral_offset_m', 'forward_projection_m', 'motor_torque_nm',
                   'native_roll_deg', 'native_pitch_deg', 'native_terrain_relative_tilt_deg', 'native_clearance_m',
                   'native_forward_projection_m', 'native_base_x_m', 'native_base_y_m',
                   'native_nonwheel_ground_candidate_count')
    canonical.update({key: [] for key in array_names})
    metadata = reset['episode_metadata']
    c.require(all(metadata.get(key) == canonical[key] for key in ('task_schema', 'reference_schema', 'reward_schema'))
              and metadata['control_dt_s'] == .01 and metadata['normal_dt_s'] == .002
              and metadata['max_steps'] == stage['controls'] and metadata['observation_size'] == 99
              and metadata['policy_action_size'] == 16 and metadata['pure_test_components'] is False,
              'actual reset task/schema/timing differs')
    families, ramps = Counter(), Counter()
    vx = yaw = z = 0.
    pi, previous_vx, stopped = np.zeros(4), None, False
    for tick, row in enumerate(rows):
        info = row['info']
        c.require(row['control_index'] == row['episode_tick'] == tick and row['episode_index'] == 0
                  and row['actor'] == stage['actor']
                  and row['checkpoint_sha256'] == (c.MODEL_SHA if stage['actor'] == 'B' else None)
                  and row['policy_predict_called'] is (stage['actor'] == 'B')
                  and row['control_stages18'] == info['controller_record']
                  and info['completed_control_intervals'] == tick+1, 'script control identity differs')
        for key in base.STATE_KEYS:
            c.require(base.equal(row['pre_state'][key], states[key][tick])
                      and base.equal(row['post_state'][key], states[key][tick+1]), 'state/control join differs: '+key)
        c.require(base.equal(row['input_observation99'], states['observation'][tick])
                  and base.equal(row['action16'], row['raw_action16'])
                  and base.equal(row['action16'], row['policy_input_action']), 'policy 99D/16D join differs')
        action = np.asarray(row['action16'])
        c.require(action.shape == (16,) and np.isfinite(action).all()
                  and (stage['actor'] != 'zero' or not np.any(action)), 'wrong zero/action shape')
        raw = c.raw_command(stage, tick)
        c.require(row['raw_command'] == info['raw_operator_command'] == raw, 'fixed script command changed')
        vx += float(np.clip(raw['forward_velocity_mps']-vx, -.005, .005))
        yaw += float(np.clip(raw['yaw_rate_rps']-yaw, -.006, .006))
        consumed = info['consumed_command']
        c.require(base.close(consumed['forward_velocity_mps'], vx)
                  and base.close(consumed['yaw_rate_rps'], yaw)
                  and consumed['lateral_velocity_mps'] == 0. and consumed['clearance_m'] == .455
                  and consumed['jump_requested'] is False, 'script command servo differs')
        adapter = info['controller_record']
        after = False if vx != 0. else (True if previous_vx is not None and previous_vx != 0. else stopped)
        c.require(adapter['mode'] == 'eval' and adapter['controller_variant'] == 'combined'
                  and adapter['test_actor_probe'] is False and adapter['previous_servo_forward_mps'] == previous_vx
                  and adapter['stop_latched_before'] is stopped and adapter['stop_latched_after'] is after,
                  'script hidden stop memory differs')
        calc = base.check_controller(row, tick, pre_qpos=states['qpos'][tick], pre_qvel=states['qvel'][tick],
                                     pre_body_forward_mps=float(base._body_com_velocity(states['qpos'][tick], states['qvel'][tick], binding)[0]),
                                     servo_yaw_rps=yaw)
        actual = calc['calculation']
        c.require(base.close(actual['wheel_integral_before_nm'], pi)
                  and base.close(actual['wheel_common_reference_z_before_rad_s'], z)
                  and base.equal(actual['consumed_joint_position_rad'], states['qpos'][tick][binding['qpos_addresses']])
                  and base.equal(actual['consumed_joint_velocity_rad_s'], states['qvel'][tick][binding['dof_addresses']]),
                  'script actual joints/PI/filter memory differs')
        pi, z = calc['computed']['wheel_integral_after_nm'], calc['computed']['wheel_common_reference_z_after_rad_s']
        previous_vx, stopped = vx, after
        five = native[5*tick:5*tick+5]
        base.check_native(row, five, calc, binding, geometry, 5*tick)
        for entry in five:
            q = np.asarray(entry['after']['qpos'])
            ground, normal, _ = base._ground_hit(geometry, q[0], q[1])
            values = (entry['roll_deg'], entry['pitch_deg'],
                      math.degrees(math.acos(float(np.clip(physics._rotation(q[3:7])[:, 2]@normal, -1., 1.)))),
                      float(q[2]-ground), float(forward@(q[:2]-q0[:2])), float(q[0]), float(q[1]),
                      entry['nonwheel_contact_count'])
            for key, value in zip(array_names[9:], values):
                canonical[key].append(value)
            canonical['map_escape_count'] += int(abs(q[0]) > 11 or abs(q[1]) > 6)
            present, ramp_present = set(), set()
            for contact in entry['contacts']:
                loaded, family = base.check_contact(contact, geometry, binding)
                if loaded:
                    present.add(family)
                    if contact['terrain_geom_name'] in ('terrain_ramp_up', 'terrain_ramp_deck', 'terrain_ramp_down'):
                        ramp_present.add(contact['terrain_geom_name'])
            families.update(present)
            ramps.update(ramp_present)
        q, velocity = states['qpos'][tick+1], states['qvel'][tick+1]
        ground, _, gid = base._ground_hit(geometry, q[0], q[1])
        body_vx = float(base._body_com_velocity(q, velocity, binding)[0])
        heading = math.atan2(physics._rotation(q[3:7])[1, 0], physics._rotation(q[3:7])[0, 0])-yaw0
        metrics = info['metrics']
        c.require(base.close(metrics['body_com_vx_mps'], body_vx, atol=1e-6)
                  and base.close(metrics['body_yaw_rate_rps'], velocity[5], atol=1e-6)
                  and base.close(metrics['clearance_m'], q[2]-ground, atol=1e-6)
                  and metrics['ground_geom_id'] == gid and base.close(row['reward'], sum(info['reward_terms'].values())),
                  'post-step metric/reward arithmetic differs')
        c.require(all(info.get(key) == canonical[key] for key in ('task_schema', 'reference_schema', 'reward_schema')),
                  'per-control task identity changed')
        values = (vx, yaw, body_vx, float(velocity[5]), heading, float(q[2]-ground),
                  float(lateral@(q[:2]-q0[:2])), float(forward@(q[:2]-q0[:2])),
                  np.stack([trace['applied_nm'] for trace in row['native_actuator_traces']]).T)
        for key, value in zip(array_names[:9], values):
            canonical[key].append(value)
        c.require(tick == count-1 or (row['terminated'] is False and row['truncated'] is False), 'early done followed by more controls')
    last = rows[-1]
    c.require(last['terminated'] is canonical['terminated'] and last['truncated'] is canonical['truncated']
              and last['info']['terminal_reason'] == canonical['stop_reason'], 'actual terminal not bound to final record')
    canonical['positive_wheel_load_native_counts_by_family'] = dict(families)
    canonical['ramp_positive_wheel_load_native_counts_by_geom'] = {name: ramps[name] for name in ('terrain_ramp_up', 'terrain_ramp_deck', 'terrain_ramp_down')}
    if stage['terrain'] == 'ramp':
        # GUI26 has no endpoint geom cache. Reconstruct from the newly validated
        # C24 compiled robot binding; bridge checks the unchanged shared robot.
        step_run = run.parent/'step_pair'
        step_construction = step.document(step_run/'construction_receipt.json')
        compiled = step_construction['geometry_binding']
        step.robot_binding_bridge24(compiled, construction['geometry_binding'])
        reconstructed = step.reconstruct24(compiled['kinematics24'], states['qpos'][-1], states['qvel'][-1])
        bounds = step.collision_bounds24(compiled['kinematics24'], reconstructed,
                                        {int(k): v for k, v in compiled['wheel_index_by_body_id'].items()})
        canonical['wheel_collision_world_x_min_m_by_index'] = {
            str(index): min(item['minimum_world_m'][0] for item in bounds if item['wheel_index'] == index)
            for index in range(4)}
        canonical['ramp_geom_world_x_max_m_by_name'] = {
            item['name']: float(np.asarray(item['position_m'])[0]+np.abs(physics._rotation(item['quaternion_wxyz'])[0])@np.asarray(item['size_m']))
            for item in geometry.values() if item['name'] in ('terrain_ramp_up', 'terrain_ramp_deck', 'terrain_ramp_down')}
    pairing = numeric_identity(initial, reset['control_reset_state'])
    pairing['compiled_robot_course'] = hashlib.sha256(json.dumps(
        {'binding': construction['geometry_binding'], 'geometry': construction['compiled_geometry']},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return canonical, pairing


def read_stage(release_path, stage_name):
    release = c.validate_release(release_path)
    stage = next((item for item in release['plan']['stages'] if item['name'] == stage_name), None)
    c.require(stage is not None, 'unknown fixed stage')
    run = Path(release['plan']['output'])/stage_name
    base, step, scorer = helpers(release)
    common = {'schema': 'b22-suite31-stage-readback-v1', 'status': 'COMPLETE', 'stage': stage_name,
              'release_identity': c.identity(release_path), 'reader_identity': c.identity(Path(__file__)),
              'model_calls_by_reader': 0, 'physics_calls_by_reader': 0, 'retry_permitted': False}
    if stage['kind'] == 'step':
        session = c.read_json(run/'session.json')
        config = c.read_json(c.ROOT/c.MANIFEST)
        expected_worker = str(c.ROOT/'runtime/d1_b22/research/continuation24/worker24.py')
        c.require(session['output_directory'] == str(run)
                  and session['argv'] == [expected_worker, '--session', str(run/'session.json'), '--output', str(run)]
                  and all(session['source_hashes'].get(name) == value for name, value in release['inputs'].items())
                  and all(session.get(key) == value for key, value in config['shared_session'].items()),
                  'step output/worker/source identity differs')
        original = step.read_run(run)
        common.update(actual_counts={'controls': original['actual_completed_controls'],
                                     'normal_native': original['actual_normal_native_returns'],
                                     'compiler_native': original['actual_compiler_native_returns']},
                      cases=original['numeric_scores'], original_C24_readback=original,
                      artifacts=artifact_identities(run, base))
    else:
        session, worker, construction, count = script_sources(run, stage, release, base)
        if stage['terrain'] == 'ramp':
            evidence = c.read_json(run.parent/'step_pair_readback.json')
            c.require(evidence['status'] == 'COMPLETE' and evidence['release_identity'] == c.identity(release_path),
                      'fresh independently checked C24 kinematic evidence unavailable')
            for name, expected in evidence['artifacts'].items():
                c.require(c.identity(run.parent/'step_pair'/name) == expected, 'fresh C24 artifact changed: '+name)
        canonical, pairing = canonical_script(run, stage, session, worker, construction, count, base, step)
        scored = scorer.score_case(canonical, expected_seed=stage['seed'], mirror=stage['mirror'])
        common.update(actual_counts={'controls': count, 'normal_native': 5*count, 'compiler_native': 2},
                      cases=[scored], pairing_identity=pairing, artifacts=artifact_identities(run, base))
    return common


def aggregate(reports, pair_scorer):
    expected = c.stages()
    c.require(len(reports) == len(expected) and [r['stage'] for r in reports] == [s['name'] for s in expected],
              'suite matrix missing, duplicate or reordered')
    for report, stage in zip(reports, expected):
        c.require(report['status'] == 'COMPLETE'
                  and report['actual_counts']['normal_native'] == 5*report['actual_counts']['controls']
                  and 0 < report['actual_counts']['controls'] <= stage['controls']
                  and report['actual_counts']['compiler_native'] == 2, 'stage incomplete or over budget')
    indexed = {r['stage']: r for r in reports}
    original_six, mirrored = [], []
    for row in c.CASES:
        profile, _, _, _, _, _, _, _, _, mirror = row
        name = profile+('_mirror' if mirror else '')
        zero, policy = indexed[name+'_zero'], indexed[name+'_B']
        c.require(zero['pairing_identity'] == policy['pairing_identity'], 'B/zero initial complete state or robot pairing differs')
        scores = [zero['cases'][0], policy['cases'][0]]
        c.require([s['actor'] for s in scores] == ['zero', 'final_policy']
                  and all(s['mirror'] is mirror for s in scores), 'paired scoring actor/mirror differs')
        (mirrored if mirror else original_six).extend(scores)
    contribution = pair_scorer(original_six)
    seven_passed = all(indexed[(row[0]+('_mirror' if row[-1] else ''))+'_B']['cases'][0]['task_passed'] for row in c.CASES)
    step_passed = all(item['task_qualified'] for item in reports[0]['cases'])
    return {'status': 'COMPLETE', 'original_six_contribution': contribution,
            'RL_contribution_passed': contribution['RL_contribution_passed'],
            'script_cases': [r['cases'][0] for r in reports[1:]],
            'mirror_scores_separate': mirrored, 'real_15mm_scores_separate': reports[0]['cases'],
            'all_seven_B_script_tasks_passed': seven_passed,
            'both_real_15mm_tasks_passed': step_passed,
            'finite_suite_task_qualification_passed': seven_passed and step_passed,
            'scope': 'finite deterministic paired cases; no new training, statistical generalization or real-robot claim'}


def compare(release_path):
    release = c.validate_release(release_path)
    root = Path(release['plan']['output'])
    reports, identities = [], {}
    for stage in release['plan']['stages']:
        path = root/(stage['name']+'_readback.json')
        report = c.read_json(path)
        c.require(report['release_identity'] == c.identity(release_path)
                  and report['reader_identity'] == c.identity(Path(__file__)), 'stage reader/release identity differs')
        for name, expected in report['artifacts'].items():
            c.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe stage artifact path')
            c.require(c.identity(root/stage['name']/name) == expected, 'raw stage artifact changed: '+name)
        identities[path.name] = c.identity(path)
        reports.append(report)
    _, _, scorer = helpers(release)
    return {'schema': 'b22-suite31-comparison-v1', 'release_identity': c.identity(release_path),
            'stage_reports': identities, **aggregate(reports, scorer.score_pairs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stage')
    group.add_argument('--compare', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.release) if args.compare else read_stage(args.release, args.stage)
    c.write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
