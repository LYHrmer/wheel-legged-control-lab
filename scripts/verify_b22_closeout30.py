"""Read one saved v27 flat_1p6 GUI session; never load a model or run physics."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'runtime/d1_b22/runtime_manifest_v27.json'
MANIFEST_SHA = 'd3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09'
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'
OUTPUT = ROOT/'results/b22_v27_closeout_20261018_01'
SCORE_SOURCE = 'runtime/d1_b22/research/course_impl08/rl16_heldout_score_08.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def manifest():
    require(identity(ROOT/MANIFEST)['sha256'] == MANIFEST_SHA, 'v27 manifest identity changed')
    return read_json(ROOT/MANIFEST)


def pinned_module(name, path, expected):
    require(identity(path) == expected, 'pinned helper identity changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_spec():
    return {'schema': 'd1-c23-gui-worker-v1', 'mode': 'script', 'render': True,
            'profile': 'flat_1p6', 'actor': 'B', 'control_limit': 1600, 'seconds': 16.,
            'seed': 181202, 'mirror': False, 'soft_s': 180, 'close_s': 235, 'hard_s': 240,
            'qualification': False, 'frame_limit': 1000, 'require_event_driver': False,
            'execution_contract_id': 'C26_B22_user_session_v1'}


def check_archives(output):
    count, identities = 0, {}
    for path in sorted(output.rglob('*.manifest.json')):
        record = read_json(path)
        require(record.get('schema') == 'd1-archive-transaction-13-v1' and record.get('payloads'),
                'invalid atomic archive manifest: '+str(path))
        for item in record['payloads']:
            filename = item['file']
            require(Path(filename).name == filename, 'archive payload leaves its folder')
            payload = path.parent/filename
            actual = identity(payload)
            require({'file': filename, **actual} == item, 'archive payload SHA/bytes mismatch: '+str(payload))
            identities[str(payload.relative_to(output))] = actual
        count += 1
    for name in ('command_identity.json', 'worker_receipt.json', 'render_frames.json',
                 'episode_0/controls.jsonl.gz', 'episode_0/states.npz'):
        require(name in identities, 'missing committed archive payload: '+name)
    return count, identities


def check_frames(output, session_sha):
    from PIL import Image
    ready = read_json(output/'window_ready.json')
    require(type(ready.get('x11_window')) is int and ready['x11_window'] > 0
            and ready.get('session_sha256') == session_sha, 'real X11 window is not bound to this session')
    frames = read_json(output/'render_frames.json')
    require(isinstance(frames, list) and 3 <= len(frames) <= 1000, 'render frame count is missing or outside cap')
    captures = {}
    for frame in frames:
        capture = frame.get('capture')
        if capture:
            path = Path(capture['path'])
            require(path.parent == output, 'capture path leaves this session')
            actual = identity(path)
            require(actual['sha256'] == capture.get('png_sha256') and actual['bytes'] == capture.get('bytes'),
                    'rendered PNG identity mismatch')
            with Image.open(path) as picture:
                require(picture.format == 'PNG' and picture.size == (800, 500) and picture.mode == 'RGB',
                        'rendered PNG has a wrong signature/viewport')
                pixels = picture.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes()
            require(any(pixels) and hashlib.sha256(pixels).hexdigest() == capture.get('pixels_sha256'),
                    'decoded GUI pixels are empty or differ from render capture')
            captures[path.name] = {**actual, 'control_index': capture.get('control_index'),
                                   'sim_time_s': capture.get('sim_time_s')}
    require(set(captures) >= {'frame_initial.png', 'frame_drive.png', 'frame_final.png'}, 'initial/drive/final GUI captures missing')
    require(captures['frame_initial.png']['control_index'] < captures['frame_drive.png']['control_index']
            <= captures['frame_final.png']['control_index'], 'capture progress did not advance')
    return {'x11_window': ready['x11_window'], 'render_frames': len(frames), 'captures': captures}


def check_controls(output):
    import numpy as np
    command = read_json(output/'command_identity.json')
    require(all(command.get(key) == value for key, value in {
        'mode': 'script', 'profile': 'flat_1p6', 'actor': 'B', 'seed': 181202,
        'terrain': 'flat', 'control_limit': 1600}.items()), 'fixed task identity differs')
    raw = command.get('raw_commands')
    require(isinstance(raw, list) and len(raw) == 1600, 'fixed command table is incomplete')
    encoded = json.dumps(raw, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(hashlib.sha256(encoded).hexdigest() == command.get('raw_command_sha256'), 'raw command SHA differs')
    with np.load(output/'episode_0/states.npz', allow_pickle=False) as archive:
        states = {key: archive[key] for key in archive.files}
    for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'observation'):
        array = states[name]
        require(array.ndim == 2 and array.shape[0] == 1601 and bool(np.isfinite(array).all()), 'state endpoints missing/nonfinite: '+name)
    require(states['observation'].shape == (1601, 99), 'actor observation endpoint shape differs')
    require(states['time'].shape == (1601,) and np.allclose(states['time'], np.arange(1601)*.01, rtol=0., atol=1e-10),
            'control endpoint clock is incomplete')
    vx, wz, clearance = [], [], []
    with gzip.open(output/'episode_0/controls.jsonl.gz', 'rt') as stream:
        for tick, line in enumerate(stream):
            require(tick < 1600, 'control archive exceeds cap')
            row = json.loads(line)
            expected = {'forward_velocity_mps': 1.6 if 175 <= tick < 895 else 0.,
                        'lateral_velocity_mps': 0., 'yaw_rate_rps': 0., 'clearance_m': .455, 'jump_requested': False}
            require(raw[tick] == expected and row.get('raw_command') == expected, 'command/control table differs at tick '+str(tick))
            require(row.get('control_index') == tick and row.get('episode_tick') == tick and row.get('episode_index') == 0,
                    'control indices skipped, duplicated, or reset')
            require(row.get('actor') == 'B' and row.get('checkpoint_sha256') == MODEL_SHA
                    and row.get('policy_predict_called') is True, 'control actor/checkpoint differs')
            require(np.array_equal(np.asarray(row['input_observation99']), states['observation'][tick]), 'control observation breaks endpoint chain')
            action = np.asarray(row['raw_action16'])
            require(action.shape == (16,) and bool(np.isfinite(action).all()) and bool((np.abs(action) <= 1.).all()), 'invalid policy action')
            require(len(row.get('native_actuator_traces', [])) == 5, 'control has fewer than five native actuator traces')
            require(row.get('terminated') is False and (tick == 1599 or row.get('truncated') is False), 'fixed task ended early')
            metrics = row['info']['metrics']
            sample = [metrics['body_com_vx_mps'], metrics['body_yaw_rate_rps'], metrics['clearance_m']]
            require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in sample), 'nonfinite task metrics')
            vx.append(sample[0]); wz.append(sample[1]); clearance.append(sample[2])
    require(len(vx) == 1600, 'control archive is incomplete')
    hold, final = np.asarray(vx[495:895]), np.asarray(vx[1500:1600])
    result = {'hold_window': [495, 895], 'hold_ticks': len(hold), 'hold_mean_com_vx_mps': float(np.mean(hold)),
              'hold_rms_error_mps': float(np.sqrt(np.mean((hold-1.6)**2))),
              'hold_fraction_above_1p5': float(np.sum(hold > 1.5)/400),
              'final_window': [1500, 1600], 'final_ticks': len(final), 'final_mean_abs_com_vx_mps': float(np.mean(np.abs(final))),
              'final_mean_abs_yaw_rps': float(np.mean(np.abs(wz[1500:1600]))),
              'final_clearance_std_m': float(np.std(clearance[1500:1600], ddof=0)),
              'min_control_clearance_m': min(clearance)}
    require(1.52 <= result['hold_mean_com_vx_mps'] <= 1.68 and result['hold_rms_error_mps'] <= .08
            and result['hold_fraction_above_1p5'] >= .9, 'original flat_1p6 speed checks failed')
    require(result['final_mean_abs_com_vx_mps'] <= .05 and result['final_mean_abs_yaw_rps'] <= .05
            and result['final_clearance_std_m'] <= .02 and result['min_control_clearance_m'] >= .28,
            'original final-window/clearance checks failed')
    return result


def verify(output, *, manifest_value=None):
    output = Path(output).resolve()
    launch = output.with_name(output.name+'_launch')
    report = {'schema': 'b22-v27-closeout-readback30-v1', 'status': 'FAIL', 'errors': [],
              'output': str(output), 'observed_counts': None, 'qualification_granted': False,
              'scope': 'one flat_1p6 GUI/control demonstration; no new six-task, RL superiority or real-robot qualification'}
    try:
        config = manifest() if manifest_value is None else manifest_value
        require(identity(ROOT/SCORE_SOURCE) == config['source_hashes'][str(ROOT/SCORE_SOURCE)], 'original numeric check source changed')
        session, request = read_json(output/'session.json'), read_json(launch/'request.json')
        spec = request['spec']
        require(all(spec.get(key) == value and session.get(key) == value for key, value in expected_spec().items()), 'fixed session specification differs')
        require(spec.get('output_directory') == session.get('output_directory') == str(output), 'output identity differs')
        go_path = launch/'source_go.json'
        require(request.get('source_go_path') == str(go_path) and session.get('source_go_path') == str(go_path)
                and session.get('source_go_identity') == identity(go_path), 'source GO is not bound to session')
        sources = dict(config['source_hashes'])
        sources[str(ROOT/MANIFEST)] = identity(ROOT/MANIFEST)
        go = read_json(go_path)
        require(go.get('repository') == str(ROOT) and go.get('worker') == config['worker'] and go.get('inputs') == sources
                and go.get('arms') == {'user_script': spec}, 'source GO differs from v27')
        require(session.get('source_hashes') == {**sources, str(go_path): identity(go_path), str(launch/'request.json'): identity(launch/'request.json')}, 'session source closure differs')
        launcher_path = ROOT/'scripts/play_b22_v27.py'
        launcher = pinned_module('closeout_v27_launcher', launcher_path, config['source_hashes'][str(launcher_path)])
        stored = read_json(launch/'v27_completion.json')
        fresh = launcher.completion(output, spec, stored['host_returncode'])
        require(stored == fresh and fresh['status'] == 'PASS', 'v27 completion receipts did not pass independent readback')
        worker = read_json(output/'worker_receipt.json')
        result = worker['result']
        native = result['C_final']
        report['observed_counts'] = {**fresh['observed_counts'], 'compiler_attempted': native.get('construction_attempts'),
                                     'compiler_returned': native.get('construction_returns')}
        require(type(native.get('construction_attempts')) is int and 0 <= native['construction_attempts'] <= 2
                and native.get('construction_returns') == native['construction_attempts']
                and native.get('control_attempts') == native.get('control_returns') == 8000 and native.get('violations') == 0,
                'C normal/compiler counts differ')
        calls = result['model_calls']['counts']
        expected_calls = {'load': 1, 'torch_load': 3, 'predict': 1601,
                          **dict.fromkeys(('train', 'learn', 'save', 'backward', 'forward', 'evaluate_actions', 'predict_values'), 0)}
        require(result['model_calls'].get('limits') == spec.get('model_limits') == expected_calls, 'model API limits differ')
        for name, expected in expected_calls.items():
            # The frozen ModelCounter creates entries lazily before a call;
            # absent zero-limit entries mean no call, not a missing ledger.
            observed = calls.get(name, {'attempted': 0, 'returned': 0} if expected == 0 else {})
            require(observed.get('attempted') == observed.get('returned') == expected,
                    'model API counts differ: '+name)
        require(result.get('policy_predictions') == 1600 and result.get('input_reset_used') is False
                and result.get('terminated') is False and result.get('stop_reason') in ('control_cap', 'time_limit', 'horizon_cap')
                and worker.get('warnings') == result.get('warnings') == [], 'control session stopped, reset, or warned')
        archive_count, payloads = check_archives(output)
        gui = check_frames(output, identity(output/'session.json')['sha256'])
        metrics = check_controls(output)
        report.update(status='PASS', archives_checked=archive_count, payload_sha256=payloads, gui=gui, metrics=metrics,
                      metric_source={SCORE_SOURCE: identity(ROOT/SCORE_SOURCE)},
                      session_identity=identity(output/'session.json'), v27_manifest_identity=identity(ROOT/MANIFEST))
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, EOFError) as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.report.exists(), 'readback report already exists')
    result = verify(args.output)
    write_json(args.report, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
