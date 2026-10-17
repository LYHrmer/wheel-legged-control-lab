"""Synthetic receipts/arrays and fake processes only; no model, GUI or physics."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


reader, runner = module('verify_b22_closeout30'), module('run_b22_closeout30')


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def archive(path):
    save(path.with_name(path.name+'.manifest.json'), {'schema': 'd1-archive-transaction-13-v1',
         'payloads': [{'file': path.name, **reader.identity(path)}]})


@pytest.fixture
def completed(tmp_path):
    output = tmp_path/'session'
    launch = tmp_path/'session_launch'
    output.mkdir(); launch.mkdir(); (output/'episode_0').mkdir()
    limits = {'load': 1, 'torch_load': 3, 'predict': 1601,
              **dict.fromkeys(('train', 'learn', 'save', 'backward', 'forward', 'evaluate_actions', 'predict_values'), 0)}
    spec = {**reader.expected_spec(), 'output_directory': str(output), 'model_limits': limits}
    config = {'worker': str(ROOT/'runtime/d1_b22/research/continuation26/gui26/run_gui26.py'),
              'source_hashes': {str(ROOT/name): reader.identity(ROOT/name) for name in ('scripts/play_b22_v27.py', reader.SCORE_SOURCE)}}
    sources = dict(config['source_hashes']); sources[str(ROOT/reader.MANIFEST)] = reader.identity(ROOT/reader.MANIFEST)
    go = launch/'source_go.json'; request = launch/'request.json'
    save(go, {'repository': str(ROOT), 'worker': config['worker'], 'inputs': sources, 'arms': {'user_script': spec}})
    save(request, {'source_go_path': str(go), 'spec': spec})
    session = {**spec, 'source_go_path': str(go), 'source_go_identity': reader.identity(go),
               'source_hashes': {**sources, str(go): reader.identity(go), str(request): reader.identity(request)}}
    save(output/'session.json', session)
    session_identity = reader.identity(output/'session.json')
    save(output/'preflight_ready.json', {'monotonic_s': 123., 'session_sha256': session_identity['sha256'], 'full_source_sha256_checked': True})
    save(output/'host_receipt.json', {'execution_started_monotonic': 123., 'failure': None, 'exit_code': 0,
         'source_mismatches': [], 'owned_no_orphans': True, 'reservation_closed': True, 'reserved_controls': 1600})
    save(output/'supervisor_receipt.json', {'failure': None, 'exit_code': 0, 'cleanup': {'remaining': {}}})
    result = {'completed_controls': 1600, 'python': {'native_returned': 8000, 'native_attempted': 8000,
              'native_failed': 0, 'control_completed': 1600, 'forbidden_entries': 0},
              'C_final': {'construction_attempts': 2, 'construction_returns': 2, 'control_attempts': 8000, 'control_returns': 8000, 'violations': 0},
              'model_calls': {'limits': limits, 'counts': {key: {'attempted': value, 'returned': value} for key, value in limits.items() if value}},
              'policy_predictions': 1600, 'input_reset_used': False, 'terminated': False, 'stop_reason': 'time_limit', 'warnings': []}
    save(output/'worker_receipt.json', {'session_identity': session_identity, 'failure': None, 'archive_failed': False,
         'execution_contract_id': spec['execution_contract_id'], 'result': result, 'warnings': []})
    save(launch/'v27_completion.json', {'status': 'PASS', 'errors': [], 'observed_counts': {'completed_controls': 1600, 'native_returned': 8000},
         'host_returncode': 0, 'qualification_granted': False})
    save(output/'window_ready.json', {'x11_window': 12345, 'session_sha256': session_identity['sha256']})
    frames = []
    for name, tick in (('initial', 0), ('drive', 800), ('final', 1600)):
        path = output/f'frame_{name}.png'
        picture = Image.new('RGB', (800, 500), (17, 23, tick % 256))
        picture.save(path, format='PNG')
        frames.append({'capture': {'path': str(path), 'png_sha256': reader.identity(path)['sha256'], 'bytes': path.stat().st_size,
                                  'pixels_sha256': hashlib.sha256(picture.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes()).hexdigest(),
                                  'control_index': tick, 'sim_time_s': tick*.01}})
    save(output/'render_frames.json', frames)
    raw = [{'forward_velocity_mps': 1.6 if 175 <= tick < 895 else 0., 'lateral_velocity_mps': 0.,
            'yaw_rate_rps': 0., 'clearance_m': .455, 'jump_requested': False} for tick in range(1600)]
    command = {'mode': 'script', 'profile': 'flat_1p6', 'actor': 'B', 'seed': 181202, 'terrain': 'flat',
               'control_limit': 1600, 'raw_commands': raw,
               'raw_command_sha256': hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    save(output/'command_identity.json', command)
    qpos = np.zeros((1601, 7)); qpos[:, 2] = .455; qpos[:, 3] = 1.
    np.savez(output/'episode_0/states.npz', qpos=qpos, qvel=np.zeros((1601, 6)), ctrl=np.zeros((1601, 16)),
             qacc_warmstart=np.zeros((1601, 6)), observation=np.zeros((1601, 99)), time=np.arange(1601)*.01)
    rows = [{'control_index': tick, 'episode_tick': tick, 'episode_index': 0, 'raw_command': raw[tick],
             'actor': 'B', 'checkpoint_sha256': reader.MODEL_SHA, 'policy_predict_called': True,
             'input_observation99': [0.]*99, 'raw_action16': [0.]*16, 'native_actuator_traces': [{}]*5,
             'terminated': False, 'truncated': tick == 1599,
             'info': {'metrics': {'body_com_vx_mps': raw[tick]['forward_velocity_mps'], 'body_yaw_rate_rps': 0., 'clearance_m': .455}}}
            for tick in range(1600)]
    def write_rows():
        with gzip.open(output/'episode_0/controls.jsonl.gz', 'wt') as stream:
            for row in rows: stream.write(json.dumps(row)+'\n')
        archive(output/'episode_0/controls.jsonl.gz')
    write_rows()
    for name in ('command_identity.json', 'worker_receipt.json', 'render_frames.json', 'episode_0/states.npz'):
        archive(output/name)
    return SimpleNamespace(output=output, launch=launch, config=config, rows=rows, write_rows=write_rows,
                           verify=lambda: reader.verify(output, manifest_value=config))


def test_saved_complete_demo_and_lazy_zero_api_entries_pass(completed):
    result = completed.verify()
    assert result['status'] == 'PASS', result['errors']
    assert result['qualification_granted'] is False
    assert result['observed_counts'] == {'completed_controls': 1600, 'native_returned': 8000, 'compiler_attempted': 2, 'compiler_returned': 2}
    assert result['metrics']['hold_mean_com_vx_mps'] == pytest.approx(1.6)


@pytest.mark.parametrize('fault', ['ready_missing', 'empty_success', 'compiler', 'train', 'load_missing', 'orphans', 'screenshot',
                                  'truncated_png_rehashed', 'pixels_sha', 'progress', 'manifest'])
def test_bad_receipt_gui_and_archive_are_rejected(completed, fault):
    output = completed.output
    worker_path = output/'worker_receipt.json'; worker = reader.read_json(worker_path)
    if fault == 'ready_missing': (output/'preflight_ready.json').unlink()
    if fault == 'empty_success': worker['result']['completed_controls'] = 0
    if fault == 'compiler': worker['result']['C_final']['construction_attempts'] = 3
    if fault == 'train': worker['result']['model_calls']['counts']['train'] = {'attempted': 1, 'returned': 1}
    if fault == 'load_missing': del worker['result']['model_calls']['counts']['load']
    if fault == 'orphans':
        path = output/'host_receipt.json'; data = reader.read_json(path); data['owned_no_orphans'] = False; save(path, data)
    if fault == 'screenshot': (output/'frame_drive.png').write_bytes(b'wrong')
    if fault in ('truncated_png_rehashed', 'pixels_sha'):
        path = output/'render_frames.json'; data = reader.read_json(path)
        if fault == 'truncated_png_rehashed':
            picture = output/'frame_drive.png'; picture.write_bytes(picture.read_bytes()[:24])
            data[1]['capture']['png_sha256'] = reader.identity(picture)['sha256']
            data[1]['capture']['bytes'] = picture.stat().st_size
        else:
            data[1]['capture']['pixels_sha256'] = '0'*64
        save(path, data); archive(path)
    if fault == 'progress':
        path = output/'render_frames.json'; data = reader.read_json(path); data[1]['capture']['control_index'] = 0; save(path, data); archive(path)
    if fault == 'manifest': (output/'episode_0/states.npz.manifest.json').unlink()
    save(worker_path, worker); archive(worker_path)
    assert completed.verify()['status'] == 'FAIL'


@pytest.mark.parametrize('fault', ['missing_tick', 'wrong_command', 'missing_native', 'wrong_checkpoint', 'observation', 'slow', 'not_stopped'])
def test_rehashed_bad_control_records_do_not_pass(completed, fault):
    row = completed.rows[600]
    if fault == 'missing_tick': row['control_index'] = 599
    if fault == 'wrong_command': row['raw_command'] = dict(row['raw_command'], forward_velocity_mps=.8)
    if fault == 'missing_native': row['native_actuator_traces'].pop()
    if fault == 'wrong_checkpoint': row['checkpoint_sha256'] = '0'*64
    if fault == 'observation': row['input_observation99'][0] = 1.
    if fault == 'slow':
        for item in completed.rows[495:895]: item['info']['metrics']['body_com_vx_mps'] = 0.
    if fault == 'not_stopped':
        for item in completed.rows[1500:]: item['info']['metrics']['body_com_vx_mps'] = .1
    completed.write_rows()
    assert completed.verify()['status'] == 'FAIL'


def test_rehashed_endpoint_missing_clock_is_rejected(completed):
    path = completed.output/'episode_0/states.npz'
    with np.load(path, allow_pickle=False) as loaded: data = {key: loaded[key] for key in loaded.files}
    data['time'][400] += .001
    np.savez(path, **data); archive(path)
    assert completed.verify()['status'] == 'FAIL'


class FakeProcess:
    def __init__(self, *, running=2, code=0):
        self.left, self.code, self.returncode = running, code, None

    def poll(self):
        if self.left:
            self.left -= 1
        else:
            self.returncode = self.code
        return self.returncode


@pytest.fixture
def supervision(tmp_path, monkeypatch):
    output = tmp_path/'one_session'
    monkeypatch.setattr(runner.readback, 'OUTPUT', output)
    now, spawned, cleaned = [0.], [], []
    def sleep(seconds): now[0] += seconds
    def launch(argv, **kwargs):
        assert '--headless' not in argv and argv[-1] == str(output)
        process = FakeProcess(); spawned.append(process); return process
    def cleanup(root, process):
        cleaned.append(process)
        if process is not None and process.returncode is None: process.returncode = -15
        return {'remaining': {}, 'signalled_identities': {}, 'elapsed_s': 0.}
    host = SimpleNamespace(subreaper=lambda: None, clean_descendants=cleanup)
    defaults = {'memory_reader': lambda: {'effective_available_bytes': 7*1024**3}, 'process_factory': launch,
                'host_loader': lambda: host, 'verifier': lambda output: {'status': 'PASS'},
                'provenance': lambda: {'git_commit': 'f'*40}, 'clock': lambda: now[0], 'sleeper': sleep}
    return SimpleNamespace(output=output, now=now, spawned=spawned, cleaned=cleaned, host=host,
                           run=lambda **kwargs: runner.run(output, **{**defaults, **kwargs}))


def test_fake_supervisor_completes_and_refuses_second_attempt(supervision):
    result = supervision.run()
    assert result['status'] == 'PASS' and len(supervision.spawned) == 1
    assert supervision.cleaned == supervision.spawned
    with pytest.raises(ValueError, match='one-shot'): supervision.run()


@pytest.mark.parametrize('fault', ['admission', 'reserve', 'memory_unknown', 'wall', 'exit', 'orphans', 'readback', 'source_changed'])
def test_fake_supervisor_stops_without_retry_or_forged_success(supervision, fault):
    options = {}
    samples = [0]
    def memory():
        samples[0] += 1
        return {'effective_available_bytes': (5 if fault == 'admission' else 7 if samples[0] == 1 else 2)*1024**3}
    if fault in ('admission', 'reserve'): options['memory_reader'] = memory
    if fault == 'memory_unknown': options['memory_reader'] = lambda: {'effective_available_bytes': None}
    if fault == 'wall': options['sleeper'] = lambda seconds: supervision.now.__setitem__(0, 900.)
    if fault == 'exit': options['process_factory'] = lambda *args, **kwargs: FakeProcess(code=1)
    if fault == 'orphans': supervision.host.clean_descendants = lambda *args: {'remaining': {12: 44}}
    if fault == 'readback': options['verifier'] = lambda path: {'status': 'FAIL', 'observed_counts': None}
    if fault == 'source_changed':
        count = [0]
        def source(): count[0] += 1; return {'version': count[0]}
        options['provenance'] = source
    result = supervision.run(**options)
    assert result['status'] == 'FAIL' and result['failure']
    assert len(supervision.spawned) <= 1
    if fault in ('admission', 'memory_unknown'): assert not supervision.spawned


def test_effective_ram_uses_tightest_ancestor_and_not_swap(tmp_path):
    proc, groups = tmp_path/'proc', tmp_path/'groups'
    (proc/'self').mkdir(parents=True); (groups/'a/b').mkdir(parents=True)
    (proc/'meminfo').write_text('MemAvailable: 8388608 kB\nSwapFree: 999999999 kB\n')
    (proc/'self/cgroup').write_text('0::/a/b\n')
    (groups/'memory.max').write_text('max')
    (groups/'a/memory.max').write_text(str(10*1024**3)); (groups/'a/memory.current').write_text(str(7*1024**3))
    (groups/'a/b/memory.max').write_text(str(8*1024**3)); (groups/'a/b/memory.current').write_text(str(2*1024**3))
    assert runner.read_memory(proc, groups)['effective_available_bytes'] == 3*1024**3
    (groups/'a/memory.current').unlink()
    with pytest.raises(OSError): runner.read_memory(proc, groups)


def test_repeated_stop_signals_cannot_interrupt_owned_cleanup(supervision):
    def cleanup(*args):
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        return {'remaining': {}}
    supervision.host.clean_descendants = cleanup
    assert supervision.run()['status'] == 'PASS'


def test_post_run_git_failure_is_recorded_without_losing_report(supervision):
    calls = [0]
    def provenance():
        calls[0] += 1
        if calls[0] > 1: raise subprocess.CalledProcessError(1, ['git'])
        return {'git_commit': 'f'*40}
    result = supervision.run(provenance=provenance)
    assert result['status'] == 'FAIL' and 'CalledProcessError' in result['failure']
    assert (supervision.output.with_name(supervision.output.name+'_closeout')/'report.json').exists()
