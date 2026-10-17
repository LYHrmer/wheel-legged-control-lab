"""Fixed B22/zero verification matrix; no simulation or learning imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'runtime/d1_b22/runtime_manifest_v27.json'
MANIFEST_SHA = 'd3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09'
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'
AUTHORED = ('scripts/b22_suite31_contract.py', 'scripts/run_b22_suite31.py',
            'scripts/read_b22_suite31.py', 'scripts/run_b22_worker31.py',
            'tests/test_b22_suite31.py', 'tests/test_b22_suite31_audit.py',
            'docs/b22_suite31.md')
ACTORS = ('zero', 'B')
# profile, canonical case, terrain, speed, yaw, seed, horizon, hold, release, mirror
CASES = (
    ('flat_0p6', 'flat_0p6', 'flat', .6, 0., 181201, 1600, (295, 695), 695, False),
    ('flat_1p6', 'flat_1p6', 'flat', 1.6, 0., 181202, 1600, (495, 895), 895, False),
    ('yaw_1p2', 'flat_1p2_yaw', 'flat', 1.2, .3, 181203, 1600, (415, 815), 815, False),
    ('bumps_0p4', 'bumps_0p4', 'bumps', .4, 0., 181204, 1600, (255, 655), 655, False),
    ('rough_0p35', 'rough_0p35', 'rough', .35, 0., 181205, 1600, (245, 645), 645, False),
    ('ramp_0p45_complete', 'ramp_0p45_complete', 'ramp', .45, 0., 181206, 1800,
     (600, 1000), 1355, False),
    ('yaw_1p2', 'flat_1p2_yaw', 'flat', 1.2, .3, 181403, 1600, (415, 815), 815, True),
)
LIMITS = {'controls': 25200, 'normal_native': 126000, 'compiler_native': 30,
          'logical_arm_records': 16, 'worker_sessions': 15,
          'total_wall_s': 3600, 'stage_wall_s': 900, 'reader_wall_s': 180,
          'start_effective_bytes': 6 << 30, 'reserve_effective_bytes': 3 << 30,
          'start_free_disk_bytes': 2 << 30, 'reserve_free_disk_bytes': 1 << 30}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
            size += len(data)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_pinned(path, expected, name):
    require(identity(path) == expected, 'helper identity changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stages():
    # The single C24 process compiles once, then records both logical arms.
    result = [{'name': 'step_pair', 'kind': 'step', 'controls': 2400, 'compiler_native': 2}]
    for row in CASES:
        profile, case, terrain, speed, yaw, seed, cap, hold, release, mirror = row
        for actor in ACTORS:
            result.append({'name': profile+('_mirror' if mirror else '')+'_'+actor,
                           'kind': 'script', 'profile': profile, 'case_id': case,
                           'terrain': terrain, 'speed_mps': speed, 'yaw_amplitude_rps': yaw,
                           'seed': seed, 'controls': cap, 'hold': list(hold),
                           'release_tick': release, 'mirror': mirror, 'actor': actor,
                           'compiler_native': 2})
    return result


def plan(output=None):
    return {'schema': 'b22-suite31-plan-v1',
            'owner_instruction': '把这些优化点，在这台电脑能力之内的都完成',
            'scope': 'seven fixed script B/zero pairs and one paired real 15mm step; no training',
            'output': None if output is None else str(Path(output).resolve()),
            'stages': stages(), 'limits': dict(LIMITS), 'retry_permitted': False,
            'reuse_closeout30_as_a_paired_case': False, 'keyboard_in_this_suite': False,
            'contribution_gate': 'unchanged frozen C18 original six; mirror and step separate'}


def raw_command(stage, tick):
    require(stage['kind'] == 'script' and type(tick) is int
            and 0 <= tick < stage['controls'], 'invalid fixed command request')
    speed = stage['speed_mps'] if 175 <= tick < stage['release_tick'] else 0.
    begin, end = stage['hold']
    yaw = 0.
    if stage['yaw_amplitude_rps'] and begin <= tick < end:
        yaw = stage['yaw_amplitude_rps'] * (1 if tick-begin < 100 or tick-begin >= 300 else -1)
        if stage['mirror']:
            yaw = -yaw
    return {'forward_velocity_mps': speed, 'lateral_velocity_mps': 0., 'yaw_rate_rps': yaw,
            'clearance_m': .455, 'jump_requested': False}


def model_limits(actor, controls, keys):
    limits = {key: 0 for key in keys}
    if actor == 'B':
        limits.update(load=1, torch_load=3, predict=controls+1)
    return limits


def source_identity(root=ROOT):
    paths = list(AUTHORED)
    subprocess.run(['git', 'ls-files', '--error-unmatch', *paths], cwd=root,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *paths], cwd=root, check=True)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                  text=True).strip(),
            'authored': {str(root/p): identity(root/p) for p in paths}}


def registered_config(root=ROOT):
    require(identity(root/MANIFEST)['sha256'] == MANIFEST_SHA, 'v27 manifest changed')
    config = read_json(root/MANIFEST)
    helper = root/'scripts/b22_runtime_v27.py'
    module = load_pinned(helper, config['source_hashes'][str(helper)], 'suite31_v27')
    checked, report = module.inspect(root)
    require(report['status'] == 'READY' and checked == config, 'v27 environment is not READY')
    return config, report


def validate_release(path, root=ROOT, *, check_sources=True):
    release = read_json(path)
    output = Path(release['plan']['output'])
    require(release.get('schema') == 'b22-suite31-release-v1'
            and release.get('plan') == plan(output)
            and release.get('v27_manifest') == {'bytes': 856947, 'sha256': MANIFEST_SHA}
            and Path(release.get('repository', '')) == root
            and output.parent == root/'results' and output.name.startswith('b22_suite31_'),
            'release scope/matrix/repository differs')
    require(identity(root/MANIFEST) == release['v27_manifest'], 'v27 manifest changed')
    config = read_json(root/MANIFEST)
    authored = release['source']['authored']
    require(set(authored) == {str(root/name) for name in AUTHORED}, 'authored source set differs')
    expected = {**config['source_hashes'], **authored, str(root/MANIFEST): identity(root/MANIFEST)}
    require(release.get('inputs') == expected, 'release input closure differs from fixed v27 and new sources')
    if check_sources:
        for name, expected in release['inputs'].items():
            require(Path(name).is_absolute() and not Path(name).is_symlink()
                    and identity(name) == expected, 'release input changed: '+name)
        require(all(release['inputs'].get(name) == expected
                    for name, expected in release['source']['authored'].items()),
                'new execution source closure missing')
    return release


def case_completion(stage, worker):
    """A legitimate policy task failure is data; an execution error stops the suite."""
    if stage['kind'] == 'step':
        require(worker.get('schema') == 'd1-true15mm-worker-24-v1'
                and worker.get('execution_complete') is True and worker.get('failure') is None
                and worker.get('cleanup_errors') == [] and worker.get('warnings') == []
                and worker.get('archive_failed') is False, 'step worker engineering failure')
        result, py = worker['result'], worker['python']
        count = result['completed_controls']
        require(result['full_comparison_executed'] is True and len(result['cases']) == 2,
                'step pair was not collected')
    else:
        require(worker.get('schema') == 'd1-c23-gui-worker-receipt-v1'
                and worker.get('failure') is None and worker.get('warnings') == []
                and worker.get('archive_failed') is False, 'script worker engineering failure')
        result = worker['result']
        py = result['python']
        count = result['completed_controls']
        require(result.get('warnings') == [] and result.get('input_reset_used') is False,
                'unexpected reset/warning')
        normal = count == stage['controls'] and result.get('terminated') is False
        failed = result.get('terminated') is True and result.get('truncated') is False
        require((normal and result.get('truncated') is True
                 and result.get('stop_reason') in ('time_limit', 'horizon_cap', 'control_cap'))
                or (failed and result.get('stop_reason') in
                    ('fall_or_low_clearance', 'nonwheel_ground_contact', 'course_map_boundary')),
                'unknown/partial task ending')
    require(type(count) is int and 1 <= count <= stage['controls']
            and py.get('control_completed') == count
            and py.get('native_attempted') == py.get('native_returned') == 5*count
            and py.get('native_failed') == 0 and py.get('forbidden_entries') == 0,
            'actual task/native accounting differs')
    return count
