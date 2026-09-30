"""Launch the qualified B22/C18 policy GUI with a fresh bounded result folder."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'runtime/d1_b22'
PROFILES = ('flat_0p6', 'flat_1p6', 'yaw_1p2', 'bumps_0p4',
            'rough_0p35', 'ramp_0p45_complete')
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify bytes; zero model/physics calls')
    parser.add_argument('--mode', choices=('script', 'keyboard'), default='script')
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--actor', choices=('B', 'zero'), default='B')
    parser.add_argument('--mirror', action='store_true', help='mirror the fixed yaw script')
    parser.add_argument('--seconds', type=int, default=60, help='keyboard session cap, 12..60 s')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path, help='must not exist; otherwise a unique directory is created')
    args = parser.parse_args()
    profile = args.profile or ('flat_1p6' if args.mode == 'script' else 'yaw_1p2')
    if args.mode == 'keyboard' and (profile != 'yaw_1p2' or args.headless or args.mirror):
        parser.error('keyboard uses the bounded yaw_1p2 window; no headless or mirror mode')
    if args.mirror and profile != 'yaw_1p2':
        parser.error('--mirror only applies to yaw_1p2')
    if not 12 <= args.seconds <= 60:
        parser.error('--seconds must be in 12..60')
    config_path = BUNDLE / 'runtime_manifest_v26.json'
    if not config_path.is_file():
        parser.error('B22 runtime bundle has not been sealed; see current delivery report')
    config = json.loads(config_path.read_text())
    sources = config['source_hashes']
    changed = [name for name, expected in sources.items()
               if not Path(name).is_file() or identity(Path(name)) != expected]
    if changed:
        parser.error('runtime source/dependency mismatch: ' + ', '.join(changed[:3]))
    if identity(BUNDLE / 'checkpoint/final_model.zip')['sha256'] != MODEL_SHA:
        parser.error('this entry accepts only the frozen B22 checkpoint')
    if args.check:
        print(json.dumps({'status': 'ready', 'checkpoint': MODEL_SHA, 'controller': 'C18 combined',
                          'observation': 99, 'action': 16, 'verified_files': len(sources),
                          'model_loads': 0, 'controls': 0, 'normal_native': 0,
                          'profiles': PROFILES, 'keyboard_profile': 'yaw_1p2'}, ensure_ascii=False, indent=2))
        return 0
    horizon = (1800 if profile == 'ramp_0p45_complete' else 1600) if args.mode == 'script' else 100 * args.seconds
    output = args.output or ROOT / 'results' / (
        'b22_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + str(time.time_ns() % 1000000000))
    output = output.resolve()
    if output.exists():
        parser.error('output exists; omit --output for an automatically unique directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    launch = output.with_name(output.name + '_launch')
    launch.mkdir(exist_ok=False)
    limits = {key: 0 for key in config['model_limit_keys']}
    if args.actor == 'B':
        limits.update(load=1, torch_load=3, predict=horizon + 1)
    spec = {'schema': 'd1-c23-gui-worker-v1', 'mode': args.mode, 'render': not args.headless,
            'profile': profile, 'actor': args.actor, 'control_limit': horizon, 'seconds': horizon * .01,
            'seed': config['mirror_seed'] if args.mirror else config['profile_seeds'][profile],
            'mirror': args.mirror,
            'output_directory': str(output), 'model_limits': limits,
            'soft_s': 180, 'close_s': 235, 'hard_s': 240, 'qualification': False,
            'frame_limit': 1000, 'require_event_driver': False,
            'execution_contract_id': 'C26_B22_user_session_v1'}
    arm = 'user_' + args.mode
    go = {'decision': 'GO', 'scope': 'user initiated bounded session, not additional qualification evidence',
          'repository': str(ROOT), 'worker': config['worker'], 'inputs': sources,
          'runtime_environment': config['runtime_environment'],
          'shared_session': config['shared_session'], 'arms': {arm: spec}, 'x11_events_by_arm': {}}
    save(launch / 'source_go.json', go)
    save(launch / 'request.json', {'source_go_path': str(launch / 'source_go.json'), 'arm': arm, 'spec': spec})
    print('B22 + C18 | 99D/16D | ' + profile + ' | actor=' + args.actor, flush=True)
    print('Output: ' + str(output), flush=True)
    print('W forward; Q/E turn (keyboard); Space/X stop; R simulation reset; Esc exit.', flush=True)
    return subprocess.run(['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
                           '--request', str(launch / 'request.json')], check=False).returncode


if __name__ == '__main__':
    sys.exit(main())


