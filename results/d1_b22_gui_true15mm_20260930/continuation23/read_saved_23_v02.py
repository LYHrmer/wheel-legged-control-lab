"""Root-owned bounded launch of the frozen saved-only GUI reader."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=('script_headless', 'script_gui', 'keyboard_gui'))
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    run = folder/(args.arm+'_01')
    session = json.loads((run/'session.json').read_text())
    environment = os.environ.copy()
    for key, value in session['runtime_environment'].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
        environment.pop(key, None)
    result_path = folder/('independent_'+args.arm+'_v02.json')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B',
        '/home/lyh/wheel-legged-control-lab/runtime/d1_b22/research/continuation23/root23/reader23_v02.py',
        '--run', str(run), '--output', str(result_path)]
    if args.arm == 'script_gui':
        command.extend(['--pair-headless', str(folder/'script_headless_01')])
    start = time.monotonic()
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)
    with (folder/('reader_'+args.arm+'_v02.log')).open('x') as stream:
        stream.write(result.stdout+result.stderr)
    with (folder/('reader_'+args.arm+'_host_v02.json')).open('x') as stream:
        json.dump({'command': command, 'exit_code': result.returncode,
                   'elapsed_s': time.monotonic()-start, 'model_calls': 0,
                   'physics': 0, 'timeout_s': 300}, stream, indent=2)
    print(json.dumps({'arm': args.arm, 'exit_code': result.returncode,
                      'elapsed_s': time.monotonic()-start}), flush=True)
    print((result.stdout+result.stderr)[-6000:], flush=True)
    if result.returncode == 0:
        report = json.loads(result_path.read_text())
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ('episodes', 'source_hashes')}, indent=2)[:12000])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

