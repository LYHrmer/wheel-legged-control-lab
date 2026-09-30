"""One bounded saved-data diagnosis; checks archived block identities first."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

C = Path(__file__).resolve().parent
W = C.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (C / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    source = C / 'diagnose_saved_rough18.py'
    expected = 'b232ce8a0d7ef4e92adf5dc0149351045772514fdc162d2a47ddb0b4301a8153'
    if sha(source) != expected:
        raise ValueError('diagnostic source changed')
    inputs = {str(source): expected}
    for run in ('continuation16/eval_01', 'continuation17/qualification_primary_01'):
        for actor in ('zero', 'grouped_continue'):
            folder = W / run / 'heldout' / ('rough_0p35_' + actor)
            path = folder / 'case_receipt.json'
            receipt = json.loads(path.read_text())
            inputs[str(path)] = sha(path)
            if receipt['completed_controls'] != 1600 or not receipt['record_valid']:
                raise ValueError('rough archive incomplete')
            for block in receipt['controller_record_blocks']:
                path = folder / block['file']
                digest = sha(path)
                if digest != block['sha256']:
                    raise ValueError('control block changed')
                inputs[str(path)] = digest
    argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(source)]
    write('diagnosis_reservation_18.json', {'argv': argv, 'inputs': inputs,
          'model_calls': 0, 'physical_steps': 0, 'child_wall_limit_s': 90,
          'cleanup_limit_s': 5, 'retry_permitted': False})
    start = time.monotonic()
    proc = None
    failure = None
    try:
        with (C / 'diagnosis_18.log').open('xb') as stream:
            proc = subprocess.Popen(argv, cwd=C, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            if proc.wait(timeout=90) != 0:
                raise RuntimeError('pure diagnosis exited unsuccessfully')
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        result = {'failure': failure, 'elapsed_s': time.monotonic()-start,
                  'exit_code': None if proc is None else proc.returncode,
                  'changed_inputs': [name for name, digest in inputs.items() if sha(Path(name)) != digest],
                  'model_calls': 0, 'physical_steps': 0,
                  'output_sha256': sha(C / 'saved_rough_diagnosis_18.json')
                  if (C / 'saved_rough_diagnosis_18.json').exists() else None}
        write('diagnosis_receipt_18.json', result)
        print(json.dumps(result))
    return 0 if failure is None and not result['changed_inputs'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
