"""Run an archived reader while rejecting reads from original project data/code."""
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import sys

original = Path('/home/lyh/wheel-legged-control-lab')
script, study, output, receipt = map(lambda x: Path(x).absolute(), sys.argv[1:5])
counts = Counter()
blocked = []


def audit(event, args):
    if event not in {'open', 'os.listdir', 'os.scandir'} or not args:
        return
    value = args[0]
    if not isinstance(value, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(value)).absolute()
    if path.is_relative_to(original) and not path.is_relative_to(original / '.local-deps'):
        blocked.append({'event': event, 'path': str(path)})
        raise PermissionError(f'migration reader accessed original project: {path}')
    if path.is_relative_to(study):
        counts[event] += 1
        if event == 'open':
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                blocked.append({'event': event, 'path': str(path), 'reason': 'input_write'})
                raise PermissionError(f'migration reader attempted input write: {path}')


sys.path.insert(0, str(script.parent))
sys.argv = [str(script), str(study), '--output', str(output)]
sys.addaudithook(audit)
success = False
try:
    runpy.run_path(str(script), run_name='__main__')
    success = True
finally:
    with receipt.open('x') as f:
        json.dump({'status': 'passed' if success else 'failed', 'script': str(script),
                   'study': str(study), 'blocked_accesses': blocked,
                   'study_access_counts': dict(counts),
                   'scope': 'CPython audit events reject original project reads except shared .local-deps and reject writes to migrated study; not an OS sandbox or dependency migration test.'}, f, indent=2)
        f.write('\n')
