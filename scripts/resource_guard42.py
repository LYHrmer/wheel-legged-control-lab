"""Read-only cgroup-v2 guard; caller supplies its preregistered unit scope.

Global headroom excludes only that exact owned unit's independent hard budget.
This module creates no cgroup/process, reads no environment and sends no signal.
"""
import os
import re
import threading
from pathlib import Path, PurePosixPath

HIGH_BYTES = 2048 * 1024**2
MAX_BYTES = 3072 * 1024**2
_MAXIMA = {}
_LOCK = threading.Lock()


class ResourceGuardError(ValueError):
    def __init__(self, snapshot):
        self.snapshot = snapshot
        super().__init__('; '.join(snapshot.get('errors', [])) or 'invalid resource snapshot')


def _read(path):
    # Refuse a control-file symlink, including in injected test filesystems.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        raw = os.read(fd, 65537)
        if len(raw) > 65536:
            raise ValueError('oversized control file: ' + str(path))
        return raw.decode('ascii').strip()
    finally:
        os.close(fd)


def _number(value):
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError('nonnegative integer required: ' + repr(value))
    return int(value)


def _scope(value):
    if not isinstance(value, str) or value == '/' or '\x00' in value:
        raise ValueError('owned_scope must name a preregistered non-root absolute scope')
    p = PurePosixPath(value)
    if not p.is_absolute() or str(p) != value or '..' in p.parts or value.startswith('//'):
        raise ValueError('owned_scope must be canonical and absolute')
    return p


def _self_scope(proc_root):
    rows = _read(proc_root / 'self/cgroup').splitlines()
    unified = [row[3:] for row in rows if row.startswith('0::')]
    if len(unified) != 1:
        raise ValueError('exactly one unified self cgroup required')
    return unified[0]


def _directory(root, relative):
    path = root
    for part in relative.parts[1:]:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink cgroup directory: ' + str(path))
    if path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError('noncanonical cgroup directory: ' + str(path))
    return path


def _events(path):
    result = {}
    for line in _read(path).splitlines():
        pair = line.split()
        if len(pair) != 2 or pair[0] in result:
            raise ValueError('malformed/duplicate memory event')
        result[pair[0]] = _number(pair[1])
    required = {'low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill'}
    if not required <= result.keys():
        raise ValueError('missing memory event counters')
    return result


def snapshot_memory(owned_scope, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    """Return diagnostics even on failure; check_memory enforces acceptance.

    sampled_max_bytes is this process's maximum observed memory.current for the
    same scope/inode. It is not a substitute for the kernel memory.peak.
    """
    result = {'available_ram_bytes': None, 'cgroup_remaining_bytes': None,
              'effective_available_bytes': None, 'owned': {}, 'raw_ancestors': [],
              'errors': [], 'accounting_scope': 'host_and_external_ancestors_excluding_exact_owned_unit'}
    proc_root, cgroup_root = Path(proc_root), Path(cgroup_root)
    try:
        relative = _scope(owned_scope)
        if not cgroup_root.is_absolute() or cgroup_root.resolve(strict=True) != cgroup_root:
            raise ValueError('cgroup_root must be canonical and absolute')
        actual = _self_scope(proc_root)
        result['self_scope'] = actual
        if actual != owned_scope:
            raise ValueError('self scope differs from preregistered owned_scope')
        rows = [line.split(':', 1)[1].split() for line in _read(proc_root / 'meminfo').splitlines()
                if line.startswith('MemAvailable:')]
        if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != 'kB':
            raise ValueError('missing/malformed MemAvailable')
        result['available_ram_bytes'] = _number(rows[0][0]) * 1024
        path = _directory(cgroup_root, relative)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        owned = {'path': owned_scope, 'filesystem_path': str(path), 'inode': stat.st_ino,
                 'device': stat.st_dev, 'current_bytes': None, 'high_bytes': None,
                 'max_bytes': None, 'remaining_bytes': None, 'swap_current_bytes': None,
                 'swap_max_bytes': None, 'oom_group': None, 'events': None,
                 'peak_bytes': None, 'sampled_max_bytes': None}
        result['owned'] = owned
        controls = {'high_bytes': 'memory.high', 'max_bytes': 'memory.max',
                    'swap_max_bytes': 'memory.swap.max', 'oom_group': 'memory.oom.group'}
        for key, name in controls.items():
            owned[key] = _number(_read(path / name))
        owned['current_bytes'] = _number(_read(path / 'memory.current'))
        owned['remaining_bytes'] = max(0, owned['max_bytes'] - owned['current_bytes'])
        owned['swap_current_bytes'] = _number(_read(path / 'memory.swap.current'))
        owned['events'] = _events(path / 'memory.events')
        try:
            owned['peak_bytes'] = _number(_read(path / 'memory.peak'))
        except FileNotFoundError:
            # Older kernels can lack this counter; absence is never a zero peak.
            owned['peak_unavailable_reason'] = 'memory.peak not provided by kernel'
        cache_key = (os.getpid(), str(proc_root), str(cgroup_root), owned_scope, *identity)
        with _LOCK:
            _MAXIMA[cache_key] = max(_MAXIMA.get(cache_key, 0), owned['current_bytes'])
            owned['sampled_max_bytes'] = _MAXIMA[cache_key]
        result['raw_ancestors'].append({'path': owned_scope, 'owned': True,
                                      'current_bytes': owned['current_bytes'],
                                      'max_bytes': owned['max_bytes'],
                                      'remaining_bytes': owned['remaining_bytes']})
        external = []
        ancestor = path.parent
        while True:
            if ancestor == cgroup_root and not (ancestor / 'memory.max').exists():
                result['raw_ancestors'].append({'path': '/', 'owned': False,
                                              'root_limit_not_exposed': True})
            else:
                # Every non-root ancestor must be present; unknown is not infinity.
                if ancestor.resolve(strict=True) != ancestor:
                    raise ValueError('symlink/noncanonical external ancestor')
                maximum = _read(ancestor / 'memory.max')
                maximum = None if maximum == 'max' else _number(maximum)
                current = _number(_read(ancestor / 'memory.current'))
                remaining = None if maximum is None else max(0, maximum - current)
                row = {'path': '/' + str(ancestor.relative_to(cgroup_root)) if ancestor != cgroup_root else '/',
                       'owned': False, 'max_bytes': maximum, 'current_bytes': current,
                       'remaining_bytes': remaining}
                result['raw_ancestors'].append(row)
                if remaining is not None:
                    external.append(remaining)
            if ancestor == cgroup_root:
                break
            ancestor = ancestor.parent
        result['cgroup_remaining_bytes'] = min(external) if external else None
        result['effective_available_bytes'] = min([result['available_ram_bytes'], *external])
        # Fail closed if identity, membership or configured hard limits changed mid-read.
        after = path.stat()
        if identity != (after.st_dev, after.st_ino) or path.resolve(strict=True) != path:
            raise ValueError('owned scope inode/path changed during snapshot')
        if _self_scope(proc_root) != owned_scope:
            raise ValueError('self scope changed during snapshot')
        for key, name in controls.items():
            if owned[key] != _number(_read(path / name)):
                raise ValueError('owned control changed during snapshot: ' + name)
    except (OSError, ValueError, UnicodeError) as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    owned = result['owned']
    if owned:
        expected = {'high_bytes': HIGH_BYTES, 'max_bytes': MAX_BYTES,
                    'swap_max_bytes': 0, 'oom_group': 1}
        for key, value in expected.items():
            if owned.get(key) != value:
                result['errors'].append('unexpected owned ' + key)
        if owned.get('swap_current_bytes') != 0:
            result['errors'].append('owned swap usage is unknown or nonzero')
        for key, value in (owned.get('events') or {}).items():
            if key.startswith('oom') and value:
                result['errors'].append('owned OOM event: ' + key)
    return result


def check_memory(snapshot):
    """Accept a complete snapshot, or raise while retaining its diagnostics."""
    if snapshot.get('errors') or type(snapshot.get('effective_available_bytes')) is not int:
        raise ResourceGuardError(snapshot)
    return snapshot


def read_memory(owned_scope, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    return check_memory(snapshot_memory(owned_scope, proc_root, cgroup_root))
