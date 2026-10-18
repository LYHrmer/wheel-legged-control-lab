"""Pin the installed MuJoCo runtime files without importing any runtime package."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import sys
from email.parser import Parser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = Path('/home/lyh/.local/lib/python3.10/site-packages')
ALIAS_SITE_ROOT = ROOT / '.local-deps'
VERSION = '3.12.0'
SCHEMA = 'b22-mujoco-package-inventory39-v1'
DISTRIBUTION_SHA256 = {
    'METADATA': '5e43553c4fc3471a3b48709a0c75c703c2687f4a655597b1d1a351158511c3cf',
    'WHEEL': '3132d1279617fd3c9de8ca1e68e37eab31df7e274538c66041aba630deba470b',
    'RECORD': '50b3e6aa47bc50500a473678d6a60d105c34b1c0d4bfbf7e50e3d60bd3ad7c2d',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(path, *, directory=False):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            'C39 package path is not canonical/non-symlink: ' + str(path))
    require(path.is_dir() if directory else path.is_file(), 'C39 non-regular package path: ' + str(path))
    return path


def identity(path):
    data = plain(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def runtime_code(path):
    """Importable code/native libraries; generated __pycache__ is excluded by callers."""
    return path.suffix in ('.py', '.pyc', '.so', '.pyd', '.dll', '.dylib') or '.so.' in path.name


def installation(site_root):
    site_root = plain(site_root, directory=True)
    package = plain(site_root / 'mujoco', directory=True)
    distribution = plain(site_root / ('mujoco-' + VERSION + '.dist-info'), directory=True)
    inputs = {str(distribution / name): identity(distribution / name) for name in DISTRIBUTION_SHA256}
    require(all(inputs[str(distribution / name)]['sha256'] == expected
                for name, expected in DISTRIBUTION_SHA256.items()), 'C39 original distribution identity changed')
    metadata = Parser().parsestr((distribution / 'METADATA').read_text())
    require(metadata.get_all('Name') == ['mujoco'] and metadata.get_all('Version') == [VERSION],
            'C39 installed distribution name/version differs')
    records = {}
    for row in csv.reader(io.StringIO((distribution / 'RECORD').read_text())):
        require(len(row) == 3, 'C39 invalid RECORD row')
        name = PurePosixPath(row[0])
        require(bool(row[0]) and not name.is_absolute() and '..' not in name.parts
                and str(name) == row[0] and row[0] not in records, 'C39 unsafe/duplicate RECORD path')
        records[row[0]] = row[1:]
    expected_files = {str(site_root / name) for name in records
                      if PurePosixPath(name).parts[0] == 'mujoco'
                      and '__pycache__' not in PurePosixPath(name).parts
                      and runtime_code(PurePosixPath(name))}
    actual_files = set()
    for path in package.rglob('*'):
        require(not path.is_symlink(), 'C39 package symlink forbidden: ' + str(path))
        if '__pycache__' in path.parts or not runtime_code(path):
            continue
        plain(path)
        actual_files.add(str(path))
    require(actual_files == expected_files and str(package / '__init__.py') in actual_files,
            'C39 runtime code set differs from original RECORD')
    for filename in sorted(actual_files):
        path = Path(filename)
        actual = identity(path)
        digest, size = records[str(path.relative_to(site_root))]
        canonical = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(actual['sha256'])).decode().rstrip('=')
        require(digest == canonical and size == str(actual['bytes']), 'C39 RECORD digest/size differs: ' + filename)
        inputs[filename] = actual
    return {'roots': {'mujoco': str(package)},
            'distribution': {'name': 'mujoco', 'version': VERSION, 'path': str(distribution)},
            'code_files': sorted(actual_files), 'inputs': dict(sorted(inputs.items()))}


def inventory39(*, site_root=None):
    """Freeze both audited namespace roots without importing either installation.

    The optional primary site root is for temporary-file fixtures. Distribution
    hashes and the secondary root are source constants, never discovered/rebased
    from runtime imports. NumPy/Gymnasium/libs retain their old release bindings.
    """
    primary = installation(SITE_ROOT if site_root is None else site_root)
    alias = installation(ALIAS_SITE_ROOT)
    primary_root, alias_root = Path(primary['roots']['mujoco']), Path(alias['roots']['mujoco'])
    require(primary_root != alias_root, 'C39 namespace roots must be distinct')
    primary_code = {str(Path(path).relative_to(primary_root)): primary['inputs'][path]
                    for path in primary['code_files']}
    alias_code = {str(Path(path).relative_to(alias_root)): alias['inputs'][path]
                  for path in alias['code_files']}
    require(primary_code == alias_code, 'C39 secondary namespace code differs from primary')
    return {'schema': SCHEMA, 'roots': primary['roots'],
            'namespace_roots': [str(primary_root), str(alias_root)],
            'distribution': primary['distribution'], 'alias_distribution': alias['distribution'],
            'code_files': sorted(primary['code_files'] + alias['code_files']),
            'inputs': dict(sorted({**primary['inputs'], **alias['inputs']}.items())),
            'scope': 'Current MuJoCo Python/native files plus pinned original distribution records; '
                     'no new claim that files absent from old38 were historically unchanged.'}


def validate_inventory39(saved, *, site_root=None):
    current = inventory39(site_root=site_root)
    require(json.dumps(saved, sort_keys=True, allow_nan=False)
            == json.dumps(current, sort_keys=True, allow_nan=False), 'C39 saved package inventory changed')


def verify_loaded_roots39(saved, modules=None):
    """Check already-loaded MuJoCo modules; do not import or discover a package."""
    validate_inventory39(saved)
    modules = sys.modules if modules is None else modules
    package = Path(saved['roots']['mujoco'])
    root_module = modules.get('mujoco')
    require(root_module is not None and list(getattr(root_module, '__path__', ())) == saved['namespace_roots']
            and getattr(root_module, '__file__', None) == str(package / '__init__.py'),
            'C39 loaded MuJoCo namespace root differs')
    for name, module in tuple(modules.items()):
        if name != 'mujoco' and not name.startswith('mujoco.'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is not None:
            path = plain(filename)
            require(path.is_relative_to(package) and str(path) in saved['code_files'] and str(path) in saved['inputs']
                    and identity(path) == saved['inputs'][str(path)], 'C39 unbound loaded MuJoCo origin: ' + name)
            continue
        # rendering is a real namespace package with no __init__.py in this wheel.
        namespace = list(getattr(module, '__path__', ()))
        relative = Path(*name.split('.')[1:])
        expected = [str(Path(root) / relative) for root in saved['namespace_roots']]
        require(namespace == expected, 'C39 loaded MuJoCo namespace differs: ' + name)
        for origin in namespace:
            directory = plain(origin, directory=True)
            require(any(directory.is_relative_to(Path(root)) for root in saved['namespace_roots'])
                    and any(Path(code).is_relative_to(directory) for code in saved['code_files']),
                    'C39 loaded MuJoCo namespace escaped its bound package')
