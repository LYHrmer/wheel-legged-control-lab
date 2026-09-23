"""Compile only. Does not execute the resulting native probe."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path('/home/lyh/wheel-legged-control-lab')
WORK = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923')
LIBDIR = Path('/home/lyh/.local/lib/python3.10/site-packages/mujoco')
SOURCE = ROOT / 'scripts/d1_contact_native_ccd_probe.c'
EXPECTED_SOURCE = '27bca448180d267c2954f127279c70939f85420d5d1b5da4fe991d63da8cf888'
BUILD = WORK / 'native_ccd_build_02'


def identity(path):
    path = Path(path)
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def main():
    if identity(SOURCE)['sha256'] != EXPECTED_SOURCE:
        raise RuntimeError('C source differs from the intended reviewed identity')
    BUILD.mkdir()
    binary = BUILD / 'd1_contact_native_ccd_probe'
    depfile = BUILD / 'dependencies.d'
    argv = ['rtk', 'proxy', 'cc', '-std=c11', '-O0', '-fno-fast-math', '-ffp-contract=off',
            '-Wall', '-Wextra', '-Werror', '-MD', '-MF', str(depfile),
            '-I' + str(LIBDIR / 'include'), '-I' + str(WORK / 'kernel_preparation_02'),
            str(SOURCE), '-o', str(binary), '-L' + str(LIBDIR),
            '-Wl,-rpath,' + str(LIBDIR), '-l:libmujoco.so.3.12.0', '-ldl', '-lm']
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    (BUILD / 'stdout.log').write_text(result.stdout)
    (BUILD / 'stderr.log').write_text(result.stderr)
    deps = {}
    if depfile.is_file():
        raw = depfile.read_text().replace('\\\n', ' ').split(':', 1)[1]
        for item in shlex.split(raw):
            path = Path(item)
            if not path.is_absolute():
                path = ROOT / path
            deps[str(path)] = identity(path)
    receipt = {'compile_argv': argv, 'compiler_version': 'cc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0',
               'exit_code': result.returncode, 'source': identity(SOURCE), 'dependencies': deps,
               'binary': identity(binary) if binary.is_file() else None,
               'binary_path': str(binary), 'binary_executed': False,
               'new_model_compiles': 0, 'new_control': 0, 'new_native': 0}
    (BUILD / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'dependencies'}, indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
