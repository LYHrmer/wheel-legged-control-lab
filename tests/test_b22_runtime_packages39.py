"""Pure file/module fixtures: no installed runtime import, native load or physics."""
import base64
import copy
import csv
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installation(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'scripts/b22_runtime_packages39.py'
    spec = importlib.util.spec_from_file_location('package39_pure_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    site = tmp_path / 'site-packages'
    package = site / 'mujoco'
    distribution = site / 'mujoco-3.12.0.dist-info'
    code = {
        '__init__.py': b'# fake root source\n',
        '_enums.cpython-310-x86_64-linux-gnu.so': b'not a real shared object',
        'libmujoco.so.3.12.0': b'not a real engine',
        'plugin/libactuator.so': b'not a real plugin',
        'rendering/classic/__init__.py': b'# fake classic\n',
        'rendering/classic/renderer.py': b'# fake renderer\n',
    }
    for name, value in code.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    distribution.mkdir()
    (distribution / 'METADATA').write_text('Metadata-Version: 2.1\nName: mujoco\nVersion: 3.12.0\n')
    (distribution / 'WHEEL').write_text('Wheel-Version: 1.0\nTag: cp310-cp310-manylinux_x86_64\n')

    def record():
        files = [*(package / name for name in code), distribution / 'METADATA', distribution / 'WHEEL']
        with (distribution / 'RECORD').open('w', newline='') as stream:
            writer = csv.writer(stream)
            for path in files:
                raw = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
                writer.writerow((path.relative_to(site).as_posix(), 'sha256=' + digest, len(raw)))
            writer.writerow(((distribution / 'RECORD').relative_to(site).as_posix(), '', ''))

    def repin():
        # Fixtures declare their own pretend installation; production constants stay fixed.
        monkeypatch.setattr(module, 'DISTRIBUTION_SHA256', {
            name: hashlib.sha256((distribution / name).read_bytes()).hexdigest()
            for name in ('METADATA', 'WHEEL', 'RECORD')
        })

    record()
    repin()
    monkeypatch.setattr(module, 'SITE_ROOT', site)
    alias_site = tmp_path / 'secondary-site-packages'
    shutil.copytree(site, alias_site)
    monkeypatch.setattr(module, 'ALIAS_SITE_ROOT', alias_site)
    return SimpleNamespace(api=module, site=site, package=package, distribution=distribution,
                           alias_package=alias_site / 'mujoco', alias_site=alias_site,
                           code=code, record=record, repin=repin)


def test_complete_runtime_package_includes_versioned_engine_and_plugins(installation):
    f = installation
    before = set(sys.modules)
    value = f.api.inventory39()
    assert value['roots'] == {'mujoco': str(f.package)}
    assert value['distribution']['version'] == '3.12.0'
    assert value['namespace_roots'] == [str(f.package), str(f.alias_package)]
    assert set(value['code_files']) == {str(root / name) for root in (f.package, f.alias_package) for name in f.code}
    assert len(value['inputs']) == 2 * (len(f.code) + 3)
    assert str(f.package / 'libmujoco.so.3.12.0') in value['inputs']
    assert str(f.package / 'plugin/libactuator.so') in value['inputs']
    assert f.api.validate_inventory39(value) is None
    assert not {name.split('.')[0] for name in set(sys.modules) - before} & {
        'numpy', 'gymnasium', 'mujoco', 'torch', 'stable_baselines3', 'cv2'
    }


@pytest.mark.parametrize('name', ['unexpected.py', 'unexpected.pyc', 'plugin/libunexpected.so', 'unexpected.so.4'])
def test_added_unrecorded_runtime_code_is_rejected(installation, name):
    f = installation
    saved = f.api.inventory39()
    (f.package / name).write_bytes(b'unknown runtime code')
    with pytest.raises(ValueError, match='code set'):
        f.api.validate_inventory39(saved)


def test_new_record_cannot_silently_rebaseline_an_added_module(installation):
    f = installation
    saved = f.api.inventory39()
    f.code['unexpected.py'] = b'new source'
    (f.package / 'unexpected.py').write_bytes(f.code['unexpected.py'])
    f.record()
    with pytest.raises(ValueError, match='original distribution'):
        f.api.validate_inventory39(saved)


@pytest.mark.parametrize('name', ['__init__.py', '_enums.cpython-310-x86_64-linux-gnu.so',
                                  'plugin/libactuator.so', 'libmujoco.so.3.12.0'])
def test_same_length_runtime_mutation_fails_original_record(installation, name):
    f = installation
    path = f.package / name
    path.write_bytes(b'x' * path.stat().st_size)
    with pytest.raises(ValueError, match='digest/size'):
        f.api.inventory39()


def test_missing_recorded_code_is_not_accepted(installation):
    f = installation
    (f.package / 'rendering/classic/renderer.py').unlink()
    with pytest.raises(ValueError, match='code set'):
        f.api.inventory39()


@pytest.mark.parametrize('name', ['METADATA', 'WHEEL', 'RECORD'])
def test_each_original_distribution_file_is_pinned(installation, name):
    f = installation
    path = f.distribution / name
    path.write_bytes(path.read_bytes() + b'\nchanged')
    with pytest.raises(ValueError, match='original distribution'):
        f.api.inventory39()


def test_wrong_distribution_version_fails_even_when_fixture_hash_is_pinned(installation):
    f = installation
    path = f.distribution / 'METADATA'
    path.write_text(path.read_text().replace('3.12.0', '3.12.1'))
    f.repin()
    with pytest.raises(ValueError, match='name/version'):
        f.api.inventory39()


@pytest.mark.parametrize('record_path', ['/outside.py', '../escape.py', 'mujoco/../escape.py',
                                          './mujoco/new.py', 'mujoco/__init__.py'])
def test_unsafe_or_duplicate_record_path_is_rejected(installation, record_path):
    f = installation
    with (f.distribution / 'RECORD').open('a') as stream:
        csv.writer(stream).writerow((record_path, 'sha256=unused', 1))
    f.repin()
    with pytest.raises(ValueError, match='unsafe/duplicate'):
        f.api.inventory39()


@pytest.mark.parametrize('replacement', ['md5=not-sha256', 'sha256=incorrect'])
def test_record_original_digest_is_checked_not_just_file_presence(installation, replacement):
    f = installation
    path = f.distribution / 'RECORD'
    rows = list(csv.reader(path.read_text().splitlines()))
    rows[0][1] = replacement
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    f.repin()
    with pytest.raises(ValueError, match='digest/size'):
        f.api.inventory39()


def test_record_original_size_is_checked(installation):
    f = installation
    path = f.distribution / 'RECORD'
    rows = list(csv.reader(path.read_text().splitlines()))
    rows[0][2] = str(int(rows[0][2]) + 1)
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    f.repin()
    with pytest.raises(ValueError, match='digest/size'):
        f.api.inventory39()


@pytest.mark.parametrize('where', ['source', 'plugin', 'namespace_directory', 'metadata', 'site_root'])
def test_symlink_paths_are_rejected(installation, tmp_path, where):
    f = installation
    outside = tmp_path / 'outside'
    outside.mkdir()
    if where in ('source', 'plugin', 'metadata'):
        path = f.distribution / 'RECORD' if where == 'metadata' else f.package / (
            '__init__.py' if where == 'source' else 'plugin/libactuator.so')
        target = outside / 'saved'
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif where == 'namespace_directory':
        (f.package / 'extra_namespace').symlink_to(outside, target_is_directory=True)
    else:
        alias = tmp_path / 'site-alias'
        alias.symlink_to(f.site, target_is_directory=True)
        with pytest.raises(ValueError, match='non-symlink'):
            f.api.inventory39(site_root=alias)
        return
    with pytest.raises(ValueError, match='symlink'):
        f.api.inventory39()


def test_bytecode_cache_and_nonruntime_data_do_not_expand_runtime_inventory(installation):
    f = installation
    saved = f.api.inventory39()
    cache = f.package / '__pycache__'
    cache.mkdir()
    (cache / '__init__.cpython-310.pyc').write_bytes(b'not imported')
    (f.package / 'typing.pyi').write_bytes(b'not executable source')
    (f.package / 'test_model.xml').write_bytes(b'not parsed or loaded')
    assert f.api.inventory39() == saved


@pytest.mark.parametrize('mutator', [
    lambda value: value['code_files'].pop(),
    lambda value: value['inputs'].pop(next(iter(value['inputs']))),
    lambda value: value['distribution'].update(version='3.12.1'),
    lambda value: value['roots'].update(mujoco='/outside'),
    lambda value: value.update(unregistered_field=True),
])
def test_saved_inventory_requires_exact_structure(installation, mutator):
    f = installation
    saved = f.api.inventory39()
    mutator(saved)
    with pytest.raises(ValueError, match='saved package inventory'):
        f.api.validate_inventory39(saved)


def loaded_modules(f):
    return {
        'mujoco': SimpleNamespace(__file__=str(f.package / '__init__.py'),
                                  __path__=[str(f.package), str(f.alias_package)]),
        'mujoco._enums': SimpleNamespace(__file__=str(f.package / '_enums.cpython-310-x86_64-linux-gnu.so')),
        'mujoco.rendering': SimpleNamespace(__file__=None,
                                            __path__=[str(root / 'rendering') for root in (f.package, f.alias_package)]),
        'mujoco.rendering.classic': SimpleNamespace(__file__=str(f.package / 'rendering/classic/__init__.py')),
    }


def test_loaded_root_extensions_and_real_namespace_are_checked_without_import(installation):
    f = installation
    saved = f.api.inventory39()
    assert f.api.verify_loaded_roots39(saved, modules=loaded_modules(f)) is None


@pytest.mark.parametrize('where', ['missing_root', 'extra_root', 'wrong_init', 'external_module',
                                  'unknown_fileless_module', 'external_namespace', 'empty_namespace'])
def test_loaded_namespace_or_origin_escape_is_rejected(installation, tmp_path, where):
    f = installation
    saved = f.api.inventory39()
    modules = loaded_modules(f)
    outside = tmp_path / 'other-package'
    outside.mkdir()
    (outside / 'foreign.py').write_text('# foreign\n')
    if where == 'missing_root':
        modules.pop('mujoco')
    elif where == 'extra_root':
        modules['mujoco'].__path__.append(str(outside))
    elif where == 'wrong_init':
        modules['mujoco'].__file__ = str(f.package / 'rendering/classic/__init__.py')
    elif where == 'external_module':
        modules['mujoco.foreign'] = SimpleNamespace(__file__=str(outside / 'foreign.py'))
    elif where == 'unknown_fileless_module':
        modules['mujoco.foreign'] = SimpleNamespace(__file__=None)
    elif where == 'external_namespace':
        modules['mujoco.rendering'].__path__ = [str(outside)]
    else:
        empty = f.package / 'empty'
        empty.mkdir()
        modules['mujoco.empty'] = SimpleNamespace(__file__=None, __path__=[str(empty)])
    with pytest.raises(ValueError, match='C39'):
        f.api.verify_loaded_roots39(saved, modules=modules)


def test_source_drift_after_inventory_is_caught_before_loaded_origin_acceptance(installation):
    f = installation
    saved = copy.deepcopy(f.api.inventory39())
    path = f.package / 'plugin/libactuator.so'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='digest/size'):
        f.api.verify_loaded_roots39(saved, modules=loaded_modules(f))


@pytest.mark.parametrize('change', ['code_mutation', 'added_plugin', 'record_change'])
def test_secondary_installation_is_independently_bound(installation, change):
    f = installation
    saved = f.api.inventory39()
    if change == 'code_mutation':
        path = f.alias_package / 'plugin/libactuator.so'
        path.write_bytes(b'x' * path.stat().st_size)
    elif change == 'added_plugin':
        (f.alias_package / 'plugin/libnew.so').write_bytes(b'unregistered')
    else:
        path = f.alias_site / 'mujoco-3.12.0.dist-info/RECORD'
        path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='C39'):
        f.api.validate_inventory39(saved)


def test_bound_secondary_code_is_not_an_allowed_actual_module_origin(installation):
    f = installation
    saved = f.api.inventory39()
    modules = loaded_modules(f)
    modules['mujoco._enums'].__file__ = str(f.alias_package / '_enums.cpython-310-x86_64-linux-gnu.so')
    assert modules['mujoco._enums'].__file__ in saved['inputs']
    with pytest.raises(ValueError, match='unbound loaded MuJoCo origin'):
        f.api.verify_loaded_roots39(saved, modules=modules)


@pytest.mark.parametrize('paths', ['primary_only', 'reordered', 'duplicated'])
def test_loaded_namespace_requires_exactly_both_original_paths_in_order(installation, paths):
    f = installation
    saved = f.api.inventory39()
    modules = loaded_modules(f)
    value = modules['mujoco'].__path__
    modules['mujoco'].__path__ = value[:1] if paths == 'primary_only' else (
        list(reversed(value)) if paths == 'reordered' else value + value[:1])
    with pytest.raises(ValueError, match='namespace root differs'):
        f.api.verify_loaded_roots39(saved, modules=modules)
