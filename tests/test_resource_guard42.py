"""Pure filesystem fixtures only: no unit, allocation probe, model or physics."""
import importlib.util
from pathlib import Path

import pytest


def load_guard():
    here = Path(__file__).resolve().parent
    paths = [here / 'resource_guard_candidate.py',
             here.parent / 'tools27/resource_guard27.py',
             here.parent / 'scripts/resource_guard42.py']
    source = next(p for p in paths if p.is_file())
    spec = importlib.util.spec_from_file_location('resource_guard_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    g = load_guard()
    proc = tmp_path / 'proc'; cg = tmp_path / 'cgroup'
    (proc / 'self').mkdir(parents=True); cg.mkdir()
    scope = '/user.slice/test.service'; owned = cg / scope.lstrip('/')
    owned.mkdir(parents=True)
    (proc / 'self/cgroup').write_text('0::' + scope + '\n')
    (proc / 'meminfo').write_text('MemTotal: 16000000 kB\nMemAvailable: 6291456 kB\n')
    values = {'memory.high': g.HIGH_BYTES, 'memory.max': g.MAX_BYTES,
              'memory.current': 1100 * 1024**2, 'memory.swap.max': 0,
              'memory.swap.current': 0, 'memory.oom.group': 1,
              'memory.peak': 1200 * 1024**2,
              'memory.events': 'low 0\nhigh 3\nmax 1\noom 0\noom_kill 0\noom_group_kill 0'}
    for name, value in values.items():
        (owned / name).write_text(str(value) + '\n')
    (owned.parent / 'memory.max').write_text('max\n')
    (owned.parent / 'memory.current').write_text(str(4 * 1024**3))
    return g, scope, proc, cg, owned


def read(f):
    g, scope, proc, cg, _ = f
    return g.read_memory(scope, proc, cg)


def test_own_limit_is_independent_of_global_headroom(fixture):
    value = read(fixture)
    assert value['effective_available_bytes'] == 6 * 1024**3
    assert value['cgroup_remaining_bytes'] is None
    assert value['owned']['remaining_bytes'] < 3 * 1024**3
    assert value['owned']['peak_bytes'] == 1200 * 1024**2
    assert value['raw_ancestors'][0]['owned'] is True


def test_external_limit_still_restricts_effective(fixture):
    _, _, _, _, owned = fixture
    (owned.parent / 'memory.max').write_text(str(7 * 1024**3))
    value = read(fixture)
    assert value['effective_available_bytes'] == 3 * 1024**3
    # Caller retains the 4GiB stop: guard never disguises the external shortage.


def test_host_memory_is_never_replaced_by_own_headroom(fixture):
    _, _, proc, _, _ = fixture
    (proc / 'meminfo').write_text('MemAvailable: 4000000 kB\n')
    assert read(fixture)['effective_available_bytes'] == 4000000 * 1024


def test_missing_peak_is_null_and_sample_max_is_honest(fixture):
    _, _, _, _, owned = fixture
    (owned / 'memory.peak').unlink()
    first = read(fixture)
    assert first['owned']['peak_bytes'] is None
    (owned / 'memory.current').write_text(str(1400 * 1024**2))
    assert read(fixture)['owned']['sampled_max_bytes'] == 1400 * 1024**2
    (owned / 'memory.current').write_text(str(1000 * 1024**2))
    assert read(fixture)['owned']['sampled_max_bytes'] == 1400 * 1024**2


@pytest.mark.parametrize('name,value', [
    ('memory.high', 'max'), ('memory.high', str(1536 * 1024**2)),
    ('memory.max', str(2048 * 1024**2)), ('memory.max', 'max'),
    ('memory.swap.max', '1'), ('memory.swap.current', '1'),
    ('memory.oom.group', '0'), ('memory.current', '-1'),
    ('memory.peak', 'unknown'),
])
def test_unknown_or_wrong_owned_controls_fail_with_diagnostics(fixture, name, value):
    g, scope, proc, cg, owned = fixture
    (owned / name).write_text(value)
    snapshot = g.snapshot_memory(scope, proc, cg)
    assert snapshot['errors']
    with pytest.raises(g.ResourceGuardError) as caught:
        g.check_memory(snapshot)
    assert caught.value.snapshot is snapshot


@pytest.mark.parametrize('event', ['oom', 'oom_kill', 'oom_group_kill'])
def test_any_oom_event_fails(fixture, event):
    g, _, _, _, owned = fixture
    path = owned / 'memory.events'
    path.write_text(path.read_text().replace(event + ' 0\n', event + ' 1\n'))
    with pytest.raises(g.ResourceGuardError, match='OOM'):
        read(fixture)


@pytest.mark.parametrize('raw', ['high 0\n', 'low 0\nhigh nan\n', 'low 0\nlow 1\n'])
def test_unknown_or_malformed_events_fail(fixture, raw):
    g, _, _, _, owned = fixture
    (owned / 'memory.events').write_text(raw)
    with pytest.raises(g.ResourceGuardError):
        read(fixture)


@pytest.mark.parametrize('scope', ['/', 'user.slice/test.service', '/user.slice/../test.service',
                                   '/user.slice//test.service', '/user.slice/test.service/'])
def test_scope_cannot_select_arbitrary_ancestor_or_escape(fixture, scope):
    g, _, proc, cg, _ = fixture
    with pytest.raises(g.ResourceGuardError):
        g.read_memory(scope, proc, cg)


def test_scope_mismatch_rejected_before_excluding_any_limit(fixture):
    g, _, proc, cg, _ = fixture
    with pytest.raises(g.ResourceGuardError, match='self scope differs'):
        g.read_memory('/other.service', proc, cg)


@pytest.mark.parametrize('name', ['memory.max', 'memory.current'])
def test_unknown_nonroot_ancestor_is_not_infinite(fixture, name):
    g, _, _, _, owned = fixture
    (owned.parent / name).unlink()
    with pytest.raises(g.ResourceGuardError):
        read(fixture)


def test_control_file_symlink_rejected(fixture):
    g, _, _, _, owned = fixture
    target = owned.parent / 'alias'
    target.write_text(str(g.MAX_BYTES))
    (owned / 'memory.max').unlink()
    (owned / 'memory.max').symlink_to(target)
    with pytest.raises(g.ResourceGuardError):
        read(fixture)


def test_directory_symlink_rejected(fixture):
    g, scope, proc, cg, owned = fixture
    actual = owned.with_name('other.service')
    owned.rename(actual); owned.symlink_to(actual, target_is_directory=True)
    with pytest.raises(g.ResourceGuardError, match='symlink'):
        g.read_memory(scope, proc, cg)


def test_mid_snapshot_limit_change_is_rejected(fixture, monkeypatch):
    g, scope, proc, cg, owned = fixture
    original = g._read
    reads = 0
    def changing(path):
        nonlocal reads
        value = original(path)
        if path == owned / 'memory.high':
            reads += 1
            if reads == 2:
                return str(g.HIGH_BYTES - 1)
        return value
    monkeypatch.setattr(g, '_read', changing)
    with pytest.raises(g.ResourceGuardError, match='changed during snapshot'):
        g.read_memory(scope, proc, cg)


def test_self_move_mid_snapshot_is_rejected(fixture, monkeypatch):
    g, scope, proc, cg, _ = fixture
    reads = 0
    original = g._self_scope
    def changing(path):
        nonlocal reads
        reads += 1
        return original(path) if reads == 1 else '/other.service'
    monkeypatch.setattr(g, '_self_scope', changing)
    with pytest.raises(g.ResourceGuardError, match='self scope changed'):
        g.read_memory(scope, proc, cg)
