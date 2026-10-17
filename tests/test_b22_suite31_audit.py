"""Execute only the real terminal-guard AST; never import a worker or model."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'runtime/d1_b22/research/continuation26/gui26/run_gui26.py'
NEW = ROOT/'scripts/run_b22_worker31.py'


def terminal_site(path):
    tree = ast.parse(path.read_text())
    return tree, next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or)
                      and [item.id for item in node.test.values if isinstance(item, ast.Name)]
                      == ['terminated', 'truncated'] and any(isinstance(item, ast.Break) for item in node.body))


def actual_guard(path, *, terminated=True, truncated=False, reason='fall_or_low_clearance', completed=12):
    _, guard = terminal_site(path)
    loop = ast.For(target=ast.Name(id='_', ctx=ast.Store()),
                   iter=ast.List(elts=[ast.Constant(0)], ctx=ast.Load()), body=[copy.deepcopy(guard)], orelse=[])
    tree = ast.Module(body=[loop, ast.Assign(targets=[ast.Name(id='finalization_reached', ctx=ast.Store())],
                                           value=ast.Constant(True))], type_ignores=[])
    namespace = {'session': {'mode': 'script', 'control_limit': 1600}, 'completed': completed,
                 'terminated': terminated, 'truncated': truncated, 'info': {'terminal_reason': reason}}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)  # noqa: S102 -- pinned guard AST only
    return namespace['finalization_reached']


@pytest.mark.parametrize('reason', ['fall_or_low_clearance', 'nonwheel_ground_contact', 'course_map_boundary'])
def test_actual_new_guard_preserves_legal_task_failures_for_finalization(reason):
    with pytest.raises(RuntimeError, match='fixed script terminated'):
        actual_guard(OLD, reason=reason)
    assert actual_guard(NEW, reason=reason) is True


@pytest.mark.parametrize('terminated,truncated,reason', [
    (False, True, 'time_limit'), (True, True, 'fall_or_low_clearance'),
    (True, False, 'owner_soft_deadline'), (True, False, 'unregistered_stop'),
])
def test_actual_new_guard_does_not_relabel_engineering_or_partial_stops(terminated, truncated, reason):
    with pytest.raises(RuntimeError, match='without a legitimate task terminal'):
        actual_guard(NEW, terminated=terminated, truncated=truncated, reason=reason)


def test_real_full_horizon_still_reaches_normal_finalization():
    assert actual_guard(NEW, terminated=False, truncated=True, reason='time_limit', completed=1600)


def test_same_owner_retains_archive_and_receipt_after_the_real_terminal_loop():
    tree, guard = terminal_site(NEW)
    owner = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'owner')
    assignments = [node for node in ast.walk(owner) if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) == "shared['result']" for target in node.targets)]
    assert len(assignments) == 1 and assignments[0].lineno > guard.end_lineno
    expected = {'finish_episode', 'runtime.unbind', 'writer.drain'}
    calls = {ast.unparse(node.func) for node in ast.walk(owner)
             if isinstance(node, ast.Call) and guard.end_lineno < node.lineno < assignments[0].lineno}
    assert expected <= calls
    assert isinstance(assignments[0].value, ast.Dict)
    assert 'completed_controls' in [item.value for item in assignments[0].value.keys if isinstance(item, ast.Constant)]
