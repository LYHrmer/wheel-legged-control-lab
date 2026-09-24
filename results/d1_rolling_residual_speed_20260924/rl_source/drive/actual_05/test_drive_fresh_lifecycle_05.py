"""Six pure source-tied regressions for the never-reset transfer guard."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from drive_fresh_lifecycle_05 import assert_never_reset_inactive

REPO = Path("/home/lyh/wheel-legged-control-lab")
LOCOMOTION = REPO / "src/wheel_legged_control/d1/locomotion_env.py"
HEADING = REPO / "scripts/d1_heading_tracking_env.py"
REQUIRED_NONE = ("loop", "decision", "last_transition", "schedule",
                 "_heading_context", "_reference_loop", "_reference_initial")


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _literal_assignments(method: ast.FunctionDef) -> dict[str, object]:
    values = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                    and target.value.id == "self":
                try:
                    values[target.attr] = ast.literal_eval(value)
                except (TypeError, ValueError):
                    pass
    return values


class _Fresh:
    @property
    def heading_decision(self):
        return self._heading_context


def _fresh_from_frozen_constructor() -> _Fresh:
    locomotion = _literal_assignments(_method(LOCOMOTION, "D1LocomotionEnv", "__init__"))
    heading = _literal_assignments(_method(HEADING, "D1HeadingTrackingEnv", "__init__"))
    all_fields = {**locomotion, **heading}
    assert all(name in all_fields and all_fields[name] is None for name in REQUIRED_NONE)
    assert all_fields["_active"] is False
    assert all_fields["_episode_metadata"] == {}
    assert "_steps" not in all_fields
    value = _Fresh()
    for name in (*REQUIRED_NONE, "_active", "_episode_metadata"):
        setattr(value, name, all_fields[name])
    value.plant = SimpleNamespace(data=SimpleNamespace(time=0.0))
    return value


def test_real_constructor_shape_is_accepted_without_mutation():
    fresh = _fresh_from_frozen_constructor()
    before = dict(vars(fresh))
    result = assert_never_reset_inactive(fresh)
    assert result["steps_attribute_absent"] and result["physical_time_zero"]
    assert vars(fresh) == before and not hasattr(fresh, "_steps")
    heading_property = _method(HEADING, "D1HeadingTrackingEnv", "heading_decision")
    assert any(isinstance(node, ast.Return)
               and ast.unparse(node.value) == "self._heading_context"
               for node in ast.walk(heading_property))
    reset = _literal_assignments(_method(LOCOMOTION, "D1LocomotionEnv", "reset"))
    assert reset["_steps"] == 0


def test_zero_valued_steps_still_means_reset_has_started():
    fresh = _fresh_from_frozen_constructor()
    fresh._steps = 0
    with pytest.raises(RuntimeError, match="reset lifecycle"):
        assert_never_reset_inactive(fresh)


def test_active_or_prepared_fields_are_rejected():
    for field, replacement in [("_active", True),
                               *((field, object()) for field in REQUIRED_NONE)]:
        fresh = _fresh_from_frozen_constructor()
        setattr(fresh, field, replacement)
        with pytest.raises(RuntimeError):
            assert_never_reset_inactive(fresh)


def test_nonempty_metadata_or_missing_initialized_field_is_rejected():
    fresh = _fresh_from_frozen_constructor()
    fresh._episode_metadata = {"already_reset": True}
    with pytest.raises(RuntimeError, match="metadata"):
        assert_never_reset_inactive(fresh)
    fresh = _fresh_from_frozen_constructor()
    del fresh._reference_loop
    with pytest.raises(RuntimeError, match="_reference_loop"):
        assert_never_reset_inactive(fresh)


def test_advanced_nonfinite_or_missing_physical_time_is_rejected():
    for time in (0.002, float("nan"), float("inf")):
        fresh = _fresh_from_frozen_constructor()
        fresh.plant.data.time = time
        with pytest.raises(RuntimeError, match="time"):
            assert_never_reset_inactive(fresh)
    fresh = _fresh_from_frozen_constructor()
    del fresh.plant.data.time
    with pytest.raises(RuntimeError, match="time"):
        assert_never_reset_inactive(fresh)


def test_actual_transfer_calls_same_guard_before_and_after():
    source = Path(__file__).with_name("drive_damping_transfer_env_05.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "install_drive_controller")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "assert_never_reset_inactive"]
    assert len(calls) == 2
    assert all(len(node.args) == 1 and ast.unparse(node.args[0]) == "inner" for node in calls)
    assert not any(isinstance(node, ast.Attribute) and node.attr == "_steps"
                   for node in ast.walk(function))
