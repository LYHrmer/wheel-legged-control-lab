"""Pure C42 reward-interface checks; no environment, model or physics import."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import math
import struct
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/b22_tracking_reward42.py"
SPEC = importlib.util.spec_from_file_location("tracking_reward42_tests", HELPER)
reward42 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward42)

ORIGINAL_SOURCES = {
    "runtime/d1_b22/research/course_impl08/full_drive_env_08.py":
        "d3895e09e7f3492e58dce3ffe8d572b0223f1639f0ee6c1581c673a7ed1ff787",
    "runtime/d1_b22/research/upright11/world_upright_course_11.py":
        "504cf329cabf889be5e2a6803b442114a6153437c87c992318bb12f8049efdb0",
}


def terms():
    return {"tracking_vx": 1.9997, "tracking_vy": .99, "tracking_yaw": .9998,
            "height": -.001, "attitude": -.002, "actual_torque": -.0001,
            "action_change": -.0002, "leg_speed": -.0003, "termination": 0.}


@pytest.fixture
def actual_reward():
    """Compile only the two pinned reward methods, with synthetic trace data.

    No original module import, constructor, reset, step, engine or policy runs.
    The source methods still compute their original numerical reward terms.
    """
    np = pytest.importorskip("numpy")
    nodes = []
    for name, cls, bases in (
        (next(iter(ORIGINAL_SOURCES)), "FullDriveCourseEnv", []),
        (next(reversed(ORIGINAL_SOURCES)), "WorldUprightCourseEnv",
         [ast.Name(id="FullDriveCourseEnv", ctx=ast.Load())]),
    ):
        path = ROOT / name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SOURCES[name]
        tree = ast.parse(raw)
        source_class = next(n for n in tree.body
                            if isinstance(n, ast.ClassDef) and n.name == cls)
        method = next(n for n in source_class.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_reward")
        nodes.append(ast.ClassDef(name=cls, bases=bases, keywords=[],
                                  body=[copy.deepcopy(method)], decorator_list=[]))
    namespace = {"np": np, "math": math,
                 "LEG_INDICES": tuple(i for i in range(16) if i % 4 != 3)}
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   "<pinned-original-reward-methods>", "exec")
    exec(code, namespace)  # noqa: S102 -- only two SHA-pinned numeric methods
    env = namespace["WorldUprightCourseEnv"]()
    env.plant = SimpleNamespace(
        physics_steps=5, actuator_torque_limit_nm=np.full(16, 8.),
        last_control_interval_actuator_traces=[SimpleNamespace(applied_nm=np.zeros(16))
                                              for _ in range(5)],
    )
    command = SimpleNamespace(forward_velocity_mps=1.2, lateral_velocity_mps=0.,
                              yaw_rate_rps=.3, clearance_m=.455)
    transition = SimpleNamespace(
        decision=SimpleNamespace(motion_command=command,
                                 context=SimpleNamespace(previous_normalized_action=np.zeros(16))),
        receipt=SimpleNamespace(normalized_action=np.zeros(16)),
        truth=SimpleNamespace(joint_velocity=np.zeros(16)),
    )

    def calculate(*, vx_error=0., yaw_error=0., height_error=0., terminated=False):
        metrics = {"body_com_vx_mps": 1.2 + vx_error, "body_com_vy_mps": 0.,
                   "body_yaw_rate_rps": .3 + yaw_error, "clearance_m": .455 + height_error,
                   "relative_roll_rad": .2, "relative_pitch_rad": -.1,
                   "task_roll_error_rad": 0., "task_pitch_error_rad": 0.}
        return env._reward(transition, metrics, terminated)

    return calculate


def test_alpha1_preserves_every_original_value_and_order_without_mutation():
    original = terms()
    original["tracking_vx"] = 1e-25
    original["tracking_yaw"] = 1e-30
    snapshot = original.copy()
    result = reward42.apply_tracking_deficit42(MappingProxyType(original), 1)
    assert result is not original and result == snapshot and list(result) == list(original)
    assert original == snapshot
    for name in original:
        assert struct.pack("!d", result[name]) == struct.pack("!d", original[name])


@pytest.mark.parametrize("alpha", [1, 8])
def test_actual_nine_term_world_reward_interface_only_changes_two_tracking_terms(actual_reward, alpha):
    original = actual_reward(vx_error=.03, yaw_error=-.04, height_error=.005, terminated=True)
    snapshot = original.copy()
    result = reward42.apply_tracking_deficit42(original, alpha)
    assert len(original) == 9 and set(original) == set(reward42.REWARD_TERMS42)
    assert original["attitude"] == 0.
    assert original["termination"] == -10.
    assert original == snapshot and result is not original
    for name in original:
        if name not in ("tracking_vx", "tracking_yaw"):
            assert struct.pack("!d", result[name]) == struct.pack("!d", original[name])
    if alpha == 1:
        assert result == original
    else:
        assert result["tracking_vx"] == 2. + 8. * (original["tracking_vx"] - 2.)
        assert result["tracking_yaw"] == 1. + 8. * (original["tracking_yaw"] - 1.)
        expected_delta = 7. * (original["tracking_vx"] + original["tracking_yaw"] - 3.)
        assert math.isclose(sum(result.values()) - sum(original.values()), expected_delta,
                            rel_tol=0., abs_tol=2e-15)


def test_actual_reward_counterexample_distinguishes_tracking_from_height_tradeoff(actual_reward):
    tracking_better = actual_reward(height_error=.006)
    height_better = actual_reward(vx_error=.005)
    # These synthetic states use the real old reward methods. They do not
    # represent a new trajectory or claim that learning will improve.
    assert sum(height_better.values()) > sum(tracking_better.values())
    modified_tracking = reward42.apply_tracking_deficit42(tracking_better, 8)
    modified_height = reward42.apply_tracking_deficit42(height_better, 8)
    assert sum(modified_tracking.values()) > sum(modified_height.values())
    assert tracking_better["tracking_vx"] > height_better["tracking_vx"]


def test_perfect_tracking_is_unchanged_and_maximal_deficit_is_finite(actual_reward):
    perfect = actual_reward()
    assert reward42.apply_tracking_deficit42(perfect, 8) == perfect
    worst = terms()
    worst.update(tracking_vx=0., tracking_yaw=0.)
    value = reward42.apply_tracking_deficit42(worst, 8)
    assert value["tracking_vx"] == -14. and value["tracking_yaw"] == -7.
    assert all(math.isfinite(x) for x in value.values())


@pytest.mark.parametrize("alpha", [0, 2, 7.999, 9, True, False, "8", None, float("nan"), float("inf")])
def test_rejects_unregistered_alpha(alpha):
    with pytest.raises((TypeError, ValueError)):
        reward42.apply_tracking_deficit42(terms(), alpha)


@pytest.mark.parametrize("fault", ["missing", "extra", "bool", "string", "nan", "inf",
                                   "vx_negative", "vx_too_large", "yaw_negative", "yaw_too_large"])
def test_rejects_wrong_original_reward_interface(fault):
    value = terms()
    if fault == "missing":
        value.pop("termination")
    elif fault == "extra":
        value["hidden_shaping"] = 1.
    elif fault in ("bool", "string", "nan", "inf"):
        value["height"] = {"bool": True, "string": "0", "nan": float("nan"),
                           "inf": float("inf")}[fault]
    else:
        key = "tracking_vx" if fault.startswith("vx_") else "tracking_yaw"
        value[key] = -1e-12 if fault.endswith("negative") else (2.00001 if key.endswith("vx") else 1.00001)
    before = value.copy()
    with pytest.raises((TypeError, ValueError)):
        reward42.apply_tracking_deficit42(value, 8)
    assert list(value) == list(before)
    for key in before:
        assert value[key] is before[key]


def test_helper_cold_import_needs_only_standard_library():
    code = '''
import importlib.abc, importlib.util, sys
blocked = {'torch', 'stable_baselines3', 'mujoco', 'gymnasium', 'numpy', 'cv2', 'onnxruntime'}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unexpected framework import: ' + fullname)
sys.meta_path.insert(0, Deny())
spec = importlib.util.spec_from_file_location('cold_reward42', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = dict.fromkeys(module.REWARD_TERMS42, 0.)
value.update(tracking_vx=2., tracking_vy=1., tracking_yaw=1.)
assert module.apply_tracking_deficit42(value, 8) == value
assert not blocked.intersection(sys.modules)
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(HELPER)],
                            check=False, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
