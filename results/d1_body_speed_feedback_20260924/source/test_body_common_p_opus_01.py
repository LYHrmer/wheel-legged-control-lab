"""Eight pure tests for the fixed common wheel-P candidate (provider: actual Opus).

Four cases exercise the pure arithmetic of ``body_speed_math_06``. Four cases run
the ACTUAL ``BodyCommonPStageController.compute`` body, extracted from
``body_speed_controller_06.py`` by AST and compiled against an explicit fake
parent/provider/state/result seam, with the REAL pure math and the REAL frozen
protection (``drive_damping_math_04.protect_requested``).

No engine, gym, torch, scripts or wheel_legged_control import; no environment, no
model, no file written. Expected values are hand-derived in
``claude_body_tests_design_01.md``; algebraic identities use atol 1e-10 /
rtol 1e-12 and copied arrays are compared exactly.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from body_speed_math_06 import (
    WHEEL_KP,
    WHEEL_RADIUS_M,
    WHEELS,
    body_common_increment,
    finite_array,
    finite_scalar,
)
from drive_damping_math_04 import (
    POSITION_HIGH,
    POSITION_LOW,
    TORQUE_LIMIT,
    VELOCITY_LIMIT,
    protect_requested,
)

ATOL = 1e-10
RTOL = 1e-12
LEGS = tuple(i for i in range(16) if i not in WHEELS)
#: Read literally from scripts/d1_turn_yaw_authority.py (WHEEL_RADIUS_M = 0.087);
#: that module is never imported here because it pulls the engine-facing package.
ORIGINAL_WHEEL_RADIUS_M = 0.087
CONTROLLER_PATH = Path(__file__).resolve().parent / "body_speed_controller_06.py"
EXTRACTED_NAMES = (
    "BODY_RECORD_SCHEMA",
    "_readonly",
    "BodyCommonPRecord",
    "BodyWheelLegResult",
    "BodyCommonPStageController",
)
EXPECTED_SOURCE_MRO = (
    "BodyCommonPRollingController",
    "RollingResidualCompositionController",
    "StopTurnCompositionController",
    "BodyCommonPStageController",
    "DriveDampingStageController",
    "TurnYawAuthorityController",
    "D1WheelLegController",
)
#: Field order of the original wheel_legged_control D1WheelLegResult dataclass.
ORIGINAL_RESULT_FIELDS = (
    "nominal_joint_target_rad",
    "nominal_wheel_speed_rad_s",
    "joint_target_rad",
    "wheel_speed_target_rad_s",
    "leg_extension_target_m",
    "requested_extension_m",
    "clipped_action",
    "leg_pd_nm",
    "support_nm",
    "wheel_nm",
    "requested_torque_nm",
    "torque_nm",
    "support_force_n",
    "joint_target_rate_limited",
    "torque_limited",
    "memory_before",
    "memory_after",
    "effective_yaw_request_rps",
)


# --- explicit fake seam (no production class is imported) --------------------
@dataclass(frozen=True)
class FakeMemory:
    """Stand-in for D1ControllerMemory: only the wheel integral is read."""

    wheel_integral_nm: np.ndarray


@dataclass(frozen=True)
class FakeDriveDampingRecord:
    """Stand-in for the actual intermediate drive-stage record."""

    base_requested_torque_nm: np.ndarray
    base_protected_torque_nm: np.ndarray
    total_requested_torque_nm: np.ndarray
    total_protected_torque_nm: np.ndarray
    delta_torque_nm: np.ndarray


@dataclass(frozen=True)
class FakeAuthorityRecord:
    """Stand-in for the original TurnYaw INTERMEDIATE authority record."""

    wheel_request_nm: np.ndarray
    wheel_torque_nm: np.ndarray
    wheel_integral_before_nm: np.ndarray
    wheel_integral_after_nm: np.ndarray


@dataclass(frozen=True)
class FakeD1WheelLegResult:
    nominal_joint_target_rad: np.ndarray
    nominal_wheel_speed_rad_s: np.ndarray
    joint_target_rad: np.ndarray
    wheel_speed_target_rad_s: np.ndarray
    leg_extension_target_m: np.ndarray
    requested_extension_m: np.ndarray
    clipped_action: np.ndarray
    leg_pd_nm: np.ndarray
    support_nm: np.ndarray
    wheel_nm: np.ndarray
    requested_torque_nm: np.ndarray
    torque_nm: np.ndarray
    support_force_n: np.ndarray
    joint_target_rate_limited: np.ndarray
    torque_limited: np.ndarray
    memory_before: FakeMemory
    memory_after: FakeMemory
    effective_yaw_request_rps: float


@dataclass(frozen=True)
class FakeDriveWheelLegResult(FakeD1WheelLegResult):
    """Preserves the intermediate drive field the new result must carry through."""

    drive_damping: FakeDriveDampingRecord


@dataclass(frozen=True)
class Scenario:
    body_velocity: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    parent_requested: np.ndarray
    parent_wheel_nm: np.ndarray
    parent_leg_pd_nm: np.ndarray
    nominal_wheel_speed_rad_s: np.ndarray
    wheel_speed_target_rad_s: np.ndarray
    pi_before: np.ndarray
    pi_step: np.ndarray
    raw_forward: float
    control_time_s: float = 1.23
    age_s: float = 0.004
    sequence: int = 7


class FakeDriveParent:
    """Explicit fake of the frozen drive/turn/original ancestors.

    ``_checked_zero_action`` and ``_require_binding`` mirror the frozen
    TurnYawAuthorityController validators textually; ``compute`` advances a mock
    PI memory EXACTLY once per call and publishes a real-shaped drive result.
    """

    def __init__(self, *, scenario: Scenario, wheel_kp: float = WHEEL_KP,
                 wheel_ki: float = 3.0, bind: bool = True, **kwargs) -> None:
        self.scenario = scenario
        self.wheel_kp = wheel_kp
        self.wheel_ki = wheel_ki
        self.extra_kwargs = kwargs
        self.binding = _make_binding(scenario) if bind else None
        self.wheel_integral_nm = scenario.pi_before.copy()
        self.compute_calls = 0
        self.reset_calls = 0
        self.received_actions: list = []
        self.received_ground_height: list = []
        self.last_result = None
        self.last_returned_torque = None
        self._last_authority = None

    @property
    def last_authority(self):
        return self._last_authority

    def reset(self) -> None:
        self.reset_calls += 1
        self.last_result = None
        self._last_authority = None

    @staticmethod
    def _checked_zero_action(action) -> np.ndarray:
        array = np.asarray(action)
        if isinstance(action, (bool, np.bool_)) or array.dtype.kind not in ("f", "i", "u"):
            raise TypeError("action must be a real numeric array of eight zeros")
        values = np.ascontiguousarray(array, dtype=np.float64)
        if values.shape != (8,):
            raise ValueError("action must have shape (8,)")
        if not np.isfinite(values).all():
            raise ValueError("action must contain eight finite values")
        if np.any(values != 0.0):
            raise ValueError("only exactly zero residual actions are accepted")
        return values

    def _require_binding(self, command, state):
        if self.binding is None:
            raise RuntimeError("the stage requires a bound raw command for this tick")
        time_s = finite_scalar(state.control_time_s, "state.control_time_s")
        if abs(self.binding.control_time_s - time_s) > 1e-10:
            raise ValueError("bound raw control_time_s does not match the state time")
        forward = finite_scalar(command.forward_velocity_mps, "command.forward_velocity_mps")
        if forward != self.binding.forward_velocity_mps:
            raise ValueError("servo forward velocity must equal the bound raw forward")
        return self.binding

    def compute(self, command, state, action, *, ground_height_m: float = 0.0):
        self.compute_calls += 1
        self.received_actions.append(action)
        self.received_ground_height.append(ground_height_m)
        scenario = self.scenario
        memory_before = FakeMemory(wheel_integral_nm=self.wheel_integral_nm.copy())
        # The one and only PI advance of this tick.
        self.wheel_integral_nm = self.wheel_integral_nm + scenario.pi_step
        memory_after = FakeMemory(wheel_integral_nm=self.wheel_integral_nm.copy())
        requested = scenario.parent_requested.copy()
        protected, limited = protect_requested(requested, scenario.position, scenario.velocity)
        drive = FakeDriveDampingRecord(
            base_requested_torque_nm=requested.copy(),
            base_protected_torque_nm=protected.copy(),
            total_requested_torque_nm=requested.copy(),
            total_protected_torque_nm=protected.copy(),
            delta_torque_nm=np.zeros(16),
        )
        result = FakeDriveWheelLegResult(
            nominal_joint_target_rad=scenario.position.copy(),
            nominal_wheel_speed_rad_s=scenario.nominal_wheel_speed_rad_s.copy(),
            joint_target_rad=scenario.position.copy(),
            wheel_speed_target_rad_s=scenario.wheel_speed_target_rad_s.copy(),
            leg_extension_target_m=np.zeros(4),
            requested_extension_m=np.zeros(4),
            clipped_action=np.zeros(8),
            leg_pd_nm=scenario.parent_leg_pd_nm.copy(),
            support_nm=np.zeros(16),
            wheel_nm=scenario.parent_wheel_nm.copy(),
            requested_torque_nm=requested,
            torque_nm=protected,
            support_force_n=np.zeros(4),
            joint_target_rate_limited=np.zeros(16, dtype=bool),
            torque_limited=limited,
            memory_before=memory_before,
            memory_after=memory_after,
            effective_yaw_request_rps=0.0,
            drive_damping=drive,
        )
        self.last_result = result
        self._last_authority = FakeAuthorityRecord(
            wheel_request_nm=requested[list(WHEELS)].copy(),
            wheel_torque_nm=protected[list(WHEELS)].copy(),
            wheel_integral_before_nm=memory_before.wheel_integral_nm,
            wheel_integral_after_nm=memory_after.wheel_integral_nm,
        )
        self.last_returned_torque = protected.copy()
        return self.last_returned_torque


# --- AST extraction of the actual stage/record/result ------------------------
_SOURCE_TEXT = CONTROLLER_PATH.read_text(encoding="utf-8")
_SOURCE_TREE = ast.parse(_SOURCE_TEXT, filename=str(CONTROLLER_PATH))


def _top_level_name(node) -> str | None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
        return node.name
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)):
        return node.targets[0].id
    return None


def _load_actual_definitions() -> dict:
    """Compile the actual selected top-level nodes against the fake seam.

    The real ClassDef objects are compiled, so the zero-argument ``super()``
    closure inside the actual ``compute``/``reset``/``__init__`` keeps working.
    """
    selected = [node for node in _SOURCE_TREE.body if _top_level_name(node) in EXTRACTED_NAMES]
    found = [_top_level_name(node) for node in selected]
    missing = [name for name in EXTRACTED_NAMES if name not in found]
    if missing:
        raise AssertionError(f"actual source no longer defines {missing}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    harness = ModuleType("body_common_p_ast_harness")
    sys.modules[harness.__name__] = harness
    namespace = harness.__dict__
    namespace.update({
        "np": np,
        "dataclass": dataclass,
        "fields": fields,
        "WHEELS": WHEELS,
        "WHEEL_KP": WHEEL_KP,
        "WHEEL_RADIUS_M": WHEEL_RADIUS_M,
        "ORIGINAL_WHEEL_RADIUS_M": ORIGINAL_WHEEL_RADIUS_M,
        "body_common_increment": body_common_increment,
        "finite_array": finite_array,
        "finite_scalar": finite_scalar,
        "protect_requested": protect_requested,
        "DriveWheelLegResult": FakeDriveWheelLegResult,
        "DriveDampingStageController": FakeDriveParent,
    })
    exec(compile(module, filename=str(CONTROLLER_PATH), mode="exec"), namespace)  # noqa: S102 -- selected local AST under import barrier
    return namespace


_ACTUAL = _load_actual_definitions()


def _make_binding(scenario: Scenario):
    return SimpleNamespace(forward_velocity_mps=scenario.raw_forward, yaw_rate_rps=0.0,
                           control_time_s=scenario.control_time_s)


def _make_state(scenario: Scenario, **overrides):
    values = {
        "base_linear_velocity_body": scenario.body_velocity.copy(),
        "base_rotation": np.eye(3),
        "joint_position": scenario.position.copy(),
        "joint_velocity": scenario.velocity.copy(),
        "control_time_s": scenario.control_time_s,
        "age_s": scenario.age_s,
        "sequence": scenario.sequence,
        "foot_jacobian": np.zeros((4, 3, 4)),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_command(scenario: Scenario):
    return SimpleNamespace(forward_velocity_mps=scenario.raw_forward, yaw_rate_rps=0.0)


def _build_stage(scenario: Scenario, **kwargs):
    return _ACTUAL["BodyCommonPStageController"](scenario=scenario, **kwargs)


def _plain_scenario(*, raw_forward: float = 0.2) -> Scenario:
    """No rated clipping and no outward suppression anywhere."""
    position = np.tile((0.1, 0.5, -1.5, 0.0), 4)
    velocity = np.array((0.2, -0.3, 0.4, 10.0,
                         0.1, 0.0, 0.0, 12.5,
                         0.0, 0.0, 0.0, 9.25,
                         0.0, 0.0, 0.0, 11.0))
    requested = np.array((1.0, -2.0, 3.0, 2.0,
                          1.5, -1.0, 0.5, -2.5,
                          0.0, 0.0, 0.0, 1.0,
                          -1.0, 2.0, -0.5, 0.75))
    wheel_nm = np.zeros(16)
    wheel_nm[list(WHEELS)] = (2.0, -2.5, 1.0, 0.75)
    leg_pd = requested.copy()
    leg_pd[list(WHEELS)] = 0.0
    return Scenario(
        body_velocity=np.array((0.87, 0.0, 0.0)),
        position=position,
        velocity=velocity,
        parent_requested=requested,
        parent_wheel_nm=wheel_nm,
        parent_leg_pd_nm=leg_pd,
        nominal_wheel_speed_rad_s=np.array((11.0, 11.0, 9.5, 9.5)),
        wheel_speed_target_rad_s=np.array((12.0, 12.0, 10.5, 10.5)),
        pi_before=np.array((0.4, -0.2, 1.1, -0.9)),
        pi_step=np.array((0.05, -0.05, 0.10, 0.0)),
        raw_forward=raw_forward,
    )


def _protected_scenario(*, raw_forward: float = 0.25) -> Scenario:
    """Exercises rated clipping plus outward position and speed suppression."""
    position = np.array((0.785398, -1.8326, -1.5, 0.0,
                         0.1, 0.5, -1.5, 0.0,
                         0.1, 0.5, -1.5, 0.0,
                         0.1, 0.5, -1.5, 0.0))
    velocity = np.array((0.0, 0.0, 20.0, 10.0,
                         1.0, -2.0, 0.5, 12.5,
                         0.0, 0.0, 0.0, 9.25,
                         0.0, 0.0, 0.0, 30.0))
    requested = np.array((5.0, -3.0, 7.0, 3.0,
                          4.0, -6.0, 2.0, -13.0,
                          90.0, 0.0, -1.0, 1.5,
                          0.0, 0.0, 0.0, 0.5))
    wheel_nm = np.zeros(16)
    wheel_nm[list(WHEELS)] = (3.0, -13.0, 1.5, 0.5)
    leg_pd = requested.copy()
    leg_pd[list(WHEELS)] = 0.0
    return Scenario(
        body_velocity=np.array((0.87, 0.0, 0.0)),
        position=position,
        velocity=velocity,
        parent_requested=requested,
        parent_wheel_nm=wheel_nm,
        parent_leg_pd_nm=leg_pd,
        nominal_wheel_speed_rad_s=np.array((14.0, 14.0, 12.5, 12.5)),
        wheel_speed_target_rad_s=np.array((15.0, 15.0, 13.0, 13.0)),
        pi_before=np.array((1.0, -1.0, 0.25, 0.5)),
        pi_step=np.array((0.2, 0.2, -0.2, 0.0)),
        raw_forward=raw_forward,
    )


# --- 1-4: actual pure common wheel-P arithmetic ------------------------------
def test_common_and_differential_identity_with_unequal_wheel_values_and_targets():
    """Unequal wheels/targets: exact common shift, differential and I preserved."""
    assert WHEEL_KP == 2.2 and WHEEL_RADIUS_M == ORIGINAL_WHEEL_RADIUS_M == 0.087
    omega = np.array((10.0, 12.5, 9.25, 11.0))
    targets = np.array((8.0, 9.0, 10.5, 12.5))
    integral = np.array((0.5, -0.25, 1.0, -2.0))
    integral_reference = integral.copy()
    body = np.array((0.87, 0.0, 0.0))

    result = body_common_increment(omega, body, raw_forward_mps=0.2)

    # Hand-derived: mean omega 10.6875, vx/r 10.0, error .6875, delta 1.5125 Nm.
    assert result["active"] is True
    assert result["mean_omega_rad_s"] == pytest.approx(10.6875, abs=ATOL, rel=RTOL)
    assert result["body_equivalent_omega_rad_s"] == pytest.approx(10.0, abs=ATOL, rel=RTOL)
    assert result["common_error_rad_s"] == pytest.approx(0.6875, abs=ATOL, rel=RTOL)
    assert result["scalar_delta_nm"] == pytest.approx(1.5125, abs=ATOL, rel=RTOL)

    delta = result["delta_torque_nm"]
    original_p = WHEEL_KP * (targets - omega)          # [-4.4, -7.7, 2.75, 3.3]
    original_request = original_p + integral
    new_request = original_request + delta[list(WHEELS)]
    assert new_request == pytest.approx(
        np.array((-2.3875, -6.4375, 5.2625, 2.8125)), abs=ATOL, rel=RTOL
    )
    # Common part: mean P becomes Kp*(mean target - vx/r) = 0, mean I = -.1875.
    assert float(np.mean(original_p + delta[list(WHEELS)])) == pytest.approx(
        WHEEL_KP * (float(np.mean(targets)) - 10.0), abs=ATOL, rel=RTOL
    )
    assert float(np.mean(new_request)) == pytest.approx(-0.1875, abs=ATOL, rel=RTOL)
    # Differential part and the original integral are untouched.
    assert (new_request - np.mean(new_request)) == pytest.approx(
        original_request - np.mean(original_request), abs=ATOL, rel=RTOL
    )
    assert np.array_equal(integral, integral_reference)
    other_integral = body_common_increment(omega, body, raw_forward_mps=0.2)
    assert np.array_equal(other_integral["delta_torque_nm"], delta)
    assert np.array_equal(omega, np.array((10.0, 12.5, 9.25, 11.0)))


def test_raw_forward_signed_zero_and_nonzero_gating():
    """Both signed zeros are inactive identity; any nonzero raw opens the gate."""
    omega = np.array((10.0, 12.5, 9.25, 11.0))
    body = np.array((0.87, 0.0, 0.0))
    zeros16 = np.zeros(16)

    for inactive_raw in (0.0, -0.0, np.float64(-0.0), 0, np.float64(0.0)):
        result = body_common_increment(omega, body, raw_forward_mps=inactive_raw)
        assert result["active"] is False
        assert result["scalar_delta_nm"] == 0.0
        assert np.array_equal(result["delta_torque_nm"], zeros16)
        # Diagnostics stay real while the gate is closed.
        assert result["common_error_rad_s"] == pytest.approx(0.6875, abs=ATOL, rel=RTOL)
        assert result["mean_omega_rad_s"] == pytest.approx(10.6875, abs=ATOL, rel=RTOL)

    forward = body_common_increment(omega, body, raw_forward_mps=0.2)
    reverse = body_common_increment(omega, body, raw_forward_mps=-0.25)
    tiny = body_common_increment(omega, body, raw_forward_mps=1e-300)
    for active in (forward, reverse, tiny):
        assert active["active"] is True
        assert type(active["active"]) is bool
        # The gate is a pure nonzero test: raw sign and magnitude never scale it.
        assert np.array_equal(active["delta_torque_nm"], forward["delta_torque_nm"])
    assert forward["scalar_delta_nm"] == pytest.approx(1.5125, abs=ATOL, rel=RTOL)


def test_four_equal_wheel_deltas_leg_zeros_units_and_sign():
    """Exactly four equal wheel entries, zero legs, radius units and true sign."""
    body = np.array((0.87, 0.0, 0.0))
    result = body_common_increment(np.array((10.0, 12.5, 9.25, 11.0)), body,
                                  raw_forward_mps=0.2)
    delta = result["delta_torque_nm"]
    assert delta.shape == (16,) and delta.dtype == np.float64
    assert all(delta[index] == result["scalar_delta_nm"] for index in WHEELS)
    assert np.array_equal(delta[list(LEGS)], np.zeros(12))
    assert int(np.count_nonzero(delta)) == 4

    # Sign: wheels faster than the body equivalent -> positive Nm and back.
    faster = body_common_increment(np.array((11.0, 11.0, 11.0, 11.0)), body,
                                  raw_forward_mps=0.2)
    slower = body_common_increment(np.array((9.0, 9.0, 9.0, 9.0)), body,
                                  raw_forward_mps=0.2)
    assert faster["scalar_delta_nm"] > 0.0 and slower["scalar_delta_nm"] < 0.0
    assert faster["scalar_delta_nm"] == pytest.approx(2.2, abs=ATOL, rel=RTOL)
    assert slower["scalar_delta_nm"] == pytest.approx(-2.2, abs=ATOL, rel=RTOL)

    # Units: Kp [Nm/(rad/s)] times (rad/s) from vx / r, r = .087 m.
    stalled = body_common_increment(np.zeros(4), body, raw_forward_mps=0.2)
    assert stalled["body_equivalent_omega_rad_s"] == pytest.approx(10.0, abs=ATOL, rel=RTOL)
    assert stalled["scalar_delta_nm"] == pytest.approx(-22.0, abs=ATOL, rel=RTOL)
    doubled = body_common_increment(np.zeros(4), np.array((1.74, 0.0, 0.0)),
                                   raw_forward_mps=0.2)
    assert doubled["body_equivalent_omega_rad_s"] == pytest.approx(20.0, abs=ATOL, rel=RTOL)

    # Only the body x component is consumed; lateral/vertical are ignored exactly.
    matched = float(0.87 / WHEEL_RADIUS_M)
    balanced = body_common_increment(np.full(4, matched), np.array((0.87, 5.0, -3.0)),
                                     raw_forward_mps=0.2)
    assert balanced["common_error_rad_s"] == 0.0
    assert np.array_equal(balanced["delta_torque_nm"], np.zeros(16))
    assert np.array_equal(
        body_common_increment(np.zeros(4), np.array((0.87, 5.0, -3.0)),
                              raw_forward_mps=0.2)["delta_torque_nm"],
        stalled["delta_torque_nm"],
    )


def test_malformed_and_nonfinite_inputs_rejected():
    """Shape/dtype errors raise TypeError; nonfinite values raise ValueError."""
    omega = np.array((10.0, 12.5, 9.25, 11.0))
    body = np.array((0.87, 0.0, 0.0))

    for bad_omega in (np.zeros(16), np.zeros((4, 1)), np.zeros(3),
                      np.array((1.0, 2.0, 3.0, 4.0), dtype=np.complex128),
                      np.array(("a", "b", "c", "d"))):
        with pytest.raises(TypeError):
            body_common_increment(bad_omega, body, raw_forward_mps=0.2)
    for bad_body in (np.zeros(2), np.zeros((3, 1)), np.zeros(16)):
        with pytest.raises(TypeError):
            body_common_increment(omega, bad_body, raw_forward_mps=0.2)
    for bad_raw in (True, np.bool_(True), None, "0.2", np.array(0.2), [0.2]):
        with pytest.raises(TypeError):
            body_common_increment(omega, body, raw_forward_mps=bad_raw)

    for bad_omega in (np.array((np.nan, 1.0, 2.0, 3.0)), np.array((np.inf, 1.0, 2.0, 3.0))):
        with pytest.raises(ValueError):
            body_common_increment(bad_omega, body, raw_forward_mps=0.2)
    for bad_body in (np.array((np.nan, 0.0, 0.0)), np.array((0.87, 0.0, -np.inf))):
        with pytest.raises(ValueError):
            body_common_increment(omega, bad_body, raw_forward_mps=0.2)
    for bad_raw in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            body_common_increment(omega, body, raw_forward_mps=bad_raw)

    # Integer/list inputs remain valid real numeric input.
    accepted = body_common_increment(np.array((10, 12, 9, 11), dtype=np.int64),
                                     [0.87, 0.0, 0.0], raw_forward_mps=1)
    assert accepted["mean_omega_rad_s"] == pytest.approx(10.5, abs=ATOL, rel=RTOL)


# --- 5-8: the actual stage compute through the pure fake seam ---------------
def test_stage_calls_parent_once_and_does_not_advance_pi_twice():
    """One parent compute per tick, one PI advance, recorded before/after exact."""
    scenario = _plain_scenario()
    stage = _build_stage(scenario)
    assert stage.last_body_common_p is None
    state = _make_state(scenario)
    torque = stage.compute(_make_command(scenario), state, np.zeros(8), ground_height_m=0.0)

    parent_before = scenario.pi_before
    parent_after = scenario.pi_before + scenario.pi_step
    assert stage.compute_calls == 1
    assert len(stage.received_actions) == 1
    checked = stage.received_actions[0]
    assert isinstance(checked, np.ndarray) and checked.shape == (8,)
    assert checked.dtype == np.float64 and np.array_equal(checked, np.zeros(8))
    assert np.array_equal(stage.wheel_integral_nm, parent_after)
    assert stage.wheel_integral_nm == pytest.approx(
        np.array((0.45, -0.25, 1.2, -0.9)), abs=ATOL, rel=RTOL
    )

    record = stage.last_body_common_p
    assert record.schema == _ACTUAL["BODY_RECORD_SCHEMA"] == "d1-drive-body-common-p-record-v1"
    assert np.array_equal(record.original_pi_integral_before_nm, parent_before)
    assert np.array_equal(record.original_pi_integral_after_nm, parent_after)
    assert record.active is True and record.provider_state_sequence == scenario.sequence
    assert record.raw_forward_mps == scenario.raw_forward
    assert record.servo_forward_mps == scenario.raw_forward
    assert record.control_time_s == scenario.control_time_s
    assert record.provider_state_age_s == scenario.age_s
    assert record.wheel_kp == WHEEL_KP and record.wheel_radius_m == WHEEL_RADIUS_M

    # The record reproduces the real pure arithmetic of this tick.
    expected = body_common_increment(scenario.velocity[list(WHEELS)], scenario.body_velocity,
                                     raw_forward_mps=scenario.raw_forward)
    assert np.array_equal(record.wheel_omega_rad_s, scenario.velocity[list(WHEELS)])
    assert record.common_error_rad_s == expected["common_error_rad_s"]
    assert record.scalar_delta_nm == expected["scalar_delta_nm"]
    assert record.scalar_delta_nm == pytest.approx(1.5125, abs=ATOL, rel=RTOL)
    assert np.array_equal(record.delta_torque_nm, expected["delta_torque_nm"])
    assert np.array_equal(torque, record.total_protected_torque_nm)
    assert isinstance(stage.last_result, _ACTUAL["BodyWheelLegResult"])
    assert stage.last_result.body_common_p is record

    # reset clears only the new diagnostic and delegates once.
    stage.reset()
    assert stage.last_body_common_p is None and stage.reset_calls == 1
    assert stage.compute_calls == 1

    # Constants guard: the stage refuses gains other than the original PI pair.
    with pytest.raises(RuntimeError):
        _build_stage(scenario, wheel_kp=2.5)
    with pytest.raises(RuntimeError):
        _build_stage(scenario, wheel_ki=4.0)


def test_stage_adds_delta_to_unprotected_request_then_original_protections():
    """Delta hits the true unprotected request; then clip, position and speed."""
    scenario = _protected_scenario()
    stage = _build_stage(scenario)
    torque = stage.compute(_make_command(scenario), _make_state(scenario), np.zeros(8))
    record = stage.last_body_common_p
    parent_protected, parent_limited = protect_requested(
        scenario.parent_requested, scenario.position, scenario.velocity
    )

    # Hand-derived: mean omega 15.4375, vx/r 10, error 5.4375, delta 11.9625 Nm.
    assert record.scalar_delta_nm == pytest.approx(11.9625, abs=ATOL, rel=RTOL)
    expected_requested = scenario.parent_requested + record.delta_torque_nm
    assert np.array_equal(record.total_requested_torque_nm, expected_requested)
    assert np.array_equal(record.parent_requested_torque_nm, scenario.parent_requested)
    assert np.array_equal(record.parent_protected_torque_nm, parent_protected)
    # Wheel 7 discriminates unprotected (-13 + delta) from protected (-12 + delta).
    assert record.total_requested_torque_nm[7] == pytest.approx(-1.0375, abs=ATOL, rel=RTOL)
    assert abs(record.total_requested_torque_nm[7]
               - (parent_protected[7] + record.scalar_delta_nm)) == pytest.approx(
        1.0, abs=ATOL, rel=RTOL)

    expected_protected = np.array((0.0, 0.0, 0.0, 12.0,
                                   4.0, -6.0, 2.0, -1.0375,
                                   80.0, 0.0, -1.0, 12.0,
                                   0.0, 0.0, 0.0, 0.0))
    expected_limited = np.array((True, True, True, True,
                                 False, False, False, False,
                                 True, False, False, True,
                                 False, False, False, True))
    assert record.total_protected_torque_nm == pytest.approx(expected_protected, abs=ATOL, rel=RTOL)
    assert np.array_equal(record.total_torque_limited, expected_limited)
    assert torque == pytest.approx(expected_protected, abs=ATOL, rel=RTOL)
    # Exactly the frozen protection of the summed request, in the original order.
    frozen_protected, frozen_limited = protect_requested(
        expected_requested, scenario.position, scenario.velocity
    )
    assert np.array_equal(record.total_protected_torque_nm, frozen_protected)
    assert np.array_equal(record.total_torque_limited, frozen_limited)
    assert np.array_equal(stage.last_result.torque_nm, frozen_protected)
    assert np.array_equal(stage.last_result.requested_torque_nm, expected_requested)
    assert np.array_equal(stage.last_result.torque_limited, frozen_limited)
    assert abs(record.total_protected_torque_nm[3]) <= TORQUE_LIMIT[3]
    assert POSITION_HIGH[0] == scenario.position[0] and POSITION_LOW[1] == scenario.position[1]
    assert VELOCITY_LIMIT[2] == scenario.velocity[2] and VELOCITY_LIMIT[15] == scenario.velocity[15]

    # Unprotected wheel sum is recorded without clipping; the actual increment
    # is what protection really allowed (wheel 15 loses the whole delta).
    expected_wheel = scenario.parent_wheel_nm + record.delta_torque_nm
    assert np.array_equal(record.total_wheel_nm, expected_wheel)
    assert np.array_equal(stage.last_result.wheel_nm, expected_wheel)
    assert record.total_wheel_nm[list(WHEELS)] == pytest.approx(
        np.array((14.9625, -1.0375, 13.4625, 12.4625)), abs=ATOL, rel=RTOL
    )
    assert np.array_equal(record.actual_protected_increment_nm,
                          frozen_protected - parent_protected)
    assert record.actual_protected_increment_nm[list(WHEELS)] == pytest.approx(
        np.array((9.0, 10.9625, 10.5, 0.0)), abs=ATOL, rel=RTOL
    )
    assert np.array_equal(record.actual_protected_increment_nm[list(LEGS)], np.zeros(12))
    assert np.array_equal(parent_limited, protect_requested(
        scenario.parent_requested, scenario.position, scenario.velocity)[1])
    assert stage.compute_calls == 1


def test_stage_preserves_targets_intermediate_records_and_inactive_identity():
    """Nominal/final targets, drive and authority intermediates, inactive path."""
    # The extracted definitions really are the cooperative subclasses in source.
    stage_node = next(node for node in _SOURCE_TREE.body
                      if isinstance(node, ast.ClassDef) and node.name == "BodyCommonPStageController")
    result_node = next(node for node in _SOURCE_TREE.body
                       if isinstance(node, ast.ClassDef) and node.name == "BodyWheelLegResult")
    assert [base.id for base in stage_node.bases] == ["DriveDampingStageController"]
    assert [base.id for base in result_node.bases] == ["DriveWheelLegResult"]
    mro_node = next(node for node in _SOURCE_TREE.body
                    if _top_level_name(node) == "EXPECTED_MRO")
    assert tuple(element.value for element in mro_node.value.elts) == EXPECTED_SOURCE_MRO
    assert tuple(field.name for field in fields(FakeD1WheelLegResult)) == ORIGINAL_RESULT_FIELDS

    scenario = _protected_scenario()
    stage = _build_stage(scenario)
    stage.compute(_make_command(scenario), _make_state(scenario), np.zeros(8))
    parent = stage.last_result
    record = stage.last_body_common_p
    parent_protected, parent_limited = protect_requested(
        scenario.parent_requested, scenario.position, scenario.velocity
    )

    # Targets and the original leg/support components pass through untouched.
    assert np.array_equal(record.nominal_wheel_speed_rad_s, scenario.nominal_wheel_speed_rad_s)
    assert np.array_equal(record.wheel_speed_target_rad_s, scenario.wheel_speed_target_rad_s)
    assert np.array_equal(parent.nominal_wheel_speed_rad_s, scenario.nominal_wheel_speed_rad_s)
    assert np.array_equal(parent.wheel_speed_target_rad_s, scenario.wheel_speed_target_rad_s)
    assert np.array_equal(parent.leg_pd_nm, scenario.parent_leg_pd_nm)
    assert np.array_equal(parent.support_nm, np.zeros(16))
    assert parent.memory_before.wheel_integral_nm is not None
    assert np.array_equal(parent.memory_before.wheel_integral_nm, scenario.pi_before)

    # The drive intermediate record is carried through as the same object and
    # is NOT the final torque; the authority record stays the intermediate too.
    drive = parent.drive_damping
    assert isinstance(drive, FakeDriveDampingRecord)
    assert np.array_equal(drive.base_requested_torque_nm, scenario.parent_requested)
    assert np.array_equal(drive.total_protected_torque_nm, parent_protected)
    assert not np.array_equal(drive.total_protected_torque_nm, parent.torque_nm)
    authority = stage.last_authority
    assert np.array_equal(authority.wheel_request_nm, scenario.parent_requested[list(WHEELS)])
    assert np.array_equal(authority.wheel_torque_nm, parent_protected[list(WHEELS)])
    assert not np.array_equal(authority.wheel_torque_nm,
                              np.asarray(parent.torque_nm)[list(WHEELS)])

    # Inactive tick: exact parent identity, diagnostics still recorded.
    inactive_scenario = _protected_scenario(raw_forward=-0.0)
    inactive_stage = _build_stage(inactive_scenario)
    returned = inactive_stage.compute(_make_command(inactive_scenario),
                                      _make_state(inactive_scenario), np.zeros(8))
    inactive = inactive_stage.last_body_common_p
    assert returned is inactive_stage.last_returned_torque
    assert inactive.active is False and inactive.scalar_delta_nm == 0.0
    assert np.array_equal(inactive.delta_torque_nm, np.zeros(16))
    assert np.array_equal(inactive.total_requested_torque_nm, inactive_scenario.parent_requested)
    assert np.array_equal(inactive.total_protected_torque_nm, parent_protected)
    assert np.array_equal(inactive.total_wheel_nm, inactive_scenario.parent_wheel_nm)
    assert np.array_equal(inactive.actual_protected_increment_nm, np.zeros(16))
    assert np.array_equal(inactive.total_torque_limited, parent_limited)
    assert inactive.common_error_rad_s == pytest.approx(5.4375, abs=ATOL, rel=RTOL)
    assert inactive_stage.compute_calls == 1
    assert np.array_equal(inactive_stage.wheel_integral_nm,
                          inactive_scenario.pi_before + inactive_scenario.pi_step)


def test_stage_rejects_invalid_action_raw_and_state_before_pi_mutation():
    """Every pre-parent rejection leaves the PI memory and records untouched."""
    cases = (
        ("nonzero residual action", ValueError, {}, {}, {"action": np.ones(8)}),
        ("wrong action shape", ValueError, {}, {}, {"action": np.zeros(7)}),
        ("non-numeric action", TypeError, {}, {}, {"action": np.array(["0"] * 8)}),
        ("missing raw binding", RuntimeError, {"bind": False}, {}, {}),
        ("nonfinite raw/servo forward", ValueError, {"raw_forward": np.inf}, {}, {}),
        ("nonfinite body velocity", ValueError, {},
         {"base_linear_velocity_body": np.array((0.87, np.nan, 0.0))}, {}),
        ("wrong joint velocity shape", TypeError, {}, {"joint_velocity": np.zeros(15)}, {}),
        ("nonfinite joint position", ValueError, {},
         {"joint_position": np.full(16, np.nan)}, {}),
        ("nonfinite provider age", ValueError, {}, {"age_s": np.inf}, {}),
        ("non-integer provider sequence", TypeError, {}, {"sequence": 7.0}, {}),
    )
    for label, error, scenario_kwargs, state_overrides, call_kwargs in cases:
        raw_forward = scenario_kwargs.pop("raw_forward", 0.2)
        scenario = _plain_scenario(raw_forward=raw_forward)
        stage = _build_stage(scenario, **scenario_kwargs)
        state = _make_state(scenario, **state_overrides)
        action = call_kwargs.get("action", np.zeros(8))
        with pytest.raises(error):
            stage.compute(_make_command(scenario), state, action)
        assert stage.compute_calls == 0, label
        assert stage.received_actions == [], label
        assert np.array_equal(stage.wheel_integral_nm, scenario.pi_before), label
        assert stage.last_result is None and stage.last_authority is None, label
        assert stage.last_body_common_p is None, label

    # A valid tick on the same construction still works afterwards.
    scenario = _plain_scenario()
    stage = _build_stage(scenario)
    stage.compute(_make_command(scenario), _make_state(scenario), np.zeros(8))
    assert stage.compute_calls == 1
    assert np.array_equal(stage.wheel_integral_nm, scenario.pi_before + scenario.pi_step)
