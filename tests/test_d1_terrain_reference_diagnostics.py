"""Geometry and intervention guards for the development-only reference probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from wheel_legged_control.d1.controllers import D1Command

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_d1_terrain_reference.py"


@pytest.fixture(scope="module")
def diagnostic() -> ModuleType:
    specification = importlib.util.spec_from_file_location("terrain_reference_diagnostic", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("yaw", (0.0, 0.7, np.pi / 2, -np.pi))
@pytest.mark.parametrize("slopes", ((0.1, 0.0), (-0.1, 0.0), (0.1, -0.15)))
def test_reference_rpy_reconstructs_world_normal(diagnostic, yaw, slopes):
    roll, pitch = diagnostic.normal_to_rpy(*slopes, yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)
    heading_rotation = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    local_z = np.asarray(
        (np.sin(pitch) * np.cos(roll), -np.sin(roll), np.cos(pitch) * np.cos(roll))
    )
    expected = np.asarray((-slopes[0], -slopes[1], 1.0))
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(heading_rotation @ local_z, expected, atol=1e-12)
    if yaw == 0.0:
        assert np.sign(pitch) == -np.sign(slopes[0])


class ReferenceState:
    base_position = np.asarray((0.0, 0.0, 0.455))
    base_rpy = np.asarray((0.01, -0.02, 0.4))
    # Deliberately arbitrary wheel z: the probe must replace it with terrain
    # height under each XY projection, including the two nominally airborne wheels.
    foot_position = np.asarray(
        ((0.25, 0.2, 12.0), (0.25, -0.2, 2.0), (-0.25, 0.2, -8.0), (-0.25, -0.2, 0.0))
    )

    @property
    def wheel_contact(self):
        raise AssertionError("oracle reference must not inspect contact flags")

    @property
    def wheel_contact_point(self):
        raise AssertionError("oracle reference must not inspect measured contact points")


class ReferencePlant:
    control_dt = 0.01

    def __init__(self):
        self.base_position = np.asarray((0.0, 0.0, 0.455))
        self.base_rpy = np.asarray((0.01, -0.02, 0.4))
        self.queries = []

    def training_ground_reference(self, x, y):
        self.queries.append((x, y))
        return SimpleNamespace(height_m=0.1 * x + 0.03 * y + 0.8, pitch_rad=-np.arctan(0.2))


def test_support_plane_uses_exactly_four_xy_ground_projections_without_contacts(diagnostic):
    plant = ReferencePlant()
    state = ReferenceState()
    original_feet = state.foot_position.copy()
    local, support, fit_error = diagnostic.references(plant, state)
    np.testing.assert_allclose(local, diagnostic.normal_to_rpy(0.2, 0.0, 0.4), atol=1e-12)
    np.testing.assert_allclose(support, diagnostic.normal_to_rpy(0.1, 0.03, 0.4), atol=1e-12)
    assert fit_error < 1e-12
    assert plant.queries == [(0.0, 0.0), *map(tuple, original_feet[:, :2])]
    np.testing.assert_array_equal(state.foot_position, original_feet)


def test_degenerate_wheel_projections_are_rejected(diagnostic):
    state = SimpleNamespace(
        base_position=np.zeros(3),
        base_rpy=np.zeros(3),
        foot_position=np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))),
    )
    with pytest.raises(ValueError, match="do not determine"):
        diagnostic.references(ReferencePlant(), state)


def test_rate_uses_same_trajectory_samples_and_does_not_add_initial_jump(diagnostic):
    local = np.asarray((0.5, 0.51, 0.53))
    support = np.asarray((0.5, 0.505, 0.515))
    local_metrics = diagnostic._rate_metrics(local, 0.01, "local")
    support_metrics = diagnostic._rate_metrics(support, 0.01, "support")
    assert local_metrics["local_rate_rms_rps"] == pytest.approx(np.sqrt(2.5))
    assert support_metrics["support_rate_rms_rps"] == pytest.approx(np.sqrt(2.5) / 2)
    assert local_metrics["local_rate_max_abs_rps"] == pytest.approx(2.0)
    assert local_metrics["local_initial_rad"] == 0.5
    single = diagnostic._rate_metrics(np.asarray((0.5,)), 0.01, "single")
    assert single["single_rate_rms_rps"] == single["single_rate_max_abs_rps"] == 0.0


def test_development_cases_are_fixed_complete_and_fresh_each_call(diagnostic):
    cases = diagnostic.development_cases()
    assert len(cases) == len({case["case_id"] for case in cases}) == 12
    assert {
        (
            case["terrain"]["amplitude_m"],
            case["terrain"]["wavelength_m"],
            case["terrain"]["phase_rad"],
        )
        for case in cases
    } == {
        (amplitude, wavelength, phase)
        for amplitude in (0.005, 0.010)
        for wavelength in (0.8, 1.0, 1.2)
        for phase in (0.0, np.pi / 2)
    }
    cases[0]["terrain"]["amplitude_m"] = 99.0
    assert diagnostic.development_cases()[0]["terrain"]["amplitude_m"] == 0.005


class FakeController:
    def __init__(self):
        self.calls = []
        self.gain = np.asarray((2.0, 3.0))

    def compute(self, command, state, residual_force_n=None, **kwargs):
        self.calls.append((command, state, residual_force_n.copy(), kwargs))
        return np.zeros(16)


class FakeEnv:
    """Only checks wrapper plumbing; not a stand-in for physical D1 results."""

    last: ClassVar[FakeEnv]

    def __init__(self, **kwargs):
        FakeEnv.last = self
        self.constructor_kwargs = kwargs
        self.plant = ReferencePlant()
        self.controller = FakeController()
        self.state = ReferenceState()
        self.resets = []
        self.actions = []
        self.closed = False

    def reset(self, **kwargs):
        self.resets.append(kwargs)
        self.step_count = 0
        self.plant.base_position[0] = 0.0

    def step(self, action):
        self.actions.append(action.copy())
        command = D1Command(
            forward_velocity_mps=0.35,
            yaw_rate_rps=0.12,
            base_height_m=0.58,
            roll_rad=0.03,
            pitch_rad=-0.02,
        )
        self.controller.compute(command, self.state, np.zeros(2), vertical_feedforward_force_n=7.0)
        self.step_count += 1
        self.plant.base_position[0] += 0.001
        return (
            None,
            0.0,
            False,
            self.step_count == 2,
            {
                "velocity_error_mps": 0.01,
                "forward_velocity_mps": 0.36,
                "clearance_error_m": 0.002,
                "clearance_m": 0.457,
                "longitudinal_force_n": 5.0,
                "torque_saturation_fraction": 0.0,
                "termination_reason": "time_limit" if self.step_count == 2 else "ongoing",
            },
        )

    def close(self):
        self.closed = True


def test_single_variable_wrapper_preserves_command_fields_seed_action_and_gains(
    diagnostic, monkeypatch
):
    monkeypatch.setattr(diagnostic, "D1TerrainResidualEnv", FakeEnv)
    case = diagnostic.development_cases()[0]
    results = diagnostic.run_case((case, 0.02, 31, 2))
    env = FakeEnv.last
    assert len(results) == len(env.resets) == 6
    assert env.constructor_kwargs == {
        "baseline": "lqr",
        "randomize": False,
        "episode_seconds": 0.02,
        "training_mode": "flat",
    }
    assert all(reset == env.resets[0] for reset in env.resets)
    assert env.resets[0] == {
        "seed": 31,
        "options": {"terrain": case["terrain"], "velocity_mps": 0.35, "height_m": 0.455},
    }
    assert all(np.array_equal(action, np.zeros(2)) for action in env.actions)
    np.testing.assert_array_equal(env.controller.gain, (2.0, 3.0))
    for command, state, residual, kwargs in env.controller.calls:
        assert (command.forward_velocity_mps, command.yaw_rate_rps, command.base_height_m) == (
            0.35,
            0.12,
            0.58,
        )
        assert state is env.state
        np.testing.assert_array_equal(residual, np.zeros(2))
        assert kwargs == {"vertical_feedforward_force_n": 7.0}
    assert env.controller.calls[0][0].roll_rad == 0.03
    assert env.controller.calls[0][0].pitch_rad == -0.02
    assert env.controller.calls[2][0].pitch_rad != -0.02
    assert env.controller.compute.__func__ is FakeController.compute
    assert env.closed
    for first, repeated in zip(results[:3], results[3:], strict=True):
        assert first["telemetry"] == repeated["telemetry"]
        assert "local_pitch_rad" in first["telemetry"][0]
        assert "support_pitch_rad" in first["telemetry"][0]


def test_archived_diagnostic_artifacts_match_their_original_manifest():
    directory = ROOT / "results" / "d1_terrain_reference_diagnosis"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["sha256"]) == {"protocol.json", "summary.json", "telemetry.json"}
    for filename, expected in manifest["sha256"].items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_unchanged"] is True
    assert len(summary["metrics"]) == 72
    assert len(summary["repeat_checks"]) == 36
    assert all(check["telemetry_identical"] for check in summary["repeat_checks"])


def test_process_local_patch_is_restored_when_a_probe_fails(diagnostic, monkeypatch):
    def fail_step(self, action):
        raise RuntimeError("synthetic step failure")

    monkeypatch.setattr(diagnostic, "D1TerrainResidualEnv", FakeEnv)
    monkeypatch.setattr(FakeEnv, "step", fail_step)
    with pytest.raises(RuntimeError, match="synthetic step failure"):
        diagnostic.run_case((diagnostic.development_cases()[0], 0.02, 31, 1))
    assert FakeEnv.last.controller.compute.__func__ is FakeController.compute
    assert FakeEnv.last.closed
