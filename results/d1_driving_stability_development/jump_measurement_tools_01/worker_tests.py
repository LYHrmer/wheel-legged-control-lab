"""Unit tests for the pure D1 jump-readiness measurement utility.

These tests exercise ``scripts/d1_jump_readiness.py`` only.  They compile the
D1 model once (model compilation and pure analytic kinematics are the only
MuJoCo work allowed here) and never construct ``MjData``, never integrate,
forward, reset or solve contacts, and never build an interactive simulation,
plant or controller.

Expected values are derived independently of the module under test: the fixed
command schedule is re-tabulated from the contract, the compiled wheel/plane
binding is cross-checked against ``refs/compiled_geometry.json``, and every
cylinder/plane clearance is compared with a brute-force enumeration of the
lowest point over both cylinder rim circles.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_WORK_DIR = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _WORK_DIR / "scripts" / "d1_jump_readiness.py"
_SRC_PATH = _WORK_DIR / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


def _load_module_under_test():
    spec = importlib.util.spec_from_file_location(
        "d1_jump_readiness_under_test", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jr = _load_module_under_test()
D1MotionCommand = type(jr.fixed_height_command(0, "hold")[0])

_REFERENCE_GEOMETRY = json.loads(
    (_WORK_DIR / "refs" / "compiled_geometry.json").read_text()
)

# MuJoCo geom/joint type codes used by the synthetic read-only models below.
_PLANE, _CYLINDER, _BOX, _MESH = 0, 5, 6, 7
_JNT_FREE, _JNT_BALL, _JNT_HINGE = 0, 1, 3

_WHEEL_RADIUS_M = 0.087
_WHEEL_HALF_LENGTH_M = 0.02
_WHEEL_MARGIN_M = 0.001

# The one permitted height schedule, re-tabulated from refs/next_contract.md
# (execution ticks, profile clearance, profile label).  Hold clearance is a flat
# 0.455 m for every tick of the matched control episode.
_CONTRACT_PROFILE_TABLE = (
    (0, 199, 0.455, "settle"),
    (200, 224, 0.405, "crouch_request"),
    (225, 239, 0.500, "extension_request"),
    (240, 274, 0.455, "return_request"),
    (275, 319, 0.435, "lower_request"),
    (320, 599, 0.455, "hold"),
)
_CONTRACT_HOLD_CLEARANCE_M = 0.455


def _expected_profile(tick: int) -> tuple[float, str]:
    """Independent contract lookup; tick 600 repeats the final held command."""

    lookup = min(int(tick), 599)
    for first, last, height, label in _CONTRACT_PROFILE_TABLE:
        if first <= lookup <= last:
            return height, label
    raise AssertionError(f"contract table does not cover tick {tick}")


# ---------------------------------------------------------------------------
# Independent geometry reference
# ---------------------------------------------------------------------------


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    return array / float(np.linalg.norm(array))


def _matrix_from_quaternion(quat) -> np.ndarray:
    """Body-to-world rotation from a MuJoCo ``(w, x, y, z)`` quaternion."""

    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _brute_force_gap(
    center, axis, radius, half_length, normal, offset=0.0, samples=60001
) -> float:
    """Lowest signed plane distance over both cylinder rim circles.

    A linear functional over a solid cylinder attains its minimum at an extreme
    point, and the extreme points of a cylinder are exactly the two rim
    circles, so enumerating the rims is an independent support calculation that
    shares no algebra with the module under test.
    """

    center = np.asarray(center, dtype=float)
    axis = _unit(axis)
    normal = np.asarray(normal, dtype=float)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(axis, helper))
    second = np.cross(axis, first)
    phi = np.linspace(0.0, 2.0 * math.pi, samples)
    rim = (
        center
        + radius * np.cos(phi)[:, None] * first
        + radius * np.sin(phi)[:, None] * second
    )
    points = np.vstack([rim + half_length * axis, rim - half_length * axis])
    return float(np.min(points @ normal) - offset)


# ---------------------------------------------------------------------------
# Read-only synthetic compiled models
# ---------------------------------------------------------------------------

# A non-identity local transform on the visual mesh geoms, copied from the real
# compiled model: a facility that read the visual geom instead of the collision
# cylinder would both mis-size and mis-orient the wheel.
_VISUAL_MESH_QUAT = (
    0.3363461230559388,
    0.6219892632064173,
    -0.3363453342839142,
    0.6219907218493073,
)
_VISUAL_MESH_SIZE = (0.021065951911499986, 0.09250137675324546, 0.09328044084893482)
_VISUAL_MESH_POS = (1.4408285473552165e-06, 3.7473298642068114e-07, -0.0007966028223543898)


def _geom_entry(
    body,
    geom_type,
    size,
    *,
    margin=0.0,
    gap=0.0,
    contype=1,
    conaffinity=1,
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    name="",
):
    return {
        "body": int(body),
        "type": int(geom_type),
        "size": tuple(float(v) for v in size),
        "margin": float(margin),
        "gap": float(gap),
        "contype": int(contype),
        "conaffinity": int(conaffinity),
        "pos": tuple(float(v) for v in pos),
        "quat": tuple(float(v) for v in quat),
        "name": str(name),
    }


def _default_geom_entries():
    """Floor plane plus a visual mesh and a collision cylinder per wheel body."""

    entries = [_geom_entry(0, _PLANE, (20.0, 20.0, 0.05), name="floor")]
    for body in (1, 2, 3, 4):
        entries.append(
            _geom_entry(
                body,
                _MESH,
                _VISUAL_MESH_SIZE,
                contype=0,
                conaffinity=0,
                pos=_VISUAL_MESH_POS,
                quat=_VISUAL_MESH_QUAT,
            )
        )
        entries.append(
            _geom_entry(
                body,
                _CYLINDER,
                (_WHEEL_RADIUS_M, _WHEEL_HALF_LENGTH_M, 0.0),
                margin=_WHEEL_MARGIN_M,
            )
        )
    return entries


class _SyntheticModel:
    """Minimal read-only stand-in exposing compiled arrays and name lookup only.

    It has no ``MjData``, no step/forward/reset entry point and no mutable
    state, so a facility that tried to refresh or solve anything would fail
    loudly instead of silently working.
    """

    class _NamedView:
        __slots__ = ("name",)

        def __init__(self, name):
            self.name = name

    def __init__(
        self,
        geoms=None,
        *,
        body_names=("world", "FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        body_jntadr=(-1, 1, 2, 3, 4),
        body_jntnum=(0, 1, 1, 1, 1),
        jnt_type=(_JNT_FREE, _JNT_HINGE, _JNT_HINGE, _JNT_HINGE, _JNT_HINGE),
        nhfield=0,
    ):
        entries = list(_default_geom_entries() if geoms is None else geoms)
        self._body_names = tuple(str(n) for n in body_names)
        self._geom_names = tuple(entry["name"] for entry in entries)
        self.nbody = len(self._body_names)
        self.ngeom = len(entries)
        self.njnt = len(jnt_type)
        self.nhfield = int(nhfield)
        self.geom_bodyid = np.array([e["body"] for e in entries], dtype=np.int32)
        self.geom_type = np.array([e["type"] for e in entries], dtype=np.int32)
        self.geom_size = np.array([e["size"] for e in entries], dtype=np.float64)
        self.geom_pos = np.array([e["pos"] for e in entries], dtype=np.float64)
        self.geom_quat = np.array([e["quat"] for e in entries], dtype=np.float64)
        self.geom_margin = np.array([e["margin"] for e in entries], dtype=np.float64)
        self.geom_gap = np.array([e["gap"] for e in entries], dtype=np.float64)
        self.geom_contype = np.array([e["contype"] for e in entries], dtype=np.int32)
        self.geom_conaffinity = np.array(
            [e["conaffinity"] for e in entries], dtype=np.int32
        )
        self.body_jntadr = np.array(body_jntadr, dtype=np.int32)
        self.body_jntnum = np.array(body_jntnum, dtype=np.int32)
        self.jnt_type = np.array(jnt_type, dtype=np.int32)

    def geom(self, index):
        return self._NamedView(self._geom_names[int(index)])

    def body(self, index):
        return self._NamedView(self._body_names[int(index)])


_SYNTHETIC_WHEEL_BODY_IDS = (1, 2, 3, 4)
_SYNTHETIC_WHEEL_GEOM_IDS = (2, 4, 6, 8)


@pytest.fixture()
def synthetic_model():
    return _SyntheticModel()


@pytest.fixture()
def synthetic_binding(synthetic_model):
    return jr.bind_wheel_plane_geometry(synthetic_model, _SYNTHETIC_WHEEL_BODY_IDS)


def _flat_frames(ngeom, wheel_geom_ids, plane_geom_id, wheel_heights, wheel_rotation=None):
    """Synchronized-style ``geom_xpos``/``geom_xmat`` arrays for a level floor."""

    rotation = _rotation_x(-0.5 * math.pi) if wheel_rotation is None else wheel_rotation
    xpos = np.zeros((ngeom, 3), dtype=np.float64)
    xmat = np.tile(np.eye(3, dtype=np.float64).reshape(9), (ngeom, 1))
    xpos[plane_geom_id] = (0.0, 0.0, 0.0)
    for slot, (gid, height) in enumerate(zip(wheel_geom_ids, wheel_heights)):
        xpos[gid] = (0.3 * (slot + 1), -0.2 * (slot + 1), float(height))
        xmat[gid] = np.asarray(rotation, dtype=np.float64).reshape(9)
    return xpos, xmat


def _binding_kwargs(**overrides):
    kwargs = dict(
        schema=jr.JUMP_READINESS_SCHEMA,
        plane_geom_id=0,
        wheel_geom_ids=_SYNTHETIC_WHEEL_GEOM_IDS,
        wheel_body_ids=_SYNTHETIC_WHEEL_BODY_IDS,
        radii_m=(_WHEEL_RADIUS_M,) * 4,
        half_lengths_m=(_WHEEL_HALF_LENGTH_M,) * 4,
        contact_margins_m=(_WHEEL_MARGIN_M,) * 4,
        model_ngeom=9,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Documented API surface
# ---------------------------------------------------------------------------


def test_public_api_surface_is_stable():
    assert set(jr.__all__) == {
        "JUMP_READINESS_SCHEMA",
        "JUMP_READINESS_PHASES",
        "JUMP_READINESS_CONDITIONS",
        "TERMINAL_PREPARED_TICK",
        "fixed_height_command",
        "WheelPlaneBinding",
        "bind_wheel_plane_geometry",
        "oriented_cylinder_gaps",
        "sample_wheel_clearance",
        "sampled_true_runs",
    }
    assert jr.TERMINAL_PREPARED_TICK == 600
    assert jr.JUMP_READINESS_CONDITIONS == ("hold", "profile")
    assert set(label for _, _, _, label in _CONTRACT_PROFILE_TABLE) <= set(
        jr.JUMP_READINESS_PHASES
    )
    for constant in (jr.JUMP_READINESS_CONDITIONS, jr.JUMP_READINESS_PHASES):
        assert isinstance(constant, tuple)


def test_module_stays_pure_and_adds_no_dependency():
    """Static check: no integration, plant, data or new third-party import."""

    tree = ast.parse(_SCRIPT_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "math", "dataclasses", "typing", "numpy", "mujoco",
                        "wheel_legged_control"}

    identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    identifiers |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    identifiers |= {
        alias.asname or alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = {
        "mj_step",
        "mj_step1",
        "mj_step2",
        "mj_forward",
        "mj_forwardSkip",
        "mj_kinematics",
        "mj_fwdPosition",
        "mj_comPos",
        "mj_collision",
        "mj_resetData",
        "MjData",
        "MjSpec",
        "D1Plant",
        "D1InteractiveSimulation",
        "D1LQRVMCController",
        "D1MPCVMCController",
        "build_d1_model",
    }
    assert identifiers.isdisjoint(forbidden)


# ---------------------------------------------------------------------------
# Fixed command schedule
# ---------------------------------------------------------------------------


def test_profile_schedule_matches_contract_table_for_every_prepared_tick():
    for tick in range(0, jr.TERMINAL_PREPARED_TICK + 1):
        command, label = jr.fixed_height_command(tick, "profile")
        height, expected_label = _expected_profile(tick)
        assert isinstance(command, D1MotionCommand)
        assert command.clearance_m == pytest.approx(height, abs=0.0)
        assert label == expected_label


def test_hold_schedule_is_a_flat_matched_control():
    labels = set()
    for tick in range(0, jr.TERMINAL_PREPARED_TICK + 1):
        command, label = jr.fixed_height_command(tick, "hold")
        assert command.clearance_m == _CONTRACT_HOLD_CLEARANCE_M
        assert label == ("settle" if tick < 200 else "hold")
        labels.add(label)
    assert labels == {"settle", "hold"}


def test_command_boundaries_change_exactly_at_the_contract_edges():
    expected = {
        199: (0.455, "settle"),
        200: (0.405, "crouch_request"),
        224: (0.405, "crouch_request"),
        225: (0.500, "extension_request"),
        239: (0.500, "extension_request"),
        240: (0.455, "return_request"),
        274: (0.455, "return_request"),
        275: (0.435, "lower_request"),
        319: (0.435, "lower_request"),
        320: (0.455, "hold"),
        599: (0.455, "hold"),
    }
    for tick, (height, label) in expected.items():
        command, actual = jr.fixed_height_command(tick, "profile")
        assert (command.clearance_m, actual) == (height, label)
    # Each request boundary is a real step in the raw command.
    for edge in (200, 225, 240, 275, 320):
        before = jr.fixed_height_command(edge - 1, "profile")[0].clearance_m
        after = jr.fixed_height_command(edge, "profile")[0].clearance_m
        assert before != after


def test_terminal_prepared_tick_600_repeats_the_final_held_command():
    for condition in jr.JUMP_READINESS_CONDITIONS:
        final_command, final_label = jr.fixed_height_command(599, condition)
        terminal_command, terminal_label = jr.fixed_height_command(600, condition)
        assert terminal_command == final_command
        assert terminal_label == final_label == "hold"
        assert terminal_command.clearance_m == _CONTRACT_HOLD_CLEARANCE_M
    with pytest.raises(ValueError, match="out of range"):
        jr.fixed_height_command(601, "profile")


def test_paired_episodes_are_identical_through_tick_199_and_first_differ_at_200():
    for tick in range(0, 200):
        hold = jr.fixed_height_command(tick, "hold")
        profile = jr.fixed_height_command(tick, "profile")
        assert hold == profile
    assert (
        jr.fixed_height_command(200, "hold")[0].clearance_m
        != jr.fixed_height_command(200, "profile")[0].clearance_m
    )


def test_raw_forward_and_yaw_are_positive_zero_so_both_mechanisms_stay_inactive():
    for condition in jr.JUMP_READINESS_CONDITIONS:
        for tick in (0, 200, 239, 320, 599, 600):
            command, _ = jr.fixed_height_command(tick, condition)
            for value in (command.forward_velocity_mps, command.yaw_rate_rps):
                assert type(value) is float
                assert value == 0.0
                assert math.copysign(1.0, value) == 1.0


def test_every_commanded_clearance_lies_inside_the_command_envelope():
    heights = set()
    for condition in jr.JUMP_READINESS_CONDITIONS:
        for tick in range(0, jr.TERMINAL_PREPARED_TICK + 1):
            clearance = jr.fixed_height_command(tick, condition)[0].clearance_m
            assert 0.38 <= clearance <= 0.53
            heights.add(clearance)
    assert heights == {0.405, 0.435, 0.455, 0.500}


def test_schedule_is_stateless_and_order_independent():
    forward = [jr.fixed_height_command(t, "profile") for t in range(0, 601)]
    backward = [jr.fixed_height_command(t, "profile") for t in range(600, -1, -1)]
    assert forward == backward[::-1]
    # No latch: repeating one tick never advances anything.
    assert [jr.fixed_height_command(300, "profile") for _ in range(3)] == [
        jr.fixed_height_command(300, "profile")
    ] * 3


def test_returned_command_is_immutable():
    command, _ = jr.fixed_height_command(0, "hold")
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        command.clearance_m = 0.5


@pytest.mark.parametrize(
    "tick",
    [True, False, 0.0, 200.0, 199.5, "200", None, np.float64(200.0), [200], b"200"],
)
def test_invalid_tick_types_are_rejected(tick):
    with pytest.raises(TypeError):
        jr.fixed_height_command(tick, "hold")


@pytest.mark.parametrize("tick", [-1, -600, 601, 1200, 6000])
def test_out_of_range_ticks_are_rejected(tick):
    with pytest.raises(ValueError):
        jr.fixed_height_command(tick, "hold")


def test_numpy_integer_ticks_are_accepted_like_python_ints():
    assert jr.fixed_height_command(np.int64(225), "profile") == jr.fixed_height_command(
        225, "profile"
    )


@pytest.mark.parametrize("condition", [True, False, None, 0, 1, b"hold", ["hold"]])
def test_invalid_condition_types_are_rejected(condition):
    with pytest.raises(TypeError):
        jr.fixed_height_command(0, condition)


@pytest.mark.parametrize(
    "condition", ["Hold", "HOLD", "profile ", "", "stationary_height_hold", "jump"]
)
def test_unknown_condition_strings_are_rejected(condition):
    with pytest.raises(ValueError):
        jr.fixed_height_command(0, condition)


# ---------------------------------------------------------------------------
# Oriented cylinder / plane clearance
# ---------------------------------------------------------------------------


def test_upright_and_horizontal_cylinders_match_independent_support_values():
    centers = np.array(
        [
            [0.0, 0.0, 0.30],
            [1.0, 0.0, 0.30],
            [0.0, 1.0, 0.30],
            [1.0, 1.0, 0.30],
        ]
    )
    axes = np.array(
        [
            [0.0, 0.0, 1.0],  # upright: flat face down, bottom at c_z - L
            [0.0, 0.0, -1.0],  # axis sign must not matter
            [0.0, 1.0, 0.0],  # horizontal rolling wheel, bottom at c_z - R
            [1.0, 0.0, 0.0],
        ]
    )
    radii = np.full(4, _WHEEL_RADIUS_M)
    halves = np.full(4, _WHEEL_HALF_LENGTH_M)
    gaps = jr.oriented_cylinder_gaps(centers, axes, radii, halves)

    expected = [
        _brute_force_gap(centers[i], axes[i], radii[i], halves[i], (0.0, 0.0, 1.0))
        for i in range(4)
    ]
    assert gaps == pytest.approx(expected, abs=1e-8)
    # Analytic closed forms for the two extreme orientations.
    assert gaps[0] == pytest.approx(0.30 - _WHEEL_HALF_LENGTH_M, abs=1e-12)
    assert gaps[1] == pytest.approx(0.30 - _WHEEL_HALF_LENGTH_M, abs=1e-12)
    assert gaps[2] == pytest.approx(0.30 - _WHEEL_RADIUS_M, abs=1e-12)
    assert gaps[3] == pytest.approx(0.30 - _WHEEL_RADIUS_M, abs=1e-12)


@pytest.mark.parametrize("tilt_deg", [0.0, 1.0, 7.5, 30.0, 45.0, 75.0, 89.0, 90.0])
def test_tilted_cylinder_support_gap_matches_independent_support_values(tilt_deg):
    tilt = math.radians(tilt_deg)
    axis = _unit((math.sin(tilt), 0.0, math.cos(tilt)))
    centers = np.tile(np.array([0.0, 0.0, 0.25]), (4, 1))
    axes = np.tile(axis, (4, 1))
    radii = np.full(4, _WHEEL_RADIUS_M)
    halves = np.full(4, _WHEEL_HALF_LENGTH_M)
    gaps = jr.oriented_cylinder_gaps(centers, axes, radii, halves)
    expected = _brute_force_gap(
        centers[0], axis, _WHEEL_RADIUS_M, _WHEEL_HALF_LENGTH_M, (0.0, 0.0, 1.0)
    )
    assert gaps == pytest.approx([expected] * 4, abs=1e-8)
    # A tilted wheel hangs lower than either extreme orientation.
    assert gaps[0] <= min(0.25 - _WHEEL_RADIUS_M, 0.25 - _WHEEL_HALF_LENGTH_M) + 1e-12


def test_tilted_plane_pins_the_axis_and_normal_dot_convention():
    """The cylinder axis is the local z axis, i.e. the third xmat column."""

    quat = (0.8087508, 0.3009374, 0.4513061, 0.2306423)
    rotation = _matrix_from_quaternion(quat)
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(rotation[:, 2])
    axis_column = rotation[:, 2]
    axis_row = rotation[2, :]
    assert not np.allclose(axis_column, axis_row)

    normal = _unit((0.2, -0.3, 0.9))
    offset = -0.15
    centers = np.tile(np.array([0.4, -0.2, 0.33]), (4, 1))
    radii = np.full(4, _WHEEL_RADIUS_M)
    halves = np.full(4, _WHEEL_HALF_LENGTH_M)

    gaps = jr.oriented_cylinder_gaps(
        centers,
        np.tile(axis_column, (4, 1)),
        radii,
        halves,
        plane_normal=normal,
        plane_offset_m=offset,
    )
    expected = _brute_force_gap(
        centers[0], axis_column, _WHEEL_RADIUS_M, _WHEEL_HALF_LENGTH_M, normal, offset
    )
    assert gaps == pytest.approx([expected] * 4, abs=1e-8)

    # Using the transposed convention would be a different, wrong answer, so the
    # choice is load bearing rather than cosmetic.
    wrong = jr.oriented_cylinder_gaps(
        centers,
        np.tile(axis_row, (4, 1)),
        radii,
        halves,
        plane_normal=normal,
        plane_offset_m=offset,
    )
    assert abs(float(wrong[0]) - expected) > 1e-4


def test_plane_offset_shifts_the_gap_rigidly():
    centers = np.tile(np.array([0.0, 0.0, 0.2]), (4, 1))
    axes = np.tile(np.array([0.0, 1.0, 0.0]), (4, 1))
    radii = np.full(4, _WHEEL_RADIUS_M)
    halves = np.full(4, _WHEEL_HALF_LENGTH_M)
    base = jr.oriented_cylinder_gaps(centers, axes, radii, halves)
    raised = jr.oriented_cylinder_gaps(
        centers, axes, radii, halves, plane_offset_m=0.05
    )
    assert raised == pytest.approx(np.asarray(base) - 0.05, abs=1e-15)


def test_penetration_is_reported_signed_and_never_clamped():
    centers = np.array(
        [
            [0.0, 0.0, _WHEEL_RADIUS_M],
            [0.0, 0.0, _WHEEL_RADIUS_M - 0.0005],
            [0.0, 0.0, _WHEEL_RADIUS_M - 0.01],
            [0.0, 0.0, _WHEEL_RADIUS_M + 0.05],
        ]
    )
    axes = np.tile(np.array([0.0, 1.0, 0.0]), (4, 1))
    gaps = jr.oriented_cylinder_gaps(
        centers, axes, np.full(4, _WHEEL_RADIUS_M), np.full(4, _WHEEL_HALF_LENGTH_M)
    )
    assert gaps == pytest.approx([0.0, -0.0005, -0.01, 0.05], abs=1e-15)
    assert float(np.min(gaps)) < 0.0


def test_gap_result_is_read_only_and_inputs_are_not_mutated():
    centers = np.tile(np.array([0.0, 0.0, 0.2]), (4, 1))
    axes = np.tile(np.array([0.0, 1.0, 0.0]), (4, 1))
    radii = np.full(4, _WHEEL_RADIUS_M)
    halves = np.full(4, _WHEEL_HALF_LENGTH_M)
    snapshots = [arr.copy() for arr in (centers, axes, radii, halves)]
    gaps = jr.oriented_cylinder_gaps(centers, axes, radii, halves)
    assert isinstance(gaps, np.ndarray)
    assert gaps.dtype == np.float64
    assert gaps.flags.writeable is False
    with pytest.raises(ValueError):
        gaps[0] = 1.0
    for arr, snapshot in zip((centers, axes, radii, halves), snapshots):
        assert np.array_equal(arr, snapshot)


@pytest.mark.parametrize(
    "normal",
    [(0.0, 0.0, 1.0000001), (0.0, 0.0, 0.9), (0.0, 0.0, 2.0), (0.0, 0.0, 0.0)],
)
def test_non_unit_plane_normal_is_rejected_not_renormalised(normal):
    centers = np.tile(np.array([0.0, 0.0, 0.2]), (4, 1))
    axes = np.tile(np.array([0.0, 1.0, 0.0]), (4, 1))
    with pytest.raises(ValueError, match="unit vector"):
        jr.oriented_cylinder_gaps(
            centers,
            axes,
            np.full(4, _WHEEL_RADIUS_M),
            np.full(4, _WHEEL_HALF_LENGTH_M),
            plane_normal=normal,
        )


def test_non_unit_cylinder_axis_is_rejected_with_its_row_index():
    centers = np.tile(np.array([0.0, 0.0, 0.2]), (4, 1))
    axes = np.tile(np.array([0.0, 1.0, 0.0]), (4, 1))
    axes[2] = (0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match=r"rows \[2\]"):
        jr.oriented_cylinder_gaps(
            centers, axes, np.full(4, _WHEEL_RADIUS_M), np.full(4, _WHEEL_HALF_LENGTH_M)
        )


def test_tiny_unit_norm_drift_inside_tolerance_is_accepted():
    axis = np.array([0.0, 0.0, 1.0 + 1e-10])
    gaps = jr.oriented_cylinder_gaps(
        np.tile(np.array([0.0, 0.0, 0.2]), (4, 1)),
        np.tile(axis, (4, 1)),
        np.full(4, _WHEEL_RADIUS_M),
        np.full(4, _WHEEL_HALF_LENGTH_M),
    )
    assert gaps == pytest.approx([0.2 - _WHEEL_HALF_LENGTH_M] * 4, abs=1e-9)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"centers_world_m": np.zeros((3, 3))}, ValueError),
        ({"centers_world_m": np.zeros((4, 2))}, ValueError),
        ({"centers_world_m": np.zeros(4)}, ValueError),
        ({"axes_world": np.zeros((4, 4))}, ValueError),
        ({"radii_m": np.full(3, 0.087)}, ValueError),
        ({"half_lengths_m": np.full((4, 1), 0.02)}, ValueError),
        ({"radii_m": np.full(4, 0.0)}, ValueError),
        ({"radii_m": np.full(4, -0.087)}, ValueError),
        ({"half_lengths_m": np.full(4, 0.0)}, ValueError),
        ({"centers_world_m": np.full((4, 3), np.nan)}, ValueError),
        ({"centers_world_m": np.full((4, 3), np.inf)}, ValueError),
        ({"radii_m": np.full(4, np.nan)}, ValueError),
        ({"plane_offset_m": float("nan")}, ValueError),
        ({"plane_offset_m": float("inf")}, ValueError),
        ({"plane_normal": np.zeros(4)}, ValueError),
        ({"centers_world_m": np.zeros((4, 3), dtype=bool)}, TypeError),
        ({"axes_world": np.zeros((4, 3), dtype=complex)}, TypeError),
        ({"radii_m": np.array(["a", "b", "c", "d"])}, TypeError),
        ({"radii_m": np.array([None] * 4, dtype=object)}, TypeError),
        ({"plane_offset_m": "0.0"}, TypeError),
        ({"plane_offset_m": True}, TypeError),
        ({"plane_offset_m": 1j}, TypeError),
    ],
)
def test_gap_input_validation(kwargs, error):
    call = {
        "centers_world_m": np.tile(np.array([0.0, 0.0, 0.2]), (4, 1)),
        "axes_world": np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
        "radii_m": np.full(4, _WHEEL_RADIUS_M),
        "half_lengths_m": np.full(4, _WHEEL_HALF_LENGTH_M),
    }
    call.update(kwargs)
    positional = (
        call.pop("centers_world_m"),
        call.pop("axes_world"),
        call.pop("radii_m"),
        call.pop("half_lengths_m"),
    )
    with pytest.raises(error):
        jr.oriented_cylinder_gaps(*positional, **call)


# ---------------------------------------------------------------------------
# Compiled-model binding: collision versus visual geometry
# ---------------------------------------------------------------------------


def test_binding_reads_collision_cylinders_not_visual_meshes(synthetic_binding):
    binding = synthetic_binding
    assert binding.schema == jr.JUMP_READINESS_SCHEMA
    assert binding.wheel_geom_ids == _SYNTHETIC_WHEEL_GEOM_IDS
    assert binding.wheel_body_ids == _SYNTHETIC_WHEEL_BODY_IDS
    assert binding.wheel_body_names == ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    assert binding.wheel_joint_ids == (1, 2, 3, 4)
    assert binding.radii_m == (_WHEEL_RADIUS_M,) * 4
    assert binding.half_lengths_m == (_WHEEL_HALF_LENGTH_M,) * 4
    assert binding.contact_margins_m == (_WHEEL_MARGIN_M,) * 4
    assert binding.plane_geom_id == 0
    assert binding.plane_geom_name == "floor"
    assert binding.plane_margin_m == 0.0
    assert binding.plane_normal == (0.0, 0.0, 1.0)
    assert binding.plane_offset_m == 0.0
    assert binding.model_ngeom == 9
    # The visual mesh ids and their non-identity transforms are never adopted.
    assert set(binding.wheel_geom_ids).isdisjoint({1, 3, 5, 7})
    assert binding.wheel_local_quaternions == ((1.0, 0.0, 0.0, 0.0),) * 4
    assert binding.wheel_local_positions == ((0.0, 0.0, 0.0),) * 4
    assert _VISUAL_MESH_SIZE[1] not in binding.radii_m


def test_binding_is_frozen_and_holds_no_model_or_array_reference(synthetic_binding):
    with pytest.raises(dataclasses.FrozenInstanceError):
        synthetic_binding.plane_offset_m = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        synthetic_binding.wheel_geom_ids = (0, 1, 2, 3)

    def _assert_plain(value):
        if isinstance(value, tuple):
            for item in value:
                _assert_plain(item)
            return
        assert type(value) in (int, float, str), f"unexpected field type {type(value)}"

    for field in dataclasses.fields(synthetic_binding):
        _assert_plain(getattr(synthetic_binding, field.name))


def test_receipt_is_json_ready_and_decoupled(synthetic_binding):
    receipt = synthetic_binding.as_receipt()
    assert json.loads(json.dumps(receipt)) == receipt
    receipt["wheel_geom_ids"][0] = -1
    receipt["radii_m"].append(9.0)
    assert synthetic_binding.wheel_geom_ids == _SYNTHETIC_WHEEL_GEOM_IDS
    assert len(synthetic_binding.radii_m) == 4
    assert synthetic_binding.as_receipt()["wheel_geom_ids"][0] == _SYNTHETIC_WHEEL_GEOM_IDS[0]


def test_binding_rejects_a_second_collidable_geom_on_a_wheel_body():
    entries = _default_geom_entries()
    entries.append(_geom_entry(1, _BOX, (0.01, 0.01, 0.01)))
    with pytest.raises(ValueError, match="exactly one collidable geom"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )


def test_binding_rejects_a_wheel_body_without_a_collision_geom():
    entries = _default_geom_entries()
    entries[2]["contype"] = 0
    entries[2]["conaffinity"] = 0
    with pytest.raises(ValueError, match="exactly one collidable geom"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )


def test_binding_rejects_a_non_cylinder_wheel_collision_geom():
    entries = _default_geom_entries()
    entries[2]["type"] = _MESH
    with pytest.raises(ValueError, match="must be a CYLINDER"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("size", (0.09, 0.02, 0.0), "radius must be"),
        ("size", (0.087, 0.03, 0.0), "half-length must be"),
        ("margin", 0.002, "margin must be"),
        ("gap", 0.001, "gap must be 0"),
        ("pos", (0.0, 0.0, 0.001), "zero local position"),
        ("quat", (0.7071067811865476, 0.7071067811865476, 0.0, 0.0), "identity local"),
    ],
)
def test_binding_rejects_unexpected_wheel_cylinder_parameters(field, value, match):
    entries = _default_geom_entries()
    entries[4][field] = value
    with pytest.raises(ValueError, match=match):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )


def test_binding_rejects_a_plane_id_that_is_not_the_floor_plane():
    model = _SyntheticModel()
    with pytest.raises(ValueError, match="not a PLANE"):
        jr.bind_wheel_plane_geometry(model, _SYNTHETIC_WHEEL_BODY_IDS, 2)
    with pytest.raises(ValueError, match="not a PLANE"):
        jr.bind_wheel_plane_geometry(model, _SYNTHETIC_WHEEL_BODY_IDS, 1)
    with pytest.raises(ValueError, match="out of range"):
        jr.bind_wheel_plane_geometry(model, _SYNTHETIC_WHEEL_BODY_IDS, 9)


def test_binding_rejects_a_renamed_or_displaced_plane():
    entries = _default_geom_entries()
    entries[0]["name"] = "terrain_floor"
    with pytest.raises(ValueError, match="must be named"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )
    entries = _default_geom_entries()
    entries[0]["pos"] = (0.0, 0.0, 0.02)
    with pytest.raises(ValueError, match="zero local position"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )
    entries = _default_geom_entries()
    entries[0]["margin"] = 0.001
    with pytest.raises(ValueError, match="margin must be"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )


def test_binding_rejects_extra_collidable_world_geometry():
    entries = _default_geom_entries()
    entries.append(_geom_entry(0, _BOX, (0.1, 0.1, 0.02), name="hurdle"))
    with pytest.raises(ValueError, match="exactly one collidable geom"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(entries), _SYNTHETIC_WHEEL_BODY_IDS
        )


def test_binding_rejects_a_height_field_model():
    with pytest.raises(ValueError, match="no height field"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(nhfield=1), _SYNTHETIC_WHEEL_BODY_IDS
        )


def test_binding_rejects_unexpected_wheel_joint_association():
    with pytest.raises(ValueError, match="must be a hinge"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(
                jnt_type=(_JNT_FREE, _JNT_BALL, _JNT_HINGE, _JNT_HINGE, _JNT_HINGE)
            ),
            _SYNTHETIC_WHEEL_BODY_IDS,
        )
    with pytest.raises(ValueError, match="exactly one joint"):
        jr.bind_wheel_plane_geometry(
            _SyntheticModel(body_jntnum=(0, 2, 1, 1, 1)), _SYNTHETIC_WHEEL_BODY_IDS
        )


def test_binding_rejects_wrong_wheel_body_identity():
    model = _SyntheticModel()
    with pytest.raises(ValueError, match="must be body 'FL_foot'"):
        jr.bind_wheel_plane_geometry(model, (2, 1, 3, 4))
    with pytest.raises(ValueError, match="four distinct ids"):
        jr.bind_wheel_plane_geometry(model, (1, 1, 3, 4))
    with pytest.raises(ValueError, match="four ids"):
        jr.bind_wheel_plane_geometry(model, (1, 2, 3))
    with pytest.raises(ValueError, match="out of range"):
        jr.bind_wheel_plane_geometry(model, (1, 2, 3, 5))
    with pytest.raises(TypeError):
        jr.bind_wheel_plane_geometry(model, (1, 2, 3, 4.0))
    with pytest.raises(TypeError):
        jr.bind_wheel_plane_geometry(model, (1, 2, 3, True))
    with pytest.raises(TypeError):
        jr.bind_wheel_plane_geometry(model, "1234")
    with pytest.raises(TypeError):
        jr.bind_wheel_plane_geometry(None, _SYNTHETIC_WHEEL_BODY_IDS)


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"schema": "other-schema"}, ValueError),
        ({"radii_m": (0.0, 0.087, 0.087, 0.087)}, ValueError),
        ({"half_lengths_m": (-0.02, 0.02, 0.02, 0.02)}, ValueError),
        ({"wheel_geom_ids": (2, 2, 6, 8)}, ValueError),
        ({"wheel_body_ids": (1, 1, 3, 4)}, ValueError),
        ({"wheel_geom_ids": [2, 4, 6, 8]}, TypeError),
        ({"radii_m": (0.087, 0.087, 0.087)}, ValueError),
        ({"radii_m": (1, 0.087, 0.087, 0.087)}, TypeError),
        ({"plane_geom_id": True}, TypeError),
        ({"plane_offset_m": float("nan")}, ValueError),
        ({"model_ngeom": -1}, ValueError),
        ({"model_ngeom": 4.0}, TypeError),
        ({"model_ngeom": 5}, ValueError),
    ],
)
def test_binding_construction_validation(overrides, error):
    with pytest.raises(error):
        jr.WheelPlaneBinding(**_binding_kwargs(**overrides))


# ---------------------------------------------------------------------------
# Sampling already-synchronized geometry
# ---------------------------------------------------------------------------


def test_sample_reflects_the_supplied_transforms_only(synthetic_binding):
    heights = [
        _WHEEL_RADIUS_M,  # resting: zero geometric gap
        _WHEEL_RADIUS_M - 0.0005,  # inside the margin, slight penetration
        _WHEEL_RADIUS_M + 0.0005,  # inside the margin, positive gap
        _WHEEL_RADIUS_M + 0.05,  # clearly clear of the floor
    ]
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, heights)
    pos_snapshot, mat_snapshot = xpos.copy(), xmat.copy()

    summary = jr.sample_wheel_clearance(synthetic_binding, xpos, xmat)

    assert summary["schema"] == jr.JUMP_READINESS_SCHEMA
    assert summary["wheel_geom_ids"] == list(_SYNTHETIC_WHEEL_GEOM_IDS)
    assert summary["wheel_bottom_gap_m"] == pytest.approx(
        [0.0, -0.0005, 0.0005, 0.05], abs=1e-15
    )
    assert summary["simultaneous_minimum_gap_m"] == pytest.approx(-0.0005, abs=1e-15)
    assert summary["plane_geom_id"] == 0
    assert summary["plane_normal"] == [0.0, 0.0, 1.0]
    assert summary["plane_offset_m"] == 0.0
    assert np.array_equal(xpos, pos_snapshot)
    assert np.array_equal(xmat, mat_snapshot)
    # Moving one supplied transform changes only that wheel's reported gap.
    xpos[_SYNTHETIC_WHEEL_GEOM_IDS[0], 2] += 0.03
    moved = jr.sample_wheel_clearance(synthetic_binding, xpos, xmat)
    assert moved["wheel_bottom_gap_m"][0] == pytest.approx(0.03, abs=1e-15)
    assert moved["wheel_bottom_gap_m"][1:] == summary["wheel_bottom_gap_m"][1:]


def test_contact_margin_stays_separate_from_the_signed_bottom_gap(synthetic_binding):
    heights = [_WHEEL_RADIUS_M + h for h in (0.0, 0.0005, 0.02, 0.05)]
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, heights)
    summary = jr.sample_wheel_clearance(synthetic_binding, xpos, xmat)

    margin = summary["contact_margin_m"]
    assert margin == _WHEEL_MARGIN_M
    # Gaps are raw signed geometry: the margin is never folded into them.
    assert summary["wheel_bottom_gap_m"] == pytest.approx(
        [0.0, 0.0005, 0.02, 0.05], abs=1e-15
    )
    assert summary["simultaneous_minimum_gap_m"] == pytest.approx(0.0, abs=1e-15)
    # The 20 mm hurdle screen must be composed by the caller as gap > 0.020 + margin.
    hurdle_screen = 0.020 + margin
    assert hurdle_screen == pytest.approx(0.021, abs=1e-12)
    assert summary["wheel_bottom_gap_m"][2] < hurdle_screen
    assert summary["wheel_bottom_gap_m"][3] > hurdle_screen


def test_sample_accepts_both_flat_and_square_rotation_layouts(synthetic_binding):
    heights = [_WHEEL_RADIUS_M + 0.01] * 4
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, heights)
    flat = jr.sample_wheel_clearance(synthetic_binding, xpos, xmat)
    square = jr.sample_wheel_clearance(synthetic_binding, xpos, xmat.reshape(9, 3, 3))
    assert flat == square
    assert flat["wheel_bottom_gap_m"] == pytest.approx([0.01] * 4, abs=1e-15)


def test_sample_wheel_axis_is_the_third_column_of_the_geom_frame(synthetic_binding):
    rotation = _rotation_z(math.radians(31.0)) @ _rotation_y(math.radians(52.0))
    axis = rotation[:, 2]
    height = 0.36
    xpos, xmat = _flat_frames(
        9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, [height] * 4, wheel_rotation=rotation
    )
    summary = jr.sample_wheel_clearance(synthetic_binding, xpos, xmat)
    expected = _brute_force_gap(
        (0.0, 0.0, height), axis, _WHEEL_RADIUS_M, _WHEEL_HALF_LENGTH_M, (0.0, 0.0, 1.0)
    )
    assert summary["wheel_bottom_gap_m"] == pytest.approx([expected] * 4, abs=1e-8)
    # For a level floor the row and column conventions share element [2][2], so
    # the discriminating convention check lives in the tilted-plane test above.
    assert float(rotation[2, 2]) == float(rotation.T[2, 2])
    assert jr.sample_wheel_clearance(
        synthetic_binding, xpos, np.tile(rotation.T.reshape(9), (9, 1))
    )["wheel_bottom_gap_m"] == pytest.approx([expected] * 4, abs=1e-8)


def test_sample_matches_the_pure_gap_facility(synthetic_binding):
    heights = [_WHEEL_RADIUS_M + 0.004 * i for i in range(4)]
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, heights)
    summary = jr.sample_wheel_clearance(synthetic_binding, xpos, xmat)
    direct = jr.oriented_cylinder_gaps(
        xpos[list(_SYNTHETIC_WHEEL_GEOM_IDS), :],
        xmat.reshape(9, 3, 3)[list(_SYNTHETIC_WHEEL_GEOM_IDS), :, 2],
        np.asarray(synthetic_binding.radii_m),
        np.asarray(synthetic_binding.half_lengths_m),
    )
    assert summary["wheel_bottom_gap_m"] == pytest.approx(list(direct), abs=1e-15)


def test_sample_rejects_a_changed_or_invalid_plane_frame(synthetic_binding):
    heights = [_WHEEL_RADIUS_M] * 4
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, heights)

    tilted = xmat.copy()
    tilted[0] = _rotation_x(math.radians(5.0)).reshape(9)
    with pytest.raises(ValueError, match="plane orientation"):
        jr.sample_wheel_clearance(synthetic_binding, xpos, tilted)

    raised = xpos.copy()
    raised[0, 2] = 0.02
    with pytest.raises(ValueError, match="plane position"):
        jr.sample_wheel_clearance(synthetic_binding, raised, xmat)


def test_sample_rejects_non_rotation_geom_frames(synthetic_binding):
    heights = [_WHEEL_RADIUS_M] * 4
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, heights)

    scaled = xmat.copy()
    scaled[_SYNTHETIC_WHEEL_GEOM_IDS[1]] = (np.eye(3) * 1.01).reshape(9)
    with pytest.raises(ValueError, match="not orthonormal"):
        jr.sample_wheel_clearance(synthetic_binding, xpos, scaled)

    reflected = xmat.copy()
    reflected[_SYNTHETIC_WHEEL_GEOM_IDS[2]] = np.diag([1.0, 1.0, -1.0]).reshape(9)
    with pytest.raises(ValueError, match="proper rotation"):
        jr.sample_wheel_clearance(synthetic_binding, xpos, reflected)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda p, m: (p[:8], m), ValueError),
        (lambda p, m: (p, m[:8]), ValueError),
        (lambda p, m: (p[:, :2], m), ValueError),
        (lambda p, m: (p, m[:, :6]), ValueError),
        (lambda p, m: (p * np.nan, m), ValueError),
        (lambda p, m: (p, m * np.inf), ValueError),
        (lambda p, m: (p.astype(bool), m), TypeError),
        (lambda p, m: (p.astype(str), m), TypeError),
    ],
)
def test_sample_geometry_array_validation(synthetic_binding, mutate, error):
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, [_WHEEL_RADIUS_M] * 4)
    bad_pos, bad_mat = mutate(xpos, xmat)
    with pytest.raises(error):
        jr.sample_wheel_clearance(synthetic_binding, bad_pos, bad_mat)


def test_sample_rejects_a_non_binding_first_argument():
    xpos, xmat = _flat_frames(9, _SYNTHETIC_WHEEL_GEOM_IDS, 0, [_WHEEL_RADIUS_M] * 4)
    for bogus in (None, {"wheel_geom_ids": _SYNTHETIC_WHEEL_GEOM_IDS}, 0):
        with pytest.raises(TypeError, match="WheelPlaneBinding"):
            jr.sample_wheel_clearance(bogus, xpos, xmat)


# ---------------------------------------------------------------------------
# Real compiled native-plane fixture (compilation only, no MjData)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_model():
    """Compile the D1 model once.  No MjData, reset, forward or step happens."""

    from wheel_legged_control.d1.model import build_d1_model

    return build_d1_model()


@pytest.fixture(scope="module")
def real_binding(real_model):
    return jr.bind_wheel_plane_geometry(
        real_model, tuple(_REFERENCE_GEOMETRY["wheel_body_ids"]), 0
    )


def test_real_compiled_binding_matches_the_reference_geometry(real_model, real_binding):
    reference = _REFERENCE_GEOMETRY
    assert int(real_model.nbody) == reference["model_nbody"]
    assert int(real_model.ngeom) == reference["model_ngeom"]

    cylinders = [g for g in reference["geometries"] if g["type"] == _CYLINDER]
    meshes = [g for g in reference["geometries"] if g["type"] == _MESH]
    plane = next(g for g in reference["geometries"] if g["type"] == _PLANE)
    assert len(cylinders) == 4

    assert real_binding.wheel_geom_ids == tuple(g["geom_id"] for g in cylinders)
    assert real_binding.wheel_body_ids == tuple(reference["wheel_body_ids"])
    assert real_binding.wheel_joint_ids == tuple(g["body_joints"][0] for g in cylinders)
    assert real_binding.radii_m == tuple(g["size"][0] for g in cylinders)
    assert real_binding.half_lengths_m == tuple(g["size"][1] for g in cylinders)
    assert real_binding.contact_margins_m == tuple(g["margin"] for g in cylinders)
    assert real_binding.plane_geom_id == plane["geom_id"]
    assert real_binding.plane_geom_name == plane["geom_name"] == "floor"
    assert real_binding.plane_margin_m == plane["margin"]
    assert real_binding.plane_normal == (0.0, 0.0, 1.0)
    assert real_binding.model_ngeom == reference["model_ngeom"]
    # Visual wheel meshes stay out of the collision binding.
    assert set(real_binding.wheel_geom_ids).isdisjoint(g["geom_id"] for g in meshes)
    assert real_binding.radii_m == (_WHEEL_RADIUS_M,) * 4
    assert real_binding.contact_margins_m == (_WHEEL_MARGIN_M,) * 4


def test_real_binding_samples_supplied_native_plane_transforms(real_binding):
    ngeom = int(_REFERENCE_GEOMETRY["model_ngeom"])
    heights = [_WHEEL_RADIUS_M + h for h in (0.0, 0.001, 0.021, 0.05)]
    xpos, xmat = _flat_frames(
        ngeom, real_binding.wheel_geom_ids, real_binding.plane_geom_id, heights
    )
    summary = jr.sample_wheel_clearance(real_binding, xpos, xmat)
    assert summary["wheel_bottom_gap_m"] == pytest.approx(
        [0.0, 0.001, 0.021, 0.05], abs=1e-15
    )
    assert summary["simultaneous_minimum_gap_m"] == pytest.approx(0.0, abs=1e-15)
    assert summary["contact_margin_m"] == _WHEEL_MARGIN_M
    assert summary["wheel_geom_ids"] == list(real_binding.wheel_geom_ids)


def test_real_binding_rejects_geometry_arrays_from_another_model(real_binding):
    ngeom = int(_REFERENCE_GEOMETRY["model_ngeom"])
    xpos, xmat = _flat_frames(
        ngeom, real_binding.wheel_geom_ids, real_binding.plane_geom_id, [0.1] * 4
    )
    with pytest.raises(ValueError, match="every geom of the bound model"):
        jr.sample_wheel_clearance(real_binding, xpos[:-1], xmat[:-1])
    with pytest.raises(ValueError, match="every geom of the bound model"):
        jr.sample_wheel_clearance(
            real_binding, np.vstack([xpos, xpos[:1]]), np.vstack([xmat, xmat[:1]])
        )


def test_geometry_facilities_never_touch_the_plant(monkeypatch, real_model, real_binding):
    """No dynamics, kinematics, collision, reset or MjData entry point is used."""

    import mujoco

    def _forbidden(*args, **kwargs):
        raise AssertionError("the readiness utility must not touch the plant")

    for name in (
        "MjData",
        "mj_step",
        "mj_step1",
        "mj_step2",
        "mj_forward",
        "mj_forwardSkip",
        "mj_kinematics",
        "mj_fwdPosition",
        "mj_comPos",
        "mj_collision",
        "mj_resetData",
        "mj_resetDataKeyframe",
    ):
        if hasattr(mujoco, name):
            monkeypatch.setattr(mujoco, name, _forbidden)

    binding = jr.bind_wheel_plane_geometry(
        real_model, tuple(_REFERENCE_GEOMETRY["wheel_body_ids"]), 0
    )
    assert binding == real_binding

    ngeom = int(_REFERENCE_GEOMETRY["model_ngeom"])
    xpos, xmat = _flat_frames(
        ngeom, binding.wheel_geom_ids, binding.plane_geom_id, [0.2] * 4
    )
    summary = jr.sample_wheel_clearance(binding, xpos, xmat)
    assert summary["simultaneous_minimum_gap_m"] == pytest.approx(
        0.2 - _WHEEL_RADIUS_M, abs=1e-15
    )
    jr.fixed_height_command(200, "profile")
    jr.sampled_true_runs(
        np.array([True, True]), np.array([0.0, 0.002]), np.array([0.002, 0.004])
    )


# ---------------------------------------------------------------------------
# Sampled-interval run reducer
# ---------------------------------------------------------------------------


def _native_intervals(count, dt=0.002, start=0.0):
    starts = start + dt * np.arange(count, dtype=np.float64)
    return starts, starts + dt


def test_runs_report_contiguous_sampled_intervals_with_durations():
    mask = np.zeros(30, dtype=bool)
    mask[4:14] = True  # ten recorded native intervals == 20 ms of sampling
    mask[20:22] = True
    starts, ends = _native_intervals(30)

    runs = jr.sampled_true_runs(mask, starts, ends)

    assert [r["first_index"] for r in runs] == [4, 20]
    assert [r["end_index_exclusive"] for r in runs] == [14, 22]
    assert [r["intervals"] for r in runs] == [10, 2]
    assert runs[0]["start_time_s"] == pytest.approx(0.008, abs=1e-12)
    assert runs[0]["end_time_s"] == pytest.approx(0.028, abs=1e-12)
    assert runs[0]["summed_interval_duration_s"] == pytest.approx(0.020, abs=1e-12)
    assert runs[1]["summed_interval_duration_s"] == pytest.approx(0.004, abs=1e-12)
    assert set(runs[0]) == {
        "first_index",
        "end_index_exclusive",
        "intervals",
        "start_time_s",
        "end_time_s",
        "summed_interval_duration_s",
    }
    # The reducer describes sampled intervals; it claims no continuous flight.
    assert not any(
        token in key
        for key in runs[0]
        for token in ("flight", "airborne", "contact", "continuous")
    )


def test_a_gap_in_native_sampling_splits_a_run():
    mask = np.ones(6, dtype=bool)
    starts, ends = _native_intervals(6)
    # Drop the interval that would have covered 0.006..0.008 s.
    keep = [0, 1, 2, 4, 5]
    runs = jr.sampled_true_runs(mask[keep], starts[keep], ends[keep])

    assert [r["intervals"] for r in runs] == [3, 2]
    assert runs[0]["end_time_s"] == pytest.approx(0.006, abs=1e-12)
    assert runs[1]["start_time_s"] == pytest.approx(0.008, abs=1e-12)
    assert sum(r["intervals"] for r in runs) == 5
    # Summed sampled duration is shorter than the wall interval it spans.
    span = runs[1]["end_time_s"] - runs[0]["start_time_s"]
    sampled = sum(r["summed_interval_duration_s"] for r in runs)
    assert sampled == pytest.approx(0.010, abs=1e-12)
    assert span > sampled


def test_float_accumulated_native_times_still_join_one_run():
    count = 500
    starts = np.array([i * 0.002 for i in range(count)], dtype=np.float64)
    ends = np.array([(i + 1) * 0.002 for i in range(count)], dtype=np.float64)
    runs = jr.sampled_true_runs(np.ones(count, dtype=bool), starts, ends)
    assert len(runs) == 1
    assert runs[0]["intervals"] == count
    assert runs[0]["summed_interval_duration_s"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([], []),
        ([False, False], []),
        ([True], [(0, 1, 1)]),
        ([True, False, True], [(0, 1, 1), (2, 3, 1)]),
        ([False, True, True, False, True], [(1, 3, 2), (4, 5, 1)]),
        ([True, True, True], [(0, 3, 3)]),
    ],
)
def test_run_segmentation_cases(mask, expected):
    mask_array = np.asarray(mask, dtype=bool)
    starts, ends = _native_intervals(mask_array.size)
    runs = jr.sampled_true_runs(mask_array, starts, ends)
    assert [
        (r["first_index"], r["end_index_exclusive"], r["intervals"]) for r in runs
    ] == expected


def test_runs_do_not_mutate_their_inputs():
    mask = np.array([True, True, False, True])
    starts, ends = _native_intervals(4)
    snapshots = (mask.copy(), starts.copy(), ends.copy())
    jr.sampled_true_runs(mask, starts, ends)
    for array, snapshot in zip((mask, starts, ends), snapshots):
        assert np.array_equal(array, snapshot)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([1, 0, 1]),
        np.array([1.0, 0.0, 1.0]),
        np.array(["True", "False", "True"]),
        [1, 0, 1],
    ],
)
def test_non_boolean_masks_are_rejected(mask):
    starts, ends = _native_intervals(3)
    with pytest.raises(TypeError, match="boolean"):
        jr.sampled_true_runs(mask, starts, ends)


def test_run_reducer_rejects_inconsistent_or_invalid_interval_times():
    starts, ends = _native_intervals(4)
    mask = np.ones(4, dtype=bool)

    with pytest.raises(ValueError, match="match the mask shape"):
        jr.sampled_true_runs(mask, starts[:3], ends)
    with pytest.raises(ValueError, match="match the mask shape"):
        jr.sampled_true_runs(mask, starts, ends[:3])
    with pytest.raises(ValueError, match="1-D"):
        jr.sampled_true_runs(mask.reshape(2, 2), starts, ends)
    with pytest.raises(ValueError, match="1-D"):
        jr.sampled_true_runs(mask, starts.reshape(2, 2), ends.reshape(2, 2))

    zero_length = ends.copy()
    zero_length[2] = starts[2]
    with pytest.raises(ValueError, match="end_time_s > start_time_s"):
        jr.sampled_true_runs(mask, starts, zero_length)

    reversed_times = ends.copy()
    reversed_times[1] = starts[1] - 0.001
    with pytest.raises(ValueError, match="end_time_s > start_time_s"):
        jr.sampled_true_runs(mask, starts, reversed_times)

    overlapping = starts.copy()
    overlapping[2] = starts[2] - 0.001
    with pytest.raises(ValueError, match="monotone and non-overlapping"):
        jr.sampled_true_runs(mask, overlapping, ends)

    nan_times = starts.copy()
    nan_times[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        jr.sampled_true_runs(mask, nan_times, ends)

    inf_times = ends.copy()
    inf_times[3] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        jr.sampled_true_runs(mask, starts, inf_times)
