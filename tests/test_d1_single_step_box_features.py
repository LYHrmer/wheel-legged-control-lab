"""D1 single-step box contact feature labeling tests (pure geometry)."""

import numpy as np
import pytest

from scripts.d1_single_step_geometry import box_contact_feature

CENTER = np.array([0.0, 0.0, 0.0075])
HALF = np.array([0.18, 0.62, 0.0075])
TOL = 1e-3

FRONT_POS = np.array([-0.18, 0.0, 0.0075])
FRONT_N = np.array([-1.0, 0.0, 0.0])
TOP_POS = np.array([0.0, 0.0, 0.015])
TOP_N = np.array([0.0, 0.0, 1.0])
EDGE_POS = np.array([-0.18, 0.0, 0.015])
EDGE_N = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)


def _call(position, normal, tolerance=TOL):
    return box_contact_feature(position, normal, CENTER, HALF, tolerance_m=tolerance)


@pytest.mark.parametrize(
    "position, normal, expected",
    [
        (FRONT_POS, FRONT_N, "front"),
        (TOP_POS, TOP_N, "top"),
        (EDGE_POS, EDGE_N, "front_top_edge"),
    ],
)
def test_true_features_are_labeled_and_supported(position, normal, expected):
    out = _call(position, normal)
    assert out["feature"] == expected
    assert out["geometric_support_valid"] is True
    assert out["normal_cone_residual"] == pytest.approx(0.0, abs=1e-9)


def test_flipped_front_normal_is_not_supported():
    out = _call(FRONT_POS, -FRONT_N)
    assert out["geometric_support_valid"] is False


def test_front_position_moved_off_face_is_not_supported():
    out = _call(np.array([-0.16, 0.0, 0.0075]), FRONT_N, tolerance=TOL)
    assert out["geometric_support_valid"] is False


def test_small_numerical_tangent_still_labels_top():
    normal = np.array([0.001, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    out = _call(TOP_POS, normal)
    assert out["feature"] == "top"
    assert out["geometric_support_valid"] is True
    assert out["normal_cone_residual"] == pytest.approx(0.001, abs=1e-5)


def test_large_tangent_component_breaks_support():
    normal = np.array([0.01, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    out = _call(TOP_POS, normal)
    assert out["geometric_support_valid"] is False
    assert out["normal_cone_residual"] > TOL


@pytest.mark.parametrize(
    "normal",
    [
        np.zeros(3),
        np.array([0.0, 0.0, 2.0]),
        np.array([np.nan, 0.0, 1.0]),
    ],
)
def test_invalid_normals_raise_value_error(normal):
    with pytest.raises(ValueError):
        _call(TOP_POS, normal)


def test_negative_tolerance_raises_value_error():
    with pytest.raises(ValueError):
        _call(TOP_POS, TOP_N, tolerance=-1e-3)


def test_residual_is_finite_scalar_float():
    out = _call(EDGE_POS, EDGE_N)
    residual = out["normal_cone_residual"]
    assert np.isscalar(residual) or np.ndim(residual) == 0
    assert np.isfinite(float(residual))
