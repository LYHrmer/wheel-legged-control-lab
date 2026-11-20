import math

import numpy as np
import pytest

from scripts import terrain_preview85_sensing as sense


def test_same_origin_and_only_nearest_beam_changes():
    old, new = (sense.RangeMount(a) for a in sense.ANGLES)
    np.testing.assert_array_equal(old.directions, sense.preview.DIRECTIONS)
    np.testing.assert_array_equal(new.directions[1:], old.directions[1:])
    np.testing.assert_allclose(
        new.directions[0], [math.cos(math.radians(82)), 0.0, -math.sin(math.radians(82))],
    )
    np.testing.assert_array_equal(sense.ORIGINS_M, sense.preview.ORIGINS + [0.16, 0, 0])
    raw = np.array([[0.1, 0.3, 0.6, 1.5], [0.14, 0.4, 1.45, -1]])
    a, b = [m.frame(5, 0.2, 0.02, raw, 0.1, -0.04) for m in (old, new)]
    np.testing.assert_array_equal(np.isnan(a.x_m), [[True, False, False, True],
                                                    [False, False, False, True]])
    np.testing.assert_array_equal(a.x_m[:, 1:], b.x_m[:, 1:])
    np.testing.assert_array_equal(a.z_m[:, 1:], b.z_m[:, 1:])
    assert a.x_m[1, 0] != b.x_m[1, 0] and a.available_s == pytest.approx(0.24)


def test_true_ground_filter_is_separate_from_controller_ranges():
    mount = sense.RangeMount(82)
    raw = np.tile(0.4 / -mount.directions[:, 2], (2, 1))
    frame = mount.frame(0, 0.0, 0.0, raw, 0.0, 0.0)
    mask = np.ones((2, 4), dtype=bool)
    mask[0, 0] = False
    oracle = sense.ground_only(frame, mask)
    query = float(np.mean(frame.x_m[0, :2]))
    assert sense.preview.query_frame(frame, 0.05, 0.0, query, 0.0).valid
    assert not sense.preview.query_frame(oracle, 0.05, 0.0, query, 0.0).valid
    assert np.isfinite(frame.x_m[0, 0]) and np.isnan(oracle.x_m[0, 0])


def sample(step, sequence, slope=1.0, *, capture=0.0, delivered=0.04):
    part = {"sequence": sequence, "reason": "valid", "slope_rad": math.radians(slope),
            "ground_reason": "unobserved", "ground_slope_rad": None,
            "captured_s": capture, "available_s": delivered}
    return {"control_s": step * 0.01, "step": step, "mounts": {"82": part}}


def test_requires_three_adjacent_ticks_of_the_same_fresh_precontact_packet():
    rows = [sample(4, 0), sample(5, 0), sample(6, 1), sample(7, 1), sample(8, 1)]
    good = sense.summarize_mount(rows, "82", 1, 0.09)
    assert good["passed"] and good["first_confirmation_s"] == pytest.approx(0.08)
    assert good["first_streak_start_s"] == pytest.approx(0.06) and good["first_qualifying_packet"] == 1
    assert not sense.summarize_mount(rows, "82", 1, 0.08)["passed"]
    assert not sense.summarize_mount(rows, "82", 1, None)["passed"]
    assert not sense.summarize_mount(rows, "82", 1, 0.09, mode="floor_returns")["passed"]
    assert not sense.summarize_mount([sample(i, 0) for i in (4, 5, 7)], "82", 1, 0.1)["passed"]
    assert not sense.summarize_mount([sample(i, 0) for i in (2, 3, 4)], "82", 1, 0.1)["passed"]
    assert not sense.summarize_mount([sample(i, 0) for i in (25, 26, 27)], "82", 1, 0.3)["passed"]
    wrong = sense.summarize_mount(rows, "82", -1, 0.09)
    assert not wrong["passed"] and wrong["precontact_wrong_packets"] == 2
    assert wrong["precontact_wrong_ticks"] == 5


def test_ground_chords_reject_estimator_tilt_on_flat_ground_with_same_online_pairs():
    mount = sense.RangeMount(82)
    base = np.array([0.0, 0.0, 0.455])
    raw = np.tile((base[2] + sense.ORIGINS_M[0, 2]) / -mount.directions[:, 2], (2, 1))
    ids = np.zeros((2, 4), dtype=int)
    true_coordinates, audit = sense.capture_truth(
        mount, raw, ids, base, np.eye(3), np.array([1.0, 0.0, 0.0]), 0,
    )
    np.testing.assert_allclose(np.array(audit["first_hit_world_xyz_m"])[..., 2], 0.0, atol=1e-12)
    frame = mount.frame(0, 0.0, 0.0, raw, -math.radians(4), 0.0)
    query = float(np.mean(frame.x_m[0, :2]))
    rows = []
    for i in (4, 5, 6):
        measured = sense.preview.query_frame(frame, i * 0.01, 0.0, query, 0.0)
        assert math.degrees(measured.slope_rad) == pytest.approx(4)
        row = sense.reading_row(measured, measured, true_coordinates)
        assert row["online_pair_start_beam_by_track"] == [0, 0]
        np.testing.assert_allclose(row["true_floor_chord_rad_by_track"], 0, atol=1e-12)
        rows.append({"step": i, "control_s": i * 0.01, "mounts": {"82": row}})
    primary = sense.summarize_mount(rows, "82", 1, 0.08)
    assert primary["passed"]
    assert sense.summarize_mount(rows, "82", 1, 0.08, mode="floor_returns")["passed"]
    assert not sense.summarize_mount(rows, "82", 1, 0.08, mode="matched_chords")["passed"]
    first = sense.first_window_ground_evidence(rows, "82", 1, primary)
    assert first["sequence"] == 0 and len(first["ticks"]) == 3 and not first["all_three_confirmed"]
    true_coordinates[0][0, 0] = np.nan
    indices, slopes = sense.matched_chords(frame, query, true_coordinates)
    assert indices == [0, 0] and slopes[0] is None
