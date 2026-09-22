"""Synthetic archive mutation tests; no physical data or simulation claims."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts.d1_single_step_scoring import score_single_step


def synthetic_archive(count=1200):
    geoms = []
    for gid in range(7):
        wheel = gid-2 if 2 <= gid <= 5 else None
        geoms.append({"geom_id": gid, "identity": f"geom{gid}", "body_id": 0 if gid < 2 else gid,
            "body_name": "world" if gid < 2 else f"body{gid}", "geom_type": 0 if gid == 0 else 6,
            "margin_m": 0. if gid < 2 else .001, "wheel_index": wheel, "collision": True,
            "terrain_kind": "plane" if gid == 0 else "box" if gid == 1 else None,
            "position_local_m": [-3.1, 0., .0075] if gid == 1 else [0., 0., 0.], "size_m": [.18, .62, .0075]})
    def pose(native):
        x = -3.8+.0004*np.clip(native-1000, 0, 4000)
        return np.r_[x, 0., .455, 1., np.zeros(19)]
    def velocity(native):
        return np.r_[.2 if 1000 <= native < 5000 else 0., np.zeros(21)]
    endpoints, trace, native = [], [], []
    for tick in range(count+1):
        q = pose(tick*5)
        bounds = []
        for geom in geoms[2:]:
            bounds.append({k: geom[k] for k in ("geom_id", "identity", "body_name", "geom_type", "margin_m", "wheel_index")}
                          | {"minimum_world_m": [q[0]-.35, -.1, 0.], "maximum_world_m": [q[0]+.35, .1, .6]})
        endpoints.append({"tick": tick, "time_s": tick*.01, "base_position_m": q[:3].tolist(), "rpy_rad": [0., 0., 0.],
            "body_vx_mps": float(velocity(tick*5)[0]), "com_vz_mps": 0., "com_velocity_mps": [0., 0., 0.],
            "collision_bounds": bounds, "nonwheel_terrain_contacts": 0})
        if tick < count:
            trace.append({"tick": tick, "action": [0.]*8,
                "raw_command": {"tick": tick, "forward_velocity_mps": .2 if 200 <= tick < 1000 else 0., "yaw_rate_rps": 0.,
                    "raw_world_height_m": .455, "prepared_world_height_m": .455, "ground_height_m": 0., "motion_clearance_m": .455},
                "stop_active": tick >= 1000, "turn_active": False, "torque_nm": [0.]*16, "callback_count_after_prepare": tick+2})
    for idx in range(count*5):
        contacts = []
        for wheel in range(4):
            box = idx == 2400 and wheel == 0
            g1, g2 = int(box), wheel+2
            frame = np.diag([-1., -1., 1.]) if box else np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
            contacts.append({"geom1": g1, "geom2": g2, "geom1_identity": geoms[g1]["identity"], "geom2_identity": geoms[g2]["identity"],
                "body1": 0, "body2": g2, "body1_name": "world", "body2_name": f"body{g2}", "frame_geom1_to_geom2": frame.tolist(),
                "position_world_m": [-3.28, 0., .0075] if box else [0., 0., 0.], "distance_m": 0., "inclusion_margin_m": .001,
                "friction": [.9, .005, .0001, .0001, .0001], "efc_address": wheel, "active": True,
                "normal_load_n": 120., "local_force_torque": [120., 0., 0., 0., 0., 0.],
                "terrain_geom_id": g1, "robot_geom_id": g2, "wheel_index": wheel,
                "normal_terrain_to_robot_world": frame[0].tolist(), "force_on_robot_world_n": (120*frame[0]).tolist(),
                "frame_orthonormal": True, "plane_normal_valid": True,
                "box_feature": {"geometric_support_valid": True, "feature": "front"} if box else None})
        native.append({"index": idx, "returned": True, "start_time_s": idx*.002, "end_time_s": (idx+1)*.002, "actual_dt_s": .002,
            "qpos_before": pose(idx), "qpos_returned": pose(idx+1), "qvel_before": velocity(idx), "qvel_returned": velocity(idx+1),
            "ctrl_nm": np.zeros(16), "xfrc_applied": np.zeros((18, 6)), "qfrc_applied": np.zeros(22),
            "contacts": {"contacts": contacts, "wheel_positive_normal_load_n": [120.]*4,
                "wheel_box_positive_normal_load_n": [120. if idx == 2400 else 0., 0., 0., 0.],
                "nonwheel_terrain_contacts": 0, "geometric_box_contact": idx == 2400}})
    return {"obstacle_enabled": True, "completed_control_intervals": count, "endpoints": endpoints, "trace": trace, "native": native,
                "geometry_manifest": {"geoms": geoms, "base_body_ipos_local_m": [0., 0., 0.]}, "error": None, "source_identity_valid": True, "terminated": count < 1200, "truncated": count == 1200}


@pytest.fixture(scope="module")
def complete_archive():
    return synthetic_archive()


def test_complete_synthetic_positive_control(complete_archive):
    score = score_single_step(complete_archive)
    assert score["record_valid"] and score["task_passed"], score
    assert score["metrics"]["final_endpoint_window_ticks"] == [1101, 1200]
    assert score["metrics"]["final_native_window_indices"] == [5500, 5999]


@pytest.mark.parametrize("mutation", ["loads", "body", "frame", "missing_nonwheel_bound", "short_native", "nan", "height", "source", "local_force", "missing_feature"])
def test_corrupt_archive_cannot_pass(complete_archive, mutation):
    p = deepcopy(complete_archive)
    if mutation == "loads":
        p["native"][-1]["contacts"]["wheel_positive_normal_load_n"][0] = 121.
    elif mutation == "body":
        p["native"][0]["contacts"]["contacts"][0]["body2"] = 99
    elif mutation == "frame":
        p["native"][0]["contacts"]["contacts"][0]["frame_geom1_to_geom2"][0] = [0., 0., 0.]
    elif mutation == "missing_nonwheel_bound":
        p["endpoints"][-1]["collision_bounds"].pop()
    elif mutation == "short_native":
        p["native"].pop()
    elif mutation == "nan":
        p["native"][0]["qvel_before"][0] = float("nan")
    elif mutation == "height":
        p["endpoints"][-1]["base_position_m"][2] = .5
    elif mutation == "source":
        p["source_identity_valid"] = False
    elif mutation == "local_force":
        p["native"][0]["contacts"]["contacts"][0]["local_force_torque"][0] = 0.
    elif mutation == "missing_feature":
        p["native"][2400]["contacts"]["contacts"][0]["box_feature"] = None
    score = score_single_step(p)
    assert not score["record_valid"] and not score["task_passed"], score


def test_base_crossing_does_not_hide_trailing_collision_geom(complete_archive):
    p = deepcopy(complete_archive)
    p["endpoints"][-1]["collision_bounds"][-1]["minimum_world_m"][0] = -3.
    score = score_single_step(p)
    assert score["record_valid"] and not score["mechanical_crossing"] and not score["task_passed"]


def test_physical_early_stop_is_valid_incomplete_record():
    score = score_single_step(synthetic_archive(3))
    assert score["record_valid"] and not score["complete"] and not score["task_passed"], score


def test_no_empty_record_pass():
    assert not score_single_step({})["record_valid"]


def test_active_zero_force_does_not_count_as_loaded(complete_archive):
    p = deepcopy(complete_archive)
    for row in p["native"][5500:5526]:
        c = row["contacts"]["contacts"][0]
        c["normal_load_n"] = 0.; c["local_force_torque"] = [0.]*6; c["force_on_robot_world_n"] = [0.]*3
        row["contacts"]["wheel_positive_normal_load_n"][0] = 0.
    score = score_single_step(p)
    assert score["record_valid"] and not score["final_stable"], score
    assert score["metrics"]["final_window_wheel_positive_load_fractions"][0] == 474/500
