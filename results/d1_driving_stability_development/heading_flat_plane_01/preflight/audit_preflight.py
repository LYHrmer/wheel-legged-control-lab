"""Reproduce bounded flat-plane preflight without any physics integration.

Run the dedicated guarded pytest suite separately and pass its JUnit receipt.
This helper checks frozen inputs, compiled-model invariants, initialization and
the same 9 turn + 28 stop saved poses. It never solves a saved-pose contact force
or advances a trajectory. mj_step/mj_step1/mj_step2 all raise if reached.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv, D1FlatPlanePlant
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
from wheel_legged_control.d1.model import D1Plant


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_snapshots(work):
    for side, condition, ticks in (
        ("left", "limit0p6", (200, 211, 230, 245, 250)),
        ("left", "limit1p0", (211, 250)),
        ("right", "limit0p6", (211,)),
        ("right", "limit1p0", (211,)),
    ):
        case = f"stationary_turn_{side}_hold"
        yield case, condition, work / "heading_turn_limit_02" / case / condition / "states.npz", ticks
    for case in ("flat_forward_stop", "flat_reverse_stop"):
        for condition in ("bypass", "release_0p5"):
            yield case, condition, work / "heading_release_01" / case / condition / "states.npz", (
                400, 401, 425, 449, 475, 500, 550,
            )


def snapshot_collision(plant, qpos, qvel):
    """Collision-only cache at saved configuration, always at audit time zero."""
    model = plant.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    assert data.time == 0.
    mujoco.mj_fwdPosition(model, data)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert data.time == 0.
    contacts = []
    for index, contact in enumerate(data.contact):
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if plant.floor_geom_id not in (geom1, geom2):
            continue
        other = geom2 if geom1 == plant.floor_geom_id else geom1
        body = int(model.geom_bodyid[other])
        if body not in plant.wheel_body_ids:
            continue
        raw = np.asarray(contact.frame).reshape(3, 3)[0].copy()
        # MuJoCo frame normal points geom1 -> geom2. Normalize direction to
        # ground -> wheel before testing +z; friction directions are unrelated.
        ground_to_wheel = raw if geom1 == plant.floor_geom_id else -raw
        horizontal = float(np.linalg.norm(ground_to_wheel[:2]))
        contacts.append({
            "contact_index": index,
            "geom1": geom1,
            "geom2": geom2,
            "wheel_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            "position_world_m": np.asarray(contact.pos).tolist(),
            "distance_m": float(contact.dist),
            "efc_address": int(contact.efc_address),
            "active_constraint": bool(contact.efc_address >= 0),
            "raw_frame_normal": raw.tolist(),
            "ground_to_wheel_normal": ground_to_wheel.tolist(),
            "horizontal_normal_magnitude": horizontal,
        })
    active = [c for c in contacts if c["active_constraint"]]
    assert all(c["horizontal_normal_magnitude"] <= 1e-12 for c in contacts)
    assert all(abs(c["ground_to_wheel_normal"][2] - 1.) <= 1e-12 for c in contacts)
    return {
        "audit_time_s": float(data.time),
        "qpos_qvel_unchanged": True,
        "wheel_floor_contact_count": len(contacts),
        "active_wheel_floor_contact_count": len(active),
        "normal_check": "passed" if active else "not_observed_no_active_contact",
        "contact_forces_reconstructed": False,
        "contacts": contacts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--pytest-junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo, work = args.repo.resolve(), args.work.resolve()
    test_path = repo / "tests/test_d1_flat_plane_env.py"
    spec = importlib.util.spec_from_file_location("plane_nonphysical_contract_tests", test_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    suites = ET.parse(args.pytest_junit).getroot()
    cases = list(suites.iter("testcase"))
    assert cases, "pytest receipt contains no tests"
    assert not any(list(suites.iter(tag)) for tag in ("failure", "error", "skipped"))
    budget_protocol = repo / "results/d1_budget_study/protocol.json"
    frozen = json.loads(budget_protocol.read_text())["source_sha256"]
    assert len(frozen) == 77
    assert all(sha(repo / path) == expected for path, expected in frozen.items())
    original_protocol = repo / "results/d1_driving_stability_development/heading_g1_01/evaluation_protocol.json"
    assert sha(original_protocol) == "cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc"
    paths = [
        Path(__file__).resolve(), test_path,
        repo / "scripts/d1_flat_plane_env.py",
        repo / "scripts/d1_heading_tracking_env.py",
        repo / "scripts/d1_heading_reference.py",
        budget_protocol, original_protocol, args.pytest_junit,
        work / "flat_plane_contract_20260920.md",
        work / "turn_contact_diagnosis_01/collision_report.json",
        work / "turn_contact_diagnosis_01/stop_collision_report.json",
        *[repo / path for path in frozen],
        *[path for _, _, path, _ in selected_snapshots(work)],
    ]
    input_hashes = {str(path.resolve()): sha(path) for path in paths}
    integration_attempts = []

    def forbidden(*_args, **_kwargs):
        integration_attempts.append(True)
        raise AssertionError("nonphysical audit reached an integration entry point")

    with patch.object(mujoco, "mj_step", forbidden), \
         patch.object(mujoco, "mj_step1", forbidden), \
         patch.object(mujoco, "mj_step2", forbidden):
        old = D1Plant(sampling_mode="synchronized", locomotion_terrain=D1LocomotionTerrainConfig())
        plane = D1FlatPlanePlant()
        checks = test_module.assert_compiled_invariants(old.model, plane.model)
        test_module.assert_zero_clock(old)
        test_module.assert_zero_clock(plane)
        plane.validate_flat_plane()
        initial_qpos = plane.data.qpos.copy()
        initial_qvel = plane.data.qvel.copy()
        snapshots = []
        for case, condition, path, ticks in selected_snapshots(work):
            with np.load(path, allow_pickle=False) as archive:
                qpos, qvel = archive["qpos"], archive["qvel"]
                for tick in ticks:
                    snapshots.append({"case": case, "condition": condition,
                                      "source_states": str(path), "source_endpoint_tick": tick,
                                      **snapshot_collision(plane, qpos[tick], qvel[tick])})
        assert len(snapshots) == 37
        np.testing.assert_array_equal(plane.data.qpos, initial_qpos)
        np.testing.assert_array_equal(plane.data.qvel, initial_qvel)
        test_module.assert_zero_clock(plane)
        callback_times = []

        def command(time_s):
            callback_times.append(time_s)
            return D1MotionCommand()

        env = D1FlatPlaneHeadingEnv(episode_seconds=.01, command_source=command)
        try:
            obs, info = env.reset(seed=55101)
            np.testing.assert_array_equal(env._prepare(), obs[:82])
            np.testing.assert_array_equal(env._prepare(), obs[:82])
            assert callback_times == [0.]
            test_module.assert_zero_clock(env.plant)
            initialization = {
                "data_time_s": float(env.plant.data.time),
                "measurement_time_s": float(env.plant.measurement_data.time),
                "qpos": env.plant.data.qpos.tolist(),
                "qvel": env.plant.data.qvel.tolist(),
                "qacc_warmstart": env.plant.data.qacc_warmstart.tolist(),
                "observation_shape": list(obs.shape), "observation_dtype": str(obs.dtype),
                "action_shape": list(env.action_space.shape),
                "command_callback_times_s": callback_times,
                "loop_plant_is_plane": env.loop.plant is env.plant,
                "provider_plant_is_plane": env.loop.provider._plant is env.plant,
                "episode_metadata": info["episode_metadata"],
            }
        finally:
            env.close()
    assert integration_attempts == []
    assert all(sha(path) == expected for path, expected in input_hashes.items())
    result = {
        "schema": "d1-flat-plane-nonphysical-preflight-v1",
        "passed": True,
        "new_physics_steps": 0,
        "integration_entry_points_guarded": ["mj_step", "mj_step1", "mj_step2"],
        "integration_attempts": 0,
        "all_data_times_zero": True,
        "mujoco_version": mujoco.__version__,
        "frozen_input_count": len(frozen), "frozen_inputs_match": True,
        "original_g1_protocol_sha256": sha(original_protocol),
        "pytest": {"passed": True, "test_count": len(cases), "junit_path": str(args.pytest_junit.resolve()),
                   "test_file_sha256": sha(test_path)},
        "compiled_invariant_fields": checks,
        "compiled_invariant_field_count": len(checks),
        "old_floor": {"geom_type": int(old.model.geom_type[old.floor_geom_id]),
                      "nhfield": int(old.model.nhfield),
                      "geom_pos": old.model.geom_pos[old.floor_geom_id].tolist()},
        "actual_plane": plane.collision_terrain_metadata,
        "initialization": initialization,
        "saved_pose_count": len(snapshots),
        "position_collision_calls_on_saved_poses": len(snapshots),
        "saved_pose_method": "new MjData per pose, qpos/qvel copied, time kept 0, mj_fwdPosition only",
        "saved_pose_frames_without_active_wheel_contact": sum(
            not s["active_wheel_floor_contact_count"] for s in snapshots),
        "saved_pose_total_active_wheel_contacts": sum(s["active_wheel_floor_contact_count"] for s in snapshots),
        "saved_pose_max_horizontal_normal": max(
            (c["horizontal_normal_magnitude"] for s in snapshots for c in s["contacts"]), default=None),
        "snapshots": snapshots,
        "input_sha256": input_hashes,
        "source_sha256": {str(repo / path): expected for path, expected in frozen.items()},
        "inputs_unchanged": True,
        "limitations": [
            "No trajectory, controller stability, stopping or turning gate is tested here.",
            "Saved poses came from old hfield trajectories; these are plane collision counterfactuals only.",
            "No historical contact force is reconstructed; no-contact frames do not pass a normal test.",
            "Construction/reset use mj_forward at time zero; saved-pose audit uses mj_fwdPosition only.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "passed", "new_physics_steps", "mujoco_version", "frozen_input_count",
        "compiled_invariant_field_count", "saved_pose_count", "saved_pose_frames_without_active_wheel_contact",
        "saved_pose_total_active_wheel_contacts", "saved_pose_max_horizontal_normal",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
