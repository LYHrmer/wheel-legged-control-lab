import csv
import gzip
import hashlib
import importlib.util
import json
import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.continuous_task import (
    STAGES,
    ContinuousTaskConfig,
    D1ContinuousTask,
    YawRatePI,
    road_height,
    scheduled_command,
    summarize_task,
)
from wheel_legged_control.d1.env import encode_d1_observation
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.training_terrain import TrainingGroundReference

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def task():
    return D1ContinuousTask()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 29},
        {"duration_s": 61},
        {"duration_s": True},
        {"duration_s": "45"},
        {"duration_s": np.nan},
        {"yaw_kp": np.inf},
        {"yaw_ki": -1},
        {"yaw_controller": "hidden"},
        {"ground_reference_mode": "auto"},
    ],
)
def test_config_rejects_ambiguous_or_unsafe_values(kwargs):
    with pytest.raises(ValueError):
        ContinuousTaskConfig(**kwargs)


def test_road_has_fixed_amplitude_grade_and_continuous_edges():
    assert road_height(-2.8) == pytest.approx(0.005)
    assert road_height(-2.4) == pytest.approx(-0.005)
    epsilon = 1e-5
    for x in (-3, -1.5, -1.1, 0.1, 0.5, 1.7):
        left = (road_height(x) - road_height(x - epsilon)) / epsilon
        right = (road_height(x + epsilon) - road_height(x)) / epsilon
        assert abs(left - right) < 1e-4
        assert abs(left) < 1e-4
    assert (road_height(-0.49) - road_height(-0.5)) / 0.01 == pytest.approx(np.tan(np.deg2rad(4)))
    assert (road_height(1.01) - road_height(1)) / 0.01 == pytest.approx(-np.tan(np.deg2rad(4)))
    assert road_height(-3.8) == 0
    assert abs(road_height(5.0)) < 1e-12


def test_actual_compiled_collision_and_query_use_same_bounds_and_profile(task):
    task.reset()
    plant = task.plant
    hfield = plant.model.geom_dataid[plant.floor_geom_id]
    np.testing.assert_array_equal(plant.model.hfield_size[hfield], task.road["size"])
    assert task.road["bounds_x_m"] == [-6, 6]
    assert task.road["bounds_y_m"] == [-3, 3]
    for x in (-5.99, -2.793, -1.103, -0.503, 0.303, 1.013, 1.703, 5.99):
        reference = plant.training_ground_reference(x, 2.0)
        geom, normal = np.zeros(1, dtype=np.int32), np.zeros(3)
        distance = mujoco.mj_ray(
            plant.model,
            plant.data,
            np.array((x, 2, 2.0)),
            np.array((0.0, 0.0, -1.0)),
            None,
            True,
            -1,
            geom,
            normal,
        )
        assert geom[0] == plant.floor_geom_id
        assert reference.height_m == pytest.approx(2 - distance, abs=1e-10)
        np.testing.assert_allclose(
            normal, (np.sin(reference.pitch_rad), 0, np.cos(reference.pitch_rad)), atol=1e-10
        )
        assert abs(reference.height_m - road_height(x)) < 4e-6


def test_schedule_is_continuous_and_duration_scaling_preserves_route():
    for _, start, _, _, _ in STAGES[1:]:
        before = scheduled_command(start - 1e-7, 45)
        after = scheduled_command(start + 1e-7, 45)
        np.testing.assert_allclose(before[1:3], after[1:3], atol=1e-10)
    for duration in (30.0, 45.0, 60.0):
        for template_time in (0, 5, 11, 18, 26, 32, 38, 44):
            scaled = scheduled_command(template_time * duration / 45, duration)
            nominal = scheduled_command(template_time, 45)
            assert scaled[0] == nominal[0]
            np.testing.assert_allclose(np.asarray(scaled[1:3]) * duration / 45, nominal[1:3])


def test_observation_preserves_oracle_v2_layout_but_names_new_control(task):
    observation, _ = task.reset(seed=0)
    command = task._command()
    base = encode_d1_observation(
        state=task.state,
        command=command,
        baseline_longitudinal_force_n=0,
        previous_applied_action=np.zeros(2),
    )
    np.testing.assert_array_equal(observation[:42], base)
    np.testing.assert_array_equal(observation[42:], (command.pitch_rad, command.roll_rad))
    assert task.control_schema == "d1-lqr-vmc-local-tangent-yaw-pi-v1"


def test_command45_adds_yaw_with_distinct_source_schemas():
    oracle = D1ContinuousTask(ContinuousTaskConfig(observation_layout="command45"))
    assert oracle.observation_schema == "d1-continuous-oracle-command45-v1"
    oracle.steps = 3200
    assert oracle.observation().shape == (45,)
    assert oracle.observation()[-1] == pytest.approx(0.08)
    estimated = D1ContinuousTask(
        ContinuousTaskConfig(observation_layout="command45", ground_reference_mode="estimated"),
        state_source_factory=D1MujocoTruthStateSource,
        ground_reference=lambda _: TrainingGroundReference(0, 0, 0),
    )
    assert estimated.observation_schema == "d1-continuous-sensor-command45-v1"
    assert estimated.observation_space.shape == (45,)


def test_sensor_training_constructor_is_pickleable():
    from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource

    config = ContinuousTaskConfig(ground_reference_mode="estimated", observation_layout="command45")
    factory = partial(D1SensorStateSource, initial_position=(-3.8, 0, 0.455), initial_rpy=(0, 0, 0))
    restored_config, restored_factory = pickle.loads(pickle.dumps((config, factory)))
    assert restored_config == config
    assert restored_factory.keywords == factory.keywords


def test_static_map_and_no_implicit_mid_episode_reset(task, monkeypatch):
    task.reset(seed=0)
    before = task.plant.model.hfield_data.copy()
    identity = id(task.plant.model)
    monkeypatch.setattr(task.plant, "reset", lambda **_: pytest.fail("hidden reset"))
    monkeypatch.setattr(task.plant, "set_training_terrain", lambda *_: pytest.fail("floor switch"))
    for _ in range(20):
        task.step(np.zeros(2))
    assert task.steps == 20 and id(task.plant.model) == identity
    np.testing.assert_array_equal(task.plant.model.hfield_data, before)


def test_estimated_control_observation_do_not_query_collision_truth(monkeypatch):
    # A fake provider tests the seam, not the sensor estimator's accuracy.
    env = D1ContinuousTask(
        ContinuousTaskConfig(ground_reference_mode="estimated"),
        state_source_factory=D1MujocoTruthStateSource,
        ground_reference=lambda _: TrainingGroundReference(0.01, 0.02, -0.01),
    )
    assert env.observation_schema != "d1-terrain-tracking-oracle-v2"
    monkeypatch.setattr(
        env.plant,
        "training_ground_reference",
        lambda *_: pytest.fail("oracle leaked into command/observation"),
    )
    assert env._command().base_height_m == pytest.approx(0.465)
    assert np.isfinite(env.observation()).all()


def test_estimated_mode_requires_explicit_providers():
    with pytest.raises(ValueError, match="providers"):
        D1ContinuousTask(ContinuousTaskConfig(ground_reference_mode="estimated"))


def test_sensor_two_axis_ground_angles_recover_the_actual_support_normal():
    from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource
    from wheel_legged_control.d1.terrain_tracking_env import terrain_normal_to_rpy

    env = D1ContinuousTask(
        ContinuousTaskConfig(ground_reference_mode="estimated"),
        state_source_factory=partial(D1SensorStateSource, initial_position=(-3.8, 0, 0.455)),
    )
    slope_x, slope_y = 0.15, 0.20
    normal = np.asarray((-slope_x, -slope_y, 1.0))
    env.source.estimator._normal = normal / np.linalg.norm(normal)
    expected_roll, expected_pitch = terrain_normal_to_rpy(slope_x, slope_y, 0.0)
    command = env._command()
    assert command.roll_rad == pytest.approx(expected_roll, abs=1e-12)
    assert command.pitch_rad == pytest.approx(expected_pitch, abs=1e-12)


def test_yaw_pi_bounds_and_antiwindup():
    controller = YawRatePI(2, 3)
    assert controller.compute(0.08, 0.01) > 0
    for _ in range(2000):
        assert abs(controller.compute(0.5, 0.01)) <= 4
    assert controller.integral_nm <= 2
    controller.reset()
    for _ in range(10):
        assert controller.compute(10, 0.01) == 4
    assert controller.integral_nm == 0  # Saturating proportional demand must not wind up.
    assert controller.compute(-0.08, 0.01) < 0


def test_checkpoint_transfer_needs_explicit_permission_and_matching_layout(task, tmp_path):
    runner = load_script("run_d1_continuous_task")
    path = tmp_path / "model.zip"
    path.write_bytes(b"test checkpoint bytes, not a trained model")
    metadata = {
        "observation_schema": task.observation_schema,
        "action_schema": task.action_schema,
        "control_schema": "d1-lqr-vmc-local-tangent-v2",
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    model = SimpleNamespace(
        observation_space=task.observation_space, action_space=task.action_space
    )
    with pytest.raises(ValueError, match="control_schema"):
        runner.validate_checkpoint(metadata, model, task, path)
    runner.validate_checkpoint(metadata, model, task, path, allow_v2_transfer=True)
    metadata["observation_schema"] = "old-42"
    with pytest.raises(ValueError, match="observation_schema"):
        runner.validate_checkpoint(metadata, model, task, path, allow_v2_transfer=True)


def test_frame_sampling_retains_initial_and_failure_endpoint():
    renderer = load_script("render_d1_continuous_task")
    assert renderer.frame_indices(13, 0.01, 20) == [0, 5, 10, 13]
    with pytest.raises(ValueError, match="integral"):
        renderer.frame_indices(13, 0.01, 30)


def test_fx_only_adapter_is_explicit_and_scale_checked(task, tmp_path):
    runner = load_script("run_d1_continuous_task")
    schema = "d1-terrain-residual-longitudinal-only-v1"
    np.testing.assert_array_equal(runner.adapt_policy_action([0.4], schema), [0.4, 0])
    with pytest.raises(ValueError, match="unregistered"):
        runner.adapt_policy_action([0.4], "unknown-even-if-shape-one")
    with pytest.raises(ValueError, match="shape"):
        runner.adapt_policy_action([0.4, 0.5], schema)
    path = tmp_path / "checkpoint.zip"
    path.write_bytes(b"fake checkpoint")
    metadata = {
        "observation_schema": task.observation_schema,
        "control_schema": task.control_schema,
        "action_schema": schema,
        "model_sha256": runner.sha256(path),
        "residual_force_scale_n": [11.25],
    }
    model = SimpleNamespace(
        observation_space=task.observation_space, action_space=SimpleNamespace(shape=(1,))
    )
    runner.validate_checkpoint(metadata, model, task, path)
    metadata["residual_force_scale_n"] = [45.0]
    with pytest.raises(ValueError, match="force scale"):
        runner.validate_checkpoint(metadata, model, task, path)
    del metadata["residual_force_scale_n"]
    with pytest.raises(ValueError, match="lacks"):
        runner.validate_checkpoint(metadata, model, task, path, allow_v2_transfer=True)


def test_shared_compressed_model_checks_both_hashes(tmp_path):
    renderer = load_script("render_d1_continuous_task")
    directory = tmp_path / "run"
    directory.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    raw = b"compiled model test bytes"
    content_sha = hashlib.sha256(raw).hexdigest()
    path = models / f"{content_sha}.mjb.gz"
    path.write_bytes(gzip.compress(raw, mtime=0))
    artifact = {
        "path": f"../models/{path.name}",
        "sha256": renderer.sha256(path),
        "uncompressed_sha256": content_sha,
    }
    manifest = {"compiled_model": artifact}
    assert renderer.compiled_model_bytes(directory, manifest) == raw
    artifact["sha256"] = "wrong"
    with pytest.raises(ValueError, match="compressed model hash"):
        renderer.compiled_model_bytes(directory, manifest)
    artifact["path"] = "/some/unrelated/model"
    with pytest.raises(ValueError, match="location"):
        renderer.compiled_model_bytes(directory, manifest)


def test_recording_reader_rejects_missing_or_altered_states(tmp_path):
    renderer = load_script("render_d1_continuous_task")
    (tmp_path / "model.mjb").write_bytes(b"fake; reader does not load MuJoCo")
    model_sha = renderer.sha256(tmp_path / "model.mjb")
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            {
                "control_dt_s": 0.01,
                "mode": "zero",
                "checkpoint_sha256": None,
                "compiled_model_sha256": model_sha,
            }
        )
    )
    (tmp_path / "summary.json").write_text("{}")
    with (tmp_path / "telemetry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "time_s"))
        writer.writeheader()
        writer.writerows([{"step": 1, "time_s": 0.01}, {"step": 2, "time_s": 0.02}])
    np.savez(
        tmp_path / "states.npz",
        time_s=[0, 0.01, 0.02],
        qpos=np.zeros((3, 23)),
        qvel=np.zeros((3, 22)),
    )
    manifest = {"sha256": {path.name: renderer.sha256(path) for path in tmp_path.iterdir()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    renderer.read_recording(tmp_path)
    (tmp_path / "states.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.read_recording(tmp_path)


def test_real_45_second_zero_rollout_covers_terrain_and_turns(task):
    """Regression for the long-episode yaw oscillation and near-zero-turn loophole."""
    task.reset(seed=0)
    rows = []
    for _ in range(task.max_steps):
        _, _, terminated, truncated, info = task.step(np.zeros(2))
        rows.append(info)
        if terminated or truncated:
            break
    report = summarize_task(rows)
    assert report["duration_s"] == 45
    assert report["completed"] and report["required_terrain_visited"]
    assert report["quality_pass"], report
    turn = {item["stage"]: item for item in report["stage_summaries"]}
    assert turn["turn_left"]["yaw_change_rad"] > 0.15
    assert turn["turn_right"]["yaw_change_rad"] < -0.10
    # A stationary-yaw trace must not pass merely because the robot survived.
    for row in rows:
        if row["stage"].startswith("turn_"):
            row["yaw_rad"] = 0
    assert not summarize_task(rows)["quality_pass"]
