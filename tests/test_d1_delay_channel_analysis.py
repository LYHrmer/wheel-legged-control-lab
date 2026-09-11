"""Self-contained artifact fixtures exercise rejection without running MuJoCo."""
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_delay_channel_study.py"
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().with_name("analyze_delay_channel_study.py")
spec = importlib.util.spec_from_file_location("delay_analysis_under_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
analyze = module.analyze


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path):
    specifications = [("fusion_noise", m, a) for m, a in ((0, 0), (10, 0), (20, 0), (30, 0), (0, 10), (0, 20), (0, 30))]
    specifications += [("fusion_ideal", 0, 0), ("fusion_ideal", 30, 0), ("oracle", 0, 0), ("oracle", 0, 30), ("truth_delayed", 30, 0)]
    profiles = [{"name": f"{source}_m{m}_a{a}", "source": source, "measurement_ms": m, "actuator_ms": a} for source, m, a in specifications]
    gains = {"wheel_kp": 0.55, "wheel_ki": 1.5, "yaw_feedback_gain": 4.0, "leg_feedback_scale": 1.0, "attitude_feedback_scale": 0.25}
    protocol = {"schema": "d1-delay-channel-diagnostic-v1", "profiles": profiles, "seeds": [17],
                    "duration_s": 3.0, "schedule_duration_s": 60.0, "terrain_suite": "v1", "split": "development", "roads": [0],
                    "controller": gains, "zero_residual": True, "workers": 1, "measurement_dt_s": 0.01,
                    "physics_dt_s": 0.002, "actuator_gain": 1.0, "actuator_time_constant_s": 0.0, "caveats": []}
    write_json(tmp_path / "protocol.json", protocol)
    source = tmp_path / "source/candidate/analyze_delay_channel_study.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(SCRIPT.read_bytes())
    manifest = {"candidate/analyze_delay_channel_study.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    write_json(tmp_path / "source_sha256.json", manifest)
    metrics = {"velocity_rmse_mps": 2.0, "yaw_rmse_rps": 3.0, "height_rmse_m": 4.0, "attitude_rmse_rad": 5.0}
    n = 300
    request = np.repeat(np.arange(n)[:, None] / 1000, 16, axis=1)
    limited = np.minimum(request, 0.2)
    controller = np.zeros((n, 8, 16))
    controller[:, 0], controller[:, 1] = request, limited
    for profile in profiles:
        directory = tmp_path / f"{profile['name']}_road0_seed17"
        directory.mkdir()
        delay = profile["actuator_ms"] // 2
        physical_input = np.repeat(limited, 5, axis=0)
        delayed = np.concatenate((np.zeros((delay, 16)), physical_input))[:len(physical_input)]
        traces = np.zeros((n * 5, 5, 16))
        traces[:, 0], traces[:, 1], traces[:, 2], traces[:, 3] = physical_input, physical_input, delayed, delayed
        np.savez_compressed(directory / "trajectory.npz", qpos=np.zeros((n + 1, 23)),
                            estimated_states=np.zeros((n, 12)), actuator_trace=traces.reshape(n, 5, 5, 16), controller_trace=controller)
        rows = []
        for i in range(n):
            time = (i + 1) * 0.01
            age = min(time, profile["measurement_ms"] / 1000)
            rows.append({"time_s": time, "measurement_time_s": time-age, "measurement_age_s": age,
                "velocity_error_mps": 2.0, "yaw_rate_error_rps": 3.0, "height_error_m": 4.0,
                "roll_error_rad": 3.0, "pitch_error_rad": 4.0,
                "measurement_aligned_position_error_norm_m": 0.0, "measurement_aligned_velocity_error_norm_mps": 0.0,
                "controller_torque_clipped_fraction": float(i > 200), "controller_torque_changed_fraction": float(i > 200),
                "controller_position_guard_fraction": 0.0, "controller_velocity_guard_fraction": 0.0,
                "controller_joint_target_limited_fraction": 0.0})
        with (directory / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        write_json(directory / "result.json", {"error": None, "executed_steps": n, "duration_s": 3.0,
            "requested_duration_s": 3.0, "completed": True, "terminal_reason": "time_limit", "episode_seed": 17,
            "partial_horizon_tracking": metrics, "full_horizon_tracking": metrics, "profile": profile})
        write_json(directory / "episode.json", {"source_manifest": manifest, "controller_parameters": gains, "profile": profile, "episode_seed": 17,
            "measurement_seed": 1814784298, "actuator": {"gain": 1.0, "time_constant_s": 0.0, "delay_steps": delay, "torque_limit_nm": [0.2] * 16}})
    return tmp_path


def test_independent_analysis(artifacts):
    result = analyze(artifacts)
    assert result["case_count"] == 12
    assert all(case["completed"] for case in result["cases"])


@pytest.mark.parametrize("damage", ["missing_case", "source_hash", "delay_queue", "protocol_profile", "fabricated_metrics", "low_level_clip", "result_profile", "result_seed", "episode_profile", "episode_seed"])
def test_reject_damaged_artifacts(artifacts, damage):
    case = artifacts / "fusion_noise_m0_a0_road0_seed17"
    if damage == "missing_case":
        (case / "result.json").rename(case / "result.missing")
    elif damage == "source_hash":
        source = artifacts / "source/candidate/analyze_delay_channel_study.py"
        source.write_bytes(source.read_bytes() + b"\n# changed\n")
    elif damage == "protocol_profile":
        path = artifacts / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["profiles"].pop()
        write_json(path, protocol)
    elif damage == "fabricated_metrics":
        path = case / "result.json"
        result = json.loads(path.read_text())
        result["full_horizon_tracking"]["velocity_rmse_mps"] = 0.0
        result["partial_horizon_tracking"]["velocity_rmse_mps"] = 0.0
        write_json(path, result)
    elif damage in ("result_profile", "result_seed", "episode_profile", "episode_seed"):
        document, field = damage.split("_")
        path = case / f"{document}.json"
        metadata = json.loads(path.read_text())
        if field == "profile":
            metadata["profile"]["source"] = "oracle"
        else:
            metadata["episode_seed"] = 29
        write_json(path, metadata)
    else:
        path = case / "trajectory.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if damage == "delay_queue":
            arrays["actuator_trace"][4, 0, 2, 0] += 1.0
        else:
            arrays["controller_trace"][250, 0, 0] = 0.0
        np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError):
        analyze(artifacts)


def test_formal_study_requires_controller_trace(artifacts):
    protocol_path = artifacts / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["duration_s"], protocol["seeds"] = 60.0, [17, 29]
    write_json(protocol_path, protocol)
    for profile in protocol["profiles"]:
        original = artifacts / f"{profile['name']}_road0_seed17"
        for seed in (17, 29):
            directory = artifacts / f"{profile['name']}_road0_seed{seed}"
            if seed == 29:
                shutil.copytree(original, directory)
            result_path = directory / "result.json"
            result = json.loads(result_path.read_text())
            result.update(episode_seed=seed, requested_duration_s=60.0, completed=False,
                          terminal_reason="fall_or_body_contact", full_horizon_tracking=None)
            write_json(result_path, result)
            episode_path = directory / "episode.json"
            episode = json.loads(episode_path.read_text())
            episode["episode_seed"] = seed
            write_json(episode_path, episode)
    write_json(artifacts / "finished.json", {"complete": True, "case_count": 24, "expected_case_count": 24})
    assert analyze(artifacts)["case_count"] == 24
    path = artifacts / "fusion_noise_m0_a0_road0_seed17/trajectory.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "controller_trace"}
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError, match="formal controller trace missing"):
        analyze(artifacts)
