"""Independent per-case source/config and physical-metric verification."""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from analyze_delay_channel_study import analyze
from wheel_legged_control.d1.locomotion_commands import schedule_to_dict, validation_command_schedule
from wheel_legged_control.d1.locomotion_terrain import locomotion_terrain_configs
from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, JOINT_POSITION_LOW, JOINT_TORQUE_LIMIT, JOINT_VELOCITY_LIMIT

parser = argparse.ArgumentParser()
parser.add_argument("study", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
root = args.study
analysis = analyze(root)
schedule = validation_command_schedule("development", 60)
terrain = asdict(locomotion_terrain_configs("development")[0])
noise = {"gyro_std_rad_s": 0.002, "accelerometer_std_m_s2": 0.03,
         "encoder_position_std_rad": 0.0005, "encoder_velocity_std_rad_s": 0.005,
         "gyro_bias_rad_s": [0., 0., 0.], "accelerometer_bias_m_s2": [0., 0., 0.]}
cases = []
for case in analysis["cases"]:
    directory = root / case["case"]
    episode = json.loads((directory / "episode.json").read_text())
    profile = case["profile"]
    provider = episode["provider"]
    assert episode["profile"] == profile and episode["episode_seed"] == case["episode_seed"]
    assert episode["baseline"] == "wheel_leg" and episode["duration_s"] == 60
    assert episode["terrain"] == terrain
    assert episode["command_schedule"] == schedule_to_dict(schedule)
    assert episode["domain"] == {"base_mass_scale": 1., "damping_scale": 1., "friction_scale": 1., "actuator_strength_scale": 1.}
    assert all(value is None or value == [1., 1.] for value in episode["randomization"].values())
    source = profile["source"]
    if source.startswith("fusion"):
        assert provider["kind"] == "imu_encoder_fusion"
        assert provider["sensor_delay_steps"] == profile["measurement_ms"] // 10
        expected = noise if source == "fusion_noise" else {key: ([0., 0., 0.] if isinstance(value, list) else 0.) for key, value in noise.items()}
        assert provider["sensor_noise"] == expected and provider["impairments"] is None
        assert provider["initial_position_m"] == [-3.8, 0., 0.455] and provider["initial_rpy_rad"] == [0., 0., 0.]
    elif source == "oracle":
        assert provider["kind"] == "oracle" and provider["impairments"] is None and provider["sensor_noise"] is None
    else:
        assert provider["kind"] == "truth_impairment" and provider["impairments"]["delay_steps"] == 3
        assert all(value == 0 for key, value in provider["impairments"].items() if key != "delay_steps")
    np.testing.assert_array_equal(episode["actuator"]["torque_limit_nm"], JOINT_TORQUE_LIMIT)
    with (directory / "metrics.csv").open() as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    assert all(np.isfinite(list(row.values())).all() for row in rows)
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        assert all(np.isfinite(data[key]).all() for key in data.files if data[key].dtype.kind in "fi")
        assert len(data["qvel"]) == len(rows) + 1
        np.testing.assert_allclose(data["qpos"][1:, :3], [[row[key] for key in ("x_m", "y_m", "z_m")] for row in rows], rtol=0, atol=1e-12)
        low = data["controller_trace"]
        request, torque, positions, velocities = low[:, 0], low[:, 1], low[:, 6], low[:, 7]
        saturated = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
        position_guard = ((positions >= JOINT_POSITION_HIGH) & (saturated > 0)) | ((positions <= JOINT_POSITION_LOW) & (saturated < 0))
        velocity_guard = (np.abs(velocities) >= JOINT_VELOCITY_LIMIT) & (saturated * velocities > 0)
        expected = saturated.copy()
        expected[position_guard | velocity_guard] = 0
        np.testing.assert_array_equal(torque, expected)
        for field, mask in (("controller_position_guard_fraction", position_guard), ("controller_velocity_guard_fraction", velocity_guard)):
            np.testing.assert_array_equal([row[field] for row in rows], np.mean(mask, axis=1))
        traces = data["actuator_trace"]
        clips = np.mean(traces[:, :, 0] != traces[:, :, 1], axis=(1, 2))
        np.testing.assert_array_equal([row["torque_input_clipped_fraction"] for row in rows], clips)
        np.testing.assert_allclose(case["mean_input_clipped_fraction"], np.mean(clips), rtol=0, atol=1e-12)
        np.testing.assert_allclose(case["measurement_age_max_s"], max(row["measurement_age_s"] for row in rows), rtol=0, atol=1e-12)
        power = np.mean(np.sum(np.abs(traces[:, :, 3] * traces[:, :, 4]), axis=2), axis=1)
        np.testing.assert_allclose([row["mechanical_power_w"] for row in rows], power, rtol=0, atol=1e-12)
        for row in rows:
            command = asdict(schedule.cmd_at(row["decision_time_s"]))
            for key, value in command.items():
                np.testing.assert_allclose(row["command_" + key], value, rtol=0, atol=1e-10)
    cases.append({"case": case["case"], "completed": case["completed"], "rows": len(rows),
                  "terminal_flags": {key: int(rows[-1][key]) for key in ("body_contact_detected", "attitude_above_0p85", "clearance_below_0p22")},
                  "sources_and_additional_metrics_verified": True})
assert json.loads((root / "source_consistency.json").read_text()) == {"changed_files": [], "checked_files": 71, "consistent": True}
report = {"case_count": len(cases), "completed": sum(case["completed"] for case in cases),
          "verified_source_files": analysis["verified_source_files"], "cases": cases,
          "additional_checks": ["provider kind, noise, delay, priors", "actual development terrain, command schedule, every command row", "domain and disabled randomization", "all numeric CSV/NPZ values finite", "qpos base position and qvel length", "low-level position and velocity protective zeroing", "per-axis channel clipping", "mechanical power", "maximum measurement age"],
          "analysis_checks": ["24-case matrix", "source hashes", "finished marker", "case identities", "mandatory formal controller traces", "independent RMSE", "full-horizon gating", "physical actuator delay queue", "measurement timestamp alignment", "low-level clipping"],
          "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with args.output.open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({"case_count": len(cases), "completed": report["completed"], "all_checks_passed": True,
                  "output": str(args.output), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
