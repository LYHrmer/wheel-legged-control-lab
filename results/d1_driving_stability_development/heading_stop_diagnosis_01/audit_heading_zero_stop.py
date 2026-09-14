"""Read existing G1 records; reconstruct wheel PI, never instantiate/step a plant."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(value):
    return {"mean": float(np.mean(value)), "rms": float(np.sqrt(np.mean(value**2))),
            "min": float(np.min(value)), "max": float(np.max(value))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"schema": "d1-heading-zero-stop-readonly-v1", "new_physics_steps": 0,
              "plant_instantiated": False, "repository_modified": False,
              "helper_sha256": sha(Path(__file__)), "cases": {}, "source_sha256": {}}
    sources = ["scripts/probe_d1_heading_g1.py", "scripts/d1_heading_tracking_env.py",
               "src/wheel_legged_control/d1/wheel_leg_controller.py",
               "src/wheel_legged_control/d1/control_loop.py",
               "src/wheel_legged_control/d1/locomotion_observation.py",
               "src/wheel_legged_control/d1/model.py"]
    protocol = json.loads((args.input / "protocol.json").read_text())
    for relative in sources:
        path = args.repo / relative
        value = sha(path)
        assert protocol["input_sha256"][str(path)] == value, relative
        report["source_sha256"][relative] = value
    for name in ("flat_forward_stop", "flat_reverse_stop"):
        root = args.input / name / "zero"
        with gzip.open(root / "trace.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        with np.load(root / "states.npz") as data:
            obs = data["observations"].copy()
            qvel = data["qvel"].copy()
            positions = data["truth_positions_world_m"].copy()
            assert np.array_equal(data["requested_actions"], np.zeros((800, 8)))
            assert np.array_equal(data["applied_actions"], np.zeros((800, 8)))
        metadata = json.loads((root / "episode_metadata.json").read_text())["episode_metadata"]
        assert metadata["baseline"] == "wheel_leg"
        assert metadata["actuator"]["delay_steps"] == 0
        assert metadata["actuator"]["time_constant_s"] == 0
        assert metadata["actuator"]["gain"] == 1
        assert np.array_equal((qvel[:, 6:] / np.tile([20., 20., 20., 30.], 4)).astype(np.float32), obs[:, 22:38])
        assert np.all(obs[:, 68:70] == 0) and np.all(obs[:, 70:74] == 1)
        assert np.all(obs[:, 41:43] == 0)
        assert all(row["heading_task"]["user_command_before"]["forward_velocity_mps"] == 0
                   and row["heading_task"]["servo_command_before"]["forward_velocity_mps"] == 0
                   for row in rows[400:])
        assert all(not any(sample["wrench_world"]) for row in rows for sample in row["physics_wrench_samples"])
        wheels = qvel[:, 6:][:, [3, 7, 11, 15]]
        target = obs[:, 56:60].astype(float) * 30.
        integral = obs[:, 64:68].astype(float) * 4.
        error = target[:-1] - wheels[:-1]
        next_i = np.clip(integral[:-1] + .03 * error, -4., 4.)
        request = 2.2 * error + next_i
        expected_i = np.where((np.abs(request) <= 12) | (request * error < 0), next_i, integral[:-1])
        recurrence_error = float(np.max(np.abs(expected_i - integral[1:])))
        assert recurrence_error < 1e-7
        torque = 2.2 * error + integral[1:]
        assert np.max(np.abs(torque)) < 12 and np.max(np.abs(wheels)) < 30
        # Conservatively use a full float32 spacing for each decoded field.
        bound = 2.2 * 30 * np.abs(np.spacing(obs[:-1, 56:60])).astype(float)
        bound += 4 * np.abs(np.spacing(obs[1:, 64:68])).astype(float)
        vx = np.r_[obs[0, 0], [row["body_forward_mps"] for row in rows]]
        pitch = np.r_[0., [row["actual_roll_pitch_rad"][1] for row in rows]]
        assert np.array_equal(vx.astype(np.float32), obs[:, 0])
        summary = json.loads((root / "summary.json").read_text())
        late_speed = float(np.max(np.abs(vx[500:800])))
        late_path = float(np.linalg.norm(np.diff(positions[500:701, :2], axis=0), axis=1).sum())
        assert late_speed == summary["late_stop_max_abs_body_vx_mps"]
        assert late_path == summary["late_stop_cumulative_planar_path_m"]
        windows = {}
        for start, end in ((4, 5), (5, 6), (6, 7), (7, 8), (5, 8)):
            selection = slice(start * 100, end * 100)
            windows[f"{start}to{end}s"] = {
                "body_vx_mps": stats(vx[selection]),
                "wheel_mean_times_radius_mps": stats(wheels.mean(axis=1)[selection] * .087),
                "pitch_rad": stats(pitch[selection]),
                "reconstructed_mean_wheel_torque_nm": stats(torque.mean(axis=1)[selection]),
            }
        samples = []
        for tick in (399, 400, 401, 405, 410, 425, 450, 475, 500, 550, 600, 650, 700, 750, 799):
            samples.append({"decision_tick": tick, "decision_time_s": tick / 100,
                            "body_vx_mps": float(vx[tick]),
                            "wheel_mean_times_radius_mps": float(wheels[tick].mean() * .087),
                            "nominal_wheel_mean_times_radius_mps": float(target[tick].mean() * .087),
                            "wheel_integral_mean_nm_before": float(integral[tick].mean()),
                            "reconstructed_wheel_mean_torque_nm_this_interval": float(torque[tick].mean()),
                            "actual_pitch_rad": float(pitch[tick])})
        report["cases"][name] = {
            "input_sha256": {p.name: sha(p) for p in root.iterdir() if p.is_file()},
            "checks_passed": True, "transitions": len(rows),
            "baseline": metadata["baseline"], "controller_schema": metadata["controller_schema"],
            "controller_parameters": metadata["controller_parameters"],
            "distance_reference_present": False, "pitch_reference_max_abs_rad": 0.,
            "zero_forward_command_from_tick": 400, "zero_residual_actions": True,
            "external_wrench_all_zero": True, "integral_recurrence_max_error_nm": recurrence_error,
            "reconstructed_torque_max_quantization_bound_nm": float(np.max(bound)),
            "reconstructed_wheel_torque_peak_abs_nm": float(np.max(np.abs(torque))),
            "wheel_integral_peak_abs_nm": float(np.max(np.abs(integral))),
            "stop_nominal_wheel_common_equivalent_speed_peak_abs_mps": float(np.max(np.abs(target[400:].mean(axis=1) * .087))),
            "late_speed_gate_mps": late_speed, "late_speed_limit_mps": .03,
            "late_planar_path_gate_m": late_path, "late_planar_path_limit_m": .05,
            "parking_gates_passed": late_speed <= .03 and late_path <= .05,
            "recorded_failed_gates": summary["gates"]["failed"],
            "terminal_reason": rows[-1]["terminal_reason"], "samples": samples, "windows": windows,
        }
    report["torque_method"] = "tau[k] = 2.2*(30*obs[k,56:60]-qvel[k,6:][wheel]) + 4*obs[k+1,64:68]. Observations store float32; torque is reconstructed, not directly recorded. All wheel torque/speed bounds are inactive; actuator has unit gain, no lag/delay. No wheel position limit exists."
    report["interpretation"] = "Confirmed abrupt nominal-wheel stop and body/wheel speed separation with decaying/reversing motion; no distance reference or pitch target leftover, no sustained integral/torque saturation. Leg/body elastic-motion excitation is a supported mechanism hypothesis, not a proven unique cause. Eight seconds cannot establish asymptotic steady state."
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit_checks_passed": True, "new_physics_steps": 0,
                      "parking_gates_passed": {name: item["parking_gates_passed"] for name, item in report["cases"].items()},
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
