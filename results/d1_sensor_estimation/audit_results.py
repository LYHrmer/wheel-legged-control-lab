"""Recompute the sensor development report without rerunning simulation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from wheel_legged_control.d1.sensor_estimation import (
    D1ProprioceptiveEstimator,
    D1SensorMeasurements,
)


def main() -> None:
    directory = Path(__file__).resolve().parent
    repository = directory.parents[1]
    summary = json.loads((directory / "development_multiseed/summary.json").read_text())
    metric_errors = []
    default_compatibility = []
    groups = {}
    checks = {}
    disk_replay_checks = []
    for recorded_path, expected in summary["source_sha256"].items():
        path = Path(recorded_path)
        # The evaluator records its absolute __file__; permit another clone.
        local_path = repository / "scripts" / path.name if path.is_absolute() else repository / path
        checks[str(local_path.relative_to(repository))] = (
            hashlib.sha256(local_path.read_bytes()).hexdigest() == expected
        )
    for case in summary["cases"]:
        name = f"{case['mode']}_{case['terrain']}_{case['noise']}_delay{case['delay_steps']}_seed{case['seed']}"
        rows = list(csv.DictReader((directory / "development_multiseed" / f"{name}.csv").open()))
        pairs = {
            "velocity_estimation_rmse_mps": (
                "estimate_velocity_body_x_mps",
                "truth_velocity_body_x_mps",
            ),
            "clearance_estimation_rmse_m": ("estimate_clearance_m", "truth_clearance_m"),
            "ground_pitch_rmse_rad": ("estimate_ground_pitch_rad", "truth_ground_pitch_rad"),
            "tracking_rmse_mps": ("truth_velocity_body_x_mps", "command_velocity_mps"),
        }
        for metric, (estimate, truth) in pairs.items():
            value = math.sqrt(
                sum((float(row[estimate]) - float(row[truth])) ** 2 for row in rows) / len(rows)
            )
            metric_errors.append(abs(value - case[metric]))
        pitch = math.sqrt(sum(float(row["pitch_error_rad"]) ** 2 for row in rows) / len(rows))
        metric_errors.append(abs(pitch - case["pitch_rmse_rad"]))
        assert len(rows) == case["steps"]
        assert all(
            abs(
                float(row["time_s"])
                - float(row["measurement_time_s"])
                - float(row["measurement_age_s"])
            )
            < 1e-10
            for row in rows
        )
        if case["mode"] == "closed_loop" and case["noise"] == "bias" and case["seed"] == 17:
            with np.load(
                directory / "development_multiseed" / f"{name}_measurements.npz"
            ) as archive:
                assert set(archive.files) == {
                    "sequence",
                    "time_s",
                    "gyro_rad_s",
                    "specific_force_m_s2",
                    "joint_position_rad",
                    "joint_velocity_rad_s",
                    "wheel_contact",
                }
                packets = [
                    D1SensorMeasurements(
                        **{
                            key: int(archive[key][index])
                            if key == "sequence"
                            else archive[key][index]
                            for key in archive.files
                        }
                    )
                    for index in range(len(archive["sequence"]))
                ]
            observer = D1ProprioceptiveEstimator()
            state = observer.reset(
                packets[0], initial_position=tuple(summary["initial_position_prior_m"])
            )
            for index, row in enumerate(rows, start=1):
                packet = packets[max(0, index - case["delay_steps"])]
                if index > case["delay_steps"]:
                    state = observer.update(packet)
                published = replace(
                    state,
                    sequence=index,
                    control_time_s=packets[index].time_s,
                    measurement_time_s=packet.time_s,
                )
                actual = [
                    published.base_position[0],
                    published.base_position[2],
                    published.base_linear_velocity_body[0],
                    published.base_linear_velocity_world[2],
                    published.base_rpy[1],
                    published.measurement_time_s,
                    published.age_s,
                ]
                expected = [
                    float(row[key])
                    for key in (
                        "estimate_x_m",
                        "estimate_height_m",
                        "estimate_velocity_body_x_mps",
                        "estimate_velocity_world_z_mps",
                        "estimate_pitch_rad",
                        "measurement_time_s",
                        "measurement_age_s",
                    )
                ]
                assert np.array_equal(actual, expected)
            disk_replay_checks.append(name)
        if case["seed"] == 17 and case["delay_steps"] == 0:
            old_name = f"{case['mode']}_{case['terrain']}_{case['noise']}_seed17.csv"
            old_rows = list(csv.DictReader((directory / "development_v1" / old_name).open()))
            default_compatibility.append(
                len(old_rows) == len(rows)
                and all(
                    all(float(row[key]) == float(old[key]) for key in old)
                    for row, old in zip(rows, old_rows, strict=True)
                )
            )
        group_name = f"{case['mode']}/{case['terrain']}/delay{case['delay_steps']}"
        groups.setdefault(group_name, []).append(case)
    group_statistics = {}
    for name, cases in groups.items():
        group_statistics[name] = {
            "count": len(cases),
            "completed": sum(case["termination_reason"] == "time_limit" for case in cases),
            "minimum_duration_s": min(case["duration_s"] for case in cases),
            **{
                metric: sum(case[metric] for case in cases) / len(cases)
                for metric in (
                    "velocity_estimation_rmse_mps",
                    "pitch_rmse_rad",
                    "ground_pitch_rmse_rad",
                    "clearance_estimation_rmse_m",
                    "tracking_rmse_mps",
                )
            },
        }
    audit = {
        "cases": len(summary["cases"]),
        "control_steps": sum(case["steps"] for case in summary["cases"]),
        "bitwise_replay_cases": sum(case["replay_bitwise_equal"] for case in summary["cases"]),
        "max_summary_recomputation_error": max(metric_errors),
        "source_hash_matches": checks,
        "delay_zero_matches_pre_delay_csv_cases": sum(default_compatibility),
        "delay_zero_compatibility_comparisons": len(default_compatibility),
        "disk_measurement_replay_matches_csv_cases": disk_replay_checks,
        "groups": group_statistics,
    }
    assert all(checks.values())
    assert all(default_compatibility)
    assert max(metric_errors) < 1e-12
    (directory / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
