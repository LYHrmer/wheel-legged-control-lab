"""Recompute saved probe metrics without importing the controller or runner."""

import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(values):
    return math.sqrt(math.fsum(v * v for v in values) / len(values))


def audit():
    count = steps = artifact_count = source_count = 0
    max_metric_error = 0.0
    directories = []
    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir() or not (directory / "protocol.json").is_file():
            continue
        protocol = json.loads((directory / "protocol.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["source_unchanged"]
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, expected in manifest.items():
            path = directory / name
            assert path.resolve().is_relative_to(directory.resolve())
            assert sha(path) == expected, path
            artifact_count += 1
        with tarfile.open(directory / "source.tar.gz") as archive:
            for name, expected in protocol["source_sha256"].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
                source_count += 1
        metrics = {item["case"]: item for item in summary["cases"]}
        traces = {}
        for case in protocol["cases"]:
            rows = list(csv.DictReader((directory / f"{case}.csv").open()))
            traces[case] = rows
            count += 1
            steps += len(rows)
            for index, row in enumerate(rows):
                time = (index + 1) * 0.01
                assert abs(float(row["time_s"]) - time) < 1e-10
                assert int(row["step"]) == index + 1
                assert abs(float(row["height_error_m"]) - (float(row["height_m"]) - 0.455)) < 1e-12
                if "measurement_time_s" in row:
                    delay = protocol.get("sensor_delay_steps", 0) * 0.01
                    assert abs(float(row["measurement_time_s"]) - max(0, time - delay)) < 1e-10
                    assert abs(float(row["target_time_s"]) - index * 0.01) < 1e-10
                for joint in range(16):
                    limit = 12 if joint % 4 == 3 else 80
                    assert abs(float(row[f"torque_{joint}_nm"])) <= limit + 1e-12
                if index < len(rows) - 1:
                    assert not int(row["fallen"])
            tail = rows[-100:]
            recomputed = {
                "tail_height_rmse_m": rms([float(r["height_m"]) - 0.455 for r in tail]),
                "tail_vx_rmse_mps": rms(
                    [float(r["vx_mps"]) - float(r["command_vx_mps"]) for r in tail]
                ),
                "tail_yaw_rmse_rps": rms(
                    [float(r["yaw_rate_rps"]) - float(r["command_yaw_rps"]) for r in tail]
                ),
                "yaw_change_rad": float(rows[-1]["yaw_rad"]) - float(rows[0]["yaw_rad"]),
            }
            for name, value in recomputed.items():
                error = abs(value - metrics[case][name])
                assert error < 1e-12, (directory, case, name, error)
                max_metric_error = max(max_metric_error, error)
            expected_steps = 800 if case in ("start_stop", "turn_left", "turn_right") else 400
            completed = len(rows) == expected_steps and not int(rows[-1]["fallen"])
            assert completed == metrics[case]["completed"]
            if case.startswith("turn_") and "turn_70_quality_pass" in metrics[case]:
                command_integral = math.fsum(float(r["command_yaw_rps"]) for r in rows) * 0.01
                progress = recomputed["yaw_change_rad"] / command_integral
                assert abs(progress - metrics[case]["turn_progress_fraction"]) < 1e-12
                passed = completed and progress >= 0.7 and recomputed["tail_yaw_rmse_rps"] <= 0.05
                assert passed == metrics[case]["turn_70_quality_pass"]
        for check in summary.get("action_response_checks", []):
            case = check["case"]
            _, axis, sign = case.split("_")
            axis = int(axis)
            key = f"wheel_z_body_{axis}_m" if axis < 4 else f"wheel_qd_{axis - 4}_rad_s"
            mean = lambda rows, key=key: (
                math.fsum(float(r[key]) for r in rows[-100:]) / min(100, len(rows))
            )
            delta = mean(traces[case]) - mean(traces["stand"])
            assert abs(delta - check["delta"]) < 1e-12
            direction = 1 if sign == "plus" else -1
            signed = -direction * delta if axis < 4 else direction * delta
            assert (
                signed >= check["minimum_signed_response"] and metrics[case]["completed"]
            ) == check["passed"]
        directories.append(
            {
                "directory": directory.name,
                "cases": len(metrics),
                "manifest_sha256": sha(directory / "manifest.json"),
            }
        )
    result = {
        "experiment_directories": directories,
        "episodes_checked": count,
        "control_steps_checked": steps,
        "artifact_hashes_checked": artifact_count,
        "archived_source_hashes_checked": source_count,
        "max_rmse_or_angle_difference": max_metric_error,
        "auditor_sha256": sha(Path(__file__)),
    }
    (ROOT / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
