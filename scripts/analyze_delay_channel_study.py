"""Reject incomplete matrices or inconsistent recorded delay clocks and queues."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def check_protocol(protocol):
    specifications = [("fusion_noise", m, a) for m, a in ((0, 0), (10, 0), (20, 0), (30, 0), (0, 10), (0, 20), (0, 30))]
    specifications += [("fusion_ideal", 0, 0), ("fusion_ideal", 30, 0), ("oracle", 0, 0), ("oracle", 0, 30), ("truth_delayed", 30, 0)]
    expected = [{"name": f"{source}_m{m}_a{a}", "source": source, "measurement_ms": m, "actuator_ms": a} for source, m, a in specifications]
    assert protocol["schema"] == "d1-delay-channel-diagnostic-v1"
    assert protocol["profiles"] == expected
    assert protocol["duration_s"] in (3.0, 60.0)
    assert protocol["seeds"] == ([17] if protocol["duration_s"] == 3 else [17, 29])
    assert protocol["schedule_duration_s"] == 60.0
    assert (protocol["terrain_suite"], protocol["split"], protocol["roads"]) == ("v1", "development", [0])
    assert protocol["controller"] == {"wheel_kp": 0.55, "wheel_ki": 1.5, "yaw_feedback_gain": 4.0, "leg_feedback_scale": 1.0, "attitude_feedback_scale": 0.25}
    assert protocol["zero_residual"] and protocol["workers"] == 1
    assert protocol["measurement_dt_s"] == 0.01 and protocol["physics_dt_s"] == 0.002
    assert protocol["actuator_gain"] == 1.0 and protocol["actuator_time_constant_s"] == 0.0


def analyze(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    check_protocol(protocol)
    manifest = json.loads((directory / "source_sha256.json").read_text())
    for relative, expected in manifest.items():
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts
        target = directory / "source" / relative
        assert not target.is_symlink() and target.resolve().is_relative_to((directory / "source").resolve())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, relative
    expected_cases = {f"{p['name']}_road0_seed{s}" for p in protocol["profiles"] for s in protocol["seeds"]}
    actual_cases = {p.parent.name for p in directory.glob("*/result.json")}
    assert actual_cases == expected_cases, {"missing": sorted(expected_cases - actual_cases), "extra": sorted(actual_cases - expected_cases)}
    cases, matrix = [], []
    for profile in protocol["profiles"]:
        group = []
        for seed in protocol["seeds"]:
            name = f"{profile['name']}_road0_seed{seed}"
            case_dir = directory / name
            result = json.loads((case_dir / "result.json").read_text())
            episode = json.loads((case_dir / "episode.json").read_text())
            assert result["profile"] == profile and result["episode_seed"] == seed, (name, "result identity")
            assert episode["profile"] == profile and episode["episode_seed"] == seed, (name, "episode identity")
            assert result["error"] is None, {"case": name, "error": result["error"]}
            assert episode["source_manifest"] == manifest
            assert episode["controller_parameters"] == protocol["controller"]
            assert episode["actuator"]["gain"] == 1.0 and episode["actuator"]["time_constant_s"] == 0.0
            actuator_steps = profile["actuator_ms"] // 2
            assert episode["actuator"]["delay_steps"] == actuator_steps
            with (case_dir / "metrics.csv").open(newline="") as stream:
                rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
            # Independently recompute with NumPy, without importing the runner's summary helper.
            field_map = {"velocity_rmse_mps": "velocity_error_mps", "yaw_rmse_rps": "yaw_rate_error_rps", "height_rmse_m": "height_error_m"}
            metrics = {key: float(np.sqrt(np.mean(np.square([r[field] for r in rows])))) for key, field in field_map.items()}
            metrics["attitude_rmse_rad"] = float(np.sqrt(np.mean([r["roll_error_rad"] ** 2 + r["pitch_error_rad"] ** 2 for r in rows])))
            assert result["executed_steps"] == len(rows)
            assert result["duration_s"] == rows[-1]["time_s"]
            assert result["requested_duration_s"] == protocol["duration_s"]
            complete = result["terminal_reason"] == "time_limit" and len(rows) == round(protocol["duration_s"] / 0.01)
            assert result["completed"] == complete
            assert (result["full_horizon_tracking"] is not None) == complete
            for key, value in metrics.items():
                np.testing.assert_allclose(result["partial_horizon_tracking"][key], value, rtol=0, atol=1e-12)
                if complete:
                    np.testing.assert_allclose(result["full_horizon_tracking"][key], value, rtol=0, atol=1e-12)
            data = np.load(case_dir / "trajectory.npz", allow_pickle=False)
            assert protocol["duration_s"] != 60.0 or "controller_trace" in data, (name, "formal controller trace missing")
            assert len(data["qpos"]) == len(rows) + 1
            assert len(data["estimated_states"]) == len(rows)
            traces = data["actuator_trace"]
            assert traces.shape == (len(rows), 5, 5, 16), traces.shape
            physical = traces.reshape(-1, 5, 16)
            limited, delayed, applied = physical[:, 1], physical[:, 2], physical[:, 3]
            expected_delayed = np.concatenate((np.zeros((actuator_steps, 16)), limited))[:len(limited)]
            assert np.array_equal(delayed, expected_delayed), (name, "actuator delay queue")
            assert np.array_equal(applied, delayed), (name, "pure delay unexpectedly changes torque")
            diagnostics = None
            if "controller_trace" in data:
                controller = data["controller_trace"]
                assert controller.shape == (len(rows), 8, 16)
                np.testing.assert_array_equal(np.repeat(controller[:, 1], 5, axis=0), physical[:, 0])
                limits = np.asarray(episode["actuator"]["torque_limit_nm"])
                saturated = np.clip(controller[:, 0], -limits, limits)
                np.testing.assert_array_equal(np.mean(controller[:, 0] != saturated, axis=1), [r["controller_torque_clipped_fraction"] for r in rows])
                np.testing.assert_array_equal(np.mean(controller[:, 0] != controller[:, 1], axis=1), [r["controller_torque_changed_fraction"] for r in rows])
                diagnostics = {key: float(np.mean([r[key] for r in rows])) for key in (
                    "controller_torque_clipped_fraction", "controller_torque_changed_fraction",
                    "controller_position_guard_fraction", "controller_velocity_guard_fraction",
                    "controller_joint_target_limited_fraction")}
            t = np.asarray([row["time_s"] for row in rows])
            np.testing.assert_allclose(t, np.arange(1, len(rows) + 1) * 0.01, atol=1e-9, rtol=0)
            mt = np.asarray([row["measurement_time_s"] for row in rows])
            np.testing.assert_allclose(t - mt, np.minimum(t, profile["measurement_ms"] / 1000), atol=1e-9, rtol=0)
            np.testing.assert_allclose([row["measurement_age_s"] for row in rows], t - mt, atol=1e-9, rtol=0)
            if profile["source"] in ("oracle", "truth_delayed"):
                assert max(row["measurement_aligned_position_error_norm_m"] for row in rows) < 1e-12
                assert max(row["measurement_aligned_velocity_error_norm_mps"] for row in rows) < 1e-12
            cases.append({"case": name, "measurement_seed": episode["measurement_seed"], "controller_diagnostics": diagnostics, **result})
            group.append(result)
        matrix.append({"profile": profile["name"], "completed": sum(r["completed"] for r in group),
                       "cases": len(group), "durations_s": [r["duration_s"] for r in group],
                       "terminal_reasons": [r["terminal_reason"] for r in group]})
    measurement_seeds = {c["episode_seed"]: c["measurement_seed"] for c in cases}
    assert all(c["measurement_seed"] == measurement_seeds[c["episode_seed"]] for c in cases)
    if (directory / "source_consistency.json").exists():
        assert json.loads((directory / "source_consistency.json").read_text())["consistent"]
    if protocol["duration_s"] == 60.0:
        finished = json.loads((directory / "finished.json").read_text())
        assert finished["complete"] and finished["case_count"] == len(expected_cases) == finished["expected_case_count"]
    paired = []
    for case in cases:
        reference = next(c for c in cases if c["episode_seed"] == case["episode_seed"] and c["profile"]["name"] == "fusion_noise_m0_a0")
        full = case["full_horizon_tracking"]
        ref_full = reference["full_horizon_tracking"]
        paired.append({"case": case["case"], "baseline": reference["case"],
                       "full_horizon_rmse_difference": {k: full[k] - ref_full[k] for k in full} if full and ref_full else None})
    return {"schema": "d1-delay-channel-diagnostic-analysis-v1", "case_count": len(cases),
            "verified_source_files": len(manifest), "matrix": matrix, "cases": cases,
            "paired_against_fusion_noise_zero": paired,
            "limits": protocol["caveats"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k in ("case_count", "verified_source_files", "matrix")}), flush=True)


if __name__ == "__main__":
    main()
