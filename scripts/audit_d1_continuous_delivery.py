"""Read-only artifact checks extending the existing continuous-policy analysis.

No MuJoCo environment or trained policy is instantiated. Outputs are new files;
the experiment, original analysis, and video sidecars are never rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def check_manifest(directory):
    inventory = read_json(directory / "manifest.json")["sha256"]
    for relative, expected in inventory.items():
        assert sha(directory / relative) == expected, relative
    return len(inventory)


def check_sources(run):
    source = read_json(run / "source.json")
    assert sha(run / "source.tar.gz") == source["archive_sha256"]
    with tarfile.open(run / "source.tar.gz", "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        assert set(members) == set(source["sha256"])
        for name, expected in source["sha256"].items():
            assert hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest() == expected
    differences = [
        name for name, expected in source["sha256"].items() if sha(ROOT / name) != expected
    ]
    assert differences == ["scripts/render_d1_continuous_task.py"], differences
    return {"archived_files": len(members), "current_source_changes": differences}


def check_training(run):
    reports = []
    for seed in (19000, 20000, 21000):
        directory = run / f"training_seed{seed}"
        with np.load(directory / "training_samples.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        assert arrays["raw_gaussian"].shape == (32768, 4, 2)
        for array in arrays.values():
            assert np.isfinite(array).all()
        np.testing.assert_array_equal(
            arrays["clipped_action"], np.clip(arrays["raw_gaussian"], -1, 1)
        )
        scale_error = float(np.max(np.abs(arrays["scaled_reward"] - 0.01 * arrays["raw_reward"])))
        assert scale_error < 2e-9
        monitor_error, episodes, completed_samples = 0.0, 0, 0
        for worker in range(4):
            cursor = 0
            with (directory / f"worker_{worker}.monitor.csv").open() as stream:
                next(stream)
                for row in csv.DictReader(stream):
                    length = int(row["l"])
                    raw_return = float(
                        np.sum(arrays["raw_reward"][cursor : cursor + length, worker])
                    )
                    monitor_error = max(monitor_error, abs(raw_return - float(row["r"])))
                    cursor += length
                    episodes += 1
            assert cursor <= 32768
            completed_samples += cursor
        # SB3 Monitor serializes the episode return rounded to six decimals.
        assert monitor_error <= 5.1e-7
        reports.append(
            {
                "seed": seed,
                "physical_samples": 131072,
                "completed_episodes": episodes,
                "completed_episode_samples": completed_samples,
                "unfinished_episode_samples": 131072 - completed_samples,
                "maximum_reward_scale_error": scale_error,
                "maximum_monitor_return_rounding_error": monitor_error,
                "actual_sample_clip_fraction": (np.abs(arrays["raw_gaussian"]) > 1)
                .mean(axis=(0, 1))
                .tolist(),
            }
        )
    return reports


def check_telemetry(run):
    count, total_steps, maximum_error = 0, 0, 0.0
    for directory in sorted((run / "rollouts").iterdir()):
        if directory.name == "models":
            continue
        with (directory / "telemetry.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        stored = read_json(directory / "summary.json")
        for row in rows:
            f = lambda key, row=row: float(row[key])
            calculated = {
                "velocity_error_mps": f("velocity_mps") - f("command_velocity_mps"),
                "yaw_rate_error_rps": f("yaw_rate_rps") - f("command_yaw_rate_rps"),
                "clearance_m": f("z_m") - f("ground_height_m"),
                "clearance_error_m": f("z_m") - f("ground_height_m") - 0.455,
                "residual_longitudinal_n": 11.25 * f("action_longitudinal"),
                "residual_vertical_n": 20 * f("action_vertical"),
            }
            for key, expected in calculated.items():
                maximum_error = max(maximum_error, abs(expected - f(key)))
            assert abs(f("action_longitudinal")) <= 1 and abs(f("action_vertical")) <= 1
            assert row["ground_reference_mode"] == "estimated"
        assert maximum_error < 1e-12
        assert sum(int(row["truncated"]) for row in rows) == 1
        assert sum(int(row["terminated"]) for row in rows) == 0
        for stage in stored["stage_summaries"]:
            selected = [row for row in rows if row["stage"] == stage["stage"]]
            assert len(selected) == stage["steps"]
            assert sum(float(row["stage_time_s"]) >= 1 for row in selected) == stage["steady_steps"]
            turn = float(selected[-1]["yaw_rad"]) - float(selected[0]["yaw_rad"])
            command = float(selected[-1]["reference_yaw_rad"]) - float(
                selected[0]["reference_yaw_rad"]
            )
            assert turn == stage["yaw_change_rad"]
            assert command == stage["commanded_yaw_change_rad"]
            fraction = turn / command if abs(command) > 0.1 else None
            assert fraction == stage["turn_progress_fraction"]
        count += 1
        total_steps += len(rows)
    return {
        "rollouts": count,
        "control_steps": total_steps,
        "maximum_primitive_error": maximum_error,
    }


def check_replay(delivery):
    replay = delivery / "replay_sensor45_cli"
    checks = check_manifest(replay)
    for name in ("dev_train19000_step131072_seed17_45s", "zero_seed17_45s"):
        original = delivery / "sensor45_v1/rollouts" / name
        assert sha(replay / name / "telemetry.csv") == sha(original / "telemetry.csv")
        with np.load(replay / name / "states.npz") as a, np.load(original / "states.npz") as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    return {
        "replay_artifact_sha_checks": checks,
        "byte_identical_recordings": 2,
        "new_physics_run": False,
    }


def check_videos(delivery):
    reports = []
    for stem, name in (
        ("seed19000_sensor45_45s", "dev_train19000_step131072_seed17_45s"),
        ("zero_sensor45_45s", "zero_seed17_45s"),
    ):
        sidecar = read_json(delivery / "videos" / f"{stem}.json")
        video = delivery / "videos" / f"{stem}.mp4"
        assert sha(video) == sidecar["video_sha256"]
        assert sha(ROOT / "scripts/render_d1_continuous_task.py") == sidecar["renderer_sha256"]
        recording = delivery / "sensor45_v1/rollouts" / name
        assert sha(recording / "manifest.json") == sidecar["input_manifest_sha256"]
        for relative, expected in sidecar["input_sha256"].items():
            assert sha(recording / relative) == expected
        assert sidecar["frame_state_indices"] == list(range(0, 4501, 5))
        metadata = read_json(recording / "protocol.json")
        assert sidecar["checkpoint_sha256"] == metadata["checkpoint_sha256"]
        assert sidecar["observation_schema"] == "d1-continuous-sensor-command45-v1"
        decode = subprocess.run(
            [
                "rtk",
                "proxy",
                "ffmpeg",
                "-hide_banner",
                "-threads",
                "1",
                "-i",
                str(video),
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "null",
                "-",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        progress = dict(line.split("=", 1) for line in decode.stdout.splitlines() if "=" in line)
        assert int(progress["frame"]) == 901 and progress["progress"] == "end"
        assert int(progress["out_time_us"]) == 45050000
        assert "960x540" in decode.stderr and "20 fps" in decode.stderr
        assert "Error" not in decode.stderr
        reports.append(
            {
                "file": video.name,
                "sha256": sha(video),
                "decoded_frames": 901,
                "duration_s": 45.05,
                "decode_exit_code": decode.returncode,
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=ROOT / "results/d1_continuous_policy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.delivery / "sensor45_v1"
    report = {
        "schema": "d1-continuous-delivery-supplement-v1",
        "auditor_sha256": sha(Path(__file__)),
        "original_manifest_sha256": sha(run / "manifest.json"),
        "experiment_artifact_sha_checks": check_manifest(run),
        "analysis_artifact_sha_checks": check_manifest(args.delivery / "analysis_sensor45_v1"),
        "sources": check_sources(run),
        "training": check_training(run),
        "telemetry_primitive_and_stage_checks": check_telemetry(run),
        "replay": check_replay(args.delivery),
        "videos": check_videos(args.delivery),
        "limits": [
            "This supplement checks archived artifacts; it does not independently reproduce training.",
            "The separate analyzer recomputes RMSE/reward/quality from CSV; this supplement checks primitive error fields and stored stage turn numbers.",
            "Raw training means/std/log_prob were not recorded in this experiment; raw likelihood ratios cannot be reconstructed from training_samples.npz.",
            "Video decode and input provenance are machine-checked. Visual inspection is separately recorded in the delivery note.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
