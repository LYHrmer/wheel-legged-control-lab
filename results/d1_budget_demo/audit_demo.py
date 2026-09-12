"""Audit matching inputs and actual command response; decode media completely."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/lyh/wheel-legged-control-lab")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def columns(records, names):
    return np.asarray([[float(row[name]) for name in names] for row in records])


def decode(path):
    command = ["rtk", "proxy", "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-threads", "1", "-i", str(path), "-map", "0:v:0", "-f", "null", "-",
        "-progress", "pipe:1", "-nostats"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    progress = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert progress["progress"] == "end", progress
    return {"command": command, "decoded_frames": int(progress["frame"]),
        "duration_us": int(progress["out_time_us"]), "sha256": sha(path),
        "stderr": result.stderr}


def pair():
    first, second = (read(HERE / name / "protocol.json") for name in ("zero", "ppo"))
    reference = read(ROOT / "results/d1_budget_study/zero_development/road0_seed1017_episode.json")
    keys = ("terrain", "provider", "controller_parameters", "actuator", "domain",
        "command_seed", "measurement_seed", "duration_s", "action_mode", "source_schema")
    source_checks = {key: first["episode"][key] == reference[key] for key in keys}
    assert all(source_checks.values()), source_checks
    assert first["episode"] == second["episode"]
    assert first["compiled_model_sha256"] == second["compiled_model_sha256"]
    commands = ("decision_time_s", "command_vx_mps", "command_yaw_rps", "command_clearance_m")
    zrows, prows = (rows(HERE / name / "telemetry.csv") for name in ("zero", "ppo"))
    assert np.array_equal(columns(zrows, commands), columns(prows, commands))
    with np.load(HERE / "zero/states.npz") as zstates, np.load(HERE / "ppo/states.npz") as pstates:
        assert np.array_equal(zstates["time_s"], pstates["time_s"])
        assert np.array_equal(zstates["qpos"][0], pstates["qpos"][0])
        assert np.array_equal(zstates["qvel"][0], pstates["qvel"][0])
    assert second["model_sha256"] == read(HERE / "task.json")["policy_sha256"]
    summaries = {name: read(HERE / name / "summary.json") for name in ("zero", "ppo")}
    assert all(s["source_unchanged"] and s["completed"] and s["steps"] == 6000
        for s in summaries.values())
    return {"formal_development_case_matches": source_checks,
        "all_episode_metadata_equal": True, "compiled_model_equal": True,
        "initial_qpos_qvel_equal": True, "all_6000_executed_commands_equal": True,
        "state_timestamps_equal": True, "predetermined_policy_sha256_matches": True,
        "summaries": summaries, "claim_limit": "One fixed development case, not aggregate holdout performance"}


def gui():
    directory = HERE / "gui_retry"
    events = read(directory / "events.json")
    summary = read(directory / "rollout/summary.json")
    assert summary["stop_reason"] == "keyboard_escape", summary
    assert summary["source_unchanged"]
    command_rows = columns(rows(directory / "rollout/telemetry.csv"),
        ("command_vx_mps", "command_yaw_rps", "command_clearance_m"))
    unique = [command_rows[0].tolist()]
    for row in command_rows[1:]:
        if not np.array_equal(row, unique[-1]):
            unique.append(row.tolist())
    expected = [[0, 0, .455], [.05, 0, .455], [.1, 0, .455], [.1, .05, .455],
        [.1, .05, .46], [0, 0, .46], [0, 0, .465], [0, 0, .46],
        [.05, 0, .46], [.05, -.05, .46], [0, 0, .46], [-.05, 0, .46],
        [-.05, .05, .46], [0, 0, .46], [0, 0, .465], [0, 0, .46]]
    assert np.array_equal(unique, expected), {"observed": unique, "expected": expected}
    assert all(item["target_pid"] == events["window_pid"] and
        item["target_window"] == events["window_id"] for item in events["events"])
    return {"kind": "automated real X11 GUI integration; not human manual acceptance",
        "all_events_target_owned_window": True, "observed_command_changes": unique,
        "checks": {"forward_increment": True, "reverse": True, "left_right_yaw": True,
            "raise_lower_clearance": True, "space_clears_motion": True,
            "motion_expires_without_input": True, "height_key_does_not_revive_motion": True,
            "escape_ends_rollout": True}, "summary": summary,
        "window_video": decode(directory / "window.mp4")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui-only", action="store_true")
    args = parser.parse_args()
    report = {} if args.gui_only else {"pair": pair()}
    if (HERE / "gui_retry/events.json").exists():
        report["gui"] = gui()
    if not args.gui_only:
        report["media"] = {str(path.relative_to(HERE)): decode(path)
            for path in sorted(HERE.glob("*.mp4"))}
    output = HERE / ("gui_audit.json" if args.gui_only else
        "audit.json" if "gui" in report else "pair_audit.json")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))
