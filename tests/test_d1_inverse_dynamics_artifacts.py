"""Regression checks at the agreed artifact-reader seams, without rendering."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
read_plot = runpy.run_path(str(SCRIPTS / "plot_d1_inverse_dynamics.py"))["_read_rollout"]
read_render = runpy.run_path(str(SCRIPTS / "render_d1_inverse_dynamics_mobility.py"))["_read"]
summarize_timing = runpy.run_path(str(SCRIPTS / "benchmark_d1_inverse_dynamics.py"))["_summary"]


def test_latency_summary_keeps_the_worst_call_including_tail_outliers() -> None:
    timings = [1.] * 999 + [25.]
    summary = summarize_timing([{
        "compute_ms": timings, "accepted": 1000, "rejected": 0, "all_phase_rejections": 0,
    }])
    assert summary["p99_ms"] == 1.
    assert summary["max_ms"] == 25.
    assert summary["p99_9_ms"] == float(np.percentile(timings, 99.9))
    assert summary["over_10ms_fraction"] == .001


def _write(directory: Path, summary: dict, rows: list[dict]) -> None:
    path = directory / "steps.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["artifacts"]["steps"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def recording(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    rows, episodes = [], []
    for scenario in ("stand", "drive_brake", "push"):
        for seed in (21, 22, 23):
            episodes.append({"scenario": scenario, "seed": seed, "completed": True,
                             "validation_passed": True, "applied_steps": 600,
                             "attempted_steps": 600, "elapsed_s": 6.0})
            rows.extend({"scenario": scenario, "seed": seed, "step": step,
                         "applied": True, "status": "solved", "output_time_s": (step + 1) / 100}
                        for step in range(600))
    summary = {"protocol_complete": True, "validation_passed": True,
               "validation_status": "passed", "episodes": episodes,
               "artifacts": {"steps": {"filename": "steps.csv.gz"}}}
    _write(tmp_path, summary, rows)
    return tmp_path, summary, rows


def test_matching_complete_nine_episode_recording_can_be_labelled_passed(recording) -> None:
    directory, _, _ = recording
    _, drive, complete = read_plot(directory)

    assert complete is True
    assert set(drive) == {21, 22, 23}
    assert [len(drive[seed]) for seed in (21, 22, 23)] == [600, 600, 600]
    assert drive[22][-1]["output_time_s"] == "6.0"


def test_complete_recording_without_hash_cannot_be_labelled_passed(recording) -> None:
    directory, summary, _ = recording
    assert read_plot(directory)[2] is True
    del summary["artifacts"]["steps"]["sha256"]
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    _, drive, complete = read_plot(directory)

    assert complete is False
    assert len(drive[22]) == 600  # Still useful diagnostically, never certified.


def test_truncated_episode_cannot_pass_even_when_the_recomputed_hash_matches(recording) -> None:
    directory, summary, rows = recording
    assert read_plot(directory)[2] is True
    rows[:] = [row for row in rows if not (
        row["scenario"] == "drive_brake" and row["seed"] == 22 and row["step"] == 599
    )]
    _write(directory, summary, rows)

    _, drive, complete = read_plot(directory)

    assert complete is False
    assert len(drive[22]) == 599


@pytest.mark.parametrize("failure", ("overall_gate", "overall_status", "episode_gate"))
def test_failed_summary_cannot_be_hidden_by_complete_successful_csv_rows(recording, failure) -> None:
    directory, summary, _ = recording
    assert read_plot(directory)[2] is True
    if failure == "overall_gate":
        summary["validation_passed"] = False
    elif failure == "overall_status":
        summary["validation_status"] = "failed"
    else:
        summary["episodes"][0]["validation_passed"] = False
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    _, drive, complete = read_plot(directory)

    assert complete is False
    assert len(drive[22]) == 600


@pytest.mark.parametrize("reader,arguments", (
    pytest.param(read_plot, (), id="plot"),
    pytest.param(read_render, ("drive_brake", 22), id="renderer"),
))
def test_wrong_csv_hash_is_rejected_before_accepting_the_recording(recording, reader, arguments):
    directory, summary, _ = recording
    summary["artifacts"]["steps"]["sha256"] = "0" * 64
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        reader(directory, *arguments)


def test_renderer_selects_the_complete_requested_episode_without_other_seeds(recording) -> None:
    directory, _, _ = recording

    _, episode, rows, hashes = read_render(directory, "drive_brake", 22)

    assert episode["scenario"] == "drive_brake" and episode["seed"] == 22
    assert len(rows) == 600
    assert {(row["scenario"], row["seed"]) for row in rows} == {("drive_brake", "22")}
    assert rows[0]["step"] == "0" and rows[-1]["step"] == "599"
    assert set(hashes) == {"summary_sha256", "steps_sha256"}


def test_renderer_rejects_a_step_gap_even_when_count_and_hash_still_match(recording) -> None:
    directory, summary, rows = recording
    assert len(read_render(directory, "drive_brake", 22)[2]) == 600
    changed = next(row for row in rows if (
        row["scenario"] == "drive_brake" and row["seed"] == 22 and row["step"] == 300
    ))
    changed["step"] = 301  # Leave 600 rows, but duplicate 301 and omit 300.
    _write(directory, summary, rows)

    with pytest.raises(ValueError, match="contiguous"):
        read_render(directory, "drive_brake", 22)
