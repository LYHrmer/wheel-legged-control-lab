"""Actual recorded files must pass provenance/clock checks before any renderer."""

import hashlib
import json
import shutil

import numpy as np
import pytest

from scripts.render_d1_locomotion import frame_indices, read_recording, render
from scripts.run_d1_locomotion import run
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    path = tmp_path_factory.mktemp("recorded") / "run"
    run(D1LocomotionEnv(episode_seconds=0.03), path, seed=17)
    return path


def update_record(path, name, value):
    (path / name).write_text(json.dumps(value))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_real_archive_has_exact_model_clock_and_pose(recorded):
    protocol, summary, states, rows, manifest = read_recording(recorded)
    assert summary["steps"] == 3 and summary["completed"]
    assert len(rows) + 1 == len(states["qpos"]) == 4
    assert protocol["compiled_model_sha256"] == manifest["model.mjb"]
    assert states["actuator_applied_nm"].shape == (3, 5, 16)


@pytest.mark.parametrize(
    "n,fps,expected",
    [(300, 20, [*range(0, 301, 5)]), (307, 20, [*range(0, 306, 5), 307]), (3, 20, [0, 3])],
)
def test_initial_and_non_grid_terminal_frames_are_kept(n, fps, expected):
    assert frame_indices(n, 0.01, fps) == expected
    assert 0 < len(expected) / fps - n * 0.01 < 2 / fps + 1e-12


@pytest.mark.parametrize("fps", [0, -1, True, 30, 101])
def test_invalid_fps_rejected(fps):
    with pytest.raises(ValueError):
        frame_indices(300, 0.01, fps)


@pytest.mark.parametrize(
    "name,value", [("source_unchanged", False), ("duration_s", 0.02), ("steps", 0)]
)
def test_summary_tampering_rejected_even_with_updated_manifest(recorded, tmp_path, name, value):
    path = tmp_path / "copy"
    shutil.copytree(recorded, path)
    summary = json.loads((path / "summary.json").read_text())
    summary[name] = value
    update_record(path, "summary.json", summary)
    with pytest.raises(ValueError):
        read_recording(path)


def test_partial_record_does_not_have_to_reach_requested_duration(recorded, tmp_path):
    path = tmp_path / "partial"
    shutil.copytree(recorded, path)
    protocol = json.loads((path / "protocol.json").read_text())
    protocol["episode"]["duration_s"] = 60.0
    update_record(path, "protocol.json", protocol)
    summary = json.loads((path / "summary.json").read_text())
    summary.update(completed=False, stop_reason="viewer_closed")
    update_record(path, "summary.json", summary)
    assert not read_recording(path)[1]["completed"]


def test_policy_label_cannot_claim_ppo_without_model_hash(recorded, tmp_path):
    path = tmp_path / "false_policy"
    shutil.copytree(recorded, path)
    protocol = json.loads((path / "protocol.json").read_text())
    protocol["policy"] = "trusted_local_ppo"
    update_record(path, "protocol.json", protocol)
    with pytest.raises(ValueError, match="policy/hash"):
        read_recording(path)


def test_bad_source_hash_and_path_cannot_be_rendered(recorded, tmp_path):
    path = tmp_path / "bad_hash"
    shutil.copytree(recorded, path)
    (path / "telemetry.csv").write_text("corrupted\n")
    with pytest.raises(ValueError, match="hash"):
        read_recording(path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["../elsewhere"] = "0" * 64
    (path / "manifest.json").write_text(json.dumps(manifest))
    # Existing corrupted file may reject before the newly added unsafe name.
    with pytest.raises(ValueError):
        read_recording(path)


def test_existing_video_is_not_overwritten_before_input_access(tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="new"):
        render(tmp_path / "missing", output)
    assert output.read_bytes() == b"keep"


def test_nonfinite_state_rejected_after_hash_recomputed(recorded, tmp_path):
    path = tmp_path / "nan"
    shutil.copytree(recorded, path)
    with np.load(path / "states.npz") as file:
        state = {key: file[key].copy() for key in file.files}
    state["qpos"][1, 0] = np.nan
    np.savez_compressed(path / "states.npz", **state)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["states.npz"] = hashlib.sha256((path / "states.npz").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="qpos"):
        read_recording(path)
