"""Failure evidence without integrating a MuJoCo model."""
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import d1_probe_archive as archive


@pytest.mark.parametrize("nonfinite", [False, True])
def test_partial_native_failure_keeps_state_wrench_time_and_restores_bindings(tmp_path, monkeypatch, nonfinite):
    data = SimpleNamespace(time=0., qpos=np.zeros(7), qvel=np.zeros(6),
                           ctrl=np.arange(16.), xfrc_applied=np.zeros((2, 6)))
    plant = SimpleNamespace(model=object(), data=data, base_body_id=1, step=lambda *a, **k: None)
    output = tmp_path / "partial"
    states = archive.ProbeStateArchive(plant, output)
    initial_env, initial_observer = archive.g1.D1HeadingTrackingEnv, archive.g1.NativeWrenchObserver

    def fake_native(_model, state):
        state.time += .002
        state.qpos[0] += 1.
        if state.qpos[0] == 2.:
            if nonfinite:
                state.qvel[0] = np.nan
                state.ctrl[0] = np.inf
            raise RuntimeError("synthetic failure after partial interval")

    monkeypatch.setattr(archive.mujoco, "mj_step", fake_native)

    def fake_episode(folder, case, _model, _gates):
        folder.mkdir()
        archive.g1.D1HeadingTrackingEnv()
        states.capture("reset")
        try:
            with archive.g1.NativeWrenchObserver(plant, case):
                archive.mujoco.mj_step(plant.model, plant.data)
                archive.mujoco.mj_step(plant.model, plant.data)
        finally:
            states.capture("failed_step")
            states.close()

    monkeypatch.setattr(archive.g1, "run_episode", fake_episode)
    with pytest.raises(RuntimeError, match="partial interval"):
        archive.run_archived_g1_episode(output, {}, "zero", {}, lambda: object())
    assert archive.g1.D1HeadingTrackingEnv is initial_env
    assert archive.g1.NativeWrenchObserver is initial_observer
    assert archive.mujoco.mj_step is fake_native
    with np.load(output / "execution_states.npz") as saved:
        assert saved["time_s"].tolist() == [0., .004, .004]
        assert saved["qpos"][:, 0].tolist() == [0., 2., 2.]
    receipt = json.loads((output / "native_entry_receipt.json").read_text())
    assert receipt["observed_native_calls"] == 2
    assert receipt["returned_native_calls"] == 1
    assert receipt["summed_actual_dt_s"] == .004
    assert receipt["archival_error"] is None
    with gzip.open(output / "native_physics_entries.jsonl.gz", "rt") as stream:
        entries = [json.loads(line) for line in stream]
    assert entries[-1]["returned"] is False
    assert entries[-1]["qpos_after_error"][0] == 2.
    assert entries[-1]["ctrl_nm"] == list(range(16))
    if nonfinite:
        assert entries[-1]["qvel_after_error"][0] == {"nonfinite_float": "NaN"}
        with np.load(output / "execution_states.npz") as saved:
            assert np.isnan(saved["qvel"][-1, 0])
        with np.load(output / "native_nonfinite_values.npz") as saved:
            assert np.isnan(saved["values"]).any()


def test_existing_output_is_not_modified(tmp_path):
    marker = tmp_path / "retained.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        archive.run_archived_g1_episode(tmp_path, {}, "zero", {}, lambda: object())
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == "preserve"
