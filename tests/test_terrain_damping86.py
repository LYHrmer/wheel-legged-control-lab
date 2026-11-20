import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest


def _tiny_recording(path, *, no_signal=False):
    seeds = (8561, 8573, 8587)

    def address(start):
        values = np.arange(start, start + 16, dtype="<i4")
        return {"dtype": "<i4", "shape": [16], "hex": values.tobytes().hex()}

    def row(t, seed):
        # The wheel channels carry large unrelated signals, and must be excluded.
        q = np.full(16, seed / 100.0)
        v = np.tile((t, 2 * t, -3 * t, 100 * t), 4).astype(float)
        if no_signal and seed == seeds[-1]:
            v *= -1
        return {"step": t, "time_s": t * .01, "qpos": [0.] * 7 + q.tolist(),
                "qvel": [0.] * 6 + v.tolist(), "applied_torque_nm": [float(t)] * 16}

    cp = [{"source": {"fields": {"estimator": {"fields": {
        "_qadr": address(7), "_dadr": address(6),
    }}}}}]
    with ZipFile(path, "w") as z:
        for seed in seeds:
            for terrain in ("up", "down"):
                start = f"trials/action/{terrain}_{seed}/"
                z.writestr(start + "checkpoint_python.json", json.dumps(cp))
                for arm in ("prefix", "negative", "zero", "positive"):
                    times = range(1, 6) if arm == "prefix" else range(6, 9)
                    z.writestr(start + arm + ".jsonl", "".join(
                        json.dumps(row(t, seed)) + "\n" for t in times))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_leave_seed_out_and_deduplicate_shared_prefix_and_branches(tmp_path, monkeypatch):
    from scripts import terrain_damping86 as damping

    path = tmp_path / "recordings.zip"
    monkeypatch.setattr(damping, "ARCHIVE_SHA256", _tiny_recording(path))
    report, model = damping.fit_predictor(path)
    assert report["passed"] and model is not None
    assert len(report["folds"]) == 3
    # Three prefix windows, three per branch, both terrains. Six are unique.
    assert all(s["raw"] == 24 and s["unique"] == 6 for s in report["samples_by_seed"])
    for fold in report["folds"]:
        assert fold["validation_unique"] == 6 and fold["train_unique"] == 12
        assert fold["hold_rmse_rad_s"] == pytest.approx((56 / 3) ** .5)
        assert fold["predictor_rmse_rad_s"] < 1e-12
        assert fold["passed"]
    assert json.loads(json.dumps(model)) == model


@dataclass(frozen=True)
class SensorPacket:
    control_time_s: float
    measurement_time_s: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    pitch: float = .18


class TestLowLevel:
    requested_transfer_n = 0.

    def compute(self, command, state, **kw):
        return state


def test_live_uses_two_actual_past_intervals_and_only_edits_leg_velocities():
    from scripts.terrain_damping86 import attach_damping, record_applied

    # Estimate delta from prior and latest torques, respectively one and twice.
    weights = np.zeros((4, 12, 3))
    weights[:, 6:9] = np.eye(3)
    weights[:, 9:12] = 2 * np.eye(3)
    model = {"schema": "d1-leg-damping-ridge86-v1", "dt_s": .01,
             "mean": np.zeros((4, 12)).tolist(), "scale": np.ones((4, 12)).tolist(),
             "weights": weights.tolist(), "delta_intercept": np.zeros((4, 3)).tolist(),
             "velocity_min": np.full((4, 3), -8).tolist(), "velocity_max": np.full((4, 3), 8).tolist()}
    original = TestLowLevel()
    ep = SimpleNamespace(controller=SimpleNamespace(low_level=original))
    attach_damping(ep, model)
    with pytest.raises(TypeError, match="already attached"):
        attach_damping(ep, model)
    low = ep.controller.low_level
    q = np.arange(16, dtype=float)
    v = np.tile([.5, 1.5, 5.5, 21.], 4)
    low.requested_transfer_n = -7
    assert original.requested_transfer_n == -7
    for tick in range(3):
        state = SensorPacket(tick * .01, 0., q, v)
        assert low.compute(None, state) is state
        # Huge first interval has expired when the first non-repeated packet arrives.
        record_applied(ep, np.full(16, (50., 1., 2.)[tick]), (tick + 1) * .01)
    clone = deepcopy(low)
    state = SensorPacket(.03, .01, q, v)
    for item in (low, clone):
        corrected = item.compute(None, state)
        assert corrected.pitch == .18 and corrected.joint_position is q
        np.testing.assert_array_equal(corrected.joint_velocity, np.tile([5.5, 6.5, 8., 21.], 4))
    np.testing.assert_array_equal(state.joint_velocity, np.tile([.5, 1.5, 5.5, 21.], 4))


def test_failed_fold_withholds_final_model_and_preserves_output(tmp_path, monkeypatch):
    from scripts import terrain_damping86 as damping

    path = tmp_path / "recordings.zip"
    monkeypatch.setattr(damping, "ARCHIVE_SHA256", _tiny_recording(path, no_signal=True))
    out = tmp_path / "fresh"
    report = damping.fit_damping(path, out)
    assert not report["passed"]
    assert any(not f["passed"] for f in report["folds"])
    assert json.loads((out / "fit.json").read_text()) == report
    assert not (out / "model.json").exists()
    with pytest.raises(FileExistsError):
        damping.fit_damping(path, out)


def test_modified_source_fails_hash_before_opening_as_zip(tmp_path):
    from scripts.terrain_damping86 import fit_predictor

    archive = tmp_path / "recordings.zip"
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="published"):
        fit_predictor(archive)
