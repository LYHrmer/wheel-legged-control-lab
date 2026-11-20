import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import terrain_axle86 as axle

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/assets/project85/recordings.zip"


@pytest.mark.parametrize(("nominal", "transfer", "expected"), [
    ([3, 3, 6, 6], 4, [5, 5, 4, 4]),
    ([10, 2, 5, 7], 4, [10, 6, 5, 3]),
    ([10, 0, 5, 0], 4, [10, 0, 5, 0]),
    ([10, 0, 5, 0], -8, [5, 0, 10, 0]),
    ([3, 9, 3, 3], 7, [6, 10, 0, 2]),
    ([3, 9, 3, 3], 0, [3, 9, 3, 3]),
])
def test_side_sums_limits_and_equal_split_before_redistribution(nominal, transfer, expected):
    before = nominal.copy()
    result, applied = axle.side_transfers(nominal, transfer, 10)
    assert list(result) == expected
    assert applied == sum(expected[:2]) - sum(nominal[:2])
    for side in (0, 1):
        assert nominal[side] + nominal[side + 2] == result[side] + result[side + 2]
    assert all(0 <= x <= 10 for x in result)
    assert nominal == before


def test_zero_and_infeasible_force_bounds_fail_closed():
    for f, r, c in (([1] * 3, 1, 5), ([1, 1, 1, 6], 1, 5), ([0] * 4, float("nan"), 1),
                    ([float("inf")] * 4, 1, 1), ([0] * 4, 0, 0)):
        with pytest.raises(ValueError):
            axle.side_transfers(f, r, c)


def test_runtime_uses_world_vertical_jacobian_with_original_wheel_channels(monkeypatch):
    from wheel_legged_control.d1.controllers import D1VMCController

    low = object.__new__(axle.SideTransferVMC)
    low.total_mass_kg = 10.0 / (0.65 * 9.81)
    nominal, original = np.array([10., 2., 5., 7.]), np.arange(16.)
    monkeypatch.setattr(D1VMCController, "_support_torque", lambda *_: (original, nominal))
    state = SimpleNamespace(foot_jacobian=np.arange(48.).reshape(4, 3, 4))
    low.requested_transfer_n = 0.
    assert low._support_torque(state, None, 0)[0] is original
    low.requested_transfer_n = 4.
    torque, new = low._support_torque(state, None, 0)
    np.testing.assert_allclose(new, [10, 6, 5, 3])
    expected = np.zeros((4, 4))
    expected[1, :3] = -4 * state.foot_jacobian[1, 2, :3]
    expected[3, :3] = 4 * state.foot_jacobian[3, 2, :3]
    np.testing.assert_allclose(torque, original + expected.ravel())


def test_published_zip_readback_reconstructs_controls_and_separates_each_sign():
    result = axle.audit_first_ticks(ARCHIVE)
    assert result["case_count"] == 6 and result["distinct_checkpoints"] == 3
    assert result["old_reconstruction_max_error_nm"] < 1e-12
    assert result["old"]["positive_vs_negative_count"] == 4
    assert result["candidate"]["positive_vs_negative_count"] == 6
    assert result["candidate"]["bidirectional_count"] == 2
    for case in result["cases"]:
        directions = {8561: "negative_only", 8573: "positive_only", 8587: "both"}
        assert case["candidate"]["direction_class"] == directions[case["seed"]]
        for arm in case["candidate"]["arms"].values():
            assert arm["max_side_sum_error_n"] < 1e-12 and arm["wheel_linf_vs_zero_nm"] == 0
    assert result["new_native_steps"] == 0 and result["new_training_updates"] == 0


def test_reader_works_without_site_packages_and_preserves_existing_output(tmp_path):
    target = tmp_path / "audit.json"
    cmd = [sys.executable, "-IS", str(ROOT / "scripts/terrain_axle86.py"),
           "--archive", str(ARCHIVE), "--output", str(target)]
    subprocess.run(cmd, check=True, capture_output=True)
    assert len(json.loads(target.read_bytes())["cases"]) == 6
    before = target.read_bytes()
    assert subprocess.run(cmd, check=False, capture_output=True).returncode != 0
    assert target.read_bytes() == before
