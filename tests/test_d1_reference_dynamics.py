import numpy as np
import pytest

from scripts.diagnose_d1_reference_dynamics import ReferenceProbeEnv, moving_height_velocity
from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv


@pytest.mark.parametrize(
    "pitch,vx,expected",
    [(0.0, 0.3, 0.0), (-np.pi / 4, 0.3, 0.3), (np.pi / 4, 0.3, -0.3), (-np.pi / 4, -0.3, -0.3)],
)
def test_height_chain_rule_sign(pitch, vx, expected):
    assert moving_height_velocity(pitch, vx) == pytest.approx(expected)


@pytest.mark.parametrize("variant", ["baseline", "vertical_feedforward", "attitude_tau_50ms"])
def test_flat_probe_preserves_physical_path(variant):
    original = D1TerrainTrackingEnv(training_mode="flat", randomize=False)
    probe = ReferenceProbeEnv(variant=variant, training_mode="flat", randomize=False)
    try:
        options = {"terrain": {"kind": "flat"}, "velocity_mps": 0.35}
        original.reset(seed=31, options=options)
        probe.reset(seed=31, options=options)
        for _ in range(100):
            a, b = original.step(np.zeros(2)), probe.step(np.zeros(2))
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
            np.testing.assert_array_equal(original.plant.data.qpos, probe.plant.data.qpos)
    finally:
        original.close()
        probe.close()


def test_probe_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown reference"):
        ReferenceProbeEnv(variant="combined_changes")
