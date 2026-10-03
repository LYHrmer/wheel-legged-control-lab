"""Small synthetic checks for the frozen-data diagnostic's key boundaries."""
import unittest
from unittest.mock import patch

import numpy as np

from critic32 import fit_lstsq, fit_predict, prepare


class Critic32FixtureTests(unittest.TestCase):
    def test_mc_differs_from_duration_gae(self):
        paths = {}
        for batch in range(8):
            rows = []
            for cycle in range(8*batch, 8*batch+8):
                for macro in range(4):
                    obs = [0.]*54
                    obs[0] = -1. if cycle % 2 == 0 else 1.
                    obs[1+macro] = 1.
                    obs[5] = macro/3
                    gae = sum(.95**k for k in range(4-macro))
                    rows.append({"cycle_index": cycle, "macro_index": macro,
                        "control_start": 200+100*macro,
                        "control_end": 300+100*macro,
                        "elapsed_controls": 100,
                        "observation54": obs, "old_value": 0., "gae": gae,
                        "value_target": gae, "reward": 1., "delta": 1.,
                        "duration_s": 1., "gae_factor": .95,
                        "terminal": macro == 3, "terminal_adjustment": 0.,
                        "reward_components": {
                            "progress_potential_difference": .2, "elapsed_s": 1.,
                            "backtrack_normalized": 0.,
                            "longitudinal_normalized_square_integral_s": 0.,
                            "lateral_goal_error_normalized_square_integral_s": 0.,
                            "yaw_error_normalized_square_integral_s": 0.,
                            "torque_normalized_square_integral_s": 0.}})
            paths[f"batch_{batch:02d}.json"] = {"prepared_batch": {
                "cycle_indices": list(range(8*batch, 8*batch+8)),
                "macro_count": 32, "rows": rows}}
        _, _, y, _, _, _, _ = prepare(paths)
        np.testing.assert_allclose(y[:4, 1], [4., 3., 2., 1.])
        self.assertGreater(y[0, 1], y[0, 0])

    def test_validation_labels_cannot_change_fit(self):
        x = np.zeros((256, 54))
        cell = np.tile(np.arange(8), 32)
        x[:, 0] = np.where(cell < 4, -1., 1.)
        x[:, 5] = (cell % 4)/3
        x[:, 6] = np.arange(256)/256
        y = np.column_stack((2+x[:, 6], -3+2*x[:, 6]))
        calls = []
        def stub_lstsq(a, b, rcond):
            calls.append((a.copy(), b.copy(), rcond))
            return (np.zeros((a.shape[1], b.shape[1])), np.empty(0),
                    a.shape[1], np.ones(a.shape[1]))
        with patch("numpy.linalg.lstsq", side_effect=stub_lstsq):
            first, _, counts = fit_predict(x, y, cell)
            changed = y.copy()
            changed[128:] += 1000.
            second, _, _ = fit_predict(x, changed, cell)
        for name in first:
            np.testing.assert_allclose(first[name], second[name], atol=1e-10)
        self.assertEqual(len(calls), 6)
        for before, after in zip(calls[:3], calls[3:]):
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2], 1e-12)
        self.assertEqual(counts["lstsq_calls"], 3)
        self.assertEqual(counts["model_scalar_predictions"], 2048)

    def test_ridge_does_not_penalize_intercept(self):
        x = np.column_stack((np.ones(12), np.arange(12)))
        y = np.full((12, 2), 7.)
        calls = []
        def stub_lstsq(a, b, rcond):
            calls.append((a.copy(), b.copy(), rcond))
            return np.zeros((2, 2)), np.empty(0), 2, np.ones(2)
        with patch("numpy.linalg.lstsq", side_effect=stub_lstsq):
            fit_lstsq(x, y, ridge=.01, free=1)
        self.assertEqual(len(calls), 1)
        design, target, rcond = calls[0]
        np.testing.assert_array_equal(design[:12], x)
        np.testing.assert_array_equal(target[:12], y)
        np.testing.assert_array_equal(design[12:], [[0., 0.], [0., np.sqrt(.12)]])
        np.testing.assert_array_equal(target[12:], np.zeros((2, 2)))
        self.assertEqual(rcond, 1e-12)


if __name__ == "__main__":
    unittest.main()
