"""Synthetic checks only: no saved training data, model, Torch, or physics."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_value14_data import (eligible_starts, identity, select_starts,
                                  validate_timeline, verify_closed_file)
from value14_math import bootstrap_targets, diagnostic_stats, discounted_sum


def timeline(episode_length=256, count=1024):
    control = np.arange(count, dtype=np.int64)
    episode = (control // episode_length).astype(np.int32)
    tick = (control % episode_length).astype(np.int32)
    done = tick == episode_length - 1
    return control, episode, tick, done


class Value14PureTests(unittest.TestCase):
    def test_discount_and_endpoint_offset(self):
        control, episode, tick, done = timeline()
        groups = eligible_starts(control, episode, tick, done, group_width=256)
        self.assertEqual(groups[0], [0, 64, 128])
        reward = np.arange(1024, dtype=np.float64)
        i = groups[0][1]
        self.assertEqual(i + 64, 128)
        self.assertEqual(reward[i:i + 64].shape, (64,))
        self.assertEqual(reward[i:i + 64][-1], 127)
        self.assertEqual(reward[i + 64], 128)
        expected = sum(k * 0.99 ** k for k in range(64))
        self.assertAlmostEqual(discounted_sum(np.arange(64)), expected)
        target = bootstrap_targets(np.arange(64)[None, :], [7.0])[0]
        self.assertAlmostEqual(target, expected + 0.99 ** 64 * 7)

    def test_episode_and_done_never_crossed(self):
        control, episode, tick, done = timeline(episode_length=64)
        groups = eligible_starts(control, episode, tick, done, group_width=256)
        self.assertEqual(groups, [[], [], [], []])
        control, episode, tick, done = timeline()
        done[63] = True
        with self.assertRaisesRegex(ValueError, "continuity"):
            eligible_starts(control, episode, tick, done, group_width=256)

    def test_missing_control_or_tick_rejected(self):
        control, episode, tick, done = timeline()
        control[70] = 999
        with self.assertRaisesRegex(ValueError, "control indices"):
            validate_timeline(control, episode, tick, done)
        control[70] = 70
        tick[70] += 1
        with self.assertRaisesRegex(ValueError, "continuity"):
            validate_timeline(control, episode, tick, done)

    def test_hash_rank_is_repeatable_unique_and_nonoverlapping(self):
        control, episode, tick, done = timeline()
        groups = eligible_starts(control, episode, tick, done, group_width=256)
        first = select_starts(groups, per_group=2)
        second = select_starts(groups, per_group=2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first.tolist())), 8)
        for q in range(4):
            subset = first[q * 2:(q + 1) * 2]
            self.assertTrue(np.all(subset // 256 == q))
            self.assertGreaterEqual(subset[1] - subset[0], 64)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            select_starts(groups, per_group=4)

    def test_nonfinite_and_zero_variance(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            discounted_sum([[1.0, np.nan]])
        with self.assertRaisesRegex(ValueError, "finite"):
            bootstrap_targets([[1.0]], [np.inf])
        with self.assertRaisesRegex(ValueError, "finite"):
            diagnostic_stats([1.0], [np.nan])
        result = diagnostic_stats([1.0, 3.0], [2.0, 2.0])
        self.assertIsNone(result["explained_variance"])
        self.assertEqual(result["bias_target_minus_value"], 0.0)
        self.assertEqual(result["rmse"], 1.0)

    def test_bias_sign_and_explained_variance(self):
        result = diagnostic_stats([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        self.assertEqual(result["bias_target_minus_value"], 1.0)
        self.assertEqual(result["median_error_target_minus_value"], 1.0)
        self.assertEqual(result["explained_variance"], 1.0)

    def test_closed_file_hash_and_size_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            path = run / "sample.bin"
            path.write_bytes(b"closed")
            expected = identity(path)
            self.assertEqual(expected["sha256"], hashlib.sha256(b"closed").hexdigest())
            verified = {}
            self.assertEqual(verify_closed_file(run, {"sample.bin": expected},
                                                "sample.bin", verified), path)
            self.assertEqual(verified["sample.bin"], expected)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_closed_file(run, {"sample.bin": expected}, "sample.bin", {})
            with self.assertRaisesRegex(ValueError, "not in closed manifest"):
                verify_closed_file(run, {"sample.bin": expected}, "../sample.bin", {})


if __name__ == "__main__":
    unittest.main()
