"""Synthetic saved-data checks; no Torch, model, or engine imports."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from read_training15 import (check_clip_event, compare_first_rollout,
                             gradient_digest, read_pair, validate_episode_timeline)


def event(arm="grouped"):
    pa, pc = .8, 2.
    global_alpha = .5 / (np.hypot(pa, pc) + 1e-6)
    aa = global_alpha if arm == "global" else .5 / (pa + 1e-6)
    ca = global_alpha if arm == "global" else .5 / (pc + 1e-6)
    qa, qc = pa * aa, pc * ca
    return {"step": 0, "global_step": 0, "rollout": 0, "epoch": 0, "minibatch": 0,
            "pre_actor_l2": pa, "pre_critic_l2": pc, "pre_joint_l2": float(np.hypot(pa, pc)),
            "post_actor_l2": qa, "post_critic_l2": qc,
            "post_joint_l2": float(np.hypot(qa, qc)),
            "expected_global_alpha": global_alpha,
            "hypothetical_global_alpha_on_same_pregradient": global_alpha,
            "expected_actor_alpha": aa, "expected_critic_alpha": ca,
            "actual_actor_scale": qa / pa, "actual_critic_scale": qc / pc,
            "optimizer_attempted": True, "optimizer_returned": True,
            "missing_gradients": 0, "nonfinite_gradients": 0,
            "adam_actor_parameter_delta_l2": .01,
            "adam_critic_parameter_delta_l2": .02,
            "adam_joint_parameter_delta_l2": float(np.hypot(.01, .02)),
            "adam_actor_delta_over_parameter_norm": .001,
            "adam_critic_delta_over_parameter_norm": .002,
            "pre_gradient_sha256": {"actor": "a" * 64, "critic": "b" * 64},
            "post_gradient_sha256": {"actor": "c" * 64, "critic": "d" * 64},
            "full_gradient_saved": True}


class TrainingReadbackPureTests(unittest.TestCase):
    def test_clip_formulas_and_grouped_joint_above_global_cap(self):
        check_clip_event(event("global"), "global")
        grouped = event("grouped")
        self.assertGreater(grouped["post_joint_l2"], .5)
        check_clip_event(grouped, "grouped")
        grouped["post_actor_l2"] = .6
        with self.assertRaises(ValueError):
            check_clip_event(grouped, "grouped")

    def test_gradient_hash_depends_on_name_dtype_and_bytes(self):
        a = {"pi.weight": np.array([1, 2], dtype=np.float32)}
        b = {"pi.weight": np.array([1, 3], dtype=np.float32)}
        c = {"vf.weight": np.array([1, 2], dtype=np.float32)}
        self.assertNotEqual(gradient_digest(a), gradient_digest(b))
        self.assertNotEqual(gradient_digest(a), gradient_digest(c))
        self.assertNotEqual(gradient_digest(a), gradient_digest({
            "pi.weight": np.array([1, 2], dtype=np.float64)}))

    def test_first_rollout_byte_mismatch_rejected(self):
        source = {"numeric": {"input_observation99": np.zeros((2, 99), np.float32),
                              "reward": np.ones(2, np.float64)},
                  "gaussian": {"raw_gaussian_action16": np.zeros((2, 16), np.float64)},
                  "episodes": [{"episode_index": 0, "first_control_index": 0,
                                "reset": {"episode_reset_seed": 1},
                                "schedule": {"terrain": "flat"}}]}
        peer = {**source, "numeric": {k: v.copy() for k, v in source["numeric"].items()}}
        self.assertTrue(compare_first_rollout(source, peer)["byte_exact"])
        peer["numeric"]["reward"][1] = 2
        self.assertIn("numeric:reward", compare_first_rollout(source, peer)["differences"])

    def test_early_termination_is_valid_episode_boundary(self):
        numeric = {"control_index": np.arange(6, dtype=np.int64),
                   "episode_index": np.array([0, 0, 0, 1, 1, 1]),
                   "episode_tick": np.array([0, 1, 2, 0, 1, 2]),
                   "terminated": np.array([False, False, True, False, False, False]),
                   "truncated": np.zeros(6, dtype=bool)}
        self.assertEqual(validate_episode_timeline(numeric), {0: 0, 1: 3})
        numeric["terminated"][2] = False
        with self.assertRaisesRegex(ValueError, "reset/done boundary"):
            validate_episode_timeline(numeric)

    def test_missing_arm_returns_incomplete_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            result = read_pair(Path(directory) / "global_01", Path(directory) / "grouped_01")
        self.assertFalse(result["engineering_passed"])
        self.assertFalse(result["global"]["complete"])
        self.assertFalse(result["grouped"]["complete"])
        self.assertTrue(result["global"]["errors"])


if __name__ == "__main__":
    unittest.main()
