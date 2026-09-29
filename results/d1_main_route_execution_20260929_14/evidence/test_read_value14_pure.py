"""Small synthetic corruption tests for the independent readback helpers."""

import unittest

import numpy as np

from read_value14 import (RHO, check_counts, check_gradient_batch,
                          check_values, check_window_indices, residual_stats)


class IndependentReadbackTests(unittest.TestCase):
    def sample(self):
        return {"start_indices": np.array([0]), "endpoint_indices": np.array([64]),
                "quartile": np.array([0]), "episode_tick": np.array([0]),
                "episode_index": np.array([0]),
                "rewards": np.zeros((1, 64), dtype=np.float64),
                "discounted_rewards": np.zeros(1, dtype=np.float64),
                "start_obs": np.zeros((1, 99), dtype=np.float32),
                "endpoint_obs": np.zeros((1, 99), dtype=np.float32)}

    def values(self):
        target = RHO * 2
        endpoint = {"endpoint_indices": np.array([64]),
                    "endpoint_values": np.array([2], dtype=np.float32)}
        values = {"start_indices": np.array([0]), "quartile": np.array([0]),
                  "start_values": np.array([1], dtype=np.float32),
                  "endpoint_values": endpoint["endpoint_values"].copy(),
                  "discounted_rewards": np.zeros(1, dtype=np.float64),
                  "targets_float64": np.array([target], dtype=np.float64),
                  "targets_used_float32": np.array([target], dtype=np.float32),
                  "residual": np.array([target - 1], dtype=np.float64)}
        return endpoint, values

    def test_endpoint_off_by_one_and_duplicate(self):
        sample = self.sample()
        check_window_indices(sample)
        sample["endpoint_indices"] = np.array([63])
        with self.assertRaisesRegex(ValueError, "start\+64"):
            check_window_indices(sample)

    def test_float64_and_float32_target_tamper(self):
        sample = self.sample()
        endpoint, values = self.values()
        check_values(sample, endpoint, values)
        values["targets_float64"][0] += .01
        with self.assertRaisesRegex(ValueError, "float64 target"):
            check_values(sample, endpoint, values)
        endpoint, values = self.values()
        values["targets_used_float32"][0] += .01
        with self.assertRaisesRegex(ValueError, "float32 loss target"):
            check_values(sample, endpoint, values)

    def test_raw_head_bias_gradient_tamper(self):
        names = ("mlp_extractor.value_net.0.weight", "mlp_extractor.value_net.0.bias",
                 "mlp_extractor.value_net.2.weight", "mlp_extractor.value_net.2.bias",
                 "value_net.weight", "value_net.bias")
        parameters = {name: {"group": "critic", "shape": [1], "l2_norm": 1.}
                      for name in names}
        parameters.update({f"actor.{i}": {"group": "actor"} for i in range(7)})
        gradients = {name: np.array([1. if name == "value_net.bias" else 0.],
                                    dtype=np.float32) for name in names}
        batch = {"per_parameter_l2": {name: float(abs(gradients[name][0])) for name in names},
                 "per_parameter_grad_over_parameter_norm": {
                     name: float(abs(gradients[name][0])) for name in names},
                 "critic_gradient_l2": 1., "critic_only_clip_factor": .5 / (1 + 1e-6),
                 "analytic_bias_gradient_expected": 1., "actual_bias_gradient": 1.,
                 "weighted_proxy_loss_float32": .5,
                 "actor_gradients_all_none": True,
                 "actor_gradient_counts": {"none": 7, "zero": 0, "nonzero": 0}}
        start, target = np.array([1], dtype=np.float32), np.array([0], dtype=np.float32)
        check_gradient_batch(gradients, batch, parameters, start, target)
        gradients["value_net.bias"][0] = 2
        batch["per_parameter_l2"]["value_net.bias"] = 2.
        batch["per_parameter_grad_over_parameter_norm"]["value_net.bias"] = 2.
        batch["critic_gradient_l2"] = 2.
        batch["critic_only_clip_factor"] = .5 / (2 + 1e-6)
        with self.assertRaisesRegex(ValueError, "raw head bias"):
            check_gradient_batch(gradients, batch, parameters, start, target)

    def test_accounting_tamper(self):
        counts = {"ppo_load_attempted": 1, "ppo_load_returned": 1,
                  "value_batches_attempted": 5, "value_batches_returned": 5,
                  "value_states_attempted": 1024, "value_states_returned": 1024,
                  "head_batches_attempted": 5, "head_batches_returned": 5,
                  "head_states_attempted": 1024, "head_states_returned": 1024,
                  "backward_attempted": 4, "backward_returned": 4,
                  "torch_load_attempted": 2, "torch_load_returned": 2,
                  "actor_forward_attempted": 0, "optimizer_step_attempted": 0,
                  "learn_attempted": 0, "train_attempted": 0, "save_attempted": 0,
                  "autograd_grad_attempted": 0, "clip_attempted": 0}
        state = {"policy": "same", "optimizer": "same"}
        receipt = {"counts": counts, "physics_controls": 0, "native_steps": 0,
                   "status": "passed", "source_postcheck_passed": True,
                   "cleanup_errors": [], "loaded_state": state,
                   "unchanged_state": state, "cleanup_state": state,
                   "shared_trainable_parameters": 0,
                   "final_gradients_cleared": True, "actor_forward_zero": True,
                   "physics_modules_and_libraries_absent": True,
                   "model_state_matches_recorded_final": True,
                   "historical_GAE_reconstructed": False, "on_policy_claim": False}
        check_counts(receipt)
        counts["backward_attempted"] = 5
        with self.assertRaisesRegex(ValueError, "backward_attempted"):
            check_counts(receipt)

    def test_near_constant_float32_statistics_use_float64(self):
        starts = np.array([10_000_000, 10_000_001, 10_000_001], dtype=np.float32)
        targets = starts.astype(np.float64) + 2
        stats = residual_stats(starts, targets)
        self.assertEqual(stats["value_mean"], float(np.mean(starts.astype(np.float64))))
        self.assertEqual(stats["value_std"], float(np.std(starts.astype(np.float64))))
        self.assertEqual(stats["bias_target_minus_value"], 2.0)


if __name__ == "__main__":
    unittest.main()
