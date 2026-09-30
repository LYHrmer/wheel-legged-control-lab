"""Synthetic Torch-only tests of actual-batch KL and grouped Adam audit."""

from __future__ import annotations

import math
import unittest

import torch

from learning20 import ClipAudit20, parameter_groups


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_extractor = torch.nn.Module()
        self.mlp_extractor.policy_net = torch.nn.Sequential(
            torch.nn.Linear(1, 1), torch.nn.Tanh(), torch.nn.Linear(1, 1))
        self.mlp_extractor.value_net = torch.nn.Sequential(
            torch.nn.Linear(1, 1), torch.nn.Tanh(), torch.nn.Linear(1, 1))
        self.action_net = torch.nn.Linear(1, 1)
        self.value_net = torch.nn.Linear(1, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.01)


class TinyModel:
    def __init__(self):
        self.policy = TinyPolicy()
        self._n_updates = 0


def gradients(model, actor=2., critic=8.):
    pi, vf = parameter_groups(model.policy)
    for parameter in pi.values():
        parameter.grad = torch.full_like(parameter, actor)
    for parameter in vf.values():
        parameter.grad = torch.full_like(parameter, critic)


class Learning20PureTests(unittest.TestCase):
    def test_actual_batch_identity_grouped_clip_and_rejected_kl(self):
        model = TinyModel()
        clip = ClipAudit20(model, "A", 512)
        samples = []
        clip.set_gradient_writer(lambda step, pre, post, ids:
                                 samples.append((step, dict(ids))) or {"step": step})
        clip.begin_train(0)
        self.assertEqual(clip.begin_epoch(), 0)
        clip.bind_batch(0, 0, torch.zeros(4))
        accepted = clip.evaluated(torch.full((4,), .1), torch)
        self.assertFalse(accepted["normal_kl_stop"])
        gradients(model)
        clip.intercept(torch.nn.utils.clip_grad_norm_, model.policy.parameters(), .5)
        clip.before_optimizer()
        model.policy.optimizer.step()
        clip.after_optimizer()
        clip.finish_batch()
        clip.bind_batch(0, 1, torch.zeros(4))
        rejected = clip.evaluated(torch.full((4,), .4), torch)
        self.assertTrue(rejected["normal_kl_stop"])
        self.assertFalse(rejected["optimized"])
        clip.finish_batch()
        model._n_updates = 1
        clip.finish_train(model)
        self.assertEqual(clip.train_records[0]["evaluated_minibatches"], 2)
        self.assertEqual(clip.train_records[0]["optimizer_steps"], 1)
        self.assertTrue(clip.train_records[0]["normal_kl_stop"])
        self.assertEqual(clip.events[0]["minibatch"], 0)
        self.assertEqual(clip.events[0]["rollout"], 0)
        self.assertEqual(clip.native_clip_returned, 2)
        self.assertEqual(samples, [(0, {"global_step": 0, "rollout": 0,
                                       "epoch": 0, "minibatch": 0})])
        self.assertLessEqual(clip.events[0]["post_actor_l2"], .50001)
        self.assertLessEqual(clip.events[0]["post_critic_l2"], .50001)
        self.assertGreater(clip.events[0]["adam_joint_parameter_delta_l2"], 0)

    def test_hard_kl_stop_and_zero_optimizer_rejected(self):
        model = TinyModel()
        clip = ClipAudit20(model, "B", 512)
        clip.begin_train(0)
        clip.begin_epoch()
        clip.bind_batch(0, 0, torch.zeros(4))
        with self.assertRaisesRegex(RuntimeError, "actual minibatch KL"):
            clip.evaluated(torch.full((4,), .8), torch)
        self.assertTrue(clip.evaluated_batches[0]["hard_stop"])
        self.assertEqual(clip.optimizer_returned, 0)
        self.assertTrue(math.isfinite(clip.evaluated_batches[0]["approx_kl"]))
        clip.finish_batch()
        with self.assertRaisesRegex(ValueError, "zero optimizer"):
            clip.finish_train(model)


if __name__ == "__main__":
    unittest.main()
