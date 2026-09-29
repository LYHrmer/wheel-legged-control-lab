"""Synthetic tiny Torch parameters only; no PPO model or robot environment."""

import math
import unittest

import torch

from learning15 import ClipAudit, gradient_norm, parameter_groups, scoped_clip_audit


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

    def evaluate_actions(self, *args, **kwargs):
        return None


class TinyModel:
    def __init__(self):
        self.policy = TinyPolicy()

    def train(self):
        raise RuntimeError("synthetic train error")


def gradients(model, actor=2., critic=8.):
    pi, vf = parameter_groups(model.policy)
    for parameter in pi.values():
        parameter.grad = torch.full_like(parameter, actor)
    for parameter in vf.values():
        parameter.grad = torch.full_like(parameter, critic)
    return pi, vf


class Learning15PureTests(unittest.TestCase):
    def test_global_matches_native_exactly(self):
        model = TinyModel()
        pi, vf = gradients(model)
        before = {name: parameter.grad.clone() for name, parameter in model.policy.named_parameters()}
        audit = ClipAudit(model, "global", 256)
        audit.set_gradient_writer(lambda step, pre, post, ids: {"step": step, "ids": ids})
        returned = audit.intercept(torch.nn.utils.clip_grad_norm_,
                                   model.policy.parameters(), .5)
        expected_parameters = [torch.nn.Parameter(torch.zeros_like(p))
                               for p in model.policy.parameters()]
        for p, source in zip(expected_parameters, before.values()):
            p.grad = source.clone()
        native_return = torch.nn.utils.clip_grad_norm_(expected_parameters, .5)
        self.assertEqual(float(returned), float(native_return))
        for p, expected in zip(model.policy.parameters(), expected_parameters):
            torch.testing.assert_close(p.grad, expected.grad, rtol=0, atol=0)
        self.assertLessEqual(math.hypot(gradient_norm(pi), gradient_norm(vf)), .50001)
        self.assertEqual(audit.native_clip_attempted, 1)
        self.assertEqual(audit.clip_attempted, audit.clip_returned)
        self.assertEqual(len(audit.gradient_sample_records), 1)

    def test_grouped_caps_both_and_records_real_adam_delta(self):
        model = TinyModel()
        pi, vf = gradients(model)
        audit = ClipAudit(model, "grouped", 256)
        audit.set_gradient_writer(lambda step, pre, post, ids: {"step": step, "ids": ids})
        audit.intercept(torch.nn.utils.clip_grad_norm_, model.policy.parameters(), .5)
        self.assertLessEqual(gradient_norm(pi), .50001)
        self.assertLessEqual(gradient_norm(vf), .50001)
        self.assertGreater(math.hypot(gradient_norm(pi), gradient_norm(vf)), .5)
        audit.before_optimizer()
        model.policy.optimizer.step()
        audit.after_optimizer()
        event = audit.events[0]
        self.assertTrue(event["optimizer_returned"])
        self.assertGreater(event["adam_actor_parameter_delta_l2"], 0)
        self.assertGreater(event["adam_critic_parameter_delta_l2"], 0)
        self.assertEqual(audit.native_clip_attempted, 2)
        self.assertEqual(audit.optimizer_returned, 1)

    def test_shared_parameter_rejected(self):
        model = TinyModel()
        model.policy.action_net.weight = model.policy.value_net.weight
        with self.assertRaisesRegex(ValueError, "shared trainable"):
            parameter_groups(model.policy)

    def test_nonfinite_fails_before_native_clip(self):
        model = TinyModel()
        pi, _ = gradients(model)
        next(iter(pi.values())).grad.fill_(float("nan"))
        audit = ClipAudit(model, "global", 256)
        with self.assertRaisesRegex(ValueError, "nonfinite gradient"):
            audit.intercept(torch.nn.utils.clip_grad_norm_,
                            model.policy.parameters(), .5)
        self.assertEqual(audit.native_clip_attempted, 0)
        self.assertEqual(audit.events, [])
        self.assertEqual(audit.clip_attempted, 1)
        self.assertEqual(audit.clip_returned, 0)

    def test_exception_restores_method_clip_and_hooks(self):
        model = TinyModel()
        audit = ClipAudit(model, "global", 256)
        native = torch.nn.utils.clip_grad_norm_
        pre_count = len(model.policy.optimizer._optimizer_step_pre_hooks)
        post_count = len(model.policy.optimizer._optimizer_step_post_hooks)
        with self.assertRaisesRegex(RuntimeError, "synthetic train error"):
            with scoped_clip_audit(model, audit):
                model.train()
        self.assertIs(torch.nn.utils.clip_grad_norm_, native)
        self.assertNotIn("train", model.__dict__)
        self.assertEqual(len(model.policy.optimizer._optimizer_step_pre_hooks), pre_count)
        self.assertEqual(len(model.policy.optimizer._optimizer_step_post_hooks), post_count)

    def test_soft_stop_before_next_minibatch(self):
        class Model(TinyModel):
            def train(self):
                return self.policy.evaluate_actions()

        model = Model()
        audit = ClipAudit(model, "global", 256)
        with self.assertRaisesRegex(TimeoutError, "before next PPO minibatch"):
            with scoped_clip_audit(model, audit, soft_stop=lambda: True):
                model.train()
        self.assertNotIn("evaluate_actions", model.policy.__dict__)
        self.assertNotIn("train", model.__dict__)

    def test_old_post_hook_books_step_before_archive_failure(self):
        old_completed = []

        class Model(TinyModel):
            def train(self):
                old_post = self.policy.optimizer.register_step_post_hook(
                    lambda opt, args, kwargs: old_completed.append(1))
                try:
                    gradients(self)
                    self.policy.evaluate_actions()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
                    self.policy.optimizer.step()
                finally:
                    old_post.remove()

        model = Model()
        audit = ClipAudit(model, "global", 256)
        audit.set_gradient_writer(lambda step, pre, post, ids: {"step": step})

        def fail_archive(event):
            raise OSError("archive interrupted")

        audit.on_step = fail_archive
        with self.assertRaisesRegex(OSError, "archive interrupted"):
            with scoped_clip_audit(model, audit):
                model.train()
        self.assertEqual(old_completed, [1])
        self.assertEqual(audit.optimizer_attempted, 1)
        self.assertEqual(audit.optimizer_returned, 1)
        self.assertTrue(audit.events[0]["optimizer_returned"])


if __name__ == "__main__":
    unittest.main()
