"""Development checkpoints must not perturb the training random stream."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PPO = pytest.importorskip("stable_baselines3").PPO

from scripts import run_d1_action_ablation as study


def test_real_checkpoint_evaluation_preserves_training_rng(tmp_path):
    model = PPO("MlpPolicy", study.InitializationSpaceEnv(), seed=4242, device="cpu")
    checkpoint = tmp_path / "policy.zip"
    model.save(checkpoint)
    random.seed(918)
    np.random.seed(918)
    torch.manual_seed(918)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with study.preserve_training_rng():
        loaded = PPO.load(checkpoint, device="cpu")
        loaded.predict(np.zeros(44, dtype=np.float32), deterministic=False)
        random.random()
        np.random.normal()
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0] and after_numpy[2:] == before[1][2:]
    np.testing.assert_array_equal(after_numpy[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_rng_is_restored_even_if_development_evaluation_fails():
    torch.manual_seed(111)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="test interruption"), study.preserve_training_rng():
        torch.randn(100)
        raise RuntimeError("test interruption")
    assert torch.equal(before, torch.get_rng_state())
