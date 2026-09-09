"""Residual-only interventions, reproducible replay, and descriptive error types."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def diagnostic():
    spec = importlib.util.spec_from_file_location(
        "residual_feedback_diagnostic", ROOT / "scripts/diagnose_d1_residual_feedback.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "errors,label",
    [
        ([-0.1] * 10, "bias_dominated_slow"),
        ([0.1] * 10, "bias_dominated_fast"),
        ([-0.1, 0.1] * 5, "fluctuation_dominated"),
        ([0] * 10, "negligible"),
        ([0, 0.2] * 5, "mixed"),
    ],
)
def test_bias_variance_identity_and_types(diagnostic, errors, label):
    result = diagnostic.error_decomposition(errors)
    assert result["failure_type"] == label
    assert result["rmse_mps"] ** 2 == pytest.approx(
        result["bias_mps"] ** 2 + result["variance_m2ps2"]
    )


@pytest.mark.parametrize("errors", [[], [[1]], [np.nan], [np.inf]])
def test_invalid_error_vectors_rejected(diagnostic, errors):
    with pytest.raises(ValueError):
        diagnostic.error_decomposition(errors)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("zero", [0, 0]),
        ("constant_fz", [0, 1]),
        ("full", [0.3, 1]),
        ("fx_only", [0.3, 0]),
        ("fz_only", [0, 1]),
    ],
)
def test_residual_masks_do_not_mutate_policy_action(diagnostic, mode, expected):
    action = np.array([0.3, 2.0])
    np.testing.assert_array_equal(diagnostic.select_action(mode, action), expected)
    np.testing.assert_array_equal(action, [0.3, 2])


def test_replay_ignores_counterfactual_live_action(diagnostic):
    np.testing.assert_array_equal(
        diagnostic.select_action("replay", np.ones(2), np.array([-0.2, -0.4])), [-0.2, -0.4]
    )
    for mode, action in (("unknown", [0, 0]), ("full", None), ("replay", None)):
        with pytest.raises(ValueError):
            diagnostic.select_action(mode, action)


def test_cli_refuses_old_output_before_loading_models(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "load_models", lambda *args: pytest.fail("loaded models"))
    args = diagnostic.build_parser().parse_args(["--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        diagnostic.run(args)


@pytest.mark.parametrize(
    "extra",
    [
        ["--seeds", "0", "0"],
        ["--case-indices", "24"],
        ["--case-indices", "1", "1"],
        ["--case-indices", "1"],
        ["--feedback-case-indices", "4", "4"],
    ],
)
def test_invalid_protocol_fails_before_execution(diagnostic, tmp_path, extra):
    args = diagnostic.build_parser().parse_args(["--output", str(tmp_path / "new"), *extra])
    with pytest.raises(ValueError):
        diagnostic.validate_args(args)


def test_perturbations_only_change_the_declared_variable(diagnostic):
    class ResetOnly:
        def reset(self, *, seed, options):
            self.seed, self.options = seed, options
            return np.zeros(44), {}

    case = diagnostic.RUNNER.evaluation_cases("development", "tracking-v2")[4]
    nominal, pitched, pushed = ResetOnly(), ResetOnly(), ResetOnly()
    diagnostic.configure_perturbation(nominal, case, "nominal")
    diagnostic.configure_perturbation(pitched, case, "pitch_minus")
    diagnostic.configure_perturbation(pushed, case, "push_plus")
    assert pitched.options == nominal.options | {"initial_pitch": -0.02}
    assert pushed.options == nominal.options
    assert (pushed._push_start, pushed._push_end, pushed._push_force_n) == (200, 210, 20.0)
    assert nominal.seed == pitched.seed == pushed.seed == case["environment_seed"]


def test_real_gaussian_clipping_and_nominal_replay(diagnostic):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    torch.set_num_threads(1)
    env = diagnostic.D1TerrainTrackingEnv(episode_seconds=0.1, training_mode="flat")
    try:
        model = sb3.PPO("MlpPolicy", env, n_steps=2, batch_size=2, device="cpu", seed=7)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_(torch.tensor([0.3, 3.0]))
        case = diagnostic.RUNNER.evaluation_cases("development", "tracking-v2")[1]
        nominal, metrics = diagnostic.run_episode(case, "full", model=model, seconds=0.1)
        assert metrics["vertical_upper_fraction"] == 1.0
        assert metrics["raw_mu_z_mean"] == 3.0
        assert metrics["raw_upper_probability_z_mean"] > 0.97
        assert metrics["quality_success"] is None
        assert {key for key in metrics if key.startswith("return_")} == {
            f"return_{name}" for name in (*diagnostic.REWARD_FIELDS, "total")
        }
        tape = np.asarray([[row["action_longitudinal"], row["action_vertical"]] for row in nominal])
        replay, _ = diagnostic.run_episode(case, "replay", model=model, tape=tape, seconds=0.1)
        assert diagnostic.verify_nominal_replay(nominal, replay) == 0.0
        assert all(row["policy_enabled"] == 0 for row in replay)
        assert all(row["reward"] == pytest.approx(row["reward_total"]) for row in replay)
        assert metrics["return_residual_effort"] == pytest.approx(-0.04 * (0.3**2 + 2) * 10)
        replay[0]["qpos_qvel_sha256"] = "changed"
        with pytest.raises(AssertionError, match="qpos/qvel"):
            diagnostic.verify_nominal_replay(nominal, replay)
    finally:
        env.close()


def test_real_push_has_exact_control_window_and_no_hidden_reset_steps(diagnostic):
    case = diagnostic.RUNNER.evaluation_cases("development", "tracking-v2")[1]
    rows, _ = diagnostic.run_episode(case, "zero", perturbation="push_plus", seconds=2.2)
    assert len(rows) == 220
    assert [row["step"] for row in rows if row["push_force_n"]] == list(range(201, 211))
    assert sum(row["push_force_n"] for row in rows) * 0.01 == 2.0
    assert rows[0]["time_s"] == 0.01


def test_nominal_replay_detects_changed_physical_state(diagnostic):
    row = {key: 0.0 for key in diagnostic.REPLAY_FIELDS}
    row.update(termination_reason="ongoing", qpos_qvel_sha256="unchanged")
    assert diagnostic.verify_nominal_replay([row], [row.copy()]) == 0
    with pytest.raises(AssertionError, match="physical transition"):
        diagnostic.verify_nominal_replay([row], [row | {"position_x_m": 0.001}])
    with pytest.raises(AssertionError, match="length"):
        diagnostic.verify_nominal_replay([row], [])


def test_feedback_pairing_is_by_seed_case_and_disturbance(diagnostic):
    common = {
        "training_seed": 0,
        "case_id": "case",
        "perturbation": "push_plus",
        "velocity_rmse_mps": 0.1,
        "clearance_rmse_m": 0.01,
        "episode_return": 100,
        "quality_success": 1,
    }
    rows = [
        common | {"mode": "full", "tail_velocity_rmse_mps": 0.07},
        common | {"mode": "replay", "tail_velocity_rmse_mps": 0.1},
    ]
    summary = diagnostic.paired_summary(rows, [0])
    assert len(summary) == 1
    assert summary[0]["paired_episodes"] == 1
    assert summary[0]["predeclared_threshold_met"]
    rows[1]["perturbation"] = "push_minus"
    with pytest.raises(AssertionError, match="unmatched"):
        diagnostic.paired_summary(rows, [0])
