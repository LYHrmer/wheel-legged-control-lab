import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import pytest

from wheel_legged_control import train


class _FakeEnvironment:
    observation_schema = "test-observation-v1"
    reward_schema = "test-reward-v1"


class _DummyVecEnv:
    pass


class _SubprocVecEnv:
    pass


class _FakeVectorEnvironment:
    def __init__(self) -> None:
        self.closed = False
        self.ppo_kwargs: dict[str, Any] = {}
        self.logger: Any = None

    def close(self) -> None:
        self.closed = True


class _FakePPO:
    def __init__(self, policy: str, vector_env: Any, **kwargs: Any) -> None:
        self.vector_env = vector_env
        vector_env.ppo_kwargs = kwargs
        self.num_timesteps = 0

    def set_logger(self, logger: Any) -> None:
        self.vector_env.logger = logger

    def learn(self, *, total_timesteps: int) -> None:
        self.num_timesteps = total_timesteps + 24

    def save(self, path: Path) -> None:
        path.with_suffix(".zip").write_bytes(b"fake-ppo-archive")


def test_train_once_records_reproducibility_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = train.build_parser().parse_args(
        [
            "--robot",
            "d1",
            "--state-mode",
            "estimated",
            "--contact-allocation",
            "constrained",
            "--steps",
            "100",
            "--envs",
            "2",
            "--seed",
            "9",
        ]
    )
    vector_environment = _FakeVectorEnvironment()
    vec_call: dict[str, Any] = {}

    def fake_make_vec_env(environment: type, **kwargs: Any) -> _FakeVectorEnvironment:
        vec_call.update({"environment": environment, **kwargs})
        return vector_environment

    monkeypatch.setattr(
        train,
        "_git_provenance",
        lambda: {
            "git_commit": "a" * 40,
            "git_dirty": True,
            "git_worktree_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        train,
        "_dependency_versions",
        lambda: {"numpy": "2.0-test", "stable-baselines3": "3.0-test"},
    )
    archive = train.train_once(
        args,
        output=tmp_path,
        seed=13,
        environment=_FakeEnvironment,
        ppo_class=_FakePPO,
        make_vec_env=fake_make_vec_env,
        dummy_vec_env=_DummyVecEnv,
        subproc_vec_env=_SubprocVecEnv,
    )

    config = json.loads((tmp_path / "training_config.json").read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(b"fake-ppo-archive").hexdigest()
    assert archive == tmp_path / "model.zip"
    assert vector_environment.closed
    assert vec_call["seed"] == 13
    assert vec_call["vec_env_cls"] is _SubprocVecEnv
    assert vec_call["vec_env_kwargs"] == {"start_method": "fork"}
    assert vec_call["env_kwargs"] == {
        "baseline": "lqr",
        "randomize": True,
        "state_mode": "estimated",
        "latency_compensation": "none",
        "contact_allocation": "constrained",
    }
    assert config["seed"] == 13
    assert config["state_mode"] == "estimated"
    assert config["contact_allocation"] == "constrained"
    assert config["actual_timesteps"] == 124
    assert config["git_commit"] == "a" * 40
    assert config["git_dirty"] is True
    assert config["git_worktree_sha256"] == "b" * 64
    assert config["python_version"] == platform.python_version()
    assert config["dependency_versions"]["numpy"] == "2.0-test"
    assert config["observation_schema"] == "test-observation-v1"
    assert config["reward_schema"] == "test-reward-v1"
    assert config["model_sha"] == expected_sha
    assert config["model_sha256"] == expected_sha
    expected_hyperparameters = {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "policy_kwargs": {"net_arch": [128, 128]},
    }
    assert config["ppo_hyperparameters"] == expected_hyperparameters
    for key, expected in expected_hyperparameters.items():
        assert vector_environment.ppo_kwargs[key] == expected
    assert config["learning_metrics_csv"] is None
    assert vector_environment.logger is None


def test_main_expands_multiple_runs_into_seed_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train,
        "_load_rl_dependencies",
        lambda: (_FakePPO, object(), _DummyVecEnv, _SubprocVecEnv),
    )
    calls: list[tuple[int, Path, str]] = []

    def fake_train_once(
        args: Any,
        *,
        output: Path,
        seed: int,
        **kwargs: Any,
    ) -> Path:
        calls.append((seed, output, args.state_mode))
        return output / "model.zip"

    monkeypatch.setattr(train, "train_once", fake_train_once)
    train.main(
        [
            "--robot",
            "d1",
            "--state-mode",
            "estimated",
            "--runs",
            "3",
            "--seed",
            "11",
            "--output",
            str(tmp_path),
        ]
    )

    assert calls == [
        (11, tmp_path / "seed_0011", "estimated"),
        (19, tmp_path / "seed_0019", "estimated"),
        (27, tmp_path / "seed_0027", "estimated"),
    ]
    manifest = json.loads((tmp_path / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment_seed_stride"] == 8
    assert [run["seed"] for run in manifest["runs"]] == [11, 19, 27]


def test_single_run_keeps_legacy_output_and_planar_rejects_estimated_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train,
        "_load_rl_dependencies",
        lambda: (_FakePPO, object(), _DummyVecEnv, _SubprocVecEnv),
    )
    outputs: list[Path] = []
    monkeypatch.setattr(
        train,
        "train_once",
        lambda args, *, output, **kwargs: outputs.append(output) or output / "model.zip",
    )
    train.main(["--output", str(tmp_path)])
    assert outputs == [tmp_path]

    with pytest.raises(SystemExit, match="only supported with --robot d1"):
        train.main(["--robot", "planar", "--state-mode", "estimated"])


@pytest.mark.parametrize("runs", ("0", "-2"))
def test_runs_must_be_positive(runs: str) -> None:
    with pytest.raises(SystemExit, match="must be positive"):
        train.main(["--runs", runs])


def test_latency_compensation_requires_estimated_d1_state() -> None:
    with pytest.raises(SystemExit, match="requires --state-mode estimated"):
        train.main(["--robot", "d1", "--latency-compensation", "constant_velocity"])


class _FakeLogger:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_custom_hyperparameters_reach_ppo_metadata_and_optional_logger(tmp_path: Path) -> None:
    args = train.build_parser().parse_args(
        [
            "--learning-rate",
            "0.001",
            "--n-steps",
            "64",
            "--batch-size",
            "32",
            "--n-epochs",
            "2",
            "--gamma",
            "0.97",
            "--gae-lambda",
            "0.8",
            "--clip-range",
            "0.1",
            "--ent-coef",
            "0.01",
            "--envs",
            "1",
            "--steps",
            "128",
            "--log-training-metrics",
            "--verbose",
        ]
    )
    vector_environment = _FakeVectorEnvironment()
    logger = _FakeLogger()
    logger_calls = []

    def configure_logger(path: Path, verbose: bool) -> _FakeLogger:
        logger_calls.append((path, verbose))
        return logger

    train.train_once(
        args,
        output=tmp_path,
        seed=7,
        environment=_FakeEnvironment,
        ppo_class=_FakePPO,
        make_vec_env=lambda *args, **kwargs: vector_environment,
        dummy_vec_env=_DummyVecEnv,
        subproc_vec_env=_SubprocVecEnv,
        logger_factory=configure_logger,
    )
    expected = {
        "learning_rate": 0.001,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "gamma": 0.97,
        "gae_lambda": 0.8,
        "clip_range": 0.1,
        "ent_coef": 0.01,
        "policy_kwargs": {"net_arch": [128, 128]},
    }
    config = json.loads((tmp_path / "training_config.json").read_text(encoding="utf-8"))
    assert config["ppo_hyperparameters"] == expected
    for key, value in expected.items():
        assert vector_environment.ppo_kwargs[key] == value
    assert logger_calls == [(tmp_path / "learning_metrics", True)]
    assert vector_environment.logger is logger
    assert logger.closed and vector_environment.closed
    assert config["learning_metrics_csv"] == "learning_metrics/progress.csv"


def test_logger_factory_is_not_called_by_default(tmp_path: Path) -> None:
    args = train.build_parser().parse_args([])
    vector_environment = _FakeVectorEnvironment()

    def forbidden_logger(*args: Any) -> Any:
        raise AssertionError("default training must keep the framework logger")

    train.train_once(
        args,
        output=tmp_path,
        seed=7,
        environment=_FakeEnvironment,
        ppo_class=_FakePPO,
        make_vec_env=lambda *args, **kwargs: vector_environment,
        dummy_vec_env=_DummyVecEnv,
        subproc_vec_env=_SubprocVecEnv,
        logger_factory=forbidden_logger,
    )
    assert vector_environment.logger is None
    assert not (tmp_path / "learning_metrics").exists()


@pytest.mark.parametrize("failure_phase", ["construction", "logger", "set_logger", "learn"])
def test_failed_training_closes_vector_environment_and_created_logger(
    tmp_path: Path, failure_phase: str
) -> None:
    args = train.build_parser().parse_args(["--log-training-metrics"])
    vector_environment = _FakeVectorEnvironment()
    logger = _FakeLogger()

    class FailingPPO(_FakePPO):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if failure_phase == "construction":
                raise RuntimeError("construction failed")
            super().__init__(*args, **kwargs)

        def set_logger(self, configured_logger: Any) -> None:
            if failure_phase == "set_logger":
                raise RuntimeError("set_logger failed")
            super().set_logger(configured_logger)

        def learn(self, **kwargs: Any) -> None:
            raise RuntimeError("learn failed")

    def configure_logger(*args: Any) -> _FakeLogger:
        if failure_phase == "logger":
            raise RuntimeError("logger failed")
        return logger

    with pytest.raises(RuntimeError, match=failure_phase):
        train.train_once(
            args,
            output=tmp_path,
            seed=7,
            environment=_FakeEnvironment,
            ppo_class=FailingPPO,
            make_vec_env=lambda *args, **kwargs: vector_environment,
            dummy_vec_env=_DummyVecEnv,
            subproc_vec_env=_SubprocVecEnv,
            logger_factory=configure_logger,
        )
    assert vector_environment.closed
    assert logger.closed is (failure_phase in {"set_logger", "learn"})
    assert not (tmp_path / "training_config.json").exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--learning-rate", "nan"),
        ("--learning-rate", "inf"),
        ("--learning-rate", "0"),
        ("--gamma", "nan"),
        ("--gamma", "1.01"),
        ("--gamma", "-0.01"),
        ("--gae-lambda", "inf"),
        ("--gae-lambda", "1.1"),
        ("--gae-lambda", "-1"),
        ("--clip-range", "0"),
        ("--clip-range", "1"),
        ("--clip-range", "nan"),
        ("--ent-coef", "-1"),
        ("--ent-coef", "nan"),
        ("--n-steps", "0"),
        ("--batch-size", "1"),
        ("--batch-size", "257"),
        ("--n-epochs", "0"),
        ("--steps", "0"),
        ("--envs", "0"),
        ("--runs", "0"),
    ],
)
def test_invalid_parameters_fail_before_dependencies_or_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    def forbidden_dependency_load() -> Any:
        raise AssertionError("invalid parameters reached RL dependency loading")

    monkeypatch.setattr(train, "_load_rl_dependencies", forbidden_dependency_load)
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit):
        train.main(["--envs", "1", "--output", str(output), flag, value])
    assert not output.exists()


def test_direct_train_once_rejects_invalid_parameters_before_creating_environment(
    tmp_path: Path,
) -> None:
    args = train.build_parser().parse_args(["--gamma", "nan"])

    def forbidden_environment(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid parameters reached environment creation")

    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="finite"):
        train.train_once(
            args,
            output=output,
            seed=7,
            environment=_FakeEnvironment,
            ppo_class=_FakePPO,
            make_vec_env=forbidden_environment,
            dummy_vec_env=_DummyVecEnv,
            subproc_vec_env=_SubprocVecEnv,
        )
    assert not output.exists()


def test_non_dividing_minibatch_and_endpoint_discounts_are_allowed() -> None:
    args = train.build_parser().parse_args(
        [
            "--envs",
            "1",
            "--n-steps",
            "64",
            "--batch-size",
            "40",
            "--gamma",
            "0",
            "--gae-lambda",
            "1",
        ]
    )
    config = train.build_ppo_hyperparameters(args)
    assert config["batch_size"] == 40
    assert config["gamma"] == 0.0
    assert config["gae_lambda"] == 1.0
