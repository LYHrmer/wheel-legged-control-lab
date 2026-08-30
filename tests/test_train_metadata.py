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

    def close(self) -> None:
        self.closed = True


class _FakePPO:
    def __init__(self, policy: str, vector_env: Any, **kwargs: Any) -> None:
        self.vector_env = vector_env
        self.num_timesteps = 0

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
    }
    assert config["seed"] == 13
    assert config["state_mode"] == "estimated"
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
