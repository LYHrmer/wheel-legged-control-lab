"""The final action holdout opens only for the complete, identical-protocol cohort.

Small real PPO checkpoints exercise serialization and policy identity. Test
environments are synthetic: no formal holdout terrain is simulated here.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
PPO = pytest.importorskip("stable_baselines3").PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts import evaluate_d1_action_ablation as evaluation

study = evaluation.study


class SmallUpdateEnv(gym.Env):
    """Cheap algorithm/checkpoint fixture; deliberately not a D1 holdout case."""

    observation_space = gym.spaces.Box(-5, 5, (44,), np.float32)

    def __init__(self, dimension):
        self.action_space = gym.spaces.Box(-1, 1, (dimension,), np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(44, np.float32), {}

    def step(self, action):
        self.steps += 1
        observation = np.full(44, self.steps / 100, np.float32)
        return observation, float(1 - np.square(action).sum()), False, self.steps == 8, {}


@pytest.fixture(scope="module")
def small_checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("small_action_policies")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    paths = {}
    try:
        for seed in evaluation.TRAINING_SEEDS:
            for variant in study.VARIANTS:
                dimension = 1 if variant == "fx_only" else 2
                env = DummyVecEnv([lambda d=dimension: SmallUpdateEnv(d) for _ in range(4)])
                policy = (
                    evaluation.BoundedMeanActorCriticPolicy
                    if variant == "bounded_mean"
                    else "MlpPolicy"
                )
                try:
                    model = PPO(policy, env, **study.shared.PPO_SETTINGS, seed=seed, device="cpu")
                    initialization = study.initialize_from_common_policy(model, seed)
                    model.learn(512)
                    assert model.num_timesteps == 512 and model._n_updates == 4
                    path = root / f"{variant}_{seed}.zip"
                    model.save(path)
                    paths[(variant, seed)] = (path, initialization)
                finally:
                    env.close()
        yield paths
    finally:
        torch.set_num_threads(old_threads)


@pytest.fixture
def small_budget(monkeypatch):
    monkeypatch.setattr(evaluation, "FINAL_BUDGET", 512)
    monkeypatch.setattr(evaluation, "FINAL_EPOCHS", 4, raising=False)
    if hasattr(evaluation, "FINAL_BUDGETS"):
        monkeypatch.setattr(evaluation, "FINAL_BUDGETS", (128, 256, 512))


def model_metadata(variant, seed, path, initialization):
    env = study.D1LongitudinalResidualEnv if variant == "fx_only" else study.D1TerrainTrackingEnv
    return {
        "variant": variant,
        "training_seed": seed,
        "actual_timesteps": 512,
        "ppo_epochs_completed": 4,
        "ppo": copy.deepcopy(study.shared.PPO_SETTINGS),
        **{
            name: getattr(env, name)
            for name in ("observation_schema", "action_schema", "reward_schema", "control_schema")
        },
        "residual_force_scale_n": [11.25] if variant == "fx_only" else [11.25, 20.0],
        "physical_action_mapping": "[ax,0]" if variant == "fx_only" else "[ax,az]",
        "distribution": "Normal(tanh(logits), std), samples still clipped"
        if variant == "bounded_mean"
        else "Normal(logits, std), samples clipped",
        "training_reward_scale": 0.01,
        "checkpoint_selection": "final fixed budget; no holdout used",
        "initialization": initialization,
        "model_sha256": study.shared._sha256(path),
    }


@pytest.mark.parametrize("variant", study.VARIANTS)
def test_real_updated_checkpoint_has_the_declared_distribution_and_physical_mapping(
    variant, small_checkpoints, small_budget
):
    path, initialization = small_checkpoints[(variant, 9000)]
    model = PPO.load(path, device="cpu")
    metadata = model_metadata(variant, 9000, path, initialization)
    evaluation.validate_policy(metadata, model, path)
    observation = np.zeros(44, dtype=np.float32)
    original = model.predict(observation, deterministic=True)[0]
    mapped = study.PhysicalActionPolicy(model, variant).predict(observation)[0]
    expected = np.array([original[0], 0], np.float32) if variant == "fx_only" else original
    np.testing.assert_array_equal(mapped, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_schema", "oracle-42-v0"),
        ("action_schema", "shape-compatible-but-different-action"),
        ("reward_schema", "continuous-task-reward"),
        ("control_schema", "different-yaw-controller"),
        ("residual_force_scale_n", [11.25, 30.0]),
        ("physical_action_mapping", "[az,ax]"),
        ("model_sha256", "0" * 64),
        ("actual_timesteps", 256),
        ("ppo_epochs_completed", 3),
        ("training_reward_scale", 1.0),
        ("training_seed", 10000),
    ],
)
def test_checkpoint_rejects_changed_semantics_or_training_identity(
    field, value, small_checkpoints, small_budget
):
    path, initialization = small_checkpoints[("full_gaussian", 9000)]
    model = PPO.load(path, device="cpu")
    metadata = model_metadata("full_gaussian", 9000, path, initialization)
    metadata[field] = value
    with pytest.raises(ValueError):
        evaluation.validate_policy(metadata, model, path)


def test_normal_checkpoint_cannot_claim_to_be_bounded_mean(small_checkpoints, small_budget):
    path, initialization = small_checkpoints[("full_gaussian", 9000)]
    model = PPO.load(path, device="cpu")
    metadata = model_metadata("bounded_mean", 9000, path, initialization)
    with pytest.raises(ValueError, match="policy class"):
        evaluation.validate_policy(metadata, model, path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_timesteps", 256),
        ("_n_updates", 3),
        ("gamma", 0.5),
        ("learning_rate", 1e-3),
        ("n_steps", 64),
        ("batch_size", 64),
        ("n_epochs", 2),
        ("gae_lambda", 0.8),
        ("ent_coef", 0.1),
        ("clip_range", lambda _: 0.1),
        ("lr_schedule", lambda _: 1e-3),
        ("n_envs", 2),
        ("vf_coef", 0.9),
        ("normalize_advantage", False),
        ("max_grad_norm", 2.0),
        ("target_kl", 0.1),
    ],
)
def test_claimed_final_metadata_cannot_hide_partial_or_different_loaded_training(
    field, value, small_checkpoints, small_budget
):
    path, initialization = small_checkpoints[("full_gaussian", 9000)]
    model = PPO.load(path, device="cpu")
    metadata = model_metadata("full_gaussian", 9000, path, initialization)
    setattr(model, field, value)
    with pytest.raises(ValueError):
        evaluation.validate_policy(metadata, model, path)


def test_changed_ppo_metadata_is_not_the_same_training_protocol(small_checkpoints, small_budget):
    path, initialization = small_checkpoints[("full_gaussian", 9000)]
    model = PPO.load(path, device="cpu")
    metadata = model_metadata("full_gaussian", 9000, path, initialization)
    metadata["ppo"]["gamma"] = 0.5
    with pytest.raises(ValueError):
        evaluation.validate_policy(metadata, model, path)


def test_changed_policy_architecture_is_not_hidden_by_model_metadata(
    small_checkpoints, small_budget
):
    path, initialization = small_checkpoints[("full_gaussian", 9000)]
    model = PPO.load(path, device="cpu")
    metadata = model_metadata("full_gaussian", 9000, path, initialization)
    model.policy.net_arch = [32, 32]
    with pytest.raises(ValueError, match="PPO|architecture"):
        evaluation.validate_policy(metadata, model, path)


def test_gsde_is_not_the_registered_raw_normal_parameterization(tmp_path, small_budget):
    env = DummyVecEnv([lambda: SmallUpdateEnv(2) for _ in range(4)])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = PPO(
            "MlpPolicy", env, **study.shared.PPO_SETTINGS, use_sde=True, seed=9000, device="cpu"
        )
        model.learn(512)
        path = tmp_path / "gsde.zip"
        model.save(path)
        metadata = model_metadata("full_gaussian", 9000, path, {})
        loaded = PPO.load(path, device="cpu")
        # The policy class and action dimension alone cannot distinguish gSDE.
        assert type(loaded.policy) is evaluation.ActorCriticPolicy
        with pytest.raises(ValueError, match="distribution|gSDE|Gaussian|sde"):
            evaluation.validate_policy(metadata, loaded, path)
    finally:
        env.close()
        torch.set_num_threads(old_threads)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def refresh_manifest(root):
    write_json(
        root / "manifest.json",
        {
            str(path.relative_to(root)): study.shared._sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != root / "manifest.json"
        },
    )


@pytest.fixture
def training_roots(tmp_path, small_checkpoints, small_budget):
    # Snapshot an actual project source. These miniature fixtures are not formal
    # training evidence; their purpose is testing the persisted-input boundary.
    relative = "src/wheel_legged_control/d1/ppo_action_policies.py"
    payload = (study.ROOT / relative).read_bytes()
    sources = {relative: hashlib.sha256(payload).hexdigest()}
    roots = []
    for seed in evaluation.TRAINING_SEEDS:
        root = tmp_path / f"train_{seed}"
        root.mkdir()
        with tarfile.open(root / "runtime_source.tar.gz", "w:gz") as snapshot:
            entry = tarfile.TarInfo(relative)
            entry.size = len(payload)
            snapshot.addfile(entry, io.BytesIO(payload))
        records = []
        for variant in study.VARIANTS:
            path, initialization = small_checkpoints[(variant, seed)]
            directory = root / f"{variant}_seed_{seed}"
            directory.mkdir()
            (directory / "model.zip").write_bytes(path.read_bytes())
            metadata = model_metadata(variant, seed, path, initialization)
            metadata["source_sha256"] = sources
            write_json(directory / "metadata.json", metadata)
            records.append(metadata)
        write_json(
            root / "protocol.json",
            {
                "variants": list(study.VARIANTS),
                "seeds": [seed],
                "budgets": list(getattr(evaluation, "FINAL_BUDGETS", (32768, 65536, 512))),
                "envs": 4,
                "ppo": copy.deepcopy(study.shared.PPO_SETTINGS),
                "training_reward_scale": 0.01,
                "training_mode": "mixed",
                "episode_seconds": 4.0,
                "development_cases": study.shared.evaluation_cases("development", "tracking-v2"),
                "intermediate_development_indices": list(study.INTERMEDIATE_CASES),
                "new_holdout_cases": study.frozen_holdout_cases(),
                "quality_criteria": study.shared.TRACKING_QUALITY_CRITERIA,
                "source_sha256": sources,
                "development_rng_isolation": "restore Python, numpy and torch CPU streams after checkpoint load and evaluation",
            },
        )
        write_json(
            root / "summary.json",
            {
                "training_runs": 3,
                "source_unchanged_during_run": True,
                "models": records,
            },
        )
        refresh_manifest(root)
        roots.append(root)
    return roots


def test_complete_cohort_is_sorted_by_real_seed_and_variant(training_roots):
    inputs, sources = evaluation.read_training_inputs(training_roots[::-1])
    assert [(meta["training_seed"], meta["variant"]) for _, meta in inputs] == [
        (seed, variant)
        for seed in (9000, 10000, 11000)
        for variant in ("bounded_mean", "full_gaussian", "fx_only")
    ]
    assert len(sources) == 1


def test_frozen_holdout_has_the_predeclared_twenty_four_new_cases():
    cases = study.frozen_holdout_cases()
    assert len(cases) == 24
    assert len({case["case_id"] for case in cases}) == 24
    assert [case["environment_seed"] for case in cases] == list(range(55000, 55024))
    assert [
        sum(case["terrain"]["kind"] == kind for case in cases) for kind in ("flat", "bumps", "ramp")
    ] == [4, 12, 8]
    bumps = [case["terrain"] for case in cases if case["terrain"]["kind"] == "bumps"]
    ramps = [case["terrain"] for case in cases if case["terrain"]["kind"] == "ramp"]
    assert {terrain["wavelength_m"] for terrain in bumps} == {0.85, 1.05, 1.25}
    assert {terrain["phase_rad"] for terrain in bumps} == {0.91, 2.61}
    assert {terrain["slope_deg"] for terrain in ramps} == {-3.75, -2.75, 2.75, 3.75}


@pytest.mark.parametrize("omission", ["empty", "metadata", "model"])
def test_incomplete_artifact_manifest_cannot_open_holdout(training_roots, omission):
    root = training_roots[0]
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if omission == "empty":
        manifest = {}
    else:
        filename = "metadata.json" if omission == "metadata" else "model.zip"
        manifest.pop(f"full_gaussian_seed_9000/{filename}")
    write_json(path, manifest)
    with pytest.raises(ValueError, match="manifest|artifact"):
        evaluation.read_training_inputs(training_roots)


def test_extra_unmanifested_file_cannot_be_silently_ignored(training_roots):
    (training_roots[0] / "unrecorded_model.zip").write_bytes(b"not in manifest")
    with pytest.raises(ValueError, match="manifest|artifact"):
        evaluation.read_training_inputs(training_roots)


def test_manifest_cannot_include_itself(training_roots):
    root = training_roots[0]
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["manifest.json"] = study.shared._sha256(path)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="manifest|artifact"):
        evaluation.read_training_inputs(training_roots)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", [9001]),
        ("envs", 2),
        ("training_reward_scale", 1.0),
        ("training_mode", "flat"),
        ("episode_seconds", 8.0),
        ("development_cases", []),
        ("intermediate_development_indices", [0]),
        ("variants", ["full_gaussian", "fx_only"]),
        ("ppo", {**study.shared.PPO_SETTINGS, "gamma": 0.5}),
        ("development_rng_isolation", "not isolated"),
    ],
)
def test_changed_training_protocol_is_rejected_even_with_updated_artifact_hashes(
    training_roots, field, value
):
    root = training_roots[0]
    path = root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol[field] = value
    write_json(path, protocol)
    refresh_manifest(root)
    with pytest.raises(ValueError):
        evaluation.read_training_inputs(training_roots)


def test_holdout_terrain_parameters_cannot_be_changed_after_training(training_roots):
    root = training_roots[0]
    path = root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["new_holdout_cases"][4]["terrain"]["phase_rad"] += 0.01
    write_json(path, protocol)
    refresh_manifest(root)
    with pytest.raises(ValueError, match="holdout"):
        evaluation.read_training_inputs(training_roots)


def test_duplicate_root_does_not_count_as_an_independent_training_seed(training_roots):
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.read_training_inputs([training_roots[0], *training_roots])


def test_two_completed_training_seeds_do_not_open_the_holdout(training_roots):
    with pytest.raises(ValueError, match="nine|all|seed"):
        evaluation.read_training_inputs(training_roots[:2])


def test_summary_cannot_claim_a_different_number_of_training_runs(training_roots):
    root = training_roots[0]
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary["training_runs"] = 2
    write_json(path, summary)
    refresh_manifest(root)
    with pytest.raises(ValueError, match="training_runs|summary"):
        evaluation.read_training_inputs(training_roots)


@pytest.mark.parametrize(
    "field", ["common_two_action_parameter_sha256", "initialized_parameter_sha256"]
)
def test_variants_cannot_claim_different_common_initializations(training_roots, field):
    root = training_roots[0]
    path = root / "bounded_mean_seed_9000/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["initialization"][field] = "0" * 64
    write_json(path, metadata)
    summary = json.loads((root / "summary.json").read_text())
    summary["models"][1] = metadata
    write_json(root / "summary.json", summary)
    refresh_manifest(root)
    with pytest.raises(ValueError, match="initialization"):
        evaluation.read_training_inputs(training_roots)


def test_copied_checkpoint_does_not_become_an_independent_seed(training_roots):
    source = training_roots[0] / "full_gaussian_seed_9000/model.zip"
    root = training_roots[1]
    directory = root / "full_gaussian_seed_10000"
    (directory / "model.zip").write_bytes(source.read_bytes())
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["model_sha256"] = study.shared._sha256(source)
    write_json(directory / "metadata.json", metadata)
    summary = json.loads((root / "summary.json").read_text())
    summary["models"][0] = metadata
    write_json(root / "summary.json", summary)
    refresh_manifest(root)
    with pytest.raises(ValueError, match="duplicate|copied|independent"):
        evaluation.read_training_inputs(training_roots)


def test_source_snapshot_is_checked_even_if_container_hash_is_updated(training_roots):
    root = training_roots[0]
    relative = "src/wheel_legged_control/d1/ppo_action_policies.py"
    with tarfile.open(root / "runtime_source.tar.gz", "w:gz") as snapshot:
        payload = b"different source version"
        entry = tarfile.TarInfo(relative)
        entry.size = len(payload)
        snapshot.addfile(entry, io.BytesIO(payload))
    refresh_manifest(root)
    with pytest.raises(ValueError, match="snapshot"):
        evaluation.read_training_inputs(training_roots)


def test_model_bytes_are_checked_before_loading(training_roots):
    path = training_roots[0] / "full_gaussian_seed_9000/model.zip"
    path.write_bytes(path.read_bytes() + b"unexpected appended bytes")
    with pytest.raises(ValueError, match="artifact"):
        evaluation.read_training_inputs(training_roots)


def score_row(variant, seed, case_id, value):
    return {
        "variant": variant,
        "training_seed": seed,
        "case_id": case_id,
        "completed": True,
        "quality_success": value < 10,
        "velocity_rmse_mps": value,
        "tail_velocity_rmse_mps": value / 2,
        "clearance_rmse_m": value / 100,
        "episode_return": 100 - value,
    }


def test_paired_statistics_use_same_training_seed_and_count_baseline_once():
    rows = [score_row("zero_residual", None, f"case_{i}", 1) for i in range(24)]
    for seed, base in ((9000, 2), (10000, 5), (11000, 12)):
        for variant, delta in (("full_gaussian", 0), ("bounded_mean", 2), ("fx_only", -1)):
            for index in range(24):
                rows.append(score_row(variant, seed, f"case_{index}", base + delta))
    # Input order should not determine the seed pairing.
    result = evaluation.summarize_controllers(rows[::-1])
    baseline = [row for row in result["controllers"] if row["variant"] == "zero_residual"]
    assert len(baseline) == 1 and baseline[0]["cases"] == 24
    assert baseline[0]["training_seed"] is None
    pairs = result["paired_seed_differences"]
    assert len(pairs) == 6
    for row in pairs:
        expected = 2 if row["variant"] == "bounded_mean" else -1
        assert row["delta_mean_velocity_rmse_mps"] == expected
        assert row["delta_mean_episode_return"] == -expected
        assert row["delta_mean_tail_velocity_rmse_mps"] == expected / 2
        assert row["delta_mean_clearance_rmse_m"] == pytest.approx(expected / 100)


@pytest.mark.parametrize(
    "corruption", ["duplicate", "missing", "different_case", "seeded_baseline"]
)
def test_paired_statistics_reject_incomplete_or_pseudoreplicated_cases(corruption):
    rows = [score_row("zero_residual", None, f"case_{i}", 1) for i in range(24)]
    for seed in evaluation.TRAINING_SEEDS:
        for variant in study.VARIANTS:
            rows.extend(score_row(variant, seed, f"case_{i}", 1) for i in range(24))
    if corruption == "duplicate":
        rows.append(rows[-1].copy())
    elif corruption == "missing":
        rows.pop()
    elif corruption == "different_case":
        rows[-1]["case_id"] = "not_the_same_case"
    else:
        rows[0]["training_seed"] = 9000
    with pytest.raises(ValueError, match="case|baseline|controller"):
        evaluation.summarize_controllers(rows)


def test_last_candidate_must_pass_before_any_holdout_episode(training_roots, monkeypatch, tmp_path):
    root = training_roots[-1]
    directory = root / "fx_only_seed_11000"
    path = directory / "model.zip"
    incomplete = PPO.load(path, device="cpu")
    incomplete._n_updates = 3
    incomplete.save(path)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["model_sha256"] = study.shared._sha256(path)
    write_json(directory / "metadata.json", metadata)
    summary = json.loads((root / "summary.json").read_text())
    summary["models"][-1] = metadata
    write_json(root / "summary.json", summary)
    refresh_manifest(root)

    def forbid_simulation(*args, **kwargs):
        pytest.fail("holdout opened before validating every candidate")

    monkeypatch.setattr(study, "evaluate", forbid_simulation)
    output = tmp_path / "must_not_open"
    with pytest.raises(ValueError, match="PPO updates"):
        evaluation.main(
            [
                "--training-roots",
                *(str(path) for path in training_roots),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_reserialized_duplicate_policy_cannot_be_an_independent_seed(
    training_roots, monkeypatch, tmp_path
):
    source = PPO.load(training_roots[0] / "full_gaussian_seed_9000/model.zip", device="cpu")
    root = training_roots[1]
    directory = root / "full_gaussian_seed_10000"
    path = directory / "model.zip"
    clone = PPO.load(path, device="cpu")
    clone.policy.load_state_dict(source.policy.state_dict())
    assert clone.seed == 10000
    clone.save(path)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["model_sha256"] = study.shared._sha256(path)
    write_json(directory / "metadata.json", metadata)
    summary = json.loads((root / "summary.json").read_text())
    summary["models"][0] = metadata
    write_json(root / "summary.json", summary)
    refresh_manifest(root)

    def forbid_simulation(*args, **kwargs):
        pytest.fail("copied policy opened the holdout")

    monkeypatch.setattr(study, "evaluate", forbid_simulation)
    output = tmp_path / "copied_policy_must_not_open"
    with pytest.raises(ValueError, match="copied policy parameters"):
        evaluation.main(
            [
                "--training-roots",
                *(str(path) for path in training_roots),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_complete_entrypoint_runs_one_baseline_and_nine_policies_on_identical_cases(
    training_roots, monkeypatch, tmp_path
):
    # This substitutes only the expensive simulation boundary. Provenance
    # checks, checkpoint loading and every policy validation remain real.
    calls = []

    def record_simulation(model, variant, cases, output):
        seed = None if model is None else model.seed
        calls.append((variant, seed, tuple(case["case_id"] for case in cases)))
        rows = []
        for case in cases:
            row = score_row(variant, seed, case["case_id"], 1.0)
            del row["training_seed"]
            rows.append(row)
        return rows

    monkeypatch.setattr(study, "evaluate", record_simulation)
    output = tmp_path / "orchestration_only"
    evaluation.main(
        [
            "--training-roots",
            *(str(path) for path in training_roots),
            "--output",
            str(output),
        ]
    )
    assert len(calls) == 10
    assert calls[0][:2] == ("zero_residual", None)
    assert sum(variant == "zero_residual" for variant, _, _ in calls) == 1
    assert all(len(cases) == 24 and cases == calls[0][2] for _, _, cases in calls)
    result = json.loads((output / "summary.json").read_text())
    assert result["episodes"] == 240
    assert len(result["controllers"]) == 10
    assert len(result["paired_seed_differences"]) == 6
    protocol = json.loads((output / "protocol.json").read_text())
    assert len(protocol["inputs"]) == 9
    assert "not three replicates" in protocol["baseline_replication"]
    assert all(Path(item["directory"]).is_relative_to(tmp_path) for item in protocol["inputs"])
