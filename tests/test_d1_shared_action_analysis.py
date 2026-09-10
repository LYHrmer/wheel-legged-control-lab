"""Hand-built arithmetic fixtures, not trained policies or formal study evidence."""

import hashlib
import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import analyze_d1_shared_actions as analysis

GAINS = {
    "wheel_kp": 0.55,
    "wheel_ki": 1.5,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 0.25,
}
METRICS = (
    "velocity_rmse_mps",
    "yaw_rmse_rps",
    "height_rmse_m",
    "attitude_rmse_rad",
    "mean_mechanical_power_w",
    "action_rms",
)


def case(name="road0_seed1017", *, complete=True, quality=True, duration=60.0, values=None):
    """Explicit summarized records for pure arithmetic and pairing tests only."""
    metrics = (0.03, 0.04, 0.01, 0.05, 10.0, 0.2) if values is None else values
    return {
        "case": name,
        "seed": 1017,
        "measurement_seed": 871533,
        "terrain": {"layout": "ramps", "slope_deg": 2.0},
        "schedule": {"duration_s": 60.0, "segments": [{"vx": 0.2, "yaw": 0.1}]},
        "controller_parameters": dict(GAINS),
        "completed": complete,
        "quality_pass": complete and quality,
        "executed_steps": round(duration / 0.01),
        "duration_s": duration,
        "terminal_reason": "time_limit" if complete else "fall_or_body_contact",
        "terrain_exposure": {"nonflat_fraction": 0.3},
        **dict(zip(METRICS, metrics, strict=True)),
    }


def arguments(*, mode="shared2", smoke=False):
    return {
        "baseline": "wheel_leg",
        "action_mode": mode,
        "source": "imu_encoder_fusion",
        "history": 1,
        "duration": 0.2 if smoke else 60.0,
        "steps": 512 if smoke else 32768,
        "workers": 4,
        "delay_randomization": False,
        "measurement_delay": 0,
        "terrain_suite": "action_compare_v1",
        "seed": 31001 if smoke else 31000,
        **GAINS,
    }


def check_arguments(record, *, mode="shared2", smoke=False):
    return analysis.common_arguments(
        record,
        duration=0.2 if smoke else 60.0,
        steps=512 if smoke else 32768,
        mode=mode,
        seed=31001 if smoke else 31000,
    )


def test_completed_pair_uses_right_minus_left_without_mutating_inputs():
    left = case()
    right = case(values=(0.02, 0.03, 0.015, 0.07, 14.0, 0.1))
    original = deepcopy((left, right))
    result = analysis.compare_cases([left], [right])
    assert (left, right) == original
    assert len(result) == 1
    pair = result[0]
    assert pair["both_completed"] is True
    assert pair["left_completed"] is True and pair["right_completed"] is True
    assert pair["left_duration_s"] == pair["right_duration_s"] == 60.0
    assert pair["right_minus_left"] == pytest.approx(
        {
            "velocity_rmse_mps": -0.01,
            "yaw_rmse_rps": -0.01,
            "height_rmse_m": 0.005,
            "attitude_rmse_rad": 0.02,
            "mean_mechanical_power_w": 4.0,
            "action_rms": -0.1,
        }
    )


@pytest.mark.parametrize(
    "left_complete,right_complete", [(False, True), (True, False), (False, False)]
)
def test_failed_prefix_cannot_receive_a_continuous_performance_delta(left_complete, right_complete):
    left = case(complete=left_complete, duration=60.0 if left_complete else 2.0)
    right = case(
        complete=right_complete,
        duration=60.0 if right_complete else 0.03,
        values=(0.001, 0.001, 0.001, 0.001, 1.0, 0.001),
    )
    pair = analysis.compare_cases([left], [right])[0]
    assert pair["both_completed"] is False
    assert pair["right_minus_left"] is None
    assert pair["left_completed"] is left_complete
    assert pair["right_completed"] is right_complete
    assert pair["left_duration_s"] == left["duration_s"]
    assert pair["right_duration_s"] == right["duration_s"]
    assert pair["left_terminal_reason"] == left["terminal_reason"]
    assert pair["right_terminal_reason"] == right["terminal_reason"]


def test_completed_quality_failure_remains_a_full_horizon_pair():
    pair = analysis.compare_cases([case(quality=False)], [case()])[0]
    assert pair["both_completed"] is True
    assert pair["right_minus_left"] == dict.fromkeys(METRICS, 0.0)


def test_completed_different_lengths_reject_even_if_case_labels_match():
    with pytest.raises(ValueError, match="horizons differ"):
        analysis.compare_cases([case()], [case(duration=59.99)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 1029),
        ("measurement_seed", 1017),  # Reset seed is not the sampled sensor seed.
        ("terrain", {"layout": "ramps", "slope_deg": 2.1}),
        ("schedule", {"duration_s": 60.0, "segments": [{"vx": 0.3, "yaw": 0.1}]}),
        ("controller_parameters", {**GAINS, "attitude_feedback_scale": 1.0}),
    ],
)
@pytest.mark.parametrize("complete", [True, False])
def test_pairing_rejects_changed_episode_conditions_even_for_failures(field, value, complete):
    left = case(complete=complete, duration=60.0 if complete else 1.0)
    right = deepcopy(left)
    right[field] = value
    with pytest.raises(ValueError, match=field):
        analysis.compare_cases([left], [right])


@pytest.mark.parametrize(
    "left,right",
    [
        ([case(), case()], [case()]),
        ([case()], [case(), case()]),
        ([case()], []),
        ([case()], [case("road1_seed1017")]),
    ],
)
def test_pairing_rejects_duplicates_and_missing_or_relabelled_cases(left, right):
    with pytest.raises(ValueError, match="paired case"):
        analysis.compare_cases(left, right)


def test_pairs_are_matched_by_label_not_input_order():
    first = case("road0_seed1017")
    second = case("road1_seed1017", values=(0.06, 0.08, 0.02, 0.1, 20.0, 0.4))
    paired = analysis.compare_cases([second, first], [first, second])
    assert [row["case"] for row in paired] == ["road0_seed1017", "road1_seed1017"]
    assert all(row["right_minus_left"] == dict.fromkeys(METRICS, 0.0) for row in paired)


def test_summary_excludes_failed_prefix_from_means_but_preserves_its_duration():
    first = case()
    second = case("road1_seed1017", quality=False, values=(0.09, 0.12, 0.03, 0.15, 30.0, 0.6))
    failed = case("road0_seed1029", complete=False, duration=0.02, values=(0.0,) * 6)
    failed["terrain_exposure"]["nonflat_fraction"] = 0.0
    summary = analysis.summarize_cases([first, second, failed])
    assert summary["cases"] == 3
    assert summary["completed"] == 2
    assert summary["cases"] - summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["terminal_reason_counts"] == {"time_limit": 2, "fall_or_body_contact": 1}
    assert summary["quality_passes"] == 1
    assert summary["completed_only_means"] == pytest.approx(
        dict(zip(METRICS, (0.06, 0.08, 0.02, 0.1, 20.0, 0.4), strict=True))
    )
    assert summary["all_case_duration_s"] == [60.0, 60.0, 0.02]
    assert summary["all_case_nonflat_fraction"] == [0.3, 0.3, 0.0]


@pytest.mark.parametrize("count", [0, 1, 4])
def test_no_complete_episode_has_no_completed_only_mean(count):
    rows = [case(f"arithmetic_case{i}", complete=False, duration=0.02) for i in range(count)]
    summary = analysis.summarize_cases(rows)
    assert summary["cases"] == count
    assert summary["completed"] == summary["quality_passes"] == 0
    assert summary["failed"] == count
    assert summary["terminal_reason_counts"] == ({"fall_or_body_contact": count} if count else {})
    assert summary["completed_only_means"] is None
    assert summary["all_case_duration_s"] == [0.02] * count


@pytest.mark.parametrize("mode", ["shared2", "independent8"])
@pytest.mark.parametrize("smoke", [False, True])
def test_declared_formal_and_smoke_arguments_are_accepted(mode, smoke):
    assert check_arguments(arguments(mode=mode, smoke=smoke), mode=mode, smoke=smoke) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline", "lqr"),
        ("action_mode", "independent8"),
        ("source", "oracle"),
        ("history", True),
        ("history", 1.0),
        ("duration", 60),
        ("duration", 59.99),
        ("duration", float("nan")),
        ("duration", float("inf")),
        ("steps", 32768.0),
        ("steps", True),
        ("steps", 32767),
        ("steps", 33280),
        ("workers", 4.0),
        ("workers", True),
        ("workers", 1),
        ("delay_randomization", 0),
        ("delay_randomization", True),
        ("measurement_delay", False),
        ("measurement_delay", 0.0),
        ("measurement_delay", 1),
        ("terrain_suite", "default"),
        ("seed", 31000.0),
        ("seed", True),
        ("seed", 31001),
    ],
)
def test_common_arguments_rejects_type_coercion_or_different_protocol(field, value):
    record = arguments()
    record[field] = value
    with pytest.raises(ValueError, match=field):
        check_arguments(record)


@pytest.mark.parametrize("field", tuple(arguments()))
def test_common_arguments_requires_every_declared_field(field):
    record = arguments()
    record.pop(field)
    with pytest.raises(ValueError):
        check_arguments(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wheel_kp", 0.56),
        ("wheel_ki", 3.0),
        ("yaw_feedback_gain", 0.0),
        ("leg_feedback_scale", 0.0),
        ("leg_feedback_scale", True),
        ("attitude_feedback_scale", 1.0),
        ("wheel_kp", float("inf")),
    ],
)
def test_common_arguments_rejects_changed_or_malformed_feedback_gains(field, value):
    record = arguments()
    record[field] = value
    with pytest.raises(ValueError):
        check_arguments(record)


def test_smoke_evaluation_cannot_silently_keep_formal_default_budget():
    record = arguments(smoke=True)
    record.update(mode="evaluate", steps=32768)
    with pytest.raises(ValueError, match="steps"):
        check_arguments(record, smoke=True)


def write_json(path, document):
    path.write_text(json.dumps(document) + "\n")


@pytest.fixture
def parent_gate_fixture(tmp_path):
    """Only parent headers: invalid inputs must stop before nonexistent runs."""
    names = [
        "shared2_seed31001",
        "shared2_seed31001_development",
        "shared2_seed31001_holdout",
        "independent8_seed31001",
        "independent8_seed31001_development",
        "independent8_seed31001_holdout",
        "zero_development",
        "zero_holdout",
    ]
    write_json(
        tmp_path / "protocol.json",
        {
            "schema": "d1-shared-action-study-v1",
            "kind": "smoke",
            "training_seeds": [31001],
            "action_modes": ["shared2", "independent8"],
            "timesteps_per_model": 512,
            "duration_s": 0.2,
            "runs": [{"name": name} for name in names],
        },
    )
    write_json(
        tmp_path / "summary.json",
        {
            "status": "commands_completed",
            "commands_completed": 8,
            "planned_commands": 8,
        },
    )
    hashes = {"src/arithmetic_fixture.py": "a" * 64, "scripts/study_fixture.py": "b" * 64}
    write_json(tmp_path / "source.json", {"sha256": hashes})
    write_json(tmp_path / "source_consistency.json", {"unchanged": True, "sha256": hashes})
    return tmp_path


@pytest.mark.parametrize("field", ["commands_completed", "planned_commands"])
@pytest.mark.parametrize("value", [8.0, True, 7])
def test_parent_command_counts_are_exact_integers(parent_gate_fixture, field, value):
    path = parent_gate_fixture / "summary.json"
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(ValueError, match="parent command count"):
        analysis.analyze(parent_gate_fixture)


@pytest.mark.parametrize("mutation", ["numeric_boolean", "changed_digest"])
def test_parent_source_gate_rejects_nonboolean_or_changed_inventory(parent_gate_fixture, mutation):
    path = parent_gate_fixture / "source_consistency.json"
    record = json.loads(path.read_text())
    if mutation == "numeric_boolean":
        record["unchanged"] = 1
    else:
        record["sha256"]["src/arithmetic_fixture.py"] = "c" * 64
    write_json(path, record)
    with pytest.raises(ValueError, match="parent source consistency"):
        analysis.analyze(parent_gate_fixture)


@pytest.mark.parametrize("unknown_path", [False, True])
def test_child_source_must_belong_to_unchanged_parent_inventory(parent_gate_fixture, unknown_path):
    child = parent_gate_fixture / "shared2_seed31001"
    child.mkdir()
    write_json(
        child / "protocol.json",
        {
            "arguments": {**arguments(smoke=True), "mode": "train", "split": "development"},
            "evaluation_seeds": {"development": [1017, 1029], "holdout": [1617, 1629]},
        },
    )
    hashes = (
        {"src/unknown.py": "a" * 64} if unknown_path else {"src/arithmetic_fixture.py": "c" * 64}
    )
    write_json(child / "source.json", {"sha256": hashes})
    with pytest.raises(ValueError, match="child source differs from parent inventory"):
        analysis.analyze(parent_gate_fixture)


@pytest.fixture
def checkpoint_gate_fixture(tmp_path):
    """Real archive/hash plumbing with labelled inert bytes, never an SB3 model.

    No worker logs or PPO samples are fabricated: malformed checkpoint headers
    must fail before those downstream files would be read.
    """
    source_bytes = b"# Hand-built source-audit fixture; not an experiment\n"
    source_name = "src/arithmetic_fixture.py"
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(source_name)
        info.size = len(source_bytes)
        archive.addfile(info, io.BytesIO(source_bytes))
    write_json(
        tmp_path / "source.json",
        {
            "sha256": {source_name: hashlib.sha256(source_bytes).hexdigest()},
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        },
    )
    write_json(tmp_path / "source_consistency.json", {"unchanged": True, "changed": []})
    model_bytes = b"inert checkpoint gate fixture, not a trained model"
    (tmp_path / "checkpoint.zip").write_bytes(model_bytes)
    digest = hashlib.sha256(model_bytes).hexdigest()
    write_json(
        tmp_path / "training.json",
        {
            "num_timesteps": 512,
            "checkpoint_sha256": digest,
            "policy_parameter_count": 123,
        },
    )
    write_json(
        tmp_path / "checkpoint.json",
        {
            "extra": {"num_timesteps": 512, "training_seed": 31001},
            "model_sha256": digest,
            "observation_dim": 82,
            "history_length": 1,
            "action_dim": 2,
            "action_mode": "shared2",
            "policy_action_size": 2,
            "physical_action_size": 8,
            "physical_action_schema": "d1-wheel-leg-extension-speed-v1",
            "action_schema": "d1-shared-wheel-leg-extension-speed-v1",
            "policy_to_physical_indices": [0, 0, 0, 0, 1, 1, 1, 1],
        },
    )
    return tmp_path


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("extra.num_timesteps", 512.0, "checkpoint num_timesteps"),
        ("extra.training_seed", 31001.0, "checkpoint training_seed"),
        ("observation_dim", 82.0, "observation/history"),
        ("history_length", True, "observation/history"),
        ("history_length", 1.0, "observation/history"),
        ("action_dim", 2.0, "policy dimension"),
    ],
)
def test_checkpoint_header_integer_fields_cannot_be_float_or_bool(
    checkpoint_gate_fixture, field, value, message
):
    path = checkpoint_gate_fixture / "checkpoint.json"
    record = json.loads(path.read_text())
    if field.startswith("extra."):
        record["extra"][field.split(".")[1]] = value
    else:
        record[field] = value
    write_json(path, record)
    with pytest.raises(ValueError, match=message):
        analysis.audit_training(checkpoint_gate_fixture, arguments(smoke=True), 512, [])


@pytest.mark.parametrize("value", [True, 123.0, 0, -1])
def test_parameter_count_is_a_strict_positive_integer(checkpoint_gate_fixture, value):
    path = checkpoint_gate_fixture / "training.json"
    record = json.loads(path.read_text())
    record["policy_parameter_count"] = value
    write_json(path, record)
    with pytest.raises(ValueError, match="invalid policy parameter count"):
        analysis.audit_training(checkpoint_gate_fixture, arguments(smoke=True), 512, [])


def recorded_model_paths(tmp_path, mode="shared2"):
    """Original serialized paths are intentionally absent on this machine."""
    original = tmp_path / "unavailable_machine" / "study" / f"{mode}_seed31000"
    assert not original.exists()
    training = {"output": str(original), "action_mode": mode, "seed": 31000}
    evaluation = {
        "policy": str(original / "checkpoint.zip"),
        "metadata": str(original / "checkpoint.json"),
        "action_mode": mode,
        "seed": 31000,
        "output": str(tmp_path / "downloaded_elsewhere" / f"{mode}_seed31000_development"),
    }
    return evaluation, training


@pytest.mark.parametrize("mode", ["shared2", "independent8"])
def test_original_model_references_remain_valid_after_study_relocation(tmp_path, mode):
    evaluation, training = recorded_model_paths(tmp_path, mode)
    before = deepcopy((evaluation, training))
    assert analysis.audit_policy_reference(evaluation, training) is None
    assert (evaluation, training) == before


@pytest.mark.parametrize("field", ["policy", "metadata"])
@pytest.mark.parametrize("replacement", ["shared2_seed32000", "independent8_seed31000"])
def test_relocated_policy_must_not_reference_another_seed_or_action_mode(
    tmp_path, field, replacement
):
    evaluation, training = recorded_model_paths(tmp_path)
    evaluation[field] = evaluation[field].replace("shared2_seed31000", replacement)
    with pytest.raises(ValueError):
        analysis.audit_policy_reference(evaluation, training)


@pytest.mark.parametrize("field,filename", [("policy", "best.zip"), ("metadata", "other.json")])
def test_relocated_model_reference_keeps_exact_checkpoint_filenames(tmp_path, field, filename):
    evaluation, training = recorded_model_paths(tmp_path)
    evaluation[field] = training["output"] + "/" + filename
    with pytest.raises(ValueError):
        analysis.audit_policy_reference(evaluation, training)


@pytest.mark.parametrize("field", ["policy", "metadata", "training_output"])
def test_recorded_paths_reject_parent_traversal_even_if_normalization_would_match(tmp_path, field):
    evaluation, training = recorded_model_paths(tmp_path)
    if field == "training_output":
        training["output"] += "/unused/.."
        evaluation["policy"] = training["output"] + "/checkpoint.zip"
        evaluation["metadata"] = training["output"] + "/checkpoint.json"
    else:
        basename = "checkpoint.zip" if field == "policy" else "checkpoint.json"
        evaluation[field] = training["output"] + "/unused/../" + basename
    with pytest.raises(ValueError):
        analysis.audit_policy_reference(evaluation, training)


@pytest.mark.parametrize("field", ["policy", "metadata", "training_output"])
@pytest.mark.parametrize("value", [None, "", False, "relative/checkpoint.zip"])
def test_recorded_model_paths_must_be_nonempty_absolute_posix_strings(tmp_path, field, value):
    evaluation, training = recorded_model_paths(tmp_path)
    if field == "training_output":
        training["output"] = value
    else:
        evaluation[field] = value
    with pytest.raises(ValueError):
        analysis.audit_policy_reference(evaluation, training)


@pytest.mark.parametrize(
    "record,field,value",
    [
        ("evaluation", "seed", 32000),
        ("evaluation", "seed", 31000.0),
        ("training", "seed", 31000.0),
        ("training", "seed", True),
        ("evaluation", "action_mode", "independent8"),
        ("training", "action_mode", "independent8"),
        ("evaluation", "action_mode", "legacy_force2"),
        ("training", "action_mode", True),
    ],
)
def test_model_reference_arguments_must_name_the_same_typed_mode_and_seed(
    tmp_path, record, field, value
):
    evaluation, training = recorded_model_paths(tmp_path)
    (evaluation if record == "evaluation" else training)[field] = value
    with pytest.raises(ValueError):
        analysis.audit_policy_reference(evaluation, training)


def test_coherent_paths_still_require_the_mode_seed_training_directory_name(tmp_path):
    evaluation, training = recorded_model_paths(tmp_path)
    training["output"] = training["output"].replace("shared2_seed31000", "arbitrary_model")
    evaluation["policy"] = training["output"] + "/checkpoint.zip"
    evaluation["metadata"] = training["output"] + "/checkpoint.json"
    with pytest.raises(ValueError):
        analysis.audit_policy_reference(evaluation, training)


def test_serialized_model_reference_validation_never_resolves_or_reads_old_paths(
    tmp_path, monkeypatch
):
    evaluation, training = recorded_model_paths(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("serialized path validation must not access the old filesystem")

    with monkeypatch.context() as patcher:
        for name in ("resolve", "exists", "stat", "open"):
            patcher.setattr(Path, name, forbidden)
        assert analysis.audit_policy_reference(evaluation, training) is None
