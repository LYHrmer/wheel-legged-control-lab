"""Independent tensor fixtures: these are not trained models or performance evidence."""

import hashlib
import importlib.util
import io
import json
import stat
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/audit_d1_budget_weights.py"
spec = importlib.util.spec_from_file_location("weights_audit", SOURCE)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
MEMBERS = (
    "data",
    "pytorch_variables.pth",
    "policy.pth",
    "policy.optimizer.pth",
    "_stable_baselines3_version",
    "system_info.txt",
)


def state(size=2):
    shapes = {
        "log_std": (size,),
        "action_net.weight": (size, 64),
        "action_net.bias": (size,),
        "value_net.weight": (1, 64),
        "value_net.bias": (1,),
    }
    for branch in ("policy_net", "value_net"):
        for layer, incoming in ((0, 82), (2, 64)):
            shapes[f"mlp_extractor.{branch}.{layer}.weight"] = (64, incoming)
            shapes[f"mlp_extractor.{branch}.{layer}.bias"] = (64,)
    return OrderedDict(
        (name, torch.full(shape, index / 32, dtype=torch.float32))
        for index, (name, shape) in enumerate(shapes.items())
    )


def reference(value):
    digest = hashlib.sha256()
    for name in sorted(value):
        array = value[name].numpy()
        for part in (
            name.encode(),
            repr((array.dtype, array.shape)).encode(),
            np.array(array, order="C").tobytes(),
        ):
            digest.update(part)
    return digest.hexdigest()


def archive(path, value=None, members=MEMBERS, special=None):
    buffer = io.BytesIO()
    torch.save(state() if value is None else value, buffer)
    with zipfile.ZipFile(path, "w") as result:
        for name in members:
            item = special if special is not None and special.filename == name else name
            result.writestr(item, buffer.getvalue() if name == "policy.pth" else b"not parsed")
    return path


@pytest.mark.parametrize("size,count", [(2, 19141), (8, 19537)])
def test_actual_torch_weights_only_matches_independent_bytes(tmp_path, size, count):
    tensors = state(size)
    path = archive(tmp_path / "checkpoint.zip", tensors)
    before = path.read_bytes()
    result = tool.hash_checkpoint(path, size)
    assert result["parameter_sha256"] == reference(tensors)
    assert result["parameter_count"] == count
    assert result["zip_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_sorted_names_not_insertion_order(tmp_path):
    tensors = state()
    first = tool.hash_checkpoint(archive(tmp_path / "a.zip", tensors), 2)
    second = tool.hash_checkpoint(
        archive(tmp_path / "b.zip", OrderedDict(reversed(tensors.items()))), 2
    )
    assert first["parameter_sha256"] == second["parameter_sha256"]
    tensors["log_std"][0] += 0.25
    changed = tool.hash_checkpoint(archive(tmp_path / "c.zip", tensors), 2)
    assert first["parameter_sha256"] != changed["parameter_sha256"]


@pytest.mark.parametrize("bad_size", [True, False, 2.0, 0, 4, "2"])
def test_only_exact_declared_action_sizes(tmp_path, bad_size):
    with pytest.raises((TypeError, ValueError)):
        tool.hash_checkpoint(archive(tmp_path / "model.zip"), bad_size)


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "shape", "dtype", "nan", "infinite", "sparse", "nontensor"]
)
def test_actual_tensor_payload_rejections(tmp_path, kind):
    tensors = state()
    if kind == "missing":
        tensors.pop("log_std")
    elif kind == "extra":
        tensors["undeclared"] = torch.zeros(1)
    elif kind == "shape":
        tensors["log_std"] = torch.zeros(3)
    elif kind == "dtype":
        tensors["log_std"] = tensors["log_std"].double()
    elif kind in ("nan", "infinite"):
        tensors["log_std"][0] = float("nan" if kind == "nan" else "inf")
    elif kind == "sparse":
        tensors["log_std"] = tensors["log_std"].to_sparse()
    else:
        tensors["log_std"] = [0.0, 0.0]
    with pytest.raises((TypeError, ValueError)):
        tool.hash_checkpoint(archive(tmp_path / "bad.zip", tensors), 2)


@pytest.mark.parametrize(
    "kind",
    ["missing", "duplicate", "parent", "absolute", "backslash", "link", "directory", "corrupt"],
)
def test_project_zip_format_rejections(tmp_path, kind):
    path = tmp_path / "bad.zip"
    names, special = list(MEMBERS), None
    if kind == "missing":
        names.remove("policy.pth")
    elif kind == "duplicate":
        names.append("policy.pth")
    elif kind in ("parent", "absolute", "backslash"):
        names.append({"parent": "../x", "absolute": "/x", "backslash": "x\\y"}[kind])
    elif kind in ("link", "directory"):
        special = zipfile.ZipInfo("policy.pth")
        special.create_system = 3
        special.external_attr = ((stat.S_IFLNK if kind == "link" else stat.S_IFDIR) | 0o777) << 16
    if kind == "corrupt":
        path.write_bytes(b"not a ZIP")
    else:
        archive(path, members=names, special=special)
    with pytest.raises(ValueError):
        tool.hash_checkpoint(path, 2)


def test_input_symlink_is_not_followed(tmp_path):
    original = archive(tmp_path / "model.zip")
    linked = tmp_path / "linked.zip"
    linked.symlink_to(original)
    with pytest.raises(ValueError):
        tool.hash_checkpoint(linked, 2)


def test_policy_payload_only_is_deserialized(tmp_path, monkeypatch):
    calls = []
    load = torch.load

    def observe(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observe)
    tool.hash_checkpoint(archive(tmp_path / "model.zip"), 2)
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"


def test_hash_changes_are_not_equivalent_to_training_quality(tmp_path):
    # No environment, reward, optimizer or actor forward pass is needed by this reader.
    assert "parameter_sha256" in tool.hash_checkpoint(archive(tmp_path / "model.zip"), 2)


def dump(path, value):
    path.write_text(json.dumps(value))


def load(path):
    return json.loads(path.read_text())


def mutate(path, change):
    value = load(path)
    change(value)
    dump(path, value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "hand_fixture"
    root.mkdir()
    models = [
        {"model_id": f"{mode}_seed7", "action_mode": mode, "seed": 7}
        for mode in ("shared2", "independent8")
    ]
    dump(
        root / "protocol.json",
        {"schema": "d1-budget-study-v1", "models": models, "checkpoint_budgets": [512, 1024]},
    )
    dump(root / "summary.json", {"status": "commands_completed"})
    for model in models:
        directory = root / model["model_id"]
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "updates").mkdir()
        size = 2 if model["action_mode"] == "shared2" else 8
        saved, witnesses, updates = [], [], []
        for index, budget in enumerate((512, 1024)):
            folder = directory / "checkpoints" / f"budget{budget}"
            folder.mkdir()
            tensors = state(size)
            tensors["log_std"] += index / 8
            archive(folder / "checkpoint.zip", tensors)
            parameter = reference(tensors)
            extra = {
                "training_seed": 7,
                "num_timesteps": budget,
                "checkpoint_budget": budget,
                "after_update_index": index,
                "param_sha256_after": parameter,
            }
            metadata = {
                "schema": "d1-locomotion-checkpoint-v1",
                "action_dim": size,
                "action_mode": model["action_mode"],
                "observation_dim": 82,
                "history_length": 1,
                "model_sha256": sha(folder / "checkpoint.zip"),
                "extra": extra,
            }
            dump(folder / "checkpoint.json", metadata)
            common = {
                "budget": budget,
                "after_update_index": index,
                "zip_sha256": sha(folder / "checkpoint.zip"),
                "metadata_sha256": sha(folder / "checkpoint.json"),
            }
            saved.append(
                {
                    **common,
                    "num_timesteps": budget,
                    "param_sha256_after": parameter,
                    "model_path": f"budget{budget}/checkpoint.zip",
                    "metadata_path": f"budget{budget}/checkpoint.json",
                }
            )
            witnesses.append({**common, "reloaded_policy_param_sha256": parameter})
            updates.append(
                {"audit_index": index, "num_timesteps": budget, "param_sha256_after": parameter}
            )
        dump(
            directory / "checkpoints/manifest.json",
            {
                "schema": "d1-budget-checkpoints-v1",
                "complete": True,
                "failed": False,
                "failure": None,
                "pending_budgets": [],
                "budgets": [512, 1024],
                "saved": saved,
            },
        )
        (directory / "updates/updates.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in updates)
        )
        archive(directory / "checkpoint.zip", tensors)
        with zipfile.ZipFile(directory / "checkpoint.zip", "a") as zipped:
            zipped.comment = b"Final ZIP bytes differ, identical tensor values."
        final_meta = {
            **metadata,
            "model_sha256": sha(directory / "checkpoint.zip"),
            "extra": {"training_seed": 7, "num_timesteps": 1024},
        }
        dump(directory / "checkpoint.json", final_meta)
        final = {
            "model_path": "checkpoint.zip",
            "metadata_path": "checkpoint.json",
            "num_timesteps": 1024,
            "after_update_index": 1,
            "reloaded_policy_param_sha256": parameter,
            "zip_sha256": sha(directory / "checkpoint.zip"),
            "metadata_sha256": sha(directory / "checkpoint.json"),
        }
        dump(
            directory / "checkpoint_reload.json",
            {
                "schema": "d1-budget-checkpoint-reload-v1",
                "phase": "after_training",
                "training_final_num_timesteps": 1024,
                "records": witnesses,
                "final_checkpoint": final,
            },
        )
        dump(
            directory / "training.json",
            {"num_timesteps": 1024, "checkpoint_sha256": final["zip_sha256"]},
        )
    return root


def test_complete_hand_fixture_checks_budget_and_different_final_zip(study):
    before = {str(p.relative_to(study)): sha(p) for p in study.rglob("*") if p.is_file()}
    result = tool.audit(study)
    assert result["status"] == "weights_verified"
    assert (
        result["checkpoints_checked"],
        result["root_finals_checked"],
        result["total_zip_files_checked"],
    ) == (4, 2, 6)
    for model in result["models"]:
        assert (
            model["final_checkpoint"]["parameter_sha256"]
            == model["checkpoints"][-1]["parameter_sha256"]
        )
        assert model["final_checkpoint"]["zip_sha256"] != model["checkpoints"][-1]["zip_sha256"]
    assert before == {str(p.relative_to(study)): sha(p) for p in study.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "path,change",
    [
        ("protocol.json", lambda d: d["models"][0].update(model_id="../outside")),
        ("protocol.json", lambda d: d.update(checkpoint_budgets=[512, 512])),
        ("summary.json", lambda d: d.update(status="failed")),
        ("shared2_seed7/checkpoints/manifest.json", lambda d: d.update(complete=1)),
        ("shared2_seed7/checkpoints/manifest.json", lambda d: d["saved"].reverse()),
        (
            "shared2_seed7/checkpoints/manifest.json",
            lambda d: d["saved"][0].update(param_sha256_after="0" * 64),
        ),
        (
            "shared2_seed7/checkpoints/manifest.json",
            lambda d: d["saved"][0].update(zip_sha256="0" * 64),
        ),
        (
            "shared2_seed7/checkpoints/manifest.json",
            lambda d: d["saved"][0].update(metadata_sha256="0" * 64),
        ),
        ("shared2_seed7/checkpoint_reload.json", lambda d: d.update(phase="during_training")),
        (
            "shared2_seed7/checkpoint_reload.json",
            lambda d: d["records"][0].update(reloaded_policy_param_sha256="0" * 64),
        ),
        (
            "shared2_seed7/checkpoint_reload.json",
            lambda d: d["final_checkpoint"].update(reloaded_policy_param_sha256="0" * 64),
        ),
        ("shared2_seed7/training.json", lambda d: d.update(checkpoint_sha256="0" * 64)),
    ],
)
def test_study_cross_record_mismatches_fail(study, path, change):
    mutate(study / path, change)
    with pytest.raises(ValueError):
        tool.audit(study)


def test_selected_update_hash_is_independent_target(study):
    path = study / "shared2_seed7/updates/updates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["param_sha256_after"] = "0" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="selected update"):
        tool.audit(study)


def test_sidecar_seed_checked_after_resigning_metadata_hash(study):
    folder = study / "shared2_seed7"
    path = folder / "checkpoints/budget512/checkpoint.json"
    mutate(path, lambda d: d["extra"].update(training_seed=8))
    mutate(
        folder / "checkpoints/manifest.json",
        lambda d: d["saved"][0].update(metadata_sha256=sha(path)),
    )
    mutate(
        folder / "checkpoint_reload.json",
        lambda d: d["records"][0].update(metadata_sha256=sha(path)),
    )
    with pytest.raises(ValueError, match="training_seed"):
        tool.audit(study)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "[]"])
def test_strict_json_records(study, text):
    (study / "summary.json").write_text(text)
    with pytest.raises(ValueError):
        tool.audit(study)


@pytest.mark.parametrize("kind", ["exists", "overlap", "symlink", "bad_study"])
def test_cli_checks_before_output_creation(study, tmp_path, kind):
    output = tmp_path / "report"
    if kind == "exists":
        output.mkdir()
    elif kind == "overlap":
        output = study / "report"
    elif kind == "symlink":
        output.symlink_to(study, target_is_directory=True)
    else:
        dump(study / "failure.json", {"error": "interrupted"})
    with pytest.raises(ValueError):
        tool.main([str(study), "--output", str(output)])
    assert not (output / "report.json").exists()


def test_cli_report_snapshot_and_manifest(study, tmp_path):
    output = tmp_path / "report"
    result = tool.main([str(study), "--output", str(output)])
    assert result == load(output / "report.json")
    assert set(result["input_sha256"]) == {"protocol.json", "summary.json"} | {
        f"{mode}_seed7/{relative}"
        for mode in ("shared2", "independent8")
        for relative in (
            "updates/updates.jsonl",
            "checkpoints/manifest.json",
            "checkpoint_reload.json",
            "training.json",
        )
    }
    assert all(sha(study / name) == digest for name, digest in result["input_sha256"].items())
    hashes = load(output / "manifest.json")["sha256"]
    assert set(hashes) == {"report.json", SOURCE.name}
    assert all(sha(output / name) == digest for name, digest in hashes.items())
    assert not (output / ".partial").exists()


def test_failed_output_write_retains_partial_marker(study, tmp_path, monkeypatch):
    output = tmp_path / "report"
    original = Path.open

    def fail(path, *args, **kwargs):
        if path == output / "manifest.json.partial":
            raise OSError("injected disk failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError, match="injected"):
        tool.main([str(study), "--output", str(output)])
    assert (output / ".partial").is_file()
    assert not (output / "manifest.json").exists()


def test_failed_final_rename_cannot_leave_success_manifest(study, tmp_path, monkeypatch):
    output = tmp_path / "report"
    original = Path.rename

    def fail(path, target):
        if path == output / "manifest.json.partial":
            raise OSError("injected final rename failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="final rename"):
        tool.main([str(study), "--output", str(output)])
    assert (output / "manifest.json.partial").is_file()
    assert not (output / "manifest.json").exists()
