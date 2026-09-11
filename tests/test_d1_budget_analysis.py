"""Hand-built arithmetic records and opaque fake ZIPs, never training evidence."""

import ast
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_d1_budget_study as analysis
from scripts import run_d1_budget_study as runner


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load(path):
    return json.loads(path.read_text())


def mutate(path, change):
    value = load(path)
    change(value)
    dump(path, value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(seconds):
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


def contract(mode):
    return {
        "action_mode": mode,
        "action_schema": "d1-shared-wheel-leg-extension-speed-v1"
        if mode == "shared2"
        else "d1-wheel-leg-extension-speed-v1",
        "policy_action_size": 2 if mode == "shared2" else 8,
        "physical_action_size": 8,
        "physical_action_schema": "d1-wheel-leg-extension-speed-v1",
        "policy_to_physical_indices": [0] * 4 + [1] * 4 if mode == "shared2" else list(range(8)),
    }


def episode(mode, terrain, *, seed=1017, duration=0.2):
    return {
        **analysis.EPISODE_SCHEMAS,
        **contract(mode),
        "controller_parameters": dict(analysis.GAINS),
        "terrain": terrain,
        "provider": {"kind": "imu_encoder_fusion", "sensor_delay_steps": 0},
        "duration_s": duration,
        "measurement_seed": seed + 7000,
        "schedule": {"duration_s": duration, "segments": [{"vx": 0.1, "yaw": 0.05}]},
    }


def source_archive(directory, payloads):
    archive_path = directory / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in payloads.items():
            header = tarfile.TarInfo(name)
            header.size = len(payload)
            archive.addfile(header, io.BytesIO(payload))
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    dump(directory / "source.json", {"sha256": hashes, "archive_sha256": sha(archive_path)})
    dump(directory / "source_consistency.json", {"unchanged": True, "changed": []})
    return hashes


def fake_updates(directory, mode, count=2):
    directory.mkdir()
    size = contract(mode)["policy_action_size"]
    logs, zero = [], np.zeros(128)
    log_prob = np.full(128, -size * math.log(2 * math.pi) / 2)
    for index in range(count):
        arrays = {
            "observations": np.zeros((128, 82)),
            "actions_raw": np.zeros((128, size)),
            "old_log_prob": log_prob,
            "old_values": zero,
            "advantages": zero,
            "returns": zero,
            "gae_rewards": np.zeros((128, 4)),
            "gae_values": np.zeros((128, 4)),
            "gae_episode_starts": np.zeros((128, 4)),
            "gae_last_values": np.zeros(4),
            "gae_last_dones": np.zeros(4),
            "gae_gamma": np.asarray(math.exp(-0.01 / 2)),
            "gae_lambda": np.asarray(0.95),
        }
        for label in ("before", "after"):
            arrays.update(
                {
                    f"action_mean_{label}": np.zeros((128, size)),
                    f"action_std_{label}": np.ones((128, size)),
                    f"log_prob_{label}": log_prob,
                    f"ratio_{label}": np.ones(128),
                }
            )
        np.savez(directory / f"sample_{index:06d}.npz", **arrays)
        logs.append(
            {
                "audit_index": index,
                "npz": f"sample_{index:06d}.npz",
                "num_timesteps": (index + 1) * 512,
                "n_updates": (index + 1) * 4,
                "n_audit_samples": 128,
                "clip_range": 0.2,
                "param_sha256_before": hashlib.sha256(f"fake{index}".encode()).hexdigest(),
                "param_sha256_after": hashlib.sha256(f"fake{index + 1}".encode()).hexdigest(),
                "parameters_changed": True,
                "approx_reverse_kl_before": 0.0,
                "approx_reverse_kl_after": 0.0,
                "ratio_clip_fraction_before": 0.0,
                "ratio_clip_fraction_after": 0.0,
            }
        )
    (directory / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs))
    return logs


def training_csv(directory, mode, total=1024):
    size = contract(mode)["policy_action_size"]
    row = {
        "worker": 0,
        "sample": 4,
        "reward": 1.0,
        "nonflat_now": False,
        "terminated": False,
        "raw_action_rms": 2.0,
        "applied_action_rms": 1.0,
        "policy_raw_action_rms": 2.0,
        "policy_clipped_action_rms": 1.0,
        "physical_applied_action_rms": 1.0,
        "raw_action_clipped_fraction": 1.0,
    }
    for i in range(size):
        row.update(
            {
                f"raw_action_{i}": 2.0,
                f"applied_action_{i}": 1.0,
                f"policy_raw_action_{i}": 2.0,
                f"policy_clipped_action_{i}": 1.0,
            }
        )
    row.update({f"physical_applied_action_{i}": 1.0 for i in range(8)})
    with (directory / "training_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        for index in range(total):
            writer.writerow({**row, "worker": index % 4, "sample": 4 * (index // 4 + 1)})


def checkpoint_metadata(mode, seed, budget, zip_path, extra):
    size = contract(mode)["policy_action_size"]
    return {
        **analysis.EPISODE_SCHEMAS,
        **contract(mode),
        "schema": "d1-locomotion-checkpoint-v1",
        "controller_parameters": dict(analysis.GAINS),
        "model_sha256": sha(zip_path),
        "observation_dim": 82,
        "history_length": 1,
        "action_dim": size,
        "action_space": {
            "shape": [size],
            "dtype": "float32",
            "low": [-1.0] * size,
            "high": [1.0] * size,
        },
        "observation_space": {
            "shape": [82],
            "dtype": "float32",
            "low": [-100.0] * 82,
            "high": [100.0] * 82,
        },
        "recorded_episode": episode(mode, analysis.TERRAINS["train"][0], seed=seed),
        "extra": {"training_seed": seed, "num_timesteps": budget, **extra},
    }


def training_artifacts(directory, run):
    mode, seed = run["action_mode"], run["seed"]
    updates = fake_updates(directory / "updates", mode)
    training_csv(directory, mode)
    for worker in range(4):
        record = {**episode(mode, analysis.TERRAINS["train"][worker]), "episode": 0}
        (directory / f"worker{worker}_episodes.jsonl").write_text(json.dumps(record) + "\n")
    (directory / "checkpoints").mkdir()
    saved, reloads = [], []
    for index, budget in enumerate((512, 1024)):
        target = directory / "checkpoints" / f"budget{budget}"
        target.mkdir()
        zip_path, meta_path = target / "checkpoint.zip", target / "checkpoint.json"
        zip_path.write_bytes(f"Not a Torch model. Hand fixture {mode}/{budget}.".encode())
        param_hash = updates[index]["param_sha256_after"]
        dump(
            meta_path,
            checkpoint_metadata(
                mode,
                seed,
                budget,
                zip_path,
                {
                    "checkpoint_budget": budget,
                    "after_update_index": index,
                    "param_sha256_after": param_hash,
                },
            ),
        )
        saved.append(
            {
                "budget": budget,
                "num_timesteps": budget,
                "after_update_index": index,
                "n_updates": (index + 1) * 4,
                "param_sha256_after": param_hash,
                "zip_sha256": sha(zip_path),
                "metadata_sha256": sha(meta_path),
                "model_path": f"budget{budget}/checkpoint.zip",
                "metadata_path": f"budget{budget}/checkpoint.json",
                "saved_at_utc": stamp(index + 1),
                "elapsed_s": float(index + 1),
                "save_duration_s": 0.01,
            }
        )
        reloads.append(
            {
                "budget": budget,
                "after_update_index": index,
                "reloaded_policy_param_sha256": param_hash,
                "zip_sha256": sha(zip_path),
                "metadata_sha256": sha(meta_path),
            }
        )
    dump(
        directory / "checkpoints/manifest.json",
        {
            "schema": "d1-budget-checkpoints-v1",
            "budgets": [512, 1024],
            "saved": saved,
            "pending_budgets": [],
            "complete": True,
            "failed": False,
            "failure": None,
        },
    )
    phases = []
    for index, budget in enumerate((512, 1024)):
        for phase, start, end in (
            ("update", index + 0.9, index + 0.99),
            ("save", index + 0.99, index + 1.0),
        ):
            phases.append(
                {
                    "phase": phase,
                    "budget": budget if phase == "save" else None,
                    "start_perf": 100.0 + start,
                    "end_perf": 100.0 + end,
                    "duration_s": end - start,
                    "wall_start_utc": stamp(start),
                    "wall_end_utc": stamp(end),
                }
            )
    (directory / "checkpoints/phases.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in phases)
    )
    (directory / "checkpoint.zip").write_bytes(b"Different opaque final compatibility ZIP bytes.")
    dump(
        directory / "checkpoint.json",
        checkpoint_metadata(mode, seed, 1024, directory / "checkpoint.zip", {}),
    )
    final = {
        "model_path": "checkpoint.zip",
        "metadata_path": "checkpoint.json",
        "num_timesteps": 1024,
        "after_update_index": 1,
        "reloaded_policy_param_sha256": updates[-1]["param_sha256_after"],
        "zip_sha256": sha(directory / "checkpoint.zip"),
        "metadata_sha256": sha(directory / "checkpoint.json"),
    }
    dump(
        directory / "checkpoint_reload.json",
        {
            "schema": "d1-budget-checkpoint-reload-v1",
            "phase": "after_training",
            "training_final_num_timesteps": 1024,
            "training_finished_at_utc": stamp(3),
            "started_at_utc": stamp(4),
            "elapsed_s": 0.1,
            "records": reloads,
            "final_checkpoint": final,
        },
    )
    dump(
        directory / "training.json",
        {
            "num_timesteps": 1024,
            "learn_calls": 1,
            "training_mode": "single_continuous_learn",
            "checkpoint_budgets": [512, 1024],
            "policy_action_size": 2 if mode == "shared2" else 8,
            "physical_action_size": 8,
            "policy_parameter_count": 19141 if mode == "shared2" else 19537,
            "updates": 8,
            "action_contract": contract(mode),
            "checkpoint_sha256": final["zip_sha256"],
            "wall_s": 3.0,
            "training_started_at_utc": stamp(0),
            "training_finished_at_utc": stamp(3),
        },
    )


def evaluation_artifacts(directory, run):
    records = []
    mode, split = run["action_mode"], run["split"]
    size = contract(mode)["policy_action_size"]
    action = 0.0 if run["model_name"] is None else 0.1
    for road, terrain in enumerate(analysis.TERRAINS[split]):
        for seed in analysis.EVALUATION_SEEDS[split]:
            name = f"road{road}_seed{seed}"
            rows = [
                {
                    "time_s": (tick + 1) * 0.01,
                    "x_m": -3.8 + (tick + 1) * 0.001,
                    "y_m": 0.0,
                    "z_m": 0.46,
                    "ground_height_m": 0.0,
                    "clearance_m": 0.46,
                    "velocity_mps": 0.12,
                    "command_forward_velocity_mps": 0.1,
                    "velocity_error_mps": 0.02,
                    "yaw_rate_rps": 0.06,
                    "command_yaw_rate_rps": 0.05,
                    "yaw_rate_error_rps": 0.01,
                    "command_clearance_m": 0.455,
                    "height_error_m": 0.005,
                    "roll_error_rad": 0.01,
                    "pitch_error_rad": 0.02,
                    "mechanical_power_w": 3.0,
                    "action_mean_square": action**2,
                    "nonflat_now": tick >= 10,
                }
                for tick in range(20)
            ]
            with (directory / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            qpos = np.zeros((21, 23))
            qpos[:, 3] = 1.0
            qpos[0, :3] = (-3.8, 0.0, 0.455)
            qpos[1:, :3] = [[r[k] for k in ("x_m", "y_m", "z_m")] for r in rows]
            actions = np.full((20, size), action)
            np.savez(
                directory / f"{name}.npz",
                qpos=qpos,
                actions=actions,
                policy_actions=actions,
                physical_actions=actions[:, contract(mode)["policy_to_physical_indices"]],
            )
            dump(directory / f"{name}_episode.json", episode(mode, terrain, seed=seed))
            records.append(
                {
                    "case": name,
                    "seed": seed,
                    "terrain": terrain,
                    "executed_steps": 20,
                    "duration_s": 0.2,
                    "completed": True,
                    "quality_pass": True,
                    "terminal_reason": "time_limit",
                    "velocity_rmse_mps": 0.02,
                    "yaw_rmse_rps": 0.01,
                    "height_rmse_m": 0.005,
                    "attitude_rmse_rad": math.sqrt(0.0005),
                    "mean_mechanical_power_w": 3.0,
                    "action_rms": action,
                    "terrain_exposure": {
                        "nonflat_steps": 10,
                        "nonflat_fraction": 0.5,
                        "nonflat_now": True,
                        "path_m": 0.02,
                        "nonflat_path_m": 0.01,
                        "unique_nonflat_0p25m_cells": 1,
                    },
                }
            )
    dump(directory / "evaluation.json", records)


def refresh_ledger(study):
    protocol = load(study / "protocol.json")
    records = []
    for sequence, run in enumerate(protocol["runs"], 1):
        directory = study / run["name"]
        names = [
            "protocol.json",
            "source.json",
            "source.tar.gz",
            "training.json" if run["mode"] == "train" else "evaluation.json",
        ]
        records.append(
            {
                **run,
                "sequence": sequence,
                "started_utc": stamp(sequence * 10),
                "ended_utc": stamp(sequence * 10 + 5),
                "elapsed_s": 5.0,
                "returncode": 0,
                "log": f"logs/{run['name']}.log",
                "child_artifact_sha256": {n: sha(directory / n) for n in names},
            }
        )
    (study / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    with (study / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.LEDGER_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{k: record.get(k) for k in runner.LEDGER_FIELDS},
                    "command_json": json.dumps(record["command"]),
                    "child_artifact_sha256_json": json.dumps(record["child_artifact_sha256"]),
                }
            )


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    protocol = runner.build_protocol(root, smoke=True)
    root.mkdir()
    payloads = {
        "pyproject.toml": b"# Synthetic source fixture, not package configuration.\n",
        "scripts/run_d1_locomotion_experiment.py": b"# Synthetic child source, not executed.\n",
        "src/wheel_legged_control/d1/locomotion_env.py": b"# Synthetic simulator source, not executed.\n",
    }
    (root / "logs").mkdir()
    for run in protocol["runs"]:
        directory = root / run["name"]
        directory.mkdir()
        hashes = source_archive(directory, payloads)
        args = {
            "mode": run["mode"],
            "baseline": "wheel_leg",
            "action_mode": run["action_mode"],
            "seed": run["seed"],
            "duration": 0.2,
            "steps": run["budget"] or 1024,
            "source": "imu_encoder_fusion",
            "history": 1,
            "workers": 4,
            "measurement_delay": 0,
            "delay_randomization": False,
            "terrain_suite": "budget_compare_v1",
            "split": run["split"] or "development",
            "output": str(directory),
            "policy": None,
            "metadata": None,
            "checkpoint_budgets": [512, 1024] if run["mode"] == "train" else None,
            **analysis.GAINS,
        }
        if run["checkpoint_id"]:
            checkpoint = root / run["model_name"] / "checkpoints" / f"budget{run['budget']}"
            args.update(
                policy=str(checkpoint / "checkpoint.zip"),
                metadata=str(checkpoint / "checkpoint.json"),
            )
        dump(
            directory / "protocol.json",
            {
                "schema": "d1-command-locomotion-experiment-v1",
                "arguments": args,
                "ppo_settings": deepcopy(analysis.PPO_SETTINGS),
                "quality_thresholds": analysis.audit.QUALITY,
                "reward_scale": 1.0,
                "discount_horizon_s": 2.0,
                "sensor_noise": {"synthetic_fixture_sd": 0.0},
                "terrain_suite": "budget_compare_v1",
                "terrain_splits": deepcopy(analysis.TERRAINS),
                "evaluation_seeds": analysis.EVALUATION_SEEDS,
                "action_contract": contract(run["action_mode"]),
            },
        )
        (root / "logs" / f"{run['name']}.log").write_text(
            "Hand fixture, no process or model executed.\n"
        )
        (training_artifacts if run["mode"] == "train" else evaluation_artifacts)(directory, run)
    (root / "orchestrator_source.py").write_text("# Hand fixture orchestrator, not executed.\n")
    hashes["scripts/run_d1_budget_study.py"] = sha(root / "orchestrator_source.py")
    dump(
        root / "source.json",
        {
            "sha256": hashes,
            "archived_extras": {"scripts/run_d1_budget_study.py": "orchestrator_source.py"},
        },
    )
    dump(root / "source_consistency.json", {"unchanged": True, "sha256": hashes})
    protocol["source_sha256"] = hashes
    dump(root / "protocol.json", protocol)
    dump(
        root / "summary.json",
        {
            "status": "commands_completed",
            "kind": "smoke",
            "commands_completed": 12,
            "planned_commands": 12,
            "planned_evaluation_cases": 40,
            "elapsed_s": 60.0,
        },
    )
    refresh_ledger(root)
    return root


def test_hand_smoke_checks_all_records_and_does_not_mutate_inputs(study):
    before = {str(p.relative_to(study)): sha(p) for p in study.rglob("*") if p.is_file()}
    result = analysis.analyze(study)
    assert result["kind"] == "smoke"
    assert result["total_training_timesteps"] == 2048
    assert result["checkpoint_count"] == result["audited_train_calls"] == 4
    assert result["evaluation_summary"]["cases"] == result["evaluation_summary"]["completed"] == 40
    assert result["evaluation_summary"]["quality_passes"] == 40
    assert len(result["training"]) == 2
    for training in result["training"].values():
        assert training["training_samples"]["samples"] == 1024
        assert training["training_samples"]["prefixes"]["512"]["training_action_rms"] == 2.0
        assert training["training_samples"]["prefixes"]["1024"]["policy_clip_fraction"] == 1.0
        assert training["ppo_math"]["updates_checked"] == 2
        assert (
            training["final_checkpoint"]["zip_sha256"]
            != training["checkpoints"][next(reversed(training["checkpoints"]))]["zip_sha256"]
        )
    assert before == {str(p.relative_to(study)): sha(p) for p in study.rglob("*") if p.is_file()}


def test_formal_protocol_means_six_trajectories_not_sum_of_checkpoint_budgets(tmp_path):
    protocol = runner.build_protocol(tmp_path / "not_created")
    models, checkpoints, runs = analysis.expected_matrix(protocol)
    assert (len(models), len(checkpoints), len(runs)) == (6, 24, 56)
    assert protocol["total_training_transitions"] == 1572864
    assert protocol["expected_train_calls"] == 3072
    assert protocol["planned_evaluation_cases"] == 200
    assert [c["expected_update_index"] for c in checkpoints[:4]] == [63, 127, 255, 511]
    assert protocol["evaluation_seeds"]["holdout"] == [4617, 4629]
    assert protocol["terrains"]["holdout"][0]["slope_deg"] == 1.2
    assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(training_mode="repeated_learn"),
        lambda p: p.update(checkpoint_budgets=[512, 512]),
        lambda p: p.update(total_training_transitions=3072),
        lambda p: p.update(expected_train_calls=6),
        lambda p: p.update(workers=True),
        lambda p: p.update(delay_randomization=True),
        lambda p: p.update(terrain_suite="action_compare_v1"),
        lambda p: p["evaluation_seeds"].update(holdout=[1017, 1029]),
        lambda p: p["checkpoints"][0].update(expected_update_index=1),
        lambda p: p["checkpoints"][0].update(relative_zip="../checkpoint.zip"),
        lambda p: p["models"].reverse(),
        lambda p: p["runs"].reverse(),
    ],
)
def test_fixed_protocol_rejects_design_changes(tmp_path, change):
    protocol = runner.build_protocol(tmp_path / "unused", smoke=True)
    change(protocol)
    with pytest.raises(ValueError):
        analysis.expected_matrix(protocol)


@pytest.mark.parametrize(
    "key,value",
    [
        ("worker", "1"),
        ("sample", "8"),
        ("reward", "nan"),
        ("nonflat_now", "1"),
        ("applied_action_0", "0.9"),
        ("policy_raw_action_0", "1.0"),
        ("physical_applied_action_4", "0.5"),
        ("raw_action_rms", "1.9"),
        ("raw_action_clipped_fraction", "0.0"),
    ],
)
def test_streamed_csv_rejects_corrupt_sequence_and_actions(tmp_path, key, value):
    training_csv(tmp_path, "shared2", total=4)
    path = tmp_path / "training_samples.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][key] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        analysis.audit_training_samples(
            tmp_path, total=4, contract=contract("shared2"), budgets=[4]
        )


@pytest.mark.parametrize("actual", [3, 5])
def test_streamed_csv_requires_exact_transition_count(tmp_path, actual):
    training_csv(tmp_path, "independent8", total=actual)
    with pytest.raises(ValueError):
        analysis.audit_training_samples(
            tmp_path, total=4, contract=contract("independent8"), budgets=[4]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_timesteps", 512),
        ("n_updates", 4),
        ("audit_index", 0),
        ("param_sha256_before", "0" * 64),
    ],
)
def test_ppo_trajectory_cannot_restart_or_break_hash_chain(tmp_path, field, value):
    directory = tmp_path / "updates"
    rows = fake_updates(directory, "shared2")
    rows[1][field] = value
    (directory / "updates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        analysis.audit_updates(directory, total=1024, policy_size=2)


@pytest.mark.parametrize("key", ["gae_rewards", "log_prob_after", "observations", "actions_raw"])
def test_ppo_raw_arrays_are_recomputed_and_dimensions_checked(tmp_path, key):
    directory = tmp_path / "updates"
    fake_updates(directory, "shared2")
    path = directory / "sample_000000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    if key in ("observations", "actions_raw"):
        arrays[key] = arrays[key][:, :-1]
    else:
        arrays[key].flat[0] = 1.0
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        analysis.audit_updates(directory, total=1024, policy_size=2)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}\n', '{"a":NaN}\n', "\n", "[]\n"])
def test_strict_jsonl_rejects_ambiguous_records(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        analysis.read_jsonl(path)


@pytest.fixture
def training_case(tmp_path):
    directory = tmp_path / "train"
    directory.mkdir()
    protocol = runner.build_protocol(tmp_path / "unused", smoke=True)
    model = protocol["models"][0]
    run = next(r for r in protocol["runs"] if r["name"] == model["training_run"])
    training_artifacts(directory, run)
    checkpoints = [c for c in protocol["checkpoints"] if c["model_id"] == model["model_id"]]
    return directory, model, checkpoints, protocol


def check_checkpoints(case):
    directory, model, checkpoints, protocol = case
    return analysis.audit_checkpoints(
        directory,
        model,
        checkpoints,
        protocol,
        load(directory / "training.json"),
        analysis.read_jsonl(directory / "updates/updates.jsonl"),
    )


@pytest.mark.parametrize(
    "filename,change",
    [
        ("checkpoints/manifest.json", lambda d: d.update(complete=False)),
        ("checkpoints/manifest.json", lambda d: d.update(pending_budgets=[1024])),
        ("checkpoints/manifest.json", lambda d: d.update(failed=True)),
        ("checkpoints/manifest.json", lambda d: d["saved"].reverse()),
        ("checkpoints/manifest.json", lambda d: d["saved"][0].update(after_update_index=1)),
        ("checkpoints/manifest.json", lambda d: d["saved"][0].update(elapsed_s=4.0)),
        ("checkpoints/manifest.json", lambda d: d["saved"][1].update(elapsed_s=1.5)),
        ("checkpoints/manifest.json", lambda d: d["saved"][0].update(saved_at_utc=stamp(2.5))),
        ("checkpoints/manifest.json", lambda d: d["saved"][0].update(save_duration_s=0.02)),
        ("checkpoint_reload.json", lambda d: d.update(phase="during_training")),
        ("checkpoint_reload.json", lambda d: d.update(started_at_utc=stamp(2))),
        ("checkpoint_reload.json", lambda d: d.update(training_final_num_timesteps=512)),
        (
            "checkpoint_reload.json",
            lambda d: d["records"][0].update(reloaded_policy_param_sha256="0" * 64),
        ),
        (
            "checkpoint_reload.json",
            lambda d: d["final_checkpoint"].update(reloaded_policy_param_sha256="0" * 64),
        ),
    ],
)
def test_checkpoint_manifest_and_after_learning_witness_are_required(
    training_case, filename, change
):
    mutate(training_case[0] / filename, change)
    with pytest.raises(ValueError):
        check_checkpoints(training_case)


def test_checkpoint_metadata_seed_is_checked_even_after_resigning_hash(training_case):
    directory = training_case[0]
    path = directory / "checkpoints/budget512/checkpoint.json"
    mutate(path, lambda d: d["extra"].update(training_seed=999))
    mutate(
        directory / "checkpoints/manifest.json",
        lambda d: d["saved"][0].update(metadata_sha256=sha(path)),
    )
    mutate(
        directory / "checkpoint_reload.json",
        lambda d: d["records"][0].update(metadata_sha256=sha(path)),
    )
    with pytest.raises(ValueError, match="training_seed"):
        check_checkpoints(training_case)


@pytest.mark.parametrize("name", ["checkpoint.zip", "checkpoints/budget512/checkpoint.zip"])
def test_both_final_and_intermediate_zip_bytes_are_hashed(training_case, name):
    (training_case[0] / name).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash"):
        check_checkpoints(training_case)


def test_checkpoint_phase_order_cannot_claim_save_before_completed_update(training_case):
    path = training_case[0] / "checkpoints/phases.jsonl"
    phases = analysis.read_jsonl(path)
    phases[0], phases[1] = phases[1], phases[0]
    path.write_text("".join(json.dumps(p) + "\n" for p in phases))
    with pytest.raises(ValueError, match="phase order"):
        check_checkpoints(training_case)


@pytest.mark.parametrize(
    "kind", ["failed_parent", "source_changed", "missing_extra", "wrong_ledger"]
)
def test_parent_completion_source_and_ledger_are_independent_gates(study, kind):
    if kind == "failed_parent":
        dump(study / "failure.json", {"error": "child failed"})
    elif kind == "source_changed":
        mutate(study / "source_consistency.json", lambda d: d.update(unchanged=False))
    elif kind == "missing_extra":
        (study / "orchestrator_source.py").rename(study / "not_the_declared_extra.py")
    else:
        rows = analysis.read_jsonl(study / "runs.jsonl")
        rows[0]["error_type"] = "MissingArtifact"
        (study / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        analysis.analyze(study)


def test_failed_full_horizon_is_not_counted_as_completed_or_paired_improvement(study):
    rows = analysis.analyze(study)["evaluations"]["zero_development"]
    before, after = deepcopy(rows), deepcopy(rows)
    after[0].update(completed=False, quality_pass=False, terminal_reason="fall")
    report = analysis.summarize_cases(after, 0.2)
    assert (report["completed"], report["failed"], report["full_horizon_failures"]) == (3, 1, 1)
    pairs = analysis.compare_cases(before, after)
    assert pairs[0]["right_minus_left"] is None
    after[1]["measurement_seed"] += 1
    with pytest.raises(ValueError, match="measurement_seed"):
        analysis.compare_cases(before, after)


@pytest.mark.parametrize("kind", ["existing", "overlap", "symlink", "invalid_input"])
def test_cli_rejects_unsafe_output_or_bad_input_before_creating_report(study, tmp_path, kind):
    output = tmp_path / "analysis"
    if kind == "existing":
        output.mkdir()
    elif kind == "overlap":
        output = study / "analysis"
    elif kind == "symlink":
        output.symlink_to(study, target_is_directory=True)
    else:
        mutate(study / "summary.json", lambda d: d.update(status="failed"))
    with pytest.raises(ValueError):
        analysis.main([str(study), "--output", str(output)])
    assert not (output / "analysis.json").exists()


def test_archived_analyzer_runs_after_study_migration_without_original_repo(study, tmp_path):
    original = tmp_path / "first_analysis"
    report = analysis.main([str(study), "--output", str(original)])
    manifest = load(original / "manifest.json")["sha256"]
    assert len(manifest) == 4
    assert all(sha(original / name) == digest for name, digest in manifest.items())
    moved = tmp_path / "download" / "study"
    shutil.copytree(study, moved)
    study.rename(tmp_path / "old_path_no_longer_available")
    output = tmp_path / "migrated_analysis"
    dependencies = Path(__file__).resolve().parents[1] / ".local-deps"
    environment = {
        **os.environ,
        "PYTHONPATH": str(dependencies),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    process = subprocess.run(
        [
            sys.executable,
            str(original / "analyze_d1_budget_study.py"),
            str(moved),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    restored = load(output / "analysis.json")
    for key in (
        "total_training_timesteps",
        "checkpoint_count",
        "audited_train_calls",
        "comparisons",
    ):
        assert restored[key] == report[key]
    assert restored["study"] == str(moved)


def test_analysis_dependency_closure_does_not_import_simulator_or_torch():
    forbidden = {"torch", "mujoco", "stable_baselines3", "wheel_legged_control"}
    for module in (analysis, analysis.audit, analysis.ppo_math):
        tree = ast.parse(Path(module.__file__).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        assert not forbidden.intersection(name.split(".")[0] for name in names)
