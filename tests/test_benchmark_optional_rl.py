"""The planar benchmark degrades without the [rl] extra and records what produced it.

Two defects were documented in results/benchmark/REPRODUCTION.md while experiments.py
was frozen: the residual row was meant to be optional, yet a checkpoint in the
repository made the import of stable-baselines3 unconditional; and no run recorded the
versions that produced it. These tests pin both fixes. CI installs the extra, so the
missing-extra case is simulated by blocking the import.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from wheel_legged_control import experiments

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/residual_ppo/model.zip"


@pytest.fixture
def without_rl(monkeypatch):
    """Make `import stable_baselines3` fail exactly as it does without the extra."""
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)


def test_a_missing_checkpoint_is_reported_without_importing_anything(tmp_path):
    policy, status = experiments.load_residual_policy(tmp_path / "absent.zip")
    assert policy is None and "no checkpoint" in status


def test_a_present_checkpoint_without_the_extra_degrades_instead_of_raising(without_rl):
    policy, status = experiments.load_residual_policy(CHECKPOINT)
    assert policy is None and "[rl] extra" in status


def test_runtime_versions_report_missing_packages_as_none():
    out = experiments.runtime_versions(("numpy", "a-package-that-does-not-exist"))
    assert out["packages"]["numpy"] and out["packages"]["a-package-that-does-not-exist"] is None
    assert out["python"].count(".") == 2


def test_file_sha256_is_none_for_a_missing_file_and_the_digest_otherwise(tmp_path):
    assert experiments.file_sha256(tmp_path / "absent") is None
    (tmp_path / "f").write_bytes(b"abc")
    assert experiments.file_sha256(tmp_path / "f") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_provenance_records_source_runtime_arguments_and_checkpoint(tmp_path):
    args = experiments.build_parser().parse_args(["--output", str(tmp_path), "--seed", "13"])
    path = experiments.write_provenance(tmp_path, args, "loaded")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record) == {"source", "runtime", "arguments", "checkpoint", "scenarios"}
    assert record["arguments"]["seed"] == 13 and record["scenarios"] == list(experiments.SCENARIOS)
    assert record["checkpoint"]["status"] == "loaded"
    assert record["checkpoint"]["sha256"] == experiments.file_sha256(args.policy)
    assert "mujoco" in record["runtime"]["packages"]
    assert set(record["source"]) == {"git_commit", "git_dirty", "git_worktree_sha256"}


def test_the_benchmark_runs_without_the_extra_and_writes_two_rows_and_provenance(
        tmp_path, without_rl, capsys):
    experiments.main(["--output", str(tmp_path)])
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open(encoding="utf-8")))
    assert {r["controller"] for r in rows} == {"LQR", "MPC"}
    assert len(rows) == 2 * len(experiments.SCENARIOS)
    record = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert "[rl] extra" in record["checkpoint"]["status"]
    assert "residual row skipped" in capsys.readouterr().err


def test_the_audit_without_the_extra_names_the_reason(tmp_path, without_rl):
    with pytest.raises(SystemExit, match=r"\[rl\] extra"):
        experiments.main(["--output", str(tmp_path), "--audit-episodes", "1"])
