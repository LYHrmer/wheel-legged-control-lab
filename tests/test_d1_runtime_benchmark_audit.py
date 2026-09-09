"""Independent arithmetic and deliberately corrupted benchmark artifacts."""

import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.audit_d1_runtime_benchmark import audit_directory, linear_quantile


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def rewrite_csv(path, mutate):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        names, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def artifacts(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "benchmark"
    output.mkdir()
    hashes = {}
    for name in ("src/wheel_legged_control/d1/fixture.py", "scripts/benchmark_d1_runtime.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Deliberately synthetic source fixture, not a simulation.\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name in hashes:
            archive.add(source / name, arcname=name)
    write_json(
        output / "protocol.json",
        {
            "schema": "d1-runtime-sampling-development-v1",
            "duration_s": 2.0,
            "repeats": 1,
            "control_dt_s": 0.01,
            "physics_dt_s": 0.002,
            "noise_seed": 17,
            "state_noise": "zero",
            "source_sha256": hashes,
        },
    )
    write_json(output / "source_consistency.json", {"changed_during_run": []})
    summaries = []
    for kind in ("oracle", "imu_encoder_fusion"):
        for mode in ("legacy_mixed", "synchronized"):
            rows = []
            for tick in range(1, 201):
                command = 0 if tick <= 50 else 0.2
                timing = (100, 200, 300, 600) if tick == 1 else (1, 2, 3, 6)
                rows.append(
                    {
                        "tick": tick,
                        "time_s": tick * 0.01,
                        "command_vx_mps": command,
                        "published_vx_mps": command + 0.1,
                        "height_error_m": 0.01,
                        "pitch_rad": -0.02,
                        "origin_phase_error_m": 0.002,
                        "controller_ms": timing[0],
                        "physics_and_sample_ms": timing[1],
                        "state_source_ms": timing[2],
                        "control_tick_ms": timing[3],
                    }
                )
            path = output / f"{kind}_{mode}_0.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = {
                "sampling_mode": mode,
                "state_source": kind,
                "repeat": 0,
                "steps": 200,
                "completed": True,
                "height_rmse_m": 0.01,
                "tail_velocity_rmse_mps": 0.1,
                "max_origin_phase_error_m": 0.002,
            }
            for name, value in (
                ("controller_ms", 1),
                ("physics_and_sample_ms", 2),
                ("state_source_ms", 3),
                ("control_tick_ms", 6),
            ):
                summary.update(
                    {f"{name}_{label}": value for label in ("median", "p95", "p99", "max")}
                )
            summaries.append(summary)
    write_json(output / "summary.json", summaries)
    return output, source


def test_linear_quantiles_are_interpolated_without_numpy():
    assert linear_quantile([40, 10, 30, 20], 0.5) == 25
    assert linear_quantile([40, 10, 30, 20], 0.95) == pytest.approx(38.5)
    assert linear_quantile([40, 10, 30, 20], 0.99) == pytest.approx(39.7)
    assert linear_quantile([40, 10, 30, 20], 1) == 40


def test_audit_recomputes_all_metrics_and_is_read_only(artifacts):
    output, source = artifacts
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    audit = audit_directory(output, source)
    assert audit["checked"] and audit["source_root_verified"]
    assert audit["source_file_count"] == 2
    assert len(audit["runs"]) == 4
    for run in audit["runs"]:
        assert run["steps"] == 200 and run["completed"] is True
        assert run["height_rmse_m"] == pytest.approx(0.01)
        assert run["tail_velocity_rmse_mps"] == pytest.approx(0.1)
        assert run["control_tick_ms_max"] == 6  # first 600 ms tick excluded only from timing
    assert audit["input_sha256"] == {
        name: hashlib.sha256(blob).hexdigest() for name, blob in before.items()
    }
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize(
    "field,value",
    (
        ("tick", "3"),
        ("time_s", ".02"),
        ("command_vx_mps", ".2"),
        ("published_vx_mps", "NaN"),
        ("height_error_m", "Infinity"),
        ("origin_phase_error_m", "-.1"),
        ("controller_ms", "-1"),
        ("control_tick_ms", "601"),
    ),
)
def test_invalid_csv_values_or_clock_are_rejected(artifacts, field, value):
    output, source = artifacts
    rewrite_csv(output / "oracle_legacy_mixed_0.csv", lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError):
        audit_directory(output, source)


@pytest.mark.parametrize(
    "change", ("metric", "duplicate", "missing", "extra_metric", "completion", "nonfinite")
)
def test_forged_or_incomplete_reported_summary_is_rejected(artifacts, change):
    output, source = artifacts
    summaries = read_json(output / "summary.json")
    if change == "metric":
        summaries[0]["control_tick_ms_p95"] = 7
    elif change == "duplicate":
        summaries[1] = summaries[0]
    elif change == "missing":
        summaries.pop()
    elif change == "extra_metric":
        summaries[0]["made_up_speedup"] = 100
    elif change == "completion":
        summaries[0]["completed"] = False
    else:
        summaries[0]["height_rmse_m"] = float("nan")
    write_json(output / "summary.json", summaries)
    with pytest.raises(ValueError):
        audit_directory(output, source)


@pytest.mark.parametrize(
    "field,value",
    (
        ("repeats", True),
        ("duration_s", float("nan")),
        ("physics_dt_s", 0.01),
        ("noise_seed", 18),
        ("state_noise", "unknown"),
    ),
)
def test_wrong_protocol_is_rejected(artifacts, field, value):
    output, source = artifacts
    protocol = read_json(output / "protocol.json")
    protocol[field] = value
    write_json(output / "protocol.json", protocol)
    with pytest.raises(ValueError):
        audit_directory(output, source)


def test_source_changes_during_benchmark_are_not_accepted(artifacts):
    output, source = artifacts
    write_json(output / "source_consistency.json", {"changed_during_run": ["src/changed.py"]})
    with pytest.raises(ValueError):
        audit_directory(output, source)


def test_snapshot_hash_mismatch_is_rejected_without_extracting_archive(artifacts):
    output, source = artifacts
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name in read_json(output / "protocol.json")["source_sha256"]:
            data = b"changed archived source"
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError):
        audit_directory(output, source)


@pytest.mark.parametrize("change", ("extra_input", "modified_current_source", "new_current_source"))
def test_exact_input_and_current_source_coverage(artifacts, change):
    output, source = artifacts
    if change == "extra_input":
        (output / "forgotten.csv").write_text("untracked\n")
    elif change == "modified_current_source":
        (source / "scripts/benchmark_d1_runtime.py").write_text("changed\n")
    else:
        (source / "src/wheel_legged_control/new.py").write_text("new\n")
    with pytest.raises(ValueError):
        audit_directory(output, source)


@pytest.mark.parametrize("fault", ("traversal", "symlink", "duplicate"))
def test_unsafe_or_duplicate_archive_members_are_rejected_without_extraction(artifacts, fault):
    output, source = artifacts
    names = list(read_json(output / "protocol.json")["source_sha256"])
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name in names:
            archive.add(source / name, arcname=name)
        if fault == "duplicate":
            archive.add(source / names[0], arcname=names[0])
        else:
            entry = tarfile.TarInfo("../outside.py" if fault == "traversal" else "linked.py")
            if fault == "symlink":
                entry.type = tarfile.SYMTYPE
                entry.linkname = names[0]
            archive.addfile(entry)
    with pytest.raises(ValueError):
        audit_directory(output, source)
    assert not (output.parent / "outside.py").exists()


def test_cli_writes_recomputed_audit_and_its_actual_source_snapshot(artifacts):
    benchmark, source = artifacts
    script = Path(__file__).resolve().parents[1] / "scripts/audit_d1_runtime_benchmark.py"
    output = benchmark.parent / "audit"
    before = {path.name: path.read_bytes() for path in benchmark.iterdir()}
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(benchmark),
            "--source-root",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    audit = read_json(output / "audit.json")
    assert audit["source_root_verified"] and audit["checked"]
    assert audit["audit_source_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert (output / "audit_source.py").read_bytes() == script.read_bytes()
    with (output / "recomputed.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert {path.name: path.read_bytes() for path in benchmark.iterdir()} == before
