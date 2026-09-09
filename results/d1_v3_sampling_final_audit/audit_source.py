"""Independently recompute sampling-benchmark CSV metrics and verify source hashes.

This checks artifact consistency, not authenticity or hard real-time guarantees.
Timing repeats share one deterministic task; they are not new control trials.
The benchmark measures a direct LQR/sampling path, not the complete v3 PPO loop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path

AUDIT_SCHEMA = "d1-runtime-independent-audit-v1"
PROTOCOL_SCHEMA = "d1-runtime-sampling-development-v1"
BENCHMARK_SCRIPT = "scripts/benchmark_d1_runtime.py"
FIXED_FILES = ("protocol.json", "summary.json", "source_consistency.json", "source.tar.gz")
STATE_SOURCES = ("oracle", "imu_encoder_fusion")
SAMPLING_MODES = ("legacy_mixed", "synchronized")
TIMING_FIELDS = ("controller_ms", "physics_and_sample_ms", "state_source_ms", "control_tick_ms")
CSV_FIELDS = (
    "tick",
    "time_s",
    "command_vx_mps",
    "published_vx_mps",
    "height_error_m",
    "pitch_rad",
    "origin_phase_error_m",
    *TIMING_FIELDS,
)
QUANTILES = (("median", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))


def _fail(where, message):
    raise ValueError(f"{where}: {message}")


def _load_json(path):
    def constant(value):
        _fail(path.name, f"nonfinite JSON literal {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail(path.name, f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(), parse_constant=constant, object_pairs_hook=unique_pairs)
    except json.JSONDecodeError as error:
        _fail(path.name, f"invalid JSON: {error.msg}")


def _sha256_stream(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256_file(path):
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _number(where, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(where, "must be a finite JSON number")
    return float(value)


def _integer(where, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        _fail(where, f"must be an integer in [{low}, {high}]")
    return value


def _safe_path(where, name):
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or ":" in name
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        _fail(where, f"unsafe relative path {name!r}")


def linear_quantile(values, q):
    """Sorted linear interpolation at position q*(n-1), independently of NumPy."""
    q = _number("quantile", q)
    if not 0 <= q <= 1:
        _fail("quantile", "q must lie in [0, 1]")
    ordered = sorted(_number("quantile sample", value) for value in values)
    if not ordered:
        _fail("quantile", "empty sample")
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        _fail("protocol.json", f"schema must be {PROTOCOL_SCHEMA}")
    duration = _number("protocol.json duration_s", protocol.get("duration_s"))
    if not 2 <= duration <= 10 or abs(duration / 0.01 - round(duration / 0.01)) > 1e-9:
        _fail("protocol.json duration_s", "must be 2..10 s in whole 10 ms ticks")
    repeats = _integer("protocol.json repeats", protocol.get("repeats"), 1, 10)
    for key, wanted in (("control_dt_s", 0.01), ("physics_dt_s", 0.002)):
        if _number(f"protocol.json {key}", protocol.get(key)) != wanted:
            _fail("protocol.json", f"{key} must equal {wanted}")
    _integer("protocol.json noise_seed", protocol.get("noise_seed"), 17, 17)
    if protocol.get("state_noise") != "zero":
        _fail("protocol.json", "state_noise must be the literal 'zero'")
    sources = protocol.get("source_sha256")
    if not isinstance(sources, dict) or BENCHMARK_SCRIPT not in sources or len(sources) < 2:
        _fail("protocol.json", "source_sha256 must cover package sources and benchmark script")
    for name, digest in sources.items():
        _safe_path("source_sha256", name)
        if name != BENCHMARK_SCRIPT and not (
            name.startswith("src/wheel_legged_control/") and name.endswith(".py")
        ):
            _fail("source_sha256", f"out-of-scope source {name}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest) <= set("0123456789abcdef")
        ):
            _fail("source_sha256", f"invalid SHA256 for {name}")
    return duration, repeats, round(duration / 0.01), sources


def _verify_archive(path, expected):
    observed = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = member.name
            _safe_path(path.name, name)
            if not member.isfile() or name in observed:
                _fail(path.name, f"nonregular or duplicate archive member {name}")
            with archive.extractfile(member) as handle:
                observed[name] = _sha256_stream(handle)
    if set(observed) != set(expected):
        _fail(path.name, "archive members do not exactly cover source_sha256")
    for name, digest in observed.items():
        if digest != expected[name]:
            _fail(path.name, f"archived SHA256 mismatch: {name}")


def _verify_source_root(root, expected):
    present = {
        path.relative_to(root).as_posix()
        for path in (root / "src/wheel_legged_control").rglob("*.py")
        if path.is_file()
    }
    present.add(BENCHMARK_SCRIPT)
    if present != set(expected):
        _fail("source_root", "current sources do not exactly cover archived source_sha256")
    for name in sorted(present):
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected[name]:
            _fail("source_root", f"current SHA256 mismatch: {name}")


def _read_ticks(path, planned_steps):
    rows = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader, ())) != CSV_FIELDS:
            _fail(path.name, "unexpected CSV header")
        for index, raw in enumerate(reader):
            if len(raw) != len(CSV_FIELDS):
                _fail(path.name, f"row {index} has an incorrect field count")
            try:
                row = dict(zip(CSV_FIELDS, map(float, raw)))
            except ValueError:
                _fail(path.name, f"row {index} has nonnumeric values")
            if not all(math.isfinite(value) for value in row.values()):
                _fail(path.name, f"row {index} contains nonfinite values")
            if row["tick"] != index + 1 or abs(row["time_s"] - (index + 1) * 0.01) > 1e-9:
                _fail(path.name, f"row {index} violates the sequential 10 ms clock")
            command = 0 if index < 50 else 0.2
            if abs(row["command_vx_mps"] - command) > 1e-9:
                _fail(path.name, f"row {index} has an incorrect command")
            if row["origin_phase_error_m"] < 0 or any(row[field] < 0 for field in TIMING_FIELDS):
                _fail(path.name, f"row {index} has a negative magnitude/timing")
            if (
                abs(row["control_tick_ms"] - math.fsum(row[field] for field in TIMING_FIELDS[:3]))
                > 1e-9
            ):
                _fail(path.name, f"row {index} timing components do not add up")
            rows.append(row)
    if not 2 <= len(rows) <= planned_steps or rows[-1]["time_s"] <= 1:
        _fail(path.name, "incorrect row count or empty tail window")
    return rows


def _recompute_metrics(rows, planned_steps):
    tail = [row for row in rows if row["time_s"] > 1]
    metrics = {
        "steps": len(rows),
        "completed": len(rows) == planned_steps,
        "height_rmse_m": math.sqrt(
            math.fsum(row["height_error_m"] ** 2 for row in rows) / len(rows)
        ),
        "tail_velocity_rmse_mps": math.sqrt(
            math.fsum((row["published_vx_mps"] - row["command_vx_mps"]) ** 2 for row in tail)
            / len(tail)
        ),
        "max_origin_phase_error_m": max(row["origin_phase_error_m"] for row in rows),
    }
    for field in TIMING_FIELDS:
        values = [row[field] for row in rows[1:]]  # only timing excludes first recorded tick
        metrics.update({f"{field}_{label}": linear_quantile(values, q) for label, q in QUANTILES})
    return metrics


def _compare_summary(where, expected, reported):
    if set(reported) != set(expected):
        _fail(where, "missing or extra summary keys")
    for key, want in expected.items():
        got = reported[key]
        if isinstance(want, (str, bool, int)):
            if type(got) is not type(want) or got != want:
                _fail(where, f"incorrect {key}: {got!r}, recomputed {want!r}")
        else:
            actual = _number(f"{where} {key}", got)
            if abs(actual - want) > 1e-10 + 1e-9 * abs(want):
                _fail(where, f"incorrect {key}: {actual!r}, recomputed {want!r}")


def audit_directory(input_dir: Path, source_root: Path | None = None) -> dict:
    """Read and recompute all runs; never modify inputs or extract source files."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        _fail(str(input_dir), "not a directory")
    entries = list(input_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        _fail(str(input_dir), "benchmark entries must be regular, non-symlink files")
    present = {path.name for path in entries}
    if not set(FIXED_FILES) <= present:
        _fail(str(input_dir), "required benchmark metadata missing")
    input_hashes = {path.name: _sha256_file(path) for path in sorted(entries)}
    protocol = _load_json(input_dir / "protocol.json")
    duration, repeats, planned_steps, sources = _validate_protocol(protocol)
    consistency = _load_json(input_dir / "source_consistency.json")
    if not isinstance(consistency, dict) or consistency.get("changed_during_run") != []:
        _fail("source_consistency.json", "sources changed or consistency report is malformed")
    _verify_archive(input_dir / "source.tar.gz", sources)
    if source_root is not None:
        _verify_source_root(Path(source_root), sources)
    keys = [
        (repeat, source, mode)
        for repeat in range(repeats)
        for source in STATE_SOURCES
        for mode in SAMPLING_MODES
    ]
    expected_names = set(FIXED_FILES) | {
        f"{source}_{mode}_{repeat}.csv" for repeat, source, mode in keys
    }
    if present != expected_names:
        _fail(str(input_dir), "benchmark input files do not exactly cover the protocol")
    summary = _load_json(input_dir / "summary.json")
    if not isinstance(summary, list) or len(summary) != len(keys):
        _fail("summary.json", "incorrect run count")
    reported = {}
    for entry in summary:
        if not isinstance(entry, dict):
            _fail("summary.json", "run must be an object")
        repeat = _integer("summary.json repeat", entry.get("repeat"), 0, repeats - 1)
        source, mode = entry.get("state_source"), entry.get("sampling_mode")
        if source not in STATE_SOURCES or mode not in SAMPLING_MODES:
            _fail("summary.json", "unknown state source or sampling mode")
        identity = (repeat, source, mode)
        if identity in reported:
            _fail("summary.json", f"duplicate run {identity}")
        reported[identity] = entry
    runs = []
    for repeat, source, mode in keys:
        name = f"{source}_{mode}_{repeat}.csv"
        rows = _read_ticks(input_dir / name, planned_steps)
        result = {
            "repeat": repeat,
            "state_source": source,
            "sampling_mode": mode,
            **_recompute_metrics(rows, planned_steps),
        }
        _compare_summary(name, result, reported[(repeat, source, mode)])
        runs.append(result)
    if input_hashes != {path.name: _sha256_file(path) for path in sorted(input_dir.iterdir())}:
        _fail(str(input_dir), "input artifacts changed during the audit")
    return {
        "schema": AUDIT_SCHEMA,
        "checked": True,
        "input_dir": str(input_dir),
        "duration_s": duration,
        "repeats": repeats,
        "planned_steps": planned_steps,
        "source_file_count": len(sources),
        "source_root_verified": source_root is not None,
        "runs": runs,
        "input_sha256": input_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.input.resolve()):
        parser.error("output must be outside the immutable benchmark input directory")
    result = audit_directory(args.input, args.source_root)
    result["audit_source_sha256"] = _sha256_file(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "audit_source.py")
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "recomputed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["runs"][0]))
        writer.writeheader()
        writer.writerows(result["runs"])
    print(json.dumps({"checked": True, "runs": len(result["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
