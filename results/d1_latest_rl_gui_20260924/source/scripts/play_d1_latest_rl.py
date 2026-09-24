"""Launch one bounded latest-RL GUI session from the frozen 06 evidence.

This module uses only the Python standard library. --check is read-only and
never imports the engine, constructs a model, or starts the GUI worker.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT_FREEZE_SHA256 = "5d41c92fd9bbcd5618645d6450920a1fce4e8cf64a0c09d0970d29eb760ee144"
CONTRACT_SHA256 = "d90970745b938ede81341b308c1ba0dc5792e951cb84b54067d39a3509b3fc1f"
CHECKPOINT_SHA256 = "4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e"
METADATA_SHA256 = "44b199ea13989412d274955a9fd85d6e97e58ad708a0c7e3ab7a4085ae3daaf3"
PROBE_SHA256 = "2cb674f54c8c46d5555e5d3580810a003be8d50fa5a7c1a5526dd0f027bb6660"
PROTOCOL_SHA256 = "494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733"
DSO_SHA256 = "3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c"
PUBLISHED05_MANIFEST_SHA256 = "3919d16d500d73d053d1defbf34e48f1c7bc25dfbcc91f2a1ed36635a9b23910"
PUBLISHED06_MANIFEST_SHA256 = "dd9c0cba57c382b8161691a12171b18e0dcdab4c6fd26650b2b6b9dce86d8e18"
WORKER = "run_d1_latest_rl.py"
CONTROLS = "d1_latest_rl_controls.py"
CV2_SHA256 = {
    "__init__.py": "69192904dead3ada37c8b305ee850ac7474924730881a9d5e0b2390944eb5aae",
    "config.py": "974e2d4096ee1a9a9a341df1bf33e16973683bf1ac733006de27ff2f23bc584d",
    "config-3.py": "9a7aadf724b822001f5e963b01fd4e375d45e2cbbe328d2ca7b42a440c083a1c",
    "load_config_py3.py": "9c22766454a9ac2f6869d37830b2efc699e7b8c3125484e9976c6e677c066029",
    "version.py": "e718587212508acdcb296839c287c7fa346edb621005ffb199c26397592544ed",
    "cv2.abi3.so": "9e29605abcc31c9942d0e7dffc53b4ccaa601676c794327b0ea71a570a0c26c2",
}
ELF_SHA256 = {
    "libmujoco.so.3.12.0": "bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8",
    "_functions.cpython-310-x86_64-linux-gnu.so": "c91f9a7a4f5d12023201c1c4faab893f4593cc953f53756cd9ac561d08e18a46",
    "_rollout.cpython-310-x86_64-linux-gnu.so": "f0fcc4ffe83b44829703ec5c6a7e65ee2ea7ad40648f07c8bd3515b80e49f877",
}


def identity(path: Path) -> dict[str, str | int]:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def validate_execution_evidence(receipt_path: Path, go_path: Path,
                                records: dict[str, dict]) -> None:
    """Require the actual pure round and Astra's bound GO before validation Popen."""
    repo = Path(__file__).resolve().parents[1]
    receipt_path, go_path = receipt_path.resolve(strict=True), go_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("all_passed") is not True or receipt.get("exit_code") != 0
            or receipt.get("collected") != 12
            or receipt.get("blocked_engine_imports") != []
            or receipt.get("sources_changed_during_tests") != []):
        raise RuntimeError("pure-test receipt does not show a clean 12-case round")
    outcomes = receipt.get("results")
    if (not isinstance(outcomes, list) or len(outcomes) != 12
            or any(row.get("when") != "call" or row.get("outcome") != "passed"
                   or not isinstance(row.get("nodeid"), str) for row in outcomes)
            or len({row["nodeid"] for row in outcomes}) != 12):
        raise RuntimeError("pure-test result rows are incomplete or duplicated")
    tested = receipt.get("source_sha256")
    tests = receipt.get("file_sha256")
    if not isinstance(tested, dict) or not isinstance(tests, dict) or len(tests) != 3:
        raise RuntimeError("pure-test source or test hashes are absent")
    test_paths = {str(repo / "tests" / name) for name in (
        "test_d1_latest_rl_controls_opus.py", "test_d1_latest_rl_render_opus.py",
        "test_d1_latest_rl_integration_pure.py")}
    if set(tests) != test_paths:
        raise RuntimeError("pure-test receipt names a different test set")
    for name, digest in tested.items():
        path = Path(name)
        if (not path.is_absolute() or not path.is_file()
                or identity(path)["sha256"] != digest
                or records.get(name) != identity(path)):
            raise RuntimeError(f"source changed since pure tests: {name}")
    if any(tested.get(name) != digest for name, digest in tests.items()):
        raise RuntimeError("pure-test source and test-file hashes disagree")
    go = json.loads(go_path.read_text(encoding="utf-8"))
    if (go.get("schema") != "d1-latest-rl-go-v1" or go.get("decision") != "GO"
            or go.get("contract_sha256") != CONTRACT_SHA256
            or go.get("pure_test_receipt_sha256") != identity(receipt_path)["sha256"]):
        raise RuntimeError("Astra GO identity or pure-test binding differs")
    inputs = go.get("inputs")
    required = {str(repo / "scripts" / name) for name in (
        "play_d1_latest_rl.py", "run_d1_latest_rl.py", "d1_latest_rl_controls.py",
        "d1_latest_rl_viewer.py", "d1_latest_rl_validation.py")}
    required.update(test_paths)
    required.update((str(repo / "docs/latest_rl_gui_contract_07.md"), str(receipt_path)))
    if (not isinstance(inputs, dict) or not required <= set(inputs)
            or str(go_path) in inputs):
        raise RuntimeError("Astra GO lacks required actual source/test/receipt inputs")
    for name, expected in inputs.items():
        path = Path(name)
        if (not path.is_absolute() or not path.is_file() or path.is_symlink()
                or identity(path) != expected
                or (name in records and records[name] != expected)):
            raise RuntimeError(f"Astra GO input changed: {name}")
        records[name] = expected


def write_durable(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def preflight(repo: Path, work: Path, *, validation: bool,
              validation_evidence: tuple[Path, Path] | None) -> tuple[int, dict[str, dict], dict[str, Path]]:
    base = work.parent
    frozen_path = base / "stability_20260924_body_speed01/root_execution_freeze_01.json"
    records: dict[str, dict] = {}
    frozen_count = 0
    if validation:
        if not frozen_path.is_file() or identity(frozen_path)["sha256"] != ROOT_FREEZE_SHA256:
            raise RuntimeError("pinned 06 execution freeze is missing or changed")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("schema") != "body-common-p-evaluation-freeze-v1" or len(frozen.get("files", {})) != 767:
            raise RuntimeError("pinned 06 freeze schema or 767-file set differs")
        for name, expected in frozen["files"].items():
            path = Path(name)
            if not path.is_absolute() or not path.is_file() or identity(path) != expected:
                raise RuntimeError(f"frozen 06 input changed: {name}")
            records[name] = expected
        records[str(frozen_path)] = identity(frozen_path)
        frozen_count = 767
    published05 = repo / "results/d1_rolling_residual_speed_20260924"
    published06 = repo / "results/d1_body_speed_feedback_20260924"
    for package, pinned in ((published05, PUBLISHED05_MANIFEST_SHA256),
                            (published06, PUBLISHED06_MANIFEST_SHA256)):
        manifest = package / "publication_manifest.json"
        if not manifest.is_file() or identity(manifest)["sha256"] != pinned:
            raise RuntimeError(f"published package manifest changed: {manifest}")
        entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
        if actual != set(entries) | {"publication_manifest.json"}:
            raise RuntimeError(f"published package file set changed: {package}")
        records[str(manifest)] = identity(manifest)
        for relative, expected in entries.items():
            path = package / relative
            if not path.is_file() or path.is_symlink() or identity(path) != expected:
                raise RuntimeError(f"published package source changed: {path}")
            records[str(path)] = expected
    paths = {
        "freeze": frozen_path,
        "contract": repo / "docs/latest_rl_gui_contract_07.md",
        "launcher": Path(__file__).resolve(),
        "worker": repo / "scripts" / WORKER,
        "controls": repo / "scripts" / CONTROLS,
        "checkpoint_model": published05 / "checkpoint/model.zip",
        "checkpoint_metadata": published05 / "checkpoint/model.metadata.json",
        "checkpoint_probe": published05 / "checkpoint/reload_probe.npz",
        "protocol": published05 / "evidence/rl/run_02/protocol.json",
        "library": published05 / "engine_patch/artifact/libepa01_engine.so",
        "binding_module": published05 / "engine_patch/provenance/engine_binding.py",
        "continuation_import_repair":
            published05 / "rl_source/continuation/run_evaluation_continuation_03.py",
        "published_body_source": published06 / "source",
        "published05_manifest": published05 / "publication_manifest.json",
        "published06_manifest": published06 / "publication_manifest.json",
        "original_run": published05 / "evidence/rl/run_02",
        "body_work": published06,
    }
    for key, path in paths.items():
        if key == "freeze" and not validation:
            continue
        if key in ("original_run", "body_work", "published_body_source"):
            if not path.is_dir():
                raise RuntimeError(f"required directory absent: {path}")
        elif not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required 07 or frozen source absent: {path}")
    if identity(paths["checkpoint_model"])["sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("original trained checkpoint differs")
    for key, expected in (("checkpoint_metadata", METADATA_SHA256),
                          ("checkpoint_probe", PROBE_SHA256), ("protocol", PROTOCOL_SHA256)):
        if identity(paths[key])["sha256"] != expected:
            raise RuntimeError(f"original checkpoint protocol artifact differs: {paths[key]}")
    if identity(paths["library"])["sha256"] != DSO_SHA256:
        raise RuntimeError("isolated engine DSO differs")
    if identity(paths["contract"])["sha256"] != CONTRACT_SHA256:
        raise RuntimeError("frozen 07 contract differs")
    if validation:
        original_contract = work / "new_contract.md"
        if (not original_contract.is_file()
                or identity(original_contract)["sha256"] != CONTRACT_SHA256):
            raise RuntimeError("original frozen 07 contract differs")
        records[str(original_contract)] = identity(original_contract)
    for key in ("checkpoint_model", "checkpoint_metadata", "checkpoint_probe", "library",
                "binding_module", "continuation_import_repair", "protocol"):
        path = paths[key]
        if records.get(str(path)) != identity(path):
            raise RuntimeError(f"published runtime artifact differs: {path}")
    source_files = {paths[key] for key in ("contract", "launcher", "worker", "controls")}
    source_files.update((repo / "scripts").glob("*.py"))
    source_files.update((repo / "src").rglob("*.py"))
    source_files.update((repo / "tests").glob("test_d1_latest_rl_*.py"))
    for path in source_files:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"07 source closure absent: {path}")
        records[str(path)] = identity(path)
    cv2_dir = Path("/home/lyh/.local/lib/python3.10/site-packages/cv2")
    if (cv2_dir / "config-3.10.py").exists():
        raise RuntimeError("unexpected OpenCV version-specific loader config")
    for name, digest in CV2_SHA256.items():
        path = cv2_dir / name
        if not path.is_file() or path.is_symlink() or identity(path)["sha256"] != digest:
            raise RuntimeError(f"frozen OpenCV loader input differs: {path}")
        records[str(path)] = identity(path)
    mujoco_dir = Path("/home/lyh/.local/lib/python3.10/site-packages/mujoco")
    for name, digest in ELF_SHA256.items():
        path = mujoco_dir / name
        if not path.is_file() or path.is_symlink() or identity(path)["sha256"] != digest:
            raise RuntimeError(f"frozen MuJoCo ELF differs: {path}")
        records[str(path)] = identity(path)
    if not (repo / "scripts/d1_latest_rl_viewer.py").is_file():
        raise RuntimeError("07 viewer source absent")
    if validation:
        if validation_evidence is None:
            raise RuntimeError("validation requires pure test receipt and Astra GO")
        for path in validation_evidence:
            resolved = path.resolve(strict=True)
            if not resolved.is_file() or resolved.is_symlink():
                raise RuntimeError(f"validation evidence absent: {path}")
            records[str(resolved)] = identity(resolved)
        validate_execution_evidence(*validation_evidence, records)
    for key in ("checkpoint_model", "checkpoint_metadata", "checkpoint_probe",
                "library", "binding_module", "continuation_import_repair", "protocol"):
        path = paths[key]
        records[str(path)] = identity(path)
    return frozen_count, records, paths


def child_environment(repo: Path, paths: dict[str, Path]) -> tuple[dict[str, str], dict[str, str | None]]:
    site = Path("/home/lyh/.local/lib/python3.10/site-packages")
    python_path = [site, repo / ".local-deps", repo / "src", repo,
                   paths["published_body_source"],
                   paths["binding_module"].parent,
                   paths["continuation_import_repair"].parent]
    for path in python_path:
        if not path.is_dir():
            raise RuntimeError(f"runtime Python path absent: {path}")
    selected: dict[str, str | None] = {
        "LD_LIBRARY_PATH": None, "LD_PRELOAD": str(paths["library"]), "LD_BIND_NOW": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(map(str, python_path)),
        "DISPLAY": os.environ.get("DISPLAY"),
        "XAUTHORITY": os.environ.get("XAUTHORITY"),
    }
    child = os.environ.copy()
    child.pop("LD_LIBRARY_PATH", None)
    child.update({key: value for key, value in selected.items() if value is not None})
    return child, selected


def output_path(repo: Path, work: Path, requested: Path | None,
                *, validation: bool) -> Path:
    parent = (work / "validation_runs" if validation else
              repo / "results/d1_latest_rl_sessions")
    if requested is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = parent / f"{stamp}_{uuid.uuid4().hex[:12]}"
    else:
        if not requested.is_absolute():
            raise ValueError("--output must be an absolute new path")
        output = requested.resolve()
        if output.parent != parent:
            raise ValueError(f"--output must be a direct child of {parent}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"session output already exists: {output}")
    return output


def session_value(args: argparse.Namespace, paths: dict[str, Path], records: dict[str, dict],
                  output: Path, runtime: dict[str, str | None], worker_argv: list[str]) -> dict:
    profile = args.validation_profile
    validation = profile is not None
    if profile == "gui_policy_box" and (args.actor, args.terrain) != ("rl", "box"):
        raise ValueError("gui_policy_box requires --actor rl --terrain box")
    if profile == "headless_zero_plane" and (args.actor, args.terrain) != ("zero", "plane"):
        raise ValueError("headless_zero_plane requires --actor zero --terrain plane")
    segments = [1000, 200] if validation else [1200, 1200]
    limit = sum(segments)
    return {
        "schema": "d1-latest-rl-gui-session-v1", "mode": "validation" if validation else "manual",
        "actor": "final_policy" if args.actor == "rl" else "zero",
        "cli_actor": args.actor, "terrain": args.terrain, "speed_mps": args.speed,
        "validation_profile": profile, "seed": 77351,
        "render_quality": "low" if validation else "normal",
        "output_directory": str(output), "session_path": str(output / "session.json"),
        "library": str(paths["library"]), "original_run": str(paths["original_run"]),
        "body_work": str(paths["body_work"]),
        "checkpoint_model": str(paths["checkpoint_model"]),
        "checkpoint_metadata": str(paths["checkpoint_metadata"]),
        "checkpoint_probe": str(paths["checkpoint_probe"]),
        "protocol_path": str(paths["protocol"]),
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_directory": str(paths["checkpoint_model"].parent),
        "published_body_source": str(paths["published_body_source"]),
        "binding_module": str(paths["binding_module"]),
        "continuation_import_repair": str(paths["continuation_import_repair"]),
        "published05_manifest": str(paths["published05_manifest"]),
        "published06_manifest": str(paths["published06_manifest"]),
        "contract_path": str(paths["contract"]),
        "contract_sha256": records[str(paths["contract"])]["sha256"],
        "freeze_path": str(paths["freeze"]) if validation else None,
        "freeze_sha256": ROOT_FREEZE_SHA256 if validation else None,
        "control_limit": limit, "normal_native_limit": 5 * limit,
        "compiler_native_limit": 3, "segments": segments,
        "max_segments": 2, "wallclock_limit_s": 600 if validation else 900,
        "source_hashes": records, "runtime_environment": runtime,
        "worker_argv": worker_argv, "launcher_argv": list(sys.argv),
        "pure_test_receipt_path": (str(args.test_receipt.resolve())
                                   if validation else None),
        "astra_go_path": str(args.astra_go.resolve()) if validation else None,
        "retry_permitted": False, "old_budget_reused": False,
        "R_is_simulation_reset": True, "physical_self_righting_claimed": False,
    }


def stop_owned(child: subprocess.Popen, *, reason: str) -> None:
    if child.poll() is not None:
        return
    print(f"{reason}; asking worker to save and exit", file=sys.stderr, flush=True)
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def stream_output(child: subprocess.Popen, log: Path, errors: list[str]) -> None:
    try:
        assert child.stdout is not None
        with log.open("xb") as out:
            for block in iter(lambda: os.read(child.stdout.fileno(), 4096), b""):
                out.write(block)
                out.flush()
                try:
                    sys.stdout.buffer.write(block)
                    sys.stdout.buffer.flush()
                except (BrokenPipeError, AttributeError):
                    pass
            os.fsync(out.fileno())
    except OSError as exc:
        errors.append(f"stdout_log_error:{type(exc).__name__}:{exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", choices=("rl", "zero"), default="rl")
    parser.add_argument("--terrain", choices=("plane", "box"), default="box")
    parser.add_argument("--speed", choices=(.20, .25), type=float, default=.20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="read-only source/budget preflight")
    parser.add_argument("--validation-profile", choices=("gui_policy_box", "headless_zero_plane"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--test-receipt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--astra-go", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    base = repo.parent / (repo.name + "-work") / "recovery-20260912"
    work = base / "stability_20260924_rl_gui01"
    validation = args.validation_profile is not None
    if validation and (args.test_receipt is None or args.astra_go is None):
        parser.error("validation profile requires --test-receipt and --astra-go")
    if not validation and (args.test_receipt is not None or args.astra_go is not None):
        parser.error("validation evidence flags require --validation-profile")
    evidence = ((args.test_receipt, args.astra_go) if validation else None)
    frozen_count, records, paths = preflight(
        repo, work, validation=validation, validation_evidence=evidence)
    child_env, runtime = child_environment(repo, paths)
    if not Path("/usr/bin/python3").is_file() or not os.access("/usr/bin/python3", os.X_OK):
        raise RuntimeError("required /usr/bin/python3 worker interpreter is unavailable")
    if shutil.which("rtk") is None:
        raise RuntimeError("required rtk command is unavailable")
    display_available = bool(child_env.get("DISPLAY"))
    auth = child_env.get("XAUTHORITY")
    if auth and not Path(auth).is_file():
        display_available = False
    output = output_path(repo, work, args.output, validation=validation)
    worker_argv = ["rtk", "proxy", "/usr/bin/python3", "-B", str(paths["worker"]),
                   "--session", str(output / "session.json")]
    value = session_value(args, paths, records, output, runtime, worker_argv)
    preview = {"actor": value["actor"], "terrain": value["terrain"],
               "speed_mps": value["speed_mps"], "output": str(output),
               "checkpoint_sha256_prefix": CHECKPOINT_SHA256[:12],
               "control_limit": value["control_limit"],
               "normal_native_limit": value["normal_native_limit"],
               "compiler_native_limit": value["compiler_native_limit"],
               "display_available": display_available,
               "keys": "hold W to drive; 1=0.20, 2=0.25; Space/X=stop; R=simulation reset; Esc=save/exit"}
    if args.check:
        print(json.dumps({"preflight_passed": True, "read_only": True,
                          "historical_frozen_input_count": frozen_count, **preview}, indent=2))
        return 0
    if args.validation_profile != "headless_zero_plane" and not display_available:
        raise RuntimeError("GUI display or Xauthority is unavailable; no physics was started")
    lock_path = repo / "results/d1_latest_rl_sessions/runtime.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another 07 runtime reservation holds runtime.lock") from exc
        # Repeat the hash read under the lock; a check is never an authorization to reuse an old run.
        _, locked_records, _ = preflight(
            repo, work, validation=validation, validation_evidence=evidence)
        if locked_records != records:
            raise RuntimeError("input changed between preflight and reservation")
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.validation_profile is not None:
            for prior in output.parent.glob("*/session.json"):
                try:
                    previous = json.loads(prior.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    raise RuntimeError(f"unreadable prior validation reservation: {prior}") from None
                if (previous.get("contract_sha256") == value["contract_sha256"]
                        and previous.get("validation_profile") == args.validation_profile):
                    raise RuntimeError(f"validation profile already reserved: {args.validation_profile}")
        output.mkdir(exist_ok=False)
        value["reserved_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_durable(output / "session.json", value)
        print(json.dumps(preview), flush=True)
        child = None
        pump = None
        pump_errors: list[str] = []
        reason = None
        exit_code = None
        start = time.monotonic()
        try:
            child = subprocess.Popen(worker_argv, cwd=repo, env=child_env,
                                     stdin=None, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write_durable(output / "child_pid.json", {"pid": child.pid,
                "argv": worker_argv, "parent_pid": os.getpid(),
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            pump = threading.Thread(target=stream_output,
                                    args=(child, output / "child_stdout.log", pump_errors), daemon=True)
            pump.start()
            try:
                exit_code = child.wait(timeout=value["wallclock_limit_s"])
            except subprocess.TimeoutExpired:
                reason = f"wallclock_timeout_{value['wallclock_limit_s']}s"
                stop_owned(child, reason=reason)
                exit_code = child.returncode
            except KeyboardInterrupt:
                reason = "user_interrupt"
                stop_owned(child, reason=reason)
                exit_code = child.returncode
            pump.join(timeout=5)
        except OSError as exc:
            reason = f"spawn_error:{type(exc).__name__}:{exc}"
        except KeyboardInterrupt:
            reason = "user_interrupt"
            if child is not None:
                stop_owned(child, reason=reason)
                exit_code = child.returncode
            if pump is not None:
                pump.join(timeout=5)
        if pump_errors and reason is None:
            reason = ";".join(pump_errors)
        changed = [name for name, expected in records.items()
                   if not Path(name).is_file() or identity(Path(name)) != expected]
        receipt = {"schema": "d1-latest-rl-launcher-receipt-v1",
                   "exit_code": exit_code, "reason": reason,
                   "elapsed_seconds": time.monotonic() - start,
                   "source_hash_mismatches": changed,
                   "retry_permitted": False, "physical_counts_source": "worker receipts",
                   "session_path": str(output / "session.json")}
        write_durable(output / "launcher_receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 0 if exit_code == 0 and reason is None and not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
