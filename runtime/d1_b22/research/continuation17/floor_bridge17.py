"""Stdlib-only C17 bridge from a live owner to a clean offline floor reader."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
W = HERE.parent
OLD_SESSION = W / "rl11" / "training_run_01" / "session.json"
CLI = HERE / "offline_floor17.py"
TIMEOUT_S = 60
EXECUTION_TIMEOUT_S = 55


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"offline floor input is missing or linked: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _floor_inputs(folder: Path) -> dict[str, Any]:
    files = {name: _file_identity(folder / name) for name in (
        "case_receipt.json", "schedule.json", "states.npz",
        "geometry_manifest.json",
    )}
    manifests = sorted(folder.rglob("*.manifest.json"))
    if not manifests:
        raise RuntimeError("offline floor has no archive manifests")
    archive = [_file_identity(path) for path in manifests]
    digest = hashlib.sha256(json.dumps(archive, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    return {"files": files, "archive_manifests": archive,
            "archive_manifest_aggregate_sha256": digest}


def _process_table() -> dict[int, tuple[int, str, int]]:
    """Read PID -> (PPid, state, start tick) without touching other processes."""
    table = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdecimal():
            continue
        try:
            stat = Path(entry.path, "stat").read_text(encoding="ascii")
            fields = stat[stat.rfind(")") + 2:].split()
            table[int(entry.name)] = (int(fields[1]), fields[0], int(fields[19]))
        except (OSError, IndexError, ValueError):
            continue
    return table


def _owned_descendants(pid: int) -> list[tuple[int, int, int]]:
    """Snapshot only descendants of this Popen PID, deepest first."""
    table = _process_table()
    frontier = [(pid, 0)]
    owned = []
    while frontier:
        parent, depth = frontier.pop()
        for child, (ppid, _state, birth) in table.items():
            if ppid == parent:
                owned.append((child, birth, depth + 1))
                frontier.append((child, depth + 1))
    return sorted(owned, key=lambda row: row[2], reverse=True)


def _live_owned(owned: list[tuple[int, int, int]]) -> list[int]:
    table = _process_table()
    return [pid for pid, birth, _depth in owned
            if pid in table and table[pid][2] == birth and table[pid][1] != "Z"]


def _kill_owned(process: subprocess.Popen[bytes], deadline: float,
                observed: dict[int, tuple[int, int, int]]) -> tuple[int | None, list[int], list[int]]:
    """Kill this Popen's descendants and root, then reap within the 60 s deadline."""
    owned = _owned_descendants(process.pid)
    observed.update({row[0]: row for row in owned})
    killed = []
    def kill_if_same(pid: int, birth: int) -> None:
        current = _process_table().get(pid)
        if current is not None and current[2] == birth and current[1] != "Z":
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass

    for pid, birth, _depth in owned:
        kill_if_same(pid, birth)
    # Catch children started during the first /proc scan before stopping root.
    known = {pid for pid, _birth, _depth in owned}
    for pid, birth, depth in _owned_descendants(process.pid):
        if pid not in known:
            owned.append((pid, birth, depth))
            observed[pid] = (pid, birth, depth)
            kill_if_same(pid, birth)
    if process.poll() is None:
        process.kill()
        killed.append(process.pid)
    try:
        exit_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        exit_code = process.poll()
    while _live_owned(owned) and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return exit_code, killed, _live_owned(owned)


def _clean_environment(override: Mapping[str, str | None] | None,
                       floor: Path | None) -> dict[str, str]:
    environment_session = OLD_SESSION if floor is None else floor.parent.parent / "session.json"
    with environment_session.open(encoding="utf-8") as stream:
        session = json.load(stream)
    frozen = session["runtime_environment"]
    path = frozen.get("PYTHONPATH")
    if type(path) is not str or not path:
        raise RuntimeError("frozen old session has no Python source path")
    child = os.environ.copy()
    for key, value in frozen.items():
        if key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"):
            continue
        if value is None:
            child.pop(key, None)
        else:
            child[key] = value
    if override is not None:
        for key, value in override.items():
            if key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"):
                continue
            if value is None:
                child.pop(key, None)
            elif type(value) is str:
                child[key] = value
            else:
                raise TypeError("offline environment values must be strings or None")
    child.pop("LD_PRELOAD", None)
    child.pop("LD_LIBRARY_PATH", None)
    paths = [part for part in path.split(os.pathsep) if part]
    for extra in (HERE, W / "continuation15", W, W / "continuation13"):
        name = str(extra)
        if name not in paths:
            paths.append(name)
    child["PYTHONPATH"] = os.pathsep.join(paths)
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    return child


def _run(
    *, floor: Path | None, probe: bool,
    environment: Mapping[str, str | None] | None,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    if probe == (floor is not None):
        raise ValueError("choose exactly one offline probe or floor read")
    if floor is not None and (not floor.is_absolute() or not floor.is_dir()
                              or floor.is_symlink()):
        raise ValueError("offline floor folder must be an absolute real directory")
    floor_inputs = None if floor is None else _floor_inputs(floor)
    if artifact_dir is None:
        parent = HERE / "isolated_runs"
        parent.mkdir(exist_ok=True)
        artifact_dir = parent / uuid.uuid4().hex
    if not artifact_dir.is_absolute() or artifact_dir.is_symlink():
        raise ValueError("offline artifact directory must be an absolute real path")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    result_path = artifact_dir / "result.json"
    log_path = artifact_dir / "stdout.log"
    receipt_path = artifact_dir / "bridge_receipt.json"
    rtk = shutil.which("rtk")
    if rtk is None:
        raise RuntimeError("rtk command is unavailable for the offline bridge")
    command = [rtk, "proxy", "/usr/bin/python3", "-B", str(CLI)]
    if probe:
        command.append("--probe-import")
    else:
        command.extend(("--floor", str(floor)))
    command.extend(("--output", str(result_path)))
    child_env = _clean_environment(environment, floor)
    started = time.monotonic()
    deadline = started + TIMEOUT_S
    timed_out = False
    exit_code: int | None = None
    pid: int | None = None
    killed_owned_pids: list[int] = []
    live_owned_after_reap: list[int] = []
    owned_live_at_exit: list[int] = []
    observed_owned: dict[int, tuple[int, int, int]] = {}
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command, cwd=W, env=child_env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=False,
        )
        pid = process.pid
        while process.poll() is None and time.monotonic() < started + EXECUTION_TIMEOUT_S:
            for row in _owned_descendants(pid):
                observed_owned[row[0]] = row
            time.sleep(min(0.1, max(0.0, started + EXECUTION_TIMEOUT_S
                                     - time.monotonic())))
        if process.poll() is None:
            timed_out = True
            exit_code, killed_owned_pids, live_owned_after_reap = _kill_owned(
                process, deadline, observed_owned)
        else:
            exit_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        live_owned_after_reap = sorted(set(live_owned_after_reap)
                                        | set(_live_owned(list(observed_owned.values()))))
        owned_live_at_exit = list(live_owned_after_reap)
        if live_owned_after_reap:
            table = _process_table()
            for child in live_owned_after_reap:
                record = observed_owned.get(child)
                if record is not None and child in table and table[child][2] == record[1]:
                    try:
                        os.kill(child, signal.SIGKILL)
                        killed_owned_pids.append(child)
                    except ProcessLookupError:
                        pass
            while time.monotonic() < deadline:
                live_owned_after_reap = _live_owned(list(observed_owned.values()))
                if not live_owned_after_reap:
                    break
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        log.flush()
        os.fsync(log.fileno())
    elapsed = time.monotonic() - started
    result: dict[str, Any] | None = None
    failure = None
    if timed_out:
        failure = "offline child exceeded 55-second execution cap"
    elif elapsed > TIMEOUT_S:
        failure = "offline bridge exceeded 60-second cap"
    elif owned_live_at_exit:
        failure = "offline child left owned descendants after exit"
    elif live_owned_after_reap:
        failure = "offline child left owned descendants alive"
    elif exit_code != 0:
        failure = f"offline child exited {exit_code}"
    elif not result_path.is_file() or result_path.is_symlink():
        failure = "offline child did not commit an exclusive result"
    else:
        try:
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            if not isinstance(result, dict):
                raise ValueError("offline result is not an object")
            if probe:
                if (result.get("probe_import_passed") is not True
                        or result.get("forbidden_modules_loaded") is not False):
                    raise ValueError("offline import probe did not prove clean isolation")
            elif (result.get("case_id") != "floor_0p4_600"
                  or type(result.get("numeric_gate_passed")) is not bool):
                raise ValueError("offline floor result lacks a numeric gate")
            if not probe:
                with (floor / "case_receipt.json").open(encoding="utf-8") as stream:
                    case_receipt = json.load(stream)
                with (floor / "schedule.json").open(encoding="utf-8") as stream:
                    schedule = json.load(stream)
                actor = case_receipt.get("actor")
                checkpoint = case_receipt.get("checkpoint_sha256")
                if (actor != "grouped_continue"
                        or floor.name != actor
                        or result.get("experiment_actor") != actor
                        or result.get("checkpoint_sha256") != checkpoint
                        or schedule.get("actor") != actor
                        or schedule.get("checkpoint_sha256") != checkpoint):
                    raise ValueError("offline result actor/checkpoint identity differs")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            failure = f"offline result is invalid: {error}"
    try:
        floor_inputs_after = None if floor is None else _floor_inputs(floor)
    except (OSError, RuntimeError) as error:
        floor_inputs_after = {"readback_error": str(error)}
    floor_inputs_unchanged = floor_inputs == floor_inputs_after
    if not floor_inputs_unchanged and failure is None:
        failure = "offline floor input changed during readback"
    receipt = {
        "schema": "d1-stage17-offline-floor-bridge-v1",
        "mode": "import_probe" if probe else "floor_readback",
        "floor": None if floor is None else str(floor),
        "argv": command, "child_pid": pid,
        "same_process_group": True, "new_session": False,
        "exit_code": exit_code, "timed_out": timed_out,
        "execution_timeout_s": EXECUTION_TIMEOUT_S,
        "timeout_s": TIMEOUT_S, "elapsed_s": elapsed,
        "killed_owned_pids": killed_owned_pids,
        "owned_live_at_exit": owned_live_at_exit,
        "live_owned_after_reap": live_owned_after_reap,
        "owned_descendants_reaped": not live_owned_after_reap,
        "floor_inputs": floor_inputs,
        "floor_inputs_unchanged": floor_inputs_unchanged,
        "loader_injection_removed": ("LD_PRELOAD" not in child_env
                                     and "LD_LIBRARY_PATH" not in child_env),
        "pythonpath": child_env["PYTHONPATH"],
        "bridge_source": str(Path(__file__).resolve()),
        "bridge_source_sha256": _sha256(Path(__file__).resolve()),
        "offline_source": str(CLI), "offline_source_sha256": _sha256(CLI),
        "reader_source": str(HERE / "read_eval17.py"),
        "reader_source_sha256": _sha256(HERE / "read_eval17.py"),
        "environment_session_source": str(OLD_SESSION if floor is None else floor.parent.parent / "session.json"),
        "environment_session_sha256": _sha256(OLD_SESSION if floor is None else floor.parent.parent / "session.json"),
        "stdout_log": str(log_path), "stdout_sha256": _sha256(log_path),
        "result_path": str(result_path) if result_path.exists() else None,
        "result_sha256": _sha256(result_path) if result_path.is_file() else None,
        "forbidden_modules_absent_in_child": failure is None,
        "model_loaded_by_bridge": False,
        "physics_called_by_bridge": False,
        "failure": failure,
    }
    _write_json(receipt_path, receipt)
    if failure is not None:
        raise RuntimeError(f"offline floor bridge failed; receipt: {receipt_path}; {failure}")
    assert result is not None
    return result


def read_floor_isolated(
    folder: Path, *, environment: Mapping[str, str | None] | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the saved numeric gate from a fresh 60-second offline process."""
    return _run(floor=folder, probe=False, environment=environment,
                artifact_dir=artifact_dir)


def probe_import_isolated(
    *, environment: Mapping[str, str | None] | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Exercise only clean reader imports, with no floor artifacts or physics."""
    return _run(floor=None, probe=True, environment=environment,
                artifact_dir=artifact_dir)


__all__ = ("read_floor_isolated", "probe_import_isolated", "TIMEOUT_S")
