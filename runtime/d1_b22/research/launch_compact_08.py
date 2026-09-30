"""08-Q one-shot root launcher: pure checks first, isolated X11, finite worker."""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

R = Path("/home/lyh/wheel-legged-control-lab")
W = Path(__file__).resolve().parent
B = W.parent
OLD = B / "stability_20260924_rl_gui01"
OUT = W / "compact_run_01"
CONTRACT_SHA = "604332e9ffd262bbf292dd247670d114e4054f4841a7acf0af7f2c8e8572eaf2"
GO = W / "astra_compact_go_08.json"
RESERVATION = W / "compact_root_reservation_08.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helper = load_module("pure07_launcher_helper", R / "scripts/play_d1_latest_rl.py")
identity = helper.identity
save = helper.write_durable


def source_preflight():
    _, records, paths = helper.preflight(
        R, OLD, validation=False, validation_evidence=None
    )
    scopes = (
        (B / "rl_improvement_20260914/frozen_source_before.json", "sha256", R),
        (
            B / "stability_20260920/shared_heave_formal_preflight_01.json",
            "input_sha256",
            R,
        ),
        (
            B / "stability_20260922/single_step_readiness_01/protocol.json",
            "input_sha256",
            R,
        ),
        (
            B / "stability_20260924_body_speed01/root_execution_freeze_01.json",
            "files",
            R,
        ),
        (
            B / "stability_20260924_body_speed01/run_01_archive_manifest_01.json",
            "files",
            B / "stability_20260924_body_speed01/run_01",
        ),
        (OLD / "astra_gui_review_inputs_07.json", "inputs", R),
        (OLD / "gui_policy_box_closed_manifest_01.json", "files", R),
        (OLD / "headless_zero_plane_closed_manifest_01.json", "files", R),
        (
            R / "results/d1_latest_rl_gui_20260924/publication_manifest.json",
            "files",
            R / "results/d1_latest_rl_gui_20260924",
        ),
    )
    for manifest, key, base in scopes:
        rows = json.loads(manifest.read_text())[key]
        for name, expected in rows.items():
            path = base / name
            actual = identity(path)
            if isinstance(expected, str):
                if actual["sha256"] != expected:
                    raise RuntimeError("historical hash mismatch: " + str(path))
            elif actual != expected:
                raise RuntimeError("historical identity mismatch: " + str(path))
            records[str(path)] = actual
        records[str(manifest)] = identity(manifest)
    for path in (
        W / "next_compact_contract_08q.md",
        W / "compact08/run_compact_q08.py",
        Path(__file__).resolve(),
        B / "gui_isolation/isolated_x11.py",
        B / "gui_isolation/rootfs/usr/bin/Xvfb",
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("required regular input absent: " + str(path))
        records[str(path)] = identity(path)
    if records[str(W / "next_compact_contract_08q.md")]["sha256"] != CONTRACT_SHA:
        raise RuntimeError("08-Q contract changed")
    for path in (W / "compact08").glob("*.py"):
        if path.is_symlink():
            raise RuntimeError("Q source symlink refused")
        records[str(path)] = identity(path)
    baseline = W / "profile_partial_closed_manifest_08.json"
    for name, expected in json.loads(baseline.read_text())["files"].items():
        path = (W / "profile_run_01") / name
        if identity(path) != expected:
            raise RuntimeError("closed P partial record changed: " + name)
        records[str(path)] = expected
    for path in (
        baseline,
        W / "profile_partial_readback_08.json",
        W / "astra_profile_outcome_review_08.md",
    ):
        records[str(path)] = identity(path)
    partial = json.loads((W / "profile_partial_readback_08.json").read_text())
    if (
        partial.get("readback_passed") is not True
        or partial.get("contract_passed") is not False
    ):
        raise RuntimeError("Q requires honestly closed failed P evidence")
    return records, paths


def checked_go(records):
    go = json.loads(GO.read_text())
    if go.get("decision") != "GO" or go.get("contract_sha256") != CONTRACT_SHA:
        raise RuntimeError("missing exact Astra 08-Q GO")
    for path in (
        Path(__file__).resolve(),
        W / "compact08/run_compact_q08.py",
        W / "next_compact_contract_08q.md",
    ):
        if go.get("inputs", {}).get(str(path)) != identity(path):
            raise RuntimeError("GO does not bind execution input: " + str(path))
    for name, expected in go["inputs"].items():
        if identity(Path(name)) != expected:
            raise RuntimeError("GO input drift: " + name)
        records[name] = expected
    pure = Path(go["pure_receipt_path"])
    if identity(pure)["sha256"] != go["pure_receipt_sha256"]:
        raise RuntimeError("pure receipt identity mismatch")
    receipt = json.loads(pure.read_text())
    if (
        not receipt.get("all_passed")
        or receipt.get("exit_code") != 0
        or not 1 <= receipt.get("collected", 0) <= 8
        or receipt.get("blocked_engine_imports") != []
        or receipt.get("sources_changed_during_tests") != []
    ):
        raise RuntimeError("pure receipt did not pass bounded cases")
    records[str(pure)] = identity(pure)
    records[str(GO)] = identity(GO)
    return go


def cleanup_owned_worker():
    report = {
        "child_receipt_present": False,
        "signals_sent": [],
        "live_owned_pids_before": [],
        "owned_group_alive_after_cleanup": False,
    }
    child_path = OUT / "child_pid.json"
    if not child_path.is_file():
        return report
    report["child_receipt_present"] = True
    child = json.loads(child_path.read_text())
    session = json.loads((OUT / "session.json").read_text())
    expected = [
        "rtk",
        "proxy",
        "/usr/bin/python3",
        "-B",
        str(W / "compact08/run_compact_q08.py"),
        "--session",
        str(OUT / "session.json"),
    ]
    if child["argv"] != expected or session["worker_argv"] != expected:
        raise RuntimeError("cleanup worker argv differs from reserved08 worker")
    group = child["pid"]
    if type(group) is not int or group <= 1:
        raise RuntimeError("invalid owned worker group")

    def members():
        found = []
        for folder in Path("/proc").iterdir():
            if not folder.name.isdigit():
                continue
            try:
                state = (folder / "stat").read_text().rsplit(")", 1)[1].split()
                if int(state[2]) != group or state[0] == "Z":
                    continue
                argv = (folder / "cmdline").read_bytes().split(b"\0")
                if (
                    str(W / "compact08/run_compact_q08.py").encode() not in argv
                    or str(OUT / "session.json").encode() not in argv
                ):
                    raise RuntimeError(
                        "cleanup refuses foreign process in reused group"
                    )
                found.append(int(folder.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
        return found

    report["live_owned_pids_before"] = members()
    for number, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 3)):
        if not members():
            break
        os.killpg(group, number)
        report["signals_sent"].append(int(number))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and members():
            time.sleep(0.05)
    report["owned_group_alive_after_cleanup"] = bool(members())
    return report


def run_child():
    if not RESERVATION.is_file() or OUT.exists():
        raise RuntimeError("worker has no fresh root reservation")
    records, paths = source_preflight()
    checked_go(records)
    reserve = json.loads(RESERVATION.read_text())
    if (
        reserve["go_sha256"] != identity(GO)["sha256"]
        or reserve["control_limit"] != 800
    ):
        raise RuntimeError("reservation/GO mismatch")
    child_env, environment = helper.child_environment(R, paths)
    auth = Path(child_env.get("XAUTHORITY", ""))
    if (
        not child_env.get("DISPLAY", "").startswith(":")
        or auth.parent != OUT.with_name(OUT.name + ".x11")
        or not auth.is_file()
    ):
        raise RuntimeError("worker display is not the private X11 reservation")
    command = [
        "rtk",
        "proxy",
        "/usr/bin/python3",
        "-B",
        str(W / "compact08/run_compact_q08.py"),
        "--session",
        str(OUT / "session.json"),
    ]
    session = {
        "schema": "d1-compact-equivalence-q08-v1",
        "mode": "validation",
        "mode_semantics": "new compact equivalence path; no old07 segment call",
        "actor": "final_policy",
        "terrain": "box",
        "speed_mps": 0.20,
        "seed": 77351,
        "render_quality": "low",
        "segments": [400, 400],
        "control_limit": 800,
        "normal_native_limit": 4000,
        "compiler_native_limit": 3,
        "wallclock_limit_s": 600,
        "output_directory": str(OUT),
        "session_path": str(OUT / "session.json"),
        "source_hashes": records,
        "runtime_environment": environment,
        "argv": command[4:],
        "worker_argv": command,
        "contract_path": str(W / "next_compact_contract_08q.md"),
        "contract_sha256": CONTRACT_SHA,
        "retry_permitted": False,
        "old_budget_reused": False,
        "input_origin": "scripted_logical_compact08",
        "training_protocol": str(paths["protocol"]),
        "protocol_path": str(paths["protocol"]),
        "protocol_sha256": helper.PROTOCOL_SHA256,
        "published_binding_source": str(paths["binding_module"]),
        "import_repair_source": str(paths["continuation_import_repair"]),
        **{
            key: str(paths[key])
            for key in (
                "library",
                "checkpoint_model",
                "checkpoint_metadata",
                "checkpoint_probe",
                "published_body_source",
                "binding_module",
                "continuation_import_repair",
                "published05_manifest",
                "published06_manifest",
                "original_run",
                "body_work",
            )
        },
    }
    OUT.mkdir(exist_ok=False)
    save(OUT / "session.json", session)
    process = None
    code = None
    failure = None
    started = time.monotonic()

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt("08-Q launcher interrupted")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (OUT / "worker_stdout.log").open("xb") as stream:
            process = subprocess.Popen(
                command,
                cwd=R,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            save(OUT / "child_pid.json", {"pid": process.pid, "argv": command})
            code = process.wait(timeout=600)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        if process is not None:
            helper.stop_owned(process, reason="08-Q closure")
            code = process.returncode
        signal.signal(signal.SIGTERM, previous)
        changed = [
            name
            for name, expected in records.items()
            if not Path(name).is_file() or identity(Path(name)) != expected
        ]
        save(
            OUT / "launcher_receipt.json",
            {
                "schema": "d1-full-drive-compact08-launcher-v1",
                "exit_code": code,
                "failure": failure,
                "source_hash_mismatches": changed,
                "worker_exited": process is None or process.poll() is not None,
                "elapsed_wall_s": time.monotonic() - started,
                "retry_permitted": False,
            },
        )
    return 0 if code == 0 and failure is None and not changed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--child-launcher", action="store_true")
    args = parser.parse_args()
    if args.child_launcher:
        return run_child()
    records, _ = source_preflight()
    if args.check:
        print(
            json.dumps(
                {
                    "source_preflight_passed": True,
                    "input_count": len(records),
                    "engine_imported": False,
                }
            )
        )
        return 0
    checked_go(records)
    if (
        RESERVATION.exists()
        or OUT.exists()
        or OUT.with_name(OUT.name + ".x11").exists()
    ):
        raise RuntimeError("08-Q already reserved; no retry")
    command = [
        "rtk",
        "proxy",
        "/usr/bin/python3",
        "-B",
        str(Path(__file__).resolve()),
        "--child-launcher",
    ]
    save(
        RESERVATION,
        {
            "schema": "d1-full-drive-compact08-root-reservation-v1",
            "control_limit": 800,
            "normal_native_limit": 4000,
            "compiler_native_limit": 3,
            "go_sha256": identity(GO)["sha256"],
            "argv": command,
            "automatic_retry": False,
            "output": str(OUT),
        },
    )
    isolate = load_module("private_x11_compact08", B / "gui_isolation/isolated_x11.py")
    code = None
    failure = None
    try:
        code = isolate.run(command, OUT)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        cleanup = cleanup_owned_worker()
        save(
            W / "compact_root_exit_08.json",
            {
                "exit_code": code,
                "failure": failure,
                "cleanup": cleanup,
                "retry_permitted": False,
            },
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
