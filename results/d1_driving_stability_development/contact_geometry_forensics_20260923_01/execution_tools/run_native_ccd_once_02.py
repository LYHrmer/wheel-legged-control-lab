"""Root-owned one-shot launcher; no engine import and no retry path."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if not __debug__:
        raise RuntimeError("launcher requires Python assertion checks enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    assert str(Path(__file__).resolve()) in freeze["files"]
    assert freeze["astra_review_decision"] == "GO"
    assert freeze["contract_sha256"] == "0869d5836add5dc12a295d5ac4618bff984244c9703516202f9485a29fdf4768"
    assert freeze["max_process_attempts"] == 1
    assert freeze["max_native_ccd_attempts"] == 16
    assert freeze["new_control_budget"] == freeze["new_integration_budget"] == 0
    for path, value in freeze["files"].items():
        assert sha(path) == value["sha256"], path
        assert Path(path).stat().st_size == value["bytes"], path
    argv = freeze["execution_argv"]
    assert argv[:2] == ["rtk", "proxy"]
    assert Path(argv[2]).is_absolute() and argv[2] in freeze["files"]
    assert str(args.output) == freeze["output_directory"]
    assert str(args.budget) == freeze["budget_file"]
    assert not args.output.exists() and not args.budget.exists()
    args.output.mkdir()
    activation = {
        "status": "attempted_no_retry",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "attempted_processes": 1,
        "max_process_attempts": 1,
        "max_native_ccd_attempts": 16,
        "control_budget": 0,
        "native_integration_budget": 0,
        "model_compile_budget": 0,
        "engine_model_data_allocation_budget": 0,
        "source_freeze_sha256": sha(args.freeze),
        "execution_argv": argv,
        "query_counts_authority": "durable internal attempted events, including calls without returns",
    }
    save_new(args.budget, activation)
    save_new(args.output / "activation.json", activation)
    started = time.monotonic()
    error = None
    exit_code = None
    with (args.output / "stdout.log").open("x") as out, (args.output / "stderr.log").open("x") as err:
        try:
            proc = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
            try:
                exit_code = proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                error = "45_second_timeout_no_retry"
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    exit_code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    exit_code = proc.wait(timeout=3)
        except OSError as exc:
            error = type(exc).__name__ + ": " + str(exc)
    events = []
    parse_errors = []
    event_path = args.output / freeze["event_file_basename"]
    if event_path.is_file():
        for index, line in enumerate(event_path.read_text().splitlines()):
            try:
                events.append(json.loads(line))
            except (ValueError, TypeError) as exc:
                parse_errors.append({"line": index + 1, "error": str(exc)})
    post_hash_errors = [path for path, value in freeze["files"].items() if sha(path) != value["sha256"]]
    receipt = {
        "attempted_processes": 1,
        "process_exit_code": exit_code,
        "execution_error": error,
        "elapsed_seconds": time.monotonic() - started,
        "event_file": str(event_path),
        "event_histogram": dict(collections.Counter(str(event.get("event")) for event in events)),
        "event_count": len(events),
        "event_parse_errors": parse_errors,
        "source_hash_errors_after": post_hash_errors,
        "manual_counter_reconciliation_required": True,
        "qualification_granted": False,
        "original_score_overridden": False,
        "rl_gate_open": False,
        "retry_permitted": False,
    }
    save_new(args.output / "launcher_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return int(exit_code != 0 or error is not None or bool(parse_errors) or bool(post_hash_errors))


if __name__ == "__main__":
    raise SystemExit(main())
