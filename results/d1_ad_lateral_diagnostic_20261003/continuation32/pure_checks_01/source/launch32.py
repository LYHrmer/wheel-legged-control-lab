"""Single reserved, wall-bounded saved-data diagnostic; no robot imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, obj):
    with path.open("x") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve(strict=True) for p in
             (args.contract, args.script, args.review, Path(__file__))]
    review = json.loads(args.review.read_text())
    if review.get("decision") != "GO_SINGLE_SAVED_DIAGNOSTIC":
        raise RuntimeError("Astra source review has not granted this diagnostic")
    bound = review["source_hashes"]
    for path in (paths[0], paths[1], paths[3]):
        if bound[str(path)] != identity(path)["sha256"]:
            raise RuntimeError("Source differs from reviewed diagnostic")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.execution.mkdir(parents=False, exist_ok=False)
    before = [identity(p) for p in paths]
    command = ["rtk", "proxy", "/usr/bin/python3", "-B", str(paths[1]),
               "--contract", str(paths[0]), "--output", str(args.output.resolve())]
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH="/home/lyh/wheel-legged-control-lab/.local-deps")
    write_new(args.execution / "reservation.json", {
        "argv": command, "wall_limit_s": 300, "source_before": before,
        "retry_permitted": False, "root_pid": os.getpid()})
    started = time.monotonic()
    timed_out = False
    failure = None
    child = None
    code = None
    try:
        with (args.execution / "stdout.log").open("x") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            try:
                code = child.wait(timeout=max(0.0, started + 295 - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=max(0.0, started + 300 - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        after = [identity(p) for p in paths]
        write_new(args.execution / "receipt.json", {
            "argv": command, "exit_code": code, "timed_out": timed_out,
            "elapsed_s": time.monotonic() - started, "failure": failure,
            "wrapper_pid": None if child is None else child.pid,
            "source_before": before, "source_after": after,
            "source_unchanged": before == after,
            "output_exists": args.output.exists(),
            "physics_and_model_counts_source": "diagnostic result, not inferred from exit code"})
    if code != 0 or timed_out or before != after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
