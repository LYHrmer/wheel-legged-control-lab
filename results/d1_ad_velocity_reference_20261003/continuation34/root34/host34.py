"""C34 outer deadline over the inherited owned C31 subreaper host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from host31 import clean_descendants, execute, subreaper, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--owned-worker-host", action="store_true")
    parser.add_argument("--isolated-x11", type=Path)
    parser.add_argument("--isolation-child", action="store_true")
    args = parser.parse_args()
    request_path = args.request.resolve(strict=True)
    request = json.loads(request_path.read_text())
    spec = request["spec"]
    if (request["arm"] != "development" or spec["outer_s"] != 1200
            or spec["hard_s"] != 1020 or spec["soft_s"] != 900
            or spec["render"] is not False or args.isolated_x11 is not None):
        raise RuntimeError("C34 owned headless host profile differs")
    if args.isolation_child:
        raise RuntimeError("C34 does not reserve an X11 child")
    if args.owned_worker_host:
        return execute(request_path)
    subreaper()
    command = ["rtk", "proxy", "/usr/bin/python3", "-B",
               str(Path(__file__).resolve()), "--request", str(request_path),
               "--owned-worker-host"]
    started = time.monotonic()
    process = None
    failure = None
    previous = signal.signal(signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(InterruptedError("C34 supervisor termination")))
    try:
        process = subprocess.Popen(command, start_new_session=True)
        process.wait(timeout=1195)
    except subprocess.TimeoutExpired:
        failure = "C34 outer host hard deadline"
    except BaseException as error:
        failure = type(error).__name__+": "+str(error)
    finally:
        cleanup = clean_descendants(__import__("os").getpid(), process)
        signal.signal(signal.SIGTERM, previous)
    output = Path(spec["output_directory"])
    record = {"schema": "d1-c34-supervisor-v1", "outer_limit_s": 1200,
              "elapsed_s": time.monotonic()-started, "failure": failure,
              "exit_code": None if process is None else process.returncode,
              "cleanup": cleanup, "includes_isolated_x11_startup": False}
    if output.is_dir():
        write(output/"supervisor_receipt.json", record)
    else:
        print(json.dumps(record), flush=True)
    return 0 if (failure is None and process is not None and process.returncode == 0
                 and not cleanup["remaining"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
