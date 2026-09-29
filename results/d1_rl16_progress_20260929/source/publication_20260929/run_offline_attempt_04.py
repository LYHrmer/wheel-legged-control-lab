"""Durable host for a read-only reader after daemon interruption."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

W = Path("/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01")
D = W / "rl11"
OUT = W / "publication_20260929/offline_attempt_04"
READER = D / "verify_short_rl16_partial_11_03.py"
SHA = "24d4384ffd4039343a5680c421ff7edc95bfd2021176280ec746817637e61dfe"

def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def main():
    if hashlib.sha256(READER.read_bytes()).hexdigest() != SHA:
        raise RuntimeError("reviewed reader source changed")
    OUT.mkdir(exist_ok=False)
    cmd = ["rtk", "proxy", "/usr/bin/python3", "-B", str(READER),
           "--run", str(D / "training_run_01"),
           "--output", str(D / "partial_readback_20260929_04.json"),
           "--manifest-output", str(D / "partial_closed_manifest_20260929_04.json")]
    save(OUT / "invocation.json", {"argv":cmd, "reader_sha256":SHA,
         "previous_tool_session_56761":"unrecoverable_after_daemon_restart_no_outputs_or_process",
         "operation":"saved_data_readback_only"})
    started=time.monotonic()
    with (OUT / "stdout.log").open("xb") as log:
        result=subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt={"exit_code":result.returncode,"elapsed_s":time.monotonic()-started,
             "reader_sha256":SHA,"operation":"saved_data_readback_only"}
    save(OUT / "receipt.json",receipt)
    print(json.dumps(receipt))
    if result.returncode:
        print((OUT / "stdout.log").read_text()[-10000:])
    return result.returncode
if __name__ == "__main__":
    raise SystemExit(main())

