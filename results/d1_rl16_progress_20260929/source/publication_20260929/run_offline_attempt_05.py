"""Durable host for a read-only reader after daemon interruption."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

W = Path("/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01")
D = W / "rl11"
OUT = W / "publication_20260929/offline_attempt_05"
READER = D / "verify_short_rl16_partial_11_04.py"
SHA = "da2a8ae34cf999ea8cfecfc699d201069324191d43b4fdb5044e5c788904b388"

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
           "--output", str(D / "partial_readback_20260929_05.json"),
           "--manifest-output", str(D / "partial_closed_manifest_20260929_05.json")]
    save(OUT / "invocation.json", {"argv":cmd, "reader_sha256":SHA,
         "previous_attempt_04":"reader03_incorrect_pre_gate_vs_post_gate_actor_equivalence",
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

