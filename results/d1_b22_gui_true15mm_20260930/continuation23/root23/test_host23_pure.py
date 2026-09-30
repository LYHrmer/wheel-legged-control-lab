"""Process ownership tests only; no simulation, model or project environment."""
import os
import signal
import subprocess
import time

from host23 import clean_descendants, descendants, signal_identity, subreaper


def test_subreaper_cleans_setsid_child_without_original_process_group():
    subreaper()
    child = subprocess.Popen(['rtk','proxy','/usr/bin/python3','-c',
                              'import os,time; os.setsid(); time.sleep(60)'])
    try:
        deadline = time.monotonic()+2
        escaped = []
        while time.monotonic()<deadline:
            escaped = [pid for pid in descendants(os.getpid()) if os.getpgid(pid)==pid]
            if escaped:
                break
            time.sleep(.01)
        assert escaped, 'the actual helper must have entered its separate session'
        owned = descendants(os.getpid())
        assert child.pid in owned
        signal_identity(child.pid,owned[child.pid]+1,signal.SIGKILL)
        assert child.poll() is None, 'birth mismatch must never signal another identity'
        receipt = clean_descendants(os.getpid(),child)
        assert not receipt['remaining']
        assert receipt['elapsed_s'] < 5
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1)
