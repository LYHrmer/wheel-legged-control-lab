"""Exercise the actual outer host against a fast orphaning dummy launcher."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class SubreaperTest(unittest.TestCase):
    def test_fast_setsid_orphan_is_recovered_and_missing_ready_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            launcher = folder / 'dummy_launcher.py'
            launcher.write_text(
                'import os,time\n'
                'if os.fork() == 0:\n'
                ' os.setsid()\n'
                ' time.sleep(60)\n'
                'else:\n'
                ' os._exit(0)\n')
            plan = folder / 'plan.json'
            plan.write_text(json.dumps({'inputs': {}, 'arms': {
                'headless': {'output_directory': str(folder / 'output')}}}))
            host = Path(__file__).with_name('run_stage13_root_host.py')
            result = subprocess.run(
                ['rtk', 'proxy', '/usr/bin/python3', '-B', str(host),
                 '--plan', str(plan), '--launcher', str(launcher),
                 '--arm', 'headless', '--receipt-directory', str(folder / 'receipt')],
                capture_output=True, text=True, timeout=15, check=False)
            self.assertEqual(result.returncode, 1, result.stderr)
            receipt = json.loads((folder / 'receipt/receipt.json').read_text())
            self.assertTrue(receipt['linux_child_subreaper'])
            self.assertFalse(receipt['preflight_ready_seen'])
            self.assertTrue(receipt['no_orphans'])
            self.assertFalse(receipt['surviving_owned_processes'])
            self.assertGreaterEqual(len(receipt['tracked_process_birth_ticks']), 2)


if __name__ == '__main__':
    unittest.main()
