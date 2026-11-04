"""Synthetic resource samples: never allocate memory to trigger pressure/OOM."""

import copy
import unittest

from scripts.resource_guard42 import ResourceGuardError, check_memory
from scripts.resource_pressure83 import check_pressure, parse_pressure


def memory_sample():
    return {
        "errors": [],
        "effective_available_bytes": 8150683648,
        "owned": {
            "events": {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0, "oom_group_kill": 0}
        },
        "raw_ancestors": [
            {"path": "/user.slice/user@1000.service/job83.service"},
            {"path": "/user.slice/user@1000.service"},
            {"path": "/"},
        ],
    }


def quiet():
    return {
        r["path"]: {
            "some": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 1},
            "full": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 1},
        }
        for r in memory_sample()["raw_ancestors"]
    }


class Pressure83(unittest.TestCase):
    def test_ancestor_pressure_fails_despite_7point59GiB_available(self):
        snap, psi = memory_sample(), quiet()
        psi["/user.slice/user@1000.service"]["full"]["avg10"] = 72.36
        # The frozen predecessor admits this evidence pattern; the new entry must not.
        self.assertIs(check_memory(snap), snap)
        with self.assertRaisesRegex(ResourceGuardError, "pressure"):
            check_pressure(snap, psi)

    def test_reclaim_event_fails_before_kernel_oom(self):
        snap = memory_sample()
        snap["owned"]["events"]["high"] = 10491
        self.assertIs(check_memory(snap), snap)
        with self.assertRaisesRegex(ResourceGuardError, "high"):
            check_pressure(snap, quiet())

    def test_quiet_receipt_is_preserved_without_mutating_inputs(self):
        snap, psi = memory_sample(), quiet()
        original = copy.deepcopy((snap, psi))
        self.assertIs(check_pressure(snap, psi), snap)
        self.assertEqual((snap, psi), original)

    def test_runtime_and_startup_boundary_and_unknown_pressure(self):
        for startup, limit in ((True, 1.0), (False, 10.0)):
            with self.subTest(startup=startup):
                psi = quiet()
                psi["/"]["full"]["avg10"] = limit - 0.0001
                check_pressure(memory_sample(), psi, startup=startup)
                psi["/"]["full"]["avg10"] = limit
                with self.assertRaises(ResourceGuardError):
                    check_pressure(memory_sample(), psi, startup=startup)
        for bad in (float("nan"), float("inf"), -1, True, None):
            psi = quiet()
            psi["/"]["full"]["avg10"] = bad
            with self.assertRaises(ResourceGuardError):
                check_pressure(memory_sample(), psi)
        psi = quiet()
        del psi["/"]
        with self.assertRaisesRegex(ResourceGuardError, "coverage"):
            check_pressure(memory_sample(), psi)

    def test_strict_pressure_parser(self):
        text = "some avg10=2.00 avg60=1.00 avg300=0.10 total=12\nfull avg10=1.00 avg60=0.20 avg300=0.00 total=10\n"
        self.assertEqual(parse_pressure(text)["full"]["total"], 10)
        for bad in (
            text.replace("1.00", "nan"),
            text.replace("total=12", "total=-1"),
            text.replace("avg60=1.00", "avg10=1.00"),
            text + text,
            text.splitlines()[0],
            text.replace("total=10", "total=20"),
        ):
            with self.assertRaises(ValueError):
                parse_pressure(bad)


if __name__ == "__main__":
    unittest.main()
