"""Pure reproduction of the 11-S gzip interruption and case-finally path."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "course_impl08/course_native_guard_08.py"
SPEC = importlib.util.spec_from_file_location("course_native_guard_08_archive13", SOURCE)
assert SPEC is not None and SPEC.loader is not None
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


class TerminationRequested(Exception):
    pass


def make_guard(folder: Path):
    guard = legacy.CourseNativeGuard(SimpleNamespace(native_monitor=None), mode="heldout")
    guard._segment = {
        "name": "heldout_interrupted", "folder": str(folder), "mode": "heldout",
        "control_limit": 1, "native_limit": 5,
        "attempted_start": 0, "returned_start": 0,
        "native_files": [], "train_array_files": [], "task_contact_failure_files": [],
    }
    guard._native_block = [{"native_index": 0}]
    return guard


class ArchiveInterruptionRegression(unittest.TestCase):
    def test_case_finally_preserves_original_gzip_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            guard = make_guard(Path(directory))
            with patch.object(legacy.json, "dumps", side_effect=TerminationRequested("soft stop")):
                with self.assertRaisesRegex(TerminationRequested, "soft stop"):
                    try:
                        guard._flush_block()
                    finally:
                        # The old record_heldout_case finally closes the guard again.
                        guard.finish_segment()


if __name__ == "__main__":
    unittest.main()
