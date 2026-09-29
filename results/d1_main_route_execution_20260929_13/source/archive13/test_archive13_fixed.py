"""Pure archive fault injection; no engine, policy, or native step is imported."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from continuation13.archive13 import atomic_archive_13 as archive


class ArchiveWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.writer = archive.ArchiveWriter()
        self.native = self.folder / "native_block_0000.jsonl.gz"
        self.arrays = self.folder / "native_arrays_0000.npz"
        self.rows = [{"native_index": 0, "force": 1.25}]
        self.values = {"qpos": np.asarray([[1.0, 2.0]])}

    def test_nominal_two_payload_commit_is_valid_only_with_manifest(self) -> None:
        receipt = self.writer.commit_native_block(self.native, self.rows,
                                                  self.arrays, self.values)
        manifest = self.folder / receipt["manifest"]
        self.assertTrue(archive.verify_manifest(manifest))
        self.assertEqual([item["file"] for item in receipt["payloads"]],
                         [self.native.name, self.arrays.name])
        with gzip.open(self.native, "rt", encoding="utf-8") as stream:
            self.assertEqual(json.loads(stream.readline()), self.rows[0])
        with np.load(self.arrays) as stored:
            np.testing.assert_array_equal(stored["qpos"], self.values["qpos"])
        self.assertFalse(self.writer.partial)

    def test_mid_gzip_write_keeps_partial_and_never_publishes_manifest(self) -> None:
        original = gzip.GzipFile.write

        def fail_after_write(stream, data):
            original(stream, data)
            raise OSError("write interrupted")

        with (
            patch.object(gzip.GzipFile, "write", fail_after_write),
            self.assertRaisesRegex(OSError, "write interrupted"),
        ):
            self.writer.commit_gzip_rows(self.native, self.rows)
        self.assertFalse(self.native.exists())
        self.assertTrue(self.writer.partial)
        self.assertTrue(Path(self.writer.partial[0]).exists())
        self.assertFalse(archive.verify_manifest(self.folder / (self.native.name + ".manifest.json")))

    def test_gzip_close_failure_keeps_uncommitted_partial(self) -> None:
        original = gzip.GzipFile.__exit__

        def fail_after_close(stream, exc_type, exc_value, traceback):
            original(stream, exc_type, exc_value, traceback)
            raise OSError("gzip close interrupted")

        with (
            patch.object(gzip.GzipFile, "__exit__", fail_after_close),
            self.assertRaisesRegex(OSError, "gzip close interrupted"),
        ):
            self.writer.commit_gzip_rows(self.native, self.rows)
        self.assertFalse(self.native.exists())
        self.assertTrue(self.writer.partial)
        self.assertFalse(archive.verify_manifest(self.folder / (self.native.name + ".manifest.json")))

    def test_fsync_failure_keeps_uncommitted_partial(self) -> None:
        with (
            patch.object(archive.os, "fsync", side_effect=OSError("fsync interrupted")),
            self.assertRaisesRegex(OSError, "fsync interrupted"),
        ):
            self.writer.commit_gzip_rows(self.native, self.rows)
        self.assertFalse(self.native.exists())
        self.assertTrue(self.writer.partial)

    def test_link_failure_cannot_publish_manifest(self) -> None:
        with (
            patch.object(archive.os, "link", side_effect=OSError("link interrupted")),
            self.assertRaisesRegex(OSError, "link interrupted"),
        ):
            self.writer.commit_gzip_rows(self.native, self.rows)
        self.assertFalse(self.native.exists())
        self.assertTrue(self.writer.partial)
        self.assertFalse(archive.verify_manifest(self.folder / (self.native.name + ".manifest.json")))

    def test_existing_target_is_never_overwritten(self) -> None:
        self.native.write_bytes(b"owned by earlier run")
        with self.assertRaises(FileExistsError):
            self.writer.commit_gzip_rows(self.native, self.rows)
        self.assertEqual(self.native.read_bytes(), b"owned by earlier run")
        self.assertTrue(self.writer.failed)
        self.assertFalse(archive.verify_manifest(self.folder / (self.native.name + ".manifest.json")))

    def test_second_payload_link_failure_leaves_both_unqualified(self) -> None:
        original = archive.os.link
        calls = 0

        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("second payload link interrupted")
            original(source, target)

        with (
            patch.object(archive.os, "link", fail_second),
            self.assertRaisesRegex(OSError, "second payload link interrupted"),
        ):
            self.writer.commit_native_block(self.native, self.rows,
                                            self.arrays, self.values)
        self.assertTrue(self.native.exists())  # Orphan payload is not a valid block.
        self.assertFalse(self.arrays.exists())
        self.assertFalse(archive.verify_manifest(self.folder / (self.native.name + ".manifest.json")))
        self.assertTrue(self.writer.partial)

    def test_manifest_link_failure_keeps_payloads_uncommitted(self) -> None:
        original = archive.os.link
        calls = 0

        def fail_manifest(source, target):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("manifest link interrupted")
            original(source, target)

        with (
            patch.object(archive.os, "link", fail_manifest),
            self.assertRaisesRegex(OSError, "manifest link interrupted"),
        ):
            self.writer.commit_native_block(self.native, self.rows,
                                            self.arrays, self.values)
        self.assertTrue(self.native.exists())
        self.assertTrue(self.arrays.exists())
        self.assertFalse(archive.verify_manifest(self.folder / (self.native.name + ".manifest.json")))
        self.assertTrue(self.writer.failed)
        self.assertFalse(self.writer.committed)

    def test_manifest_directory_fsync_failure_removes_own_manifest(self) -> None:
        calls = 0

        def fail_second_directory_fsync(_folder):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("manifest directory fsync interrupted")

        with (
            patch.object(archive, "_fsync_directory", fail_second_directory_fsync),
            self.assertRaisesRegex(OSError, "manifest directory fsync interrupted"),
        ):
            self.writer.commit_native_block(self.native, self.rows,
                                            self.arrays, self.values)
        self.assertGreaterEqual(calls, 2)
        self.assertTrue(self.native.exists())
        self.assertTrue(self.arrays.exists())
        self.assertFalse(archive.verify_manifest(self.folder / (self.native.name + ".manifest.json")))
        self.assertTrue(self.writer.failed)
        self.assertFalse(self.writer.committed)

    def test_failed_writer_blocks_new_payload_but_allows_one_failure_receipt(self) -> None:
        with (
            patch.object(archive.os, "link", side_effect=OSError("original link failure")),
            self.assertRaisesRegex(OSError, "original link failure"),
        ):
            self.writer.commit_gzip_rows(self.native, self.rows)
        other = self.folder / "later.json"
        with self.assertRaisesRegex(OSError, "original link failure"):
            self.writer.commit_json(other, {"must_not_publish": True})
        self.assertFalse(other.exists())
        failure_path = self.folder / "worker_failure_receipt.json"
        receipt = self.writer.commit_failure_receipt(
            failure_path, {"failure": {"type": "OSError", "message": "original link failure"},
                           "record_valid": False})
        self.assertTrue(archive.verify_manifest(self.folder / receipt["manifest"]))
        self.assertTrue(self.writer.failed)
        with self.assertRaises(RuntimeError):
            self.writer.commit_failure_receipt(failure_path, {"failure": "again"})

    def test_soft_stop_only_latches_and_does_not_interrupt_a_commit(self) -> None:
        self.writer.request_stop()
        self.writer.request_stop()
        self.assertTrue(self.writer.stop_requested)
        receipt = self.writer.commit_native_block(self.native, self.rows,
                                                  self.arrays, self.values)
        self.assertTrue(archive.verify_manifest(self.folder / receipt["manifest"]))


class GuardFinishTests(unittest.TestCase):
    def test_successful_finish_is_idempotent_and_manifest_backed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = archive.ArchiveWriter()
            guard = archive.AtomicCourseNativeGuard(SimpleNamespace(native_monitor=None), writer,
                                                    mode="heldout")
            guard._segment = {
                "name": "heldout_complete", "folder": directory, "mode": "heldout",
                "control_limit": 1, "native_limit": 5,
                "attempted_start": 0, "returned_start": 0,
                "native_files": [], "train_array_files": [],
                "archive_block_manifests": [], "task_contact_failure_files": [],
            }
            guard._native_block = [{"native_index": 0}]
            guard.attempted = guard.returned = 5
            first = guard.finish_segment()
            second = guard.finish_segment()
            self.assertEqual(first, second)
            self.assertTrue(first["record_valid"])
            self.assertEqual(len(first["archive_block_manifests"]), 1)
            self.assertTrue(archive.verify_manifest(Path(directory)
                                                    / first["archive_block_manifests"][0]))
            self.assertEqual(len(writer.committed), 2)

    def test_failed_flush_is_not_retried_or_masked_by_case_finally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = archive.ArchiveWriter()
            guard = archive.AtomicCourseNativeGuard(SimpleNamespace(native_monitor=None), writer,
                                                    mode="heldout")
            guard._segment = {
                "name": "heldout_interrupted", "folder": directory, "mode": "heldout",
                "control_limit": 1, "native_limit": 5,
                "attempted_start": 0, "returned_start": 0,
                "native_files": [], "train_array_files": [],
                "archive_block_manifests": [], "task_contact_failure_files": [],
            }
            guard._native_block = [{"native_index": 0}]
            with patch.object(writer, "commit_native_block",
                              side_effect=OSError("first archive error")) as commit:
                with self.assertRaisesRegex(OSError, "first archive error"):
                    try:
                        guard._flush_block()
                    finally:
                        sealed = guard.finish_segment()
                again = guard.finish_segment()
            self.assertEqual(commit.call_count, 1)
            self.assertEqual(sealed, again)
            self.assertFalse(sealed["record_valid"])
            self.assertEqual(sealed["archive_failure"]["message"], "first archive error")


if __name__ == "__main__":
    unittest.main()
