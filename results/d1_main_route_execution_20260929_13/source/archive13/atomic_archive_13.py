"""Durable, no-clobber archive transactions for a later bounded run.

Only a durable manifest makes payloads committed. This module performs file
I/O only; it never loads a policy, imports an engine, or runs physics.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA = "d1-archive-transaction-13-v1"
Payload = tuple[Path, Callable[[BinaryIO], None]]


def _fsync_directory(folder: Path) -> None:
    descriptor = os.open(folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"file": path.name, "bytes": size, "sha256": digest.hexdigest()}


def verify_manifest(path: Path) -> bool:
    """Verify saved bytes, not fsync durability or the enclosing run's success.

    A successful owner receipt and the writer's no-error close are separate
    requirements. An orphan payload, or a manifest removed after failed
    publication, cannot pass this check.
    """
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or not manifest.get("payloads"):
            return False
        for item in manifest["payloads"]:
            filename = item["file"]
            if Path(filename).name != filename:
                return False
            if _identity(path.parent / filename) != item:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


class ArchiveWriter:
    """Stage each payload exclusively, then link payloads and manifest last."""

    def __init__(self) -> None:
        self.committed: list[str] = []
        self.partial: list[str] = []
        self.stop_requested = False
        self._first_error: BaseException | None = None
        self.failure_receipts: list[str] = []
        self.failure_receipt_attempted = False
        self.failure_receipt_error: str | None = None

    @property
    def failed(self) -> bool:
        return self._first_error is not None

    def request_stop(self) -> None:
        """Safe for a signal callback: the owner observes this at a tick boundary."""
        self.stop_requested = True

    @staticmethod
    def _partial_path(final: Path) -> Path:
        return final.with_name(f".{final.name}.{uuid.uuid4().hex}.partial")

    @staticmethod
    def _stage(path: Path, write: Callable[[BinaryIO], None]) -> None:
        with path.open("xb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())

    def _commit(self, payloads: list[Payload]) -> dict[str, Any]:
        if self._first_error is not None:
            raise self._first_error
        try:
            return self._commit_once(payloads)
        except BaseException as error:
            self._first_error = error
            raise

    def _commit_once(self, payloads: list[Payload]) -> dict[str, Any]:
        if not payloads:
            raise ValueError("archive transaction needs at least one payload")
        finals = [path for path, _ in payloads]
        folder = finals[0].parent
        if (not folder.is_dir() or any(path.parent != folder for path in finals)
                or len(set(finals)) != len(finals)):
            raise ValueError("archive transaction requires distinct same-folder targets")
        manifest = finals[0].with_name(finals[0].name + ".manifest.json")
        if any(os.path.lexists(path) for path in (*finals, manifest)):
            raise FileExistsError("archive target or manifest already exists")

        staged: list[tuple[Path, Path]] = []
        linked: list[Path] = []
        manifest_stage: Path | None = None
        manifest_linked = False
        try:
            for final, write in payloads:
                temporary = self._partial_path(final)
                staged.append((temporary, final))
                self._stage(temporary, write)

            receipt = {
                "schema": SCHEMA,
                "payloads": [_identity(temporary) | {"file": final.name}
                             for temporary, final in staged],
            }
            for temporary, final in staged:
                os.link(temporary, final)
                linked.append(final)
            _fsync_directory(folder)

            manifest_stage = self._partial_path(manifest)
            encoded = (json.dumps(receipt, sort_keys=True, allow_nan=False,
                                  separators=(",", ":")) + "\n").encode("utf-8")
            self._stage(manifest_stage, lambda stream: stream.write(encoded))
            os.link(manifest_stage, manifest)
            manifest_linked = True
            _fsync_directory(folder)
        except BaseException:
            if manifest_linked:
                try:
                    if os.stat(manifest).st_ino == os.stat(manifest_stage).st_ino:
                        manifest.unlink()
                        _fsync_directory(folder)
                except OSError:
                    # The first publication error remains authoritative.
                    pass
            self.partial.extend(str(path) for path, _ in staged if path.exists())
            if manifest_stage is not None and manifest_stage.exists():
                self.partial.append(str(manifest_stage))
            self.partial.extend(str(path) for path in linked)
            raise

        self.committed.extend(str(path) for path in (*finals, manifest))
        for temporary in (path for path, _ in staged):
            try:
                temporary.unlink()
            except OSError:
                self.partial.append(str(temporary))
        if manifest_stage is not None:
            try:
                manifest_stage.unlink()
            except OSError:
                self.partial.append(str(manifest_stage))
        return {"manifest": manifest.name, "payloads": receipt["payloads"]}

    def commit_json(self, path: Path, value: Any) -> dict[str, Any]:
        def write(stream: BinaryIO) -> None:
            encoded = (json.dumps(value, sort_keys=True, allow_nan=False,
                                  separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(encoded)

        return self._commit([(path, write)])

    def commit_gzip_rows(self, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
        def write(stream: BinaryIO) -> None:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                for row in rows:
                    compressed.write((json.dumps(row, sort_keys=True, allow_nan=False,
                                                 separators=(",", ":")) + "\n").encode())

        return self._commit([(path, write)])

    def commit_npz(self, path: Path, arrays: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        return self._commit([(path, lambda stream: np.savez(stream, **arrays))])

    def commit_native_block(
        self,
        gzip_path: Path,
        rows: list[dict[str, Any]],
        npz_path: Path | None = None,
        arrays: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train's gzip and npz are one transaction, valid only together."""
        if (npz_path is None) != (arrays is None):
            raise ValueError("native array path and values must be supplied together")

        def write_rows(stream: BinaryIO) -> None:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                for row in rows:
                    compressed.write((json.dumps(row, sort_keys=True, allow_nan=False,
                                                 separators=(",", ":")) + "\n").encode())

        payloads: list[Payload] = [(gzip_path, write_rows)]
        if npz_path is not None:
            import numpy as np

            payloads.append((npz_path, lambda stream: np.savez(stream, **arrays)))
        return self._commit(payloads)

    def commit_failure_receipt(self, path: Path, value: dict[str, Any]) -> dict[str, Any]:
        """One independent best-effort receipt after the main writer fails.

        This does not clear its error or permit another normal block. Callers
        must mark the enclosing run failed even if this receipt is durable.
        """
        if not self.failed or self.failure_receipt_attempted:
            raise RuntimeError("failure receipt requires one failed writer attempt")
        if not isinstance(value, dict) or value.get("failure") is None:
            raise ValueError("failure receipt must disclose the run failure")
        self.failure_receipt_attempted = True
        rescue = ArchiveWriter()
        try:
            receipt = rescue.commit_json(path, value)
        except BaseException as error:
            self.partial.extend(rescue.partial)
            self.failure_receipt_error = repr(error)
            raise
        self.failure_receipts.extend(rescue.committed)
        return receipt


def writer_factory() -> ArchiveWriter:
    return ArchiveWriter()


from course_impl08.course_native_guard_08 import (
    CourseNativeGuard,
    _json_safe,
)


class AtomicCourseNativeGuard(CourseNativeGuard):
    """Use one archive transaction at the old guard's actual block flush seam."""

    def __init__(self, runtime: Any, writer: ArchiveWriter, **kwargs: Any) -> None:
        super().__init__(runtime, **kwargs)
        self.writer = writer
        self._archive_error: BaseException | None = None
        self._finished_segment_row: dict[str, Any] | None = None

    def start_segment(self, *args: Any, **kwargs: Any) -> None:
        if self._archive_error is not None:
            raise RuntimeError("failed archive cannot start another segment")
        super().start_segment(*args, **kwargs)
        self._segment["archive_block_manifests"] = []
        self._finished_segment_row = None

    def request_soft_stop(self) -> None:
        self.writer.request_stop()

    @property
    def soft_stop_requested(self) -> bool:
        return self.writer.stop_requested

    def _write_gzip_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        self.writer.commit_gzip_rows(path, rows)

    def _flush_block(self) -> None:
        if self._archive_error is not None:
            raise self._archive_error
        if not self._native_block:
            return
        if self._segment is None:
            raise RuntimeError("cannot flush native block without a segment")
        folder = Path(self._segment["folder"])
        native = folder / f"native_block_{self._block_index:04d}.jsonl.gz"
        train = self._segment["mode"] == "train"
        arrays_path = folder / f"native_arrays_{self._block_index:04d}.npz" if train else None
        try:
            arrays = None
            if train:
                import numpy as np

                arrays = {key: np.stack(values) for key, values in self._train_arrays.items()}
            receipt = self.writer.commit_native_block(native, self._native_block,
                                                      arrays_path, arrays)
        except BaseException as error:
            self._archive_error = error
            raise
        self._segment["native_files"].append(native.name)
        if arrays_path is not None:
            self._segment["train_array_files"].append(arrays_path.name)
            self._train_arrays = {}
        self._segment["archive_block_manifests"].append(receipt["manifest"])
        self._native_block = []
        self._block_index += 1
        self._controls_in_block = 0

    def _record_task_contact_failure(self, row: dict[str, Any]) -> None:
        if self._segment is None:
            raise RuntimeError("contact failure has no course segment")
        path = (Path(self._segment["folder"])
                / f"native_contact_failure_{self._contact_failure_index:04d}.json")
        try:
            self.writer.commit_json(path, _json_safe(row))
        except BaseException as error:
            if self._archive_error is None:
                self._archive_error = error
            raise
        self._segment["task_contact_failure_files"].append(path.name)
        self._contact_failure_index += 1
        if self.failure_sink is not None:
            self.failure_sink(row)

    def finish_segment(self) -> dict[str, Any]:
        """Seal once; an archive failure cannot trigger a retry in finally."""
        if self._segment is None:
            if self._finished_segment_row is not None:
                return dict(self._finished_segment_row)
            raise RuntimeError("no course segment to close")

        primary = sys.exc_info()[1]
        finish_error: BaseException | None = None
        if self._archive_error is None:
            try:
                self._flush_block()
            except BaseException as error:  # noqa: BLE001 - preserve active first error
                finish_error = error

        segment = self._segment
        native_attempted = self.attempted - segment["attempted_start"]
        native_returned = self.returned - segment["returned_start"]
        partial_interval = bool(self._interval_open or self._interval)
        if native_attempted > segment["native_limit"]:
            budget_error = RuntimeError("course native segment exceeded its reservation")
            if self._archive_error is None:
                self._archive_error = budget_error
                if finish_error is None:
                    finish_error = budget_error
        archive_failure = None if self._archive_error is None else {
            "type": type(self._archive_error).__name__,
            "message": str(self._archive_error),
        }
        valid = archive_failure is None and self.failure is None and not partial_interval
        row = {
            "name": segment["name"], "folder": segment["folder"],
            "mode": segment["mode"], "control_limit": segment["control_limit"],
            "native_limit": segment["native_limit"],
            "native_attempted": native_attempted, "native_returned": native_returned,
            "partial_native_interval": partial_interval,
            "failure": self.failure, "archive_failure": archive_failure,
            "record_valid": valid,
            "native_files": list(segment["native_files"]),
            "train_array_files": list(segment["train_array_files"]),
            "archive_block_manifests": list(segment["archive_block_manifests"]),
            "task_contact_failure_files": list(segment["task_contact_failure_files"]),
            "force_sampling_performed": segment["mode"] != "train",
            "full_contact_qualification_recorded": segment["mode"] != "train" and valid,
        }
        self.segments.append(row)
        self._segment = None
        self._interval = []
        self._interval_open = False
        self._finished_segment_row = row
        if finish_error is not None and primary is None:
            raise finish_error
        return dict(row)
