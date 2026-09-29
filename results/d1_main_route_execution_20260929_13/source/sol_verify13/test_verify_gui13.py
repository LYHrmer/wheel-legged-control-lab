"""Small saved-byte fixtures for the GUI13 reader; no engine or model imports."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from verify_gui13 import (
    ARCHIVE,
    Invalid,
    check_acks,
    check_performance,
    exact_array,
    percentile95,
    verified_payload,
)


def seal(path: Path, data: bytes, companion: tuple[Path, bytes] | None = None) -> None:
    path.write_bytes(data)
    payloads = [path]
    if companion is not None:
        companion[0].write_bytes(companion[1])
        payloads.append(companion[0])
    manifest = {"schema": ARCHIVE, "payloads": [
        {"file": item.name, "bytes": item.stat().st_size,
         "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in payloads]}
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest))


def ack_rows() -> list[dict]:
    return [
        {"prepared_tick": tick, "prior_completed_tick": tick - 1,
         "prior_completed_controls": tick, "events": list(events),
         "prepared_raw_vx_mps": value, "request_wall_ns": tick * 10,
         "ack_wall_ns": tick * 10 + 1,
         "source": "logical_script_not_hardware_keyboard"}
        for tick, events, value in (
            (175, ("W_PRESS",), 0.6),
            (425, ("W_RELEASE", "S_PRESS"), 0.0),
            (430, ("S_RELEASE",), 0.0))]


def fixture_png() -> tuple[bytes, str]:
    pixels = np.array([[[10, 20, 30], [40, 50, 60]],
                       [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(2))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload)))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return png, hashlib.sha256(pixels[::-1].tobytes()).hexdigest()


class Gui13ReaderPureTests(unittest.TestCase):
    def test_dual_payload_manifest_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            native, arrays = folder / "native.jsonl.gz", folder / "native.npz"
            seal(native, b"gzip bytes", (arrays, b"npz bytes"))
            verified_payload(native, arrays)
            verified_payload(arrays, native)
            arrays.write_bytes(b"changed")
            with self.assertRaises(Invalid):
                verified_payload(native, arrays)

    def test_missing_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "states.npz"
            path.write_bytes(b"unsealed")
            with self.assertRaises(Invalid):
                verified_payload(path)

    def test_logic_ack_order_and_main_request(self) -> None:
        rows = ack_rows()
        check_acks(rows)
        rows[1]["request_wall_ns"] = rows[1]["ack_wall_ns"] + 1
        with self.assertRaises(Invalid):
            check_acks(rows)

    def test_bitwise_numeric_comparison(self) -> None:
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = a.copy()
        self.assertTrue(exact_array(a, b))
        b[0, 1] = np.nextafter(b[0, 1], np.float32(3.0))
        self.assertFalse(exact_array(a, b))
        self.assertFalse(exact_array(a, a.astype(np.float64)))

    def test_gui_performance_pass_and_age_failure(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            worker = {"active_start_monotonic_ns": 100_000_000_000,
                      "active_end_monotonic_ns": 106_000_000_000,
                      "active_rtf": 1.0}
            case = {"logical_script_acks": ack_rows()}
            png_bytes, pixels_sha = fixture_png()
            for threshold_failure, low_fps in ((False, False), (True, False),
                                               (False, True)):
                frames = []
                for index in range(10 if low_fps else 60):
                    frames.append({"status": "rendered", "wall_ns": 100_100_000_000 +
                                   index * 100_000_000,
                                   "snapshot_age_ns": 300_000_000 if threshold_failure
                                   else 10_000_000, "capture": None})
                for filename, control in (("frame_initial.png", 0),
                                          ("frame_drive.png", 250),
                                          ("frame_final.png", 600)):
                    png = folder / filename
                    png.write_bytes(png_bytes)
                    frames.append({"status": "rendered", "wall_ns": 100_000_000_001,
                                   "snapshot_age_ns": 10_000_000,
                                   "control_index": control,
                                   "capture": {"path": str(png), "control_index": control,
                                               "bytes": png.stat().st_size,
                                               "pixels_sha256": pixels_sha,
                                               "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest()}})
                render = {"arm": "gui", "logical_acks": case["logical_script_acks"],
                          "poll_timestamps_ns": [100_000_000_000 + i * 100_000_000
                                                 for i in range(61)], "frames": frames}
                payload = folder / "render_receipt.json"
                if payload.exists():
                    payload.unlink()
                    payload.with_name(payload.name + ".manifest.json").unlink()
                seal(payload, json.dumps(render).encode())
                self.assertIs(check_performance(folder, worker, case)["passed"],
                              not (threshold_failure or low_fps))
            self.assertEqual(percentile95([1, 2, 3, 4, 100]), 100)


if __name__ == "__main__":
    unittest.main()
