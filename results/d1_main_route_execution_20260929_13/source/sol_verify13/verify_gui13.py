"""Independent saved-byte reader for the fixed GUI13 flat-0.6 development pair.

Imports only the standard library and NumPy. It never loads a model, engine,
worker module, or untrusted pickle. A failed field or archive is a failed gate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "d1-world-upright-flat0p6-gui13-v1"
ARCHIVE = "d1-archive-transaction-13-v1"
MODEL_SHA = "6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70"
STATE_KEYS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time")
N, NATIVE, SEED = 600, 3000, 88813
TORQUE_LIMITS = np.array([80.0, 80.0, 80.0, 12.0] * 4)


class Invalid(ValueError):
    """One precise saved-evidence failure."""


def need(test: bool, message: str) -> None:
    if not test:
        raise Invalid(message)


def sha(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def obj(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda value: (_ for _ in ()).throw(Invalid(value)))
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verified_payload(path: Path, companion: Path | None = None) -> None:
    """Verify manifest-last transaction and every payload before content parsing."""
    need(path.is_file() and not path.is_symlink(), f"payload absent/symlink: {path}")
    manifest_path = path.with_name(path.name + ".manifest.json")
    if companion is not None and not manifest_path.exists():
        manifest_path = companion.with_name(companion.name + ".manifest.json")
    need(manifest_path.is_file() and not manifest_path.is_symlink(),
         f"archive manifest absent: {manifest_path}")
    manifest = obj(manifest_path)
    rows = manifest.get("payloads")
    expected = {path.name} if companion is None else {path.name, companion.name}
    need(manifest.get("schema") == ARCHIVE and isinstance(rows, list)
         and len(rows) == len(expected)
         and {row.get("file") for row in rows if isinstance(row, dict)} == expected,
         f"archive transaction members differ: {manifest_path}")
    for row in rows:
        name = row["file"]
        item = manifest_path.parent / name
        need(Path(name).name == name and item.is_file() and not item.is_symlink()
             and sha(item) == {"bytes": row.get("bytes"), "sha256": row.get("sha256")},
             f"archive digest differs: {item}")


def archived_json(path: Path) -> dict:
    verified_payload(path)
    return obj(path)


def archived_rows(path: Path, *, companion: Path | None = None) -> list[dict]:
    verified_payload(path, companion)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            Invalid(value))) for line in stream]
    need(all(isinstance(row, dict) for row in rows), f"non-object row: {path}")
    return rows


def archived_npz(path: Path, *, companion: Path | None = None) -> dict[str, np.ndarray]:
    verified_payload(path, companion)
    with np.load(path, allow_pickle=False) as data:
        need(len(data.files) == len(set(data.files)), f"duplicate npz member: {path}")
        arrays = {key: data[key].copy() for key in data.files}
    need(all(array.dtype.kind in "fiub" and np.isfinite(array).all()
             for array in arrays.values()), f"nonfinite/non-numeric array: {path}")
    return arrays


def exact_array(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def png_rgb(path: Path) -> np.ndarray:
    """Decode the fixed RGB8 Pillow capture without adding an image dependency."""
    data = path.read_bytes()
    need(data.startswith(b"\x89PNG\r\n\x1a\n"), f"PNG signature missing: {path}")
    offset, width, height, payload = 8, None, None, bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        tag = data[offset + 4:offset + 8]
        end = offset + 12 + length
        need(end <= len(data), f"PNG chunk truncated: {path}")
        body = data[offset + 8:end - 4]
        need(zlib.crc32(tag + body) == struct.unpack_from(">I", data, end - 4)[0],
             f"PNG chunk CRC differs: {path}")
        if tag == b"IHDR":
            width, height, bits, colour, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", body)
            need(0 < width <= 4096 and 0 < height <= 4096 and bits == 8
                 and colour == 2 and compression == filtering == interlace == 0,
                 f"PNG RGB8 dimensions/format differ: {path}")
        elif tag == b"IDAT":
            payload.extend(body)
        elif tag == b"IEND":
            need(end == len(data), f"PNG trailing bytes: {path}")
            break
        offset = end
    need(width is not None and height is not None and bool(payload), f"PNG data missing: {path}")
    raw = zlib.decompress(payload)
    stride = width * 3
    need(len(raw) == height * (stride + 1), f"PNG raster length differs: {path}")
    image = np.empty((height, stride), dtype=np.uint8)
    for row in range(height):
        start = row * (stride + 1)
        method = raw[start]
        need(method in (0, 1, 2, 3, 4), f"PNG filter invalid: {path}")
        source = raw[start + 1:start + 1 + stride]
        for col, value in enumerate(source):
            left = int(image[row, col - 3]) if col >= 3 else 0
            above = int(image[row - 1, col]) if row else 0
            upper_left = int(image[row - 1, col - 3]) if row and col >= 3 else 0
            if method == 0:
                predictor = 0
            elif method == 1:
                predictor = left
            elif method == 2:
                predictor = above
            elif method == 3:
                predictor = (left + above) // 2
            else:
                p = left + above - upper_left
                distances = (abs(p - left), abs(p - above), abs(p - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            image[row, col] = (value + predictor) & 255
    return image.reshape(height, width, 3)


def speed(tick: int) -> float:
    return 0.6 if 175 <= tick < 425 else 0.0


def check_acks(acks: Any) -> None:
    need(isinstance(acks, list) and len(acks) == 3, "three logical event acks required")
    for row, tick, events, request in zip(
        acks, (175, 425, 430),
        (("W_PRESS",), ("W_RELEASE", "S_PRESS"), ("S_RELEASE",)),
        (0.6, 0.0, 0.0), strict=True,
    ):
        need(row.get("prepared_tick") == tick and row.get("prior_completed_tick") == tick - 1
             and row.get("prior_completed_controls") == tick
             and tuple(row.get("events", ())) == events
             and row.get("prepared_raw_vx_mps") == request
             and row.get("source") == "logical_script_not_hardware_keyboard"
             and type(row.get("request_wall_ns")) is int
             and type(row.get("ack_wall_ns")) is int
             and row["ack_wall_ns"] >= row["request_wall_ns"],
             f"logical request/ack mismatch at prepared tick {tick}")


def check_execution(output: Path, arm: str) -> tuple[dict, dict, dict, dict]:
    session = obj(output / "session.json")
    host = obj(output / "launcher_receipt.json")
    worker = archived_json(output / "worker_receipt.json")
    case = archived_json(output / "case/case_receipt.json")
    render = archived_json(output / "render_receipt.json")
    need(not list(output.rglob("*.partial")) and not list(output.rglob("*.partial.*")),
         "partial archive file remains")
    need(session.get("schema") == SCHEMA and session.get("arm") == arm
         and session.get("actor") == "final_policy" and session.get("terrain") == "flat"
         and session.get("seed") == SEED and session.get("control_limit") == N
         and session.get("normal_native_limit") == NATIVE
         and session.get("compiler_native_limit") == 2
         and session.get("retry_permitted") is False
         and session.get("evidence", {}).get("checkpoint_model_sha256") == MODEL_SHA
         and session["evidence"].get("development_seed") == SEED
         and session["evidence"].get("source_case_seed") == 88701
         and session["evidence"].get("flat0p6_final_policy_task_passed") is True
         and session["evidence"].get("original_1600_task_replayed") is False,
         "session identity/scope/budget differs")
    source = session.get("source_hashes")
    need(isinstance(source, dict) and len(source) >= 20, "frozen source closure absent")
    for name, expected in source.items():
        item = Path(name)
        need(item.is_absolute() and item.is_file() and not item.is_symlink()
             and sha(item) == expected, f"frozen source changed: {name}")
    models = [identity for name, identity in source.items()
              if Path(name).name == "final_model.zip"]
    need(len(models) == 1 and models[0].get("sha256") == MODEL_SHA,
         "final checkpoint bytes not frozen")
    plan = Path(session["plan_path"])
    need(str(plan) in source and sha(plan)["sha256"] == session.get("plan_sha256")
         and obj(plan).get("decision") == "GO", "GO plan identity differs")
    frozen_plan = obj(plan)
    need(frozen_plan.get("inputs") == {name: identity for name, identity in source.items()
                                        if name != str(plan)}
         and frozen_plan.get("evidence") == session["evidence"]
         and frozen_plan.get("arms", {}).get(arm, {}).get("seed") == SEED,
         "GO plan/session closure differs")
    need(host.get("schema") == SCHEMA and host.get("arm") == arm
         and host.get("failure") is None and host.get("exit_code") == 0
         and host.get("source_hash_mismatches") == []
         and host.get("owned_worker_cleanup", {}).get("no_orphans") is True
         and host.get("fully_reserved_budget_closed") is True
         and host.get("retry_permitted") is False, "host closure failed")
    calls = {"load_attempted": 1, "load_returned": 1,
             "predict_attempted": 601, "predict_returned": 601,
             "learn_attempted": 0, "save_attempted": 0}
    need(worker.get("schema") == SCHEMA and worker.get("arm") == arm
         and worker.get("failure") is None and worker.get("completed_controls") == N
         and worker.get("policy_predictions") == N and worker.get("model_calls") == calls
         and type(worker.get("strict_load_probe_rows")) is int
         and 1 <= worker["strict_load_probe_rows"] <= 32
         and worker.get("warnings") == [] and worker.get("thread_violations") == []
         and worker.get("writer_partial") == []
         and worker.get("qualified_for_default_GUI") is False
         and worker.get("original_1600_task_replayed") is False,
         "worker/model/closure failed")
    pauses = worker.get("pause_render_proofs")
    edges = worker.get("copy_edges")
    need(isinstance(pauses, list) and len(pauses) == 2
         and [row.get("stage") for row in pauses] == ["initial", "final"]
         and all(row.get("unchanged") is True for row in pauses)
         and isinstance(edges, dict) and edges.get("rejected") == 0
         and edges.get("measurement_to_writing", 0) >= 2
         and edges.get("reading_to_display", 0) >= 2,
         "snapshot ownership/render pauses failed")
    c = worker.get("boundary", {}).get("C_state", {})
    py = worker.get("boundary", {}).get("python", {})
    guard = worker.get("guard", {})
    segment = worker.get("native_segment", {})
    need(c.get("construction_attempts") == c.get("construction_returns") == 2
         and c.get("control_attempts") == c.get("control_returns") == NATIVE
         and c.get("violations") == 0
         and c.get("phase") == 0
         and c.get("target_model") == c.get("target_data") == 0
         and c.get("ccd_attempts") == c.get("ccd_returns")
         and py.get("control_attempted") == py.get("control_completed") == N
         and py.get("native_attempted") == py.get("native_returned") == NATIVE
         and py.get("forbidden_entries") == 0 and py.get("fatal_latched") is False,
         "C/Python physical budget differs")
    need(guard.get("mode") == "train" and guard.get("native_attempted") == NATIVE
         and guard.get("native_returned") == guard.get("native_checked") == NATIVE
         and guard.get("nonwheel_ground_contacts") == 0
         and guard.get("max_abs_roll_deg", math.inf) <= 10.0
         and guard.get("max_abs_pitch_deg", math.inf) <= 10.0
         and guard.get("failure") is None
         and segment.get("mode") == "train" and segment.get("record_valid") is True
         and segment.get("native_attempted") == segment.get("native_returned") == NATIVE
         and segment.get("partial_native_interval") is False
         and segment.get("archive_failure") is None
         and segment.get("full_contact_qualification_recorded") is False,
         "compact native guard/budget/safety failed")
    need(case.get("completed_controls") == case.get("policy_predictions") == N
         and case.get("termination") == "time_limit"
         and case.get("compact_native_only") is True
         and case.get("full_contact_force_qualified") is False
         and case.get("native_segment") == segment,
         "case receipt differs")
    check_acks(case.get("logical_script_acks"))
    mailbox = render.get("mailbox", {})
    need(render.get("arm") == arm
         and render.get("logical_acks") == case["logical_script_acks"]
         and mailbox.get("active_lease") is False
         and mailbox.get("published", 0) >= 2
         and mailbox.get("acquired", 0) == mailbox.get("released", -1),
         "render/snapshot close or logical ack differs")
    return session, worker, case, segment


def check_numeric(output: Path, case: dict, segment: dict) -> dict:
    construction = archived_json(output / "construction.json")
    actuator_ids = np.asarray(construction.get("actuator_ids", ()), dtype=np.int64)
    need(actuator_ids.shape == (16,) and set(actuator_ids.tolist()) == set(range(16)),
         "compiled actuator ordering missing")
    states = archived_npz(output / "case/states.npz")
    need(set(states) == set(STATE_KEYS), "states fields differ")
    need(all(states[key].ndim == 2 and states[key].shape[0] == N + 1
             for key in STATE_KEYS[:-1]) and states["time"].shape == (N + 1,)
         and states["observation"].shape == (N + 1, 99)
         and states["observation"].dtype == np.dtype("float32")
         and np.allclose(states["time"], np.arange(N + 1) * 0.01,
                         rtol=0, atol=1e-10), "601 numeric states/time invalid")
    rows = archived_rows(output / "case/controls.jsonl.gz")
    need(len(rows) == N, "control record count differs")
    nonzero_applied = False
    for tick, row in enumerate(rows):
        info = row.get("info", {})
        raw = info.get("raw_operator_command", {})
        used = info.get("consumed_command", {})
        native = info.get("native_interval_summary", {})
        calculation = info.get("controller_record", {}).get("calculation", {})
        obs = np.asarray(row.get("input_observation99", ()), dtype=np.float32)
        action = np.asarray(row.get("policy_input_action", ()), dtype=np.float32)
        torque = np.asarray(calculation.get("safe_torque_nm", ()), dtype=np.float64)
        applied_action = np.asarray(info.get("applied_action", ()), dtype=np.float64)
        calculated_action = np.asarray(calculation.get("applied_action", ()),
                                       dtype=np.float64)
        traces = row.get("native_actuator_traces")
        need(row.get("tick") == tick and row.get("policy_predict_called") is True
             and exact_array(obs, states["observation"][tick])
             and action.shape == (16,) and np.isfinite(action).all()
             and np.max(np.abs(action)) <= 1.0
             and raw.get("forward_velocity_mps") == speed(tick)
             and raw.get("yaw_rate_rps") == 0.0
             and used.get("yaw_rate_rps") == 0.0
             and type(used.get("forward_velocity_mps")) in (int, float)
             and 0.0 <= used["forward_velocity_mps"] <= 0.6
             and native.get("native_returns") == 5
             and native.get("nonwheel_contact_count") == 0
             and info.get("metrics", {}).get("clearance_m", -1) >= 0.28
             and torque.shape == (16,) and np.isfinite(torque).all()
             and np.all(np.abs(torque) <= TORQUE_LIMITS + 1e-9)
             and np.array_equal(action, np.asarray(info.get("policy_input_action", ()),
                                                   dtype=np.float32))
             and applied_action.shape == calculated_action.shape == (16,)
             and np.isfinite(applied_action).all()
             and np.array_equal(applied_action, calculated_action)
             and row.get("terminated") is False
             and row.get("truncated") is (tick == N - 1),
             f"control/servo/torque chain differs at tick {tick}")
        nonzero_applied |= bool(np.any(applied_action != 0))
        need(isinstance(traces, list) and len(traces) == 5,
             f"five native actuator traces missing at tick {tick}")
        for substep, trace in enumerate(traces):
            need(isinstance(trace, dict) and set(trace) == {
                "requested_nm", "limited_nm", "delayed_nm", "applied_nm",
                "joint_velocity_rps"},
                f"actuator trace fields differ at {tick}:{substep}")
            vectors = {name: np.asarray(value, dtype=np.float64)
                       for name, value in trace.items()}
            need(all(value.shape == (16,) and np.isfinite(value).all()
                     for value in vectors.values())
                 and np.array_equal(vectors["requested_nm"], torque)
                 and np.all(np.abs(vectors["limited_nm"]) <= TORQUE_LIMITS + 1e-9)
                 and np.all(np.abs(vectors["applied_nm"]) <= TORQUE_LIMITS + 1e-9),
                 f"requested/safe/applied actuator trace differs at {tick}:{substep}")
    need(nonzero_applied, "all applied actions are zero")
    native_files = segment.get("native_files")
    arrays_files = segment.get("train_array_files")
    manifests = segment.get("archive_block_manifests")
    need(isinstance(native_files, list) and isinstance(arrays_files, list)
         and isinstance(manifests, list) and len(native_files) == len(arrays_files)
         == len(manifests) == 1, "600-control compact native transaction missing")
    folder = output / "case"
    native_path, array_path = folder / native_files[0], folder / arrays_files[0]
    need(manifests[0] == native_path.name + ".manifest.json", "native block manifest differs")
    native_rows = archived_rows(native_path, companion=array_path)
    native = archived_npz(array_path, companion=native_path)
    keys = {"qpos_before", "qvel_before", "ctrl_before", "qacc_warmstart_before",
            "qpos_after", "qvel_after", "ctrl_after", "qacc_warmstart_after",
            "qacc_after", "actuator_force_after", "qfrc_actuator_after",
            "start_time_s", "end_time_s"}
    need(set(native) == keys and len(native_rows) == NATIVE
         and all(values.shape[0] == NATIVE for values in native.values()),
         "3000 native rows/arrays missing")
    need(native["ctrl_before"].shape == (NATIVE, 16)
         and np.array_equal(native["ctrl_before"], native["ctrl_after"])
         and np.all(np.abs(native["ctrl_before"]) <= TORQUE_LIMITS + 1e-9),
         "native applied actuator torque invalid")
    need(np.array_equal(native["qpos_before"][0], states["qpos"][0])
         and np.array_equal(native["qvel_before"][0], states["qvel"][0]),
         "first native state differs from reset")
    for index, row in enumerate(native_rows):
        tick, substep = divmod(index, 5)
        applied = np.asarray(rows[tick]["native_actuator_traces"][substep]["applied_nm"],
                             dtype=np.float64)
        start, end = float(row.get("start_time_s", math.nan)), float(row.get("end_time_s", math.nan))
        need(row.get("native_index") == index
             and math.isclose(start, index * 0.002, rel_tol=0, abs_tol=1e-10)
             and math.isclose(end, (index + 1) * 0.002, rel_tol=0, abs_tol=1e-10)
             and math.isclose(float(native["start_time_s"][index]), start, abs_tol=1e-12)
             and math.isclose(float(native["end_time_s"][index]), end, abs_tol=1e-12)
             and row.get("nonwheel_contact_count") == 0
             and type(row.get("contact_count")) is int and row["contact_count"] >= 0
             and abs(row.get("roll_deg", math.inf)) <= 10
             and abs(row.get("pitch_deg", math.inf)) <= 10,
             f"native integrator/safety differs at {index}")
        need(np.array_equal(native["ctrl_before"][index, actuator_ids], applied)
             and np.array_equal(native["ctrl_after"][index, actuator_ids], applied),
             f"trace-to-native-ctrl differs at {index}")
        if index:
            need(np.array_equal(native["qpos_before"][index], native["qpos_after"][index - 1])
                 and np.array_equal(native["qvel_before"][index], native["qvel_after"][index - 1]),
                 f"native state discontinuity at {index}")
        if index % 5 == 4:
            tick = index // 5
            need(np.array_equal(native["qpos_after"][index], states["qpos"][tick + 1])
                 and np.array_equal(native["qvel_after"][index], states["qvel"][tick + 1]),
                 f"control endpoint differs from native state at {tick}")
    return {"states": states, "rows": rows, "native_rows": NATIVE,
            "numeric_sha256": sha(output / "case/states.npz")["sha256"]}


def percentile95(values: list[int]) -> int:
    need(bool(values), "no samples for p95")
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def check_performance(output: Path, worker: dict, case: dict) -> dict:
    render = archived_json(output / "render_receipt.json")
    need(render.get("arm") == "gui" and render.get("logical_acks") == case["logical_script_acks"],
         "GUI logical acknowledgements differ")
    start, end = worker.get("active_start_monotonic_ns"), worker.get("active_end_monotonic_ns")
    need(type(start) is int and type(end) is int and start < end,
         "GUI active monotonic boundary missing")
    rtf = N * 0.01 / ((end - start) / 1e9)
    need(worker.get("active_rtf") is not None and
         math.isclose(float(worker["active_rtf"]), rtf, rel_tol=0.01),
         "GUI RTF boundary differs")
    polls = render.get("poll_timestamps_ns")
    need(isinstance(polls, list) and all(type(value) is int for value in polls)
         and polls == sorted(polls), "GUI poll timestamps invalid")
    active_polls = [value for value in polls if start <= value <= end]
    intervals = [b - a for a, b in itertools.pairwise(active_polls)]
    frames = render.get("frames")
    need(isinstance(frames, list), "GUI frames missing")
    drawn = [row for row in frames if row.get("status") in ("rendered", "reused")
             and type(row.get("wall_ns")) is int and start <= row["wall_ns"] <= end]
    draw_fps = len(drawn) / ((end - start) / 1e9)
    ages = [row.get("snapshot_age_ns") for row in drawn]
    need(all(type(age) is int and age >= 0 for age in ages), "GUI snapshot ages invalid")
    captures = {Path(row["capture"]["path"]).name: row for row in frames
                if isinstance(row.get("capture"), dict)}
    for name, expected_tick in (("frame_initial.png", 0), ("frame_final.png", N)):
        need(name in captures and captures[name].get("control_index") == expected_tick,
             f"GUI {name} capture/index missing")
    need("frame_drive.png" in captures
         and 175 <= captures["frame_drive.png"].get("control_index", -1) < 425,
         "GUI moving frame capture missing")
    for name, row in captures.items():
        capture = row["capture"]
        path = output / name
        need(name in {"frame_initial.png", "frame_drive.png", "frame_final.png"}
             and capture.get("path") == str(path) and path.is_file() and not path.is_symlink()
             and sha(path) == {"bytes": capture.get("bytes"),
                               "sha256": capture.get("png_sha256")},
             f"GUI capture bytes differ: {name}")
        pixels = png_rgb(path)
        need(np.any(pixels) and int(pixels.max()) > int(pixels.min())
             and hashlib.sha256(pixels[::-1].tobytes()).hexdigest()
             == capture.get("pixels_sha256"),
             f"GUI capture pixel bytes/contrast differ: {name}")
    passed = (rtf >= 0.8 and draw_fps >= 8.0 and len(intervals) >= 8
              and percentile95(intervals) <= 250_000_000
              and percentile95(ages) <= 250_000_000)
    return {"passed": passed, "active_rtf": rtf, "drawn_active_frames": len(drawn),
            "draw_fps": draw_fps,
            "poll_interval_p95_ns": percentile95(intervals),
            "snapshot_age_p95_ns": percentile95(ages),
            "captures_verified": sorted(captures)}


def validate_arm(output: Path) -> dict:
    """Verify one saved arm; return separate execution, numeric, GUI gates."""
    output = Path(output).resolve()
    report: dict[str, Any] = {"output": str(output), "arm": None,
        "execution_integrity": False, "numeric_record_integrity": False,
        "gui_performance": None, "reasons": []}
    try:
        arm = obj(output / "session.json").get("arm")
        need(arm in ("headless", "gui"), "unknown GUI13 arm")
        report["arm"] = arm
        _, worker, case, segment = check_execution(output, arm)
        report["execution_integrity"] = True
        numeric = check_numeric(output, case, segment)
        report["numeric_record_integrity"] = True
        report["numeric_sha256"] = numeric["numeric_sha256"]
        if arm == "gui":
            performance = check_performance(output, worker, case)
            report["gui_performance"] = performance["passed"]
            report["performance"] = performance
            if not performance["passed"]:
                report["reasons"].append("GUI RTF/draw/poll/age threshold failed")
    except (Invalid, OSError, ValueError, KeyError, TypeError, IndexError,
            EOFError, gzip.BadGzipFile, zipfile.BadZipFile) as error:
        report["reasons"].append(f"{type(error).__name__}: {error}")
        if report["arm"] == "gui" and report["gui_performance"] is None:
            report["gui_performance"] = False
    return report


def verify_pair(headless: Path, gui: Path) -> dict:
    """Compare two separately validated native trajectories byte for byte."""
    h, g = validate_arm(headless), validate_arm(gui)
    report = {"headless": h, "gui": g, "numeric_equivalence": False,
              "gui_performance": g["gui_performance"],
              "qualified_for_default_GUI": False, "reasons": []}
    if (h["arm"] != "headless" or g["arm"] != "gui"
            or not h["execution_integrity"] or not g["execution_integrity"]
            or not h["numeric_record_integrity"] or not g["numeric_record_integrity"]):
        report["reasons"].append("one arm has invalid execution/numeric evidence")
        return report
    try:
        h_session = obj(Path(headless) / "session.json")
        g_session = obj(Path(gui) / "session.json")
        need(h_session["source_hashes"] == g_session["source_hashes"]
             and h_session["plan_sha256"] == g_session["plan_sha256"]
             and h_session["evidence"] == g_session["evidence"],
             "paired source/plan/evidence closure differs")
        a = archived_npz(Path(headless) / "case/states.npz")
        b = archived_npz(Path(gui) / "case/states.npz")
        need(all(exact_array(a[key], b[key]) for key in STATE_KEYS),
             "601 states differ bitwise")
        x = archived_rows(Path(headless) / "case/controls.jsonl.gz")
        y = archived_rows(Path(gui) / "case/controls.jsonl.gz")
        for tick, (left, right) in enumerate(zip(x, y, strict=True)):
            for field in ("input_observation99", "policy_input_action"):
                need(exact_array(np.asarray(left[field], dtype=np.float32),
                                 np.asarray(right[field], dtype=np.float32)),
                     f"paired {field} differs at {tick}")
            for field in ("raw_operator_command", "consumed_command"):
                need(left["info"][field] == right["info"][field],
                     f"paired {field} differs at {tick}")
            for field in ("safe_torque_nm", "applied_action"):
                lcalc = left["info"]["controller_record"]["calculation"]
                rcalc = right["info"]["controller_record"]["calculation"]
                need(exact_array(np.asarray(lcalc[field], dtype=np.float64),
                                 np.asarray(rcalc[field], dtype=np.float64)),
                     f"paired controller {field} differs at {tick}")
            need(left["native_actuator_traces"] == right["native_actuator_traces"],
                 f"paired native actuator traces differ at {tick}")
        ha = archived_json(Path(headless) / "case/case_receipt.json")["native_segment"]
        ga = archived_json(Path(gui) / "case/case_receipt.json")["native_segment"]
        hn = Path(headless) / "case" / ha["native_files"][0]
        gn = Path(gui) / "case" / ga["native_files"][0]
        harray = archived_npz(Path(headless) / "case" / ha["train_array_files"][0],
                              companion=hn)
        garray = archived_npz(Path(gui) / "case" / ga["train_array_files"][0],
                              companion=gn)
        need(all(exact_array(harray[key], garray[key]) for key in harray),
             "3000 native numeric arrays differ bitwise")
        need(archived_rows(hn, companion=Path(headless) / "case" /
                           ha["train_array_files"][0])
             == archived_rows(gn, companion=Path(gui) / "case" /
                              ga["train_array_files"][0]),
             "3000 native contact/cache rows differ")
        report["numeric_equivalence"] = True
    except (Invalid, OSError, ValueError, KeyError, TypeError) as error:
        report["reasons"].append(f"{type(error).__name__}: {error}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", type=Path, required=True)
    parser.add_argument("--gui", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_pair(args.headless, args.gui)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0 if (report["numeric_equivalence"] is True
                 and report["gui_performance"] is True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
