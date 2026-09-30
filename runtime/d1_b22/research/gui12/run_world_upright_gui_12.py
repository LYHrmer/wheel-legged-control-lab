"""One bounded world-upright 99D/16D GUI session.

The stdlib preflight runs before MuJoCo, Torch or the final PPO are imported.
The launcher creates a unique output, reserves every native call, and freezes
all source/evidence inputs. One control thread owns the model, policy, native
guard, reset, and integration; the main thread owns GLFW and a leased display
copy. This source does not authorize running an unreviewed checkpoint.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import run_d1_latest_rl as common

SCHEMA = "d1-world-upright-rl16-async-gui-12-v1"
CAPABILITY_SCHEMA = "d1-world-upright-gui-capability-12-v1"
HORIZON = 1600
SETTLE_TICKS = 175
FRAME_EVERY = 5
INPUT_STALE_NS = 250_000_000
CONTROL_PERIOD_NS = 10_000_000
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
CAP_FIELDS = (
    "max_forward_mps", "max_abs_yaw_rps", "moving_yaw_allowed",
    "max_forward_with_yaw_mps", "max_abs_vx_yaw_m2_s2",
)
HERE = Path(__file__).resolve().parent
W = HERE.parent


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"sha256": common._sha256(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object: " + str(path))
    return value


def _preflight(session: dict[str, Any], session_path: Path) -> tuple[Path, dict[str, Any]]:
    """Reject an unqualified actor or altered source before engine imports."""
    if (session.get("schema") != SCHEMA or session.get("retry_permitted") is not False
            or session.get("argv") != sys.argv
            or session.get("worker_argv", [])[4:] != sys.argv
            or sys.argv[1:] != ["--session", str(session_path)]
            or common._absolute(session, "session_path") != session_path):
        raise RuntimeError("GUI session/argv/retry identity differs")
    output = common._absolute(session, "output_directory")
    if not output.is_dir() or not session.get("run_id"):
        raise RuntimeError("GUI output must be exclusively created by launcher")
    integers = (
        "control_limit", "normal_native_limit", "compiler_native_limit",
        "max_episodes", "episode_horizon", "settle_ticks",
        "frame_publish_every_controls", "input_stale_ns", "control_period_ns",
    )
    if any(type(session.get(name)) is not int for name in integers):
        raise TypeError("GUI physical cadence and limits must be explicit integers")
    limits = [session[name] for name in integers]
    segments = session.get("segment_limits")
    if (not isinstance(segments, list) or len(segments) != limits[3]
            or any(type(value) is not int or not 0 < value <= HORIZON for value in segments)
            or not 1 <= limits[3] <= 2 or sum(segments) != limits[0]
            or limits[1] != 5 * limits[0] or limits[2] != 2
            or limits[4:] != [HORIZON, SETTLE_TICKS, FRAME_EVERY,
                              INPUT_STALE_NS, CONTROL_PERIOD_NS]):
        raise RuntimeError("GUI segment or native reservation exceeds reviewed bounds")
    if session.get("native_record_mode") != "compact":
        raise RuntimeError("GUI requires the reviewed compact native recorder")
    if session.get("actor") not in ("final_policy", "zero"):
        raise RuntimeError("GUI actor must be actual final policy or recorded zero")
    if session.get("validation_profile") not in (None, "straight_1p6", "turn_1p2"):
        raise RuntimeError("unknown reviewed GUI validation profile")
    seeds = session.get("episode_reset_seeds")
    if (not isinstance(seeds, list) or len(seeds) != len(segments)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise RuntimeError("GUI episode reset seeds must be explicit uint32 values")
    sources = session.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("GUI frozen source closure is absent")
    for name, record in sources.items():
        path = Path(name)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or not isinstance(record, dict) or _identity(path) != record):
            raise RuntimeError("GUI frozen source changed: " + str(name))
    required = [Path(__file__).resolve(), W / "upright11/world_upright_course_11.py",
                W / "rl11/rl16_learning_11.py"]
    required.extend(HERE / name for name in (
        "full_drive_controls_12.py", "latest_frame_mailbox_12.py",
        "async_course_renderer_12.py",
    ))
    for key in (
        "contract_path", "go_path", "library", "binding_module",
        "continuation_import_repair", "checkpoint_manifest_path",
        "capability_path", "readback_path", "scoring_path",
    ):
        required.append(common._absolute(session, key))
    if any(str(path) not in sources for path in required):
        raise RuntimeError("GUI critical source/evidence is missing from frozen closure")
    for key, hash_key in (("contract_path", "contract_sha256"), ("go_path", "go_sha256")):
        if _identity(common._absolute(session, key))["sha256"] != session.get(hash_key):
            raise RuntimeError("GUI contract or GO identity differs")
    documents = session.get("contract_documents")
    if (not isinstance(documents, dict) or not documents
            or any(sources.get(name) != identity for name, identity in documents.items())):
        raise RuntimeError("GUI contract documents differ from source closure")
    if common._sha256(common._absolute(session, "library")) != common.DSO_SHA256:
        raise RuntimeError("isolated engine DSO differs from reviewed artifact")
    environment = session.get("runtime_environment")
    if not isinstance(environment, dict) or any(
        os.environ.get(name) != value for name, value in environment.items()
    ):
        raise RuntimeError("GUI isolated runtime environment differs")
    if ("LD_LIBRARY_PATH" in os.environ or os.environ.get("LD_PRELOAD") != session["library"]
            or os.environ.get("LD_BIND_NOW") != "1"):
        raise RuntimeError("GUI isolated engine environment is invalid")
    capability_path = common._absolute(session, "capability_path")
    capability = _load(capability_path)
    if (session.get("capability_sha256") != common._sha256(capability_path)
            or capability.get("schema") != CAPABILITY_SCHEMA
            or capability.get("qualification_passed") is not True
            or capability.get("actor") != session["actor"]
            or capability.get("terrain") != session.get("terrain")
            or capability.get("spawn_position_m") != session.get("spawn_position_m")):
        raise RuntimeError("GUI actor/terrain capability qualification differs")
    caps = capability.get("caps")
    if not isinstance(caps, dict) or any(caps.get(key) != session.get(key) for key in CAP_FIELDS):
        raise RuntimeError("GUI request ceiling differs from reviewed capability")
    for stem in ("readback", "scoring", "checkpoint_manifest"):
        evidence = common._absolute(session, stem + "_path")
        if (capability.get(stem + "_path") != str(evidence)
                or capability.get(stem + "_sha256") != common._sha256(evidence)):
            raise RuntimeError("GUI capability has altered " + stem + " evidence")
    readback = _load(common._absolute(session, "readback_path"))
    if readback.get("schema") != "d1-world-upright-short-rl16-independent-readback-11-v1":
        raise RuntimeError("GUI requires independent 11-S flat readback")
    if (readback.get("execution_complete") is not True
            or readback.get("all_twelve_full_native_controller_contact_records_verified")
            is not True or readback.get("six_initial_state_pairs_bitwise_exact") is not True):
        raise RuntimeError("11-S model/full-native saved-record readback is incomplete")
    # The actual policy can be used even if the separately reported RL
    # contribution comparison is negative. Its requested profile still needs
    # a positive physical task case, bound below to the capability.
    scores = readback.get("numeric_scores", {}).get("cases", [])
    if not isinstance(scores, list):
        raise TypeError("11-S independent numeric cases are absent")
    case_id = capability.get("forward_evidence_case_id")
    selected = [row for row in scores if row.get("case_id") == case_id
                and row.get("actor") == session["actor"]]
    if (len(selected) != 1 or selected[0].get("task_passed") is not True
            or selected[0].get("terrain") != session["terrain"]
            or selected[0].get("target_speed_mps", -1) < session["max_forward_mps"]):
        raise RuntimeError("GUI forward cap is not backed by a qualified actor/case")
    if session["moving_yaw_allowed"]:
        turning = [row for row in scores
                   if row.get("case_id") == capability.get("turn_evidence_case_id")
                   and row.get("actor") == session["actor"]]
        if (len(turning) != 1 or turning[0].get("task_passed") is not True
                or turning[0].get("terrain") != session["terrain"]
                or turning[0].get("target_speed_mps", -1)
                < session["max_forward_with_yaw_mps"]):
            raise RuntimeError("GUI moving-yaw cap lacks an actual qualified case")
    checkpoint_dir = common._absolute(session, "checkpoint_directory")
    trusted = _load(common._absolute(session, "checkpoint_manifest_path"))
    if (not checkpoint_dir.is_dir()
            or readback.get("checkpoint", {}).get("files") != trusted.get("files")
            or readback.get("checkpoint", {}).get("metadata_sha256")
            != trusted.get("metadata_sha256")):
        raise RuntimeError("GUI checkpoint is not the one independently read back")
    for name, record in trusted["files"].items():
        payload = checkpoint_dir / name
        if str(payload) not in sources or _identity(payload) != record:
            raise RuntimeError("GUI checkpoint payload differs: " + name)
    metadata = checkpoint_dir / "final_metadata.json"
    if (str(metadata) not in sources
            or common._sha256(metadata) != trusted["metadata_sha256"]):
        raise RuntimeError("GUI final checkpoint metadata differs")
    return output, capability


class LatestIntentMailbox:
    """Only detached immutable intents cross from GLFW to the control thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._intent: Any = None

    def publish(self, intent: Any) -> None:
        from full_drive_controls_12 import DriveIntent

        if not isinstance(intent, DriveIntent):
            raise TypeError("input mailbox requires an immutable DriveIntent")
        with self._lock:
            if self._intent is not None and intent.seq <= self._intent.seq:
                raise RuntimeError("GUI input sequence did not increase")
            self._intent = intent

    def latest(self) -> Any:
        with self._lock:
            return self._intent


class PreparedInputSource:
    """Consume the latest intent only at the real next-command prepare seam."""

    def __init__(self, mailbox: LatestIntentMailbox) -> None:
        self.mailbox = mailbox
        self._servo_receipt_reader: Any = None
        self._caps: Any = None
        self.episode_id = -1
        self.records: list[dict[str, Any]] = []

    def bind_servo_admission(self, receipt_reader: Any, caps: Any) -> None:
        if (self._servo_receipt_reader is not None or not callable(receipt_reader)
                or caps is None):
            raise RuntimeError("GUI servo admission must bind once")
        self._servo_receipt_reader = receipt_reader
        self._caps = caps

    def begin_episode(self, episode_id: int) -> None:
        if type(episode_id) is not int or episode_id <= self.episode_id:
            raise ValueError("GUI episode ID must increase")
        self.episode_id = episode_id
        self.records = []

    def __call__(self, tick: int, time_s: float) -> Any:
        from full_drive_command_08 import FullDriveCommand

        if (type(tick) is not int or not 0 <= tick < HORIZON
                or not math.isclose(time_s, tick * 0.01, rel_tol=0.0, abs_tol=1e-10)):
            raise RuntimeError("GUI raw callback differs from prepared tick/time")
        intent = self.mailbox.latest()
        now_ns = time.monotonic_ns()
        age_ns = None if intent is None else now_ns - intent.poll_wall_ns
        stale = bool(intent is None or age_ns < 0 or age_ns > INPUT_STALE_NS
                     or not intent.focused or intent.exit_latched or intent.reset_edge)
        settle = tick < SETTLE_TICKS
        requested_vx = 0.0 if stale or settle else intent.forward_velocity_mps
        requested_yaw = 0.0 if stale or settle else intent.yaw_rate_rps
        receipt = None if self._servo_receipt_reader is None else self._servo_receipt_reader()
        old_applied = None if receipt is None else receipt.applied
        old_vx = 0.0 if old_applied is None else old_applied.forward_velocity_mps
        old_yaw = 0.0 if old_applied is None else old_applied.yaw_rate_rps
        admitted_vx, admitted_yaw = requested_vx, requested_yaw
        defer_reason = None
        if self._caps is None:
            raise RuntimeError("GUI servo admission must bind before reset")
        turn_limit = self._caps.max_forward_with_yaw_mps
        if admitted_yaw != 0.0 and old_vx > turn_limit:
            admitted_yaw = 0.0
            defer_reason = "servo_forward_above_qualified_turn_speed"
        if admitted_vx > turn_limit and old_yaw != 0.0:
            admitted_vx = min(admitted_vx, turn_limit)
            admitted_yaw = 0.0
            defer_reason = "servo_yaw_nonzero_defer_higher_forward"
        command = FullDriveCommand(
            admitted_vx, 0.0, admitted_yaw, 0.455, False,
        )
        self.records.append({
            "episode_id": self.episode_id, "prepared_tick": tick,
            "prepared_time_s": time_s, "prepared_wall_ns": now_ns,
            "input_seq": None if intent is None else intent.seq,
            "input_poll_wall_ns": None if intent is None else intent.poll_wall_ns,
            "input_age_ns": age_ns, "input_stale": stale,
            "settle_gate": settle, "raw_command": asdict(command),
            "requested_intent_vx_mps": requested_vx,
            "requested_intent_yaw_rps": requested_yaw,
            "prior_applied_servo_vx_mps": old_vx,
            "prior_applied_servo_yaw_rps": old_yaw,
            "admission_defer_reason": defer_reason,
            "intent": None if intent is None else asdict(intent),
        })
        return command


def _make_thread_owned_runtime(stop_event: threading.Event):
    """Create the C/physics API fence after the stdlib preflight."""
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime

    class ThreadOwnedRuntime(RollingEngineRuntime):
        def __init__(self, library: Path, *, control_limit: int, construction_limit: int):
            # Parent __init__ invokes the virtual allowed setter.
            self.owner_ident = threading.get_ident()
            self.thread_violations: list[dict[str, Any]] = []
            self._violation_lock = threading.Lock()
            self._stop_event = stop_event
            self._constructing = False
            self._actual_plant: Any = None
            super().__init__(library, control_limit=control_limit,
                             construction_limit=construction_limit)

        def _owner(self, entry: str) -> None:
            actual = threading.get_ident()
            if actual == self.owner_ident:
                return
            with self._violation_lock:
                self.thread_violations.append({
                    "entry": entry, "expected": self.owner_ident,
                    "actual": actual, "wall_ns": time.monotonic_ns(),
                })
            self._fatal = True
            self._stop_event.set()
            raise RuntimeError("non-owner engine runtime call: " + entry)

        @property
        def allowed(self):
            self._owner("allowed.get")
            return RollingEngineRuntime.allowed.fget(self)

        @allowed.setter
        def allowed(self, value):
            self._owner("allowed.set")
            return RollingEngineRuntime.allowed.fset(self, value)

        def __enter__(self):
            self._owner("__enter__")
            return super().__enter__()

        def __exit__(self, *args):
            self._owner("__exit__")
            return super().__exit__(*args)

        def construct(self, factory):
            self._owner("construct")
            self._constructing = True
            try:
                env = super().construct(factory)
            finally:
                self._constructing = False
            self._actual_plant = env.unwrapped.plant
            return env

        def bind(self, env):
            self._owner("bind")
            if env.unwrapped.plant is not self._actual_plant:
                raise RuntimeError("GUI attempted to bind another model/data pair")
            return super().bind(env)

        def unbind(self):
            self._owner("unbind")
            return super().unbind()

        def start_segment(self, name, limit):
            self._owner("start_segment")
            return super().start_segment(name, limit)

        def control_step(self, env, action):
            self._owner("control_step")
            return super().control_step(env, action)

        def _step(self, model, data, *args, **kwargs):
            self._owner("_step")
            return super()._step(model, data, *args, **kwargs)

        def _forward(self, *args, **kwargs):
            self._owner("_forward")
            if not self._constructing:
                plant = self._actual_plant
                if (plant is None or len(args) != 2 or kwargs or args[0] is not plant.model
                        or args[1] not in (plant.data, plant.measurement_data)):
                    return self._forbidden()
            return super()._forward(*args, **kwargs)

        def _setconst(self, *args, **kwargs):
            self._owner("_setconst")
            return self._forbidden()

        def _forbidden(self, *args, **kwargs):
            self._owner("_forbidden")
            self._stop_event.set()
            return super()._forbidden(*args, **kwargs)

        def state(self):
            self._owner("state")
            return super().state()

        def ledger_receipt(self):
            self._owner("ledger_receipt")
            return super().ledger_receipt()

    return ThreadOwnedRuntime


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One compressed, exclusive, fsynced block; never overwrite a prefix."""
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw) as packed:
            for row in rows:
                packed.write((json.dumps(common._jsonable(row), sort_keys=True,
                                         allow_nan=False) + "\n").encode())
        raw.flush()
        os.fsync(raw.fileno())
    return {"path": str(path), **_identity(path), "rows": len(rows)}


def _save_arrays(path: Path, rows: dict[str, list[Any]]) -> dict[str, Any]:
    import numpy as np

    with path.open("xb") as stream:
        np.savez(stream, **{key: np.asarray(values) for key, values in rows.items()})
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), **_identity(path),
            "samples": len(rows["time"])}


def _fence_copy_data(mj: Any, *, plant: Any, slots: tuple[Any, ...],
                     display: Any, mailbox: Any, owner_ident: int,
                     main_ident: int, stop_event: threading.Event):
    """Allow only reviewed data-copy edges; restore after both threads stop."""
    original = mj.mj_copyData
    if len({id(item) for item in (*slots, display, plant.data, plant.measurement_data)}) != 6:
        raise RuntimeError("GUI copy-data buffers are not six distinct MjData objects")
    edges = {"live_to_measurement": 0, "measurement_to_writing": 0,
             "reading_to_display": 0, "rejected": 0}

    def checked(destination, model, source):
        ident = threading.get_ident()
        if model is not plant.model:
            edges["rejected"] += 1
            stop_event.set()
            raise RuntimeError("GUI mj_copyData used another model")
        if (ident == owner_ident and destination is plant.measurement_data
                and source is plant.data):
            edge = "live_to_measurement"
        elif (ident == owner_ident and destination in slots
              and source is plant.measurement_data):
            edge = "measurement_to_writing"
        elif (ident == main_ident and destination is display
              and mailbox.is_reading_buffer(source)):
            edge = "reading_to_display"
        else:
            edges["rejected"] += 1
            stop_event.set()
            raise RuntimeError("GUI mj_copyData crossed the owner/snapshot fence")
        edges[edge] += 1
        return original(destination, model, source)

    mj.mj_copyData = checked
    return original, edges
