"""Bounded latest-frame mailbox for the 12 world-upright GUI (pure stdlib).

Ownership model
---------------
Three pre-allocated, opaque data slots are supplied by the future runner. The
mailbox itself never calls an engine function: the ``copy_into`` callback is the
owner's ``mj_copyData`` and is invoked only on the slot the producer already
claimed.

* Producer: exactly one thread. Bound lazily on the first ``publish`` call so
  constructing the mailbox on the main thread claims nothing.
* Consumer: exactly one thread. Bound lazily on the first ``acquire_latest``
  call. Only one consumer lease may be active at a time.

Slot states are FREE -> WRITING -> READY -> READING -> FREE. The producer never
selects READING or WRITING, so an active consumer lease or an in-flight copy is
never overwritten. With three slots and a single producer plus a single
consumer there is always a FREE or READY slot during normal operation; the
"dropped" branch is a defensive invariant, not a queue.

Thread identity caveat: ``threading.get_ident()`` values are OS-level and can
theoretically be reused after a thread exits. The mailbox lives for one
session, and its producer/consumer threads outlive it; reuse within that
lifetime is out of scope.

The metadata copy holds only immutable primitives (bool/int/float/str/None), so
no live NumPy view or mutable dict/list alias can leak into a published frame.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

_FREE = "FREE"
_WRITING = "WRITING"
_READY = "READY"
_READING = "READING"

_REQUIRED_METADATA = ("episode_id", "control_index", "sim_time_s", "source_wall_ns")
_MISSING = object()


def _freeze_primitive(name: str, value: object) -> object:
    """Return the value if it is an immutable primitive, else raise."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, str):
        return value
    raise TypeError(
        f"{name} must be an immutable primitive (bool/int/float/str/None), "
        f"got {type(value).__name__}"
    )


def _freeze_metadata(metadata: object) -> dict:
    """Validate and detach frame metadata into a primitive-only dict."""
    if isinstance(metadata, (str, bytes)) or not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping of string keys")
    out: dict = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        out[key] = _freeze_primitive(f"metadata[{key!r}]", value)
    for key in _REQUIRED_METADATA:
        if key not in out:
            raise ValueError(f"metadata is missing required key {key!r}")
    episode_id = out["episode_id"]
    if isinstance(episode_id, bool) or not isinstance(episode_id, int):
        raise TypeError("episode_id must be an int")
    control_index = out["control_index"]
    if (
        isinstance(control_index, bool)
        or not isinstance(control_index, int)
        or control_index < 0
    ):
        raise ValueError("control_index must be a non-negative int")
    sim_time_s = out["sim_time_s"]
    if (
        isinstance(sim_time_s, bool)
        or not isinstance(sim_time_s, (int, float))
        or not math.isfinite(float(sim_time_s))
        or sim_time_s < 0.0
    ):
        raise ValueError("sim_time_s must be a finite non-negative number")
    source_wall_ns = out["source_wall_ns"]
    if (
        isinstance(source_wall_ns, bool)
        or not isinstance(source_wall_ns, int)
        or source_wall_ns < 0
    ):
        raise ValueError("source_wall_ns must be a non-negative int")
    return out


@dataclass
class _FrameSlot:
    index: int
    buffer: object
    state: str = _FREE
    frame_seq: int = -1
    metadata: Mapping = field(default_factory=dict)
    published_wall_ns: int = -1


class FrameLease:
    """One consumer's read-only claim on a published frame.

    ``data`` is the opaque slot buffer; ``metadata`` is an immutable mapping of
    primitives. The consumer must call ``release()`` once, from the same thread
    that acquired the lease, usually in a ``finally`` block.
    """

    __slots__ = (
        "_mailbox",
        "_owner_ident",
        "_released",
        "_slot_index",
        "data",
        "frame_seq",
        "metadata",
        "published_wall_ns",
    )

    def __init__(
        self,
        mailbox: LatestFrameMailbox,
        slot_index: int,
        owner_ident: int,
        data: object,
        metadata: Mapping,
        frame_seq: int,
        published_wall_ns: int,
    ) -> None:
        self._mailbox = mailbox
        self._slot_index = slot_index
        self._owner_ident = owner_ident
        self.data = data
        self.metadata = metadata
        self.frame_seq = frame_seq
        self.published_wall_ns = published_wall_ns
        self._released = False

    def release(self) -> None:
        self._mailbox._release_lease(self)


class LatestFrameMailbox:
    """Fixed three-slot latest-frame mailbox; pure stdlib, no engine imports."""

    def __init__(self, slots: Sequence[object]) -> None:
        if isinstance(slots, (str, bytes)) or not isinstance(slots, Sequence):
            raise TypeError("slots must be a sequence of exactly 3 distinct objects")
        items = list(slots)
        if len(items) != 3:
            raise ValueError("exactly 3 slots are required")
        if len({id(item) for item in items}) != 3:
            raise ValueError("slots must be 3 distinct objects")
        self._slots = tuple(_FrameSlot(index, item) for index, item in enumerate(items))
        self._lock = threading.Lock()
        self._producer_ident: int | None = None
        self._consumer_ident: int | None = None
        self._active_slot: _FrameSlot | None = None
        self._counters = {
            "published": 0,
            "dropped": 0,
            "overwritten": 0,
            "discarded_obsolete": 0,
            "acquired": 0,
            "released": 0,
            "copy_failed": 0,
        }

    # -- introspection ------------------------------------------------------
    def slot_buffers(self) -> tuple[object, ...]:
        """Identity view of the three opaque buffers (no engine involvement)."""
        with self._lock:
            return tuple(slot.buffer for slot in self._slots)

    def is_reading_buffer(self, buffer: object) -> bool:
        """Permit the renderer copy only while it owns this exact slot lease."""
        with self._lock:
            return bool(self._active_slot is not None
                        and self._active_slot.state == _READING
                        and self._active_slot.buffer is buffer
                        and self._consumer_ident == threading.get_ident())

    def _first_free_or_oldest_ready(self) -> _FrameSlot | None:
        """Caller holds the lock. Prefer FREE, else the oldest READY slot."""
        free: _FrameSlot | None = None
        oldest_ready: _FrameSlot | None = None
        for slot in self._slots:
            if slot.state == _FREE and free is None:
                free = slot
            elif (slot.state == _READY
                  and (oldest_ready is None or slot.frame_seq < oldest_ready.frame_seq)):
                oldest_ready = slot
        return free if free is not None else oldest_ready

    # -- producer -----------------------------------------------------------
    def publish(
        self,
        copy_into: Callable[[object], None],
        metadata: Mapping[str, object],
    ) -> dict:
        """Copy one frame into a FREE or oldest-READY slot and publish it.

        ``copy_into`` runs outside the lock and only on the claimed slot. On a
        copy exception the slot returns to FREE, other frames are untouched and
        the exception propagates. Returns a detached status dict; it never waits.
        """
        if not callable(copy_into):
            raise TypeError("copy_into must be callable")
        frozen = _freeze_metadata(metadata)
        ident = threading.get_ident()
        if self._producer_ident is None:
            self._producer_ident = ident
        elif self._producer_ident != ident:
            raise RuntimeError("LatestFrameMailbox is single-producer")

        with self._lock:
            slot = self._first_free_or_oldest_ready()
            if slot is None:
                self._counters["dropped"] += 1
                return {
                    "status": "dropped",
                    "reason": "no_free_or_ready_slot",
                    "frame_seq": None,
                    "slot_index": None,
                    "wall_ns": time.monotonic_ns(),
                }
            if slot.state == _READY:
                self._counters["overwritten"] += 1
            slot.state = _WRITING

        try:
            copy_into(slot.buffer)
        except BaseException:
            with self._lock:
                slot.state = _FREE
                slot.frame_seq = -1
                slot.metadata = {}
                slot.published_wall_ns = -1
                self._counters["copy_failed"] += 1
            raise

        with self._lock:
            slot.metadata = frozen
            self._counters["published"] += 1
            slot.frame_seq = self._counters["published"]
            slot.published_wall_ns = time.monotonic_ns()
            slot.state = _READY
        return {
            "status": "published",
            "reason": None,
            "frame_seq": slot.frame_seq,
            "slot_index": slot.index,
            "wall_ns": slot.published_wall_ns,
        }

    # -- consumer -----------------------------------------------------------
    def acquire_latest(self) -> FrameLease | None:
        """Claim the newest READY frame, discarding obsolete READY frames."""
        ident = threading.get_ident()
        if self._consumer_ident is None:
            self._consumer_ident = ident
        elif self._consumer_ident != ident:
            raise RuntimeError("LatestFrameMailbox is single-consumer")

        with self._lock:
            if self._active_slot is not None:
                raise RuntimeError("only one active consumer lease is allowed")
            newest: _FrameSlot | None = None
            for slot in self._slots:
                if slot.state == _READY and (
                    newest is None or slot.frame_seq > newest.frame_seq
                ):
                    newest = slot
            if newest is None:
                return None
            for slot in self._slots:
                if slot.state == _READY and slot is not newest:
                    slot.state = _FREE
                    slot.frame_seq = -1
                    slot.metadata = {}
                    slot.published_wall_ns = -1
                    self._counters["discarded_obsolete"] += 1
            newest.state = _READING
            self._active_slot = newest
            self._counters["acquired"] += 1
            lease = FrameLease(
                self,
                newest.index,
                ident,
                newest.buffer,
                MappingProxyType(dict(newest.metadata)),
                newest.frame_seq,
                newest.published_wall_ns,
            )
        return lease

    def _release_lease(self, lease: FrameLease) -> None:
        with self._lock:
            if lease._released:
                raise RuntimeError("FrameLease was already released")
            if lease._owner_ident != threading.get_ident():
                raise RuntimeError("FrameLease released by the wrong thread")
            slot = self._slots[lease._slot_index]
            if slot is not self._active_slot or slot.state != _READING:
                raise RuntimeError("FrameLease slot is not the active reading slot")
            lease._released = True
            slot.state = _FREE
            slot.frame_seq = -1
            slot.metadata = {}
            slot.published_wall_ns = -1
            self._active_slot = None
            self._counters["released"] += 1

    # -- statistics ---------------------------------------------------------
    def snapshot_statistics(self) -> dict:
        """Detached, truthful counters and current slot states."""
        with self._lock:
            return {
                "published": self._counters["published"],
                "dropped": self._counters["dropped"],
                "overwritten": self._counters["overwritten"],
                "discarded_obsolete": self._counters["discarded_obsolete"],
                "acquired": self._counters["acquired"],
                "released": self._counters["released"],
                "copy_failed": self._counters["copy_failed"],
                "slot_states": [slot.state for slot in self._slots],
                "slot_frame_seq": [slot.frame_seq for slot in self._slots],
                "active_lease": self._active_slot is not None,
                "producer_thread": self._producer_ident,
                "consumer_thread": self._consumer_ident,
            }


def _intent_field(intent: object, name: str) -> object:
    value = getattr(intent, name, _MISSING)
    if value is _MISSING:
        try:
            value = intent[name]  # type: ignore[index]
        except (KeyError, TypeError):
            pass
    if value is _MISSING:
        raise ValueError(f"intent is missing required field {name!r}")
    return value


def _intent_extras(intent: object, required: set[str]) -> dict:
    if isinstance(intent, Mapping):
        items = list(intent.items())
    elif dataclasses.is_dataclass(intent) and not isinstance(intent, type):
        items = [(f.name, getattr(intent, f.name)) for f in dataclasses.fields(intent)]
    elif hasattr(intent, "_asdict") and callable(intent._asdict):
        items = list(intent._asdict().items())  # type: ignore[union-attr]
    else:
        namespace = getattr(intent, "__dict__", None)
        items = [] if namespace is None else list(namespace.items())
    out: dict = {}
    for key, value in items:
        if key in required:
            continue
        if not isinstance(key, str):
            raise TypeError("intent extra fields must have string names")
        out[key] = _freeze_primitive(f"intent[{key!r}]", value)
    return out


def _require_nonneg_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


@dataclass(frozen=True)
class InputSnapshot:
    """Immutable primitive-only input packet returned by ``latest()``."""

    seq: int
    poll_wall_ns: int
    reset_generation: int
    exit_latched: bool
    payload: tuple = ()


class LatestInputMailbox:
    """Latest-wins input mailbox; no engine, no history queue.

    Accepts an immutable dataclass, a primitive-only mapping, or a plain object
    with the required fields. ``seq`` must strictly increase, ``poll_wall_ns``
    and ``reset_generation`` must not decrease, and once ``exit_latched`` is set
    it is sticky and can never be lost by a later overwrite.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._producer_ident: int | None = None
        self._consumer_ident: int | None = None
        self._latest: InputSnapshot | None = None
        self._last_seq = -1
        self._last_poll_wall_ns = -1
        self._last_reset_generation = 0
        self._exit_latched = False
        self._published = 0
        self._overwritten = 0

    def publish(self, intent: object) -> InputSnapshot:
        ident = threading.get_ident()
        if self._producer_ident is None:
            self._producer_ident = ident
        elif self._producer_ident != ident:
            raise RuntimeError("LatestInputMailbox is single-producer")

        seq = _require_nonneg_int(_intent_field(intent, "seq"), "seq")
        poll_wall_ns = _require_nonneg_int(
            _intent_field(intent, "poll_wall_ns"), "poll_wall_ns"
        )
        reset_generation = _require_nonneg_int(
            _intent_field(intent, "reset_generation"), "reset_generation"
        )
        exit_latched = _intent_field(intent, "exit_latched")
        if not isinstance(exit_latched, bool):
            raise TypeError("exit_latched must be a bool")
        extras = _intent_extras(
            intent, {"seq", "poll_wall_ns", "reset_generation", "exit_latched"}
        )

        with self._lock:
            if seq <= self._last_seq:
                raise ValueError("seq must strictly increase")
            if poll_wall_ns < self._last_poll_wall_ns:
                raise ValueError("poll_wall_ns must not decrease")
            if reset_generation < self._last_reset_generation:
                raise ValueError("reset_generation must not decrease")
            latched = self._exit_latched or exit_latched
            snapshot = InputSnapshot(
                seq=seq,
                poll_wall_ns=poll_wall_ns,
                reset_generation=reset_generation,
                exit_latched=latched,
                payload=tuple(extras.items()),
            )
            if self._latest is not None:
                self._overwritten += 1
            self._published += 1
            self._latest = snapshot
            self._last_seq = seq
            self._last_poll_wall_ns = poll_wall_ns
            self._last_reset_generation = reset_generation
            self._exit_latched = latched
        return snapshot

    def latest(self) -> InputSnapshot | None:
        ident = threading.get_ident()
        if self._consumer_ident is None:
            self._consumer_ident = ident
        elif self._consumer_ident != ident:
            raise RuntimeError("LatestInputMailbox is single-consumer")
        with self._lock:
            return self._latest

    def snapshot_statistics(self) -> dict:
        with self._lock:
            return {
                "published": self._published,
                "overwritten": self._overwritten,
                "latest_seq": self._latest.seq if self._latest is not None else None,
                "reset_generation": self._last_reset_generation,
                "exit_latched": self._exit_latched,
                "producer_thread": self._producer_ident,
                "consumer_thread": self._consumer_ident,
            }
