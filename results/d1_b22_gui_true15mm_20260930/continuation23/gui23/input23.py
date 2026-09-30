"""Pure C23 keyboard authority state; owner may run faster than GUI polling.

Every GLFW poll is one immutable snapshot. Repeating an unchanged, recent
snapshot at 100 Hz retains a held command without repeating an R/stop edge.
An expired, reordered, unfocused or stopped input disarms W until a *new*
snapshot observes its release. No model, viewer or physical API is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading

PROFILES = {
    'flat_0p6':(.6,0.), 'flat_1p6':(1.6,0.),
    'yaw_1p2':(1.2,.3), 'bumps_0p4':(.4,0.),
    'rough_0p35':(.35,0.), 'ramp_0p45_complete':(.45,0.),
}
SUPPORTED_KEYS = frozenset({'w','q','e','r','space','x','escape'})


@dataclass(frozen=True,slots=True)
class Event23:
    kind: str
    key: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ('press','release','focus_lost','focus_gained','close'):
            raise ValueError('unknown keyboard/window event kind')
        if self.kind in ('press','release') and (type(self.key) is not str or not self.key):
            raise ValueError('key event needs a nonempty string key')
        if self.kind not in ('press','release') and self.key is not None:
            raise ValueError('non-key event cannot carry a key')


@dataclass(frozen=True,slots=True)
class Snapshot23:
    sequence: int
    timestamp_ns: int
    held: frozenset[str]
    focused: bool
    closed: bool
    events: tuple[Event23,...] = ()
    events_truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence<0:
            raise ValueError('snapshot sequence must be nonnegative integer')
        if type(self.timestamp_ns) is not int or self.timestamp_ns<0:
            raise ValueError('snapshot timestamp must be monotonic nanoseconds')
        if (type(self.held) is not frozenset
                or any(type(key) is not str for key in self.held)):
            raise ValueError('held keys must be an immutable string set')
        if type(self.focused) is not bool or type(self.closed) is not bool:
            raise ValueError('focus/closed must be bool')
        if type(self.events) is not tuple or any(type(x) is not Event23 for x in self.events):
            raise ValueError('events must be an immutable Event23 tuple')
        if type(self.events_truncated) is not bool:
            raise ValueError('event overflow flag must be bool')


@dataclass(frozen=True,slots=True)
class Decision23:
    forward_mps: float
    yaw_rps: float
    reason: str
    consumed_sequence: int | None
    reset_requested: bool = False
    exit_requested: bool = False
    reused_snapshot: bool = False
    clearance_m: float = .455
    lateral_mps: float = 0.
    jump_requested: bool = False


class InputController23:
    def __init__(self, profile: str, *, ttl_s: float = .25) -> None:
        if profile not in PROFILES or not math.isfinite(ttl_s) or ttl_s<=0.:
            raise ValueError('unknown fixed profile or invalid TTL')
        self.profile = profile
        self.ttl_ns = int(ttl_s*1e9)
        if self.ttl_ns<=0:
            raise ValueError('TTL is below one nanosecond')
        self._last: Snapshot23|None = None
        self._last_now_ns = -1
        self._needs_release = False
        self._reset_used = False
        self._exit = False

    @property
    def reset_used(self) -> bool:
        return self._reset_used

    def _zero(self, reason: str, sequence: int|None=None, *,
              reset: bool=False, reused: bool=False) -> Decision23:
        return Decision23(0.,0.,reason,sequence,reset,self._exit,reused)

    def update(self, snapshot: Snapshot23, *, now_ns: int) -> Decision23:
        if not isinstance(snapshot,Snapshot23) or type(now_ns) is not int:
            raise TypeError('owner needs one immutable snapshot and integer monotonic time')
        # Closing and Escape can only remove authority, even for a stale clock.
        if snapshot.closed or 'escape' in snapshot.held or any(
            e.kind=='close' or e.kind=='press' and e.key=='escape' for e in snapshot.events
        ):
            self._exit = True
            self._needs_release = True
            return self._zero('exit_requested')
        if self._exit:
            return self._zero('exit_latched')
        if now_ns<0 or now_ns<self._last_now_ns:
            self._needs_release = True
            return self._zero('backward_owner_clock')
        self._last_now_ns = now_ns
        if snapshot.timestamp_ns>now_ns:
            self._needs_release = True
            return self._zero('future_snapshot')
        if now_ns-snapshot.timestamp_ns>self.ttl_ns:
            self._needs_release = True
            return self._zero('stale_snapshot')
        prior = self._last
        if prior is not None and snapshot.sequence<prior.sequence:
            self._needs_release = True
            return self._zero('reordered_snapshot')
        repeated = prior is not None and snapshot.sequence==prior.sequence
        if repeated and snapshot!=prior:
            self._needs_release = True
            return self._zero('same_sequence_changed')
        if not repeated:
            if prior is not None and snapshot.timestamp_ns<prior.timestamp_ns:
                self._needs_release = True
                return self._zero('backward_snapshot_clock')
            self._last = snapshot
        keys = snapshot.held
        if snapshot.events_truncated or not snapshot.focused or any(
            e.kind=='focus_lost' for e in snapshot.events if not repeated
        ):
            self._needs_release = True
            return self._zero('event_overflow' if snapshot.events_truncated else 'focus_lost',
                              snapshot.sequence,reused=repeated)
        if (any(key not in SUPPORTED_KEYS for key in keys)
                or not repeated and any(e.key not in SUPPORTED_KEYS
                    for e in snapshot.events if e.kind in ('press','release'))):
            self._needs_release = True
            return self._zero('unsupported_key',snapshot.sequence,reused=repeated)
        if not repeated:
            stop_press = any(e.kind=='press' and e.key in ('space','x')
                             for e in snapshot.events)
            if stop_press:
                self._needs_release = True
                return self._zero('operator_stop',snapshot.sequence)
        if 'space' in keys or 'x' in keys:
            self._needs_release = True
            return self._zero('operator_stop',snapshot.sequence,reused=repeated)
        if not repeated:
            r_press = any(e.kind=='press' and e.key=='r' for e in snapshot.events)
            r_rising = 'r' in keys and (prior is None or 'r' not in prior.held)
            if r_press or r_rising:
                self._needs_release = True
                request = not self._reset_used
                self._reset_used = True
                return self._zero('simulation_reset_request' if request else 'reset_limit',
                                  snapshot.sequence,reset=request)
        if 'w' not in keys:
            if not repeated:
                self._needs_release = False
            return self._zero('released',snapshot.sequence,reused=repeated)
        if self._needs_release:
            return self._zero('release_w_to_rearm',snapshot.sequence,reused=repeated)
        forward,amplitude = PROFILES[self.profile]
        yaw = amplitude*(int('q' in keys)-int('e' in keys))
        return Decision23(forward,yaw,'held_command',snapshot.sequence,
                          reused_snapshot=repeated)


class LatestInput23:
    """Latest held state plus all unconsumed short callback edges.

    Several polls can precede one owner read. Their events accumulate in
    callback order and are delivered once with the newest held/focus state.
    A repeated read returns the identical delivered snapshot, so the owner
    cannot retrigger a one-shot edge.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._publisher: int|None = None
        self._consumer: int|None = None
        self._snapshot: Snapshot23|None = None
        self._delivered: Snapshot23|None = None
        self._pending_events: list[Event23] = []
        self._pending_overflow = False
        self.published = 0
        self.reads = 0

    def publish(self, snapshot: Snapshot23) -> None:
        if not isinstance(snapshot,Snapshot23):
            raise TypeError('input mailbox accepts only immutable snapshots')
        ident = threading.get_ident()
        with self._lock:
            if self._publisher is None:
                self._publisher = ident
            elif ident!=self._publisher:
                raise RuntimeError('input publisher changed thread')
            old = self._snapshot
            if old is not None and (snapshot.sequence<=old.sequence
                                    or snapshot.timestamp_ns<old.timestamp_ns):
                raise ValueError('input mailbox snapshot order differs')
            self._snapshot = snapshot
            if len(self._pending_events)+len(snapshot.events)>4096:
                self._pending_events.clear()
                self._pending_overflow = True
            elif not self._pending_overflow:
                self._pending_events.extend(snapshot.events)
            self._pending_overflow |= snapshot.events_truncated
            self._delivered = None
            self.published += 1

    def latest(self) -> Snapshot23|None:
        ident = threading.get_ident()
        with self._lock:
            if self._consumer is None:
                self._consumer = ident
            elif ident!=self._consumer:
                raise RuntimeError('input consumer changed thread')
            self.reads += 1
            if self._snapshot is None:
                return None
            if self._delivered is None:
                sample = self._snapshot
                self._delivered = Snapshot23(
                    sample.sequence,sample.timestamp_ns,sample.held,
                    sample.focused,sample.closed,tuple(self._pending_events),
                    self._pending_overflow)
                self._pending_events.clear()
                self._pending_overflow = False
            return self._delivered
