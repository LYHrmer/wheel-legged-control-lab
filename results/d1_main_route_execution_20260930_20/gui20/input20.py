"""Pure polled keyboard safety seam for the future current-policy GUI.

No GUI, policy, reset or physics is performed here. R requests one simulation
reset from its owner; it is not physical self-righting.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

PROFILES = MappingProxyType({
    'flat_0p6': (.6, 0.), 'flat_1p6': (1.6, 0.),
    'yaw_1p2': (1.2, .3), 'bumps_0p4': (.4, 0.),
    'rough_0p35': (.35, 0.), 'ramp_0p45_complete': (.45, 0.),
})


@dataclass(frozen=True)
class Snapshot20:
    sequence: int
    timestamp: float
    held: frozenset[str] = frozenset()
    focused: bool = True
    closed: bool = False

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('sequence must be nonnegative integer')
        if type(self.held) is not frozenset or not all(type(k) is str for k in self.held):
            raise ValueError('held keys must be an immutable string set')
        if type(self.focused) is not bool or type(self.closed) is not bool:
            raise ValueError('focus/close must be explicit bools')


@dataclass(frozen=True)
class InputResult20:
    forward_mps: float
    yaw_rps: float
    reason: str
    consumed_sequence: int | None
    reset_requested: bool = False
    exit_requested: bool = False
    lateral_mps: float = 0.
    clearance_m: float = .455
    jump_requested: bool = False


class InputController20:
    def __init__(self, profile, *, ttl_s=.25):
        if profile not in PROFILES or not math.isfinite(ttl_s) or ttl_s <= 0.:
            raise ValueError('unknown fixed profile or invalid TTL')
        self._profile, self._ttl = profile, ttl_s
        self._sequence, self._timestamp, self._now = -1, -math.inf, -math.inf
        self._keys, self._needs_release = frozenset(), False
        self._reset_used, self._exit = False, False

    @property
    def profile(self):
        return self._profile

    def update(self, snapshot: Snapshot20, *, now: float) -> InputResult20:
        if not isinstance(snapshot, Snapshot20):
            raise TypeError('an immutable input snapshot is required')
        def stopped(reason, sequence=None, reset=False):
            return InputResult20(0., 0., reason, sequence, reset, self._exit)
        # Exit can only remove authority; it must survive stale or invalid clocks.
        if snapshot.closed or 'escape' in snapshot.held:
            self._exit = True
            return stopped('exit_requested')
        if self._exit:
            return stopped('exit_latched')
        if not (math.isfinite(now) and math.isfinite(snapshot.timestamp)):
            self._needs_release = True
            return stopped('invalid_clock')
        if now < self._now or snapshot.timestamp < self._timestamp:
            self._needs_release = True
            return stopped('backward_clock')
        self._now = now
        if snapshot.timestamp > now:
            self._needs_release = True
            return stopped('future_snapshot')
        if snapshot.sequence <= self._sequence or now-snapshot.timestamp > self._ttl:
            self._needs_release = True
            return stopped('stale_snapshot')
        self._sequence, self._timestamp = snapshot.sequence, snapshot.timestamp
        keys = snapshot.held
        rising_r = 'r' in keys and 'r' not in self._keys
        self._keys = keys
        if not snapshot.focused:
            self._needs_release = True
            return stopped('focus_lost', snapshot.sequence)
        if rising_r:
            self._needs_release = True
            request = not self._reset_used
            self._reset_used = True
            return stopped('simulation_reset_request' if request else 'reset_limit', snapshot.sequence, request)
        if 'space' in keys or 'x' in keys:
            self._needs_release = True
            return stopped('operator_stop', snapshot.sequence)
        if 'w' not in keys:
            self._needs_release = False
            return stopped('released', snapshot.sequence)
        if self._needs_release:
            return stopped('release_w_to_rearm', snapshot.sequence)
        speed, amplitude = PROFILES[self._profile]
        yaw = amplitude * (int('q' in keys) - int('e' in keys))
        return InputResult20(speed, yaw, 'held_command', snapshot.sequence)
