"""Root-owned counted access to the separately frozen experimental engine.

The caller supplies the isolated DSO and all limits from its reviewed contract.
Importing this module does not construct a model. Constructing the runtime is
allowed only inside the uniquely reserved root experiment process.
"""
from __future__ import annotations

from scripts.d1_single_step_records import PhysicsCallLedger


class RollingEngineRuntime(PhysicsCallLedger):
    def __init__(self, library, *, control_limit, construction_limit):
        from engine_binding import EngineGuard

        self.guard = EngineGuard(library)
        self.proof = self.guard.binding_proof()
        if not self.proof["passed"]:
            raise RuntimeError("experimental engine binding proof failed")
        initial = self.guard.snapshot()
        keys = ("construction_attempts", "construction_returns", "control_attempts",
                "control_returns", "ccd_attempts", "ccd_returns", "violations")
        if any(initial[key] for key in keys):
            raise RuntimeError("engine counters must start at zero")
        self.guard.arm(construction_limit=construction_limit, control_limit=5 * control_limit)
        self.control_limit = control_limit
        self._allowed = None
        self.constructed = []
        self.segments = []
        self._segment = None
        self._inside_control = False
        self._tick_attempted = self._tick_returned = 0
        self._fatal = False
        self.native_monitor = None
        self.active_plant = None
        super().__init__(native_limit=5 * control_limit)

    @property
    def allowed(self):
        return self._allowed

    @allowed.setter
    def allowed(self, value):
        if value is None:
            self.guard.set_phase(0)
            self.guard.set_target(0, 0)
        else:
            self.guard.set_target(int(value[0]._address), int(value[1]._address))
            self.guard.set_phase(2)
        self._allowed = value

    def construct(self, factory):
        if self._fatal or self._inside_control or self.allowed is not None:
            raise RuntimeError("unbind the previous model before constructing another")
        before = self.guard.snapshot()
        self.guard.set_phase(1)
        try:
            env = factory()
        except BaseException:
            self._fatal = True
            raise
        finally:
            self.guard.set_phase(0)
            after = self.guard.snapshot()
            self.constructed.append({"before": before, "after": after})
        return env

    def bind(self, env):
        if self._fatal or self._inside_control:
            raise RuntimeError("cannot bind after a fatal error or during a control tick")
        plant = env.unwrapped.plant
        self.allowed = (plant.model, plant.data)
        self.active_plant = plant

    def unbind(self):
        self.allowed = None
        self.active_plant = None

    def start_segment(self, name, limit):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("segment limit must be a positive integer")
        if self._fatal or self._inside_control:
            raise RuntimeError("cannot change segment during a tick or after failure")
        if any(row["name"] == name for row in self.segments):
            raise RuntimeError("a segment cannot be restarted")
        if sum(row["limit"] for row in self.segments) + limit > self.control_limit:
            raise RuntimeError("segment reservations exceed total control budget")
        row = {"name": name, "limit": limit, "attempted": 0, "completed": 0,
               "native_attempted": 0, "native_returned": 0}
        self.segments.append(row)
        self._segment = row

    def control_step(self, env, action):
        plant = env.unwrapped.plant
        if (self.allowed is None or plant.model is not self.allowed[0]
                or plant.data is not self.allowed[1]):
            raise RuntimeError("control step target is not the bound plant")
        if (self._fatal or self._inside_control or self._segment is None
                or self.control_attempted >= self.control_limit
                or self._segment["attempted"] >= self._segment["limit"]):
            raise RuntimeError("control segment or total budget exhausted")
        self.control_attempted += 1
        self._segment["attempted"] += 1
        self._inside_control = True
        self._tick_attempted = self._tick_returned = 0
        try:
            result = env.step(action)
            if self._tick_attempted != 5 or self._tick_returned != 5:
                raise RuntimeError("each completed control must execute exactly five native steps")
        except BaseException:
            self._fatal = True
            self.guard.set_phase(0)
            raise
        finally:
            self._inside_control = False
        self.control_completed += 1
        self._segment["completed"] += 1
        return result

    def _step(self, model, data, *args, **kwargs):
        if (self._fatal or not self._inside_control or self._segment is None
                or self._tick_attempted >= 5
                or self._segment["native_attempted"] >= 5 * self._segment["limit"]):
            self._fatal = True
            self.guard.set_phase(0)
            return self._forbidden()
        before = (self.native_monitor.before(model, data)
                  if self.native_monitor is not None else None)
        self._tick_attempted += 1
        self._segment["native_attempted"] += 1
        try:
            result = super()._step(model, data, *args, **kwargs)
            self._tick_returned += 1
            self._segment["native_returned"] += 1
            if self.native_monitor is not None:
                self.native_monitor.after(model, data, before)
            return result
        except BaseException:
            self._fatal = True
            self.guard.set_phase(0)
            raise

    def state(self):
        return self.guard.snapshot()

    def ledger_receipt(self):
        return {**self.receipt(), "control_limit": self.control_limit,
                "segments": self.segments, "construction_records": self.constructed,
                "fatal_latched": self._fatal}

    def __exit__(self, *args):
        try:
            self.unbind()
        finally:
            super().__exit__(*args)
