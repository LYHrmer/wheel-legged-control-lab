"""Read-only 10-simulation-FPS viewer for the bounded 08-Q compact probe.

Every control may poll GLFW, but only scheduled ticks or pending captures copy
the synchronized measurement state and draw. The parent LatestRLViewer keeps
its exact before/after integrator and C/Python counter proof on every draw.
"""

from __future__ import annotations

import math
import time
from typing import Any

from scripts.d1_latest_rl_viewer import LatestRLViewer


class CompactViewer(LatestRLViewer):
    DRAW_STRIDE = 10
    WINDOW_SIZE = (800, 500)

    def __init__(self, model: Any, data: Any, commands: Any, **kwargs: Any) -> None:
        if "render_quality" in kwargs and kwargs["render_quality"] != "low":
            raise ValueError("08-Q viewer requires low render quality")
        kwargs["render_quality"] = "low"
        super().__init__(model, data, commands, **kwargs)
        self.glfw.set_window_size(self.window, *self.WINDOW_SIZE)
        self.draw_calls: list[dict[str, Any]] = []
        self.source_ticks_skipped: list[int] = []
        self.requested_window_size = self.WINDOW_SIZE

    def copy_and_sync(self, plant: Any, runtime: Any, *, source_tick: int) -> bool:
        if type(source_tick) is not int or source_tick < 0:
            raise ValueError("08-Q source tick must be a nonnegative integer")
        scheduled = source_tick % self.DRAW_STRIDE == 0
        capture_pending = bool(self._capture_requests)
        if not scheduled and not capture_pending:
            # No display-data copy, C read, digest, scene update, or render.
            self.source_ticks_skipped.append(source_tick)
            self.skipped_frames += 1
            return False
        # The 24-Hz wall throttle in the old viewer is irrelevant to a fixed
        # simulation-tick schedule. Force each planned draw to be attempted.
        self._last_render = -math.inf
        start = time.perf_counter_ns()
        returned = False
        rendered = False
        try:
            rendered = super().copy_and_sync(plant, runtime, source_tick=source_tick)
            if not rendered:
                raise RuntimeError("08-Q scheduled or capture frame was not rendered")
            returned = True
            return True
        finally:
            self.draw_calls.append({
                "source_tick": source_tick,
                "scheduled_stride": scheduled,
                "forced_capture": capture_pending,
                "draw_wall_ns": time.perf_counter_ns() - start,
                "returned": returned,
                "rendered": rendered,
                "measurement_copy_and_counters_checked_by_parent": returned,
            })
