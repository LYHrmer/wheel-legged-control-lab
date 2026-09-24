"""Render the already refreshed endpoint of the bounded latest-RL plant.

The frozen KeyboardViewer owns GLFW input and camera behavior. Its historical
``sync`` calls ``mj_forward`` on a display copy; this viewer intentionally does
not. The synchronized plant's measurement data already has current kinematics
after each reset and control interval. Drawing must not add a collision query.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.d1_keyboard_viewer import KeyboardViewer


class LatestRLViewer(KeyboardViewer):
    """Old GLFW input/camera with a read-only, no-forward rendering path."""

    def __init__(self, model: Any, data: Any, commands: Any, **kwargs: Any) -> None:
        super().__init__(model, data, commands, **kwargs)
        self.rendered_frames = 0
        self.skipped_frames = 0
        self.frame_receipts: list[dict[str, Any]] = []
        self.screenshot_receipts: list[dict[str, Any]] = []
        self._capture_requests: list[tuple[str, Path]] = []
        self._source_tick = 0

    def request_capture(self, frame_id: str, path: Path) -> None:
        if not frame_id or not isinstance(frame_id, str):
            raise ValueError("frame id must be a nonempty string")
        target = Path(path)
        if target.exists():
            raise FileExistsError(target)
        self._capture_requests.append((frame_id, target))

    def sync(self) -> bool:
        now = time.monotonic()
        if now - self._last_render < 1.0 / 24.0:
            self.skipped_frames += 1
            return False
        self._last_render = now
        glfw, mj = self.glfw, self.mujoco
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            self.skipped_frames += 1
            return False
        # The caller has copied plant.measurement_data to self.data. In
        # particular, do not call mj_forward, mj_step or collision here.
        mj.mjv_updateScene(
            self.model, self.data, self.opt, self._perturb, self.cam,
            mj.mjtCatBit.mjCAT_ALL, self.scene,
        )
        self.scene.flags[mj.mjtRndFlag.mjRND_WIREFRAME] = 0
        if self.render_quality == "low":
            self.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
            self.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
        from scripts.d1_terrain_display import append_terrain_grid

        append_terrain_grid(self.model, self.scene, self.cam.lookat[:2])
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjr_render(viewport, self.scene, self.context)
        vx, yaw, clearance, seconds = self._status
        controls = self.controls_help or (
            "Hold W\n1 / 2\nSpace / X\nR\nC / Esc",
            "Forward while held\nSelect 0.20 / 0.25 m/s\nStop request\nSimulator reset\nCamera / save and exit",
        )
        left = controls[0] + "\nTarget\nTerrain\nTime"
        right = (
            controls[1] + "\n"
            f"v {vx:+.2f} m/s | yaw {yaw:+.2f} rad/s | height {clearance:.3f} m\n"
            f"{self.terrain_label}\n{seconds:.1f} s"
        )
        mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_TOPLEFT,
                       viewport, left, right, self.context)
        if self.extra_help:
            mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport, self.extra_help, "", self.context)
        if not self._focused:
            mj.mjr_overlay(mj.mjtFont.mjFONT_BIG, mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport, "Click this window to drive", "Motion request cleared",
                           self.context)
        if self._capture_requests:
            from PIL import Image

            rgb = np.empty((height, width, 3), dtype=np.uint8)
            mj.mjr_readPixels(rgb, None, viewport, self.context)
            if not np.any(rgb):
                raise RuntimeError("captured GUI frame is empty")
            pixels_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            frame_id, target = self._capture_requests.pop(0)
            with target.open("xb") as stream:
                Image.fromarray(rgb[::-1]).save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
            self.screenshot_receipts.append({
                "frame_id": frame_id, "source_tick": self._source_tick,
                "source_time_s": float(self.data.time), "path": str(target),
                "pixels_sha256": pixels_sha,
                "png_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bytes": target.stat().st_size,
            })
        glfw.swap_buffers(self.window)
        self.rendered_frames += 1
        return True

    @staticmethod
    def _raw_c_state(runtime: Any) -> tuple[int, ...]:
        """Read the fixed DSO counters without repeated ELF hashing per frame."""
        from engine_binding import _CState

        state = _CState()
        library = runtime.guard._lib
        if library.epa01_get_state(ctypes.byref(state), ctypes.sizeof(state)) != 0:
            raise RuntimeError("cannot read native guard before rendering")
        return tuple(int(getattr(state, name)) for name, _ in _CState._fields_)

    def copy_and_sync(self, plant: Any, runtime: Any, *, source_tick: int) -> bool:
        """Copy a synchronized endpoint and prove rendering changed no dynamics."""
        if self.model is not plant.model or plant.sampling_mode != "synchronized":
            raise RuntimeError("display must share the synchronized compiled model")
        measurement = plant.measurement_data
        live = plant.data
        if (self.data is live or self.data is measurement
                or int(self.data._address) in (int(live._address), int(measurement._address))):
            raise RuntimeError("display data must be an independent allocation")
        for name in ("qpos", "qvel", "ctrl"):
            if not np.array_equal(getattr(measurement, name), getattr(live, name)):
                raise RuntimeError(f"measurement endpoint differs from live {name}")
        if float(measurement.time) != float(live.time):
            raise RuntimeError("measurement endpoint time differs from live time")
        before_arrays = {
            name: getattr(live, name).copy()
            for name in ("qpos", "qvel", "ctrl", "qacc_warmstart")
        }
        before_measurement = {
            name: getattr(measurement, name).copy() for name in ("qpos", "qvel", "ctrl")
        }
        before_time = float(live.time)
        before_measurement_time = float(measurement.time)
        before_c = self._raw_c_state(runtime)
        before_python = (runtime.control_attempted, runtime.control_completed,
                         runtime.attempted, runtime.returned, runtime.failed,
                         runtime.forbidden_calls, runtime.forward_calls,
                         runtime.setconst_calls, runtime.advanced_substeps)
        source_digest = hashlib.sha256()
        for name in ("qpos", "qvel", "ctrl"):
            source_digest.update(before_measurement[name].tobytes())
        source_digest.update(before_measurement_time.hex().encode("ascii"))
        counter_digest = hashlib.sha256(repr((before_c, before_python)).encode("ascii")).hexdigest()
        self.mujoco.mj_copyData(self.data, self.model, measurement)
        self.cam.lookat[:] = measurement.qpos[:3]
        self._source_tick = source_tick
        rendered = self.sync()
        after_python = (runtime.control_attempted, runtime.control_completed,
                        runtime.attempted, runtime.returned, runtime.failed,
                        runtime.forbidden_calls, runtime.forward_calls,
                        runtime.setconst_calls, runtime.advanced_substeps)
        if (self._raw_c_state(runtime) != before_c or after_python != before_python
                or float(live.time) != before_time
                or float(measurement.time) != before_measurement_time
                or any(not np.array_equal(getattr(live, name), value)
                       for name, value in before_arrays.items())
                or any(not np.array_equal(getattr(measurement, name), value)
                       for name, value in before_measurement.items())):
            raise RuntimeError("rendering modified native/Python counters or live integrator")
        if rendered:
            self.frame_receipts.append({
                "frame_index": self.rendered_frames - 1,
                "source_tick": source_tick,
                "source_time_s": before_measurement_time,
                "measurement_source_sha256": source_digest.hexdigest(),
                "counter_snapshot_sha256": counter_digest,
                "C_counters_before_after": list(before_c),
                "python_counters_before_after": list(before_python),
                "native_counter_delta": 0, "python_counter_delta": 0,
                "live_integrator_unchanged": True,
                "measurement_unchanged": True,
                "screenshot_count": len(self.screenshot_receipts),
            })
        return rendered
