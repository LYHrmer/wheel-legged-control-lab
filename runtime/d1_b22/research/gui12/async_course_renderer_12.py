"""Standalone main-thread-only GLFW renderer for the 09 async course GUI.

This module has no import-time GLFW/MuJoCo side effects: both are imported
inside ``__init__`` only. The renderer owns GLFW, the GL context, the camera
and the scene on the constructing main thread, and asserts that identity on
every GUI method.

Physics is never advanced here. The only engine data operation is
``mj_copyData`` from a leased snapshot into the renderer's own display buffer;
everything else is visualization (``mjv_updateScene``, ``mjr_render``,
``mjr_overlay`` and, on capture, ``mjr_readPixels``). No ``mj_forward``,
``mj_step``, collision/kinematics, model mutation, data allocation or policy
path exists in this file. There is deliberately no per-frame engine
counter-difference check, because physics advances concurrently with control.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from typing import Any

_MAX_PENDING_EVENTS = 4096


class AsyncCourseRenderer:
    """Render the latest leased snapshot at most ``max_fps`` times per second."""

    def __init__(
        self,
        model: Any,
        display_data: Any,
        frame_mailbox: Any,
        *,
        key_codes: Any,
        width: int = 800,
        height: int = 500,
        max_fps: float = 15.0,
        title: str = "D1 RL course",
    ) -> None:
        self._main_thread = threading.get_ident()
        self._closed = False
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive int")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive int")
        if (
            isinstance(max_fps, bool)
            or not isinstance(max_fps, (int, float))
            or not math.isfinite(float(max_fps))
            or max_fps <= 0.0
        ):
            raise ValueError("max_fps must be a finite positive number")
        if not isinstance(title, str):
            raise TypeError("title must be a str")
        acquire_latest = getattr(frame_mailbox, "acquire_latest", None)
        slot_buffers = getattr(frame_mailbox, "slot_buffers", None)
        if not callable(acquire_latest) or not callable(slot_buffers):
            raise TypeError(
                "frame_mailbox must expose acquire_latest() and slot_buffers()"
            )
        slots = tuple(slot_buffers())
        if any(slot is display_data for slot in slots):
            raise ValueError("display_data must not be one of the mailbox slots")

        self.model = model
        self.display_data = display_data
        self.frame_mailbox = frame_mailbox
        self.key_codes = frozenset(int(key) for key in key_codes)
        self.width = int(width)
        self.height = int(height)
        self.max_fps = float(max_fps)
        self.title = title

        import glfw
        import mujoco

        self.glfw = glfw
        self.mj = mujoco
        if int(getattr(model, "nplugin", 0)) != 0:
            raise ValueError("renderer refuses a model with plugins")

        self.window = None
        self.context = None
        self.scene = None
        self._focused = True
        self._mouse: tuple[float, float] | None = None
        self._pending_events: list[dict] = []
        self._event_overflow = 0
        self._last_event_ns = 0
        self._held_keys: frozenset = frozenset()
        self._start_monotonic = time.monotonic()
        self._last_render_monotonic = float("-inf")
        self._has_snapshot = False
        self._last_source: dict | None = None
        self._rendered = 0
        self._reused = 0
        self._skipped = 0
        self._captures = 0
        self._copies = 0

        try:
            if not glfw.init():
                raise RuntimeError("GLFW initialization failed")
            glfw.window_hint(glfw.SAMPLES, 0)  # MSAA off
            self.window = glfw.create_window(
                self.width, self.height, self.title, None, None
            )
            if not self.window:
                raise RuntimeError("GLFW window creation failed")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            self._reset_camera()
            self.opt = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.opt)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.context = mujoco.MjrContext(
                model, mujoco.mjtFontScale.mjFONTSCALE_150
            )
            self._perturb = mujoco.MjvPerturb()
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_window_focus_callback(self.window, self._on_focus)
            glfw.set_cursor_pos_callback(self.window, self._on_mouse)
            glfw.set_scroll_callback(self.window, self._on_scroll)
        except BaseException:
            self.close()
            raise

    # -- threading and camera ----------------------------------------------
    def _assert_main_thread(self) -> None:
        if threading.get_ident() != self._main_thread:
            raise RuntimeError(
                "AsyncCourseRenderer methods must run on the creating main thread"
            )

    def _reset_camera(self) -> None:
        self.cam.distance = 4.5
        self.cam.azimuth = 135.0
        self.cam.elevation = -28.0

    # -- GLFW callbacks (append events only) --------------------------------
    def _next_event_ns(self) -> int:
        now = time.monotonic_ns()
        if now <= self._last_event_ns:
            now = self._last_event_ns + 1
        self._last_event_ns = now
        return now

    def _append_event(self, event: dict) -> None:
        if len(self._pending_events) >= _MAX_PENDING_EVENTS:
            self._pending_events.pop(0)
            self._event_overflow += 1
        self._pending_events.append(event)

    def _on_key(self, _window, key, _scancode, action, mods) -> None:
        # Preserve short R/Esc press-and-release edges in callback order.
        self._append_event(
            {
                "type": "key",
                "wall_ns": self._next_event_ns(),
                "key": int(key),
                "action": int(action),
                "mods": int(mods),
            }
        )

    def _on_focus(self, _window, focused) -> None:
        self._focused = bool(focused)
        self._append_event(
            {
                "type": "focus",
                "wall_ns": self._next_event_ns(),
                "focused": bool(focused),
            }
        )

    def _on_mouse(self, window, xpos, ypos) -> None:
        previous, self._mouse = self._mouse, (xpos, ypos)
        if (
            previous is None
            or self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT)
            != self.glfw.PRESS
        ):
            return
        dx = xpos - previous[0]
        dy = ypos - previous[1]
        self.cam.azimuth -= 0.25 * dx
        self.cam.elevation = min(-8.0, max(-80.0, self.cam.elevation - 0.20 * dy))

    def _on_scroll(self, _window, _xoffset, yoffset) -> None:
        self.cam.distance = min(
            15.0, max(1.5, self.cam.distance * math.exp(-0.10 * yoffset))
        )

    # -- input poll ---------------------------------------------------------
    def poll(self) -> dict:
        """Drain GLFW events and return an authoritative detached input record."""
        self._assert_main_thread()
        self.glfw.poll_events()
        wall_ns = time.monotonic_ns()
        events = list(self._pending_events)
        self._pending_events.clear()
        overflow = self._event_overflow
        self._event_overflow = 0
        focused = bool(
            self.glfw.get_window_attrib(self.window, self.glfw.FOCUSED)
        )
        self._focused = focused
        held = frozenset(
            key
            for key in self.key_codes
            if focused and self.glfw.get_key(self.window, key) == self.glfw.PRESS
        )
        self._held_keys = held
        window_close = bool(self.glfw.window_should_close(self.window))
        return {
            "wall_ns": wall_ns,
            "wall_s": time.monotonic() - self._start_monotonic,
            "focused": focused,
            "window_close": window_close,
            "held_keys": sorted(int(key) for key in held),
            "events": events,
            "events_truncated": overflow > 0,
        }

    # -- snapshot copy / render --------------------------------------------
    def _copy_latest_lease(self) -> tuple[dict | None, int]:
        """Copy the newest lease into the owned display buffer, release, return.

        The lease is always released in ``finally`` before any scene/render
        work, and no producer buffer is read after release.
        """
        lease = self.frame_mailbox.acquire_latest()
        if lease is None:
            return None, 0
        start = time.perf_counter_ns()
        try:
            self.mj.mj_copyData(self.display_data, self.model, lease.data)
            copy_ns = time.perf_counter_ns() - start
            meta = dict(lease.metadata)
            source = {
                "frame_seq": int(lease.frame_seq),
                "metadata": meta,
                "episode_id": meta.get("episode_id"),
                "control_index": meta.get("control_index"),
                "sim_time_s": meta.get("sim_time_s"),
                "source_wall_ns": meta.get("source_wall_ns"),
                "published_wall_ns": int(lease.published_wall_ns),
            }
            self._copies += 1
            return source, copy_ns
        finally:
            lease.release()

    def _draw_hud(self, meta: dict, viewport) -> None:
        mj = self.mj

        def fmt(value):
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)

        right = (
            f"req {fmt(meta.get('requested_com_speed_mps'))} | "
            f"app {fmt(meta.get('applied_com_speed_mps'))} | "
            f"act {fmt(meta.get('actual_com_speed_mps'))} m/s\n"
            f"policy {fmt(meta.get('policy_mode'))} | "
            f"terrain {fmt(meta.get('terrain_state'))} | "
            f"stop {fmt(meta.get('stop_state'))}"
        )
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            self.title,
            right,
            self.context,
        )

    def _capture_frame(self, viewport, width, height, meta, capture_path):
        from pathlib import Path

        import numpy as np

        target = Path(capture_path)
        if target.exists():
            raise FileExistsError(str(target))
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        self.mj.mjr_readPixels(rgb, None, viewport, self.context)
        if not np.any(rgb):
            raise RuntimeError("captured GUI frame is empty")
        pixels_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        from PIL import Image

        with target.open("xb") as stream:
            Image.fromarray(rgb[::-1]).save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        self._captures += 1
        return {
            "path": str(target),
            "pixels_sha256": pixels_sha,
            "png_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
            "episode_id": meta.get("episode_id"),
            "control_index": meta.get("control_index"),
            "sim_time_s": meta.get("sim_time_s"),
        }

    def _render_owned_scene(
        self, source: dict, capture_path
    ) -> tuple[int, dict | None, int, int]:
        """Draw the owned display buffer; never touch any producer buffer."""
        glfw, mj = self.glfw, self.mj
        start = time.perf_counter_ns()
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            raise RuntimeError("framebuffer has non-positive size")
        # Camera target follows the copied qpos of the owned display buffer.
        self.cam.lookat[:] = self.display_data.qpos[:3]
        mj.mjv_updateScene(
            self.model,
            self.display_data,
            self.opt,
            self._perturb,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        # Wireframe off (keeps terrain visible); shadow/reflection off.
        self.scene.flags[mj.mjtRndFlag.mjRND_WIREFRAME] = 0
        self.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
        self.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjr_render(viewport, self.scene, self.context)
        self._draw_hud(source["metadata"], viewport)
        capture = None
        if capture_path is not None:
            capture = self._capture_frame(
                viewport, width, height, source["metadata"], capture_path
            )
        glfw.swap_buffers(self.window)
        render_ns = time.perf_counter_ns() - start
        return render_ns, capture, int(width), int(height)

    def _skip_record(self, reason: str, wall_ns: int, total_start: int) -> dict:
        return {
            "status": "skipped",
            "reason": reason,
            "wall_ns": wall_ns,
            "wall_s": time.monotonic() - self._start_monotonic,
            "copy_wall_ns": 0,
            "render_wall_ns": 0,
            "total_wall_ns": time.perf_counter_ns() - total_start,
            "frame_seq": None,
            "control_index": None,
            "episode_id": None,
            "sim_time_s": None,
            "source_wall_ns": None,
            "published_wall_ns": None,
            "snapshot_age_ns": None,
            "width": None,
            "height": None,
            "rendered_count": self._rendered,
            "reused_count": self._reused,
            "skipped_count": self._skipped,
            "capture": None,
            "source_metadata": None,
        }

    def render_latest(self, *, force: bool = False, capture_path=None) -> dict:
        """Render at most ``max_fps``; ``force`` bypasses the wall throttle.

        Returns a detached JSON-ready record; the runner owns persistence.
        """
        self._assert_main_thread()
        if capture_path is not None:
            capture_path = os.fspath(capture_path)
        total_start = time.perf_counter_ns()
        wall_ns = time.monotonic_ns()
        now = time.monotonic()
        if not force and (now - self._last_render_monotonic) < (1.0 / self.max_fps):
            self._skipped += 1
            return self._skip_record("max_fps_throttle", wall_ns, total_start)

        source, copy_ns = self._copy_latest_lease()
        reason = "rendered"
        if source is None:
            if not self._has_snapshot:
                self._skipped += 1
                return self._skip_record("no_snapshot_yet", wall_ns, total_start)
            source = self._last_source
            reason = "reused"
        else:
            self._last_source = source
            self._has_snapshot = True

        render_ns, capture, width, height = self._render_owned_scene(
            source, capture_path
        )
        self._last_render_monotonic = now
        if reason == "rendered":
            self._rendered += 1
        else:
            self._reused += 1
        return {
            "status": reason,
            "reason": reason,
            "wall_ns": wall_ns,
            "wall_s": now - self._start_monotonic,
            "copy_wall_ns": copy_ns,
            "render_wall_ns": render_ns,
            "total_wall_ns": time.perf_counter_ns() - total_start,
            "frame_seq": source.get("frame_seq"),
            "control_index": source.get("control_index"),
            "episode_id": source.get("episode_id"),
            "sim_time_s": source.get("sim_time_s"),
            "source_wall_ns": source.get("source_wall_ns"),
            "published_wall_ns": source.get("published_wall_ns"),
            "snapshot_age_ns": time.monotonic_ns() - source["published_wall_ns"],
            "width": width,
            "height": height,
            "rendered_count": self._rendered,
            "reused_count": self._reused,
            "skipped_count": self._skipped,
            "capture": capture,
            "source_metadata": source.get("metadata") if reason == "rendered" else None,
        }

    # -- teardown -----------------------------------------------------------
    def close(self) -> None:
        """Idempotent main-thread teardown of GL/window/context only."""
        self._assert_main_thread()
        if self._closed:
            return
        if self.context is not None:
            self.context.free()
            self.context = None
        self.scene = None
        if self.window is not None:
            self.glfw.destroy_window(self.window)
            self.window = None
        if self.glfw is not None:
            self.glfw.terminate()
        self._closed = True
