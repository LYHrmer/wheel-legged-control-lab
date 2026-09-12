"""GLFW window for D1 driving, with keys owned only by the driving interface.

This viewer receives a separate model/data copy. It never advances physics.
GLFW is imported only when opening a window; the headless runner needs no GUI.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

if __package__:
    from .d1_terrain_display import append_terrain_grid
else:
    from d1_terrain_display import append_terrain_grid


class KeyboardViewer:
    def __init__(self, model, data, commands, *, terrain_label=""):
        import glfw
        import mujoco

        self.glfw, self.mujoco = glfw, mujoco
        self.model, self.data, self.commands = model, data, commands
        self.terrain_label = terrain_label
        self.extra_help = ""
        self._last_render = -math.inf
        self.events = []
        self._clock_start = time.monotonic()
        self._simulation_time = 0.0
        self._focused = True
        self._mouse = None
        self._status = (0.0, 0.0, 0.455, 0.0)
        self.window = self.context = self.scene = None
        if not glfw.init():
            raise RuntimeError("Cannot open the current desktop with GLFW")
        try:
            glfw.window_hint(glfw.SAMPLES, 4)
            self.window = glfw.create_window(1280, 800, "D1 driving | hold WASD", None, None)
            if not self.window:
                raise RuntimeError("GLFW could not create the D1 driving window")
            glfw.make_context_current(self.window)
            # The control loop supplies pacing; vsync must not reduce it to 60 Hz.
            glfw.swap_interval(0)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            self.reset_camera()
            self.opt = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.opt)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._perturb = mujoco.MjvPerturb()
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_window_focus_callback(self.window, self._on_focus)
            glfw.set_cursor_pos_callback(self.window, self._on_mouse)
            glfw.set_scroll_callback(self.window, self._on_scroll)
        except BaseException:
            self.close()
            raise

    def reset_camera(self):
        self.cam.distance = 4.5
        self.cam.azimuth = 135.0
        self.cam.elevation = -28.0

    def _record(self, **event):
        self.events.append(
            {
                "wall_time_s": time.monotonic() - self._clock_start,
                "simulation_time_s": self._simulation_time,
                **event,
            }
        )

    def _on_key(self, _window, key, _scancode, action, _mods):
        self._record(type="key", key=int(key), action=int(action))
        # Driving keys are polled as states. No native Simulate shortcuts run.
        if action == self.glfw.PRESS and key == self.glfw.KEY_C:
            self.reset_camera()
        if action == self.glfw.PRESS and key == self.glfw.KEY_ESCAPE:
            self.commands.update_pressed({self.glfw.KEY_ESCAPE})

    def _on_focus(self, _window, focused):
        self._focused = bool(focused)
        self._record(type="focus", focused=self._focused)
        if not self._focused:
            self.commands.update_pressed(set(), focused=False)

    def _on_mouse(self, window, xpos, ypos):
        previous, self._mouse = self._mouse, (xpos, ypos)
        if (
            previous is None
            or self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT) != self.glfw.PRESS
        ):
            return
        dx, dy = xpos - previous[0], ypos - previous[1]
        self.cam.azimuth -= 0.25 * dx
        self.cam.elevation = min(-8.0, max(-80.0, self.cam.elevation - 0.20 * dy))

    def _on_scroll(self, _window, _xoffset, yoffset):
        self.cam.distance = min(15.0, max(1.5, self.cam.distance * math.exp(-0.10 * yoffset)))

    def poll(self, simulation_time_s):
        self._simulation_time = float(simulation_time_s)
        self.glfw.poll_events()
        focused = bool(self.glfw.get_window_attrib(self.window, self.glfw.FOCUSED))
        self._focused = focused
        keys = {ord(c) for c in "WASDRF"} | {self.glfw.KEY_SPACE, self.glfw.KEY_ESCAPE}
        pressed = {
            key
            for key in keys
            if focused and self.glfw.get_key(self.window, key) == self.glfw.PRESS
        }
        self.commands.update_pressed(pressed, focused=focused)

    def set_status(self, command, simulation_time_s):
        self._status = (
            command.forward_velocity_mps,
            command.yaw_rate_rps,
            command.clearance_m,
            float(simulation_time_s),
        )

    def is_running(self):
        return self.window is not None and not self.glfw.window_should_close(self.window)

    def lock(self):
        return nullcontext()

    def sync(self):
        now = time.monotonic()
        # Read keys and step control at 100 Hz; render fewer frames on this CPU.
        if now - self._last_render < 1.0 / 24.0:
            return
        self._last_render = now
        glfw, mj = self.glfw, self.mujoco
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        mj.mj_forward(self.model, self.data)
        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            self._perturb,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        # No keyboard callback ever changes scene flags or geometry visibility.
        self.scene.flags[mj.mjtRndFlag.mjRND_WIREFRAME] = 0
        append_terrain_grid(self.model, self.scene, self.cam.lookat[:2])
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjr_render(viewport, self.scene, self.context)
        vx, yaw, clearance, seconds = self._status
        left = "Hold W/S\nHold A/D\nHold R/F\nSpace\nMouse / wheel\nC / Esc\nTarget\nTerrain\nTime"
        right = (
            "Forward / reverse (release stops request)\nLeft / right\nRaise / lower body\nStop request\nOrbit / zoom\nReset camera / exit\n"
            f"v {vx:+.2f} m/s | yaw {yaw:+.2f} rad/s | height {clearance:.3f} m\n"
            f"{self.terrain_label}\n{seconds:.1f} s"
        )
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            left,
            right,
            self.context,
        )
        if self.extra_help:
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                self.extra_help,
                "",
                self.context,
            )
        if not self._focused:
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_BIG,
                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                "Click this window to drive",
                "Motion request cleared",
                self.context,
            )
        glfw.swap_buffers(self.window)

    def close(self):
        if self.context is not None:
            self.context.free()
            self.context = None
        if self.scene is not None:
            self.scene = None
        if self.window is not None:
            self.glfw.destroy_window(self.window)
            self.window = None
        self.glfw.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False
