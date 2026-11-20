"""Two short, on-policy D1 segments for the admitted PPO smoke."""

import math

import numpy as np

from scripts import terrain_baseline86 as baseline
from scripts import terrain_preview85_action as pulse
from scripts import terrain_preview85_sensing as sensing
from scripts.terrain_preview85_common import baseline_command


def advance(ep, request, damping):
    row = pulse._advance(ep, request)
    if damping is not None:
        from scripts.terrain_damping86 import record_applied

        record_applied(ep, row["applied_torque_nm"], row["time_s"])
    return row


def policy_observation(state, command, reading):
    valid = reading["reason"] == "valid" and reading["slope_rad"] is not None
    age = 1.0 if reading["captured_s"] is None else (state.control_time_s - reading["captured_s"]) / 0.25
    vector = np.asarray([
        state.base_rpy[0] / 0.4, state.base_rpy[1] / 0.4, state.base_angular_velocity_body[0] / 5,
        state.base_angular_velocity_body[1] / 5, state.base_linear_velocity_body[0] / 0.35,
        command.forward_velocity_mps / 0.35, reading["slope_rad"] / math.radians(12) if valid else 0.0,
        float(valid), age,
    ])
    if not np.isfinite(vector).all():
        raise ValueError("nonfinite causal PPO observation")
    return np.clip(vector, -5, 5)


def reward(state, command, reading, operating_pitch, applied, limits, fallen):
    valid = reading["reason"] == "valid" and reading["slope_rad"] is not None
    reference = operating_pitch + (-reading["slope_rad"] if valid else command.pitch_rad)
    pitch_cost = ((state.base_rpy[1] - reference) / 0.10) ** 2
    speed_cost = ((state.base_linear_velocity_body[0] - command.forward_velocity_mps) / 0.35) ** 2
    effort = np.mean(np.square(np.asarray(applied) / limits))
    return -float(pitch_cost + 0.25 * speed_cost + 0.01 * effort + 10 * fallen)


class PhysicalSmokeEnv:
    def __init__(self, directory, model, damping):
        self.directory, self.model, self.damping = directory, model, damping
        directory.mkdir()
        self.segment = 0
        self.controls = 0
        self.captures = self.records = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for stream in (self.captures, self.records):
            if stream is not None:
                stream.close()

    def _observe(self):
        self.command = baseline_command(self.ep)
        self.reading = self.clock.observe(self.ep, self.command, self.captures)["82"]
        return policy_observation(self.ep.state, self.command, self.reading)

    def _advance(self, request):
        row = advance(self.ep, request, self.damping)
        self.controls += 1
        if row["torque_bound_violations"]:
            raise ValueError("D1 torque bound violated during PPO smoke")
        if row["termination_reason"]:
            self.ep.state = self.ep.source.read()
        return row

    def reset(self, *, seed):
        if self.segment >= 2:
            raise ValueError("registered physical rollout has only two segments")
        self.close()
        terrain = ("up", "down")[self.segment]
        self.segment += 1
        directory = self.directory / terrain
        directory.mkdir()
        self.captures = (directory / "captures.jsonl").open("x")
        self.records = (directory / "ticks.jsonl").open("x")
        self.ep = baseline.new_episode(terrain, seed, self.model, normalized_input_cost=True,
                                       side_allocator=True, damping_model=self.damping)
        self.clock = baseline.range_clock()
        self.tick = 0
        for _ in range(250):
            self._observe()
            row = self._advance(0.0)
            sensing.write_row(self.records, {**row, "stage": "warmup"})
            if row["termination_reason"]:
                raise ValueError(f"{terrain} warmup ended at {row['time_s']}: {row['termination_reason']}")
        return self._observe()

    def step(self, action):
        if self.tick >= 64 or not math.isfinite(action) or abs(action) > 1:
            raise ValueError("invalid PPO interval or transfer action")
        low = self.ep.controller.low_level
        row = self._advance(action * 0.02 * low.total_mass_kg * low.gravity_mps2)
        self.tick += 1
        value = reward(self.ep.state, self.command, self.reading, self.model.operating_state[1],
                       row["applied_torque_nm"], self.ep.plant.actuator_torque_limit_nm,
                       row["termination_reason"] == "fallen")
        sensing.write_row(self.records, {**row, "stage": "ppo", "reward": value,
                                         "measured_slope_rad": self.reading["slope_rad"]})
        info = {k: row[k] for k in ("requested_transfer_n", "allocated_transfer_n", "applied_torque_nm")}
        return self._observe(), value, row["termination_reason"] is not None, self.tick == 64, info
