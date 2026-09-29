"""Two bounded pure cases for the real 11-S schedule and wrapper interfaces.

Root owns execution. No engine, plant construction, PPO or physics is imported
or run by these tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from budget_spec_11 import BudgetSpec
from full_drive_command_08 import QualifiedCommandCaps
from full_drive_schedule_08 import COURSE_SPAWNS_M
from full_drive_servo_08 import FullDriveCommandServo
from short_corridor_11 import assert_nominal_corridor_clearance
from short_episode_plan_11 import FrozenShortPlan
from short_heldout_recipe_11 import ORDER, TASKS, heldout_schedule
from short_plan_recipe_11 import PLAN_SCHEMA, build_plan, select_episode

W = Path(__file__).resolve().parent.parent


def test_real_selector_all_six_heldouts_and_short_flat_footprint_shape():
    plan = build_plan(source_sha256="1" * 64)
    assert plan.schema == PLAN_SCHEMA
    assert [(plan.episode(i).raw_commands[175].forward_velocity_mps,
             plan.episode(i).raw_commands[175].yaw_rate_rps)
            for i in range(8)] == [
                (0.4, 0.0), (0.6, 0.0), (0.8, 0.0), (1.0, 0.0),
                (1.2, 0.0), (1.6, 0.0), (0.6, 0.0), (0.9, 0.0),
            ]
    turn = select_episode(10)  # Slot 2 after eight warmups.
    assert turn.terrain == "flat"
    assert turn.raw_commands[450].yaw_rate_rps != 0.0
    assert turn.raw_commands[550].yaw_rate_rps == -turn.raw_commands[450].yaw_rate_rps
    assert turn.raw_commands[750].yaw_rate_rps == turn.raw_commands[450].yaw_rate_rps
    assert turn.raw_commands[850].yaw_rate_rps == 0.0
    assert select_episode(10).command_sha256 == turn.command_sha256

    assert len(ORDER) == len(TASKS) == 6
    for index, name in enumerate(ORDER):
        schedule = heldout_schedule(name)
        task = TASKS[index]
        assert schedule.case_id == name and schedule.control_cap == task.horizon
        assert len(schedule.raw_commands) == (1800 if index == 5 else 1600)
        assert all(row.forward_velocity_mps == 0.0 for row in schedule.raw_commands[:175])
        assert all(row.forward_velocity_mps == task.speed_mps
                   for row in schedule.raw_commands[175:task.release_tick])
        assert all(row.forward_velocity_mps == 0.0
                   for row in schedule.raw_commands[task.release_tick:])
        assert schedule.command_source(task.horizon - 1,
                                       (task.horizon - 1) * 0.01) == schedule.raw_commands[-1]
        assert task.final_start == task.horizon - 100
    ramp = heldout_schedule(ORDER[-1])
    assert ramp.raw_commands[1354].forward_velocity_mps == 0.45
    assert ramp.raw_commands[1355].forward_velocity_mps == 0.0
    with pytest.raises(ValueError):
        ramp.command_source(1800, 18.0)

    # Reuse the closed real 92-row compiled world map, with a deliberately
    # synthetic flat-spawn robot envelope. This tests the new 1001-path pure
    # geometry implementation, not a claim about a new physical reset.
    geometry = json.loads((W / "upright11/reference_run_01/ramp_zero_reference/"
                           "geometry_manifest.json").read_text())
    rows = geometry["world_collision_geoms"]
    spawn = COURSE_SPAWNS_M["flat"]
    robot_id = max(row["geom_id"] for row in rows) + 1
    manifest = {
        "model_identity": {"source": "actual_compiled_course_reset_geometry_v1",
                           "model_ngeom": robot_id + 1, "synthetic_test_envelope": True},
        "base_world_position_m": spawn,
        "robot_collision_geoms": [{"geom_id": robot_id, "body_id": 1,
                                   "world_center_m": spawn,
                                   "rbound_m": 0.05, "collision": True}],
        "terrain_world_geoms": rows,
    }
    schedule = select_episode(0)
    servo = FullDriveCommandServo(QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False))
    x, y = spawn[:2]
    path = [(x, y)]
    for tick, raw in enumerate(schedule.raw_commands):
        applied = servo.advance(tick, raw).applied
        x += applied.forward_velocity_mps * 0.01
        path.append((x, y))
    path = np.asarray(path)
    assert path.shape == (1001, 2)
    assert assert_nominal_corridor_clearance(manifest, path)["checked_centerline_segments"] == 1000
    with pytest.raises(TypeError):
        assert_nominal_corridor_clearance(
            manifest, np.concatenate((path, np.repeat(path[-1:], 600, axis=0))),
        )


def test_actual_short_wrapper_truncates_at_1000_and_zero_control_reset_keeps_budget(
    monkeypatch, tmp_path,
):
    # Install only the legacy Gym base and the world Env identity before
    # loading the actual new wrapper. No engine/model import can be reached.
    gym = types.ModuleType("gymnasium")

    class FakeGymEnv:
        @property
        def unwrapped(self):
            return self

    class FakeGymWrapper:
        def __init__(self, env):
            self.env = env

    gym.Env = FakeGymEnv
    gym.Wrapper = FakeGymWrapper
    monkeypatch.setitem(sys.modules, "gymnasium", gym)

    world = types.ModuleType("world_upright_course_11")
    world.WORLD_UPRIGHT_REFERENCE_SCHEMA = "pure-world-reference"
    world.WORLD_UPRIGHT_REWARD_SCHEMA = "pure-world-reward"
    world.WORLD_UPRIGHT_TASK_SCHEMA = "pure-world-task"

    caps = QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False)
    base = select_episode(11)  # bumps: no flat footprint is claimed by a fake plant.

    def select(index):
        choice = json.loads(base.choice_json)
        choice["episode_index"] = index
        return replace(
            base, episode_index=index, case_id=f"pure_bumps_{index}",
            choice_json=json.dumps(choice, sort_keys=True, separators=(",", ":")),
        )

    class FakeWorldUprightEnv(FakeGymEnv):
        def __init__(self):
            super().__init__()
            self.mode, self.max_steps, self.pure_test_components = "train", 1000, False
            self.caps = caps
            self.plant = SimpleNamespace(last_control_interval_actuator_traces=[])
            self.reset_calls = 0
            self.tick = 0
            self.command_source = None

        def reset(self, *, seed=None, options=None):
            self.reset_calls += 1
            self.tick = 0
            self.command_source = options["command_source"]
            self.last_seed = seed
            return np.zeros(99, dtype=np.float32), {
                "episode_metadata": {"reference_schema": subject.WORLD_UPRIGHT_REFERENCE_SCHEMA,
                                     "measurement_seed": 17},
            }

        def step(self, action):
            tick = self.tick
            self.tick += 1
            raw = self.command_source(tick, tick * 0.01)
            zero16 = np.zeros(16)
            calc = {name: np.zeros(shape) for name, shape in STAGE_ARRAYS.items()}
            calc["applied_action"] = zero16
            self.plant.last_control_interval_actuator_traces = [
                SimpleNamespace(requested_nm=zero16, delayed_nm=zero16, applied_nm=zero16)
                for _ in range(5)
            ]
            truncated = self.tick == 1000
            info = {
                "task_schema": subject.WORLD_UPRIGHT_TASK_SCHEMA,
                "reward_schema": subject.WORLD_UPRIGHT_REWARD_SCHEMA,
                "reference_schema": subject.WORLD_UPRIGHT_REFERENCE_SCHEMA,
                "completed_control_intervals": self.tick,
                "raw_operator_command": asdict(raw),
                "consumed_command": asdict(raw),
                "native_interval_summary": {"native_returns": 5,
                                            "nonwheel_contact_count": 0},
                "controller_record": {"calculation": calc},
                "reward_terms": {name: 0.0 for name in REWARD_TERMS},
                "metrics": {"body_com_vx_mps": 0.0, "body_com_vy_mps": 0.0,
                            "body_yaw_rate_rps": 0.0, "x_m": 0.0, "y_m": 0.0,
                            "clearance_m": 0.455, "relative_roll_rad": 0.0,
                            "relative_pitch_rad": 0.0, "task_roll_error_rad": 0.0,
                            "task_pitch_error_rad": 0.0},
                "policy_input_action": action,
                "policy_clipped_action": action,
                "applied_action": zero16,
                "terminal_reason": "time_limit" if truncated else None,
            }
            return np.zeros(99, dtype=np.float32), 0.0, False, truncated, info

    world.WorldUprightCourseEnv = FakeWorldUprightEnv
    monkeypatch.setitem(sys.modules, "world_upright_course_11", world)
    spec = importlib.util.spec_from_file_location(
        "short_curriculum_11", Path(__file__).with_name("short_curriculum_11.py"),
    )
    subject = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, subject)
    spec.loader.exec_module(subject)
    from rl16_curriculum_08 import REWARD_TERMS, STAGE_ARRAYS

    class FakeRuntime:
        control_completed = 0

        def control_step(self, env, action):
            result = env.step(action)
            self.control_completed += 1
            return result

        def state(self):
            return {"count": self.control_completed}

        def receipt(self):
            return {"completed": self.control_completed}

    budget = BudgetSpec(1000, 1000, 250, 4)
    runtime = FakeRuntime()
    env = FakeWorldUprightEnv()
    wrapper = subject.ShortCurriculumWrapper(
        env, runtime, tmp_path, budget=budget,
        plan=FrozenShortPlan(PLAN_SCHEMA, "2" * 64, select),
        caps=caps, measurement_seed=88623,
    )
    _, info = wrapper.reset(seed=88621)
    expected_seed = int(np.random.SeedSequence([88623, 0]).generate_state(
        1, dtype=np.uint32,
    )[0])
    assert env.last_seed == expected_seed != 88621
    assert info["episode_metadata"]["measurement_seed"] == 17
    for index in range(1000):
        _, _, terminated, truncated, _ = wrapper.step(np.zeros(16))
        assert not terminated and truncated == (index == 999)
        if index == 999:
            # DummyVecEnv resets before its rollout callback observes this
            # terminal transition. It must not select a new target or step.
            wrapper.reset()
            assert runtime.control_completed == 1000 and env.reset_calls == 2
            assert wrapper._postbudget_reset_count == 1
        prior = wrapper.last_completed_record
        wrapper.record_transition({
            "num_timesteps": index + 1,
            "done": prior["done"], "truncated": prior["truncated"],
            "terminal_reason": prior["terminal_reason"],
            "completed_control_intervals": prior["episode_tick"] + 1,
            "reward_before_bootstrap": 0.0,
            "raw_gaussian_action": np.zeros(16),
            "sb3_clipped_action": np.zeros(16),
            "effective_applied_action": np.zeros(16),
            "clip_active": False, "gate_zeroed_action": False,
        })
    assert len(wrapper.closed_episodes) == 1
    assert wrapper.closed_episodes[0]["completed_controls"] == 1000
    assert wrapper.closed_episodes[0]["truncated"] is True
    assert wrapper.gaussian_records == 1000
    with pytest.raises(RuntimeError):
        wrapper.step(np.zeros(16))
    assert wrapper.finalize_training()["zero_control_postbudget_resets"] == 1
