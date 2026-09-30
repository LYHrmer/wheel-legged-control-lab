"""One physical owner and bounded recorder for a frozen short-episode plan.

The plan selects each episode by index before reset, never by the preceding
outcome. This wrapper creates no model and makes exactly one counted physical
call per ``step``. Root must supply the plan, budget, seeds and live runtime.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from budget_spec_11 import BudgetSpec
from full_drive_command_08 import QualifiedCommandCaps
from full_drive_servo_08 import FullDriveCommandServo
from rl16_curriculum_08 import (
    REWARD_TERMS,
    STAGE_ARRAYS,
    _finite_array,
    _finite_number,
    _NumericBlocks,
    _sha256,
    _write_exclusive_json,
    _write_exclusive_npz,
)
from short_corridor_11 import assert_nominal_corridor_clearance
from short_episode_plan_11 import FrozenShortPlan, ShortEpisodeSchedule
from geometry21 import qualify_reset21, compare_reset_to_template21
from geometry_runtime22 import (capture_actual_reset_geometry22,
                                load_sealed_templates22)
from recipes22 import SPEC as SPEC22
from world_upright_course_11 import (
    WORLD_UPRIGHT_REFERENCE_SCHEMA,
    WORLD_UPRIGHT_REWARD_SCHEMA,
    WORLD_UPRIGHT_TASK_SCHEMA,
    WorldUprightCourseEnv,
)

SHORT_CURRICULUM_SCHEMA = "d1-course-world-upright-short1000-curriculum-v1"
SHORT_BLOCK_SCHEMA = "d1-course-world-upright-short1024-numeric-block-v1"
EPISODE_CONTROLS = 1000
BLOCK_CONTROLS = 1024


class Curriculum20(gym.Wrapper):
    """Record one actual 99D/16D training stream without outcome selection."""

    def __init__(
        self, env: WorldUprightCourseEnv, runtime: Any, output_dir: str | Path,
        *, budget: BudgetSpec, plan: FrozenShortPlan, caps: QualifiedCommandCaps,
        measurement_seed: int, writer: Any, source_offset: int = 0,
    ) -> None:
        if (type(env) is not WorldUprightCourseEnv or env.mode != "train"
                or env.max_steps != EPISODE_CONTROLS or env.pure_test_components):
            raise ValueError("short curriculum needs the actual 1000-tick upright train env")
        if not isinstance(budget, BudgetSpec) or not isinstance(plan, FrozenShortPlan):
            raise TypeError("short curriculum requires explicit budget and reviewed plan")
        if not isinstance(caps, QualifiedCommandCaps) or caps != env.caps:
            raise ValueError("curriculum caps differ from the actual env")
        if type(measurement_seed) is not int or not 0 <= measurement_seed < 2**32:
            raise ValueError("measurement stream needs an explicit uint32 seed")
        if getattr(runtime, "control_completed", None) != 0:
            raise ValueError("training must start on a fresh counted runtime")
        super().__init__(env)
        self.runtime, self.budget, self.plan, self.caps = runtime, budget, plan, caps
        self.measurement_seed = measurement_seed
        self.writer, self.source_offset = writer, source_offset
        (self._geometry_templates22,self._geometry_reference_controls22,
         self._geometry_input_identities22) = load_sealed_templates22(SPEC22)
        self._full_rows, self._full_files = [], []
        self._pre_state20 = None
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            raise ValueError("root must precreate the output directory")
        self._numeric = _NumericBlocks(self.output_dir / "training_numeric_blocks", "controls")
        self._gaussian = _NumericBlocks(self.output_dir / "training_gaussian_blocks", "gaussian")
        self._events = (self.output_dir / "curriculum_events.ndjson").open("x", encoding="utf-8")
        self.completed_controls = 0
        self.gaussian_records = 0
        self.closed_episodes: list[dict[str, Any]] = []
        self._episode: dict[str, Any] | None = None
        self._active_schedule: ShortEpisodeSchedule | None = None
        self._last_closed_schedule: ShortEpisodeSchedule | None = None
        self._last_measurement_seed: int | None = None
        self._last_observation: np.ndarray | None = None
        self.last_completed_record: dict[str, Any] | None = None
        self._pending_gaussian_index: int | None = None
        self._postbudget_reset_count = 0
        self._fatal = False
        self._finalized = False
        self._event({
            "event": "curriculum_initialized", "schema": SHORT_CURRICULUM_SCHEMA,
            "budget_spec": budget.as_dict(),
            "budget_spec_sha256": budget.canonical_sha256(),
            "plan_schema": plan.schema, "plan_source_sha256": plan.source_sha256,
            "measurement_seed_stream_seed": measurement_seed,
        })

    def _event(self, value: dict[str, Any]) -> None:
        self._events.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())

    def _nominal_path(self, schedule: ShortEpisodeSchedule) -> np.ndarray:
        servo = FullDriveCommandServo(self.caps)
        x, y = schedule.spawn_position_m[:2]
        heading = 0.0
        points = [(x, y)]
        for tick, raw in enumerate(schedule.raw_commands):
            applied = servo.advance(tick, raw).applied
            x += math.cos(heading) * applied.forward_velocity_mps * 0.01
            y += math.sin(heading) * applied.forward_velocity_mps * 0.01
            heading += applied.yaw_rate_rps * 0.01
            points.append((x, y))
        path = np.asarray(points, dtype=np.float64)
        if not np.isfinite(path).all():
            raise ValueError("short plan nominal path is nonfinite")
        if (np.max(np.abs(path[:, 0])) >= 10.5
                or np.max(np.abs(path[:, 1])) >= 5.8
                or schedule.terrain == "flat" and np.max(path[:, 1]) >= -3.6):
            raise ValueError("selected short episode nominal path exceeds its lane")
        return path

    def _flat_footprint_after_reset(self, schedule: ShortEpisodeSchedule,
                                    path: np.ndarray) -> dict[str, Any]:
        plant = self.env.unwrapped.plant
        before_c, before_python = self.runtime.state(), self.runtime.receipt()
        model, live = plant.model, plant.data
        data = plant.measurement_data
        if (data is live or any(
            not np.array_equal(np.asarray(getattr(data, field)),
                               np.asarray(getattr(live, field)))
            for field in ("qpos", "qvel", "ctrl")
        ) or float(data.time) != float(live.time)):
            raise RuntimeError("flat reset measurement and live state differ")
        manifest = {
            "base_world_position_m": data.qpos[:3].copy().tolist(),
            "model_identity": {
                "model_address": int(model._address), "data_address": int(data._address),
                "source": "actual_compiled_course_reset_geometry_v1",
                "model_ngeom": int(model.ngeom),
            },
            "robot_collision_geoms": [
                {"geom_id": gid, "body_id": int(model.geom_bodyid[gid]),
                 "world_center_m": data.geom_xpos[gid].copy().tolist(),
                 "rbound_m": float(model.geom_rbound[gid]), "collision": True}
                for gid in range(model.ngeom)
                if model.geom_bodyid[gid] != 0
                and (model.geom_contype[gid] or model.geom_conaffinity[gid])
            ],
            "terrain_world_geoms": plant.collision_terrain_metadata["world_collision_geoms"],
        }
        stem = f"training_episode_{schedule.episode_index:06d}_flat_corridor"
        geometry_file = self.output_dir / f"{stem}_geometry.json"
        path_file = self.output_dir / f"{stem}_nominal_path.npz"
        receipt_file = self.output_dir / f"{stem}_precheck.json"
        _write_exclusive_json(geometry_file, manifest)
        _write_exclusive_npz(path_file, xy_m=path)
        report = assert_nominal_corridor_clearance(manifest, path)
        after_c, after_python = self.runtime.state(), self.runtime.receipt()
        unchanged = before_c == after_c and before_python == after_python
        receipt = {
            "schema": "d1-course-short-flat-reset-footprint-v1",
            "case_id": schedule.case_id, "nominal_only_not_actual_trajectory": True,
            "geometry_file": geometry_file.name, "geometry_sha256": _sha256(geometry_file),
            "centerline_file": path_file.name, "centerline_sha256": _sha256(path_file),
            "report": report, "counters_unchanged": unchanged,
            "passed": unchanged,
        }
        _write_exclusive_json(receipt_file, receipt)
        if not unchanged:
            raise RuntimeError("flat footprint check changed physical counters")
        return {"file": receipt_file.name, "sha256": _sha256(receipt_file)}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self._fatal or self._finalized:
            raise RuntimeError("fatal/finalized curriculum cannot reset")
        if self.completed_controls == self.budget.total_controls:
            if (seed is not None or options not in (None, {}) or self._episode is not None
                    or self._postbudget_reset_count or self._last_closed_schedule is None
                    or self._pending_gaussian_index != self.budget.total_controls - 1
                    or self.last_completed_record is None
                    or self.last_completed_record["done"] is not True):
                raise RuntimeError("only the last done transition may auto-reset at budget")
            schedule = self._last_closed_schedule
            observation, info = self.env.reset(
                seed=self._last_measurement_seed,
                options={"command_source": schedule.command_source,
                         "spawn_position_m": schedule.spawn_position_m},
            )
            _finite_array(observation, (99,), "post-budget zero-control reset")
            self._postbudget_reset_count = 1
            self._event({"event": "zero_control_postbudget_reset",
                         "completed_global_controls": self.completed_controls,
                         "reused_case_id": schedule.case_id, "new_training_target_selected": False})
            return observation, info
        if self._episode is not None or options not in (None, {}):
            raise RuntimeError("reset cannot replace an active physical episode")
        try:
            index = len(self.closed_episodes)
            schedule = self.plan.episode(index)
            schedule_record = schedule.record(self.caps)
            path = self._nominal_path(schedule)
            if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
                raise TypeError("SB3 reset seed must be None or uint32")
            episode_reset_seed = int(np.random.SeedSequence(
                [self.measurement_seed, index + self.source_offset],
            ).generate_state(1, dtype=np.uint32)[0])
            schedule_record.update({
                "plan_schema": self.plan.schema,
                "plan_source_sha256": self.plan.source_sha256,
                "episode_reset_seed": episode_reset_seed,
                "episode_reset_seed_origin": (
                    f"SeedSequence([{self.measurement_seed},source_episode_index])"
                ),
                "sb3_reset_seed_observed": seed,
                "start_completed_global_controls": self.completed_controls,
                "nominal_path_end_xy_m": path[-1].tolist(),
            })
            schedule_file = self.output_dir / f"training_episode_{index:06d}_schedule.json"
            _write_exclusive_json(schedule_file, schedule_record)
            observation, info = self.env.reset(
                seed=episode_reset_seed,
                options={"command_source": schedule.command_source,
                         "spawn_position_m": schedule.spawn_position_m},
            )
            if info["episode_metadata"].get("reference_schema") != WORLD_UPRIGHT_REFERENCE_SCHEMA:
                raise RuntimeError("new episode did not publish the world-upright reference")
            provider_seed = info["episode_metadata"].get("measurement_seed")
            if type(provider_seed) is not int or not 0 <= provider_seed < 2**32:
                raise RuntimeError("actual provider measurement seed is missing")
            reset_file = self.output_dir / f"training_episode_{index:06d}_reset.json"
            _write_exclusive_json(reset_file, {
                "episode_index": index,
                "episode_reset_seed": episode_reset_seed,
                "provider_measurement_seed": provider_seed,
                "actual_episode_metadata": info["episode_metadata"],
            })
            footprint = (self._flat_footprint_after_reset(schedule, path)
                         if schedule.terrain == "flat" else None)
            self._last_observation = _finite_array(
                observation, (99,), "initial short episode observation",
            ).astype(np.float32)
            self._episode = {
                "episode_index": index, "source_episode_index": index + self.source_offset, "case_id": schedule.case_id,
                "terrain": schedule.terrain,
                "episode_reset_seed": episode_reset_seed,
                "provider_measurement_seed": provider_seed,
                "reset_file": reset_file.name,
                "raw_command_sha256": schedule.command_sha256,
                "schedule_file": schedule_file.name,
                "flat_footprint_precheck": footprint,
                "start_control_index": self.completed_controls,
                "completed_controls": 0, "reward_sum": 0.0,
                "drive_sse": 0.0, "drive_count": 0,
            }
            self._active_schedule = schedule
            from run_rl16_training_08 import _capture_state
            from state21 import control_state20
            self._pre_state20 = _capture_state(self.env, observation)
            reset_control = control_state20(self.env)
            self.writer.commit_npz(self.output_dir / f"training_episode_{index:06d}_initial_state.npz", self._pre_state20)
            self.writer.commit_json(self.output_dir / f"training_episode_{index:06d}_control_reset.json", reset_control)
            before_geometry = (self.runtime.state(),self.runtime.receipt())
            geometry = capture_actual_reset_geometry22(self.env)
            geometry_prefix = self.output_dir/f'training_episode_{index:06d}_geometry21'
            self.writer.commit_json(Path(str(geometry_prefix)+'_reset_manifest.json'),geometry)
            qualification,qualified_path = qualify_reset21(
                schedule,geometry,self._pre_state20,kind='train',
                source_index=index+self.source_offset)
            expected_geometry,expected_initial = self._geometry_templates22[schedule.terrain]
            bridge = compare_reset_to_template21(
                expected_geometry,geometry,expected_initial,self._pre_state20,
                self._geometry_reference_controls22[schedule.terrain],reset_control)
            if not bridge['passed']:
                self.writer.commit_json(Path(str(geometry_prefix)+'_reset_homotopy.json'),bridge)
                raise RuntimeError('actual training reset differs from finite source-derived template')
            if before_geometry!=(self.runtime.state(),self.runtime.receipt()):
                raise RuntimeError('geometry capture or qualification changed physical counters')
            self.writer.commit_json(Path(str(geometry_prefix)+'_receipt.json'),qualification)
            self.writer.commit_npz(Path(str(geometry_prefix)+'_nominal_path.npz'),
                                   {'path_xy':qualified_path})
            self.writer.commit_json(Path(str(geometry_prefix)+'_reset_homotopy.json'),bridge)
            self._event({"event": "episode_reset", **self._episode})
            return observation, info
        except BaseException:
            self._fatal = True
            raise

    def _numeric_row(self, observation: np.ndarray, reward: float, terminated: bool,
                     truncated: bool, info: dict[str, Any]) -> dict[str, Any]:
        episode, schedule = self._episode, self._active_schedule
        if episode is None or schedule is None:
            raise RuntimeError("completed control lacks its selected short episode")
        if (info.get("task_schema") != WORLD_UPRIGHT_TASK_SCHEMA
                or info.get("reward_schema") != WORLD_UPRIGHT_REWARD_SCHEMA
                or info.get("reference_schema") != WORLD_UPRIGHT_REFERENCE_SCHEMA):
            raise RuntimeError("completed control differs from the world-upright task")
        tick = info["completed_control_intervals"] - 1
        raw = info["raw_operator_command"]
        if (type(tick) is not int or tick != episode["completed_controls"]
                or raw != asdict(schedule.raw_commands[tick])):
            raise RuntimeError("actual raw command differs from the saved short schedule")
        interval, calc = info["native_interval_summary"], info["controller_record"]["calculation"]
        if interval["native_returns"] != 5:
            raise RuntimeError("completed control lacks five native returns")
        traces = self.env.unwrapped.plant.last_control_interval_actuator_traces
        if len(traces) != 5:
            raise RuntimeError("completed control lacks five actual actuator traces")
        terms, metrics = info["reward_terms"], info["metrics"]
        reward_terms = np.asarray([_finite_number(terms[name], name) for name in REWARD_TERMS])
        if not math.isclose(float(np.sum(reward_terms)), reward, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("saved reward decomposition differs from returned reward")
        command_fields = ("forward_velocity_mps", "lateral_velocity_mps",
                          "yaw_rate_rps", "clearance_m")
        raw_command = np.asarray([_finite_number(raw[name], name) for name in command_fields])
        servo_command = np.asarray([
            _finite_number(info["consumed_command"][name], name) for name in command_fields
        ])
        row = {
            "control_index": np.asarray(self.completed_controls, dtype=np.int64),
            "episode_index": np.asarray(episode["episode_index"], dtype=np.int32),
            "episode_tick": np.asarray(tick, dtype=np.int32),
            "input_observation99": _finite_array(observation, (99,), "input observation").astype(np.float32),
            "policy_env_input16": _finite_array(info["policy_input_action"], (16,), "env action"),
            "policy_clipped16": _finite_array(info["policy_clipped_action"], (16,), "clipped action"),
            "effective_action16": _finite_array(info["applied_action"], (16,), "effective action"),
            "raw_command_vx_vy_yaw_clearance": raw_command,
            "servo_command_vx_vy_yaw_clearance": servo_command,
            "reward_terms9": reward_terms,
            "reward": np.asarray(reward, dtype=np.float64),
            "terminated": np.asarray(terminated), "truncated": np.asarray(truncated),
            "body_com_vx_mps": np.asarray(_finite_number(metrics["body_com_vx_mps"], "COM vx")),
            "body_com_vy_mps": np.asarray(_finite_number(metrics["body_com_vy_mps"], "COM vy")),
            "body_yaw_rate_rps": np.asarray(_finite_number(metrics["body_yaw_rate_rps"], "yaw rate")),
            "base_x_m": np.asarray(_finite_number(metrics["x_m"], "base x")),
            "base_y_m": np.asarray(_finite_number(metrics["y_m"], "base y")),
            "clearance_m": np.asarray(_finite_number(metrics["clearance_m"], "clearance")),
            "geometry_relative_roll_rad": np.asarray(_finite_number(
                metrics["relative_roll_rad"], "geometry relative roll")),
            "geometry_relative_pitch_rad": np.asarray(_finite_number(
                metrics["relative_pitch_rad"], "geometry relative pitch")),
            "task_roll_error_rad": np.asarray(_finite_number(metrics["task_roll_error_rad"], "task roll")),
            "task_pitch_error_rad": np.asarray(_finite_number(metrics["task_pitch_error_rad"], "task pitch")),
            "native_nonwheel_contacts": np.asarray(interval["nonwheel_contact_count"], dtype=np.int32),
            "actuator_delayed5x16": np.stack([
                _finite_array(trace.delayed_nm, (16,), "delayed actuator torque") for trace in traces
            ]),
            "actuator_applied5x16": np.stack([
                _finite_array(trace.applied_nm, (16,), "applied actuator torque") for trace in traces
            ]),
        }
        for field, shape in STAGE_ARRAYS.items():
            row[f"stage_{field}"] = _finite_array(calc[field], shape, field)
        if not np.array_equal(row["effective_action16"], calc["applied_action"]):
            raise ValueError("saved effective action differs from the actual controller")
        if not all(np.array_equal(row["stage_safe_torque_nm"], trace.requested_nm)
                   for trace in traces):
            raise ValueError("actual actuator request differs from the controller safe torque")
        from run_rl16_training_08 import _capture_state
        from scripts.run_d1_latest_rl import _jsonable
        post = _capture_state(self.env, self.env._last_observation)
        # The final terminal observation is returned by runtime; it is replaced
        # with that exact array below after this numeric hook completes.
        self._full_rows.append(_jsonable({
            'control_index': self.completed_controls, 'episode_index': episode['episode_index'],
            'source_episode_index': episode['episode_index'] + self.source_offset,
            'episode_tick': tick, 'terrain': schedule.terrain,
            'input_observation99': observation, 'policy_input_action': info['policy_input_action'],
            'reward': reward, 'terminated': terminated, 'truncated': truncated,
            'info': info, 'pre_state': self._pre_state20, 'post_state': post,
            'native_actuator_traces': [asdict(trace) for trace in traces],
        }))
        self._pre_state20 = post
        return row

    def _flush_full20(self):
        if self._full_rows:
            path = self.output_dir / f'full_control_records_{len(self._full_files):04d}.jsonl.gz'
            receipt = self.writer.commit_gzip_rows(path, self._full_rows)
            self._full_files.append({'file': path.name, 'rows': len(self._full_rows),
                                    'archive_manifest': receipt['manifest'], **receipt['payloads'][0]})
            self._full_rows = []

    def step(self, action):
        if self._fatal or self._finalized or self._episode is None:
            raise RuntimeError("short curriculum has no active episode")
        if self.completed_controls >= self.budget.total_controls:
            raise RuntimeError("short training control budget is exhausted")
        if self._pending_gaussian_index is not None:
            raise RuntimeError("previous control lacks its sampled Gaussian receipt")
        before = getattr(self.runtime, "control_completed", None)
        if before != self.completed_controls or self._last_observation is None:
            raise RuntimeError("runtime count or prepared observation differs")
        try:
            observation, reward, terminated, truncated, info = self.runtime.control_step(self.env, action)
            if (getattr(self.runtime, "control_completed", None) != before + 1
                    or type(terminated) is not bool or type(truncated) is not bool
                    or terminated and truncated):
                raise RuntimeError("runtime did not complete exactly one physical control")
            checked_reward = _finite_number(reward, "physical reward")
            row = self._numeric_row(self._last_observation, checked_reward,
                                    terminated, truncated, info)
            exact_observation = _finite_array(observation, (99,), "returned actual observation").astype(np.float32)
            self._full_rows[-1]['post_state']['observation'] = exact_observation.tolist()
            self._pre_state20['observation'] = exact_observation.copy()
            if len(self._full_rows) == 200:
                self._flush_full20()
            self._numeric.append(row)
            index = self.completed_controls
            self.completed_controls += 1
            self._pending_gaussian_index = index
            self.last_completed_record = {
                "control_index": index,
                "episode_index": self._episode["episode_index"],
                "episode_tick": int(row["episode_tick"]),
                "policy_env_input_action": row["policy_env_input16"].copy(),
                "policy_clipped_action": row["policy_clipped16"].copy(),
                "effective_action": row["effective_action16"].copy(),
                "reward": checked_reward, "done": terminated or truncated,
                "truncated": truncated, "terminal_reason": info.get("terminal_reason"),
            }
            self._last_observation = _finite_array(
                observation, (99,), "returned actual observation",
            ).astype(np.float32)
            self._episode["completed_controls"] += 1
            self._episode["reward_sum"] += checked_reward
            if float(row["servo_command_vx_vy_yaw_clearance"][0]) > 0.0:
                delta = (float(row["body_com_vx_mps"])
                         - float(row["servo_command_vx_vy_yaw_clearance"][0]))
                self._episode["drive_sse"] += delta * delta
                self._episode["drive_count"] += 1
            if terminated or truncated:
                self._close_episode(info, terminated=terminated, truncated=truncated)
            return observation, checked_reward, terminated, truncated, info
        except BaseException:
            self._fatal = True
            raise

    def _close_episode(self, info: dict[str, Any], *, terminated: bool, truncated: bool) -> None:
        episode = self._episode
        if episode is None:
            raise RuntimeError("closing a missing short episode")
        count = episode["drive_count"]
        rms = math.sqrt(episode["drive_sse"] / count) if count else None
        closed = {
            **episode, "closed": True, "terminated": terminated,
            "truncated": truncated, "task_failure": terminated,
            "ended_by": info.get("terminal_reason"),
            "drive_tracking_rms_mps": rms,
            "end_completed_global_controls": self.completed_controls,
        }
        self.closed_episodes.append(closed)
        self._event({"event": "episode_closed", **closed})
        self._last_closed_schedule = self._active_schedule
        self._last_measurement_seed = episode["episode_reset_seed"]
        self._episode = None
        self._active_schedule = None

    def record_transition(self, record: dict[str, Any]) -> None:
        """Pair SB3's sampled Gaussian after the matching counted env.step."""
        prior = self.last_completed_record
        index = self._pending_gaussian_index
        if not isinstance(record, dict) or prior is None or index is None:
            raise RuntimeError("rollout callback has no unmatched physical control")
        if (type(record.get("num_timesteps")) is not int
                or record["num_timesteps"] != index + 1
                or type(record.get("done")) is not bool
                or record["done"] != prior["done"]
                or type(record.get("truncated")) is not bool
                or record["truncated"] != prior["truncated"]
                or record.get("terminal_reason") != prior["terminal_reason"]
                or type(record.get("completed_control_intervals")) is not int
                or record["completed_control_intervals"] != prior["episode_tick"] + 1
                or np.float32(_finite_number(
                    record.get("reward_before_bootstrap"), "prebootstrap reward",
                )) != np.float32(prior["reward"])):
            raise RuntimeError("rollout callback differs from the actual completed control")
        raw = _finite_array(record["raw_gaussian_action"], (16,), "raw Gaussian")
        clipped = _finite_array(record["sb3_clipped_action"], (16,), "SB3 clipped")
        effective = _finite_array(record["effective_applied_action"], (16,), "effective")
        if (not np.array_equal(np.clip(raw, -1.0, 1.0), clipped)
                or not np.array_equal(clipped, prior["policy_env_input_action"])
                or not np.array_equal(effective, prior["effective_action"])
                or type(record.get("clip_active")) is not bool
                or record["clip_active"] != bool(np.any(raw != clipped))
                or type(record.get("gate_zeroed_action")) is not bool
                or record["gate_zeroed_action"] != bool(not np.any(effective) and np.any(clipped))):
            raise RuntimeError("Gaussian, clipped and effective actions do not match")
        if index != self.gaussian_records:
            raise RuntimeError("Gaussian transition was repeated or skipped")
        self._gaussian.append({
            "control_index": np.asarray(index, dtype=np.int64),
            "episode_index": np.asarray(prior["episode_index"], dtype=np.int32),
            "episode_tick": np.asarray(prior["episode_tick"], dtype=np.int32),
            "raw_gaussian_action16": raw,
            "clipped_action16": clipped,
            "effective_action16": effective,
        })
        self.gaussian_records += 1
        self._pending_gaussian_index = None

    def _seal(self, *, complete: bool, reason: str) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("training recorder has already been sealed")
        self._flush_full20()
        self._numeric.flush()
        self._gaussian.flush()
        partial = None
        if self._episode is not None:
            partial = {**self._episode, "closed": False,
                       "end_completed_global_controls": self.completed_controls,
                       "reason": reason}
            _write_exclusive_json(self.output_dir / "training_partial_episode.json", partial)
        result = {
            "schema": SHORT_BLOCK_SCHEMA, "complete": complete, "reason": reason,
            "budget_spec": self.budget.as_dict(),
            "budget_spec_sha256": self.budget.canonical_sha256(),
            "completed_controls": self.completed_controls,
            "gaussian_records": self.gaussian_records,
            "pending_gaussian_control_index": self._pending_gaussian_index,
            "closed_episodes": len(self.closed_episodes), "partial_episode": partial,
            "full_control_blocks20": self._full_files,
            "env_mode": "train", "archive_force_sampling_mode": "heldout/full_contact",
            "source_episode_offset": self.source_offset,
            "numeric_blocks": self._numeric.files,
            "gaussian_blocks": self._gaussian.files,
            "zero_control_postbudget_resets": self._postbudget_reset_count,
            "runtime_control_completed": getattr(self.runtime, "control_completed", None),
        }
        _write_exclusive_json(self.output_dir / "training_blocks_manifest.json", result)
        self._event({"event": "training_sealed", "complete": complete,
                     "reason": reason, "completed_controls": self.completed_controls})
        self._events.close()
        self._finalized = True
        return result

    def finalize_training(self) -> dict[str, Any]:
        if (self._fatal or self.completed_controls != self.budget.total_controls
                or self.gaussian_records != self.completed_controls
                or self._pending_gaussian_index is not None
                or getattr(self.runtime, "control_completed", None) != self.completed_controls):
            raise RuntimeError("short training does not satisfy the explicit full budget")
        return self._seal(complete=True, reason="explicit_budget_completed")

    def seal_partial(self, reason: str) -> dict[str, Any]:
        if type(reason) is not str or not reason:
            raise ValueError("partial seal requires a concrete reason")
        self._fatal = True
        return self._seal(complete=False, reason=reason)


__all__ = ("SHORT_BLOCK_SCHEMA", "SHORT_CURRICULUM_SCHEMA", "Curriculum20")
