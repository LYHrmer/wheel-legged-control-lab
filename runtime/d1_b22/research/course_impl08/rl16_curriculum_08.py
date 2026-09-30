"""One-environment, counted 08-R curriculum and numeric training recorder.

The caller constructs and binds the real course env/runtime and owns the hard
process reservation. This wrapper does not create a plant, import MuJoCo, run
an engine function, or pick a checkpoint. Its sole physical call is
``runtime.control_step(self.env, action)``. It keeps the command RNG separate
from measurement seeds and archives every actual consumed control once.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from course_corridor_08 import assert_nominal_corridor_clearance
from full_drive_command_08 import QualifiedCommandCaps
from full_drive_schedule_08 import precheck_nominal_path, training_schedule
from full_drive_servo_08 import CONTROL_DT_S, FullDriveCommandServo

CURRICULUM_SCHEMA = "d1-course-rl16-65536x4-seeded-curriculum-v1"
BLOCK_SCHEMA = "d1-course-rl16-1024-control-numeric-block-v1"
TOTAL_TRAINING_CONTROLS = 262144
LEVEL_WINDOW_CONTROLS = 65536
BLOCK_CONTROLS = 1024
EPISODE_CONTROLS = 1600
REWARD_TERMS = (
    "tracking_vx", "tracking_vy", "tracking_yaw", "height", "attitude",
    "actual_torque", "action_change", "leg_speed", "termination",
)
STAGE_ARRAYS = {
    "consumed_joint_position_rad": (16,),
    "consumed_joint_velocity_rad_s": (16,),
    "consumed_nominal_joint_target_rad": (16,),
    "consumed_nominal_wheel_speed_rad_s": (4,),
    "consumed_support_torque_nm": (16,),
    "consumed_base_rotation_world_from_body": (3, 3),
    "consumed_foot_jacobian_world": (4, 3, 4),
    "consumed_wheel_omega_rad_s": (4,),
    "geometric_joint_target_rad": (16,),
    "rate_limited_joint_target_rad": (16,),
    "position_limited_joint_target_rad": (16,),
    "joint_target_rad": (16,),
    "wheel_speed_target_rad_s": (4,),
    "wheel_speed_error_rad_s": (4,),
    "wheel_integral_before_nm": (4,),
    "wheel_integral_candidate_nm": (4,),
    "wheel_integral_after_nm": (4,),
    "leg_pd_nm": (16,),
    "base_wheel_torque_nm": (16,),
    "base_request_torque_nm": (16,),
    "projected_jx": (4, 3),
    "relative_forward_mps": (4,),
    "leg_damping_delta_torque_nm": (16,),
    "common_wheel_delta_torque_nm": (16,),
    "final_request_torque_nm": (16,),
    "safe_torque_nm": (16,),
}


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be a real array of shape {shape}")
    result = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer,
                                                                      np.floating)):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _write_exclusive_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_npz(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


class _NumericBlocks:
    """One exclusive NPZ per 1024 completed controls; no per-tick disk writes."""

    def __init__(self, directory: Path, stem: str) -> None:
        self.directory = directory
        self.stem = stem
        self.directory.mkdir(exist_ok=False)
        self.pending: dict[str, list[np.ndarray]] = {}
        self.count = 0
        self.files: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        if not row or (self.pending and set(row) != set(self.pending)):
            raise ValueError("numeric block schema changed within training")
        checked: dict[str, np.ndarray] = {}
        for key, value in row.items():
            array = np.asarray(value)
            if array.dtype.kind not in "fiub" or not np.isfinite(array).all():
                raise ValueError(f"numeric training block field {key} is invalid")
            checked[key] = array.copy()
        for key, array in checked.items():
            self.pending.setdefault(key, []).append(array)
        self.count += 1
        if self.count == BLOCK_CONTROLS:
            self.flush()

    def flush(self) -> None:
        if not self.count:
            return
        index = len(self.files)
        path = self.directory / f"{self.stem}_{index:04d}.npz"
        arrays = {key: np.stack(values) for key, values in self.pending.items()}
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        self.files.append({
            "file": path.name, "rows": self.count,
            "first_control_index": int(arrays["control_index"][0]),
            "last_control_index": int(arrays["control_index"][-1]),
            "sha256": _sha256(path), "bytes": path.stat().st_size,
        })
        self.pending = {}
        self.count = 0


class RL16CurriculumWrapper(gym.Wrapper):
    """Current-course train wrapper with a single immutable physical owner."""

    def __init__(
        self, env: gym.Env, runtime: Any, output_dir: str | Path,
        *, caps: QualifiedCommandCaps, schedule_seed: int = 88401,
        measurement_seed: int = 88402,
        total_controls: int = TOTAL_TRAINING_CONTROLS,
    ) -> None:
        if (getattr(env, "mode", None) != "train" or getattr(env, "max_steps", None) != EPISODE_CONTROLS
                or getattr(env, "pure_test_components", True)):
            raise ValueError("curriculum requires the actual 1600-tick course train environment")
        if not isinstance(caps, QualifiedCommandCaps) or caps != env.caps:
            raise ValueError("curriculum caps must equal the physical env command caps")
        if (type(schedule_seed) is not int or schedule_seed < 0
                or type(measurement_seed) is not int or measurement_seed < 0
                or type(total_controls) is not int or total_controls != TOTAL_TRAINING_CONTROLS):
            raise ValueError("curriculum requires its fixed seeds and 262144-control bound")
        if not callable(getattr(runtime, "control_step", None)):
            raise TypeError("runtime must provide the counted control_step seam")
        if getattr(runtime, "control_completed", None) != 0:
            raise ValueError("new training runtime must start at zero completed controls")
        super().__init__(env)
        self.runtime = runtime
        self.caps = caps
        self.schedule_seed = schedule_seed
        self.measurement_seed_stream_seed = measurement_seed
        self._command_rng = np.random.default_rng(schedule_seed)
        self._measurement_rng = np.random.default_rng(measurement_seed)
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            raise ValueError("root launcher must precreate the unique output directory")
        self._numeric = _NumericBlocks(self.output_dir / "training_numeric_blocks", "controls")
        self._gaussian = _NumericBlocks(self.output_dir / "training_gaussian_blocks", "gaussian")
        self._events = (self.output_dir / "curriculum_events.ndjson").open("x", encoding="utf-8")
        self.completed_controls = 0
        self.gaussian_records = 0
        self.level = 0
        self._level_episode_counts = [0, 0, 0, 0]
        self._flat_episode_counts = [0, 0, 0, 0]
        self._recent: list[deque[bool]] = [deque(maxlen=10) for _ in range(4)]
        self.closed_episodes: list[dict[str, Any]] = []
        self._episode: dict[str, Any] | None = None
        self._active_schedule = None
        self._last_closed_schedule = None
        self._postbudget_reset_count = 0
        self._last_observation: np.ndarray | None = None
        self.last_completed_record: dict[str, Any] | None = None
        self._pending_gaussian_index: int | None = None
        self._fatal = False
        self._finalized = False
        self._event({
            "event": "curriculum_initialized", "schema": CURRICULUM_SCHEMA,
            "schedule_seed": schedule_seed, "measurement_seed_stream_seed": measurement_seed,
            "training_control_limit": total_controls,
        })

    def _event(self, value: dict[str, Any]) -> None:
        self._events.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())

    def _maybe_promote(self) -> None:
        boundary_level = min(self.completed_controls // LEVEL_WINDOW_CONTROLS, 3)
        if boundary_level <= self.level:
            return
        recent = self._recent[self.level]
        successes = sum(recent)
        if len(recent) == 10 and successes >= 8:
            old = self.level
            self.level += 1  # no skipping, even if later global boundaries passed
            self._event({
                "event": "level_promoted", "from": old, "to": self.level,
                "completed_global_controls": self.completed_controls,
                "recent_closed_episodes": len(recent), "recent_successes": successes,
            })

    def _new_schedule(self):
        self._maybe_promote()
        local = self._level_episode_counts[self.level]
        if self.level < 2:
            terrain = "flat"
        elif self.level == 2:
            terrain = ("flat", "bumps")[local % 2]
        else:
            terrain = ("flat", "bumps", "rough", "ramp")[local % 4]
        if terrain == "flat":
            flat_ordinal = self._flat_episode_counts[self.level]
            ceiling = (0.4, 0.9, 1.6, 1.6)[self.level]
            target = ceiling if (flat_ordinal + 1) % 4 == 0 else float(
                self._command_rng.uniform(0.0, ceiling)
            )
            yaw = 0.0 if self.level == 0 else float(self._command_rng.uniform(-0.3, 0.3))
        else:
            flat_ordinal = None
            target = float(self._command_rng.uniform(0.2, 0.4))
            yaw = 0.0
        index = len(self.closed_episodes)
        schedule = training_schedule(
            level=self.level, terrain=terrain, target_speed_mps=target,
            yaw_amplitude_rps=yaw, episode_index=index,
            flat_episode_ordinal=flat_ordinal, schedule_seed=self.schedule_seed,
        )
        self._level_episode_counts[self.level] += 1
        if terrain == "flat":
            self._flat_episode_counts[self.level] += 1
        return schedule, flat_ordinal

    def _flat_corridor_after_reset(self, schedule, *, episode_index: int) -> dict[str, Any]:
        """Pure path against this episode's actual compiled reset geom centers.

        This executes after the one necessary env reset but before any control.
        It reads native arrays only; counter snapshots prove it did not invoke
        an engine function or consume a control/native substep.
        """
        plant = self.env.unwrapped.plant
        before_c = self.runtime.state()
        before_python = self.runtime.receipt()
        model, data = plant.model, plant.data
        manifest = {
            "base_world_position_m": data.qpos[:3].copy().tolist(),
            "model_identity": {
                "model_address": int(model._address),
                "data_address": int(data._address),
                "source": "actual_compiled_course_reset_geometry_v1",
                "model_ngeom": int(model.ngeom),
            },
            "robot_collision_geoms": [
                {
                    "geom_id": gid,
                    "body_id": int(model.geom_bodyid[gid]),
                    "world_center_m": data.geom_xpos[gid].copy().tolist(),
                    "rbound_m": float(model.geom_rbound[gid]),
                    "collision": True,
                }
                for gid in range(model.ngeom)
                if model.geom_bodyid[gid] != 0
                and (model.geom_contype[gid] or model.geom_conaffinity[gid])
            ],
            "terrain_world_geoms": plant.collision_terrain_metadata["world_collision_geoms"],
        }
        servo = FullDriveCommandServo(self.caps)
        x, y = schedule.spawn_position_m[:2]
        heading = 0.0
        points = [(x, y)]
        for tick, raw in enumerate(schedule.raw_commands):
            applied = servo.advance(tick, raw).applied
            x += math.cos(heading) * applied.forward_velocity_mps * CONTROL_DT_S
            y += math.sin(heading) * applied.forward_velocity_mps * CONTROL_DT_S
            heading += applied.yaw_rate_rps * CONTROL_DT_S
            points.append((x, y))
        path = np.asarray(points, dtype=np.float64)
        stem = f"training_episode_{episode_index:06d}_flat_corridor"
        geometry_path = self.output_dir / f"{stem}_geometry.json"
        centerline_path = self.output_dir / f"{stem}_nominal_path.npz"
        receipt_path = self.output_dir / f"{stem}_precheck.json"
        _write_exclusive_json(geometry_path, manifest)
        _write_exclusive_npz(centerline_path, xy_m=path)
        report = None
        failure = None
        try:
            report = assert_nominal_corridor_clearance(manifest, path)
        except (TypeError, ValueError) as exc:
            failure = exc
        after_c = self.runtime.state()
        after_python = self.runtime.receipt()
        counters_unchanged = before_c == after_c and before_python == after_python
        result = {
            "schema": "d1-course-training-flat-reset-footprint-v1",
            "case_id": schedule.case_id,
            "nominal_only_not_actual_trajectory": True,
            "geometry_file": geometry_path.name,
            "geometry_sha256": _sha256(geometry_path),
            "centerline_file": centerline_path.name,
            "centerline_sha256": _sha256(centerline_path),
            "report": report,
            "passed": failure is None and counters_unchanged,
            "failure": None if failure is None else {
                "type": type(failure).__name__, "message": str(failure),
            },
            "counters_unchanged": counters_unchanged,
            "before_c": before_c,
            "after_c": after_c,
            "before_python": before_python,
            "after_python": after_python,
        }
        _write_exclusive_json(receipt_path, result)
        if not counters_unchanged:
            raise RuntimeError("flat footprint precheck changed engine or Python counters")
        if failure is not None:
            raise ValueError("flat reset footprint precheck rejected the selected schedule") from failure
        return {"file": receipt_path.name, "sha256": _sha256(receipt_path)}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self._fatal or self._finalized:
            raise RuntimeError("finished/fatal curriculum cannot reset")
        if self.completed_controls == TOTAL_TRAINING_CONTROLS:
            if (seed is not None or options not in (None, {}) or self._episode is not None
                    or self._postbudget_reset_count != 0
                    or self._pending_gaussian_index != TOTAL_TRAINING_CONTROLS - 1
                    or self.last_completed_record is None
                    or self.last_completed_record["done"] is not True
                    or self._last_closed_schedule is None):
                raise RuntimeError("only the final done transition may auto-reset at budget")
            try:
                schedule = self._last_closed_schedule
                observation, info = self.env.reset(
                    seed=int(self.closed_episodes[-1]["measurement_seed"]),
                    options={
                        "command_source": schedule.command_source,
                        "spawn_position_m": schedule.spawn_position_m,
                    },
                )
                _finite_array(observation, (99,), "post-budget autoreset observation")
                self._postbudget_reset_count = 1
                self._event({
                    "event": "zero_control_postbudget_reset",
                    "completed_global_controls": self.completed_controls,
                    "reused_case_id": schedule.case_id,
                    "reused_measurement_seed": self.closed_episodes[-1]["measurement_seed"],
                    "new_training_target_selected": False,
                })
                return observation, info
            except BaseException:
                self._fatal = True
                raise
        if self.completed_controls > TOTAL_TRAINING_CONTROLS:
            raise RuntimeError("training control limit was exceeded")
        if self._episode is not None:
            raise RuntimeError("cannot reset an unclosed physical training episode")
        if options not in (None, {}):
            raise ValueError("curriculum owns its explicit command and spawn reset options")
        try:
            before_rng = hashlib.sha256(json.dumps(
                self._command_rng.bit_generator.state, sort_keys=True,
            ).encode()).hexdigest()
            schedule, flat_ordinal = self._new_schedule()
            after_rng = hashlib.sha256(json.dumps(
                self._command_rng.bit_generator.state, sort_keys=True,
            ).encode()).hexdigest()
            nominal = precheck_nominal_path(schedule, self.caps)
            episode_index = self._level_episode_counts[self.level] - 1
            if type(seed) is int and seed >= 0:
                actual_measurement_seed = seed
                measurement_origin = "explicit_caller_seed"
            elif seed is None:
                actual_measurement_seed = int(self._measurement_rng.integers(0, 2**31))
                measurement_origin = "independent_measurement_rng_88402"
            else:
                raise TypeError("measurement seed must be a nonnegative integer or None")
            choice = {
                "schema": CURRICULUM_SCHEMA,
                "case_id": schedule.case_id, "episode_index": len(self.closed_episodes),
                "level": self.level, "level_episode_index": episode_index,
                "flat_episode_ordinal": flat_ordinal,
                "terrain": schedule.terrain, "spawn_position_m": schedule.spawn_position_m,
                "target_speed_mps": schedule.target_speed_mps,
                "yaw_amplitude_rps": schedule.yaw_amplitude_rps,
                "schedule_seed": self.schedule_seed,
                "command_rng_before_sha256": before_rng,
                "command_rng_after_sha256": after_rng,
                "measurement_seed": actual_measurement_seed,
                "measurement_seed_origin": measurement_origin,
                "start_completed_global_controls": self.completed_controls,
                "raw_command_sha256": schedule.command_sha256,
                "nominal_path_precheck": asdict(nominal),
                "raw_commands": [asdict(command) for command in schedule.raw_commands],
            }
            _write_exclusive_json(
                self.output_dir / f"training_episode_{len(self.closed_episodes):06d}_schedule.json",
                choice,
            )
            if not nominal.passed:
                raise ValueError("the selected seeded nominal path is infeasible; no redraw")
            observation, info = self.env.reset(
                seed=actual_measurement_seed,
                options={
                    "command_source": schedule.command_source,
                    "spawn_position_m": schedule.spawn_position_m,
                },
            )
            checked = _finite_array(observation, (99,), "initial 99D observation")
            footprint = (self._flat_corridor_after_reset(
                schedule, episode_index=len(self.closed_episodes),
            ) if schedule.terrain == "flat" else None)
            self._last_observation = checked.astype(np.float32)
            self._episode = {
                "case_id": schedule.case_id, "level": self.level,
                "terrain": schedule.terrain, "episode_index": len(self.closed_episodes),
                "start_control_index": self.completed_controls,
                "measurement_seed": actual_measurement_seed,
                "raw_command_sha256": schedule.command_sha256,
                "flat_footprint_precheck": footprint,
                "completed_controls": 0, "drive_sse": 0.0, "drive_count": 0,
                "reward_sum": 0.0,
            }
            self._active_schedule = schedule
            self._event({
                "event": "episode_reset", "case_id": schedule.case_id,
                "level": self.level, "terrain": schedule.terrain,
                "completed_global_controls": self.completed_controls,
                "measurement_seed": actual_measurement_seed,
                "raw_command_sha256": schedule.command_sha256,
                "flat_footprint_precheck": footprint,
            })
            return observation, info
        except BaseException:
            self._fatal = True
            raise

    def _numeric_row(self, *, control_index: int, observation: np.ndarray,
                     reward: float, terminated: bool, truncated: bool,
                     info: dict[str, Any]) -> dict[str, Any]:
        episode = self._episode
        if episode is None:
            raise RuntimeError("a physical control has no selected episode")
        calc = info["controller_record"]["calculation"]
        terms = info["reward_terms"]
        metrics = info["metrics"]
        raw = info["raw_operator_command"]
        applied_command = info["consumed_command"]
        if not isinstance(calc, dict) or not isinstance(terms, dict) or not isinstance(metrics, dict):
            raise TypeError("actual controller/reward/metrics records are required")
        reward_terms = np.asarray([_finite_number(terms[key], key) for key in REWARD_TERMS])
        if not math.isclose(float(np.sum(reward_terms)), reward, rel_tol=0, abs_tol=1e-9):
            raise ValueError("recorded reward terms differ from the returned reward")
        command_fields = ("forward_velocity_mps", "lateral_velocity_mps", "yaw_rate_rps",
                          "clearance_m")
        raw_vector = np.asarray([_finite_number(raw[key], key) for key in command_fields])
        servo_vector = np.asarray([
            _finite_number(applied_command[key], key) for key in command_fields
        ])
        if (raw.get("jump_requested") is not False or applied_command.get("jump_requested") is not False):
            raise ValueError("the 08-R pilot must not activate jump")
        tick = info["completed_control_intervals"] - 1
        schedule = self._active_schedule
        if (type(info["completed_control_intervals"]) is not int
                or tick != episode["completed_controls"]
                or schedule is None or raw != asdict(schedule.raw_commands[tick])):
            raise RuntimeError("consumed raw command differs from the preregistered episode")
        interval = info["native_interval_summary"]
        if interval["native_returns"] != 5:
            raise ValueError("completed training control lacks five native returns")
        traces = self.env.unwrapped.plant.last_control_interval_actuator_traces
        if len(traces) != 5:
            raise ValueError("completed training control lacks five actuator traces")
        row: dict[str, Any] = {
            "control_index": np.asarray(control_index, dtype=np.int64),
            "episode_index": np.asarray(episode["episode_index"], dtype=np.int32),
            "episode_tick": np.asarray(info["completed_control_intervals"] - 1, dtype=np.int32),
            "input_observation99": _finite_array(observation, (99,), "input observation").astype(np.float32),
            "policy_env_input16": _finite_array(info["policy_input_action"], (16,), "env input action"),
            "policy_clipped16": _finite_array(info["policy_clipped_action"], (16,), "clipped action"),
            "effective_action16": _finite_array(info["applied_action"], (16,), "effective action"),
            "raw_command_vx_vy_yaw_clearance": raw_vector,
            "servo_command_vx_vy_yaw_clearance": servo_vector,
            "reward_terms9": reward_terms,
            "reward": np.asarray(reward, dtype=np.float64),
            "terminated": np.asarray(terminated),
            "truncated": np.asarray(truncated),
            "body_com_vx_mps": np.asarray(_finite_number(metrics["body_com_vx_mps"], "COM vx")),
            "body_com_vy_mps": np.asarray(_finite_number(metrics["body_com_vy_mps"], "COM vy")),
            "body_yaw_rate_rps": np.asarray(_finite_number(metrics["body_yaw_rate_rps"], "yaw rate")),
            "base_x_m": np.asarray(_finite_number(metrics["x_m"], "base x")),
            "base_y_m": np.asarray(_finite_number(metrics["y_m"], "base y")),
            "clearance_m": np.asarray(_finite_number(metrics["clearance_m"], "clearance")),
            "relative_roll_rad": np.asarray(_finite_number(metrics["relative_roll_rad"], "relative roll")),
            "relative_pitch_rad": np.asarray(_finite_number(metrics["relative_pitch_rad"], "relative pitch")),
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
            raise ValueError("environment effective action differs from controller calculation")
        if not all(np.array_equal(row["stage_safe_torque_nm"], trace.requested_nm)
                   for trace in traces):
            raise ValueError("controller safe torque differs from actuator-channel request")
        return row

    def step(self, action):
        if self._fatal or self._finalized or self._episode is None:
            raise RuntimeError("curriculum requires an active selected physical episode")
        if self.completed_controls >= TOTAL_TRAINING_CONTROLS:
            raise RuntimeError("training control limit is exhausted")
        if self._pending_gaussian_index is not None:
            raise RuntimeError("the previous physical control has no raw Gaussian action receipt")
        before = getattr(self.runtime, "control_completed", None)
        if before != self.completed_controls:
            raise RuntimeError("runtime and curriculum completed counts differ before control")
        observation_before = self._last_observation
        if observation_before is None:
            raise RuntimeError("current control has no prepared observation")
        try:
            observation, reward, terminated, truncated, info = self.runtime.control_step(self.env, action)
            if (getattr(self.runtime, "control_completed", None) != before + 1
                    or type(terminated) is not bool or type(truncated) is not bool
                    or terminated and truncated):
                raise RuntimeError("runtime/episode completion differs from one actual control")
            index = self.completed_controls
            checked_reward = _finite_number(reward, "actual step reward")
            row = self._numeric_row(
                control_index=index, observation=observation_before,
                reward=checked_reward, terminated=terminated,
                truncated=truncated, info=info,
            )
            self._numeric.append(row)
            self.completed_controls += 1
            self._pending_gaussian_index = index
            self.last_completed_record = {
                "control_index": index,
                "policy_env_input_action": row["policy_env_input16"].copy(),
                "policy_clipped_action": row["policy_clipped16"].copy(),
                "effective_action": row["effective_action16"].copy(),
                "reward": checked_reward,
                "done": terminated or truncated,
                "truncated": truncated,
                "terminal_reason": info.get("terminal_reason"),
                "episode_index": self._episode["episode_index"],
                "episode_tick": int(row["episode_tick"]),
            }
            self._last_observation = _finite_array(
                observation, (99,), "returned 99D observation",
            ).astype(np.float32)
            self._episode["completed_controls"] += 1
            self._episode["reward_sum"] += checked_reward
            if row["servo_command_vx_vy_yaw_clearance"][0] > 0.0:
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
            raise RuntimeError("closing an episode without a physical record")
        n = episode["drive_count"]
        rms = math.sqrt(episode["drive_sse"] / n) if n else None
        task_passed = bool(not terminated and truncated and n > 0 and rms <= 0.12)
        closed = {
            **episode,
            "closed": True, "ended_by": info.get("terminal_reason"),
            "terminated": terminated, "truncated": truncated,
            "task_failure": terminated, "drive_tracking_rms_mps": rms,
            "promotion_eligible": task_passed,
            "end_completed_global_controls": self.completed_controls,
        }
        self.closed_episodes.append(closed)
        self._recent[self.level].append(task_passed)
        self._event({"event": "episode_closed", **closed})
        self._last_closed_schedule = self._active_schedule
        self._episode = None
        self._active_schedule = None

    def on_gaussian_transition(
        self, control_index: int, gaussian_action: Any, clipped_action: Any,
    ) -> None:
        """Bind a true SB3 sampled action after its env.step, even after auto-reset."""
        record = self.last_completed_record
        if (type(control_index) is not int or control_index != self._pending_gaussian_index
                or record is None or record["control_index"] != control_index
                or control_index != self.gaussian_records):
            raise RuntimeError("Gaussian callback is missing, repeated or out of control order")
        raw = _finite_array(gaussian_action, (16,), "preclip Gaussian sample")
        clipped = _finite_array(clipped_action, (16,), "SB3 clipped sample")
        if (not np.array_equal(np.clip(raw, -1.0, 1.0), clipped)
                or not np.array_equal(clipped, record["policy_env_input_action"])):
            raise ValueError("Gaussian/clipped sample does not match the actual env input")
        self._gaussian.append({
            "control_index": np.asarray(control_index, dtype=np.int64),
            "episode_index": np.asarray(record["episode_index"], dtype=np.int32),
            "episode_tick": np.asarray(record["episode_tick"], dtype=np.int32),
            "raw_gaussian_action16": raw,
            "clipped_action16": clipped,
            "effective_action16": record["effective_action"].copy(),
        })
        self.gaussian_records += 1
        self._pending_gaussian_index = None

    def record_transition(self, record: dict[str, Any]) -> None:
        """Consume one Opus rollout callback record after the matching real step.

        DummyVecEnv may already have reset the physical environment by this
        point. The comparison uses the retained completed-control record, not
        the current (possibly next-episode) env state.
        """
        if not isinstance(record, dict):
            raise TypeError("rollout callback must provide its actual record dict")
        prior = self.last_completed_record
        index = self._pending_gaussian_index
        if prior is None or index is None:
            raise RuntimeError("rollout callback has no unmatched completed control")
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
                    record.get("reward_before_bootstrap"), "callback prebootstrap reward",
                )) != np.float32(prior["reward"])):
            raise RuntimeError("rollout callback disagrees with the completed control")
        effective = _finite_array(
            record["effective_applied_action"], (16,), "callback effective action",
        )
        if not np.array_equal(effective, prior["effective_action"]):
            raise ValueError("callback effective action differs from the actual controller")
        raw = _finite_array(record["raw_gaussian_action"], (16,), "callback raw Gaussian")
        clipped = _finite_array(record["sb3_clipped_action"], (16,), "callback clipped action")
        if (type(record.get("clip_active")) is not bool
                or record["clip_active"] != bool(np.any(raw != clipped))
                or type(record.get("gate_zeroed_action")) is not bool
                or record["gate_zeroed_action"] != bool(
                    not np.any(effective) and np.any(clipped)
                )):
            raise RuntimeError("rollout callback clip/gate indicators are inconsistent")
        self.on_gaussian_transition(index, raw, clipped)

    def _seal(self, *, complete: bool, reason: str) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("training recorder was already sealed")
        self._numeric.flush()
        self._gaussian.flush()
        partial = None
        if self._episode is not None:
            partial = {
                **self._episode, "closed": False,
                "end_completed_global_controls": self.completed_controls,
                "reason": reason,
            }
            _write_exclusive_json(self.output_dir / "training_partial_episode.json", partial)
        manifest = {
            "schema": BLOCK_SCHEMA,
            "complete": complete,
            "reason": reason,
            "completed_controls": self.completed_controls,
            "gaussian_records": self.gaussian_records,
            "pending_gaussian_control_index": self._pending_gaussian_index,
            "closed_episodes": len(self.closed_episodes),
            "partial_episode": partial,
            "numeric_blocks": self._numeric.files,
            "gaussian_blocks": self._gaussian.files,
            "level_reached": self.level,
            "zero_control_postbudget_resets": self._postbudget_reset_count,
            "runtime_control_completed": getattr(self.runtime, "control_completed", None),
        }
        _write_exclusive_json(self.output_dir / "training_blocks_manifest.json", manifest)
        self._event({
            "event": "training_sealed", "complete": complete, "reason": reason,
            "completed_controls": self.completed_controls,
            "gaussian_records": self.gaussian_records,
        })
        self._events.close()
        self._finalized = True
        return manifest

    def finalize_training(self) -> dict[str, Any]:
        if (self._fatal or self.completed_controls != TOTAL_TRAINING_CONTROLS
                or self.gaussian_records != self.completed_controls
                or self._pending_gaussian_index is not None
                or getattr(self.runtime, "control_completed", None) != self.completed_controls):
            raise RuntimeError("full training record cannot be finalized")
        return self._seal(complete=True, reason="fixed_262144_control_budget_completed")

    def seal_partial(self, reason: str) -> dict[str, Any]:
        """Preserve actual completed work after a fatal/interrupted process."""
        if not reason or not isinstance(reason, str):
            raise ValueError("partial seal requires a concrete reason")
        self._fatal = True
        return self._seal(complete=False, reason=reason)
