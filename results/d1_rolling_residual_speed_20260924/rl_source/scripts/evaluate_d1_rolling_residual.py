"""Eight fixed full-record held-out episodes for the loaded shared-leg policy."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.d1_jump_readiness_records import ScratchKinematics
from scripts.d1_rolling_residual_env import D1RollingResidualEnv
from scripts.d1_rolling_residual_scoring import score_rolling_episode
from scripts.d1_rolling_residual_task import RollingEpisodeSpec
from scripts.d1_single_step_records import (
    StreamingNativeObserver,
    compiled_geometry_manifest,
    jsonable,
)
from scripts.probe_d1_single_step import endpoint_record

EVAL_SEED = 77351
EVAL_CONTROL_PER_CASE = 1200
EVAL_NATIVE_PER_CASE = 6000
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_row(stream: Any, value: Any) -> None:
    stream.write(json.dumps(jsonable(value), allow_nan=False) + "\n")
    stream.flush()


def _initial_snapshot(inner: Any, observation: np.ndarray) -> dict[str, np.ndarray]:
    return {**{key: getattr(inner.plant.data, key).copy() for key in PAIR_FIELDS[:-1]},
            "observation": np.asarray(observation).copy()}


def _case(
    runtime: Any,
    env: D1RollingResidualEnv,
    model: Any,
    folder: Path,
    *,
    spec: RollingEpisodeSpec,
    actor: str,
    checkpoint_sha256: str,
    paired_initial: dict[str, np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    folder.mkdir(exist_ok=False)
    runtime.start_segment(folder.name, EVAL_CONTROL_PER_CASE)
    runtime.bind(env)
    env.configure_next_episode(spec)
    inner = env.unwrapped
    plant = inner.plant
    geometry_manifest = compiled_geometry_manifest(plant)
    _write_json(folder / "geometry_manifest.json", geometry_manifest)
    body_ids = list(range(1, plant.model.nbody))
    scratch = ScratchKinematics(
        plant.model, body_ids=body_ids, masses=plant.model.body_mass[body_ids],
        base_body_id=plant.base_body_id,
    )
    observer = StreamingNativeObserver(plant, folder)
    endpoints: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray | float]] = {
        key: [] for key in (*PAIR_FIELDS, "time")
    }
    completed = 0
    terminated = truncated = False
    error: dict[str, str] | None = None
    initial: dict[str, np.ndarray] | None = None

    def snapshot(obs: np.ndarray, tick: int, stream: Any) -> None:
        for key in PAIR_FIELDS[:-1]:
            arrays[key].append(getattr(plant.data, key).copy())
        arrays["observation"].append(np.asarray(obs).copy())
        arrays["time"].append(float(plant.data.time))
        row = endpoint_record(inner, scratch, tick)
        endpoints.append(row)
        _write_row(stream, row)

    try:
        with gzip.open(folder / "endpoints.jsonl.gz", "xt") as ep_stream, gzip.open(
            folder / "trace.jsonl.gz", "xt"
        ) as trace_stream, observer:
            observation, info = env.reset(seed=EVAL_SEED)
            _write_json(folder / "episode_metadata.json", info)
            initial = _initial_snapshot(inner, observation)
            snapshot(observation, 0, ep_stream)
            if paired_initial is not None:
                for key in PAIR_FIELDS:
                    if not np.array_equal(initial[key], paired_initial[key]):
                        raise RuntimeError(f"paired initial state differs: {key}")
            for tick in range(EVAL_CONTROL_PER_CASE):
                raw = dict(inner.command_records[-1])
                decision = inner.heading_decision
                if decision is None or decision.tick != tick:
                    raise RuntimeError("prepared controller decision/tick is inconsistent")
                action = (np.zeros(1, dtype=np.float32) if actor == "zero"
                          else np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32))
                if action.shape != (1,) or not np.isfinite(action).all():
                    raise RuntimeError("loaded deterministic policy action is invalid")
                observation, reward, terminated, truncated, info = runtime.control_step(env, action)
                completed += 1
                control = inner._controller.controller
                receipt = info["rolling_action"]
                row = {
                    "tick": tick, "endpoint_tick": tick + 1,
                    "policy_action": info["policy_action"],
                    "clipped_policy_scalar": receipt["clipped_scalar"],
                    "forward_gate_active": receipt["forward_gate_active"],
                    "action": info["physical_action"],
                    "applied_action": info["applied_action"],
                    "raw_command": raw,
                    "servo_command": asdict(decision.servo_command),
                    "world_command": asdict(decision.decision.world_command),
                    "stop_active": control.last_damping.active,
                    "turn_active": control.last_authority.active,
                    "torque_nm": inner.last_transition.requested_torque_nm,
                    "callback_count_after_prepare": inner.raw_callback_count,
                    "stop_record": asdict(control.last_damping),
                    "authority_record": asdict(control.last_authority),
                    "controller_result": asdict(control.last_result),
                    "reward": reward, "terminated": bool(terminated),
                    "truncated": bool(truncated), "terminal_reason": info.get("terminal_reason"),
                    "metrics": info.get("metrics"), "checkpoint_sha256": checkpoint_sha256,
                }
                trace.append(row)
                _write_row(trace_stream, row)
                snapshot(observation, tick + 1, ep_stream)
                if terminated or truncated:
                    break
    except BaseException as exc:  # noqa: BLE001 -- retain partial physical evidence and stop
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.unbind()
        _write_json(folder / "receipt.json", {
            "completed_control_intervals": completed,
            "native_receipt": observer.receipt(),
            "clock_native_steps": round(float(plant.data.time) / .002),
            "error": error, "terminated": bool(terminated), "truncated": bool(truncated),
            "reconstruction": scratch.receipt(), "callback_count": inner.raw_callback_count,
            "checkpoint_sha256": checkpoint_sha256, "no_retry_or_padding": True,
        })
        np.savez_compressed(folder / "states.npz", **{
            key: np.asarray(rows) for key, rows in arrays.items()
        })
        np.savez_compressed(
            folder / "terminal_integrator.npz", qpos=plant.data.qpos,
            qvel=plant.data.qvel, ctrl=plant.data.ctrl,
            qacc_warmstart=plant.data.qacc_warmstart, time=np.array(plant.data.time),
        )
        _write_json(folder / "command_records.json", inner.command_records)
    payload = {
        "obstacle_enabled": spec.obstacle_enabled, "spec": spec.as_dict(),
        "completed_control_intervals": completed, "endpoints": endpoints,
        "trace": trace, "native": observer.entries, "error": error,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "source_identity_valid": True, "geometry_manifest": geometry_manifest,
    }
    score = score_rolling_episode(payload, spec)
    _write_json(folder / "score.json", score)
    if error is not None or not score["record_valid"]:
        raise RuntimeError(f"{folder.name} execution or record invalid: {error or score['failure_reasons']}")
    if (len(observer.entries) != 5 * completed
            or observer.attempted_calls != 5 * completed
            or observer.returned_calls != 5 * completed):
        raise RuntimeError(f"{folder.name} native/control count mismatch")
    if completed < EVAL_CONTROL_PER_CASE and not (terminated or truncated):
        raise RuntimeError(f"{folder.name} stopped without a physical terminal signal")
    if initial is None:
        raise RuntimeError("held-out initial state is absent")
    return score, initial


def run_evaluation(
    runtime: Any,
    box_env: D1RollingResidualEnv,
    loaded_model: Any,
    output: Path,
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Consume at most eight fixed cases; valid task failures remain visible."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    runtime.unbind()
    plane_spec = RollingEpisodeSpec(.2, 175, 975, False)
    plane_env = runtime.construct(lambda: D1RollingResidualEnv(plane_spec))
    environments = {"plane": plane_env, "box": box_env}
    rows: list[dict[str, Any]] = []
    try:
        for speed in (.2, .25):
            for terrain in ("plane", "box"):
                paired: dict[str, np.ndarray] | None = None
                for actor in ("zero", "final_policy"):
                    name = f"speed{round(1000 * speed):03d}_{terrain}_{actor}"
                    spec = RollingEpisodeSpec(speed, 175, 975, terrain == "box")
                    score, initial = _case(
                        runtime, environments[terrain], loaded_model, output / name,
                        spec=spec, actor=actor, checkpoint_sha256=checkpoint_sha256,
                        paired_initial=paired,
                    )
                    if actor == "zero":
                        paired = initial
                    rows.append({"case": name, "speed_mps": speed, "terrain": terrain,
                                 "actor": actor, "score": score,
                                 "paired_initial_exact": actor == "final_policy"})
                    _write_json(output / f"{name}.case.json", rows[-1])
    finally:
        runtime.unbind()
        plane_env.close()
    speed_deltas: dict[str, float | None] = {}
    for terrain in ("plane", "box"):
        means = {row["speed_mps"]: row["score"]["metrics"]["speed_window_mean_body_vx_mps"]
                 for row in rows if row["terrain"] == terrain and row["actor"] == "final_policy"}
        speed_deltas[terrain] = (None if means[.25] is None or means[.2] is None
                                 else float(means[.25] - means[.2]))
    qualified = (len(rows) == 8
                 and all(row["score"]["task_passed"] for row in rows if row["actor"] == "final_policy")
                 and all(delta is not None and delta >= .03 for delta in speed_deltas.values()))
    result = {"cases": rows, "policy_speed_mean_delta_mps": speed_deltas,
              "measured_speed_increase_passed": all(v is not None and v >= .03
                                                    for v in speed_deltas.values()),
              "qualified": qualified,
              "causal_rl_improvement_claimed": False,
              "zero_outcomes_reported": True,
              "scope": "eight deterministic simulation episodes, not statistical reliability or hardware"}
    _write_json(output / "evaluation_summary.json", result)
    return result
