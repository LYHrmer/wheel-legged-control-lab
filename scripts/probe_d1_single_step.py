"""Execute only the frozen paired 1200+1200 zero-action 15 mm step contract."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.d1_jump_readiness_records import ScratchKinematics
from scripts.d1_single_step_geometry import robot_collision_bounds
from scripts.d1_single_step_records import (
    PhysicsCallLedger,
    StreamingNativeObserver,
    compiled_geometry_manifest,
    jsonable,
)
from scripts.qualify_d1_single_step import ROOT, source_manifest

CONTRACT_SHA256 = "1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238"


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_row(stream, row):
    stream.write(json.dumps(jsonable(row), allow_nan=False)+"\n")
    stream.flush()


def quaternion_rpy(q):
    w, x, y, z = q
    return np.array([np.arctan2(2*(w*x+y*z), 1.-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1., 1.)),
                     np.arctan2(2*(w*z+x*y), 1.-2*(y*y+z*z))])


def endpoint_record(env, scratch, tick):
    plant = env.plant
    state = scratch.reconstruct(plant.data.qpos, plant.data.qvel,
        label="returned_control_endpoint", time_s=float(plant.data.time))
    truth = env.decision.context.state if tick == 0 else env.last_transition.truth
    rpy = quaternion_rpy(plant.data.qpos[3:7])
    if not np.allclose(rpy, truth.base_rpy, rtol=0., atol=1e-12):
        raise RuntimeError("endpoint quaternion differs from synchronized truth")
    nonwheel = 0
    for c in plant._measurement_data.contact:
        g1, g2 = int(c.geom1), int(c.geom2)
        t1, t2 = g1 in plant.terrain_geom_ids, g2 in plant.terrain_geom_ids
        if t1 != t2:
            g = g2 if t1 else g1
            nonwheel += int(int(plant.model.geom_bodyid[g]) not in plant.wheel_body_ids)
    return {"tick": tick, "time_s": float(plant.data.time),
        "base_position_m": plant.data.qpos[:3].tolist(), "rpy_rad": rpy.tolist(),
        "body_vx_mps": float(truth.base_linear_velocity_body[0]),
        "com_vz_mps": float(state["com_velocity_mps"][2]),
        "com_position_m": state["com_position_m"], "com_velocity_mps": state["com_velocity_mps"],
        "collision_bounds": robot_collision_bounds(plant.model,
            SimpleNamespace(geom_xpos=state["geom_xpos"], geom_xmat=state["geom_xmat"])),
        "nonwheel_terrain_contacts": nonwheel,
        "geometry_sampling": "scratch_endpoint_kinematics_no_forward",
        "nonwheel_sampling": "synchronized_endpoint_geometric_contacts_not_dynamic_load"}


def run_case(folder, enabled, ledger, *, baseline=None):
    from scripts.d1_single_step_env import D1SingleStepEnv

    folder.mkdir(exist_ok=False)
    env = D1SingleStepEnv(obstacle_enabled=enabled)
    plant = env.plant
    geometry_manifest = compiled_geometry_manifest(plant)
    write_json(folder/"geometry_manifest.json", geometry_manifest)
    body_ids = list(range(1, plant.model.nbody))
    scratch = ScratchKinematics(plant.model, body_ids=body_ids,
        masses=plant.model.body_mass[body_ids], base_body_id=plant.base_body_id)
    observer = StreamingNativeObserver(plant, folder,
        baseline=None if baseline is None else baseline["native"])
    endpoints, trace = [], []
    arrays = {key: [] for key in ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time")}
    error, terminated, truncated = None, False, False
    completed = 0

    def snapshot(obs, tick, stream):
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            arrays[key].append(getattr(plant.data, key).copy())
        arrays["observation"].append(obs.copy()); arrays["time"].append(float(plant.data.time))
        row = endpoint_record(env, scratch, tick)
        endpoints.append(row); write_row(stream, row)

    try:
        with gzip.open(folder/"endpoints.jsonl.gz", "xt") as ep_stream, gzip.open(folder/"trace.jsonl.gz", "xt") as trace_stream:  # noqa: SIM117
            with observer:
                obs, info = env.reset(seed=77301)
                write_json(folder/"episode_metadata.json", info)
                snapshot(obs, 0, ep_stream)
                if baseline is not None:
                    for key in ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation"):
                        if not np.array_equal(arrays[key][0], baseline["arrays"][key][0]):
                            raise RuntimeError("initial paired state differs: "+key)
                ledger.allowed = (plant.model, plant.data)
                for tick in range(1200):
                    if ledger.control_attempted >= 2400:
                        raise RuntimeError("control budget exhausted")
                    raw = dict(env.command_records[-1])
                    before = env.heading_decision
                    action = np.zeros(8, dtype=np.float32)
                    ledger.control_attempted += 1
                    obs, reward, terminated, truncated, info = env.step(action)
                    completed += 1; ledger.control_completed += 1
                    control = env._controller.controller
                    row = {"tick": tick, "endpoint_tick": tick+1, "action": action,
                        "raw_command": raw, "servo_command": asdict(before.servo_command),
                        "world_command": asdict(before.decision.world_command),
                        "stop_active": control.last_damping.active,
                        "turn_active": control.last_authority.active,
                        "torque_nm": env.last_transition.requested_torque_nm,
                        "callback_count_after_prepare": env.raw_callback_count,
                        "stop_record": asdict(control.last_damping),
                        "authority_record": asdict(control.last_authority),
                        "controller_result": asdict(control.last_result),
                        "reward": reward, "terminated": bool(terminated), "truncated": bool(truncated),
                        "terminal_reason": info.get("terminal_reason"), "metrics": info.get("metrics")}
                    trace.append(row); write_row(trace_stream, row)
                    snapshot(obs, tick+1, ep_stream)
                    if terminated or truncated:
                        break
    except BaseException as exc:  # noqa: BLE001 -- preserve partial physics without retry
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        ledger.allowed = None
        receipt = {"completed_control_intervals": completed, "native_receipt": observer.receipt(),
            "clock_native_steps": round(float(plant.data.time)/.002), "error": error,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "prefix_compared_native": observer.prefix_compared_native,
            "prefix_errors": observer.prefix_errors, "reconstruction": scratch.receipt(),
            "callback_count": env.raw_callback_count, "no_retry_or_padding": True}
        write_json(folder/"receipt.json", receipt)
        np.savez_compressed(folder/"terminal_integrator.npz", qpos=plant.data.qpos,
            qvel=plant.data.qvel, ctrl=plant.data.ctrl, qacc_warmstart=plant.data.qacc_warmstart,
            time=np.array(plant.data.time))
        env.close()
        np.savez_compressed(folder/"states.npz", **{key: np.asarray(rows) for key, rows in arrays.items()})
        write_json(folder/"command_records.json", env.command_records)
    return {"obstacle_enabled": enabled, "completed_control_intervals": completed,
        "endpoints": endpoints, "trace": trace, "native": observer.entries, "arrays": arrays,
        "error": error, "terminated": bool(terminated), "truncated": bool(truncated),
        "source_identity_valid": True, "receipt": receipt, "geometry_manifest": geometry_manifest}


def load_case(folder):
    protocol = json.loads((folder.parent/"protocol.json").read_text())
    manifest = json.loads((folder.parent/"manifest.json").read_text())
    source_valid = all((ROOT/p).is_file() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest
                       for p, digest in protocol["input_sha256"].items())
    archive_valid = all((folder.parent/p).is_file() and hashlib.sha256((folder.parent/p).read_bytes()).hexdigest() == row["sha256"]
                        for p, row in manifest.items())
    receipt = json.loads((folder/"receipt.json").read_text())
    info = json.loads((folder/"episode_metadata.json").read_text())
    out = {"obstacle_enabled": info["episode_metadata"]["collision_terrain"]["obstacle_enabled"],
           **{k: receipt[k] for k in ("completed_control_intervals", "error", "terminated", "truncated")},
           "source_identity_valid": source_valid and archive_valid, "receipt": receipt}
    out["geometry_manifest"] = json.loads((folder/"geometry_manifest.json").read_text())
    for name in ("endpoints", "trace", "native"):
        with gzip.open(folder/(name+".jsonl.gz"), "rt") as stream:
            out[name] = [json.loads(line) for line in stream]
    return out


def main():
    from scripts.d1_single_step_scoring import score_single_step

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if not preflight["passed"] or not preflight["root_execution_authorized"]:
        raise ValueError("root-qualified preflight required")
    if hashlib.sha256(args.contract.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise ValueError("frozen contract identity changed")
    inputs = source_manifest()
    if inputs != preflight["input_sha256"]:
        raise ValueError("source manifest changed after preflight")
    frozen = json.loads((ROOT/"results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    if len(frozen) != 77 or any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h for p, h in frozen.items()):
        raise ValueError("frozen77 changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"protocol.json", {"contract_sha256": CONTRACT_SHA256,
        "input_sha256": inputs, "max_control": 2400, "max_native": 12000,
        "cases": ["plane_only", "single_15mm_box"], "seed": 77301,
        "nonwheel_gate": "any native or synchronized endpoint geometric terrain contact; includes margin candidates",
        "no_swept_collision_claim": True, "full_objective_complete": False})
    summaries, outputs, batch_error = {}, {}, None
    with PhysicsCallLedger() as ledger:
        try:
            for name, enabled in (("plane_only", False), ("single_15mm_box", True)):
                payload = run_case(args.output/name, enabled, ledger,
                    baseline=outputs.get("plane_only") if enabled else None)
                payload["source_identity_valid"] = inputs == source_manifest()
                score = score_single_step(payload)
                write_json(args.output/name/"score.json", score)
                summaries[name] = score; outputs[name] = payload
                print(json.dumps({"case": name, "score": score}), flush=True)
                if payload["error"] or not score["record_valid"]:
                    raise RuntimeError("record or plant failure; remaining physics stopped")
        except BaseException as error:  # noqa: BLE001 -- terminal batch receipt is mandatory
            batch_error = {"type": type(error).__name__, "message": str(error)}
        finally:
            write_json(args.output/"batch_receipt.json", {"ledger": ledger.receipt(),
                "error": batch_error, "cases": list(summaries), "scores": summaries,
                "sources_unchanged": inputs == source_manifest(), "full_objective_complete": False})
    write_json(args.output/"manifest.json", {str(p.relative_to(args.output)): {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in sorted(args.output.rglob("*")) if p.is_file()})
    return 1 if batch_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
