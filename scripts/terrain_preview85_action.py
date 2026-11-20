"""Paired, zero-training D1 support-transfer pulses starting at a loaded front wheel."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import deque
from dataclasses import fields, is_dataclass
from pathlib import Path

import mujoco
import numpy as np

from scripts import terrain_preview84_trials as baseline
from wheel_legged_control.d1.controllers import D1VMCController
from wheel_legged_control.d1.hierarchical import D1LQRVMCController

ARMS = {"negative": -1, "zero": 0, "positive": 1}
PULSE_TICKS = 16
BRANCH_TICKS = 64
MAX_PREFIX_TICKS = 400
TORQUE_EPS_NM = 1e-8
LOAD_EPS_N = 1e-8
PITCH_MIN_DEG = 0.01
WHEELS = [3, 7, 11, 15]


def transferred_forces(nominal, request_n, cap_n):
    """Transfer a signed total axle load while preserving the original four-leg sum."""
    nominal = np.asarray(nominal, dtype=float)
    if (nominal.shape != (4,) or not np.isfinite(nominal).all()
            or not math.isfinite(request_n) or not math.isfinite(cap_n)
            or cap_n <= 0 or np.any(nominal < 0) or np.any(nominal > cap_n)):
        raise ValueError("invalid nominal support or transfer bound")
    if request_n >= 0:
        available = min(np.min(cap_n - nominal[:2]), np.min(nominal[2:]))
    else:
        available = min(np.min(nominal[:2]), np.min(cap_n - nominal[2:]))
    transfer = math.copysign(min(abs(request_n), 2.0 * available), request_n)
    half = transfer / 2
    return nominal + np.asarray([half, half, -half, -half]), transfer


class SupportTransferVMC(D1VMCController):
    """Add world-vertical axle load before mapping leg forces to joint torques."""

    requested_transfer_n = 0.0

    def _support_torque(self, state, command, vertical_force_offset_n):
        torque, nominal = super()._support_torque(state, command, vertical_force_offset_n)
        self.nominal_support_n = nominal.copy()
        self.applied_transfer_n = 0.0
        if self.requested_transfer_n == 0:
            return torque, nominal
        forces, applied = transferred_forces(
            nominal, self.requested_transfer_n, self.total_mass_kg * 9.81 * 0.65,
        )
        self.applied_transfer_n = applied
        # The legacy VMC maps world -z at each foot with the estimated Jacobian.
        delta_joint = -(forces - nominal)[:, None] * state.foot_jacobian[:, 2, :]
        delta_joint[:, 3] = 0.0
        return torque + delta_joint.ravel(), forces


class SupportTransferLQR(D1LQRVMCController):
    def __init__(self, plant, linear_model=None):
        super().__init__(plant, linear_model=linear_model)
        self.low_level = SupportTransferVMC(plant)
        self.low_level.wheel_velocity_gain = 0.0


def _plain(value, native_refs):
    """Name native buffers explicitly, retain exact array bytes and RNG state in evidence."""
    ref = native_refs.get(id(value))
    if ref is not None:
        return {"native_reference": ref}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "hex": value.tobytes().hex()}
    if isinstance(value, np.random.Generator):
        return _plain(value.bit_generator.state, native_refs)
    if isinstance(value, np.generic):
        return _plain(value.item(), native_refs)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v, native_refs) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_plain(v, native_refs) for v in value]
    if is_dataclass(value):
        record = {f.name: getattr(value, f.name) for f in fields(value)}
    else:
        record = vars(value)
    return {"type": type(value).__qualname__, "fields": _plain(record, native_refs)}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _python_state(ep):
    return (vars(ep).copy(), ep.plant._last_control_interval_contact_wrench,
            ep.plant._last_actuator_traces)


def _integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


class BranchCheckpoint:
    """One live plant; three inactive native data snapshots include solver warm starts."""

    def __init__(self, ep, directory):
        plant, estimator = ep.plant, ep.source.estimator
        if plant.sampling_mode != "synchronized" or plant.actuator_channel is not None:
            raise ValueError("expected synchronized D1 with the original actuator path")
        self.live = {
            "plant": (plant.model, plant.data),
            "measurement": (plant.model, plant.measurement_data),
            "estimator": (estimator.model, estimator.data),
        }
        external = {"episode_plant": plant, "estimator_model": estimator.model,
                    "estimator_data": estimator.data}
        self.refs = {id(v): name for name, v in external.items()}
        self.shared = {id(v): v for v in external.values()}
        self.native = {}
        integration = {}
        for name, (model, data) in self.live.items():
            saved = mujoco.MjData(model)
            mujoco.mj_copyData(saved, model, data)
            self.native[name] = saved
            integration[name] = _integration_state(model, data)
        self.python = copy.deepcopy(_python_state(ep), dict(self.shared))
        py_record = _plain(_python_state(ep), self.refs)
        self.python_digest = hashlib.sha256(_bytes(py_record)).hexdigest()
        self.native_digests = {k: hashlib.sha256(v.tobytes()).hexdigest()
                               for k, v in integration.items()}
        baseline.write_json(directory / "checkpoint_python.json", py_record)
        with (directory / "checkpoint_native.npz").open("xb") as stream:
            np.savez_compressed(stream, **integration)

    def restore(self, ep):
        for name, (model, data) in self.live.items():
            mujoco.mj_copyData(data, model, self.native[name])
        attributes, contact, traces = copy.deepcopy(self.python, dict(self.shared))
        vars(ep).update(attributes)
        ep.plant._last_control_interval_contact_wrench = contact
        ep.plant._last_actuator_traces = traces
        python_sha = hashlib.sha256(_bytes(_plain(_python_state(ep), self.refs))).hexdigest()
        native_sha = {name: hashlib.sha256(_integration_state(m, d).tobytes()).hexdigest()
                      for name, (m, d) in self.live.items()}
        if python_sha != self.python_digest or native_sha != self.native_digests:
            raise ValueError("branch start differs from the saved prefix")
        return {"python_sha256": python_sha, "integration_sha256": native_sha}


def normal_loads(plant):
    """First contact-frame component is the compressive force of each loaded contact."""
    data, model = plant.measurement_data, plant.model
    loads = np.zeros(4)
    wrench = np.empty(6)
    wheel_index = {body: i for i, body in enumerate(plant.wheel_body_ids_by_leg)}
    for index, ct in enumerate(data.contact):
        g1, g2 = int(ct.geom1), int(ct.geom2)
        if ct.efc_address < 0 or plant.floor_geom_id not in (g1, g2):
            continue
        other = g2 if g1 == plant.floor_geom_id else g1
        leg = wheel_index.get(int(model.geom_bodyid[other]))
        if leg is not None:
            mujoco.mj_contactForce(model, data, index, wrench)
            loads[leg] += max(0.0, float(wrench[0]))
    return loads


def pulse_at(sign, step, weight_n):
    return sign * 0.02 * weight_n if step <= PULSE_TICKS else 0.0


def _log(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def _advance(ep, request_n):
    from scripts.terrain_preview85_common import baseline_command, physical_sample

    command = baseline_command(ep)
    low = ep.controller.low_level
    low.requested_transfer_n = request_n
    torque = ep.controller.compute(command, ep.state)
    ep.plant.step(torque)
    result = physical_sample(ep, torque)
    result["requested_transfer_n"] = request_n
    result["allocated_transfer_n"] = low.applied_transfer_n
    part = low.last_breakdown
    result["baseline_support_n"] = low.nominal_support_n.tolist()
    result["allocated_support_n"] = part.support_force_n.tolist()
    result["support_sum_error_n"] = float(np.sum(part.support_force_n - low.nominal_support_n))
    result["preclip_torque_nm"] = (
        part.leg_pd_nm + part.support_nm + part.wheel_velocity_nm + part.high_level_nm
    ).tolist()
    result["applied_torque_nm"] = ep.plant.data.ctrl[ep.plant.actuator_ids].tolist()
    result["joint_torque_limits_nm"] = ep.plant.actuator_torque_limit_nm.tolist()
    result["qpos"] = ep.plant.data.qpos.tolist()
    result["qvel"] = ep.plant.data.qvel.tolist()
    result["longitudinal_drive_n"] = ep.controller.last_longitudinal_force_n
    loads = normal_loads(ep.plant)
    result["wheel_normal_load_n"] = loads.tolist()
    result["wheel_loaded"] = (loads > 0).tolist()
    result["front_normal_load_n"], result["rear_normal_load_n"] = (
        float(np.sum(loads[:2])), float(np.sum(loads[2:]))
    )
    if not result["termination_reason"]:
        ep.state = ep.source.read()
    return result


def _contacted(row, toe):
    x = row["front_contact_x_m"]
    return x is not None and x >= toe


def _trigger(x, toe):
    if x is None or x < toe - 0.05:
        return "approach"
    return "checkpoint" if x < toe else "missed_precontact_band"


def compare_branches(branches, toe):
    if set(branches) != set(ARMS) or any(not rows for rows in branches.values()):
        raise ValueError("expected three measured branches")
    shortest = min(map(len, branches.values()))
    contact_tick = {name: next((i for i, r in enumerate(rows, 1) if _contacted(r, toe)), None)
                    for name, rows in branches.items()}
    start = max(contact_tick.values()) if None not in contact_tick.values() else None
    pitch_window = range(start - 1, shortest) if start is not None else range(0)
    rmse = {name: (math.degrees(math.sqrt(sum(rows[i]["pitch_error_rad"] ** 2
                                               for i in pitch_window) / len(pitch_window)))
                   if pitch_window else None)
            for name, rows in branches.items()}
    pitch_diff = rmse["positive"] - rmse["negative"] if pitch_window else None
    first = {k: r[0] for k, r in branches.items()}
    first_torques = {name: np.array(row["applied_torque_nm"]) for name, row in first.items()}
    joint_differences = (first_torques["positive"] - first_torques["negative"]).reshape(4, 4)
    torque_delta = float(np.max(np.abs(joint_differences[:, :3])))
    wheel_disagreement = max(float(np.max(np.abs(t[WHEELS] - first_torques["zero"][WHEELS])))
                             for t in first_torques.values())
    longitudinal_disagreement = max(abs(r["longitudinal_drive_n"] - first["zero"]["longitudinal_drive_n"])
                                    for r in first.values())
    sum_error = max(abs(row["support_sum_error_n"]) for rows in branches.values() for row in rows)
    load_samples = []
    for i in range(shortest):
        rows = {name: branch[i] for name, branch in branches.items()}
        if all(min(r["front_normal_load_n"], r["rear_normal_load_n"]) > 0 for r in rows.values()):
            axis = {name: row["front_normal_load_n"] - row["rear_normal_load_n"]
                    for name, row in rows.items()}
            load_samples.append((i + 1, axis["positive"] - axis["negative"]))
    max_load = max((abs(val) for _, val in load_samples), default=None)
    end = {name: r[-1]["termination_reason"] or "branch_horizon" for name, r in branches.items()}
    violations = {name: sum(row["torque_bound_violations"] for row in rows)
                  for name, rows in branches.items()}
    checks = {
        "all_three_reach_64_ticks": all(len(rows) == BRANCH_TICKS for rows in branches.values()),
        "first_applied_torques_separate": torque_delta > TORQUE_EPS_NM,
        "first_wheel_and_longitudinal_commands_match": wheel_disagreement <= TORQUE_EPS_NM
            and longitudinal_disagreement <= LOAD_EPS_N,
        "support_transfer_conserves_total": sum_error <= LOAD_EPS_N,
        "loaded_front_rear_response_separates": max_load is not None and max_load > LOAD_EPS_N,
        "common_entry_rmse_separates": pitch_diff is not None and abs(pitch_diff) >= PITCH_MIN_DEG,
        "no_added_fall": end["zero"] == "fallen" or "fallen" not in end.values(),
        "in_bounds": "out_of_bounds" not in end.values(),
        "no_added_torque_limit_violations": max(violations.values()) <= violations["zero"],
    }
    return {
        "branches": {a: {"steps": len(r), "termination": end[a], "first_contact_tick": contact_tick[a],
                         "common_entry_pitch_rmse_deg": rmse[a], "torque_bound_violations": violations[a],
                         "first_requested_transfer_n": r[0]["requested_transfer_n"],
                         "first_allocated_transfer_n": r[0]["allocated_transfer_n"],
                         "first_front_normal_load_n": r[0]["front_normal_load_n"],
                         "first_rear_normal_load_n": r[0]["rear_normal_load_n"]}
                     for a, r in branches.items()},
        "common_entry_start_tick": start if pitch_window else None,
        "common_entry_end_tick": shortest if pitch_window else None,
        "common_entry_ticks": len(pitch_window),
        "positive_minus_negative_pitch_rmse_deg": pitch_diff,
        "first_pos_neg_applied_torque_linf_nm": torque_delta,
        "first_wheel_torque_spread_vs_zero_nm": wheel_disagreement,
        "first_longitudinal_force_spread_vs_zero_n": longitudinal_disagreement,
        "max_per_tick_support_total_error_n": sum_error,
        "both_axles_loaded_all_arms_ticks": len(load_samples),
        "first_common_loaded_tick": load_samples[0][0] if load_samples else None,
        "first_common_loaded_pos_minus_neg_axle_difference_n": load_samples[0][1] if load_samples else None,
        "mean_common_loaded_pos_minus_neg_axle_difference_n": (
            sum(v for _, v in load_samples) / len(load_samples) if load_samples else None
        ),
        "max_abs_common_loaded_pos_minus_neg_axle_difference_n": max_load,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _case(output, terrain, seed, prefix_cap):
    from scripts.terrain_preview85_common import new_episode

    output.mkdir()
    baseline.write_json(output / "started.json", {"terrain": terrain, "seed": seed,
                                                   "status_until_metrics": "PARTIAL"})
    ep = new_episode(terrain, seed, controller_class=SupportTransferLQR)
    toe = ep.toe
    prefix_steps = 0
    reason = _trigger(baseline.front_contact_x(ep.plant), toe)
    with (output / "prefix.jsonl").open("x") as stream:
        while reason == "approach" and prefix_steps < prefix_cap:
            row = _advance(ep, 0.0)
            prefix_steps += 1
            _log(stream, row)
            reason = row["termination_reason"] or _trigger(row["front_contact_x_m"], toe)
    if reason == "approach":
        reason = "prefix_cap"
    case = {"terrain": terrain, "seed": seed, "toe_m": toe, "prefix_steps": prefix_steps,
            "prefix_outcome": reason, "terminal_branches": 0, "branch_steps": 0, "passed": False}
    if reason == "checkpoint":
        case["checkpoint_s"] = float(ep.plant.data.time)
        case["checkpoint_loaded_front_x_m"] = baseline.front_contact_x(ep.plant)
        checkpoint = BranchCheckpoint(ep, output)
        weight = ep.controller.low_level.total_mass_kg * ep.controller.low_level.gravity_mps2
        arms = {}
        for name, sign in ARMS.items():
            hashes = checkpoint.restore(ep)
            baseline.write_json(output / f"{name}_start.json", hashes)
            rows = []
            with (output / f"{name}.jsonl").open("x") as stream:
                for tick in range(1, BRANCH_TICKS + 1):
                    row = _advance(ep, pulse_at(sign, tick, weight))
                    row["branch_tick"] = tick
                    _log(stream, row)
                    rows.append(row)
                    if row["termination_reason"]:
                        break
            arms[name] = rows
        case.update(compare_branches(arms, toe))
        case["terminal_branches"] = len(arms)
        case["branch_steps"] = sum(len(rows) for rows in arms.values())
    baseline.write_json(output / "metrics.json", case)
    return case


def run_action_cases(output: Path, seeds: tuple[int, ...], *, max_prefix_steps=MAX_PREFIX_TICKS) -> dict:
    """Six development prefixes and up to eighteen physical branches in one process."""
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct development seeds are required")
    if not 1 <= max_prefix_steps <= MAX_PREFIX_TICKS:
        raise ValueError("prefix cap exceeds the registered limit")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    active = None
    try:
        for terrain in ("up", "down"):
            for seed in seeds:
                active = f"{terrain}_{seed}"
                cases.append(_case(output / active, terrain, seed, max_prefix_steps))
    except BaseException as exc:
        baseline.write_json(output / "partial.json", {"status": "PARTIAL", "active": active,
                                                       "completed_cases": cases, "error": repr(exc)})
        raise
    prefix_steps = sum(c["prefix_steps"] for c in cases)
    branch_steps = sum(c["branch_steps"] for c in cases)
    summary = {
        "status": "COMPLETE", "seeds": list(seeds), "max_prefix_steps_each": max_prefix_steps,
        "cases": cases, "prefix_steps": prefix_steps, "branch_steps": branch_steps,
        "task_steps": prefix_steps + branch_steps,
        "triggered_points": sum(c["prefix_outcome"] == "checkpoint" for c in cases),
        "terminal_branches": sum(c["terminal_branches"] for c in cases),
        "passed": all(c["passed"] for c in cases),
        "scope": "Independent, externally requested support pulses on development starts",
        "training_updates": 0,
    }
    baseline.write_json(output / "summary.json", summary)
    return summary
