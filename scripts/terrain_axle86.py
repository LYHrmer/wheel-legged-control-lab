"""Same-side vertical-load transfer and a standard-library audit of the 85 first ticks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import struct
from functools import cache
from pathlib import Path
from zipfile import ZipFile

ARCHIVE_SHA256 = "68925eb753a4c863d842dff87db3c9995614faf0907341763b8b8983c658ae97"
SEEDS = (8561, 8573, 8587)
SIGNS = {"negative": -1, "zero": 0, "positive": 1}
EPS_NM = 1e-8


def side_transfers(nominal, request_n, cap_n):
    """FL/FR/RL/RR. Preserve the sum on each side, minimize asymmetry at feasible total."""
    f = tuple(map(float, nominal))
    if (len(f) != 4 or not all(math.isfinite(v) for v in (*f, request_n, cap_n))
            or cap_n <= 0 or any(v < 0 or v > cap_n for v in f)):
        raise ValueError("invalid transfer request or leg support bounds")
    if request_n == 0:
        return list(f), 0.0
    if request_n > 0:
        left = min(cap_n - f[0], f[2])
        right = min(cap_n - f[1], f[3])
    else:
        left = min(f[0], cap_n - f[2])
        right = min(f[1], cap_n - f[3])
    total = min(abs(request_n), left + right)
    amount_left = min(left, max(total - right, total / 2))
    dl = math.copysign(amount_left, request_n)
    dr = math.copysign(total - amount_left, request_n)
    # The clamp only removes floating-point drift at an active endpoint.
    targets = [max(0., min(cap_n, v + d)) for v, d in zip(f, (dl, dr, -dl, -dr), strict=True)]
    return targets, math.copysign(total, request_n)


@cache
def _controller_types():
    """Keep the offline entry point importable on machines without MuJoCo or NumPy."""
    import numpy as np

    from wheel_legged_control.d1.controllers import D1VMCController
    from wheel_legged_control.d1.hierarchical import D1LQRVMCController

    class SideTransferVMC(D1VMCController):
        requested_transfer_n = 0.0

        def _support_torque(self, state, command, vertical_force_offset_n):
            torque, nominal = super()._support_torque(state, command, vertical_force_offset_n)
            self.nominal_support_n = nominal.copy()
            self.applied_transfer_n = 0.0
            if self.requested_transfer_n == 0:
                return torque, nominal
            forces, self.applied_transfer_n = side_transfers(
                nominal, self.requested_transfer_n, self.total_mass_kg * 9.81 * 0.65,
            )
            allocated = np.asarray(forces, dtype=float)
            delta_joint = -(allocated - nominal)[:, None] * state.foot_jacobian[:, 2, :]
            delta_joint[:, 3] = 0.0
            return torque + delta_joint.ravel(), allocated

    class SideTransferLQR(D1LQRVMCController):
        def __init__(self, plant, linear_model=None):
            super().__init__(plant, linear_model=linear_model)
            self.low_level = SideTransferVMC(plant)
            self.low_level.wheel_velocity_gain = 0.0

    # The classes also have stable qualified names in checkpoints and Python pickles.
    SideTransferVMC.__qualname__ = "SideTransferVMC"
    SideTransferLQR.__qualname__ = "SideTransferLQR"
    return {"SideTransferVMC": SideTransferVMC, "SideTransferLQR": SideTransferLQR}


def __getattr__(name):
    if name in {"SideTransferVMC", "SideTransferLQR"}:
        return _controller_types()[name]
    raise AttributeError(name)


def _array(field, shape, dtype="<f8"):
    if field["shape"] != list(shape) or field["dtype"] != dtype:
        raise ValueError("unexpected checkpoint array layout")
    fmt = "d" if dtype == "<f8" else "?"
    return struct.unpack(f"<{math.prod(shape)}{fmt}", bytes.fromhex(field["hex"]))


def _legacy_support(f, request, cap):
    """The published 85 rule requires each of the four legs to accept the same magnitude."""
    available = (min(cap - f[0], cap - f[1], f[2], f[3]) if request >= 0
                 else min(f[0], f[1], cap - f[2], cap - f[3]))
    total = math.copysign(min(abs(request), 2 * available), request)
    half = total / 2
    return [x + d for x, d in zip(f, (half, half, -half, -half), strict=True)], total


def _joint_controls(raw, limits, jacobian, delta):
    result = []
    for i, (tau, limit) in enumerate(zip(raw, limits, strict=True)):
        leg, joint = divmod(i, 4)
        increment = -jacobian[leg * 12 + 8 + joint] * delta[leg] if joint != 3 else 0.0
        result.append(min(limit, max(-limit, tau + increment)))
    return result


def _difference(a, b, *, wheels=False):
    return max(abs(x - y) for i, (x, y) in enumerate(zip(a, b, strict=True))
               if (i % 4 == 3) == wheels)


def _direction_summary(torques):
    negative = _difference(torques["negative"], torques["zero"])
    positive = _difference(torques["positive"], torques["zero"])
    names = {(False, False): "neither", (False, True): "positive_only",
             (True, False): "negative_only", (True, True): "both"}
    return {
        "negative_vs_zero_leg_linf_nm": negative, "positive_vs_zero_leg_linf_nm": positive,
        "positive_vs_negative_leg_linf_nm": _difference(torques["positive"], torques["negative"]),
        "direction_class": names[negative > EPS_NM, positive > EPS_NM],
    }


def _native_hashes(blob):
    """Validate the raw float64 integration vectors, including their NPY headers."""
    output = {}
    with ZipFile(io.BytesIO(blob)) as inner:
        for label in ("plant", "measurement", "estimator"):
            data = inner.read(f"{label}.npy")
            if not data.startswith(b"\x93NUMPY") or data[6] not in (1, 2, 3):
                raise ValueError("unsupported NPY file")
            width = 2 if data[6] == 1 else 4
            end = 8 + width + int.from_bytes(data[8:8 + width], "little")
            header = ast.literal_eval(data[8 + width:end].decode("latin1"))
            raw = data[end:]
            if (header["descr"] != "<f8" or header["fortran_order"]
                    or len(header["shape"]) != 1 or len(raw) != 8 * header["shape"][0]):
                raise ValueError("invalid native integration vector")
            output[label] = hashlib.sha256(raw).hexdigest()
    return output


def _case(archive, terrain, seed):
    prefix = f"trials/action/{terrain}_{seed}/"
    cp = json.loads(archive.read(prefix + "checkpoint_python.json"))
    native = _native_hashes(archive.read(prefix + "checkpoint_native.npz"))
    py_digest = hashlib.sha256(json.dumps(cp, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    signature = {"python_sha256": py_digest, "integration_sha256": native}
    records = {}
    for arm in SIGNS:
        if json.loads(archive.read(prefix + arm + "_start.json")) != signature:
            raise ValueError("recorded branches do not share the saved checkpoint")
        with archive.open(prefix + arm + ".jsonl") as src:
            records[arm] = json.loads(src.readline())
        if records[arm]["branch_tick"] != 1:
            raise ValueError("first control tick missing")
    state = cp[0]["state"]["fields"]
    low = cp[0]["controller"]["fields"]["low_level"]["fields"]
    mass, g = low["total_mass_kg"], low["gravity_mps2"]
    cap_n = mass * 9.81 * 0.65
    jac = _array(state["foot_jacobian"], (4, 3, 4))
    pos = _array(state["foot_position"], (4, 3))
    rotation = _array(state["base_rotation"], (3, 3))
    yaw = math.atan2(rotation[3], rotation[0])
    c, s = math.cos(yaw), math.sin(yaw)
    lateral = [c * pos[3*i+1] - s * pos[3*i] for i in range(4)]
    forward = [c * pos[3*i] + s * pos[3*i+1] for i in range(4)]
    z = records["zero"]
    nominal, raw, limits = z["baseline_support_n"], z["preclip_torque_nm"], z["joint_torque_limits_nm"]
    observed = {arm: r["applied_torque_nm"] for arm, r in records.items()}
    errors, candidate_torques, arms = [], {}, {}
    for arm, sign in SIGNS.items():
        record = records[arm]
        req = sign * 0.02 * mass * g
        if (record["baseline_support_n"] != nominal or record["joint_torque_limits_nm"] != limits
                or not math.isclose(req, record["requested_transfer_n"], rel_tol=1e-14, abs_tol=1e-14)):
            raise ValueError("recorded first requests or baseline states disagree")
        was, realized = _legacy_support(nominal, req, cap_n)
        if (not math.isclose(realized, record["allocated_transfer_n"], abs_tol=1e-12)
                or any(abs(x - y) > 1e-12 for x, y in zip(was, record["allocated_support_n"], strict=True))):
            raise ValueError("the archived nominal allocator could not be reproduced")
        reconstructed = _joint_controls(raw, limits, jac, [v - n for v, n in zip(was, nominal, strict=True)])
        errors.append(max(abs(x - y) for x, y in zip(reconstructed, observed[arm], strict=True)))
        force, allocated = side_transfers(nominal, req, cap_n)
        df = [v - n for v, n in zip(force, nominal, strict=True)]
        applied = _joint_controls(raw, limits, jac, df)
        candidate_torques[arm] = applied
        arms[arm] = {
            "requested_transfer_n": req, "allocated_transfer_n": allocated,
            "target_support_n": force, "delta_support_n": df, "total_delta_n": sum(df),
            "max_side_sum_error_n": max(abs(df[0] + df[2]), abs(df[1] + df[3])),
            "ideal_heading_roll_moment_nm": sum(f * y for f, y in zip(df, lateral, strict=True)),
            "ideal_heading_pitch_moment_nm": -sum(f * x for f, x in zip(df, forward, strict=True)),
            "predicted_clipped_torque_nm": applied,
            "wheel_linf_vs_zero_nm": _difference(applied, observed["zero"], wheels=True),
        }
    if max(errors) > 1e-12:
        raise ValueError("the archived clipped joint torques could not be reproduced")
    return {
        "terrain": terrain, "seed": seed, "start": signature, "nominal_support_n": nominal,
        "per_leg_support_cap_n": cap_n,
        "estimator_loaded_wheels": list(_array(state["wheel_contact"], (4,), "|b1")),
        "heading_front_minus_rear_lateral_m": [lateral[0] - lateral[2], lateral[1] - lateral[3]],
        "old_reconstruction_max_error_nm": max(errors),
        "old": {**_direction_summary(observed), "observed_clipped_torques_nm": observed},
        "candidate": {**_direction_summary(candidate_torques), "arms": arms},
    }


def audit_first_ticks(archive: Path) -> dict:
    """Read exactly six archived first ticks. No controller or numerical package is imported."""
    archive = Path(archive)
    with archive.open("rb") as src:
        digest = hashlib.sha256()
        for chunk in iter(lambda: src.read(1 << 20), b""):
            digest.update(chunk)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("input does not match the published round-85 archive")
        with ZipFile(src) as z:
            cases = [_case(z, terrain, seed) for terrain in ("up", "down") for seed in SEEDS]
    result = {
        "archive_sha256": digest.hexdigest(), "case_count": len(cases),
        "distinct_checkpoints": len({json.dumps(c["start"], sort_keys=True) for c in cases}),
        "old_reconstruction_max_error_nm": max(c["old_reconstruction_max_error_nm"] for c in cases),
        "torque_tolerance_nm": EPS_NM, "cases": cases, "new_native_steps": 0, "new_training_updates": 0,
    }
    for method in ("old", "candidate"):
        arms = [case[method] for case in cases]
        classes = [a["direction_class"] for a in arms]
        result[method] = {
            "positive_vs_negative_count": sum(a["positive_vs_negative_leg_linf_nm"] > EPS_NM for a in arms),
            "bidirectional_count": classes.count("both"), "positive_only_count": classes.count("positive_only"),
            "negative_only_count": classes.count("negative_only"), "neither_count": classes.count("neither"),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_first_ticks(args.archive)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
