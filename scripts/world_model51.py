"""Offline torque-conditioned dynamics pilot; never imports a simulator or policy.

All rollouts condition on recorded future torque requests. They are not policy
rollouts, and no closed-loop performance qualification follows from this pilot.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

TERRAINS = ("flat", "bumps", "rough", "ramp")
STATE_DIM = 45  # qpos[23], qvel[22]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def state_vector(state):
    qpos = np.asarray(state["qpos"], dtype=np.float64)
    qvel = np.asarray(state["qvel"], dtype=np.float64)
    require(qpos.shape == (23,) and qvel.shape == (22,), "state shape differs")
    result = np.r_[qpos, qvel]
    require(np.isfinite(result).all(), "nonfinite state")
    require(abs(np.linalg.norm(qpos[3:7]) - 1) < 1e-6, "invalid source quaternion")
    return result


def read_transition(row):
    """Use explicit pre/post states; post metrics cannot leak into the input."""
    before, after = row["pre_state"], row["post_state"]
    require(abs(after["time"] - before["time"] - 0.01) < 1e-9, "control dt differs")
    request = np.asarray(row["info"]["controller_record"]["calculation"]["safe_torque_nm"])
    require(request.shape == (16,) and np.isfinite(request).all(), "invalid torque request")
    traces = row["native_actuator_traces"]
    require(len(traces) == 5, "partial native interval")
    # This pilot deliberately accepts only the observed passthrough regime.
    # A delayed/lagged/gain-limited row fails admission instead of using future
    # applied torque or hidden actuator state as if it were known at tick start.
    for trace in traces:
        for field in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm"):
            require(
                np.array_equal(np.asarray(trace[field]), request),
                "non-passthrough actuator: separate protocol required",
            )
    return state_vector(before), request, state_vector(after)


def validate_episode(states, next_states, ticks, done, length=1000):
    require(len(states) == length, "incomplete episode")
    require(np.array_equal(ticks, np.arange(length)), "missing/reordered episode ticks")
    require(np.array_equal(states[1:], next_states[:-1]), "state discontinuity or reset")
    require(not np.any(done[:-1]) and bool(done[-1]), "terminal inside episode or missing close")


def extract(root, protocol, output):
    identities = protocol["inputs"]
    for rel, identity in identities.items():
        path = root / rel
        require(path.stat().st_size == identity["bytes"], f"input size changed: {rel}")
        require(sha256(path) == identity["sha256"], f"input hash changed: {rel}")
    source = root / protocol["source_run"]
    report = json.loads((source / "report.json").read_text())
    readback = json.loads((source / "train_ffyaw_s21_readback.json").read_text())
    require(report["status"] == readback["status"] == "COMPLETE", "source is partial")
    for name in ("archive_valid", "training_valid", "learning_valid", "coverage_valid"):
        require(readback[name] is True, f"source {name} failed")
    folder = source / "train_ffyaw_s21/training"
    receipt = json.loads((folder / "training_segment_receipt.json").read_text())
    require(receipt["record_valid"] and not receipt["partial_native_interval"], "partial record")
    require(receipt["failure"] is None and receipt["archive_failure"] is None, "source failure")
    events = [
        json.loads(line) for line in (folder / "curriculum_events.ndjson").read_text().splitlines()
    ]
    selected = protocol["episodes"]
    closed = {e["episode_index"]: e for e in events if e["event"] == "episode_closed"}
    for episode in selected:
        event = closed[episode]
        require(
            event["completed_controls"] == 1000
            and event["ended_by"] == "time_limit"
            and event["closed"]
            and not event["terminated"]
            and event["truncated"],
            "episode not complete at the preregistered horizon",
        )
    states, requests, targets, episodes, ticks, terrains, done = [], [], [], [], [], [], []
    files = []
    for index in range(40, 160):
        path = folder / f"full_control_records_{index:04d}.jsonl.gz"
        expected = readback["artifacts"]["training/" + path.name]
        require(
            path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
            f"trace identity differs: {path.name}",
        )
        count = 0
        with path.open("rb") as raw:
            with gzip.open(raw, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    episode, tick = row["episode_index"], row["episode_tick"]
                    require(
                        episode in selected and row["source_episode_index"] == episode,
                        "unexpected episode",
                    )
                    require(row["control_index"] == episode * 1000 + tick, "global tick differs")
                    require(row["terrain"] == closed[episode]["terrain"], "terrain differs")
                    require(not row["terminated"], "failed episode is not admitted")
                    require(bool(row["truncated"]) == (tick == 999), "unexpected truncation")
                    s, u, target = read_transition(row)
                    states.append(s)
                    requests.append(u)
                    targets.append(target)
                    episodes.append(episode)
                    ticks.append(tick)
                    terrains.append(TERRAINS.index(row["terrain"]))
                    done.append(row["truncated"])
                    count += 1
            os.posix_fadvise(raw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        require(count == 200, "trace row count differs")
        files.append(dict(path=str(path.relative_to(root)), **expected))
    data = {
        "s": np.asarray(states),
        "u": np.asarray(requests),
        "target": np.asarray(targets),
        "episode": np.asarray(episodes),
        "tick": np.asarray(ticks),
        "terrain": np.asarray(terrains),
    }
    require(len(states) == 24000, "transition count differs")
    signatures = set()
    for episode in selected:
        mask = data["episode"] == episode
        validate_episode(
            data["s"][mask], data["target"][mask], data["tick"][mask], np.asarray(done)[mask]
        )
        signature = hashlib.sha256(
            data["s"][mask].tobytes() + data["u"][mask].tobytes()
        ).hexdigest()
        require(signature not in signatures, "duplicate complete state-action episode")
        signatures.add(signature)
    write_npz(output / "transitions.npz", **data)
    write_json(
        output / "data_receipt.json",
        {
            "status": "COMPLETE",
            "transitions": 24000,
            "episodes": [closed[e] for e in selected],
            "source_policy_qualification": readback["qualification_granted"],
            "source_execution_complete": True,
            "actuator_passthrough_verified": True,
            "artifacts": files,
            "dataset_sha256": sha256(output / "transitions.npz"),
            "source_policy_training_seed": 201021,
            "exclusions": "episodes 0..7 warmup and incomplete episode 32; all experiment 48",
            "distribution": "chronological held-out trajectories of the same eight course templates",
        },
    )
    return data


def features(states, requests, terrain):
    return np.c_[states, requests, np.eye(4)[terrain]]


def fit_baseline(data, mask, ridge):
    x = features(data["s"][mask], data["u"][mask], data["terrain"][mask])
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
    z = np.c_[(x - mean) / scale, np.ones(len(x))]
    y = (data["target"][mask] - data["s"][mask]) / scale[:STATE_DIM]
    penalty = ridge * np.eye(z.shape[1])
    penalty[-1, -1] = 0
    coef = np.linalg.solve(z.T @ z / len(z) + penalty, z.T @ y / len(z))
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "coef": coef.astype(np.float32),
    }


def normalized_delta(model, x):
    z = (np.asarray(x, dtype=np.float32) - model["mean"]) / model["scale"]
    delta = z @ model["coef"][:-1] + model["coef"][-1]
    if "w0" in model:
        hidden = np.tanh(z @ model["w0"].T + model["b0"])
        hidden = np.tanh(hidden @ model["w1"].T + model["b1"])
        delta = delta + hidden @ model["w2"].T + model["b2"]
    return delta


def advance(model, states, requests, terrain):
    result = (
        states
        + normalized_delta(model, features(states, requests, terrain)) * model["scale"][:STATE_DIM]
    )
    norms = np.linalg.norm(result[:, 3:7], axis=1)
    require(np.isfinite(result).all() and np.all(norms > 1e-9), "nonfinite/divergent prediction")
    result[:, 3:7] /= norms[:, None]
    return result


def rollout(model, initial, requests, terrain):
    """No future physical state is an argument: only the initial truth and inputs."""
    state = initial.copy()
    for step in range(requests.shape[1]):
        state = advance(model, state, requests[:, step], terrain)
    return state


def evaluate(model, data, selected, horizon):
    rows = []
    for episode in selected:
        mask = data["episode"] == episode
        s, u, target, terrain = (data[k][mask] for k in ("s", "u", "target", "terrain"))
        starts = np.arange(0, len(s) - horizon + 1, horizon)
        indices = starts[:, None] + np.arange(horizon)
        predicted = rollout(model, s[starts], u[indices], terrain[starts])
        truth = target[starts + horizon - 1]
        error = predicted - truth
        norm = error / model["scale"][:STATE_DIM]
        orientation = 2 * np.arccos(
            np.clip(np.abs(np.sum(predicted[:, 3:7] * truth[:, 3:7], axis=1)), 0, 1)
        )
        rows.append(
            {
                "episode": int(episode),
                "terrain": TERRAINS[int(terrain[0])],
                "windows": len(starts),
                "velocity_nrmse": float(np.sqrt(np.mean(norm[:, 23:] ** 2))),
                "base_linear_rmse_mps": float(np.sqrt(np.mean(error[:, 23:26] ** 2))),
                "base_angular_rmse_rps": float(np.sqrt(np.mean(error[:, 26:29] ** 2))),
                "joint_velocity_rmse_rps": float(np.sqrt(np.mean(error[:, 29:] ** 2))),
                "position_rmse_m": float(np.sqrt(np.mean(error[:, :3] ** 2))),
                "orientation_rmse_rad": float(np.sqrt(np.mean(orientation**2))),
            }
        )
    return {
        "horizon": horizon,
        "physical_seconds": 0.01 * horizon,
        "episodes": rows,
        "velocity_nrmse": float(np.mean([r["velocity_nrmse"] for r in rows])),
        "by_terrain": {
            t: float(np.mean([r["velocity_nrmse"] for r in rows if r["terrain"] == t]))
            for t in TERRAINS
        },
    }


def gate(baseline, candidate, thresholds):
    reasons = []
    ratios = {}
    for scores in (baseline, candidate):
        values = [scores[h]["velocity_nrmse"] for h in ("1", "8")]
        values.extend(scores["8"]["by_terrain"].values())
        if not all(np.isfinite(v) and v >= 0 for v in values):
            return {
                "passed": False,
                "ratios": {},
                "reasons": ["nonfinite or negative error metric"],
            }
    for horizon in ("1", "8"):
        base, trial = baseline[horizon]["velocity_nrmse"], candidate[horizon]["velocity_nrmse"]
        if base <= thresholds["denominator_floor"]:
            reasons.append(f"h{horizon}: baseline too small for improvement claim")
            ratios[horizon] = None
        else:
            ratios[horizon] = trial / base
            if ratios[horizon] > thresholds["h" + horizon]:
                reasons.append(f"h{horizon}: error ratio exceeds preregistered gate")
    for terrain in TERRAINS:
        base = baseline["8"]["by_terrain"][terrain]
        trial = candidate["8"]["by_terrain"][terrain]
        if base <= thresholds["denominator_floor"]:
            if trial > thresholds["denominator_floor"]:
                reasons.append(f"{terrain}: near-zero baseline degraded")
        elif trial / base > thresholds["terrain_h8"]:
            reasons.append(f"{terrain}: h8 degradation exceeds preregistered gate")
    return {"passed": not reasons, "ratios": ratios, "reasons": reasons}


def train_residual(baseline, data, mask, seed, config, output, global_cpu_deadline):
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    network = torch.nn.Sequential(
        torch.nn.Linear(65, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 45),
    )
    torch.nn.init.zeros_(network[-1].weight)
    torch.nn.init.zeros_(network[-1].bias)
    x = features(data["s"][mask], data["u"][mask], data["terrain"][mask]).astype(np.float32)
    z = (x - baseline["mean"]) / baseline["scale"]
    y = ((data["target"][mask] - data["s"][mask]) / baseline["scale"][:STATE_DIM]).astype(
        np.float32
    )
    residual = y - normalized_delta(baseline, x)
    xt, yt = torch.from_numpy(z), torch.from_numpy(residual)
    sampler = torch.Generator().manual_seed(config["batch_seed"])
    optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
    started = time.process_time()
    curve = []
    with (output / "updates.jsonl").open("x", buffering=1) as stream:
        for update in range(config["updates"]):
            require(time.process_time() < global_cpu_deadline, "total CPU budget exhausted")
            require(
                time.process_time() - started <= config["cpu_seconds_per_seed"],
                "seed CPU budget exhausted",
            )
            indices = torch.randint(len(xt), (config["batch_size"],), generator=sampler)
            loss = torch.mean((network(xt[indices]) - yt[indices]) ** 2)
            require(bool(torch.isfinite(loss)), "nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            if update == 0 or (update + 1) % 100 == 0:
                row = {
                    "update": update + 1,
                    "mse": float(loss.detach()),
                    "cpu_s": time.process_time() - started,
                }
                curve.append(row)
                stream.write(json.dumps(row) + "\n")
    model = dict(baseline)
    for i, layer in enumerate((network[0], network[2], network[4])):
        model[f"w{i}"] = layer.weight.detach().numpy().copy()
        model[f"b{i}"] = layer.bias.detach().numpy().copy()
    with torch.no_grad():
        torch_delta = network(xt[:16]).numpy() + normalized_delta(baseline, x[:16])
    parity = float(np.max(np.abs(torch_delta - normalized_delta(model, x[:16]))))
    require(parity <= 1e-5, "NumPy export differs from Torch")
    write_npz(output / "model.npz", **model)
    return model, {
        "updates": config["updates"],
        "cpu_seconds": time.process_time() - started,
        "export_max_abs_difference": parity,
        "curve": curve,
    }
