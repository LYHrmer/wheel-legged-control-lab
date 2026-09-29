"""Freeze 512 saved-behavior value windows; no model, engine, or Torch imports."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from value14_math import discounted_sum


W = Path(__file__).resolve().parent.parent
RUN = W / "rl11/training_run_01"
CLOSED = W / "rl11/partial_closed_manifest_20260929_05.json"
ALGORITHM = "d1-value14-selection-v1"
KEYS = {
    "control_index": ((), np.int64),
    "episode_index": ((), np.int32),
    "episode_tick": ((), np.int32),
    "input_observation99": ((99,), np.float32),
    "reward": ((), np.float64),
    "reward_terms9": ((9,), np.float64),
    "terminated": ((), np.bool_),
    "truncated": ((), np.bool_),
    "servo_command_vx_vy_yaw_clearance": ((4,), np.float64),
    "effective_action16": ((16,), np.float64),
}


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def verify_closed_file(run: Path, files: dict, rel: str, verified: dict) -> Path:
    relative = Path(rel)
    if relative.is_absolute() or ".." in relative.parts or rel not in files:
        raise ValueError(f"file is not in closed manifest: {rel}")
    path = run / relative
    actual = identity(path)
    if actual != files[rel]:
        raise ValueError(f"closed file identity mismatch: {rel}")
    verified[rel] = actual
    return path


def validate_timeline(control, episode, tick, done):
    """Reject missing controls, broken episode ticks, and unmarked resets."""
    n = len(control)
    if not all(len(x) == n for x in (episode, tick, done)) or not n:
        raise ValueError("timeline lengths differ or are empty")
    if not np.array_equal(control, np.arange(n, dtype=np.int64)):
        raise ValueError("control indices are not complete and consecutive")
    if int(tick[0]) != 0 or int(episode[0]) != 0 or np.any(tick < 0):
        raise ValueError("timeline does not start at episode zero, tick zero")
    same = episode[1:] == episode[:-1]
    valid_same = (tick[1:] == tick[:-1] + 1) & ~done[:-1]
    valid_reset = (episode[1:] == episode[:-1] + 1) & (tick[1:] == 0) & done[:-1]
    if not np.all(np.where(same, valid_same, valid_reset)):
        raise ValueError("episode/tick/done continuity is broken")


def eligible_starts(control, episode, tick, done, *, window=64, group_width=16384):
    """Return four chronological candidate lists; endpoint is row start+window."""
    if window <= 0 or group_width <= window:
        raise ValueError("invalid window or group width")
    validate_timeline(control, episode, tick, done)
    groups = [[] for _ in range(4)]
    for i in np.flatnonzero(tick % window == 0):
        j = int(i) + window
        if j >= len(control) or control[j] // group_width != control[i] // group_width:
            continue
        q = int(control[i] // group_width)
        if q not in range(4):
            continue
        if (episode[j] != episode[i] or tick[j] != tick[i] + window
                or np.any(done[i:j])
                or not np.array_equal(control[i:j + 1],
                                      np.arange(control[i], control[i] + window + 1))
                or not np.array_equal(tick[i:j + 1],
                                      np.arange(tick[i], tick[i] + window + 1))):
            continue
        groups[q].append(int(i))
    return groups


def select_starts(groups, *, per_group=128, algorithm=ALGORITHM):
    """Hash-rank aligned, nonoverlapping candidates, then restore time order."""
    chosen = []
    for q, candidates in enumerate(groups):
        if len(candidates) < per_group or len(set(candidates)) != len(candidates):
            raise ValueError(f"quartile {q} has insufficient or duplicate candidates")
        ranked = sorted(candidates, key=lambda i: (
            hashlib.sha256(f"{algorithm}|{q}|{i}".encode("utf-8")).digest(), i))
        selected = sorted(ranked[:per_group])
        if any(b - a < 64 for a, b in zip(selected, selected[1:])):
            raise ValueError("selected reward windows overlap")
        chosen.extend(selected)
    return np.asarray(chosen, dtype=np.int64)


def load_numeric(run: Path, files: dict, verified: dict):
    manifest_path = verify_closed_file(
        run, files, "training/training_blocks_manifest.json", verified)
    manifest = json.loads(manifest_path.read_text())
    blocks = manifest["numeric_blocks"]
    if len(blocks) != 64 or manifest["completed_controls"] != 65536:
        raise ValueError("training block manifest is not the closed 65,536-row set")
    pieces = {key: [] for key in KEYS}
    block_names = []
    next_index = 0
    for block in blocks:
        rel = "training/training_numeric_blocks/" + block["file"]
        path = verify_closed_file(run, files, rel, verified)
        if (verified[rel] != {k: block[k] for k in ("sha256", "bytes")}
                or block["first_control_index"] != next_index
                or block["last_control_index"] != next_index + block["rows"] - 1
                or block["rows"] != 1024):
            raise ValueError(f"numeric block metadata mismatch: {rel}")
        with np.load(path, allow_pickle=False) as arrays:
            for key, (tail, dtype) in KEYS.items():
                a = arrays[key]
                if a.shape != (block["rows"], *tail) or a.dtype != np.dtype(dtype):
                    raise ValueError(f"numeric shape/dtype mismatch: {rel}:{key}")
                pieces[key].append(a)
        block_names.append(rel)
        next_index += block["rows"]
    if next_index != 65536:
        raise ValueError("numeric blocks do not cover the training run")
    data = {key: np.concatenate(value) for key, value in pieces.items()}
    done = data["terminated"] | data["truncated"]
    validate_timeline(data["control_index"], data["episode_index"],
                      data["episode_tick"], done)
    if (not np.isfinite(data["reward"]).all()
            or not np.isfinite(data["reward_terms9"]).all()
            or not np.allclose(data["reward_terms9"].sum(axis=1),
                               data["reward"], atol=1e-10, rtol=0)):
        raise ValueError("environment reward and saved reward terms disagree")
    return data, block_names


def load_schedules(run: Path, files: dict, verified: dict, episode_ids):
    result = {}
    for episode_id in sorted(set(map(int, episode_ids))):
        rel = f"training/training_episode_{episode_id:06d}_schedule.json"
        path = verify_closed_file(run, files, rel, verified)
        schedule = json.loads(path.read_text())
        choice = schedule["choice"]
        if (schedule["episode_index"] != episode_id
                or choice["episode_index"] != episode_id
                or schedule["terrain"] != choice["terrain"]):
            raise ValueError(f"schedule identity mismatch: {rel}")
        result[episode_id] = {
            "terrain": schedule["terrain"],
            "phase": choice["phase"],
            "speed_mps": choice["speed_mps"],
            "schedule_file": rel,
        }
    return result


def prepare(output: Path):
    """Create a new sample directory exclusively; prior closed files stay untouched."""
    if output.exists():
        raise FileExistsError(output)
    closed_identity = identity(CLOSED)
    closed = json.loads(CLOSED.read_text())
    files = closed["files"]
    if len(files) != closed["file_count"]:
        raise ValueError("closed manifest file count mismatch")
    verified = {}
    data, blocks = load_numeric(RUN, files, verified)
    done = data["terminated"] | data["truncated"]
    groups = eligible_starts(data["control_index"], data["episode_index"],
                             data["episode_tick"], done)
    aligned = np.flatnonzero(data["episode_tick"] % 64 == 0)
    excluded = [Counter() for _ in range(4)]
    candidate_sets = [set(group) for group in groups]
    for i in aligned:
        q = int(i // 16384)
        if i in candidate_sets[q]:
            continue
        j = i + 64
        reason = ("missing_endpoint" if j >= len(done) else
                  "crosses_quartile" if j // 16384 != q else
                  "episode_or_done_boundary")
        excluded[q][reason] += 1
    candidate_indices = np.asarray([i for group in groups for i in group], dtype=np.int64)
    candidate_ends = candidate_indices + 64
    obs = data["input_observation99"]
    if (not np.isfinite(obs[candidate_indices]).all()
            or not np.isfinite(obs[candidate_ends]).all()
            or np.max(np.abs(obs[candidate_indices])) > 5
            or np.max(np.abs(obs[candidate_ends])) > 5):
        raise ValueError("candidate observation is nonfinite or exceeds 5")
    starts = select_starts(groups)
    ends = starts + 64
    if len(starts) != 512 or len(set(starts.tolist())) != 512:
        raise ValueError("selection is not 512 distinct starts")
    schedules = load_schedules(RUN, files, verified,
                               data["episode_index"][np.concatenate(groups)])
    candidate_terrain_counts = Counter(
        f"q{q}:{schedules[int(data['episode_index'][i])]['terrain']}"
        for q, group in enumerate(groups) for i in group)
    start_obs = data["input_observation99"][starts]
    endpoint_obs = data["input_observation99"][ends]
    rewards = data["reward"][starts[:, None] + np.arange(64)]
    servo = data["servo_command_vx_vy_yaw_clearance"]
    effective = np.any(data["effective_action16"] != 0, axis=1)
    window_rows = starts[:, None] + np.arange(64)
    if (not np.isfinite(start_obs).all() or not np.isfinite(endpoint_obs).all()
            or max(float(np.max(np.abs(start_obs))),
                   float(np.max(np.abs(endpoint_obs)))) > 5
            or not np.isfinite(rewards).all()
            or not np.isfinite(servo[window_rows]).all()
            or not np.isfinite(data["effective_action16"][window_rows]).all()):
        raise ValueError("selected numeric data are nonfinite or observation exceeds 5")
    quartile = data["control_index"][starts] // 16384
    terrain = np.asarray([schedules[int(e)]["terrain"]
                          for e in data["episode_index"][starts]], dtype=np.str_)
    payload = {
        "start_indices": starts,
        "endpoint_indices": ends,
        "quartile": quartile.astype(np.int64),
        "episode_index": data["episode_index"][starts].astype(np.int64),
        "episode_tick": data["episode_tick"][starts].astype(np.int64),
        "start_obs": start_obs,
        "endpoint_obs": endpoint_obs,
        "rewards": rewards,
        "discounted_rewards": discounted_sum(rewards),
        "terrain_label": terrain,
        "start_servo_command": servo[starts],
        "effective_action_nonzero": effective[starts],
        "window_effective_action_nonzero_fraction": effective[window_rows].mean(axis=1),
        "window_mean_abs_servo_vx": np.abs(servo[window_rows, 0]).mean(axis=1),
        "window_has_yaw": np.any(np.abs(servo[window_rows, 2]) > 0.05, axis=1),
    }
    details = []
    terrain_counts, servo_counts, action_counts, yaw_counts = (Counter() for _ in range(4))
    for k, (start, end) in enumerate(zip(starts, ends)):
        episode_id = int(payload["episode_index"][k])
        choice = schedules[episode_id]
        action_fraction = float(payload["window_effective_action_nonzero_fraction"][k])
        action_bin = "zero" if action_fraction == 0 else (
            "full" if action_fraction == 1 else "partial")
        mean_abs_vx = float(payload["window_mean_abs_servo_vx"][k])
        servo_bin = "near_zero" if mean_abs_vx < 0.05 else (
            "moderate" if mean_abs_vx < 0.8 else "high")
        terrain_counts[f"q{quartile[k]}:{choice['terrain']}"] += 1
        servo_counts[f"q{quartile[k]}:{servo_bin}"] += 1
        action_counts[f"q{quartile[k]}:{action_bin}"] += 1
        yaw_counts[f"q{quartile[k]}:{bool(payload['window_has_yaw'][k])}"] += 1
        details.append({
            "start_index": int(start), "endpoint_index": int(end),
            "quartile": int(quartile[k]), "episode_index": episode_id,
            "episode_tick": int(payload["episode_tick"][k]),
            "start_block": blocks[start // 1024], "start_block_offset": int(start % 1024),
            "endpoint_block": blocks[end // 1024], "endpoint_block_offset": int(end % 1024),
            "schedule_file": choice["schedule_file"],
            "terrain": choice["terrain"], "phase": choice["phase"],
            "scheduled_speed_mps": choice["speed_mps"],
            "window_mean_abs_servo_vx": mean_abs_vx, "servo_speed_bin": servo_bin,
            "servo_yaw_active": bool(payload["window_has_yaw"][k]),
            "window_effective_action_nonzero_fraction": action_fraction,
            "effective_action_bin": action_bin,
            "reward64_sha256": hashlib.sha256(rewards[k].tobytes()).hexdigest(),
            "start_obs_sha256": hashlib.sha256(start_obs[k].tobytes()).hexdigest(),
            "endpoint_obs_sha256": hashlib.sha256(endpoint_obs[k].tobytes()).hexdigest(),
        })
    output.mkdir(parents=True, exist_ok=False)
    sample_path = output / "samples.npz"
    with sample_path.open("xb") as stream:
        np.savez(stream, **payload)
    selection = {
        "schema": "d1-value14-saved-behavior-sample-v1",
        "algorithm": ALGORITHM,
        "rank_key": "SHA256(UTF8('d1-value14-selection-v1|q|global_control_index'))",
        "quartile_width_controls": 16384,
        "window_rewards": 64,
        "endpoint": "input_observation99 at row start+64 in same episode",
        "reward_semantics": "environment_returned_prebootstrap_reward_f64; not historical PPO GAE targets",
        "selection_constraint": "episode_tick%64==0; no done in 64 rewards; endpoint in same quartile and episode",
        "batch_boundaries": [[q * 128, (q + 1) * 128] for q in range(4)],
        "counts": {"aligned_starts_by_quartile": [int(np.count_nonzero(aligned // 16384 == q))
                                                 for q in range(4)],
                   "candidates_by_quartile": [len(g) for g in groups],
                   "candidate_terrain_by_quartile": dict(sorted(candidate_terrain_counts.items())),
                   "excluded_by_quartile": [dict(sorted(x.items())) for x in excluded],
                   "selected_by_quartile": [128] * 4,
                   "selected_terrain_by_quartile": dict(sorted(terrain_counts.items())),
                   "selected_servo_bin_by_quartile": dict(sorted(servo_counts.items())),
                   "selected_effective_action_bin_by_quartile": dict(sorted(action_counts.items())),
                   "selected_yaw_by_quartile": dict(sorted(yaw_counts.items()))},
        "selected_start_indices": starts.tolist(),
        "samples_file": "samples.npz",
        "samples_identity": identity(sample_path),
        "closed_manifest_identity": closed_identity,
        "plan_identity": identity(Path(__file__).with_name("astra_plan_14.md")),
        "preparation_source_identity": identity(Path(__file__)),
        "math_source_identity": identity(Path(__file__).with_name("value14_math.py")),
        "pure_test_source_identity": identity(Path(__file__).with_name("test_value14_pure.py")),
        "verified_input_files": verified,
        "sample_details": details,
        "model_loads": 0, "policy_forwards": 0, "value_forwards": 0,
        "backwards": 0, "optimizer_steps": 0, "physics_controls": 0,
    }
    with (output / "selection.json").open("x") as stream:
        json.dump(selection, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({"selected": len(result["selected_start_indices"]),
                      "candidates": result["counts"]["candidates_by_quartile"],
                      "physics_controls": 0, "model_calls": 0}))
