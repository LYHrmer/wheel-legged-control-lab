"""
DEVELOPMENT-only closed-loop yaw ablation (action-intervention diagnostic).

classification: "closed-loop inference action intervention; not a trained policy"

No training, no re-initialization, no final/held-out scenarios.  Every branch
resets its own environment, reloads its own checkpoint against that env, and
predicts from that branch's CURRENT observation before each physics step.  The
intervention is applied AFTER model.predict() and the intervened action is the
action passed to env.step(), so it also appears in the next observation's
previous-action block.  The wheel-mean is NEVER subtracted.

All output goes into one brand-new directory (mkdir exist_ok=False, open 'x'/'xb').
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

REFERENCE_STEPS = 3200  # case dev_straight_road: 32 s at 100 Hz
CASE_NAME = "dev_straight_road"
MODES = ("full", "zero_legs", "common_wheels")
SEEDS = (48001, 48002, 48003)
VARIANTS = ("unbounded", "bounded")
CLASSIFICATION = "closed-loop inference action intervention; not a trained policy"
LR_NOTE = (
    "index-defined diagnostic only: ((a4+a6)-(a5+a7))/2 over the action "
    "index layout; true anatomical sign depends on actuator ordering"
)


# ---------------------------------------------------------------- small helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npconv(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", "replace")
    raise TypeError(f"not JSON-native: {type(obj)!r}")


def jdump(path: Path, obj) -> None:
    # exclusive create + allow_nan=False are hard requirements, so this is local.
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, allow_nan=False, default=npconv, sort_keys=False)
        fh.write("\n")


def jline(fh, obj) -> None:
    fh.write(json.dumps(obj, allow_nan=False, default=npconv) + "\n")
    fh.flush()


def manifest_for(root: Path, files, exclude: Path | None = None):
    out = []
    for p in sorted(files):
        if exclude is not None and p == exclude:
            continue
        if p.is_file():
            out.append(
                {
                    "path": os.path.relpath(p, root),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
    return out


def rms(np, vals):
    a = np.asarray(vals, dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else None


def quat_yaw(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------- metric block
def compute_metrics(np, ep, n):
    """Recompute metrics over the first n transitions (no timestep padding)."""
    if n <= 0:
        return {"steps": 0, "metrics_available": False, "reason": "no recorded transitions"}
    s = lambda k: ep[k][:n]
    yaw_seq = np.unwrap(np.asarray([ep["yaw0"]] + s("yaw_rad"), dtype=float))
    heading = yaw_seq - float(ep["yaw0"])
    y_seq = np.asarray([ep["y0"]] + s("y_m"), dtype=float)
    lat = y_seq - float(ep["y0"])
    xf, yf = float(s("x_m")[-1]), float(s("y_m")[-1])
    c, sn = math.cos(ep["yaw0"]), math.sin(ep["yaw0"])
    acts = np.asarray(s("applied_action"), dtype=float)
    terms = {}
    for row in s("reward_terms"):
        for k, v in row.items():
            terms[k] = terms.get(k, 0.0) + float(v)
    return {
        "steps": int(n),
        "metrics_available": True,
        "simulated_duration_s": float(ep["time_s"][n] - ep["time_s"][0]),
        "completion_fraction_vs_case_steps": float(n) / float(REFERENCE_STEPS),
        "reference_steps": REFERENCE_STEPS,
        "return": float(np.sum(s("reward"))),
        "reward_term_sums": terms,
        "velocity_rmse_mps": rms(np, s("velocity_error_mps")),
        "yaw_rate_rmse_rps": rms(np, s("yaw_rate_error_rps")),
        "height_rmse_m": rms(np, s("height_error_m")),
        "roll_rmse_rad": rms(np, s("roll_error_rad")),
        "pitch_rmse_rad": rms(np, s("pitch_error_rad")),
        "initial_x_m": float(ep["x0"]),
        "initial_y_m": float(ep["y0"]),
        "initial_yaw_rad": float(ep["yaw0"]),
        "final_x_m": xf,
        "final_y_m": yf,
        "final_yaw_rad": float(s("yaw_rad")[-1]),
        "signed_forward_progress_m": (xf - ep["x0"]) * c + (yf - ep["y0"]) * sn,
        "max_abs_lateral_displacement_m": float(np.max(np.abs(lat))),
        "lateral_rmse_m": rms(np, lat[1:]),
        "heading_rmse_rad": rms(np, heading[1:]),
        "final_heading_change_rad": float(heading[-1]),
        "wheel_common_mean": float(np.mean(s("wheel_common"))),
        "wheel_common_min": float(np.min(s("wheel_common"))),
        "wheel_common_max": float(np.max(s("wheel_common"))),
        "wheel_differential_rms_mean": float(np.mean(s("wheel_diff_rms"))),
        "index_defined_left_right_wheel_mean_difference_mean": float(np.mean(s("lr_diff"))),
        "index_defined_left_right_diagnostic_note": LR_NOTE,
        "mean_executed_action_8": [float(v) for v in acts.mean(axis=0)],
        "min_clearance_m": float(np.min(s("clearance_m"))),
        "max_undesired_ground_contacts": float(np.max(s("contacts"))),
    }


# ---------------------------------------------------------------- one episode
def run_episode(
    mods, np, label, dirpath, mode, seed, variant, model_path, meta_path, case, first_init
):
    D = mods
    dirpath.mkdir(parents=False, exist_ok=False)
    trace_path, states_path = dirpath / "trace.jsonl", dirpath / "states.npz"
    ep = {
        k: []
        for k in (
            "reward",
            "reward_terms",
            "velocity_error_mps",
            "yaw_rate_error_rps",
            "height_error_m",
            "roll_error_rad",
            "pitch_error_rad",
            "yaw_rad",
            "x_m",
            "y_m",
            "clearance_m",
            "contacts",
            "wheel_common",
            "wheel_diff_rms",
            "lr_diff",
            "applied_action",
            "time_s",
        )
    }
    obsl, qposl, qvell = [], [], []
    failure = None
    init_rec = None
    env = None
    terminated = truncated = False
    terminal_reason = None
    fh = open(trace_path, "x", encoding="utf-8")  # noqa: SIM115 -- Closed in finally.
    try:
        env = D["Env"](
            baseline="wheel_leg",
            action_mode="independent8",
            provider_config=D["ProvCfg"]("oracle"),
            terrain=D["TerrCfg"](**case["terrain"]),
            episode_seconds=case["episode_seconds"],
            command_source=D["case_command"](case["command"]),
        )
        obs, info = env.reset(seed=case["seed"])
        init_rec = json.loads(
            json.dumps(D["_initial_record"](env, obs, info), allow_nan=False, default=npconv)
        )
        jdump(dirpath / "initial.json", init_rec)
        if first_init is not None and init_rec != first_init:
            raise RuntimeError(
                "initial record differs from first episode; environment reset is not reproducible"
            )
        model = None
        if mode != "zero":
            model = D["load_policy"](str(model_path), str(meta_path), env)
            with open(meta_path, "r", encoding="utf-8") as mfh:
                meta_json = json.load(mfh)
            D["_check_variant"](model, meta_json, seed, variant)
        q0 = np.array(env.plant.data.qpos, dtype=float, copy=True)
        ep["yaw0"] = quat_yaw(q0[3:7])
        ep["x0"], ep["y0"] = float(q0[0]), float(q0[1])
        obsl.append(obs.copy())
        qposl.append(q0)
        qvell.append(np.array(env.plant.data.qvel, dtype=float, copy=True))
        ep["time_s"].append(float(env.plant.data.time))
        idx = 0
        while True:
            if mode == "zero":
                predicted = np.zeros(8, dtype=float)
            else:
                predicted = (
                    np.asarray(model.predict(obs, deterministic=True)[0], dtype=float)
                    .ravel()
                    .copy()
                )
            if predicted.size != 8:
                raise RuntimeError(f"expected 8-vector action, got {predicted.size}")
            action = predicted.copy()
            if mode == "zero_legs":
                action[:4] = 0.0
            elif mode == "common_wheels":
                action[4:] = float(np.mean(predicted[4:]))
            obs, reward, terminated, truncated, info = env.step(action)
            applied = np.asarray(info["applied_action"], dtype=float)
            w = applied[4:]
            common = float(np.mean(w))
            dif = float(np.sqrt(np.mean((w - common) ** 2)))
            lr = float(((w[0] + w[2]) - (w[1] + w[3])) / 2.0)
            m = info["metrics"]
            jline(
                fh,
                {
                    "transition_index": idx,
                    "predicted_action": predicted.tolist(),
                    "intervened_action": action.tolist(),
                    "intervention_mode": mode,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "applied_wheel_common_mean": common,
                    "applied_wheel_differential_rms_about_mean": dif,
                    "index_defined_left_right_wheel_mean_difference": lr,
                    "index_defined_left_right_diagnostic_note": LR_NOTE,
                    "info": info,
                },
            )
            ep["reward"].append(float(reward))
            ep["reward_terms"].append(dict(info.get("reward_terms", {})))
            for k in (
                "velocity_error_mps",
                "yaw_rate_error_rps",
                "height_error_m",
                "roll_error_rad",
                "pitch_error_rad",
                "yaw_rad",
                "x_m",
                "y_m",
            ):
                ep[k].append(float(m[k]))
            ep["clearance_m"].append(float(m["clearance_m"]))
            ep["contacts"].append(float(m["undesired_ground_contacts"]))
            ep["wheel_common"].append(common)
            ep["wheel_diff_rms"].append(dif)
            ep["lr_diff"].append(lr)
            ep["applied_action"].append(applied.tolist())
            obsl.append(obs.copy())
            qposl.append(np.array(env.plant.data.qpos, dtype=float, copy=True))
            qvell.append(np.array(env.plant.data.qvel, dtype=float, copy=True))
            ep["time_s"].append(float(env.plant.data.time))
            terminal_reason = info.get("terminal_reason")
            idx += 1
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 -- Retain diagnostic failures without retrying.
        failure = {
            "label": label,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "recorded_transitions": len(ep["reward"]),
        }
    finally:
        fh.close()
        with open(states_path, "xb") as sf:
            np.savez(
                sf,
                observation=np.asarray(obsl),
                qpos=np.asarray(qposl),
                qvel=np.asarray(qvell),
                time_s=np.asarray(ep["time_s"], dtype=float),
            )
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 -- Preserve close failures in the episode record.
                if failure is None:
                    failure = {
                        "label": label,
                        "error_type": type(exc).__name__,
                        "error": f"env.close: {exc}",
                        "traceback": traceback.format_exc(),
                        "recorded_transitions": len(ep["reward"]),
                    }
    n = len(ep["reward"])
    summary = {
        "label": label,
        "case_name": CASE_NAME,
        "split": "dev",
        "intervention_mode": mode,
        "classification": CLASSIFICATION,
        "policy_seed": seed,
        "policy_variant": variant,
        "model_sha256": sha256_file(model_path) if model_path else None,
        "metadata_sha256": sha256_file(meta_path) if meta_path else None,
        "evaluation_completed": failure is None,
        "n_steps": n,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_reason": terminal_reason,
        "state_rows": len(ep["time_s"]),
        "initial_record_matches_first_episode": None
        if init_rec is None
        else (True if first_init is None else init_rec == first_init),
        "metrics": compute_metrics(np, ep, n),
    }
    if failure is not None:
        jdump(dirpath / "failure.json", failure)
        summary["failure"] = {k: failure[k] for k in ("error_type", "error")}
    jdump(dirpath / "summary.json", summary)
    mpath = dirpath / "manifest.json"
    jdump(mpath, {"label": label, "files": manifest_for(dirpath, dirpath.iterdir(), exclude=mpath)})
    return summary, ep, init_rec, failure


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="dev-only closed-loop yaw ablation")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--evaluation-protocol", required=True)
    ap.add_argument("--frozen-source", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    models = Path(a.models).resolve()
    proto_path = Path(a.evaluation_protocol).resolve()
    frozen_path = Path(a.frozen_source).resolve()
    out = Path(a.output).resolve()
    script_sha = sha256_file(Path(__file__).resolve())

    for p in (str(repo / ".local-deps"), str(repo / "src"), str(repo)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    import numpy as np
    import torch

    torch.set_num_threads(1)
    from scripts.run_d1_wheel_common_mean_study import (  # noqa: F401
        _check_variant,
        _initial_record,
        case_command,
        read_evaluation_protocol,
        source_hashes,
        write_json,
        write_manifest,
    )
    from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy
    from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
    from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
    from wheel_legged_control.d1.state_provider import D1StateProviderConfig

    mods = {
        "Env": D1LocomotionEnv,
        "TerrCfg": D1LocomotionTerrainConfig,
        "ProvCfg": D1StateProviderConfig,
        "load_policy": load_locomotion_policy,
        "case_command": case_command,
        "_initial_record": _initial_record,
        "_check_variant": _check_variant,
    }

    with open(frozen_path, "r", encoding="utf-8") as fh:
        frozen = json.load(fh)
    fsha = dict(frozen["sha256"])

    def frozen_check():
        mismatch = [k for k, v in fsha.items() if sha256_file(repo / k) != v]
        return {
            "declared_all_77_match": bool(frozen.get("all_77_match")),
            "entries": len(fsha),
            "expected_entries": 77,
            "mismatched_paths": mismatch,
            "all_match": (not mismatch) and len(fsha) == 77 and bool(frozen.get("all_77_match")),
        }

    before = frozen_check()
    if not before["all_match"]:
        raise RuntimeError(f"frozen-source verification failed: {before}")

    protocol, cases = read_evaluation_protocol(str(proto_path), "dev")
    before_sources = source_hashes()
    protocol_sha = sha256_file(proto_path)
    frozen_sha = sha256_file(frozen_path)
    sel = [c for c in cases if c.get("name") == CASE_NAME and c.get("split") == "dev"]
    if len(sel) != 1:
        raise RuntimeError(f"expected exactly one dev {CASE_NAME} case, got {len(sel)}")
    case = sel[0]
    if int(case["seed"]) != 55101 or abs(float(case["episode_seconds"]) - 32.0) > 1e-9:
        raise RuntimeError(f"unexpected case parameters: {case}")

    plan = [("zero", "zero", None, None)]
    for seed in SEEDS:
        for variant in VARIANTS:
            for mode in MODES:
                plan.append((f"seed{seed}_{variant}_{mode}", mode, seed, variant))
    if len(plan) != 19:
        raise RuntimeError("plan must contain exactly 19 episodes")

    out.mkdir(parents=True, exist_ok=False)
    eps_dir = out / "episodes"
    eps_dir.mkdir(exist_ok=False)
    model_shas = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            mp = models / f"seed{seed}" / variant / "model.zip"
            jp = models / f"seed{seed}" / variant / "model.metadata.json"
            model_shas[f"seed{seed}/{variant}"] = {
                "model_path": str(mp),
                "model_sha256": sha256_file(mp),
                "metadata_path": str(jp),
                "metadata_sha256": sha256_file(jp),
            }

    jdump(
        out / "protocol.json",
        {
            "classification": CLASSIFICATION,
            "split": "dev",
            "development_only": True,
            "training_performed": False,
            "case": case,
            "protocol": protocol,
            "episode_labels": [p[0] for p in plan],
            "source_hashes": before_sources,
            "evaluation_protocol_file": {"path": str(proto_path), "sha256": protocol_sha},
            "models": model_shas,
            "frozen_source_file": {"path": str(frozen_path), "sha256": frozen_sha},
            "frozen_source_check_before": before,
            "script": {"path": str(Path(__file__).resolve()), "sha256": script_sha},
        },
    )
    rows, eps, failures, first_init = [], {}, [], None
    for label, mode, seed, variant in plan:
        mp = jp = None
        if mode != "zero":
            mp = models / f"seed{seed}" / variant / "model.zip"
            jp = models / f"seed{seed}" / variant / "model.metadata.json"
        summary, ep, init_rec, failure = run_episode(
            mods, np, label, eps_dir / label, mode, seed, variant, mp, jp, case, first_init
        )
        if first_init is None and init_rec is not None:
            first_init = init_rec
        rows.append(summary)
        eps[label] = ep
        if failure is not None:
            failures.append(failure)
        print(
            json.dumps(
                {
                    "event": "mode_finished",
                    "label": label,
                    "mode": mode,
                    "policy_seed": seed,
                    "policy_variant": variant,
                    "n_steps": summary["n_steps"],
                    "terminal_reason": summary["terminal_reason"],
                    "evaluation_completed": summary["evaluation_completed"],
                },
                allow_nan=False,
            ),
            flush=True,
        )

    comparisons = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            key = f"seed{seed}/{variant}"
            labels = {"zero": "zero"}
            labels.update({m: f"seed{seed}_{variant}_{m}" for m in MODES})
            counts = {m: len(eps[l]["reward"]) for m, l in labels.items()}
            if min(counts.values()) <= 0:
                comparisons[key] = {
                    "prefix_available": False,
                    "reason": "one or more branches recorded zero "
                    "transitions (exception or early abort)",
                    "recorded_steps": counts,
                }
                continue
            n = int(min(counts.values()))
            comparisons[key] = {
                "prefix_available": True,
                "common_prefix_steps": n,
                "recorded_steps": counts,
                "padding": "none",
                "note": "zero baseline is shared across comparisons (run once)",
                "prefix_metrics": {m: compute_metrics(np, eps[l], n) for m, l in labels.items()},
            }

    after = frozen_check()
    hashes_unchanged = (
        after["all_match"]
        and after["mismatched_paths"] == []
        and source_hashes() == before_sources
        and sha256_file(proto_path) == protocol_sha
        and sha256_file(frozen_path) == frozen_sha
        and sha256_file(Path(__file__).resolve()) == script_sha
        and all(
            sha256_file(Path(record[k + "_path"])) == record[k + "_sha256"]
            for record in model_shas.values()
            for k in ("model", "metadata")
        )
    )
    jdump(
        out / "integrity.json",
        {
            "frozen_source_check_after": after,
            "sources_models_protocol_and_runner_unchanged": bool(hashes_unchanged),
        },
    )
    jdump(
        out / "summary.json",
        {
            "classification": CLASSIFICATION,
            "case_name": CASE_NAME,
            "n_episodes": len(rows),
            "episodes": rows,
            "per_model_common_prefix_comparison": comparisons,
            "failures": failures,
            "all_episodes_completed": not failures,
            "initial_records_all_equal": all(
                r.get("initial_record_matches_first_episode") is not False for r in rows
            )
            and all(r.get("initial_record_matches_first_episode") is not None for r in rows),
            "frozen_source_hashes_unchanged": bool(hashes_unchanged),
        },
    )
    top = out / "manifest.json"
    files = [p for p in out.rglob("*") if p.is_file()]
    jdump(
        top,
        {
            "output_root": str(out),
            "script_sha256": script_sha,
            "files": manifest_for(out, files, exclude=top),
        },
    )
    ok = (
        (not failures)
        and hashes_unchanged
        and all(r.get("initial_record_matches_first_episode") is True for r in rows)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        sys.exit(130)
    except Exception:  # noqa: BLE001 -- Report the script failure and exit nonzero.
        traceback.print_exc()
        sys.exit(1)
