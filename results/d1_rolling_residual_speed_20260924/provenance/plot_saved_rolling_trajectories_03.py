"""Plot eight saved cases with separated title, legend, and subplot bands.

No MuJoCo, model, checkpoint or PPO import. Plots show observations; they do
not infer that the learned residual caused any difference from paired zero.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPEEDS = (.2, .25)
TERRAINS = ("plane", "box")
ACTORS = ("zero", "final_policy")
START_TICK = 175
STOP_TICK = 975
DT = .01
OLD_CASES = ("speed200_plane_zero", "speed200_plane_final_policy", "speed200_box_zero")
NEW_CASES = ("speed200_box_final_policy", "speed250_plane_zero",
             "speed250_plane_final_policy", "speed250_box_zero",
             "speed250_box_final_policy")


def _digest(path: Path) -> dict[str, Any]:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_.update(block)
    return {"sha256": hash_.hexdigest(), "bytes": path.stat().st_size}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"not a JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object saved row in {path}")
    return rows


def _case_name(speed: float, terrain: str, actor: str) -> str:
    return f"speed{round(1000 * speed):03d}_{terrain}_{actor}"


def _read_case(original: Path, continuation: Path, row: dict[str, Any]) -> dict[str, Any]:
    name = row["case"]
    source = row["source_process"]
    if source == "interrupted_run_02_complete_record" and name in OLD_CASES:
        run = original
    elif source == "new_run_03_continuation" and name in NEW_CASES:
        run = continuation
    else:
        raise ValueError(f"invalid completed-case source mapping: {name}, {source}")
    case = run / "evaluation" / name
    if not case.is_dir() or _json(case / "score.json") != row["score"]:
        raise ValueError(f"saved case or score is absent/inconsistent: {name}")
    receipt = _json(case / "receipt.json")
    count = receipt["completed_control_intervals"]
    endpoints = _rows(case / "endpoints.jsonl.gz")
    trace = _rows(case / "trace.jsonl.gz")
    if (not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 1200
            or len(endpoints) != count + 1 or len(trace) != count):
        raise ValueError(f"saved endpoint/action length mismatch in {case.name}")
    for tick, endpoint in enumerate(endpoints):
        if endpoint["tick"] != tick:
            raise ValueError(f"endpoint tick mismatch in {case.name}")
    for tick, transition in enumerate(trace):
        if transition["tick"] != tick:
            raise ValueError(f"trace tick mismatch in {case.name}")
        action = np.asarray(transition["action"], dtype=float)
        applied = np.asarray(transition["applied_action"], dtype=float)
        if (action.shape != (8,) or not np.isfinite(action).all()
                or not np.array_equal(action, applied)
                or not np.array_equal(action[:4], np.full(4, action[0]))
                or np.any(action[4:] != 0.0)):
            raise ValueError(f"saved shared-leg/wheel-zero action mismatch in {case.name}")
    if (not row["score"]["record_valid"]
            or row["score"]["metrics"]["completed_control_intervals"] != count):
        raise ValueError(f"saved case score or completion differs: {name}")
    source_files = {
        filename: _digest(case / filename)
        for filename in ("endpoints.jsonl.gz", "trace.jsonl.gz", "score.json")
    }
    return {"name": case.name, "count": count, "endpoints": endpoints,
            "trace": trace, "receipt": receipt, "source_process": source,
            "source_dir": str(case.resolve()), "source_files": source_files}


def _axes() -> tuple[Any, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4), sharex=True)
    fig.subplots_adjust(left=.09, right=.95, bottom=.11, top=.78, hspace=.34, wspace=.2)
    return fig, axes


def _decorate(ax: Any, *, speed: float, terrain: str) -> None:
    ax.axvspan(START_TICK * DT, STOP_TICK * DT, color="#eee7d9", alpha=.45, zorder=0)
    ax.grid(alpha=.25, linewidth=.6)
    ax.set_title(f"{terrain}; command {speed:.2f} m/s")


def _finish(fig: Any, axes: Any, path: Path, *, title: str, ylabel: str) -> None:
    fig.suptitle(title, fontsize=13, y=.985)
    for ax in axes[-1]:
        ax.set_xlabel("Saved control time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(.5, .925), frameon=False)
    fig.text(.5, .015, "Saved simulator observations; paired deterministic runs do not prove causal RL benefit.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot(original: Path, continuation: Path, output: Path) -> dict[str, Any]:
    launcher = _json(continuation / "root_launcher_receipt.json")
    receipt = _json(continuation / "continuation_receipt.json")
    stitch_path = continuation / "eight_case_stitch.json"
    stitched = _json(stitch_path)
    if (launcher.get("process_attempts") != 1 or launcher.get("exit_code") != 0
            or launcher.get("error") is not None
            or launcher.get("post_execution_hash_mismatches")
            or receipt.get("new_five_case_execution_valid") is not True
            or stitched.get("new_five_case_execution_valid") is not True
            or stitched.get("original_run_02_study_qualified") is not False
            or stitched.get("original_final_C_counts") is not None):
        raise RuntimeError("new continuation is not a closed, valid five-case execution")
    rows = stitched.get("cases")
    names = (*OLD_CASES, *NEW_CASES)
    if not isinstance(rows, list) or tuple(row["case"] for row in rows) != names:
        raise ValueError("stitched result does not contain the fixed eight cases")
    data = {}
    for row in rows:
        key = (row["speed_mps"], row["terrain"], row["actor"])
        data[key] = _read_case(original, continuation, row)
    expected = {(speed, terrain, actor) for speed in SPEEDS for terrain in TERRAINS
                for actor in ACTORS}
    if set(data) != expected:
        raise ValueError("a required completed case is absent or duplicated")
    output.mkdir(parents=True, exist_ok=False)
    colors = {"zero": "#667788", "final_policy": "#cf5b38"}

    fig, axes = _axes()
    for i, speed in enumerate(SPEEDS):
        for j, terrain in enumerate(TERRAINS):
            ax = axes[i, j]
            _decorate(ax, speed=speed, terrain=terrain)
            ax.plot([START_TICK * DT, STOP_TICK * DT], [speed, speed],
                    linestyle="--", color="#222222", linewidth=1.1, label="raw command")
            for actor in ACTORS:
                case = data[speed, terrain, actor]
                t = np.asarray([row["tick"] * DT for row in case["endpoints"]])
                vx = np.asarray([row["body_vx_mps"] for row in case["endpoints"]])
                ax.plot(t, vx, color=colors[actor], linewidth=1.15,
                        label="final policy" if actor == "final_policy" else "zero")
    _finish(fig, axes, output / "speed_tracking.png",
            title="Forward-speed trajectories: command, zero, and final policy",
            ylabel="Body forward velocity (m/s)")

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 5.9), sharex=True)
    fig.subplots_adjust(left=.09, right=.96, bottom=.12, top=.78, hspace=.34)
    for i, speed in enumerate(SPEEDS):
        ax = axes[i]
        ax.axvspan(START_TICK * DT, STOP_TICK * DT, color="#eee7d9", alpha=.45)
        ax.set_title(f"15 mm box; command {speed:.2f} m/s")
        ax.set_ylabel("Pitch (degrees)")
        ax.grid(alpha=.25, linewidth=.6)
        for actor in ACTORS:
            case = data[speed, "box", actor]
            t = [row["tick"] * DT for row in case["endpoints"]]
            pitch = [float(np.degrees(row["rpy_rad"][1])) for row in case["endpoints"]]
            ax.plot(t, pitch, color=colors[actor], linewidth=1.2,
                    label="final policy" if actor == "final_policy" else "zero")
    axes[-1].set_xlabel("Saved control time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(.5, .925), frameon=False)
    fig.suptitle("Box-traversal pitch from saved endpoints", fontsize=13, y=.985)
    fig.text(.5, .02, "Saved simulator observations; no causal RL claim.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(output / "box_pitch.png", dpi=170)
    plt.close(fig)

    fig, axes = _axes()
    for i, speed in enumerate(SPEEDS):
        for j, terrain in enumerate(TERRAINS):
            ax = axes[i, j]
            _decorate(ax, speed=speed, terrain=terrain)
            case = data[speed, terrain, "final_policy"]
            t = [row["tick"] * DT for row in case["trace"]]
            residual = [float(row["applied_action"][0]) * 40.0 for row in case["trace"]]
            ax.plot(t, residual, color=colors["final_policy"], linewidth=1.1,
                    label="final policy applied leg residual")
            ax.axhline(0.0, color="#667788", linewidth=.8, label="zero residual")
            ax.set_ylim(-10.8, 10.8)
    _finish(fig, axes, output / "shared_leg_residual.png",
            title="Actual applied shared-leg extension residual; all wheel residuals zero",
            ylabel="Leg extension residual (mm)")
    receipt = {
        "schema": "saved-rolling-two-process-trajectory-figures-v2",
        "original_run": str(original),
        "continuation_run": str(continuation),
        "eight_case_stitch": _digest(stitch_path),
        "original_run_02_study_qualified": False,
        "new_five_case_execution_valid": True,
        "aggregate_policy_task_and_speed_gates_passed": stitched[
            "aggregate_four_policy_task_and_speed_gates_passed"
        ],
        "no_engine_or_model_calls": True,
        "sources": {
            case["name"]: {"source_process": case["source_process"],
                           "source_dir": case["source_dir"],
                           "completed_control": case["count"],
                           "files": case["source_files"]}
            for case in data.values()
        },
        "figures": {
            name: _digest(output / name)
            for name in ("speed_tracking.png", "box_pitch.png", "shared_leg_residual.png")
        },
        "causal_rl_benefit_inferred": False,
    }
    with (output / "figure_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--continuation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.original_run.resolve(strict=True),
                          args.continuation_run.resolve(strict=True), args.output.resolve())))


if __name__ == "__main__":
    main()
