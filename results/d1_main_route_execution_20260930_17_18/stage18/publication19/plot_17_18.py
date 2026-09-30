"""Plot four fixed saved-record comparisons after C18 independent readbacks close.

This script reads archived control records only. It never loads a model,
imports a physics engine, or advances a simulation. Each trace is checked
against its case receipt and every compressed block's recorded SHA256.
"""

from __future__ import annotations

import builtins
import gzip
import hashlib
import json
from pathlib import Path
import sys


_IMPORT = builtins.__import__
_FORBIDDEN = frozenset(("torch", "mujoco", "stable_baselines3", "gym",
                        "gymnasium", "engine_binding", "wheel_legged_control"))


def _guard(name, *args, **kwargs):
    if name.partition(".")[0] in _FORBIDDEN:
        raise RuntimeError("model or physics import prohibited: " + name)
    return _IMPORT(name, *args, **kwargs)


builtins.__import__ = _guard
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORK = Path(__file__).resolve().parents[2]
C16 = WORK / "continuation16"
C17 = WORK / "continuation17"
C18 = WORK / "continuation18"
OUTPUT = C18 / "figures18"
SOURCE_HASHES: dict[str, dict[str, object]] = {}


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or linked saved input: " + str(path))
    data = path.read_bytes()
    SOURCE_HASHES[str(path)] = {"bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()}
    return data


def _doc(path: Path) -> dict:
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError("saved document is not an object: " + str(path))
    return value


def _readback(stage: Path, name: str, run: str, plural_zip: bool = False) -> dict:
    report = _doc(stage / name)
    zip_key = "checkpoint_ZIPs_verified" if plural_zip else "checkpoint_ZIP_verified"
    if (report.get("run") != str((stage / run).resolve())
            or report.get("source_closure_verified") is not True
            or report.get("physical_calls_performed_by_reader") != 0
            or report.get("model_loaded_by_reader") is not False
            or report.get(zip_key) is not True
            or not isinstance(report.get("numeric_scores"), (dict if plural_zip else list))):
        raise ValueError("independent saved readback not closed: " + str(stage / name))
    return report


def _series(stage: Path, run: str, case: str, actor: str,
            horizon: int) -> dict[str, np.ndarray]:
    folder = stage / run / "heldout" / f"{case}_{actor}"
    receipt = _doc(folder / "case_receipt.json")
    if (receipt.get("case_id") != case or receipt.get("actor") != actor
            or receipt.get("completed_controls") != horizon
            or receipt.get("record_valid") is not True
            or receipt.get("failure") is not None):
        raise ValueError("case receipt identity or horizon differs: " + str(folder))
    blocks = receipt["controller_record_blocks"]
    rows = []
    for block in blocks:
        name = block["file"]
        if (not isinstance(name, str) or Path(name).name != name
                or not name.startswith("control_records_")
                or not name.endswith(".jsonl.gz")):
            raise ValueError("unexpected record block path")
        path = folder / name
        raw = _read(path)
        if (SOURCE_HASHES[str(path)]["sha256"] != block["sha256"]
                or len(raw) != block["bytes"]):
            raise ValueError("saved control block differs from case receipt: " + str(path))
        part = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
        if len(part) != block["rows"]:
            raise ValueError("saved control block row count differs: " + str(path))
        rows.extend(part)
    if len(rows) != horizon:
        raise ValueError("control record horizon differs: " + str(folder))
    for tick, row in enumerate(rows):
        if (row["tick"] != tick or row["actor"] != actor
                or row["checkpoint_sha256"] != receipt["checkpoint_sha256"]):
            raise ValueError("saved control row identity differs: " + str(folder))
    result = {"t": (np.arange(horizon, dtype=float) + 1.0) * .01}
    for key, source in (
        ("vx", ("metrics", "body_com_vx_mps")),
        ("yaw", ("metrics", "body_yaw_rate_rps")),
        ("vx_ref", ("consumed_command", "forward_velocity_mps")),
        ("yaw_ref", ("consumed_command", "yaw_rate_rps")),
    ):
        values = np.asarray([row["info"][source[0]][source[1]] for row in rows],
                            dtype=float)
        if values.shape != (horizon,) or not np.isfinite(values).all():
            raise ValueError("nonfinite plotted field: " + key)
        result[key] = values
    return result


def _score(report: dict, case: str, actor: str) -> dict:
    matches = [row for row in report["numeric_scores"]
               if row["case_id"] == case and row["experiment_actor"] == actor]
    if len(matches) != 1:
        raise ValueError("missing unique saved numeric score: " + case + "/" + actor)
    return matches[0]


def _plot(ax, trace: dict, key: str, *, color: str, label: str,
          width: float = 1.5, style: str = "-") -> None:
    ax.plot(trace["t"], trace[key], linestyle=style, color=color,
            lw=width, label=label)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if {name.partition(".")[0] for name in sys.modules} & _FORBIDDEN:
        raise RuntimeError("model or physics module already loaded")
    # The mirror readback is required for overall C18 closure even though its
    # separate mirrored script is not plotted in this four-panel comparison.
    reports = {
        "C16": _readback(C16, "eval_readback_16.json", "eval_01", plural_zip=True),
        "C17_baseline": _readback(C17, "development_baseline_readback_17.json",
                                  "development_baseline_01"),
        "C17_primary": _readback(C17, "qualification_primary_readback_17.json",
                                 "qualification_primary_01"),
        "C18_primary": _readback(C18, "qualification_primary_readback_18.json",
                                 "qualification_primary_01"),
        "C18_mirror": _readback(C18, "qualification_mirror_readback_18.json",
                                "qualification_mirror_01"),
    }
    rough_failure = _score(reports["C17_primary"], "rough_0p35",
                           "grouped_continue")
    if rough_failure["speed"]["passed"] is not False:
        raise ValueError("C17 rough trace is not the independently scored failure")
    traces = {
        "yaw_baseline": _series(C17, "development_baseline_01", "flat_1p2_yaw",
                                "grouped_continue", 1600),
        "yaw_final": _series(C18, "qualification_primary_01", "flat_1p2_yaw",
                             "grouped_continue", 1600),
        "ramp_baseline": _series(C17, "development_baseline_01", "ramp_0p45_complete",
                                 "grouped_continue", 1800),
        "ramp_final": _series(C18, "qualification_primary_01", "ramp_0p45_complete",
                              "grouped_continue", 1800),
        "rough_old": _series(C16, "eval_01", "rough_0p35", "grouped_continue", 1600),
        "rough_failed": _series(C17, "qualification_primary_01", "rough_0p35",
                                 "grouped_continue", 1600),
        "rough_final": _series(C18, "qualification_primary_01", "rough_0p35",
                                "grouped_continue", 1600),
        "fast_zero": _series(C18, "qualification_primary_01", "flat_1p6", "zero", 1600),
        "fast_final": _series(C18, "qualification_primary_01", "flat_1p6",
                               "grouped_continue", 1600),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), layout="constrained")
    blue, orange, gray, green = "#126a8a", "#b45334", "#68727d", "#2f7d5b"
    ax = axes[0, 0]
    _plot(ax, traces["yaw_baseline"], "yaw", color=orange,
          label="C17 baseline + grouped policy")
    _plot(ax, traces["yaw_final"], "yaw", color=blue,
          label="C18 filtered controller + grouped policy")
    _plot(ax, traces["yaw_final"], "yaw_ref", color="#20252a",
          label="Consumed yaw reference", width=1.1, style="--")
    ax.set(title="Original steering script", xlim=(3.8, 8.5),
           ylabel="Body yaw rate (rad/s)")

    ax = axes[0, 1]
    _plot(ax, traces["ramp_baseline"], "vx", color=orange,
          label="C17 baseline + grouped policy")
    _plot(ax, traces["ramp_final"], "vx", color=blue,
          label="C18 filtered controller + grouped policy")
    _plot(ax, traces["ramp_final"], "vx_ref", color="#20252a",
          label="Consumed speed reference", width=1.1, style="--")
    ax.set(title="Full ramp tracking", xlim=(5.0, 13.55),
           ylabel="Base-body inertial COM $v_x$ (body axis, m/s)")

    ax = axes[1, 0]
    _plot(ax, traces["rough_old"], "vx", color=gray,
          label="C16 grouped policy")
    _plot(ax, traces["rough_failed"], "vx", color=orange,
          label="C17 grouped policy (failed speed gate)")
    _plot(ax, traces["rough_final"], "vx", color=blue,
          label="C18 filtered controller + grouped policy")
    _plot(ax, traces["rough_final"], "vx_ref", color="#20252a",
          label="Consumed speed reference", width=1.1, style="--")
    ax.set(title="Rough hold: fixed ticks 245–645", xlim=(2.45, 6.45),
           ylabel="Base-body inertial COM $v_x$ (body axis, m/s)")

    ax = axes[1, 1]
    _plot(ax, traces["fast_zero"], "vx", color=gray,
          label="C18 zero residual")
    _plot(ax, traces["fast_final"], "vx", color=green,
          label="C18 grouped policy")
    _plot(ax, traces["fast_final"], "vx_ref", color="#20252a",
          label="Consumed speed reference", width=1.1, style="--")
    ax.set(title="1.6 m/s script: drive and release", xlim=(0.0, 16.0),
           ylabel="Base-body inertial COM $v_x$ (body axis, m/s)")

    for ax in axes.flat:
        ax.set_xlabel("Simulation time (s)")
        ax.grid(alpha=.22)
        ax.legend(fontsize=7.3, loc="best")
    fig.suptitle("Saved fixed-script control tracking (unchanged grouped RL checkpoint)",
                 fontsize=13)
    OUTPUT.mkdir(exist_ok=False)
    png = OUTPUT / "tracking_17_18.png"
    svg = OUTPUT / "tracking_17_18.svg"
    fig.savefig(png, dpi=190)
    fig.savefig(svg)
    plt.close(fig)
    result = {
        "schema": "d1-saved-data-four-panel-figure-18-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": dict(sorted(SOURCE_HASHES.items())),
        "output_sha256": {str(path): {"bytes": path.stat().st_size,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in (png, svg)},
        "matplotlib_version": matplotlib.__version__,
        "model_calls": 0, "physical_calls": 0, "training_calls": 0,
        "scope": "Saved deterministic scripts; curves are different trajectories, not tick-matched counterfactuals.",
        "speed_definition": "Inertial base-body center-of-mass velocity projected onto the body forward axis; not whole-robot COM velocity.",
        "RL_claim": "No RL benefit is inferred from this visualization; use the independent paired numeric scorer.",
        "post_step_time": "Each post-step measurement and its consumed servo command are drawn at the control interval end.",
    }
    with (OUTPUT / "figure_receipt_18.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
