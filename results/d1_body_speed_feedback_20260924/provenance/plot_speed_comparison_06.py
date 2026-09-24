"""Plot saved baseline, 05 and 06 body speed in four complete fixed cases.

Offline only. Usage: python3 plot_speed_comparison_06.py --work W --output PNG
All 1,201 endpoints stay visible; 275..875 is the unchanged scoring window.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES = ("speed200_plane_final_policy", "speed200_box_final_policy",
         "speed250_plane_final_policy", "speed250_box_final_policy")
COLORS = ("#667085", "#2563eb", "#c04435")
LABELS = ("Original policy", "Drive 05", "Body speed 06")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def series(path: Path) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if [row["tick"] for row in rows] != list(range(1201)):
        raise ValueError(f"incomplete endpoint stream: {path}")
    values = np.asarray([row["body_vx_mps"] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite speed: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve(strict=True)
    parent = work.parent
    old_map = {row["case"]: row for row in read(parent / "stability_20260923_rl01/complete_cases_map_03.json")["cases"]}
    receipt = read(work / "run_01/body_evaluation_receipt.json")
    readback = read(work / "root_run_01_readback_01.json")
    if (receipt.get("execution_valid") is not True or readback.get("passed") is not True
            or tuple(row["case"] for row in receipt["cases"]) != CASES):
        raise ValueError("closed 06 execution/readback absent")
    stage05 = parent / "stability_20260924_drive_damping02/run_01/evaluation"
    records = {}
    for name in CASES:
        paths = (Path(old_map[name]["source_dir"]) / "endpoints.jsonl.gz",
                 stage05 / name / "endpoints.jsonl.gz",
                 work / "run_01/evaluation" / name / "endpoints.jsonl.gz")
        records[name] = {"paths": paths, "values": tuple(series(path) for path in paths)}
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(f"output exists or parent absent: {output}")
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.6), sharex=True, sharey=True,
                             constrained_layout=True)
    ticks = np.arange(1201)
    for ax, name in zip(axes.flat, CASES):
        target = int(name[5:8]) / 1000
        terrain = "plane" if "_plane_" in name else "15 mm box"
        ax.axvspan(275, 875, color="#f3e5b8", alpha=.35, label="Speed gate window")
        ax.axvline(175, color="#98a2b3", linewidth=.8, linestyle=":")
        ax.axvline(975, color="#98a2b3", linewidth=.8, linestyle=":")
        command = np.where((ticks >= 175) & (ticks < 975), target, 0.0)
        ax.step(ticks, command, where="post", color="#111827", linewidth=1,
                linestyle="--", label="Command")
        for label, color, values in zip(LABELS, COLORS, records[name]["values"]):
            ax.plot(ticks, values, color=color, linewidth=1, alpha=.88, label=label)
        ax.set_title(f"{terrain}, command {target:.2f} m/s", loc="left", fontsize=10)
        ax.set_xlim(0, 1200)
        ax.grid(alpha=.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Body forward speed (m/s)")
    for ax in axes[1, :]:
        ax.set_xlabel("Control endpoint tick")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, 1.06))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"plot": str(output), **digest(output), "cases": list(CASES),
                      "all_1201_endpoints_shown": True, "speed_window_endpoints": [275, 875],
                      "physical_geometry_rescored": False}))


if __name__ == "__main__":
    main()
