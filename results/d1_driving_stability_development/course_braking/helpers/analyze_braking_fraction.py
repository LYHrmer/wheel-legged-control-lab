#!/usr/bin/env python3
"""Audit the recorded alpha comparison, including motion hidden by net drift."""
import argparse
import json
from pathlib import Path

import numpy as np
from probe_gui_baseline import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="gui_baseline root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.input.resolve()
    directory = root / "braking_fraction_01"
    summaries = json.loads((directory / "summary.json").read_text())
    rows, hashes, checks = [], {}, []
    for summary in summaries:
        case, alpha = summary["case"]["name"], summary["alpha"]
        data_dir = directory / f"{case}_alpha{str(alpha).replace('.', 'p')}"
        for name in ("states.npz", "telemetry.csv", "summary.json", "protocol.json"):
            hashes[str((data_dir / name).relative_to(root))] = digest(data_dir / name)
        cut = round(summary["release_time_s"]/.01)
        with np.load(data_dir / "states.npz") as state:
            qpos, times = state["qpos"], state["segment_time_s"]
            position = qpos[cut:, 0]-qpos[cut, 0]
            dx = np.diff(position)
            vx = dx/np.diff(times[cut:])
            metrics = {
                "case": case, "alpha": alpha,
                "reference_edge": summary["release_reference_events"][0],
                "release_net_dx_m": float(position[-1]),
                "release_max_forward_excursion_m": float(np.max(position)),
                "release_max_backward_excursion_m": float(max(0., -np.min(position))),
                "release_total_backward_travel_m": float(-np.sum(dx[dx < 0])),
                "release_max_retreat_from_prior_peak_m": float(np.max(np.maximum.accumulate(position)-position)),
                "release_max_backward_world_speed_mps": float(max(0., -np.min(vx))),
                "last_second_mean_abs_world_vx_mps": float(np.mean(np.abs(vx[-100:]))),
                "time_to_vx_below_0p05_for_0p2s_s": summary["time_to_vx_below_0p05_for_0p2s_s"],
            }
            rows.append(metrics)
            with np.load(directory / f"{case}_alpha0p0/states.npz") as reference:
                same = {key: bool(np.array_equal(state[key][:cut+1], reference[key][:cut+1]))
                        for key in ("qpos", "qvel", "segment_time_s")}
                same["pre_release_torque"] = bool(np.array_equal(state["applied_torque_nm"][:cut],
                                                                  reference["applied_torque_nm"][:cut]))
                assert all(same.values()), (case, alpha, same)
            repeated = None
            if alpha in (0., 1.):
                previous = root / ("forward_02" if alpha == 0. else "forward_braking_01") / case
                with np.load(previous / "states.npz") as reference:
                    repeated = {key: bool(np.array_equal(state[key], reference[key])) for key in state.files}
                    assert all(repeated.values()), (case, alpha, repeated)
                hashes[str((previous / "states.npz").relative_to(root))] = digest(previous / "states.npz")
            checks.append({"case": case, "alpha": alpha, "pre_release_identical": same,
                           "full_endpoint_replay_identical": repeated})
    result = {"all_passed": True, "rows": rows, "checks": checks, "input_sha256": hashes,
              "notes": "Net drift alone can hide forward motion followed by retreat; excursions and accumulated backward travel are both reported. All velocities are world-x finite differences. No physics run by this audit."}
    write_json(args.output, result)
    print(json.dumps({"all_passed": True, "rows": len(rows), "rough3": [r for r in rows if 'rough' in r['case']]}, indent=2))


if __name__ == "__main__":
    main()
