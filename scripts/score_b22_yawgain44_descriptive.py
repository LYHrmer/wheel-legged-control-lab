"""Descriptive SSE over saved controls for the yaw-law exploration.

Usage: score_b22_yawgain44_descriptive.py RUN_DIR [GLOB]. Drive windows and the zero reference come
from the completed 43 readback (same task schedules). Not the qualification reader."""

import gzip
import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
ref = Path("/home/lyh/wheel-legged-control-lab/results/b22_symmetry43_physics_20261018_01")
rb = json.loads((ref / "readback.json").read_text())
win = {c["case_id"]: c["rl_terms"]["drive_window"] for c in rb["cases"] if c["actor"] == "zero"}
zero43 = {c["case_id"]: c["rl_terms"] for c in rb["cases"] if c["actor"] == "zero"}
out = {}
for gain_dir in sorted(RUN.glob(sys.argv[2] if len(sys.argv) > 2 else "K*")):
    for case, (a, b) in win.items():
        folder = gain_dir / "heldout" / f"{case}_zero"
        if not (folder / "case_receipt.json").exists():
            continue
        rec = json.loads((folder / "case_receipt.json").read_text())
        vx = yaw = 0.0
        n = 0
        roll = 0.0
        ticks = 0
        for blk in rec["controller_record_blocks"]:
            with gzip.open(folder / blk["file"], "rt") as f:
                for line in f:
                    r = json.loads(line)
                    ticks += 1
                    i = r["info"]
                    m = i["metrics"]
                    roll = max(roll, abs(m.get("world_roll_deg", 0) or 0))
                    if a <= r["tick"] < b:
                        c = i["consumed_command"]
                        vx += ((c["forward_velocity_mps"] - m["body_com_vx_mps"]) / 0.25) ** 2
                        yaw += ((c["yaw_rate_rps"] - m["body_yaw_rate_rps"]) / 0.4) ** 2
                        n += 1
        out.setdefault(gain_dir.name, {})[case] = {
            "sse_vx": vx,
            "sse_yaw": yaw,
            "sse": vx + yaw,
            "drive_ticks": n,
            "controls": ticks,
            "ref43_zero_sse": zero43[case]["sse_total"],
        }
for g, d in out.items():
    tot = sum(v["sse"] for v in d.values())
    ref_tot = sum(v["ref43_zero_sse"] for v in d.values())
    print(
        g,
        "pooled",
        round(tot, 3),
        "vs 43 zero",
        round(ref_tot, 3),
        "ratio",
        round(tot / ref_tot, 4) if len(d) == 6 else "partial",
    )
    for c, v in d.items():
        print(
            "  ",
            c,
            "sse",
            round(v["sse"], 3),
            "vx",
            round(v["sse_vx"], 3),
            "yaw",
            round(v["sse_yaw"], 3),
            "| 43zero",
            round(v["ref43_zero_sse"], 3),
            "n",
            v["drive_ticks"],
            v["controls"],
        )
