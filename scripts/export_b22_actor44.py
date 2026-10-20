"""NumPy export + SB3 parity of a completed 44 training arm (zero physics).

A thin driver over the completed-42 probe: the probe's fixed numerical protocol (384 SB3 and
384 NumPy predictions, 192 mirror projections, tolerance 1e-6) is run unchanged against the
final checkpoint of one 44 arm; only the checkpoint folder, its SHA and the output identities
are rebound.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "b22_yawtrain44_20261018_02"
RUN = ROOT / "results" / DEFAULT_RUN  # replaced by --run
ARMS = ("ffcontrol", "ffyaw", "ffrough", "ffsse", "ffyaw_s21", "ffsse_s21")


def load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe44_base", ROOT / "scripts/probe_b22_symmetry43.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind(probe, arm: str, revision: str = "20261018_01") -> None:
    run_tag = RUN.name
    unit = f"wheel-{run_tag}-numeric-{arm}-20261018-01.service"
    probe.FOLDER = RUN / f"train_{arm}/final_checkpoint"
    probe.MODEL_SHA = probe.sha(probe.FOLDER / "final_model.zip")
    probe.OUTPUT = ROOT / f"results/{run_tag}_numeric_{arm}_{revision}"
    probe.PLAN = ROOT / f"results/{run_tag}_numeric_plan_{arm}_{revision}.json"
    probe.UNIT = unit
    uid = os.getuid()
    probe.SCOPE = f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}"
    probe.AUTHORED = (*probe.AUTHORED, "scripts/export_b22_actor44.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run", help="results/ directory name of the completed training run")
    parser.add_argument("--revision", default="20261018_01", help="numeric output revision suffix")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    global RUN
    if args.run:
        RUN = ROOT / "results" / args.run
    probe = load_probe()
    bind(probe, args.arm, args.revision)
    sys.argv = [
        sys.argv[0],
        *(["--prepare"] if args.prepare else ["--execute"] if args.execute else []),
    ]
    return probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
