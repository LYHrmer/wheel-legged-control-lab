"""Single-variable checks of the flat-heightfield delayed standing failure.

Diagnostic model/ground-query overrides are scoped to each fresh plant. They
never change the production defaults or inject ground truth into feedback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import probe_d1_delay_parity as probe
from wheel_legged_control.d1.training_terrain import TrainingGroundReference

VARIANTS = (
    "reference",
    "solver_iterations_100",
    "ccd_tolerance_1e-8",
    "fixed_attitude",
    "leg_pd_quarter",
    "attitude_feedback_quarter",
)


def run_variant(name):
    if name not in VARIANTS:
        raise ValueError("unknown diagnostic variant")
    original_build = probe.build_case
    parameters = {}

    def build(**kwargs):
        plant, loop = original_build(**kwargs)
        parameters.update(
            solver_iterations_before=int(plant.model.opt.iterations),
            ccd_tolerance_before=float(plant.model.opt.ccd_tolerance),
            ccd_iterations=int(plant.model.opt.ccd_iterations),
            disableflags=int(plant.model.opt.disableflags),
        )
        if name == "solver_iterations_100":
            plant.model.opt.iterations = 100
        elif name == "ccd_tolerance_1e-8":
            plant.model.opt.ccd_tolerance = 1e-8
        elif name == "fixed_attitude":
            get_ground = loop.provider.ground_reference

            def level_reference():
                # Flat nominal prior, NOT a query of the true terrain. Retain
                # the same measured height and all other delayed feedback.
                return TrainingGroundReference(get_ground().height_m, 0.0, 0.0)

            loop.provider.ground_reference = level_reference
        elif name == "leg_pd_quarter":
            loop.controller.controller.leg_kp *= 0.25
            loop.controller.controller.leg_kd *= 0.25
        elif name == "attitude_feedback_quarter":
            loop.controller.controller.roll_pitch_kp *= 0.25
            loop.controller.controller.roll_pitch_kd *= 0.25
        parameters.update(
            solver_iterations_after=int(plant.model.opt.iterations),
            ccd_tolerance_after=float(plant.model.opt.ccd_tolerance),
            leg_kp=loop.controller.controller.leg_kp.tolist(),
            leg_kd=loop.controller.controller.leg_kd.tolist(),
            attitude_kp=loop.controller.controller.roll_pitch_kp,
            attitude_kd=loop.controller.controller.roll_pitch_kd,
        )
        return plant, loop

    with patch.object(probe, "build_case", build):
        rows, summary = probe.run_case("heightfield_bare")
    return rows, {**summary, "variant": name, "parameters": parameters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS[:4])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must not repeat")
    args.output.mkdir(parents=True)
    sources = (*probe.SOURCE_FILES, "scripts/diagnose_d1_heightfield_delay.py")
    hashes = {p: hashlib.sha256((probe.ROOT / p).read_bytes()).hexdigest() for p in sources}
    protocol = {
        "schema": "d1-heightfield-delay-diagnosis-v1",
        "variants": args.variants,
        "hypotheses": {
            "solver_iterations_100": "insufficient constraint convergence; more iterations should improve standing",
            "ccd_tolerance_1e-8": "convex collision precision; tighter tolerance should change early contact loss",
            "fixed_attitude": "measured support attitude amplifies feedback; a level prior should improve standing",
            "leg_pd_quarter": "delayed leg feedback is too stiff; one-quarter PD scale should reduce oscillation",
            "attitude_feedback_quarter": "delayed body feedback is too stiff; one-quarter attitude feedback scale should reduce oscillation",
        },
        "reference": probe.protocol(["heightfield_bare"], 4.0, True, probe.MEASUREMENT_SEED),
        "mujoco_version": mujoco.__version__,
        "source_sha256": hashes,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(probe.ROOT / path, arcname=path)
    summaries = []
    for name in args.variants:
        rows, summary = run_variant(name)
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    changed = [
        p for p in sources if hashlib.sha256((probe.ROOT / p).read_bytes()).hexdigest() != hashes[p]
    ]
    if changed:
        raise RuntimeError(f"source changed during probe: {changed}")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
