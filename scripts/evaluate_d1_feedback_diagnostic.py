"""Development-only zero-residual feedback-scale contrast, no checkpoint loading.

This archived diagnostic preceded the public configuration flags. Its explicit
runtime overrides are labelled separately from production controller parameters.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_d1_locomotion_experiment as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("leg_pd_quarter", "attitude_feedback_quarter"), required=True
    )
    parser.add_argument("--measurement-delay", type=int, default=2, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    args.output.mkdir(parents=True)
    original = experiment.D1LocomotionEnv

    class DiagnosticEnv(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            controller = self._controller.controller
            if args.variant == "leg_pd_quarter":
                controller.leg_kp *= 0.25
                controller.leg_kd *= 0.25
            else:
                controller.roll_pitch_kp *= 0.25
                controller.roll_pitch_kd *= 0.25
            controller.control_schema = "d1-feedback-development-diagnostic-v1"
            self.diagnostic = {
                "variant": args.variant,
                "leg_kp": controller.leg_kp.tolist(),
                "leg_kd": controller.leg_kd.tolist(),
                "attitude_kp": controller.roll_pitch_kp,
                "attitude_kd": controller.roll_pitch_kd,
            }

        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            self._episode_metadata["runtime_diagnostic"] = self.diagnostic
            info["episode_metadata"] = self.episode_metadata
            return observation, info

    settings = experiment.parser().parse_args(
        [
            "evaluate",
            "--output",
            str(args.output),
            "--source",
            "imu_encoder_fusion",
            "--duration",
            "60",
            "--split",
            "development",
            "--wheel-kp",
            "0.55",
            "--wheel-ki",
            "1.5",
            "--measurement-delay",
            str(args.measurement_delay),
        ]
    )
    experiment.write_json(
        args.output / "protocol.json",
        {
            "schema": "d1-feedback-development-diagnostic-v1",
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(settings).items()
            },
            "runtime_diagnostic": args.variant,
            "quality_thresholds": experiment.QUALITY,
            "scope": "development only; zero action; no model selection or checkpoint load",
        },
    )
    hashes = experiment.snapshot(args.output)
    own_name = str(Path(__file__).relative_to(experiment.ROOT))
    hashes[own_name] = experiment.sha256(__file__)
    # Rebuild once to add this actual override implementation, not just imports.
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for name in hashes:
            archive.add(experiment.ROOT / name, arcname=name, recursive=False)
    experiment.write_json(
        args.output / "source.json",
        {"sha256": hashes, "archive_sha256": experiment.sha256(args.output / "source.tar.gz")},
    )
    try:
        with patch.object(experiment, "D1LocomotionEnv", DiagnosticEnv):
            experiment.evaluate(settings, args.output)
        experiment.verify_source(args.output, hashes)
    except Exception as error:
        experiment.write_json(
            args.output / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
