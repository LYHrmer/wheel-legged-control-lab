"""Replay the preselected sensor45 final model and its paired zero baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "continuous_replay_training_helpers", ROOT / "scripts/train_d1_continuous_policy.py"
)
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


def verify_simulation_source(hashes: dict) -> None:
    # The renderer is archived for presentation provenance but is never used
    # by evaluate(). Post-training text/lighting changes cannot alter physics.
    simulation = {
        key: value for key, value in hashes.items() if key != "scripts/render_d1_continuous_task.py"
    }
    TRAINING.verify_source(simulation)


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    hashes = TRAINING.ROLLOUT.sha256
    inventory = json.loads((root / "source.json").read_text())["sha256"]
    verify_simulation_source(inventory)
    checkpoint = root / "checkpoints/seed19000/step131072/model.zip"
    metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
    model = PPO.load(checkpoint, device="cpu")
    probe = TRAINING.make_task()
    TRAINING.ROLLOUT.validate_checkpoint(metadata, model, probe, checkpoint)
    del probe
    output.mkdir(parents=True)
    checks = []
    case = {"seed": 17, "duration_s": 45.0}
    for policy, meta, name in (
        (model, metadata, "dev_train19000_step131072_seed17_45s"),
        (None, None, "zero_seed17_45s"),
    ):
        summary = TRAINING.evaluate(policy, meta, output / name, case, inventory)
        original = root / "rollouts" / name
        with (
            np.load(original / "states.npz", allow_pickle=False) as before,
            np.load(output / name / "states.npz", allow_pickle=False) as after,
        ):
            identical = {
                key: bool(np.array_equal(before[key], after[key]))
                for key in ("qpos", "qvel", "time_s")
            }
        telemetry_identical = hashes(original / "telemetry.csv") == hashes(
            output / name / "telemetry.csv"
        )
        if not all(identical.values()) or not telemetry_identical:
            raise AssertionError(f"saved-checkpoint replay differs from original: {name}")
        checks.append(
            {
                "rollout": name,
                "state_fields_byte_identical": identical,
                "telemetry_byte_identical": telemetry_identical,
                "completed": summary["completed"],
                "quality_pass": summary["quality_pass"],
            }
        )
    result = {
        "model_sha256": hashes(checkpoint),
        "original_manifest_sha256": hashes(root / "manifest.json"),
        "same_saved_checkpoint_reloaded": True,
        "replay_script_sha256": hashes(Path(__file__)),
        "source_verification": "all frozen simulation/training dependencies; unused renderer excluded",
        "checks": checks,
    }
    TRAINING.ROLLOUT.write_json(output / "replay_check.json", result)
    TRAINING.write_manifest(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.output), indent=2))
