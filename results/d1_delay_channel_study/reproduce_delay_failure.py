"""Actual zero-residual D1 closed loop; red means early termination."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from wheel_legged_control.d1.locomotion_commands import validation_command_schedule
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv, LOCOMOTION_SPAWN_POSITION_M
from wheel_legged_control.d1.locomotion_terrain import locomotion_terrain_configs
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise
from wheel_legged_control.d1.state_provider import D1StateProviderConfig
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegControlConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    source = D1StateProviderConfig(
        "imu_encoder_fusion", sensor_delay_steps=3,
        sensor_noise=D1SensorNoise(0.002, 0.03, 0.0005, 0.005),
        initial_position_m=LOCOMOTION_SPAWN_POSITION_M, initial_rpy_rad=(0, 0, 0),
    )
    schedule = validation_command_schedule("development", 60.0)
    start = perf_counter()
    env = D1LocomotionEnv(
        episode_seconds=3.0, terrain=locomotion_terrain_configs("development")[0],
        provider_config=source, command_source=schedule.cmd_at,
        wheel_leg_control=D1WheelLegControlConfig(0.55, 1.5, 4, 1, 0.25),
    )
    rows = []
    try:
        _, reset = env.reset(seed=17)
        (args.output / "episode.json").write_text(json.dumps(reset, indent=2))
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros(8))
            rows.append(info["metrics"])
            if terminated or truncated:
                break
        result = {"completed": bool(truncated and not terminated),
                  "duration_s": rows[-1]["time_s"], "executed_steps": len(rows),
                  "terminal_reason": info["terminal_reason"],
                  "nonflat_fraction": info["terrain_exposure"]["nonflat_fraction"],
                  "measurement_seed": reset["episode_metadata"]["measurement_seed"],
                  "wall_seconds": perf_counter() - start,
                  "scope": "first 3 s of fixed 60 s development schedule, road0 seed17"}
        (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        assert result["completed"], f"delayed closed loop terminated: {result['terminal_reason']}"
    finally:
        env.close()


if __name__ == "__main__":
    main()
