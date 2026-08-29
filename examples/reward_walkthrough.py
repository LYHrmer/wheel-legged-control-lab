"""Print reward terms and visualize the pitch/velocity reward landscape."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wheel_legged_control.rewards import calculate_reward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/learning/reward_landscape.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    example_state = np.array([0.0, np.deg2rad(5.0), 0.03, 0.40, 0.0, 0.0])
    example_action = np.array([0.50, -0.50])
    breakdown = calculate_reward(example_state, 0.0, 0.0, example_action)
    print("Representative state reward:")
    for name, value in breakdown.as_dict().items():
        print(f"  {name:>20}: {value:+.4f}")

    pitch_deg = np.linspace(-20.0, 20.0, 161)
    velocity_error = np.linspace(-1.5, 1.5, 161)
    landscape = np.empty((len(pitch_deg), len(velocity_error)))
    for row, pitch in enumerate(np.deg2rad(pitch_deg)):
        for column, velocity in enumerate(velocity_error):
            state = np.array([0.0, pitch, 0.0, velocity, 0.0, 0.0])
            landscape[row, column] = calculate_reward(
                state, 0.0, 0.0, np.zeros(2)
            ).total

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    image = axes[0].imshow(
        landscape,
        origin="lower",
        aspect="auto",
        extent=(velocity_error[0], velocity_error[-1], pitch_deg[0], pitch_deg[-1]),
        cmap="viridis",
    )
    axes[0].set_xlabel("velocity error [m/s]")
    axes[0].set_ylabel("pitch [deg]")
    axes[0].set_title("Total reward, other errors = 0")
    figure.colorbar(image, ax=axes[0], label="reward")

    components = breakdown.as_dict()
    components.pop("total")
    colors = ["#3b82f6" if value >= 0.0 else "#ef4444" for value in components.values()]
    axes[1].barh(list(components), list(components.values()), color=colors)
    axes[1].axvline(0.0, color="0.2", linewidth=0.8)
    axes[1].set_xlabel("weighted contribution")
    axes[1].set_title("Representative reward breakdown")
    axes[1].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

