"""Show every D1 residual-reward component for one controlled transition."""

from __future__ import annotations

import numpy as np

from wheel_legged_control.d1.env import D1ResidualEnv


def main() -> None:
    env = D1ResidualEnv(randomize=False, episode_seconds=0.1)
    env.reset(seed=5, options={"scenario": "nominal"})
    _, reward, _, _, info = env.step(np.asarray((0.2, -0.1), dtype=np.float32))
    print("D1 reward breakdown")
    for name, value in info["reward_terms"].items():
        print(f"  {name:24s} {value:+.6f}")
    assert np.isclose(reward, info["reward_terms"]["total"])
    env.close()


if __name__ == "__main__":
    main()
