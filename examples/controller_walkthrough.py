"""Inspect the open-loop model and compare LQR/MPC recovery from five degrees."""

from time import perf_counter

import numpy as np

from wheel_legged_control.controllers import LinearMPCController, LQRController, TrackingCommand
from wheel_legged_control.model import WheelLeggedPlant


def simulate(controller_name: str) -> dict[str, float]:
    plant = WheelLeggedPlant()
    initial_state = np.array([0.0, np.deg2rad(5.0), 0.0, 0.0, 0.0, 0.0])
    plant.reset(initial_state)
    controller = (
        LQRController(plant)
        if controller_name == "LQR"
        else LinearMPCController(plant, horizon=20)
    )
    controller.reset(initial_state)
    pitches = []
    solve_times = []
    start = perf_counter()
    for _ in range(200):
        control = controller.compute(plant.state(), TrackingCommand())
        plant.step(control)
        pitches.append(abs(np.rad2deg(plant.state()[1])))
        solve_times.append(float(getattr(controller, "last_solve_ms", 0.0)))
    return {
        "final_pitch_deg": pitches[-1],
        "max_pitch_deg": max(pitches),
        "solve_p95_ms": float(np.percentile(solve_times, 95)),
        "rollout_ms": (perf_counter() - start) * 1e3,
    }


def main() -> None:
    plant = WheelLeggedPlant()
    a, _ = plant.linearize()
    lqr = LQRController(plant)
    np.set_printoptions(precision=4, suppress=True)
    print("Open-loop discrete eigenvalues:")
    print(np.linalg.eigvals(a))
    print("\nLQR closed-loop eigenvalues:")
    print(lqr.closed_loop_eigenvalues)
    print("\nFive-degree recovery (4 s):")
    for name in ("LQR", "MPC"):
        print(name, simulate(name))


if __name__ == "__main__":
    main()

