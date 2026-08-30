"""Print the identified D1 model and run short LQR/MPC closed-loop checks."""

from __future__ import annotations

import numpy as np

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.hierarchical import D1LQRVMCController, D1MPCVMCController
from wheel_legged_control.d1.linear_model import identify_sagittal_model
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def main() -> None:
    np.set_printoptions(precision=5, suppress=True)
    linear_model = identify_sagittal_model()
    print("state order: [x, pitch, x_velocity, pitch_rate]")
    print("identified A:\n", linear_model.a)
    print("identified B [total longitudinal force N]:\n", linear_model.b)

    for controller_type in (D1LQRVMCController, D1MPCVMCController):
        plant = D1Plant()
        controller = controller_type(plant, linear_model=linear_model)
        states = D1MujocoTruthStateSource(plant)
        state = states.reset()
        velocities = []
        pitches = []
        for step in range(300):
            command = D1Command(forward_velocity_mps=0.5 if step >= 50 else 0.0)
            push = np.asarray((120.0, 0.0, 0.0)) if 200 <= step < 212 else None
            plant.step(controller.compute(command, state), push_force_world_n=push)
            state = states.read()
            velocities.append(plant.base_velocity(local=False)[0][0])
            pitches.append(plant.base_rpy[1])
        velocity_error = np.asarray(velocities[50:]) - 0.5
        print(
            f"{controller.baseline_name}: "
            f"velocity RMSE={np.sqrt(np.mean(velocity_error**2)):.3f} m/s, "
            f"pitch RMSE={np.rad2deg(np.sqrt(np.mean(np.asarray(pitches) ** 2))):.3f} deg, "
            f"fallen={plant.has_fallen()}, solve={controller.last_solve_ms:.3f} ms"
        )


if __name__ == "__main__":
    main()
