import numpy as np
import pytest

from wheel_legged_control.d1.interactive import (
    D1InteractiveSimulation,
    run_scripted_demo,
)
from wheel_legged_control.d1.model import D1Plant


def test_base_local_velocity_uses_visible_base_link_frame() -> None:
    plant = D1Plant()
    plant.data.qvel[:6] = np.asarray((0.2, -0.1, 0.3, 0.5, -0.4, 0.2))
    import mujoco

    mujoco.mj_forward(plant.model, plant.data)
    world_linear, world_angular = plant.base_velocity(local=False)
    local_linear, local_angular = plant.base_velocity(local=True)
    rotation = plant.data.xmat[plant.base_body_id].reshape(3, 3)
    np.testing.assert_allclose(local_linear, rotation.T @ world_linear)
    np.testing.assert_allclose(local_angular, rotation.T @ world_angular)


def test_course_arena_contains_physical_terrain() -> None:
    flat = D1Plant(arena="flat")
    course = D1Plant(arena="course")
    assert len(flat.terrain_geom_ids) == 1
    assert len(course.terrain_geom_ids) > 80
    assert course.model.ngeom > flat.model.ngeom


def test_keyboard_turn_and_guarded_jump_are_physical() -> None:
    simulation = D1InteractiveSimulation()
    for _ in range(100):
        simulation.step()
    simulation.teleop.forward_velocity_mps = 0.15
    simulation.teleop.yaw_rate_rps = 0.35
    for _ in range(250):
        simulation.step()
    assert simulation.plant.base_rpy[2] > np.deg2rad(3.0)
    assert not simulation.plant.has_fallen()

    simulation.reset("jump")
    for _ in range(100):
        simulation.step()
    simulation.teleop.handle_key(ord(" "))
    heights = []
    contacts = []
    status = None
    for _ in range(140):
        status = simulation.step()
        heights.append(float(simulation.plant.base_position[2]))
        contacts.append(simulation.plant.wheel_ground_contacts)
    assert status is not None and status.completed_jumps == 1
    assert max(heights) > 0.51
    assert min(contacts) == 0
    assert not simulation.plant.has_fallen()


def test_residual_policy_can_be_toggled_and_is_gated_during_jump() -> None:
    class ZeroPolicy:
        def predict(self, observation: np.ndarray, deterministic: bool) -> tuple[np.ndarray, None]:
            assert observation.shape == (42,)
            assert deterministic
            return np.zeros(2, dtype=np.float32), None

    simulation = D1InteractiveSimulation(residual_policy=ZeroPolicy())
    assert simulation.teleop.handle_key(ord("l")) is None
    status = simulation.step()
    assert status.rl_mode == "on"
    simulation.teleop.handle_key(ord(" "))
    status = simulation.step()
    assert status.jump_phase == "crouch"
    assert status.rl_mode == "gated"


@pytest.mark.parametrize("zone", ("rough", "ramp", "stairs", "bumps", "jump"))
def test_scripted_course_zone_reaches_its_acceptance_target(zone: str) -> None:
    metrics = run_scripted_demo(zone)
    assert metrics["success"] == 1
    assert metrics["step_time_p95_ms"] < 10.0
    if zone == "jump":
        assert metrics["cleared_hurdles"] >= 1
        assert metrics["jump_height_gain_m"] > 0.05
