from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wheel_legged_control.d1 import interactive
from wheel_legged_control.d1.interactive import (
    D1InteractiveSimulation,
    build_parser,
    run_scripted_demo,
    run_viewer,
    write_course_audit,
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


def test_estimated_interactive_state_is_sampled_after_each_physics_step() -> None:
    simulation = D1InteractiveSimulation(
        state_mode="estimated",
        state_delay_steps=2,
        sensor_noise=1.0,
        seed=5,
    )

    ages = [simulation.step().state_age_ms for _ in range(30)]

    assert max(ages) == pytest.approx(20.0)
    assert not simulation.plant.has_fallen()


def test_interactive_can_apply_short_horizon_latency_compensation() -> None:
    simulation = D1InteractiveSimulation(
        state_mode="estimated",
        latency_compensation="constant_velocity",
        state_delay_steps=2,
        sensor_noise=0.0,
        seed=5,
    )

    statuses = [simulation.step() for _ in range(5)]

    assert statuses[-1].latency_compensation == "constant_velocity"
    assert statuses[-1].compensation_status == "applied"
    assert statuses[-1].state_age_ms == pytest.approx(20.0)
    assert statuses[-1].compensation_horizon_ms == pytest.approx(20.0)


def test_interactive_exposes_constrained_contact_allocation_diagnostics() -> None:
    simulation = D1InteractiveSimulation(contact_allocation="constrained")

    statuses = [simulation.step() for _ in range(30)]
    status = statuses[-1]

    assert status.contact_allocation == "constrained"
    assert status.allocation_status in {"converged", "feasible_nonconverged"}
    assert status.allocation_status_reason.startswith("slsqp_")
    assert status.allocation_wrench_tracking_status in {"tracked", "limited"}
    assert status.allocation_solve_ms > 0.0
    assert status.allocation_constraint_violation <= 1e-7
    assert status.allocation_force_error_norm_n >= 0.0
    assert status.allocation_moment_error_norm_nm >= 0.0


def test_scripted_demo_records_contact_allocation_diagnostics() -> None:
    metrics = run_scripted_demo("start", contact_allocation="constrained")

    assert metrics["contact_allocation"] == "constrained"
    assert metrics["allocation_solve_p95_ms"] > 0.0
    assert metrics["allocation_solve_p99_ms"] >= metrics["allocation_solve_p95_ms"]
    assert metrics["allocation_constraint_violation_max"] >= 0.0
    assert metrics["allocation_force_error_rms_n"] >= 0.0
    assert metrics["allocation_moment_error_rms_nm"] >= 0.0
    for name in (
        "allocation_converged_ratio",
        "allocation_feasible_nonconverged_ratio",
        "allocation_fallback_ratio",
        "allocation_wrench_limited_ratio",
        "allocation_no_contact_ratio",
    ):
        assert 0.0 <= metrics[name] <= 1.0


def test_course_audit_records_one_contact_allocation_mode(tmp_path) -> None:
    records = write_course_audit(tmp_path, contact_allocation="constrained")

    assert len(records) == 6
    assert {record["contact_allocation"] for record in records} == {"constrained"}
    csv_text = (tmp_path / "course_metrics.csv").read_text(encoding="utf-8")
    markdown = (tmp_path / "course_metrics.md").read_text(encoding="utf-8")
    assert "contact_allocation" in csv_text.splitlines()[0]
    assert "allocation_solve_p95_ms" in csv_text.splitlines()[0]
    assert "Contact allocation: `constrained`" in markdown
    assert "Allocation P95 [ms]" in markdown


def test_viewer_reports_the_selected_contact_allocation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import mujoco.viewer

    class OneStepViewer:
        def __init__(self) -> None:
            self.cam = SimpleNamespace()
            self._running = True

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def is_running(self) -> bool:
            running = self._running
            self._running = False
            return running

        def sync(self) -> None:
            return None

    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *args, **kwargs: OneStepViewer())

    run_viewer("lqr", contact_allocation="constrained")

    output = capsys.readouterr().out
    assert "alloc=constrained/" in output
    assert "alloc_ms=" in output
    assert "force_err=" in output
    assert "moment_err=" in output


def test_interactive_cli_accepts_constrained_contact_allocation() -> None:
    args = build_parser().parse_args(["--contact-allocation", "constrained"])

    assert args.contact_allocation == "constrained"


def test_interactive_cli_forwards_one_contact_allocation_mode_to_every_run_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        interactive,
        "run_viewer",
        lambda *args, **kwargs: calls.append(("viewer", kwargs["contact_allocation"])),
    )
    monkeypatch.setattr(
        interactive,
        "run_scripted_demo",
        lambda *args, **kwargs: calls.append(("demo", kwargs["contact_allocation"])) or {},
    )
    monkeypatch.setattr(
        interactive,
        "write_course_audit",
        lambda *args, **kwargs: calls.append(("audit", kwargs["contact_allocation"])) or [],
    )
    monkeypatch.setattr(
        interactive,
        "render_course_overview",
        lambda *args, **kwargs: calls.append(("overview", kwargs["contact_allocation"])),
    )

    common = ["--contact-allocation", "constrained"]
    interactive.main(common)
    interactive.main(common + ["--demo-zone", "start"])
    interactive.main(common + ["--audit-output", str(tmp_path / "audit")])
    interactive.main(common + ["--overview", str(tmp_path / "overview.png")])

    assert calls == [
        ("viewer", "constrained"),
        ("demo", "constrained"),
        ("audit", "constrained"),
        ("overview", "constrained"),
    ]


@pytest.mark.parametrize("zone", ("start", "rough", "ramp", "stairs", "bumps", "jump"))
def test_scripted_course_zone_reaches_its_acceptance_target(zone: str) -> None:
    metrics = run_scripted_demo(zone)
    assert metrics["success"] == 1, metrics
    assert metrics["step_time_p95_ms"] < 10.0
    assert metrics["compensation_applied_ratio"] == 0.0
    assert metrics["compensation_horizon_p95_ms"] == 0.0
    assert metrics["compensation_rejected_steps"] == 0
    if zone == "jump":
        assert metrics["cleared_hurdles"] >= 1
        assert metrics["jump_height_gain_m"] > 0.05
