"""Exact 11-S twelve-actor heldout task table, independent of train choices."""

from __future__ import annotations

from dataclasses import dataclass

from full_drive_command_08 import FullDriveCommand
from full_drive_schedule_08 import COURSE_SPAWNS_M
from short_heldout_11 import WorldUprightHeldoutSchedule

HELDOUT_TABLE_SCHEMA = "d1-world-upright-short-rl16-six-matched-tasks-v1"


@dataclass(frozen=True, slots=True)
class HeldoutTask:
    case_id: str
    terrain: str
    speed_mps: float
    yaw_amplitude_rps: float
    seed: int
    horizon: int
    hold_start: int
    hold_end: int
    release_tick: int
    final_start: int


TASKS = (
    HeldoutTask("flat_0p6", "flat", 0.6, 0.0, 88701, 1600, 295, 695, 695, 1500),
    HeldoutTask("flat_1p6", "flat", 1.6, 0.0, 88702, 1600, 495, 895, 895, 1500),
    HeldoutTask("flat_1p2_yaw", "flat", 1.2, 0.3, 88703, 1600, 415, 815, 815, 1500),
    HeldoutTask("bumps_0p4", "bumps", 0.4, 0.0, 88704, 1600, 255, 655, 655, 1500),
    HeldoutTask("rough_0p35", "rough", 0.35, 0.0, 88705, 1600, 245, 645, 645, 1500),
    HeldoutTask("ramp_0p45_complete", "ramp", 0.45, 0.0, 88706, 1800,
                600, 1000, 1355, 1700),
)
TASK_BY_ID = {task.case_id: task for task in TASKS}
ORDER = tuple(task.case_id for task in TASKS)


def heldout_schedule(case_id: str) -> WorldUprightHeldoutSchedule:
    """Construct only the requested preregistered raw task, no random draws."""
    if type(case_id) is not str or case_id not in TASK_BY_ID:
        raise ValueError("unknown 11-S heldout task")
    task = TASK_BY_ID[case_id]
    if (task.hold_end - task.hold_start != 400
            or task.final_start != task.horizon - 100
            or not 175 < task.hold_start < task.hold_end <= task.release_tick
            or task.release_tick >= task.final_start):
        raise RuntimeError("heldout table window identities differ")
    raw = []
    for tick in range(task.horizon):
        speed = task.speed_mps if 175 <= tick < task.release_tick else 0.0
        yaw = 0.0
        if task.yaw_amplitude_rps and task.hold_start <= tick < task.hold_end:
            within_hold = tick - task.hold_start
            yaw = (task.yaw_amplitude_rps
                   if within_hold < 100 or within_hold >= 300
                   else -task.yaw_amplitude_rps)
        raw.append(FullDriveCommand(
            forward_velocity_mps=speed, lateral_velocity_mps=0.0,
            yaw_rate_rps=yaw, clearance_m=0.455, jump_requested=False,
        ))
    return WorldUprightHeldoutSchedule(
        case_id=task.case_id, terrain=task.terrain, seed=task.seed,
        spawn_position_m=COURSE_SPAWNS_M[task.terrain], raw_commands=tuple(raw),
    )


__all__ = ("HELDOUT_TABLE_SCHEMA", "ORDER", "TASKS", "HeldoutTask", "heldout_schedule")
