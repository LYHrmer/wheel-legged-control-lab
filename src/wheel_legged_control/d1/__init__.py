"""Full-body D1 simulation layer.

The planar teaching model remains the source of the small LQR/MPC examples.
This package adds a 16-actuator MuJoCo plant for validating those ideas on a
robot with explicit links, wheels, contacts, inertias, and actuator limits.
"""

from .controllers import D1Command, D1VMCController
from .env import D1ResidualEnv
from .hierarchical import D1LQRVMCController, D1MPCVMCController
from .model import (
    D1_JOINT_NAMES,
    D1_LEG_JOINT_NAMES,
    D1_WHEEL_JOINT_NAMES,
    D1Plant,
)

__all__ = [
    "D1_JOINT_NAMES",
    "D1_LEG_JOINT_NAMES",
    "D1_WHEEL_JOINT_NAMES",
    "D1Command",
    "D1LQRVMCController",
    "D1MPCVMCController",
    "D1Plant",
    "D1ResidualEnv",
    "D1VMCController",
]
