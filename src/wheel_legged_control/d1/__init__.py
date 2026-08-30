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
from .state_estimation import (
    D1EstimatorImpairments,
    D1LatencyCompensationResult,
    D1MujocoTruthStateSource,
    D1NoisyDelayedStateSource,
    D1StateEstimate,
    compensate_d1_state_constant_velocity,
    make_default_d1_estimator_impairments,
    prepare_d1_control_state,
)

__all__ = [
    "D1_JOINT_NAMES",
    "D1_LEG_JOINT_NAMES",
    "D1_WHEEL_JOINT_NAMES",
    "D1Command",
    "D1EstimatorImpairments",
    "D1LQRVMCController",
    "D1LatencyCompensationResult",
    "D1MPCVMCController",
    "D1MujocoTruthStateSource",
    "D1NoisyDelayedStateSource",
    "D1Plant",
    "D1ResidualEnv",
    "D1StateEstimate",
    "D1VMCController",
    "compensate_d1_state_constant_velocity",
    "make_default_d1_estimator_impairments",
    "prepare_d1_control_state",
]
