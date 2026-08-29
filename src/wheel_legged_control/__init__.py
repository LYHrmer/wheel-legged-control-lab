"""Learning-augmented control for a planar wheel-legged robot."""

from .controllers import LinearMPCController, LQRController, TrackingCommand
from .model import WheelLeggedPlant
from .rewards import RewardBreakdown, calculate_reward

__all__ = [
    "LQRController",
    "LinearMPCController",
    "RewardBreakdown",
    "TrackingCommand",
    "WheelLeggedPlant",
    "calculate_reward",
]
