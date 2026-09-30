"""Read-only snapshot of reset control state; no constructors or forwards."""
from dataclasses import asdict
from scripts.run_d1_latest_rl import _jsonable


def control_state20(env):
    c, servo, decision = env.controller, env._servo, env.decision
    return _jsonable({
        'variant': c.variant, 'wheel_integral_nm': c._wheel_integral_nm.copy(),
        'wheel_common_reference_z_rad_s': c.wheel_common_reference_z_rad_s,
        'previous_servo_forward_mps': c._previous_servo_forward_mps,
        'stop_latched': c._stop_latched,
        'servo_last_tick': servo._last_tick,
        'servo_applied': asdict(servo._applied),
        'servo_receipt': None if servo._receipt is None else asdict(servo._receipt),
        'provider_sequence': decision.context.state.sequence,
        'provider_control_time_s': decision.context.state.control_time_s,
        'previous_action': decision.context.previous_normalized_action.copy(),
        'context': asdict(decision.context),
    })
