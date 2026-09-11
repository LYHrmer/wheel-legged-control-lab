"""延迟通道指标：离散步数换算与部分/完整时域跟踪汇总。"""

import math

_CHANNEL_STEP_MS = {"measurement": 10, "actuator": 2}
_TOL = 1e-8
_FIELDS = (
    "time_s",
    "velocity_error_mps",
    "yaw_rate_error_rps",
    "height_error_m",
    "roll_error_rad",
    "pitch_error_rad",
    "measurement_age_s",
    "torque_input_clipped_fraction",
)


def delay_steps(delay_ms, channel):
    """把毫秒延迟换算为通道离散步数；不可整除或未知通道抛 ValueError。"""
    if isinstance(delay_ms, bool) or not isinstance(delay_ms, int):
        raise TypeError("delay_ms 必须是非布尔整数")
    if delay_ms < 0:
        raise ValueError("delay_ms 必须非负")
    if not isinstance(channel, str) or channel not in _CHANNEL_STEP_MS:
        raise ValueError("channel 必须是 'measurement' 或 'actuator'")
    step_ms = _CHANNEL_STEP_MS[channel]
    if delay_ms % step_ms != 0:
        raise ValueError(
            f"delay_ms={delay_ms} 不能被通道 {channel} 的 {step_ms} ms 步长整除"
        )
    return delay_ms // step_ms


def _finite(value, name):
    """校验为有限实数并转成 float。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限值")
    return number


def _rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def summarize(rows, requested_duration_s, terminal_reason):
    """汇总一次运行的跟踪指标，只在真正跑满时域时给出 full_horizon_tracking。"""
    if not isinstance(rows, (list, tuple)) or len(rows) == 0:
        raise ValueError("rows 不能为空")
    if not isinstance(terminal_reason, str):
        raise TypeError("terminal_reason 必须是字符串")
    requested = _finite(requested_duration_s, "requested_duration_s")
    if requested <= 0.0:
        raise ValueError("requested_duration_s 必须为正")

    columns = {name: [] for name in _FIELDS}
    previous_time = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"rows[{index}] 必须是字典")
        for name in _FIELDS:
            if name not in row:
                raise ValueError(f"rows[{index}] 缺少字段 {name}")
            columns[name].append(_finite(row[name], f"rows[{index}].{name}"))
        time_s = columns["time_s"][-1]
        if time_s <= 0.0:
            raise ValueError(f"rows[{index}].time_s 必须为正")
        if previous_time is not None and time_s <= previous_time:
            raise ValueError(f"time_s 必须严格递增，在 rows[{index}] 处违反")
        if time_s > requested + _TOL:
            raise ValueError(f"rows[{index}].time_s 超出 requested_duration_s")
        previous_time = time_s

    duration_s = previous_time
    completed = terminal_reason == "time_limit" and abs(duration_s - requested) <= _TOL

    rolls = columns["roll_error_rad"]
    pitches = columns["pitch_error_rad"]
    attitude_ms = sum(r * r + p * p for r, p in zip(rolls, pitches)) / len(rolls)
    partial = {
        "velocity_rmse_mps": _rmse(columns["velocity_error_mps"]),
        "yaw_rmse_rps": _rmse(columns["yaw_rate_error_rps"]),
        "height_rmse_m": _rmse(columns["height_error_m"]),
        "attitude_rmse_rad": math.sqrt(attitude_ms),
    }
    clipped = columns["torque_input_clipped_fraction"]
    return {
        "executed_steps": len(rows),
        "duration_s": duration_s,
        "requested_duration_s": requested,
        "completed": completed,
        "terminal_reason": terminal_reason,
        "partial_horizon_tracking": partial,
        "full_horizon_tracking": dict(partial) if completed else None,
        "measurement_age_max_s": max(columns["measurement_age_s"]),
        "mean_input_clipped_fraction": sum(clipped) / len(clipped),
    }
