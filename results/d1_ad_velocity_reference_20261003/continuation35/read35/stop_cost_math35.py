"""C35 stop, retention and full-scene torque-square arithmetic on saved data."""
from __future__ import annotations

import math
import numpy as np

from task_math35 import entry_heading35


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise AssertionError(reason)


def stop_metrics35(side_qpos, side_qvel, *, stop_request_control: int,
                   handoff_control: int, support_margin_by_control,
                   all_four_loaded_by_control, pending_swing_by_control) -> dict:
    """A 50-control real low-speed/all-four tail before handoff is mandatory."""
    q = np.asarray(side_qpos, dtype=np.float64)
    v = np.asarray(side_qvel, dtype=np.float64)
    margins = np.asarray(support_margin_by_control, dtype=np.float64)
    loaded = np.asarray(all_four_loaded_by_control, dtype=bool)
    pending = np.asarray(pending_swing_by_control, dtype=bool)
    need(q.ndim == v.ndim == 2 and len(q) == len(v)
         and q.shape[1] >= 7 and v.shape[1] >= 6
         and np.isfinite(q).all() and np.isfinite(v).all()
         and margins.shape == loaded.shape == pending.shape == (len(q)-1,),
         'missing finite actual stop states/support')
    need(type(stop_request_control) is int and type(handoff_control) is int
         and 0 <= stop_request_control < handoff_control <= len(q)-1,
         'invalid actual stop/handoff control boundary')
    _, left = entry_heading35(q[0])
    displacement = (q[stop_request_control:handoff_control+1, :2]
                    - q[stop_request_control, :2])@left
    tail = slice(handoff_control-50, handoff_control)
    need(handoff_control >= 50, 'handoff lacks 50-control stable tail')
    need(np.isfinite(margins[tail]).all(),
         'handoff tail lacks actual four-contact static margins')
    # Control k uses actual post-state k+1 and all five native support samples.
    origin_speed = np.linalg.norm(v[1:, :3], axis=1)
    angular_speed = np.linalg.norm(v[1:, 3:6], axis=1)
    stable = bool(np.all(loaded[tail]) and not np.any(pending[tail])
                  and np.all(margins[tail] >= .005)
                  and np.all(origin_speed[tail] <= .01)
                  and np.all(angular_speed[tail] <= .05))
    delay = handoff_control-stop_request_control
    distance = float(np.max(np.abs(displacement)))
    return {'stop_delay_controls': delay,
            'maximum_lateral_excursion_after_stop_request_m': distance,
            'stable_last_50_controls': stable,
            'no_pending_swing_last_50_controls': bool(not np.any(pending[tail])),
            'minimum_tail_static_support_margin_m': float(np.min(margins[tail])),
            'maximum_tail_origin_speed_mps': float(np.max(origin_speed[tail])),
            'maximum_tail_world_angular_speed_rps': float(np.max(angular_speed[tail])),
            'passed': bool(delay <= 250 and distance <= .03 and stable)}


def torque_cost35(*, full_scene_mean: float, full_scene_integral_s: float,
                  kind: str, commanded_distance_m: float | None,
                  c33_same_direction_integral_s: float | None) -> dict:
    """Mean≤.050 for every case; distance efficiency only for completed pulses."""
    need(kind in ('in_place', 'continuous', 'cancel')
         and math.isfinite(full_scene_mean) and full_scene_mean >= 0.
         and math.isfinite(full_scene_integral_s) and full_scene_integral_s >= 0.,
         'invalid actual normalized full-scene torque-square cost')
    mean_passed = full_scene_mean <= .050
    if kind != 'continuous':
        need(commanded_distance_m is None and c33_same_direction_integral_s is None,
             'zero/cancel has no qualified cost-per-distance denominator')
        return {'full_scene_tau2_mean': full_scene_mean,
                'full_scene_tau2_integral_s': full_scene_integral_s,
                'mean_gate_passed': mean_passed,
                'commanded_distance_cost_denominator_m': None,
                'tau2_integral_per_commanded_meter': None,
                'distance_efficiency_gate': None,
                'passed': mean_passed}
    need(commanded_distance_m is not None
         and c33_same_direction_integral_s is not None
         and math.isfinite(commanded_distance_m) and commanded_distance_m > 0.
         and math.isfinite(c33_same_direction_integral_s)
         and c33_same_direction_integral_s >= 0.,
         'completed continuous pulse lacks C33 paired cost denominator')
    ratio = full_scene_integral_s/commanded_distance_m
    old_ratio = c33_same_direction_integral_s/.03
    efficiency = bool(ratio <= old_ratio)
    return {'full_scene_tau2_mean': full_scene_mean,
            'full_scene_tau2_integral_s': full_scene_integral_s,
            'mean_gate_passed': mean_passed,
            'commanded_distance_cost_denominator_m': commanded_distance_m,
            'tau2_integral_per_commanded_meter': ratio,
            'c33_same_direction_integral_s': c33_same_direction_integral_s,
            'c33_distance_cost_denominator_m': .03,
            'c33_tau2_integral_per_commanded_meter': old_ratio,
            'distance_efficiency_gate': efficiency,
            'passed': bool(mean_passed and efficiency)}
