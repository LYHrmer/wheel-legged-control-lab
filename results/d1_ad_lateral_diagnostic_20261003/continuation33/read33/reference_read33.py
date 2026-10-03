"""Independent C33 body-reference reconstruction from saved controls only."""
from __future__ import annotations

import math
import numpy as np

from reader26 import require, close
from side_math27 import compiled_side_geometry
from reference_read30 import verify_reference30

R = .05
K = 4. / (1. - 2.*R)
J = K / R
KNOTS = (0., R, .5-R, .5+R, 1.-R, 1.)
JERKS = (J, 0., -J, 0., J)
SHAPE = 'symmetric_trapezoidal_acceleration_r0.05'


def integrated_curve33(elapsed_s, duration_s):
    """Integrate constant jerk over five intervals; independent of runtime hinges."""
    t, duration = float(elapsed_s), float(duration_s)
    require(math.isfinite(t) and math.isfinite(duration) and duration > 0.,
            'C33 invalid saved S clock')
    u = min(1., max(0., t/duration))
    if u == 0.:
        return 0., 0., 0.
    if u == 1.:
        return 1., 0., 0.
    x = v = a = 0.
    for begin, end, jerk in zip(KNOTS[:-1], KNOTS[1:], JERKS):
        step = min(max(u-begin, 0.), end-begin)
        x += v*step + .5*a*step*step + jerk*step**3/6.
        v += a*step + .5*jerk*step*step
        a += jerk*step
        if u <= end:
            break
    return x, v/duration, a/(duration*duration)


def planned_duration33(span, height, alpha):
    span = np.asarray(span, dtype=float)
    require(span.shape == (3,) and np.isfinite(span).all()
            and math.isfinite(float(height)) and height > 0.
            and float(alpha) in (.85, .925, 1.), 'C33 invalid planned body span/height/alpha')
    return max(.30, math.sqrt(K*np.linalg.norm(span[:2])*height/(9.81*.015*alpha)))


def verify_reference33(episode, events, kin, binding, case_id, native_minima,
                       native_contacts, *, alpha33):
    # C30 still verifies every foot reference, contact/shape permission and
    # actual post-IK target. It has no body timing polynomial to override.
    original = verify_reference30(episode, events, kin, binding, case_id,
                                  native_minima, native_contacts)
    body_ticks = recenter_ticks = 0
    planned_peaks = []
    rows, states = episode['rows'], episode['states']
    for i, row in enumerate(rows):
        if row['actor'] != 'traditional_side':
            continue
        rec, skill = row['info']['controller_record'], row['skill30']
        before, after = rec['diagnostic_before'], rec['diagnostic_after']
        metadata = skill['body_curve33']
        require(metadata['shape'] == SHAPE
                and close(metadata['ramp_fraction_of_total_duration'], R)
                and close(metadata['peak_normalized_acceleration'], K)
                and metadata['alpha'] == alpha33
                and close(metadata['effective_zmp_budget_m'], .015*alpha33),
                'C33 shape or budget metadata differs from sealed case alpha')
        source = before['phase']
        tick = metadata['reference_tick']
        if source in ('shift', 'recenter'):
            require(tick is not None and tick['source_phase'] == source,
                    'C33 body phase omitted its consumed reference tick')
            origin = np.asarray(before['body_from'], dtype=float)
            span = np.asarray(before['body_to'], dtype=float)-origin
            duration = float(before['shift_time_s'])
            elapsed = float(before['phase_time_s'])+.01
            q, v, a = integrated_curve33(elapsed, duration)
            require(close(tick['body_from_m'], origin)
                    and close(tick['body_span_m'], span)
                    and close(tick['duration_s'], duration)
                    and close(tick['elapsed_s'], elapsed)
                    and close(tick['position_fraction'], q)
                    and close(tick['acceleration_fraction_per_s2'], a),
                    'C33 tick origin/span/clock or integrated S shape differs')
            expected_velocity = span*v if after['phase'] == source else np.zeros(3)
            require(close(skill['body_reference_velocity_mps'], expected_velocity)
                    and close(skill['body_reference_position_m'], after['pose'])
                    and close(skill['body_reference_accel_mps2'], after['body_accel']),
                    'C33 body velocity/state telemetry differs from consumed side reference')
            if after['phase'] == source:
                require(close(after['pose'], origin+span*q)
                        and close(after['body_accel'], span*a),
                        'C33 live-phase pose/acceleration differs from integrated S')
            elif after['phase'] in ('unload', 'done') and after['failure'] is None:
                require(elapsed >= duration and q == 1.
                        and close(after['pose'], origin+span)
                        and close(after['body_accel'], [0., 0., 0.]),
                        'C33 successful body phase exit missed its zero-derivative endpoint')
            else:
                require(after['failure'] is not None
                        and close(after['body_accel'], [0., 0., 0.]),
                        'C33 abnormal body exit bypassed old abort acceleration reset')
            body_ticks += 1
            recenter_ticks += int(source == 'recenter')
        else:
            require(tick is None and close(skill['body_reference_velocity_mps'], [0., 0., 0.]),
                    'C33 non-body phase reported an S reference tick/velocity')
            if source == 'load' and after['phase'] == 'recenter':
                g = compiled_side_geometry(kin, states['qpos'][i], states['qvel'][i],
                                           binding['wheel_map'])
                height = float(np.clip(g['whole_com_m'][2]-g['contact_points_m'][:, 2].min(), .10, 1.))
                span = np.asarray(after['body_to'])-np.asarray(after['body_from'])
                duration = planned_duration33(span, height, alpha33)
                require(close(after['body_from'], before['pose'])
                        and close(after['pose'], before['pose'])
                        and close(after['shift_time_s'], duration)
                        and close(after['body_accel'], [0., 0., 0.]),
                        'C33 recenter entry differs from old load gate or S duration')
                planned_peaks.append(K*np.linalg.norm(span[:2])*height/(9.81*duration**2))
    for event in events:
        span = np.asarray(event['consumed_body_to_m'])-np.asarray(event['consumed_body_from_m'])
        height = float(event['plan_com_height_m'])
        duration = planned_duration33(span, height, alpha33)
        require(close(event['consumed_shift_time_s'], duration),
                'C33 latch planned shift exceeds or mismatches its S duration')
        planned_peaks.append(K*np.linalg.norm(span[:2])*height/(9.81*duration**2))
    terminal = episode['cycle_receipt31']['terminal_kind']
    require(terminal in ('success', 'safe_cancel', 'controlled_failure')
            and body_ticks > 0
            and (terminal != 'success' or recenter_ticks > 0)
            and all(peak <= .015*alpha33+1e-12 for peak in planned_peaks),
            'C33 omitted required body/recenter phase or exceeded planned ZMP budget')
    return dict(original, body_curve33_verified=True, body_curve33_body_ticks=body_ticks,
                body_curve33_recenter_ticks=recenter_ticks,
                body_curve33_max_planned_zmp_m=max(planned_peaks),
                body_curve33_budget_m=.015*alpha33)
