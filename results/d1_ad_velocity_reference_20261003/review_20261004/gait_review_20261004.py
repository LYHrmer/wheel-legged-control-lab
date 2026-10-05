"""Reproduce every derived number in docs/ad_lateral_gait_review_20261004.md.

Pure arithmetic over frozen C35 contract parameters, the saved pre-state geometry of
the single C35 run and the URDF link lengths. No MuJoCo, no model load, no physics,
no fitting. Run with plain python3; prints the tables used in the document.
"""
from __future__ import annotations

import math

# --- frozen C35 contract reference parameters (astra_plan/contract35.json) ----------
TRANSFER_S, LIFT_S, LOWER_S, DWELL_S, LOAD_S = .06, .12, .12, .08, .06
OVERHEAD_S = TRANSFER_S + LIFT_S + LOWER_S + DWELL_S + LOAD_S
PEAK_FOOT_MPS = .16
QUINTIC_PEAK_OVER_MEAN = 1.875          # max s'(u) of 10u^3-15u^4+6u^5
QUINTIC_PEAK_ACCEL = 10 / math.sqrt(3)  # max |s''(u)| of the same curve
LANDING_CAP_M = .060
PROBE_DEPTH_M, PROBE_SPEED_MPS = .015, .030
LIFTOFF_EXTRA_S, TOUCHDOWN_EXTRA_S = .4, .6
REFERENCE_ZMP_M = .015
PAIRS_PER_CYCLE = 2                     # phase_transition: strictly serial, no flight
K_S_PER_M = QUINTIC_PEAK_OVER_MEAN / PEAK_FOOT_MPS

# --- measured pre-state of the single C35 side control (case35_00.json) ------------
MASS_KG, G = 48.146865260000006, 9.81
COM_M = (-7.996515049003027, -4.699912570342095, 0.38758374348429603)
CONTACT_M = ((-7.830779743774686, -4.463196398561754),
             (-7.830752426854562, -4.9367834637916825),
             (-8.20556744122596, -4.463397910908332),
             (-8.205603917946572, -4.936580236718218))
# saved raw contact slots, two per wheel (native_contact_failure_0000.json)
RAW_SLOTS_M = {
    0: ((-7.830779743774686, -4.463196398561753), (-7.830791631874375, -4.503195345138564)),
    1: ((-7.830752426854562, -4.9367834637916825), (-7.830739857247978, -4.896784568303424)),
    2: ((-8.20556744122596, -4.463397910908332), (-8.205579297481748, -4.503396649623779)),
    3: ((-8.205603917946572, -4.936580236718218), (-8.205591339771466, -4.896581396826926)),
}

# --- qualified B22 keyboard envelope (docs/b22_gui.md) -----------------------------
KEYBOARD_YAW_RPS = .3

# --- old serial Fast controller, for method validation (main_plan_20261003_ad.md) --
OLD_PER_LEG_OVERHEAD_S = .12 + .104 + .15
OLD_LEGS, OLD_STEP_M = 4, .04


def cycle_speed(step_m, *, extra_s=0., overhead_s=OVERHEAD_S,
                k=K_S_PER_M, pairs=PAIRS_PER_CYCLE):
    """Steady-state lateral speed: foot displacement over the serial gait cycle."""
    horizontal_s = max(.08, k*step_m)
    return step_m/(pairs*(overhead_s+horizontal_s+extra_s))


def required_step_m(speed_mps, *, overhead_s=OVERHEAD_S, k=K_S_PER_M,
                    pairs=PAIRS_PER_CYCLE):
    denominator = 1-pairs*speed_mps*k
    return math.inf if denominator <= 0 else pairs*speed_mps*overhead_s/denominator


def required_peak_foot_mps(speed_mps, step_m=LANDING_CAP_M, *, overhead_s=OVERHEAD_S,
                           pairs=PAIRS_PER_CYCLE):
    available = step_m/(pairs*speed_mps)-overhead_s
    return math.inf if available <= 0 else QUINTIC_PEAK_OVER_MEAN*step_m/available


def perpendicular_unit(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy)
    return (-dy/length, dx/length), length


def diagonal_statics(swing, support):
    """Toppling moment about a two-point support line vs the contact-patch capacity."""
    anchor, far = CONTACT_M[support[0]], CONTACT_M[support[1]]
    unit, length = perpendicular_unit(anchor, far)
    offset = (COM_M[0]-anchor[0])*unit[0]+(COM_M[1]-anchor[1])*unit[1]
    arms = []
    for wheel in support:
        projected = [(point[0]-anchor[0])*unit[0]+(point[1]-anchor[1])*unit[1]
                     for point in RAW_SLOTS_M[wheel]]
        arms.append((max(projected)-min(projected))/2)
    arm = min(arms)
    return {'swing': swing, 'support': support, 'diagonal_length_m': length,
            'com_offset_m': abs(offset), 'required_nm': MASS_KG*G*abs(offset),
            'patch_half_extent_m': arm, 'capacity_nm': MASS_KG*G*arm}


def quintic_duration_s(distance_m, accel_limit_mps2):
    return math.sqrt(QUINTIC_PEAK_ACCEL*distance_m/accel_limit_mps2)


def s_curve(lateral_m, drive_mps, yaw_rps=KEYBOARD_YAW_RPS):
    """Symmetric two-arc lane change: heading restored, forward travel is the cost."""
    radius = drive_mps/yaw_rps
    cosine = 1-lateral_m/(2*radius)
    if cosine < -1:
        return None
    theta = math.acos(cosine)
    seconds = 2*theta/yaw_rps
    return {'radius_m': radius, 'theta_rad': theta, 'seconds': seconds,
            'forward_m': 2*radius*math.sin(theta), 'lateral_rate_mps': lateral_m/seconds}


def break_even_m(step_speed_mps, drive_mps, yaw_rps=KEYBOARD_YAW_RPS):
    """Lateral travel below which stepping beats the S-curve."""
    low, high = 1e-6, 2*(drive_mps/yaw_rps)*.999
    for _ in range(200):
        middle = (low+high)/2
        curve = s_curve(middle, drive_mps, yaw_rps)
        if middle/step_speed_mps < curve['seconds']:
            low = middle
        else:
            high = middle
    return low


def main() -> None:
    print('== 0. method validation against the published old serial bound ==')
    old = OLD_STEP_M/(OLD_LEGS*(OLD_PER_LEG_OVERHEAD_S+K_S_PER_M*OLD_STEP_M))
    print(f'  old serial bound recomputed: {old:.6f} m/s  (published 0.01187 m/s)')

    print('\n== 1. frozen C35 schedule, step at the 60 mm landing cap ==')
    print(f'  overhead per pair {OVERHEAD_S:.2f} s, k = 1.875/0.16 = {K_S_PER_M:.5f} s/m')
    print(f'  contract-allowed extra time per pair: probe <= '
          f'{PROBE_DEPTH_M/PROBE_SPEED_MPS:.2f} s, wait <= '
          f'{LIFTOFF_EXTRA_S+TOUCHDOWN_EXTRA_S:.2f} s')
    for extra in (0., .1, .2, .3569, .5, 1., 1.5):
        print(f'  extra {extra:.4f} s/pair -> {cycle_speed(LANDING_CAP_M, extra_s=extra):.5f} m/s')

    print('\n== 2. structural asymptote and the only first-order lever ==')
    print(f'  v_inf = v_peak/3.75 = {PEAK_FOOT_MPS/3.75:.6f} m/s (independent of step and overhead)')
    for target in (.020, .025, .030, .040):
        step = required_step_m(target)
        flag = 'OK' if step <= LANDING_CAP_M else 'EXCEEDS 60 mm cap'
        print(f'  target {target:.3f} m/s -> step {step*1000:8.2f} mm  {flag}')
    for target in (.020, .025, .030, .040):
        peak = required_peak_foot_mps(target)
        print(f'  target {target:.3f} m/s at 60 mm -> peak foot {peak:.4f} m/s '
              f'({peak/PEAK_FOOT_MPS:.2f}x frozen)')
    clearance_fraction = .012/.035
    low, high = 0., 1.
    for _ in range(200):                      # quintic time to reach 12 mm of a 35 mm lift
        middle = (low+high)/2
        value = 10*middle**3-15*middle**4+6*middle**5
        low, high = (middle, high) if value < clearance_fraction else (low, middle)
    print(f'  quintic lift reaches 12 mm clearance at {low*LIFT_S:.4f} s of the {LIFT_S:.2f} s lift')
    merged = OVERHEAD_S-2*(LIFT_S-low*LIFT_S)
    print(f'  merging lift/lower into one arc -> overhead {merged:.4f} s, '
          f'{cycle_speed(LANDING_CAP_M, overhead_s=merged):.5f} m/s')

    print('\n== 3. two-support statics at the measured pre-state ==')
    centre = [sum(point[axis] for point in CONTACT_M)/4 for axis in (0, 1)]
    print(f'  mass {MASS_KG:.6f} kg, weight {MASS_KG*G:.2f} N, COM height {COM_M[2]:.5f} m')
    print(f'  contact rectangle centre ({centre[0]:.6f}, {centre[1]:.6f})')
    print(f'  COM forward of centre by {(COM_M[0]-centre[0])*1000:.2f} mm, '
          f'lateral {(COM_M[1]-centre[1])*1000:.2f} mm')
    for swing, support in (([0, 3], [1, 2]), ([1, 2], [0, 3])):
        row = diagonal_statics(swing, support)
        print(f'  swing {swing} on support {support} (diagonal {row["diagonal_length_m"]*1000:.1f} mm)')
        print(f'    COM perpendicular offset {row["com_offset_m"]*1000:7.2f} mm'
              f' -> required {row["required_nm"]:6.2f} N*m')
        print(f'    patch half extent        {row["patch_half_extent_m"]*1000:7.2f} mm'
              f' -> capacity {row["capacity_nm"]:6.2f} N*m'
              f'  deficit {row["required_nm"]-row["capacity_nm"]:5.2f} N*m'
              f' ({row["required_nm"]/row["capacity_nm"]:.2f}x)')

    print('\n== 4. time cost of the per-pair body shift that statics requires ==')
    accel_limit = REFERENCE_ZMP_M*G/COM_M[2]
    shift_m = diagonal_statics([0, 3], [1, 2])['com_offset_m']
    shift_s = quintic_duration_s(shift_m, accel_limit)
    print(f'  reference accel limit h*a/g<=0.015 -> {accel_limit:.4f} m/s^2')
    print(f'  quintic shift of {shift_m*1000:.2f} mm needs {shift_s:.4f} s per pair')
    for overlap, label in ((0., 'no overlap'),
                           (LOWER_S+DWELL_S+LOAD_S, 'overlapped with lower+dwell+load')):
        extra = max(0., shift_s-overlap)
        speed = cycle_speed(LANDING_CAP_M, extra_s=extra)
        print(f'  {label:34s} +{extra:.4f} s/pair -> {speed:.5f} m/s')

    print('\n== 5. nominal crouch reach from URDF link lengths ==')
    thigh = calf = .25
    knee = -1.5
    reach = math.sqrt(thigh**2+calf**2+2*thigh*calf*math.cos(knee))
    print(f'  thigh {thigh} m, calf {calf} m, calf joint {knee} rad -> hip-to-wheel {reach:.4f} m')
    print(f'  planar estimate at the 4 rad/s reference clip: {reach*4:.3f} m/s foot speed')
    print(f'  frozen 0.16 m/s uses {PEAK_FOOT_MPS/(reach*4)*100:.1f}% of that estimate')

    print('\n== 6. S-curve lane change at the qualified 0.3 rad/s keyboard yaw ==')
    for lateral in (.05, .10, .15, .30, .50):
        for drive in (.3, 1.2):
            row = s_curve(lateral, drive)
            print(f'  lateral {lateral:.3f} m at {drive:.1f} m/s -> {row["seconds"]:.3f} s, '
                  f'forward {row["forward_m"]:.3f} m, rate {row["lateral_rate_mps"]:.5f} m/s')
    for label, speed in (('optimised gait, see motion_optimisation_20261005', .07317),
                         ('frozen timing ceiling', cycle_speed(LANDING_CAP_M)),
                         ('frozen incl. the 0.975 s landing-lead statics cap', .02365),
                         ('statically sound, overlapped body shift',
                          cycle_speed(LANDING_CAP_M,
                                      extra_s=max(0., shift_s-(LOWER_S+DWELL_S+LOAD_S))))):
        print(f'  stepping ({label}, {speed:.5f} m/s) wins only below '
              f'{break_even_m(speed, .3)*1000:.1f} mm of lateral travel')


if __name__ == '__main__':
    main()
