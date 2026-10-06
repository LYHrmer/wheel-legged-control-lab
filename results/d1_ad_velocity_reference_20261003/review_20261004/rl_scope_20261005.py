"""Where can reinforcement learning actually raise the A/D lateral speed?

The four preceding scripts showed that a 3.1x speed gain is available from reference-design
corrections alone, with no learning. Before spending any training budget this script asks
the complementary question: after those corrections, what is left that is genuinely
state-dependent, and how much is it worth?

Three results, all pure arithmetic over the URDF tree and the frozen contract constants:
no MuJoCo, no model construction, no physics, no model load, no training.

  1. The peak body reaction from the swing pair CANNOT be reduced by profile shaping at a
     fixed step and window. In-phase bang-bang already attains the minimum, so a policy
     that only reshapes the swing has no headroom there.
  2. That reaction is fully determined by the planned trajectory, so it is a feedforward
     term, not a learning problem. The frozen body force law has no such term.
  3. What IS state-dependent is the contact-admission wait, and that is where the only
     real headroom lies. This script prices it and derives the RL contribution gate.
"""
from __future__ import annotations

import math

from motion_optimisation_20261005 import (
    ALIGNED_CALF_RAD,
    ALIGNED_THIGH_RAD,
    CONTROL_DT_S,
    LOAD_S,
    REF_QACC_CLIP_RPS2,
    REF_QVEL_CLIP_RPS,
    TARGET_FLOOR_MPS,
    TARGET_GOAL_MPS,
    direction_cost,
    gait,
    jacobian,
    minimum_time_s,
)
from posture_statics_20261004 import (
    LANDING_CAP_M,
    G,
    angle_vector,
    forward,
    parse,
    uniform,
)

SWING_LINKS = ('thigh', 'calf', 'foot')
BODY_KP, BODY_KD = 26.0, 9.0          # contract35.json reference.body
RL_GATE_FRACTION = .20                # main_plan_20261003_ad.md proposed RL contribution gate
SHORT_LIFT_M = .015


def effective_lateral_mass(masses, joints, leg, angles, step=1e-6):
    """Mass seen at the foot laterally: sum over swing links of m_i * d(com_i)/d(y_foot).

    Computed as sum m_i * (dp_i/dq) * (dq/dy_foot), so it is the exact first-order
    momentum transfer per unit foot acceleration for this chain and pose.
    """
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    columns = jacobian(masses, joints, leg, angles)
    rates, _ = _solve_lateral(columns)
    keys = [f'{leg}_{name}_joint' for name in ('hip', 'thigh', 'calf')]
    links = [f'{leg}_{name}' for name in SWING_LINKS]

    def com_world(frames, link):
        rotation, position = frames[link]
        local = masses[link][1]
        return tuple(position[axis]+sum(rotation[axis][k]*local[k] for k in range(3))
                     for axis in range(3))

    total = [0., 0., 0.]
    mass_sum = 0.
    for index, key in enumerate(keys):
        shifted = []
        for sign in (+1, -1):
            perturbed = dict(angles)
            perturbed[key] = perturbed[key]+sign*step
            frames, _, _, _ = forward(masses, joints, perturbed, (0., 0., 0.), identity)
            shifted.append(frames)
        for link in links:
            a, b = com_world(shifted[0], link), com_world(shifted[1], link)
            for axis in range(3):
                total[axis] += (masses[link][0]*(a[axis]-b[axis])/(2*step)*rates[index])
    for link in links:
        mass_sum += masses[link][0]
    return total, mass_sum


def _solve_lateral(columns):
    from motion_optimisation_20261005 import solve3
    return solve3(columns, (0., 1., 0.))


def minimum_peak_acceleration(distance_m, seconds):
    """Smallest achievable max|a| for rest-to-rest travel: symmetric bang-bang."""
    return 4*distance_m/seconds**2


def quintic_peak_acceleration(distance_m, seconds):
    return 10/math.sqrt(3)*distance_m/seconds**2


def main() -> None:
    masses, joints = parse()
    angles = angle_vector(uniform(ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD))
    body_weight_n = sum(mass for mass, _ in masses.values())*G
    columns = jacobian(masses, joints, 'FL', angles)
    worst_lateral, _ = direction_cost(columns, (0., 1., 0.))
    worst_vertical, _ = direction_cost(columns, (0., 0., 1.))
    lateral_speed_cap = REF_QVEL_CLIP_RPS/worst_lateral
    lateral_accel_cap = REF_QACC_CLIP_RPS2/worst_lateral
    vertical_speed_cap = REF_QVEL_CLIP_RPS/worst_vertical
    vertical_accel_cap = REF_QACC_CLIP_RPS2/worst_vertical

    print('== 1. the swing reaction cannot be shaped away ==')
    momentum, swing_mass = effective_lateral_mass(masses, joints, 'FL', angles)
    effective = abs(momentum[1])
    print(f'  swing-link mass (thigh+calf+foot) {swing_mass:.4f} kg')
    print(f'  effective lateral mass at the foot {effective:.4f} kg '
          f'({effective/swing_mass:.4f} of the link mass), cross terms '
          f'({momentum[0]:+.4f}, -, {momentum[2]:+.4f}) kg')
    horizontal_s = minimum_time_s(LANDING_CAP_M, lateral_speed_cap, lateral_accel_cap)
    quantised_s = math.ceil(horizontal_s/CONTROL_DT_S-1e-9)*CONTROL_DT_S
    print(f'  optimised horizontal window {horizontal_s:.4f} s -> quantised {quantised_s:.2f} s '
          f'for a {LANDING_CAP_M*1000:.0f} mm step')
    # Both swing feet must go rest-to-rest over the same window, so their summed COM path is
    # fixed: S(0)=0, S(T)=2D, S'(0)=S'(T)=0. The minimum of max|S''| is 4*(2D)/T^2, and two
    # in-phase bang-bang profiles already achieve it. Staggering cannot beat it.
    single_peak = minimum_peak_acceleration(LANDING_CAP_M, quantised_s)
    summed_peak = minimum_peak_acceleration(2*LANDING_CAP_M, quantised_s)
    print(f'  per-leg minimum peak foot acceleration {single_peak:.4f} m/s^2, '
          f'summed-path minimum {summed_peak:.4f} m/s^2')
    print(f'  two in-phase bang-bang legs give exactly {2*single_peak:.4f} m/s^2 of summed '
          f'acceleration, which equals the summed-path minimum: the bound is attained, so '
          f'no staggering, phase offset or reshaping can lower the peak')
    reaction_n = effective*2*single_peak
    print(f'  peak lateral reaction on the body {reaction_n:.1f} N, '
          f'{reaction_n/body_weight_n*100:.1f}% of the {body_weight_n:.0f} N body weight')
    print(f'  quintic for comparison needs {quintic_peak_acceleration(LANDING_CAP_M, quantised_s):.4f}'
          f' m/s^2 per leg, so the minimum-time profile is also the gentlest available at '
          f'this window')
    print('  therefore the only ways to cut the reaction are a shorter step or a longer '
          'window, both of which are slower: this is a speed/disturbance trade, not '
          'something a policy can optimise away')

    print('\n== 2. the reaction is a feedforward term, not a learning problem ==')
    print('  the frozen body force law is m*(g + a_ref + 26*(p_ref-p) + 9*(v_ref-vCOM));')
    print('  it carries no swing-reaction term, so the reaction enters as an unmodelled')
    print('  disturbance that only the body PD rejects.')
    print('  but the swing plan is known at latch time, so -sum(m_i * a_i,ref) is exactly')
    print(f'  computable: {reaction_n:.1f} N of feedforward, leaving only tracking error')
    for error_fraction in (.05, .10, .20):
        print(f'    if the swing legs track their reference to {error_fraction*100:.0f}%, the '
              f'residual disturbance is {reaction_n*error_fraction:.1f} N, '
              f'{reaction_n*error_fraction/body_weight_n*100:.1f}% of body weight')
    print('  this is a seventh reference-level correction and it is free; it should be in')
    print('  the fixed baseline, not delegated to a policy')

    print('\n== 3. what is actually state-dependent: the contact-admission wait ==')
    config = {'horizontal_s': horizontal_s,
              'lift_s': minimum_time_s(SHORT_LIFT_M, vertical_speed_cap, vertical_accel_cap),
              'lower_s': minimum_time_s(SHORT_LIFT_M, vertical_speed_cap, vertical_accel_cap),
              'overlap_s': LOAD_S}
    print('  the frozen gates that can stall a pair, from contract35 and clarifications:')
    print('    lift admission: both swing wheels whole-shape min Z > 12 mm AND no active')
    print('      terrain contact in all five previous native substeps')
    print('    transfer->lift: at least 0.06 s AND current plus each previous five substeps')
    print('      satisfy the intended stance-pair load gate, hold at most 0.40 s more')
    print('    touchdown: actual positive contact on both returning wheels, then the dwell,')
    print('      with at most 0.60 s of extra allowance')
    print(f'  {"wait/pair":>12} {"speed":>10} {"vs zero wait":>13} {"floor":>7} {"goal":>6}')
    baseline = None
    rows = []
    for controls in (0, 2, 5, 10, 15, 20, 30, 40):
        row = gait(LANDING_CAP_M, **{**config, 'wait_s': controls*CONTROL_DT_S})
        if baseline is None:
            baseline = row['speed_mps']
        rows.append((controls, row['speed_mps']))
        print(f'  {controls:9d} ct {row["speed_mps"]:10.5f} '
              f'{(row["speed_mps"]/baseline-1)*100:+12.1f}% '
              f'{"ok" if row["speed_mps"] >= TARGET_FLOOR_MPS else "FAIL":>7} '
              f'{"ok" if row["speed_mps"] >= TARGET_GOAL_MPS else "FAIL":>6}')
    print('  a policy cannot remove a gate, but it can shape the approach so the gate is')
    print('  satisfied sooner: slower, flatter touchdown for cleaner load onset, and a')
    print('  liftoff that clears 12 mm with less overshoot. The table is the price list.')

    print('\n== 4. the RL contribution gate, computed ==')
    for label, wait_controls in (('ideal fixed baseline, zero wait', 0),
                                 ('plausible fixed baseline, 10 controls', 10),
                                 ('pessimistic fixed baseline, 20 controls', 20)):
        fixed = gait(LANDING_CAP_M,
                     **{**config, 'wait_s': wait_controls*CONTROL_DT_S})['speed_mps']
        gate = fixed*(1+RL_GATE_FRACTION)
        recoverable = baseline
        print(f'  {label}: fixed {fixed:.5f} m/s -> RL must exceed '
              f'{gate:.5f} m/s to clear the {RL_GATE_FRACTION*100:.0f}% gate')
        if wait_controls == 0:
            print('    with zero wait there is no admission headroom left, so the gate '
                  'cannot be cleared by closing waits at all')
        else:
            headroom = recoverable/fixed-1
            verdict = 'reachable' if headroom >= RL_GATE_FRACTION else 'NOT reachable'
            print(f'    closing the wait entirely would give {recoverable:.5f} m/s, '
                  f'{headroom*100:+.1f}%: gate {verdict} by admission alone')

    print('\n== 5. scope for a bounded RL contract ==')
    print('  the policy MAY propose, inside the frozen envelope:')
    print('    - a residual on the landing target inside the unchanged 60 mm ball')
    print('    - a residual on the touchdown approach speed and the lower-phase profile')
    print('    - a residual on the lift profile subject to the unchanged 12 mm clearance')
    print('    - the decision to extend a dwell instead of aborting a marginal pair')
    print('  the policy MAY NOT:')
    print('    - change any contact, load, clearance, pose, torque or no-flight gate')
    print('    - change the integration, the actuator limits or the landing cap')
    print('    - choose the step length or the commanded velocity (those are the operator\'s)')
    print('    - replace the swing-reaction feedforward of section 2, which belongs to the')
    print('      fixed baseline')
    print('  the reward must be speed per qualified pair exchange with the full safety and')
    print('  retention gates as hard terminations, and the comparison must be against the')
    print('  SAME corrected fixed controller, not against the frozen 0.0236 m/s baseline')
    print('  a bounded budget in the established shape, to be frozen by the next contract:')
    controls_per_episode = 200+1200+400
    for episodes, label in ((64, 'smoke'), (512, 'development')):
        print(f'    {label}: {episodes} episodes x {controls_per_episode} controls = '
              f'{episodes*controls_per_episode} controls, '
              f'{episodes*controls_per_episode*5} normal native substeps')
    print('  this is a proposal only: source_go35.json carries training_authorized false '
          'and its single physical attempt is consumed, so no training may start without a '
          'new Astra-signed GO.')


if __name__ == '__main__':
    main()
