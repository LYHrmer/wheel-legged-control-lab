"""Pure C34 complete-pair ranking and finite Stage B template selection."""
from __future__ import annotations


def best34(pairs):
    eligible = [pair for pair in pairs if pair.get('eligible') is True
                and pair.get('complete') is True and len(pair.get('case_ids', ())) == 2]
    return min(eligible, key=lambda pair: (
        pair['max_side_duration_s'], pair['mean_full_tau2_mean'], pair['beta']
    )) if eligible else None


def select34(pairs):
    eligible = [pair for pair in pairs if pair.get('eligible') is True
                and pair.get('complete') is True and len(pair.get('case_ids', ())) == 2]
    if not eligible:
        return None
    largest = max(pair['distance_m'] for pair in eligible)
    return best34(pair for pair in eligible if pair['distance_m'] == largest)


def selected_stage_b_cases34(selected_a_beta):
    """Only the nonzero A winner adds a second 40 mm pair after beta0."""
    if selected_a_beta not in (0., .5, 1.):
        raise ValueError('C34 Stage A beta differs')
    if selected_a_beta == 0.:
        return []
    tag = {.5: '050', 1.: '100'}[selected_a_beta]
    return [dict(beta=selected_a_beta, case_id=f'd040_beta{tag}_{side}',
                 direction=direction, distance_m=.04, initial_yaw_rad=0.)
            for side, direction in (('left', 1), ('right', -1))]
