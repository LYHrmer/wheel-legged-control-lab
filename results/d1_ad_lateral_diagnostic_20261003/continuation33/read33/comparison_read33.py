"""Saved C30 baselines, unchanged task gates and fixed-alpha selection."""
from __future__ import annotations

from pathlib import Path

from reader26 import require
from pairing_read31 import metrics31
from pairing_read33 import pair_entry33
from audit_pairs30 import Evidence, read_case

W = Path(__file__).resolve().parents[2]


def prior_baselines33(session):
    evidence = Evidence()
    prior = {}
    for direction in ('left', 'right'):
        for mode in ('zero', 'fixed_nonzero'):
            name = mode+'_'+direction
            key = 'fixed' if mode == 'fixed_nonzero' else 'zero'
            expected_path = W/'continuation30'/f'independent_{name}_01.json'
            require(Path(session['baseline_paths33'][direction][key]).resolve()
                    == expected_path.resolve(),
                    'C33 online baseline source is not the preregistered C30 '+name)
            item = dict(evidence_valid=False, qualification_passed=False, errors=[])
            raw = read_case(W/'continuation30', name, evidence, item)
            require(item['evidence_valid'] and item['qualification_passed'],
                    'C33 reused C30 baseline lacks independent qualification: '+name)
            prior[name] = (item, raw)
    critical = {'session.json', 'worker_receipt.json', 'host_receipt.json',
                'supervisor_receipt.json', 'initial_state.npz', 'states.npz',
                'controls.jsonl.gz', 'reset.json', 'macro_transitions30.json'}
    for path, wanted in evidence.inputs.items():
        if Path(path).name in critical or Path(path).name.startswith('independent_'):
            require(session['source_hashes'].get(path) == wanted,
                    'C33 source closure omitted reused old baseline: '+path)
    return prior, evidence.inputs


def qualification33(report, snapshot, direction, alpha, prior):
    label = 'left' if direction == 1 else 'right'
    zero, zero_snapshot = prior['zero_'+label]
    fixed, fixed_snapshot = prior['fixed_nonzero_'+label]
    pair_zero = pair_entry33(snapshot, zero_snapshot, alpha33=alpha)
    pair_fixed = pair_entry33(snapshot, fixed_snapshot, alpha33=alpha)
    actual = metrics31(report)
    z, f = zero['metrics'], fixed['metrics']
    retention = actual['retention_ratio']
    gates = dict(
        exact_saved_entry_pair_zero=pair_zero['passed'],
        exact_saved_entry_pair_fixed=pair_fixed['passed'],
        inherited_full_physical_task=report['qualification_passed'],
        goal_error_le_zero_plus_2mm=actual['goal_error_m'] <= z['goal_error_m']+.002,
        retention_ge_max_0p90_and_zero_minus_0p02=(retention is not None and
            z['retention_ratio'] is not None and retention >= max(.90, z['retention_ratio']-.02)),
        full_tau2mean_le_1p10fixed=actual['full_torque_square_mean'] <= 1.10*f['full_torque_square_mean'],
        full_tau2mean_le_1p20zero=actual['full_torque_square_mean'] <= 1.20*z['full_torque_square_mean'])
    engineering_speed = dict(cycle_le_0p95oldfixed=actual['cycle_s'] <= .95*f['cycle_s'],
                             cycle_le_0p90oldzero=actual['cycle_s'] <= .90*z['cycle_s'])
    return dict(eligible=all(gates.values()), gates=gates,
                engineering_speed_gates=engineering_speed,
                engineering_speed_passed=all(engineering_speed.values()),
                actual=actual, old_zero=z, old_fixed=f,
                pair_zero=pair_zero, pair_fixed=pair_fixed,
                torque_cost_is_energy=False,
                full_tau2_integral_separate=report['macros']['full_scene_components'][
                    'torque_normalized_square_integral_s'])


def select_alpha33(development):
    """Choose one complete eligible L/R pair by max cycle, mean tau², alpha."""
    pairs = []
    for alpha in (.85, .925, 1.):
        by_direction = development.get(alpha, {})
        if set(by_direction) != {1, -1}:
            continue
        left, right = by_direction[1], by_direction[-1]
        if not left['eligible'] or not right['eligible']:
            continue
        lm, rm = left['actual'], right['actual']
        pairs.append(dict(alpha=alpha,
            worst_cycle_s=max(lm['cycle_s'], rm['cycle_s']),
            mean_full_tau2=(lm['full_torque_square_mean']+rm['full_torque_square_mean'])/2.,
            both_engineering_speed_gates=(left['engineering_speed_passed'] and
                                           right['engineering_speed_passed'])))
    ranked = sorted(pairs, key=lambda row: (row['worst_cycle_s'], row['mean_full_tau2'], row['alpha']))
    return dict(selected_alpha33=ranked[0]['alpha'] if ranked else None,
                eligible_pairs=ranked,
                fixed_engineering_gain=bool(ranked and ranked[0]['both_engineering_speed_gates']),
                RL_speed_benefit_proven=False)
