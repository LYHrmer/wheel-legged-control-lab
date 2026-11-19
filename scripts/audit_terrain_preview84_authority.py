"""Optimistic pitch-authority bound on archived trajectories, without running control."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit(trials: Path, registered: Path) -> dict:
    plan = json.loads(registered.read_text())['paired_trials']
    share = plan['max_extra_share_of_remaining_longitudinal_force']
    threshold = plan['correct_delta_min_rad']
    cases = []
    for terrain in ('flat', 'up', 'down'):
        for seed in plan['seed']:
            case_dir = trials / 'cases' / f'{terrain}_{seed}_off'
            metrics = json.loads((case_dir / 'metrics.json').read_text())
            with (case_dir / 'telemetry.csv').open(newline='') as source:
                rows = list(csv.DictReader(source))
            if len(rows) != metrics['steps']:
                raise ValueError(f'incomplete telemetry in {case_dir.name}')
            capture = metrics['first_front_contact_s']
            if capture is None:
                raise ValueError(f'no time anchor in {case_dir.name}')
            required_force = abs(metrics['pitch_gain_n_per_rad']) * threshold / share
            count = longest = streak = 0
            for row in rows:
                if float(row['control_s']) > capture - plan['control_dt_s'] + 1e-9:
                    break
                possible = float(row['force_headroom_n']) >= required_force
                count += int(possible)
                streak = streak + 1 if possible else 0
                longest = max(streak, longest)
            cases.append({'terrain': terrain, 'seed': seed, 'min_headroom_n': required_force,
                          'eligible_ticks_before_contact': count, 'longest_streak': longest,
                          'eligible_streak': longest >= plan['consecutive_correct_ticks']})
    return {'cases': cases, 'integration_steps': 0,
            'registration_sha256': hashlib.sha256(registered.read_bytes()).hexdigest(),
            'meaning': 'Bound on these recorded states, ignoring geometry, sign and slew.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', required=True, type=Path)
    parser.add_argument('--registered', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.output.open('x') as dst:
        json.dump(audit(args.trials, args.registered), dst, indent=2, allow_nan=False)
        dst.write('\n')
