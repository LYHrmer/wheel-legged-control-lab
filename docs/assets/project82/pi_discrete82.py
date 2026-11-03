"""Stdlib-only algebraic audit of all twelve frozen PI49 trajectories.

No simulator, controller, numerical package, optimization or forward rollout is
imported. Continuous reference inverse dynamics and the recorded implicit-fast
discrete balance are deliberately reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import resource
import signal
import time
from pathlib import Path

ROOT = Path('/home/lyh/wheel-legged-control-lab')
STUDY = 'results/pi49_preflight_20261022_01'
REGISTRATION = 'reports/control_candidates49_preregistration.json'
SEEDS = (490101, 490102, 490103)
DT = 0.002
N = 3000
LOAD_INERTIA = 0.5 * 0.5 * 0.087**2
TOTAL_INERTIA = LOAD_INERTIA + 0.017
DAMPING = 0.105
LIMIT = 2.0
CAP = 2 << 20
ATOL, RTOL = 1e-10, 1e-12
SOURCES = (
    'src/wheel_legged_control/actuator_bench.py',
    'src/wheel_legged_control/wheel_control.py',
    'scripts/evaluate_wheel_control.py',
    'scripts/control_candidates49.py',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(t):
    """Frozen stress command: 18*sin(2*pi*1*t), with analytic derivative."""
    phase = 2 * math.pi * t
    return 18 * math.sin(phase), 36 * math.pi * math.cos(phase)


def friction(v, scenario):
    extra = 0.08 * math.exp(-(v / 0.3)**2) if scenario == 'mismatch' else 0.0
    return (0.043 + extra) * math.tanh(v / 0.05)


def stats(values, times):
    index = max(range(len(values)), key=lambda i: abs(values[i]))
    count = sum(abs(value) > LIMIT for value in values)
    return {
        'rows': len(values), 'abs_gt_2Nm_count': count,
        'abs_gt_2Nm_fraction': count / len(values),
        'peak_abs_Nm': abs(values[index]), 'peak_signed_Nm': values[index],
        'peak_index': index, 'peak_time_s': times[index],
        'minimum_abs_distance_to_2Nm': min(abs(abs(v) - LIMIT) for v in values),
    }


def audit(report, deadline):
    bound = report['inputs_sha256']

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(str(path) not in bound or bound[str(path)] == digest, 'input changed during read')
        bound[str(path)] = digest
        return raw

    def load(relative):
        return json.loads(read(ROOT / relative))

    registration = load(REGISTRATION)
    summary = load(STUDY + '/summary.json')
    manifest = load(STUDY + '/manifest.json')
    require(summary['registration_sha256'] == bound[str(ROOT / REGISTRATION)], 'registration SHA')
    require(manifest['summary.json'] == bound[str(ROOT / STUDY / 'summary.json')], 'summary SHA')
    require(summary['status'] == 'complete' and summary['candidate'] == 'pi'
            and summary['phase'] == 'preflight', 'complete PI preflight required')
    for name in SOURCES:
        read(ROOT / name)
        require(bound[str(ROOT / name)] == registration['source_sha256'][name], 'frozen source: ' + name)
    report['source_verification_before'] = 'COMPLETE'
    report['historical_decision'] = {
        'accepted': summary['decision']['accepted'], 'gates': summary['decision']['gates'],
        'details': summary['decision']['details'],
        'legacy_meaning_literal': summary['decision'].get('meaning'),
        'meaning_note': 'Legacy meaning is phase-only text; accepted=false remains authoritative.',
    }
    episodes = summary['episodes']
    identities = [(e['seed'], e['scenario'], e['profile'], e['arm']) for e in episodes]
    expected = set(itertools.product(SEEDS, ('matched', 'mismatch'), ('stress',), ('baseline', 'candidate')))
    require(len(identities) == 12 and set(identities) == expected, 'complete unique twelve-case grid')
    for ep in episodes:
        deadline()
        require(ep['completed'] is True and ep['steps'] == N and ep['physics_steps'] == N,
                'complete 3000-transition source episode required')
        identity = {key: ep[key] for key in ('seed', 'scenario', 'profile', 'arm')}
        case = '_'.join(str(identity[key]) for key in ('seed', 'scenario', 'profile', 'arm'))
        relative = case + '/trajectory.csv'
        raw = read(ROOT / STUDY / relative)
        require(bound[str(ROOT / STUDY / relative)] == manifest[relative], 'trajectory SHA: ' + case)
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        require(len(rows) == N, 'trajectory row count: ' + case)
        item = dict(identity, status='INCOMPLETE', rows_checked=0, checks=0, max_check_gaps={})
        report['episodes'].append(item)
        continuous, discrete, times, residuals = [], [], [], []

        def number(index, key, rows=rows, case=case):
            value = float(rows[index][key])
            require(math.isfinite(value), f'{case} row {index}: nonfinite {key}')
            return value

        def close(a, b, label, index, item=item, case=case):
            require(math.isfinite(a) and math.isfinite(b), f'{case} row {index}: nonfinite {label}')
            gap = abs(a - b)
            item['checks'] += 1
            maximum = item['max_check_gaps'].get(label)
            if maximum is None or gap > maximum['absolute_gap']:
                item['max_check_gaps'][label] = {'absolute_gap': gap, 'index': index}
            require(gap <= ATOL + RTOL * abs(b),
                    f'{case} row {index}: {label}, gap={gap!r}')

        for k in range(N):
            if k % 128 == 0:
                deadline()
            t = number(k, 'time_s')
            r, a = number(k, 'reference_velocity_rad_s'), number(k, 'reference_acceleration_rad_s2')
            analytic_r, analytic_a = reference(k * DT)
            close(t, k * DT, 'fixed_pretransition_time', k)
            close(r, analytic_r, 'analytic_reference_velocity', k)
            close(a, analytic_a, 'analytic_reference_acceleration', k)
            # Existing rows provide the next reference at k<2999. The final
            # target at t=6.0 is the same fixed analytic command, not a new run.
            rn = number(k + 1, 'reference_velocity_rad_s') if k + 1 < N else reference(N * DT)[0]
            close(rn, reference((k + 1) * DT)[0], 'next_reference_at_fixed_time', k)
            v, vn = number(k, 'actual_velocity_rad_s'), number(k, 'next_velocity_rad_s')
            applied = number(k, 'applied_torque_nm')
            command = number(k, 'command_torque_nm')
            require(abs(command) <= LIMIT and abs(applied) <= LIMIT, 'recorded torque bound')
            if k == 0:
                close(v, 0.0, 'known_initial_velocity', k)
                close(number(k, 'position_rad'), 0.0, 'known_initial_position', k)
            else:
                close(v, number(k - 1, 'next_velocity_rad_s'), 'velocity_pre_next_continuity', k)
                close(number(k, 'position_rad'), number(k - 1, 'next_position_rad'),
                      'position_pre_next_continuity', k)
            close(applied, number(k - 2, 'command_torque_nm') if k >= 2 else 0.0,
                  'actual_two_step_command_delay', k)
            close(r - v, number(k, 'tracking_error_rad_s'), 'tracking_error_is_pretransition', k)
            inverse = TOTAL_INERTIA * (vn - v) / DT + DAMPING * vn + friction(v, ep['scenario'])
            close(inverse, applied, 'implicitfast_recorded_inverse_balance', k)
            residuals.append(inverse - applied)
            continuous.append(TOTAL_INERTIA * a + DAMPING * r + friction(r, ep['scenario']))
            discrete.append(TOTAL_INERTIA * (rn - r) / DT + DAMPING * rn + friction(r, ep['scenario']))
            times.append(t)
            item['rows_checked'] = k + 1
            report['rows_checked'] += 1
        worst = max(range(N), key=lambda i: abs(residuals[i]))
        item.update(
            status='COMPLETE', continuous_reference_inverse=stats(continuous, times),
            discrete_reference_inverse=stats(discrete, times),
            recorded_discrete_balance={
                'max_abs_Nm': abs(residuals[worst]), 'signed_residual_at_max_Nm': residuals[worst],
                'index': worst, 'time_s': times[worst],
            },
            discrete_minus_continuous_max_abs_Nm=max(abs(d - c) for d, c in zip(discrete, continuous, strict=True)),
            last_target={'time_s': N * DT, 'velocity_rad_s': reference(N * DT)[0],
                         'origin': 'fixed analytic stress reference; no physics continuation'},
        )
    require(report['rows_checked'] == 36000, 'complete audit row count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output already exists; no overwrite or resume')
    initial = args.output.with_name(args.output.name + '.initial.json')
    report = {
        'schema': 'pi-discrete82-readback-v1', 'status': 'INCOMPLETE',
        'verification_after': 'NOT_RUN', 'new_physics_steps': 0, 'new_model_calls': 0,
        'new_parameter_updates': 0, 'rows_checked': 0, 'episodes': [], 'inputs_sha256': {},
        'parameters': {'load_inertia_kg_m2': LOAD_INERTIA, 'armature_kg_m2': 0.017,
                       'total_inertia_kg_m2': TOTAL_INERTIA, 'damping_Nm_s_per_rad': DAMPING,
                       'coulomb_amplitude_Nm': 0.043, 'mismatch_extra_stribeck_Nm': 0.08,
                       'friction_smoothing_rad_s': 0.05, 'stribeck_velocity_rad_s': 0.3,
                       'dt_s': DT, 'torque_limit_Nm': LIMIT, 'command_delay_steps': 2},
        'reference': 'r(t)=18*sin(2*pi*t), dr/dt=36*pi*cos(2*pi*t); t=k*0.002',
        'next_reference': 'k<2999: next CSV row; k=2999: fixed analytic reference at t=6.0',
        'continuous_formula': 'M*analytic_reference_acceleration[k]+b*r[k]+friction(r[k])',
        'discrete_formula': 'M*(r[k+1]-r[k])/dt+b*r[k+1]+friction(r[k])',
        'recorded_balance': 'M*(v_next-v_pre)/dt+b*v_next+friction(v_pre)-applied_torque',
        'check_tolerance': {'absolute_Nm_or_native_unit': ATOL, 'relative': RTOL},
        'interpretation_limits': [
            'Known contact-free synthetic rotor; these are not identified hardware parameters.',
            'Continuous instantaneous and discrete sample-to-sample inverse torques are different objects.',
            'Discrete feasibility assumes exact reference at both interval endpoints; it is not an RMSE lower bound.',
            'The reference inverse does not undo the two-step command delay or supply unavailable startup commands.',
            'All cases share the prescribed reference and plant parameters within scenario; repeated counts are not independent experiments.',
            'Torque saturation is not solely inertial feedforward: damping, friction, P and accepted I also contribute.',
            'No controller is changed, no candidate gate is relaxed, and the original PI49 rejection is unchanged.',
        ],
    }
    script = Path(__file__).resolve()
    report['inputs_sha256'][str(script)] = hashlib.sha256(script.read_bytes()).hexdigest()
    with initial.open('x') as stream:
        stream.write(json.dumps({'status': 'INCOMPLETE', 'schema': report['schema'],
                                 'script_sha256': report['inputs_sha256'][str(script)]}) + '\n')
    cpu_start, wall_start = time.process_time(), time.monotonic()

    def stop(signum, _frame):
        raise InterruptedError('readback interrupted: signal ' + str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU, signal.SIGALRM):
        signal.signal(sig, stop)
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30))
    signal.alarm(60)

    def deadline():
        require(time.process_time() - cpu_start < 25, 'cooperative CPU budget')
        require(time.monotonic() - wall_start < 55, 'cooperative wall budget')

    try:
        audit(report, deadline)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - retain interrupted readback prefix.
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            for path, digest in report['inputs_sha256'].items():
                require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,
                        'bound input changed: ' + path)
            report['verification_after'] = 'COMPLETE'
        except BaseException as exc:  # noqa: BLE001 - late failure revokes completion.
            report['status'] = 'INCOMPLETE'
            report['verification_after'] = 'FAILED'
            report['verification_failure'] = type(exc).__name__ + ': ' + str(exc)
        report['cpu_seconds'] = time.process_time() - cpu_start
        report['wall_seconds'] = time.monotonic() - wall_start
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        require(len(text.encode()) + initial.stat().st_size <= CAP, '2MiB combined artifact cap')
        with args.output.open('x') as stream:
            stream.write(text)
        signal.alarm(0)
    print(json.dumps({key: report[key] for key in ('status', 'verification_after', 'rows_checked', 'cpu_seconds')}))
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
