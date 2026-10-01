"""One bounded, saved-only C31/C30 post-evaluation description.

Root executes this only after the independent v2 evaluation closes. No
runtime, model, reader, engine, training or native simulator is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


W = Path(__file__).resolve().parents[2]
P = W / 'continuation31'
C30 = W / 'continuation30'
OUTPUT = P / 'post_eval_saved_diagnostic_01'
BASELINES = ('zero_left', 'zero_right', 'fixed_nonzero_left', 'fixed_nonzero_right')
EVAL_CASES = (
    'nominal_learned_left', 'nominal_learned_right',
    'heldout_left_zero', 'heldout_left_fixed', 'heldout_left_learned',
    'heldout_right_zero', 'heldout_right_fixed', 'heldout_right_learned',
    'cancel_learned_left', 'cancel_learned_right',
)
LIMITS = np.tile(np.array([80., 80., 80., 12.]), 4)
MAX_CONTROLS = 30800
MAX_ENDPOINTS = 30814
MAX_LATCHES = 312


def identity(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


class Inputs:
    def __init__(self) -> None:
        self.used: dict[str, dict] = {}

    def mark(self, path: Path) -> Path:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(W):
            raise ValueError('missing, linked or outside saved input: ' + str(path))
        current = identity(path)
        self.used[str(path)] = current
        if not path.name.endswith('.manifest.json'):
            manifest = path.with_name(path.name+'.manifest.json')
            if manifest.exists():
                proof = json.loads(self.mark(manifest).read_text(encoding='utf-8'))
                if (proof.get('schema') != 'd1-archive-transaction-13-v1' or
                        proof.get('payloads') != [{'file':path.name,**current}]):
                    raise ValueError('saved payload differs from its atomic manifest: '+str(path))
        return path

    def document(self, path: Path) -> dict:
        return json.loads(self.mark(path).read_text(encoding='utf-8'))

    def arrays(self, path: Path) -> dict:
        with np.load(self.mark(path), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    def controls(self, path: Path) -> list[dict]:
        self.mark(path)
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]


def stats(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    if not np.isfinite(array).all():
        raise ValueError('saved diagnostic value is non-finite')
    return {'count': len(array), 'mean': float(array.mean()),
            'std': float(array.std()), 'min': float(array.min()),
            'max': float(array.max())}


def fixed_base(event: dict) -> list[float]:
    start = np.asarray(event['teacher_body_from_m'], dtype=float)
    target = np.asarray(event['teacher_body_to_m'], dtype=float)
    toward = start[:2] - target[:2]
    norm = float(np.linalg.norm(toward))
    world = np.zeros(2) if norm == 0 else min(.008, norm) * toward / norm
    yaw = float(event['initial_body_yaw_rad'])
    c, s = math.cos(yaw), math.sin(yaw)
    return [float(c * world[0] + s * world[1]),
            float(-s * world[0] + c * world[1]), .5]


def latches(events: list[dict], *, source: str) -> list[dict]:
    result = []
    for e in events:
        base = fixed_base(e)
        proposed = [float(x) for x in e['proposed_action']]
        projected = [float(x) for x in e['projected_action']]
        consumed = [float(x) for x in e['consumed_action']]
        if not all(len(v) == 3 and np.isfinite(v).all() for v in
                   (np.asarray(base), np.asarray(proposed),
                    np.asarray(projected), np.asarray(consumed))):
            raise ValueError('incomplete finite saved latch action')
        result.append(dict(source=source, control_index=e['control_index'],
            leg_index=e['leg_index'], macro_index=e['macro_index'],
            fixed_base_action=base, proposed_action=proposed,
            proposed_residual_from_fixed_base=[proposed[j]-base[j] for j in range(3)],
            projected_action=projected, consumed_action=consumed,
            projected_xy_fallback_m=[projected[j]-consumed[j] for j in range(2)],
            reason=e['reason'],
            body_from_m=e['consumed_body_from_m'], body_to_m=e['consumed_body_to_m'],
            lambda_proposed=proposed[2], lambda_consumed=consumed[2]))
    return result


def torque(rows: list[dict], begin: int, end: int) -> dict:
    if begin < 0 or end > len(rows) or begin >= end:
        raise ValueError('invalid torque interval')
    per_control = []
    for row in rows[begin:end]:
        traces = row['native_actuator_traces']
        if len(traces) != 5:
            return {'available': False, 'reason': 'missing five saved applied torque traces'}
        matrix = np.asarray([t['applied_nm'] for t in traces], dtype=float)
        if matrix.shape != (5, 16) or not np.isfinite(matrix).all():
            raise ValueError('saved 5x16 native torque shape/nonfinite differs')
        per_control.append(float(np.mean((matrix / LIMITS) ** 2)))
    return {'available': True, 'controls': end-begin,
            'normalized_torque_square_mean': float(np.mean(per_control)),
            'normalized_torque_square_time_integral_s': float(.01 * sum(per_control)),
            'interpretation': 'squared normalized applied torque, not energy'}


def phase_before_counts(rows: list[dict], side: list[int]) -> Counter:
    """Use the phase entering compute, including the transition-producing tick."""
    return Counter(rows[i]['info']['controller_record']['diagnostic_before']['phase']
                   for i in side)


def task_windows(name: str, task: dict, side: list[int], n: int) -> tuple[int,int]:
    if not side or side != list(range(side[0],side[-1]+1)):
        raise ValueError(name + ': side interval is missing or fragmented')
    begin, end = side[0], side[-1]+1
    if ([begin, end] != task['side_control_window'] or
            task['retention_control_window'] != [end,end+400] or
            n != end+400 or begin != 200):
        raise ValueError(name + ': preparation/side/retention boundary differs')
    return begin,end


def goal_rate_fields(task: dict) -> dict:
    """A safe cancellation does not complete the 30 mm lateral goal."""
    if task['is_cancellation_case']:
        return {'nominal_goal_m':None,'nominal_goal_per_cycle_mps':None,
                'actual_net_per_cycle_mps':None}
    return {'nominal_goal_m':.03,
            'nominal_goal_per_cycle_mps':.03/task['side_duration_s'],
            'actual_net_per_cycle_mps':task['actual_net_per_cycle_mps']}


def analyze_case(name: str, folder: Path, report: dict, inputs: Inputs,
                 *, source: str) -> dict:
    episode = folder / 'episode_0' if source == 'C30' else folder
    rows = inputs.controls(episode / 'controls.jsonl.gz')
    states = inputs.arrays(episode / 'states.npz')
    macros = inputs.document(episode / ('macro_transitions30.json' if source == 'C30'
                                        else 'macro_transitions31.json'))
    if source == 'C30':
        task = report['task']
        native_report = report['reference']
        qualified = report['qualification_passed']
    else:
        task = report['task']
        native_report = report['reference']
        qualified = report['qualification_passed']
    n = len(rows)
    if len(states['qpos']) != n+1 or len(states['qvel']) != n+1:
        raise ValueError(name + ': saved control/endpoint count differs')
    side = [i for i, row in enumerate(rows) if row['actor'] == 'traditional_side']
    begin, end = task_windows(name,task,side,n)
    phases = Counter()
    phase_by_leg = Counter()
    nominal_times = defaultdict(set)
    post_nominal = Counter()
    ready_reasons = Counter()
    overlap = Counter()
    eligibility = defaultdict(lambda: {'first_observed_eligible_control': None,
        'horizontal_start_control': None, 'start_prior_5T_contact_free': None,
        'start_actual_whole_wheel_min_z_m': None})
    for i in side:
        row = rows[i]
        rec = row['info']['controller_record']
        before, after = rec['diagnostic_before'], rec['diagnostic_after']
        phase, leg = before['phase'], before['leg']
        leg_key = 'none' if leg is None else str(leg)
        phases[phase] += 1
        phase_by_leg[(leg_key, phase)] += 1
        nominal = (before['shift_time_s'] if phase in ('shift', 'recenter')
                   else before['nominal_duration_s'])
        if nominal is not None:
            nominal_times[(leg_key, phase)].add(round(float(nominal), 12))
            elapsed = float(before['phase_time_s']) + .01
            if elapsed >= float(nominal) and after['phase'] == phase:
                post_nominal[(leg_key, phase)] += 1
                ready = rec.get('ready_predicates') or {}
                if ready:
                    for key in ('speed_ready', 'margin_ready', 'all_wheels_contact'):
                        if key in ready and ready[key] is False:
                            ready_reasons[(leg_key, phase, key)] += 1
                else:
                    ready_reasons[(leg_key, phase, 'unknown')] += 1
        skill = row.get('skill30')
        if skill is None:
            continue
        active_leg = skill['leg_index']
        if active_leg is None:
            continue
        leg_id = str(active_leg)
        v = np.asarray(skill['foot_reference_velocity_mps'], dtype=float)
        if v.shape != (3,):
            raise ValueError(name + ': foot reference velocity shape differs')
        if np.linalg.norm(v[:2]) > 1e-10 and abs(v[2]) > 1e-10:
            overlap[leg_id] += 1
        gate = skill.get('gate') or {}
        clearance = gate.get('actual_whole_wheel_min_z_m')
        eligible = bool(gate.get('previous_contact_free_5') is True and
                        clearance is not None and float(clearance) > .012)
        if eligible and eligibility[leg_id]['first_observed_eligible_control'] is None:
            eligibility[leg_id]['first_observed_eligible_control'] = i
        if skill['horizontal_start_control_index'] == i:
            eligibility[leg_id].update(horizontal_start_control=i,
                start_prior_5T_contact_free=gate.get('previous_contact_free_5'),
                start_actual_whole_wheel_min_z_m=gate.get('actual_whole_wheel_min_z_m'))
    if phases != phase_before_counts(rows,side) or sum(phases.values()) != end-begin:
        raise ValueError(name + ': compute-before phase counts do not close')
    actual_latches = latches(macros['events'], source=name)
    if not 1 <= len(actual_latches) <= 4:
        raise ValueError(name + ': real latch count differs')
    full = torque(rows, 0, n)
    side_torque = torque(rows, begin, end)
    hold_torque = torque(rows, end, end+400)
    signed_full = report['macros']['full_scene_components']
    if full['available'] and (
            abs(full['normalized_torque_square_mean'] -
                signed_full['torque_normalized_square_mean']) > 2e-9 or
            abs(full['normalized_torque_square_time_integral_s'] -
                signed_full['torque_normalized_square_integral_s']) > 2e-9):
        raise ValueError(name + ': saved traces differ from independent full torque cost')
    by_leg = {}
    for key, count in sorted(phase_by_leg.items()):
        leg, phase = key
        by_leg.setdefault(leg, {})[phase] = dict(controls=count, seconds=.01*count,
            saved_nominal_duration_s=sorted(nominal_times.get(key, ())),
            post_nominal_same_phase_controls=post_nominal[key],
            observed_not_ready={reason: value for (l,p,reason),value in
                ready_reasons.items() if (l,p) == key})
    return dict(case=name, source=source, qualified_in_independent_report=qualified,
        is_cancellation_case=task['is_cancellation_case'],
        preparation_controls=begin, side_control_window=[begin,end],
        side_controls=end-begin, side_seconds=.01*(end-begin),
        side_duration_interpretation=('time to safe cancellation handoff, not completed lateral skill cycle'
            if task['is_cancellation_case'] else 'completed lateral skill cycle'),
        retention_control_window=[end,end+400], retention_controls=400,
        phase_before_compute_controls=dict(sorted(phases.items())),
        phase_by_leg=by_leg, latches=actual_latches,
        post_nominal_interpretation='observed same-phase controls after saved nominal clock; readiness flags may overlap and do not identify a unique causal delay',
        real_foot_xy_z_overlap_controls_by_leg=dict(sorted(overlap.items())),
        observed_prior5T_and_wholewheel_gate_by_leg=dict(eligibility),
        independent_horizontal_native_geometry_passed=
            native_report['horizontal_native_geometry_passed'],
        early_lower_enabled=native_report['early_lower_enabled'],
        signed_net_lateral_m=task['signed_lateral_m'],
        **goal_rate_fields(task),
        retention_ratio=task['retention_ratio'],
        retention_interpretation=('cancellation safety hold, not successful goal retention'
            if task['is_cancellation_case'] else 'completed 30 mm goal retention'),
        retained_signed_lateral_m=(None if task['retention_ratio'] is None else
                                   task['signed_lateral_m']*task['retention_ratio']),
        rollback_m=task['max_rollback_m'], goal_error_m=task['final_xy_error_m'],
        max_absolute_longitudinal_drift_m=task['max_abs_longitudinal_drift_m'],
        yaw_error_rad=task['final_yaw_error_rad'],
        independent_full_torque_square_mean=signed_full['torque_normalized_square_mean'],
        independent_full_torque_square_integral_s=
            signed_full['torque_normalized_square_integral_s'],
        saved_trace_full=full, saved_trace_side=side_torque,
        saved_trace_retention=hold_torque, torque_cost_is_energy=False)


def training_summary(inputs: Inputs, report: dict) -> dict:
    if report['training_valid'] is not True or report['record_valid'] is not True:
        raise ValueError('C31 independent training readback is not valid')
    if len(report['cycles']) != 64:
        raise ValueError('C31 training report lacks 64 independently read cycles')
    train = P/'train_01'
    actions, rewards, overlap = [], [], []
    reasons = Counter()
    by_direction_leg = defaultdict(list)
    reward_by_direction_leg = defaultdict(list)
    for index in range(64):
        episode = train/f'episode_{index}'
        macros = inputs.document(episode/'macro_transitions31.json')
        receipt = inputs.document(episode/'cycle_receipt.json')
        reported = report['cycles'][index]
        if (receipt['cycle_index'] != index or reported['cycle_index'] != index or
                receipt['terminal_kind'] != reported['independent_terminal_kind'] or
                len(macros['events']) != 4):
            raise ValueError('C31 archived training macro/cycle/report identity differs')
        for event in macros['events']:
            action = latches([event], source=f'train_{index}')[0]
            actions.append(action)
            reasons[event['reason']] += 1
            by_direction_leg[(receipt['direction'],event['leg_index'])].append(action)
        spans = reported['reference']['overlap_controls_per_macro']
        overlap.extend(spans.values() if isinstance(spans, dict) else spans)
    if len(actions) != 256:
        raise ValueError('C31 training latch count differs from actual 256')
    value_ev = []
    for batch in range(8):
        document = inputs.document(train/f'batch_{batch:02d}.json')
        rows = document['prepared_batch']['rows']
        if len(rows) != document['prepared_batch']['macro_count']:
            raise ValueError('C31 actual batch rows differ')
        rewards.extend(float(row['reward']) for row in rows)
        for row in rows:
            direction = 1 if row['cycle_index'] % 2 == 0 else -1
            order = [2,0,3,1] if direction == 1 else [3,1,2,0]
            reward_by_direction_leg[(direction,order[row['macro_index']])].append(float(row['reward']))
        target = np.asarray([row['value_target'] for row in rows],dtype=float)
        old = np.asarray([row['old_value'] for row in rows],dtype=float)
        variance = float(np.var(target))
        value_ev.append({'batch_index':batch,'actual_macro_samples':len(rows),
            'pre_update_collection_explained_variance':None if variance<=0 else
                float(1-np.var(target-old)/variance)})
    def summary(group):
        return dict(proposed_xy_residual_mm=[stats(1000*a['proposed_residual_from_fixed_base'][j]
                for a in group) for j in range(2)],
            consumed_xy_mm=[stats(1000*a['consumed_action'][j] for a in group)
                            for j in range(2)],
            lambda_consumed=stats(a['lambda_consumed'] for a in group))
    return {'cycles':64,'real_latches':len(actions),
            'by_direction_leg':{f'{direction}_{leg}':summary(items)
                for (direction,leg),items in sorted(by_direction_leg.items())},
            'all_actions':summary(actions),'projection_reasons':dict(sorted(reasons.items())),
            'actual_macro_reward_by_direction_leg':{
                f'{direction}_{leg}':stats(items) for (direction,leg),items in
                sorted(reward_by_direction_leg.items())},
            'actual_overlap_controls_per_leg_macro':stats(overlap),
            'actual_macro_reward':stats(rewards),
            'pre_update_batch_value_EV':value_ev,
            'value_EV_scope':'different collected batch and pre-update GAE target each time; not a fixed validation curve or final critic metric',
            'training_raw_control_or_native_rescanned':False}


def bind_report(path: Path, execution_path: Path, run: Path,
                review_path: Path, inputs: Inputs) -> tuple[dict, dict]:
    report, execution = inputs.document(path), inputs.document(execution_path)
    session = inputs.document(run/'session.json')
    worker = inputs.document(run/'worker_receipt.json')
    host = inputs.document(run/'host_receipt.json')
    supervisor = inputs.document(run/'supervisor_receipt.json')
    review = inputs.document(review_path)
    entry = Path(review['entry'])
    actual_entry = report['source']['actual_reader_sources']['__main__']
    inputs.mark(entry)
    go_path = Path(session['source_go_path'])
    go = inputs.document(go_path)
    if (execution['exit_code'] != 0 or execution['failure'] is not None
            or execution['source_mismatches'] != []
            or execution['output_identity'] != inputs.used[str(path)]
            or Path(execution['output_path']).resolve() != path.resolve()
            or execution['source_review_identity'] != inputs.used[str(review_path)]
            or execution['owned_child_reaped'] is not True
            or report['run'] != str(run) or report['session_identity'] != inputs.used[str(run/'session.json')]
            or report['worker_identity'] != inputs.used[str(run/'worker_receipt.json')]
            or report['source']['review_identity'] != inputs.used[str(review_path)]
            or report['source']['review_path'] != str(review_path)
            or report['source']['source_GO_identity'] != session['source_go_identity']
            or session['source_go_identity'] != inputs.used[str(go_path)]
            or go['decision'] != 'GO'
            or actual_entry != {'path':str(entry),**inputs.used[str(entry)]}
            or str(entry) not in execution['command']
            or report['source']['passed'] is not True
            or host['exit_code'] != 0 or host['reservation_closed'] is not True
            or host['owned_no_orphans'] is not True or host['failure'] is not None
            or host['source_mismatches'] != []
            or supervisor['exit_code'] != 0 or supervisor['failure'] is not None
            or supervisor['cleanup']['remaining'] != {}
            or worker['result']['status'] != 'complete' or review['decision'] != 'GO'
            or session['arm'] not in ('train','evaluation')):
        raise ValueError('C31 independent report/execution/source/closed host identity differs')
    return report, execution


def write_exclusive(folder: Path, result: dict, markdown: str) -> None:
    if folder != OUTPUT or folder.exists() or folder.is_symlink():
        raise ValueError('output must be the one uncreated C31 post-eval directory')
    folder.mkdir(mode=0o755)
    for name, content in (('diagnostic.json',json.dumps(result,sort_keys=True,
                                     indent=2,allow_nan=False)+'\n'),
                          ('summary.md',markdown)):
        with (folder/name).open('x',encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-readback',type=Path,required=True)
    parser.add_argument('--evaluation-execution',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = Inputs()
    contract = inputs.document(P/'post_eval_saved_diagnostic_contract31.json')
    inputs.mark(P/'post_eval_saved_diagnostic_addendum31_01.md')
    if contract['contract_id'] != 'C31_post_eval_saved_difference_diagnostic_v1':
        raise ValueError('saved diagnostic contract drifted')
    training_report,_ = bind_report(P/'training_readback_02.json',
        P/'training_readback_execution_02/receipt.json',P/'train_01',
        P/'reader_source_review_31_v2.json',inputs)
    eval_report,_ = bind_report(args.evaluation_readback.absolute(),
        args.evaluation_execution.absolute(),P/'eval_01',
        P/'eval_reader_source_review_31_v2.json',inputs)
    if (training_report['training_valid'] is not True or
            eval_report['record_valid'] is not True or
            set(eval_report['cases']) != set(EVAL_CASES)):
        raise ValueError('training/evaluation independent case set is incomplete')
    baseline = {}
    cases = {}
    for name in BASELINES:
        report = inputs.document(C30/f'independent_{name}_01.json')
        if report['qualification_passed'] is not True:
            raise ValueError('C30 baseline was not independently qualified: '+name)
        baseline[name] = analyze_case(name,C30/f'{name}_01',report,inputs,source='C30')
    prior = eval_report['paired_gain_evaluation']['prior_baseline_input_identities']
    for path in (C30/f'independent_{name}_01.json' for name in BASELINES):
        if prior.get(str(path)) != inputs.used[str(path)]:
            raise ValueError('C30 signed baseline identity differs from frozen evaluation pairing')
    for index,name in enumerate(EVAL_CASES):
        report = eval_report['cases'][name]
        cases[name] = analyze_case(name,P/'eval_01'/f'episode_{index}',report,inputs,source='C31')
    actual_controls = sum(x['preparation_controls']+x['side_controls']+400
        for x in (*baseline.values(),*cases.values()))
    actual_latches = sum(len(x['latches']) for x in (*baseline.values(),*cases.values()))
    if (actual_controls > MAX_CONTROLS or actual_controls+14 > MAX_ENDPOINTS or
            actual_latches+256 > MAX_LATCHES):
        raise ValueError('saved diagnostic trajectory/control/latch cap exceeded')
    train = training_summary(inputs,training_report)
    comparison = eval_report['paired_gain_evaluation']
    if set(comparison['comparisons']) != {
            'nominal_left','nominal_right','heldout_left','heldout_right'}:
        raise ValueError('four fixed scientific comparison groups differ')
    branch = ('physical_or_record_failure' if not eval_report['physical_qualification_passed']
              else 'qualified' if eval_report['learned_research_candidate_qualified']
              else 'gain_gate_failed')
    elapsed = time.monotonic()-started
    if elapsed > 900:
        raise TimeoutError('saved-only diagnostic exceeded preregistered 900 s')
    result = {'schema':'d1-c31-post-eval-saved-diagnostic-v1',
        'contract_id':contract['contract_id'],'status':branch,
        'inputs':inputs.used,'training':train,'baseline_cases':baseline,
        'evaluation_cases':cases,'formal_comparison_from_independent_v2':comparison,
        'formal_physical_qualification':eval_report['physical_qualification_passed'],
        'formal_research_candidate_qualification':eval_report['learned_research_candidate_qualified'],
        'actual_streamed_controls':actual_controls,'actual_latches_all_including_training':actual_latches+256,
        'new_model_calls':0,'new_physics_steps':0,'native_payload_rescanned':False,
        'actor_or_value_replayed':False,'torque_cost_is_energy':False,
        'causal_claim':False,'elapsed_s':elapsed,
        'next_research_question':None if branch=='qualified' else
            ('Which saved phase/geometry limit or actually consumed action projection most constrained the four formal gain gates?'
             if branch=='gain_gate_failed' else None),
        'GUI_qualified':False}
    group_lines = ['| Condition/direction | Formal passed | Learned/zero cycle | Learned/fixed cycle |',
                   '|---|---:|---:|---:|']
    for group in ('nominal_left','nominal_right','heldout_left','heldout_right'):
        row = comparison['comparisons'][group]
        group_lines.append(f"| {group} | {row['passed']} | "
                           f"{row['cycle_learned_over_zero']:.6f} | "
                           f"{row['cycle_learned_over_fixed']:.6f} |")
    markdown = ('# C31 saved-only post-evaluation diagnosis\n\n'
        f'Formal branch: **{branch}**. This document quotes the independent v2 gates; '
        'it does not replace them.\n\n'+'\n'.join(group_lines)+'\n\n'
        f'Controls descriptively read: {actual_controls}; genuine latches including training: {actual_latches+256}. '
        'The 64 training cycles contributed macro/cycle/batch records only.\n\n'
        'Phase durations, action residuals, actual overlap, whole-wheel/prior-5T eligibility and torque-square '
        'cost are in `diagnostic.json`. Same-phase post-nominal counts are observed waiting, not a single '
        'speed cause. Torque-square is not energy, and the 400-control retention is a qualification window. '
        'No new model/physics/native payload was run or rescanned.\n')
    write_exclusive(args.output_directory.absolute(),result,markdown)


if __name__=='__main__':
    main()
