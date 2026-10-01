"""Saved C31/C30 common-entry pairing and preregistered per-direction gains."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from reader26 import require, saved_document

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W/'continuation30/pair_read30'))
from audit_pairs30 import Evidence, read_case, canonical, byte_equal, ARRAYS


def entry_snapshot31(run, session, construction, episode):
    prefix = hashlib.sha256()
    rows, states = episode['rows'], episode['states']
    require(len(rows) > 200, 'paired actual preparation/entry is missing')
    for row in rows[:200]:
        require(row['actor'] == 'B' and row['policy_predict_called'] is True
                and row['skill30'] is None and not row['macro_events30'], 'preparation is not unchanged B22')
        chosen = {key: row[key] for key in ('input_observation99', 'raw_action16', 'action16',
            'policy_input_action', 'raw_command', 'controller_handoff_receipt', 'side_start_receipt')}
        chosen['info'] = {key: row['info'][key] for key in
            ('controller_record', 'raw_operator_command', 'consumed_command', 'servo_receipt')}
        prefix.update(canonical(chosen)+b'\n')
    original_reset = episode['original_reset31']
    origin = saved_document(Path(run)/'runtime31_module_origins.json')
    loaded = saved_document(Path(run)/'loaded_origins_final.json')
    loaded_paths = set(loaded['module_origins'].values())
    loaded_paths.update(row['defining_source'] for row in loaded['synthetic_module_aliases'].values())
    return dict(initial={key: episode['initial'][key].copy() for key in ARRAYS},
        prefix_arrays={key: states[key][:201].copy() for key in ARRAYS},
        reset_memory=original_reset['control_reset_state'], reset_metadata=original_reset['episode_metadata'],
        prefix_sha256=prefix.hexdigest(), entry=rows[200], first_event=rows[200]['macro_events31'][0],
        compiled=construction['side_kinematic_binding'], profile=construction['side_profile'], origins=origin,
        checkpoint_sha256=session['checkpoint_sha256'],
        loaded_sources={path: session['source_hashes'][path] for path in sorted(loaded_paths)},
        frozen_sources=session['source_hashes'])


def pair_entry31(a, b, *, prior_C30=False):
    checks = {}
    for key in ARRAYS:
        checks['initial_'+key+'_bytes'] = byte_equal(a['initial'][key], b['initial'][key])
        checks['preparation_through_entry_'+key+'_bytes'] = byte_equal(a['prefix_arrays'][key], b['prefix_arrays'][key])
    for key in ('reset_memory', 'prefix_sha256', 'compiled', 'profile', 'checkpoint_sha256'):
        checks[key] = canonical(a[key]) == canonical(b[key])
    # Episode index labels a reset invocation; it does not alter oracle dynamics.
    metadata_a = {k: v for k,v in a['reset_metadata'].items() if k != 'episode_index'}
    metadata_b = {k: v for k,v in b['reset_metadata'].items() if k != 'episode_index'}
    checks['reset_metadata_except_episode_label'] = metadata_a == metadata_b
    common_runtime = set(a['origins']) & set(b['origins'])
    required = {'controller18', 'hybrid_adapter27', 'side_runtime27', 'd1_fast_side_step',
                'd1_side_step', 'state21', 'world_upright_course_11', 'runtime30', 'side_skill30',
                'kinematics24', 'full_drive_command_08', 'engine_binding', 'run_rl16_training_08'}
    checks['unchanged_required_reset_and_control_sources'] = required <= common_runtime and all(
        a['origins'][name] == b['origins'][name] for name in common_runtime)
    common_loaded = set(a['loaded_sources']) & set(b['loaded_sources'])
    checks['old_numerical_loaded_sources_retained'] = set(b['loaded_sources']) <= set(a['frozen_sources']) and all(
        a['frozen_sources'][path] == b['loaded_sources'][path] for path in b['loaded_sources'])
    checks['shared_numerical_loaded_identities'] = all(a['loaded_sources'][p] == b['loaded_sources'][p] for p in common_loaded)
    common_frozen = set(a['frozen_sources']) & set(b['frozen_sources'])
    checks['all_shared_frozen_identities_equal'] = all(a['frozen_sources'][p] == b['frozen_sources'][p] for p in common_frozen)
    ea, eb = a['entry'], b['entry']
    for key in ('input_observation99', 'raw_command', 'side_start_receipt'):
        checks['entry_'+key] = canonical(ea[key]) == canonical(eb[key])
    for key in ('preview_consumed', 'wheel_memory_before'):
        checks['entry_'+key] = canonical(ea['info']['controller_record'][key]) == canonical(eb['info']['controller_record'][key])
    for key in ('raw_operator_command', 'consumed_command', 'servo_receipt'):
        checks['entry_'+key] = canonical(ea['info'][key]) == canonical(eb['info'][key])
    for key in ('teacher_body_from_m', 'teacher_body_to_m', 'teacher_shift_time_s', 'plan_com_height_m', 'initial_body_yaw_rad'):
        checks['teacher_before_action_'+key] = canonical(a['first_event'][key]) == canonical(b['first_event'][key])
    return dict(passed=all(checks.values()), checks=checks, prior_C30_baseline=prior_C30,
        proof_kind='direct_saved_reset_prefix_and_source_derived_unsaved_entry_memory_equivalence',
        direct_evidence='7 initial and 201 prefix arrays; complete reset memory; 200 real B22 action/controller/servo records; entry provider/PI/z/servo; original teacher plan',
        source_derived_scope='unsaved tick-200 context follows unchanged reset/B22 sources and identical full saved preparation',
        all_runtime_or_episode_metadata_claimed_identical=False, ignored_metadata_fields=['episode_index'],
        address_fields_compared=False, post_action_waypoint_compared=False,
        seed_alone_is_not_pairing_evidence=True)


def metrics31(report):
    task = report['task']
    return dict(cycle_s=task['side_duration_s'], goal_error_m=task['final_xy_error_m'],
        retention_ratio=task['retention_ratio'],
        full_torque_square_mean=report['macros']['full_scene_components']['torque_normalized_square_mean'],
        signed_lateral_m=task['signed_lateral_m'])


def gain_gates31(learned, zero, fixed, pair_zero, pair_fixed, qualified):
    retention = learned['retention_ratio']
    gates = dict(actual_entry_paired_with_zero=bool(pair_zero), actual_entry_paired_with_fixed=bool(pair_fixed),
        all_three_full_task_qualified=bool(qualified),
        cycle_le_0p90zero=learned['cycle_s'] <= .90*zero['cycle_s'],
        cycle_le_0p95fixed=learned['cycle_s'] <= .95*fixed['cycle_s'],
        goal_error_le_zero_plus_2mm=learned['goal_error_m'] <= zero['goal_error_m']+.002,
        retention_ge_max_0p90_and_zero_minus_0p02=retention is not None and zero['retention_ratio'] is not None
            and retention >= max(.90, zero['retention_ratio']-.02),
        full_tau2mean_le_1p20zero=learned['full_torque_square_mean'] <= 1.20*zero['full_torque_square_mean'],
        full_tau2mean_le_1p10fixed=learned['full_torque_square_mean'] <= 1.10*fixed['full_torque_square_mean'])
    def ratio(num, den):
        return num/den if den != 0 else None
    return dict(passed=all(gates.values()), gates=gates, learned=learned, zero=zero, fixed=fixed,
        cycle_learned_over_zero=ratio(learned['cycle_s'], zero['cycle_s']),
        cycle_learned_over_fixed=ratio(learned['cycle_s'], fixed['cycle_s']),
        torque_mean_learned_over_zero=ratio(learned['full_torque_square_mean'], zero['full_torque_square_mean']),
        torque_mean_learned_over_fixed=ratio(learned['full_torque_square_mean'], fixed['full_torque_square_mean']),
        torque_cost_is_energy=False)


def comparison31(session, reports, snapshots):
    evidence = Evidence()
    comparisons, pairs, prior = {}, {}, {}
    base = W/'continuation30'
    for direction in ('left', 'right'):
        for mode in ('zero', 'fixed_nonzero'):
            case = mode+'_'+direction
            item = dict(evidence_valid=False, qualification_passed=False, errors=[])
            raw = read_case(base, case, evidence, item)
            require(item['evidence_valid'] and item['qualification_passed'], 'original C30 baseline is not qualified')
            prior[case] = (item, raw)
        name = 'nominal_learned_'+direction
        z, zr = prior['zero_'+direction]
        f, fr = prior['fixed_nonzero_'+direction]
        pairs[name+'_zero'] = pair_entry31(snapshots[name], zr, prior_C30=True)
        pairs[name+'_fixed'] = pair_entry31(snapshots[name], fr, prior_C30=True)
        comparisons['nominal_'+direction] = gain_gates31(metrics31(reports[name]), z['metrics'], f['metrics'],
            pairs[name+'_zero']['passed'], pairs[name+'_fixed']['passed'], reports[name]['qualification_passed'])
        names = {mode: 'heldout_'+direction+'_'+mode for mode in ('learned', 'zero', 'fixed')}
        pairs['heldout_'+direction+'_zero'] = pair_entry31(snapshots[names['learned']], snapshots[names['zero']])
        pairs['heldout_'+direction+'_fixed'] = pair_entry31(snapshots[names['learned']], snapshots[names['fixed']])
        comparisons['heldout_'+direction] = gain_gates31(*(metrics31(reports[names[mode]]) for mode in ('learned', 'zero', 'fixed')),
            pairs['heldout_'+direction+'_zero']['passed'], pairs['heldout_'+direction+'_fixed']['passed'],
            all(reports[names[mode]]['qualification_passed'] for mode in names))
    # Reused evidence must have been bound before this evaluation, not selected afterwards.
    critical_names = {'session.json', 'worker_receipt.json', 'host_receipt.json', 'supervisor_receipt.json',
                      'initial_state.npz', 'states.npz', 'controls.jsonl.gz', 'reset.json', 'macro_transitions30.json'}
    for path, expected in evidence.inputs.items():
        if Path(path).name in critical_names or Path(path).name.startswith('independent_'):
            require(session['source_hashes'].get(path) == expected, 'prior baseline evidence absent from eval frozen closure: '+path)
    cancels = {}
    for direction in ('left', 'right'):
        row = reports['cancel_learned_'+direction]
        delay = row['task']['cancel_to_handoff_controls']
        cancels[direction] = bool(row['qualification_passed'] and row['independent_terminal_kind'] == 'safe_cancel'
            and delay is not None and 0 < delay <= 300)
    gates = dict(all_ten_physical_tasks_qualified=all(r['qualification_passed'] for r in reports.values()),
        all_four_condition_direction_gain_gates=all(c['passed'] for c in comparisons.values()),
        all_actual_entry_pairs=all(p['passed'] for p in pairs.values()), both_learned_cancels_qualified=all(cancels.values()))
    return dict(passed=all(gates.values()), gates=gates, comparisons=comparisons, entry_pairs=pairs,
        learned_cancel_gates=cancels, prior_baseline_input_identities=evidence.inputs,
        nominal_reused_C30_not_new_heldout=True,
        heldout_conditions=[dict(direction=1, initial_yaw_rad=.04), dict(direction=-1, initial_yaw_rad=-.04)],
        deterministic_paired_evidence_not_statistical_independence=True,
        no_direction_averaging_to_hide_failed_gate=True)
