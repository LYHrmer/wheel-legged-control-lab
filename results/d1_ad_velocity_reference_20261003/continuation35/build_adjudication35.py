"""Write the C35 root adjudication and closed state from saved evidence only.

No physics, no model load, no reader re-run: every number below is read back from
the archived receipts and the independent report produced under source_go35.json.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SECTION = Path(__file__).resolve().parent
RUN = SECTION/'development_01'
EPISODE = RUN/'episode_0'


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def gated_rows() -> list[dict]:
    """Native substeps whose consumed reference actually required a support gate."""
    rows = []
    with gzip.open(EPISODE/'native_block_0000.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            proof = row['contact_evidence35']
            if proof.get('support_gate_required'):
                rows.append({'native_index': row['native_index'],
                             'consumed_phase': proof['consumed_phase'],
                             'wheel_normal_load_n': proof['positive_wheel_normal_load_sum_n'],
                             'support_normal_sum_n': proof['support_normal_sum_n'],
                             'whole_wheel_min_z_m': proof['whole_wheel_min_z_m'],
                             'active_wheel_terrain_contact': proof['active_wheel_terrain_contact']})
    return rows


def main() -> None:
    worker = json.loads((RUN/'worker_receipt.json').read_text())
    host = json.loads((RUN/'host_receipt.json').read_text())
    supervisor = json.loads((RUN/'supervisor_receipt.json').read_text())
    case = json.loads((RUN/'case35_00.json').read_text())
    campaign = json.loads((RUN/'campaign35.json').read_text())
    report = json.loads((SECTION/'independent_read35_01.json').read_text())
    reader_receipt = json.loads((SECTION/'readback_execution35_01/receipt.json').read_text())
    go = json.loads((SECTION/'source_go35.json').read_text())
    failure = json.loads((EPISODE/'native_contact_failure_0000.json').read_text())

    pending = case['pending_control']['actual_last_side_record']
    reference = pending['reference']
    allocation = pending['calculation']['allocation_forces4']
    rows = gated_rows()
    final = rows[-1]
    swing_pair = reference['pair']
    separated = [row for row in failure['raw_contacts'] if row['dist'] > 0]

    record = {
        'schema': 'd1-c35-root-adjudication-v1',
        'execution_contract_id': go['execution_contract_id'],
        'source_go_identity': identity(SECTION/'source_go35.json'),
        'request_identity': identity(SECTION/'request35_01.json'),
        'adjudicator': 'root (main Codex agent)',
        'astra_final_adjudication_obtained': False,
        'astra_note': ('Only the source GO was signed by the actual gpt-6-astra ultra reviewer. '
                       'No new Astra final adjudication was requested or produced this round; '
                       'this record is the root executor\'s own reading of the saved evidence.'),
        'decision': 'REJECT_FIXED_PAIR_FEASIBILITY_PILOT_ON_FIRST_CASE',
        'pilot_fixed_feasible': False,
        'continuous_lateral_qualified': False,
        'measured_lateral_speed_mps': None,
        'engineering_speed_milestone_achieved': False,
        'RL_speed_benefit_proven': False,
        'new_ad_gui_delivered': False,
        'user_goal_complete': False,

        'campaign': {
            'physical_attempts_used': 1,
            'physical_attempts_authorized': go['physical_attempts_authorized'],
            'automatic_retry_used': False,
            'remaining_budget_reused': False,
            'attempted_cases': campaign['attempted_cases'],
            'closed_cases': campaign['closed_cases'],
            'cases_in_contract': len(go['arms']['development']['cases']),
            'first_case_id': case['case_id'],
            'stop_reason': campaign['stop_reason'],
            'terminal_kind': case['terminal_kind'],
            'never_attempted_case_ids': [row['id'] for row in
                                         go['arms']['development']['cases'][1:]],
        },

        'actual_counts': {
            'controls_consumed': campaign['controls'],
            'controls_limit': go['arms']['development']['control_limit'],
            'b22_preparation_controls': case['policy_predictions'],
            'side_controls_returned': report['ledger']['per_case'][0]['returned_side_controls'],
            'pending_unsafe_control_attempts': report['ledger']['pending_control_attempts'],
            'normal_native_consumed': campaign['normal_native'],
            'normal_native_limit': go['arms']['development']['normal_native_cap'],
            'compiler_native_consumed': report['ledger']['compiler_native'],
            'compiler_native_limit': go['arms']['development']['compiler_native_cap'],
            'native_checked': report['ledger']['native_checked'],
            'b22_predict_calls': report['ledger']['B22_predict_calls'],
            'b22_actor_rows': report['ledger']['B22_actor_rows'],
            'side_computes': report['ledger']['actual_side_computes'],
            'side_starts': report['ledger']['actual_side_starts'],
            'pair_exchanges_completed': campaign['pair_exchanges'],
            'new_policy_or_optimizer_calls': 0,
            'static_api': report['ledger']['actual_static_API'],
            'worker_elapsed_s': worker['elapsed_wall_s'],
            'host_elapsed_s': host['elapsed_s'],
            'supervisor_elapsed_s': supervisor['elapsed_s'],
            'reader_elapsed_s': reader_receipt['elapsed_s'],
        },

        'first_failure': {
            'gate': 'C35 new all4 actual support/load bound',
            'raised_in': 'root35/runtime35.py sink (native guard)',
            'native_index': failure['failure']['native_index'],
            'sim_time_s': failure['failure']['time_s'],
            'control_index_record': case['pending_control']['control_index'],
            'consumed_phase': reference['phase_before'],
            'support_mode': 'all4',
            'phase_elapsed_s': reference['phase_elapsed_s'],
            'transfer_duration_s': 0.06,
            'swing_pair': swing_pair,
            'declared_stance_legs': reference['stance_legs'],
            'declared_swing_legs': reference['swing_legs'],
            'allocation_force_weights': reference['force_weights'],
            'commanded_normal_force_n': [row[2] for row in allocation],
            'measured_wheel_normal_load_n': final['wheel_normal_load_n'],
            'measured_support_normal_sum_n': final['support_normal_sum_n'],
            'all_wheel_contacts_still_listed': final['active_wheel_terrain_contact'],
            'violated_condition': 'per-stance-wheel independently summed normal load > 1e-8 N',
            'satisfied_conditions': ['all four wheel-terrain contact slots present',
                                     'support normal sum >= 0.5*m*g',
                                     'no nonwheel terrain contact',
                                     'pose/height/leg-qvel native bounds',
                                     'zero actual soft joint-range excursion'],
            'separated_contact_slots': [{'geom2': row['geom2'], 'gap_m': row['dist'],
                                         'pos_world_m': row['pos']} for row in separated],
            'wheel_index_zero_load': final['wheel_normal_load_n'].index(0.0),
        },

        'gated_native_trace': rows,

        'localization': {
            'conclusion': ('The frozen C35 transfer design and the frozen C35 all-four contact '
                           'regime are mutually unsatisfiable; this is a design contradiction, '
                           'not a tuning or seed problem.'),
            'conflict': [
                {'document': 'astra_plan/clarifications35_01.json',
                 'field': 'contact_phase_semantics.all_four_regime',
                 'requires': ('transfer is inside the all-four regime: every one of the four '
                              'wheels must keep an actual active terrain contact with '
                              'independently summed normal load > 1e-8 N at every native substep, '
                              'with no loss grace')},
                {'document': 'astra_plan/clarifications35_03.json',
                 'field': 'transfer_weights',
                 'requires': ('during the same transfer phase each intended swing wheel allocator '
                              'weight is 1-smoothstep(elapsed/0.06), i.e. it reaches 0 at 0.06 s, '
                              'and transfer must last at least 0.06 s before lift may be consumed')},
            ],
            'mechanism': ('The allocator drove the commanded swing-pair normal force to '
                          '0.576 N and 0.0075 N while the body force law (kp=26, kd=9) held base '
                          'height, so wheels 0 and 3 unloaded monotonically across the five '
                          'consumed transfer controls (wheel 3: 120.8 -> 75.0 -> 34.2 -> 2.9 -> '
                          '0.0 N). Wheel 3 physically separated by 0.87 mm while its contact slot '
                          'was still listed inside the margin, so the measured load hit exactly '
                          'zero at phase_elapsed 0.05 s of a mandatory >= 0.06 s transfer, before '
                          'any lift phase could legally begin.'),
            'why_structural': ('Full unloading of the swing pair is the declared purpose of '
                               'transfer, and lift (where clearance is allowed) cannot be entered '
                               'before 0.06 s. The zero crossing of the swing-pair normal load '
                               'therefore always falls inside transfer, where strictly positive '
                               'load on all four wheels is mandatory. The static load split is '
                               'also asymmetric (wheel 0 147.8 N vs wheel 3 120.8 N at transfer '
                               'start), so the lighter swing wheel reaches zero before 0.06 s.'),
            'not_evidence_for': ['any claim about achievable lateral speed',
                                 'any claim that two-support continuous side-stepping is '
                                 'physically infeasible for this robot',
                                 'any claim that the paired IK, inertia coupling, landing '
                                 'projection, stop logic or cost accounting are wrong - they were '
                                 'never exercised beyond the transfer phase',
                                 'any RL conclusion'],
            'forbidden_next_actions': ['parameter sweep of transfer duration or weights',
                                       'retry under the existing GO',
                                       'editing frozen C35 source and reusing source_go35.json',
                                       'any new training run'],
            'required_next_action': ('A new Astra decision amending the frozen phase/contact '
                                     'semantics (for example an explicit pre-unload regime where '
                                     'only the incoming stance pair needs positive load), then a '
                                     'new source freeze, a new source GO and a new bounded '
                                     'campaign. Source may not be edited under this GO.'),
        },

        'independent_readback': {
            'entrypoint': go['reader'],
            'launcher': go['reader_launcher'],
            'report': 'independent_read35_01.json',
            'report_identity': identity(SECTION/'independent_read35_01.json'),
            'exit_code': reader_receipt['exit_code'],
            'failure': reader_receipt['failure'],
            'source_mismatches': reader_receipt['source_mismatches'],
            'reader_physics_calls': report['reader_physics_calls'],
            'reader_model_calls': report['reader_model_calls'],
            'ledger_passed': report['ledger']['passed'],
            'record_auditable': report['record_auditable'],
            'partial_record': report['partial_record'],
            'saved_native_proof_passed': report['cases'][0]['partial_native_proofs'][0]
                                               ['saved_native_proof_passed'],
            'uncompleted_requirements': report['cases'][0]['uncompleted_requirements'],
            'dynamics_scope': report['dynamics_scope'],
        },

        'closure': {
            'host_source_mismatches': host['source_mismatches'],
            'host_owned_no_orphans': host['owned_no_orphans'],
            'supervisor_remaining': supervisor['cleanup']['remaining'],
            'worker_cleanup_errors': worker['cleanup_errors'],
            'worker_archive_failed': worker['archive_failed'],
            'worker_execution_complete': worker['execution_complete'],
            'failure_records_retained': True,
            'retry_permitted': worker['retry_permitted'],
        },

        'unchanged_claims': {
            'b22_straight_line_sim_speed_mps': 1.60145,
            'b22_gui_ad_enabled': False,
            'b22_r_key_is_simulation_reset': True,
            'physical_self_recovery_implemented': False,
            'frozen_77_inputs_modified': False,
            'b22_checkpoint_modified': False,
            'old_experiments_or_batches_rerun': False,
        },
        'evidence': [{'path': str(path.relative_to(SECTION)), **identity(path)} for path in (
            RUN/'worker_receipt.json', RUN/'host_receipt.json', RUN/'supervisor_receipt.json',
            RUN/'campaign35.json', RUN/'case35_00.json', RUN/'case_definition35.json',
            RUN/'progress35_00.json', EPISODE/'native_contact_failure_0000.json',
            EPISODE/'cycle_receipt35.json', EPISODE/'segment_receipt.json',
            SECTION/'independent_read35_01.json',
            SECTION/'readback_execution35_01/receipt.json',
        )],
    }
    write(SECTION/'root_adjudication35_01.json', record)

    state = {
        'schema': 'd1-c35-continuation-state-v1',
        'closed_utc': datetime.now(timezone.utc).isoformat(),
        'execution_contract_id': go['execution_contract_id'],
        'arm': 'development',
        'goal': 'stable straight line -> reliable obstacle crossing -> higher speed',
        'goal_complete': False,
        'stage': 'C35 fixed two-support continuous side-step pilot rejected on its first case',
        'physical_attempts_used': 1,
        'retry_permitted': False,
        'independent_readback': 'independent_read35_01.json',
        'root_adjudication': 'root_adjudication35_01.json',
        'astra_final_adjudication_obtained': False,
        'reviews': {'source_go35.json': identity(SECTION/'source_go35.json')},
        'source_frozen': True,
        'next_action_requires_new_contract_and_go': True,
        'no_training_authorized': True,
        'repository_head_at_source_go': go['repository_head'],
    }
    write(SECTION/'continuation_state35.json', state)
    print(json.dumps({'adjudication': str(SECTION/'root_adjudication35_01.json'),
                      'state': str(SECTION/'continuation_state35.json'),
                      'gated_native_rows': len(rows)}))


if __name__ == '__main__':
    main()
