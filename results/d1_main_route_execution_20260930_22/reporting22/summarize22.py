"""Independently aggregate closed C22 saved results under the frozen claim gates.

Standard-library JSON/hash arithmetic only. Does not import a scorer, load a
checkpoint, execute a servo, simulate, select another model, or change defaults.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

C = Path(__file__).resolve().parent.parent
CONTRACT = 'C22_fixed_B_eval_repair_v1'
ACTORS = ('zero','old','A','B')
SIX = ('flat_0p6','flat_1p6','flat_1p2_yaw','bumps_0p4','rough_0p35','ramp_0p45_complete')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing regular summary input: '+str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            size += len(block)
            digest.update(block)
    return {'sha256':digest.hexdigest(),'bytes':size}


def near(a,b):
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)


def ratio(numerator,denominator):
    require(all(math.isfinite(x) and x >= 0 for x in (numerator,denominator)), 'invalid ratio operands')
    return {'numerator':numerator,'denominator':denominator,
            'defined':denominator>0,'ratio':numerator/denominator if denominator>0 else None}


def under(value,bound):
    n,d = value['numerator'],value['denominator']
    return n/d <= bound if d>0 else n==0 and bound>=1.


def checked_ratio(n,d,saved):
    value = ratio(n,d)
    require(near(n,saved['numerator']) and near(d,saved['denominator'])
            and value['defined'] == saved['defined']
            and ((value['ratio'] is None and saved['ratio'] is None)
                 or (value['ratio'] is not None and saved['ratio'] is not None
                     and near(value['ratio'],saved['ratio']))), 'saved comparison differs from independent arithmetic')
    return value


def score_table(rows):
    table = {}
    for row in rows:
        key = (row['case_id'],row['experiment_actor'])
        require(key not in table and key[1] in ACTORS,'duplicate or unknown case/actor')
        table[key] = row
    return table


def complete_safe(row):
    return all((row['record']['record_valid'],row['record']['full_horizon'],
                row['safety']['passed'],row['rl_terms']['drive_complete'],
                row['hold_terms']['drive_complete']))


def pool(readback,case_specs):
    expected = [row['case_id'] for row in case_specs]
    require(len(expected)==len(set(expected))==4, 'new split is not exactly four fixed cases')
    table = score_table([r for r in readback['numeric_scores'] if r['case_id'] in expected])
    require(set(table)=={(name,actor) for name in expected for actor in ACTORS}, 'new split missing prelisted records')
    valid = all(complete_safe(row) for row in table.values())
    saved = readback['equal_case_pool']
    require(valid is saved['complete'], 'saved equal-case completeness differs')
    diagnostics, actors = {}, {}
    for actor in ACTORS:
        diagnostics[actor] = {}
        for name in expected:
            row = table[name,actor]
            drive,hold = row['rl_terms'],row['hold_terms']
            count = drive['drive_completed_ticks']-hold['drive_completed_ticks']
            sse = drive['sse_total']-hold['sse_total']
            require(count >= 0 and sse >= -1e-12, 'hold is not a drive subset')
            diagnostics[actor][name] = {
                'drive_error_mean':drive['sse_total']/drive['drive_completed_ticks'] if drive['drive_completed_ticks'] else None,
                'hold_error_mean':hold['sse_total']/hold['drive_completed_ticks'] if hold['drive_completed_ticks'] else None,
                'outside_hold_transition_diagnostic':{'sse':sse,'ticks':count,'error_mean':sse/count if count else None},
                'complete_safe':complete_safe(row),'task_passed':row['task_passed']}
    if not valid:
        return {'complete':False,'actors':{},'comparisons':{},'family_B_over_A':{},
                'case_diagnostics':diagnostics,'failed_record_keys':[list(k) for k,r in table.items() if not complete_safe(r)]}
    families = {family:[row['case_id'] for row in case_specs if row['terrain']==terrain]
                for family,terrain in (('yaw','flat'),('bumps','bumps'))}
    require(all(len(names)==2 for names in families.values()),'family weights changed')
    for actor in ACTORS:
        entries = [table[name,actor] for name in expected]
        data = {'equal_case_error_mean':sum(r['rl_terms']['sse_total']/r['rl_terms']['drive_completed_ticks'] for r in entries)/4,
            'equal_case_torque_cost_mean':sum(r['rl_terms']['torque_cost_mean'] for r in entries)/4,
            'all_tasks_passed':all(r['task_passed'] for r in entries),'families':{}}
        for family,names in families.items():
            data['families'][family] = {metric:sum(diagnostics[actor][name][metric] for name in names)/2
                for metric in ('drive_error_mean','hold_error_mean')}
        require(near(data['equal_case_error_mean'],saved['actors'][actor]['equal_case_error_mean'])
                and near(data['equal_case_torque_cost_mean'],saved['actors'][actor]['equal_case_torque_cost_mean'])
                and data['all_tasks_passed'] is saved['actors'][actor]['all_tasks_passed'], 'saved equal-case actor aggregate differs')
        actors[actor] = data
    comparisons = {'B/'+a:{'error':checked_ratio(actors['B']['equal_case_error_mean'],actors[a]['equal_case_error_mean'],saved['comparisons']['B/'+a]['error']),
        'cost':checked_ratio(actors['B']['equal_case_torque_cost_mean'],actors[a]['equal_case_torque_cost_mean'],saved['comparisons']['B/'+a]['cost'])} for a in ('A','old','zero')}
    family = {f:{metric:checked_ratio(actors['B']['families'][f][metric],actors['A']['families'][f][metric],saved['family_B_over_A'][f][metric])
        for metric in ('drive_error_mean','hold_error_mean')} for f in families}
    return {'complete':True,'actors':actors,'comparisons':comparisons,'family_B_over_A':family,'case_diagnostics':diagnostics,
            'case_weight':.25,'family_case_weight':.5}


def original_rl(actor,readback,regression_table):
    """Recompute the frozen six-pair prerequisite/cost/conversion/SSE gates."""
    saved = readback['six_task_zero_pairs'][actor]
    pairs = saved['pairs']
    require(len(pairs)==6 and {p['case_id'] for p in pairs}==set(SIX), 'original six pairing differs')
    prerequisite = True
    conversions,regressions,common,no_worse = [],[],[],[]
    for p in pairs:
        z,r = regression_table[p['case_id'],'zero'],regression_table[p['case_id'],actor]
        conversion = not z['task_passed'] and r['task_passed']
        regression = z['task_passed'] and not r['task_passed']
        both = z['task_passed'] and r['task_passed']
        require(p['task_conversion'] is conversion and p['capability_regression'] is regression
                and p['common_task_passing'] is both
                and near(p['sse_policy'],r['rl_terms']['sse_total'])
                and near(p['sse_zero'],z['rl_terms']['sse_total']), 'original pair task/SSE differs')
        prerequisite &= z['record']['record_valid'] and r['record']['record_valid'] and not regression and r['record']['full_horizon'] and r['safety']['passed']
        if conversion: conversions.append(p['case_id'])
        if regression: regressions.append(p['case_id'])
        if both:
            common.append(p)
            if p['sse_policy'] <= 1.02*p['sse_zero']:
                no_worse.append(p['case_id'])
    ticks = sum(p['common_drive_ticks'] for p in pairs)
    n = sum(p['common_cost_sum_policy'] for p in pairs)
    d = sum(p['common_cost_sum_zero'] for p in pairs)
    cost_ratio = ratio(n/ticks,d/ticks) if ticks else None
    cost = all(p['common_drive_ticks']>0 for p in pairs) and cost_ratio is not None and under(cost_ratio,1.20)
    branch_a = bool(conversions) and not regressions
    pooled = ratio(sum(p['sse_policy'] for p in common),sum(p['sse_zero'] for p in common))
    branch_b = len(common)>=4 and pooled['denominator']>0 and under(pooled,.85) and len(no_worse)>=4
    passed = bool(prerequisite and cost and (branch_a or branch_b))
    require(prerequisite is saved['prerequisites']['passed'] and cost is saved['cost_gate']['passed']
            and branch_a is saved['branch_a_task_conversion']['passed']
            and branch_b is saved['branch_b_sse_improvement']['passed']
            and passed is saved['RL_contribution_passed']
            and sorted(no_worse)==sorted(saved['branch_b_sse_improvement']['pairs_no_worse_within_1p02']),
            'original six independent contribution gate differs')
    return {'RL_contribution_passed':passed,'prerequisites_passed':bool(prerequisite),
        'cost_gate_passed':bool(cost),'cost_ratio':cost_ratio,'conversion_branch_passed':bool(branch_a),
        'conversion_cases':conversions,'capability_regression_cases':regressions,
        'sse_branch_passed':bool(branch_b),'common_passing_pair_count':len(common),
        'pooled_sse_ratio':pooled,'within_1p02_cases':no_worse,'within_1p02_count':len(no_worse),
        'mirror_excluded':True,'zero_role':'reference baseline; no self-comparison contribution claim'}


def summarize():
    inputs = {}
    def bind(path):
        path=Path(path); actual=identity(path); inputs[str(path)]=actual; return path
    def read(path):
        return json.loads(bind(path).read_text())
    def readback(path,source):
        value=read(path); host=read(path.with_name(path.stem+'_host_receipt.json'))
        require(host['exit_code']==0 and host['failure'] is None and host['controls']==host['model_calls']==0
                and host['output_identity']==identity(path) and host['reader_identity']==identity(bind(source)), 'readback host/source/output mismatch')
        return value
    spec=read(C/'spec22.json')
    for name in ('training_contract_22.md','evaluation_contract_22.md','plan_22.md'):
        bind(C/name)
    require(spec['execution_contract_id']==CONTRACT,'contract mismatch')
    claim=spec['evaluation']['final_claim_gates']
    for key,expected in {'final_B_over_A_error_max':.90,'final_B_over_old_error_max':.95,
            'each_final_family_drive_B_over_A_error_max':1.02,'final_B_over_zero_cost_max':1.20,
            'final_B_over_A_cost_max':1.10}.items():
        require(claim[key]==expected,'frozen final numeric gate changed: '+key)
    ledger=read(C/'execution_ledger_22.json')
    require(ledger['execution_contract_id']==CONTRACT and ledger['all_three_workers_executed'] is True
            and ledger['all_executed_workers_independently_verified'] is True,
            'all executed workers and independent readers must close first')
    for path,expected in ledger['inputs'].items():
        require(identity(bind(path))==expected,'ledger input changed: '+path)
    a=readback(Path(spec['A_reuse']['reader']),C.parent/'continuation20/read_training20.py')
    b=readback(C/'train_B_1_readback.json',C/'read_training22.py')
    dev=readback(C/'development_1_readback.json',C/'read_eval22.py')
    final=readback(C/'final_1_readback.json',C/'read_eval22.py')
    reuse=read(C/'a_reuse_22.json'); pair=read(C/'first1024_pair_22.json')
    decision=read(C/'final_execution_decision_22.json')
    for evidence in (pair,decision):
        for path,expected in evidence['inputs'].items():
            require(identity(bind(path))==expected,'pair/decision input changed: '+path)
    require(pair['passed'] is all(pair['fields'].values()) and pair['passed'] is True
            and decision['final_execution_allowed'] is True and all(decision['execution_gates'].values())
            and decision['coverage_or_performance_selects_final'] is False
            and decision['final_scores_inspected'] is False,'fixed final execution eligibility differs')
    require(dev['evaluation_split']=='development' and final['evaluation_split']=='final'
            and all(v['execution_contract_id']==CONTRACT and v['stage']==1 and v['source_closure_verified'] is True for v in (b,dev,final))
            and dev['A_reuse']==final['A_reuse']==reuse and reuse['passed'] is True
            and dev['regression_reuse']['complete'] is True and final['regression_reuse']['complete'] is True
            and dev['original_six_pool']==final['original_six_pool']
            and dev['six_task_zero_pairs']==final['six_task_zero_pairs'], 'fixed A/B/source or regression reuse changed')
    regression_ids={r['case_id'] for r in spec['evaluation']['regression']}
    reg_rows=[r for r in dev['numeric_scores']+dev['regression_reuse']['scores'] if r['case_id'] in regression_ids]
    reg=score_table(reg_rows)
    require(set(reg)=={(case,actor) for case in regression_ids for actor in ACTORS},'original seven incomplete')
    final_reg=score_table(final['regression_reuse']['scores'])
    require(final_reg==reg,'final reused original-seven scores differ from development')
    dp,fp=pool(dev,spec['evaluation']['development']),pool(final,spec['evaluation']['final_sealed'])
    for value,case_specs in ((dev,spec['evaluation']['development']), (final,spec['evaluation']['final_sealed'])):
        cases={r['case_id'] for r in case_specs}
        pairs={r['case_id']:r for r in value['new_actor_initial_pairs'] if r['case_id'] in cases}
        require(set(pairs)==cases and all(r['actors']==list(ACTORS) and r['five_initial_arrays_bitwise_equal'] is True
                and r['complete_control_reset_state_equal'] is True for r in pairs.values()),'new four-case full initial pairing differs')
    tasks={}
    for actor in ACTORS:
        groups={'original_seven':[reg[name,actor] for name in sorted(regression_ids)],
            'development_four':[r for r in dev['numeric_scores'] if r['experiment_actor']==actor and r['case_id'] in {x['case_id'] for x in spec['evaluation']['development']}],
            'final_four':[r for r in final['numeric_scores'] if r['experiment_actor']==actor]}
        tasks[actor]={group:{'passed':all(r['task_passed'] for r in rows),'passed_count':sum(r['task_passed'] for r in rows),
            'case_count':len(rows),'failed_cases':[{'case_id':r['case_id'],'reasons':r['task_reasons']} for r in rows if not r['task_passed']]}
            for group,rows in groups.items()}
    training_valid=bool(a['training_valid'] and b['training_valid'] and reuse['passed'])
    coverage=bool(a['coverage_valid'] and b['coverage_valid'])
    gates={'two_valid_training_records_and_A_reuse':training_valid,'both_arms_expected_coverage':coverage,
        'fresh_first1024_and_complete_new_case_initial_pairs':pair['passed'],
        'all_final_records_full_horizon_safe':fp['complete'],
        'original_regression_reuse_complete':dev['regression_reuse']['complete'] and final['regression_reuse']['complete'],
        'B_all_original_seven_pass':tasks['B']['original_seven']['passed'],
        'B_all_final_four_pass':tasks['B']['final_four']['passed']}
    if fp['complete']:
        for gate,key,metric,bound in (('final_B_over_A_error','B/A','error',.90),
                ('final_B_over_old_error','B/old','error',.95),('final_B_over_zero_cost','B/zero','cost',1.20),
                ('final_B_over_A_cost','B/A','cost',1.10)):
            gates[gate]=under(fp['comparisons'][key][metric],bound)
        for family in ('yaw','bumps'):
            gates['final_'+family+'_drive_B_over_A_error']=under(fp['family_B_over_A'][family]['drive_error_mean'],1.02)
    else:
        gates['complete_pool_required_for_numeric_claim']=False
    course_passed=all(gates.values())
    rl={actor:original_rl(actor,dev,reg) for actor in ('A','B')}
    regression_cost=dev['original_six_pool']
    original_cost_description={'complete':regression_cost['complete'],'role':'development descriptive gate and reported regression cost; not an additional final curriculum-benefit gate',
        'comparisons':regression_cost.get('comparisons',{}),'mirror_excluded':True}
    if regression_cost['complete']:
        original_cost_description['B_zero_cost_at_most_1p20']=under(regression_cost['comparisons']['B/zero']['cost'],1.20)
        original_cost_description['B_A_cost_at_most_1p10']=under(regression_cost['comparisons']['B/A']['cost'],1.10)
    candidate_manifest=read(C/'train_B_1/final_checkpoint_manifest.json')
    require(candidate_manifest['files']['final_model.zip']==b['actual_learning']['checkpoint_model_identity'], 'research candidate differs from verified unique B final')
    for path,expected in inputs.items():
        require(identity(path)==expected,'summary input changed during aggregation: '+path)
    return {'schema':'d1-c22-execution-scientific-summary-v1','execution_contract_id':CONTRACT,
        'training':{'A_reused_from_C20':True,'A_training_valid':a['training_valid'],'B_training_valid':b['training_valid'],
            'A_actual_learning':a['actual_learning']['actual'],'B_actual_learning':b['actual_learning']['actual'],
            'A_coverage_valid':a['coverage_valid'],'B_coverage_valid':b['coverage_valid'],
            'A_coverage':a['coverage'],'B_coverage':b['coverage'],'first1024_pair_valid':pair['passed']},
        'final_execution_eligibility':decision,
        'curriculum_B_over_A_final_claim':{'passed':course_passed,'gates':gates,
            'failed_gates':[name for name,value in gates.items() if not value],
            'interpretation':'finite_preregistered_curriculum_package_benefit_supported' if course_passed else
                ('mechanism_inconclusive_due_to_insufficient_coverage' if not coverage else 'preregistered_curriculum_benefit_not_established'),
            'no_final_hold_threshold_added':True,'development_numeric_gates_do_not_select_or_gate_final_claim':True,
            'scope':'single A/B training pair and finite sealed commands; neither statistical reliability nor domain-randomization proof'},
        'original_six_RL_vs_zero':rl,'task_qualification_by_actor':tasks,
        'development_equal_case':dp,'final_equal_case':fp,
        'development_descriptive_gates':decision['development_descriptive_gates'],
        'development_descriptive_gates_passed':decision['development_descriptive_gates_passed'],
        'original_six_regression_cost_descriptive':original_cost_description,
        'transition_diagnostic_definition':'drive-window raw normalized SSE minus hold-window SSE, divided by the difference in completed ticks; includes startup and any post-hold drive samples, excludes post-release braking, and is not a local yaw-reversal window; descriptive only',
        'cost_definition':'mean squared actual torque normalized by per-motor limits over all 5 native steps and 16 motors, then prescribed case weighting; not energy',
        'research_candidate':{'actor':'B','stage':1,'checkpoint_path':str(C/'train_B_1/final_checkpoint/final_model.zip'),
            'checkpoint_identity':candidate_manifest['files']['final_model.zip'],'selection':'fixed before development/final scores; unique complete final checkpoint',
            'qualified_for_reported_original_seven_and_final_four':tasks['B']['original_seven']['passed'] and tasks['B']['final_four']['passed'],
            'curriculum_benefit_gate_passed':course_passed,'RL_vs_zero_gate_passed':rl['B']['RL_contribution_passed'],
            'GUI_qualification_established_by_this_run':False,'default_policy_changed':False,'automatic_promotion_performed':False},
        'actual_execution_ledger':{'path':str(C/'execution_ledger_22.json'),'identity':identity(C/'execution_ledger_22.json'),
            'actual_totals':ledger['actual_totals'],'actual_native_including_compiler':ledger['actual_native_including_compiler']},
        'pure_geometry_counting_boundary':'191200 counts only historical C21 plus new C22 formal preflight servo intervals. Reader intervals are separate categorized work. Runtime reset qualification also integrates pure nominal paths; these figures are not a complete all-system pure-servo-call ledger. Robot/native/model calls use the closed actual execution ledger.',
        'further_training_or_physics_authorized_by_summary':False,'inputs':inputs,
        'new_model_calls_by_this_script':0,'new_physics_calls_by_this_script':0,'new_servo_calls_by_this_script':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=C/'execution_summary_22.json')
    args=parser.parse_args()
    require(not args.output.exists(),'summary output exists; no overwrite')
    result=summarize(); result['summary_source_identity']=identity(Path(__file__))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'output':str(args.output),'identity':identity(args.output),
        'curriculum_final_claim':result['curriculum_B_over_A_final_claim']['passed'],
        'B_RL_vs_zero':result['original_six_RL_vs_zero']['B']['RL_contribution_passed']},sort_keys=True))


if __name__=='__main__':
    main()
