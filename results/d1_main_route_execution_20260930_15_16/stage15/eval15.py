"""One engine owner evaluates both floors and the preregistered actor/task table."""
from pathlib import Path


def evaluate(session, runtime, guard, env, output, writer, counter):
    from rl16_learning_11 import load_and_verify_final
    from recipes15 import ORDER, floor_schedule, heldout_schedule
    from heldout15 import record_case
    from read_eval15 import read_floor
    from run_rl16_training_08 import _boundary
    policies = {}
    shas = {}
    floors = {}
    cases = []
    for arm in ('global','grouped'):
        manifest = session['eval_manifests'][arm]
        counter.phase = 'probe'
        policy, report = load_and_verify_final(manifest['folder'], manifest, return_model=True)
        policies[arm] = policy
        shas[arm] = manifest['files']['final_model.zip']['sha256']
        writer.commit_json(output / (arm+'_load.json'), report)
    import json
    construction = json.loads((output/'construction_receipt.json').read_text())
    writer.commit_json(output/'eval_identity.json', {
        'source_hashes':session['source_hashes'],
        'checkpoint_sha256_by_actor':{arm+'_continue':sha for arm,sha in shas.items()},
        'checkpoint_paths_by_actor':{arm+'_continue':str(Path(session['eval_manifests'][arm]['folder'])/'final_model.zip') for arm in ('global','grouped')},
        'actual_geometry_binding':construction['geometry_binding'],
        'compiled_geometry':construction['compiled_geometry'], 'warnings':[]})
    floorroot = output / 'floor'
    floorroot.mkdir(exist_ok=False)
    initial = None
    for arm in ('global','grouped'):
        actor = arm + '_continue'
        folder = floorroot / actor
        counter.phase = 'floor:'+arm
        receipt, initial = record_case(runtime, guard, env, policies[arm],
            floor_schedule(), actor, folder, writer=writer, pair_initial=initial,
            checkpoint_sha256=shas[arm])
        if not receipt['record_valid']:
            raise RuntimeError('floor record invalid: '+arm)
        gate = read_floor(folder)
        writer.commit_json(output / (arm+'_floor_gate.json'), gate)
        floors[arm] = gate
        writer.commit_json(output / (arm+'_floor_progress.json'), {'receipt':str(folder/'case_receipt.json'),
            'boundary':_boundary(runtime),'numeric_gate_passed':gate['numeric_gate_passed']})
    enabled = [arm for arm in ('global','grouped') if floors[arm]['numeric_gate_passed']]
    heldroot = output/'heldout'
    heldroot.mkdir(exist_ok=False)
    if enabled:
        for case_id in ORDER:
            pair_initial = None
            for arm in ('zero',*enabled):
                actor = 'zero' if arm == 'zero' else arm + '_continue'
                folder = heldroot / (case_id+'_'+actor)
                counter.phase = 'heldout:'+case_id+':'+arm
                receipt, initial = record_case(runtime, guard, env, policies.get(arm),
                    heldout_schedule(case_id), actor, folder, writer=writer,
                    pair_initial=pair_initial, checkpoint_sha256=shas.get(arm))
                if not receipt['record_valid']:
                    raise RuntimeError('heldout record invalid: '+case_id+':'+arm)
                if arm=='zero':pair_initial=initial
                cases.append({key:receipt[key] for key in (
                    'case_id','actor','seed','completed_controls','record_valid','terminated',
                    'truncated','stop_reason','policy_predictions','checkpoint_sha256')})
                writer.commit_json(output / f'progress_case_{len(cases):02d}.json',
                    {'case':cases[-1],'boundary':_boundary(runtime)})
    return {'floors':floors,'enabled_arms':enabled,'cases':cases,
            'completed_controls':runtime.control_completed,
            'full_comparison_executed':len(cases)==18,
            'qualification_pending_independent_readback':True}
