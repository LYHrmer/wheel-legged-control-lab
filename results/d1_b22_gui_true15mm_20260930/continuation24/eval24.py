"""One matched zero/B evaluation on the same real 15 mm box model."""
from pathlib import Path
import json


def evaluate(session, runtime, guard, env, output, writer, counter):
    from rl16_learning_11 import load_and_verify_final
    from geometry24 import make_schedule, CONTRACT_ID, CASE_ID
    from heldout24 import record_case
    from run_rl16_training_08 import _boundary
    manifest = session['checkpoint_manifest']
    folder = session['checkpoint_folder']
    relocation=session['checkpoint_relocation']
    original_path=Path(relocation['original_manifest_path'])
    original=json.loads(original_path.read_text())
    if (relocation['changed_fields']!=['folder']
            or original['folder']!=relocation['original_folder']
            or relocation['current_folder']!=folder
            or manifest!={**original,'folder':folder}
            or session['source_hashes'].get(str(original_path))!=relocation['original_manifest_identity']):
        raise RuntimeError('C24 checkpoint relocation exceeds exact folder-only allowance')
    checkpoint = manifest['files']['final_model.zip']['sha256']
    if manifest['folder'] != folder or checkpoint != session['checkpoint_sha256']:
        raise RuntimeError('C24 checkpoint identity differs from session')
    counter.phase = 'probe'
    policy, load_report = load_and_verify_final(folder, manifest, return_model=True)
    writer.commit_json(output/'B_load.json',load_report)
    construction = json.loads((output/'construction_receipt.json').read_text())
    writer.commit_json(output/'eval_identity.json',dict(
        source_hashes=session['source_hashes'], execution_contract_id=CONTRACT_ID,
        controller_variant='combined', checkpoint_sha256=checkpoint,
        checkpoint_path=str(Path(folder)/'final_model.zip'),
        actual_geometry_binding=construction['geometry_binding'],
        compiled_geometry=construction['compiled_geometry'], warnings=[]))
    root = output/'heldout'
    root.mkdir(exist_ok=False)
    pair = None
    cases = []
    for actor in ('zero','B'):
        counter.phase = 'heldout:'+CASE_ID+':'+actor
        receipt, initial = record_case(runtime,guard,env,None if actor=='zero' else policy,
            make_schedule(),actor,root/(CASE_ID+'_'+actor),writer=writer,
            pair_initial=pair,checkpoint_sha256=None if actor=='zero' else checkpoint)
        if not receipt['record_valid']:
            raise RuntimeError('C24 physical/archive record invalid')
        if pair is None:
            pair = initial
        cases.append({k:receipt[k] for k in ('case_id','actor','seed',
            'completed_controls','record_valid','terminated','truncated','stop_reason',
            'policy_predictions','checkpoint_sha256','readonly_mj_objectVelocity_calls')})
        writer.commit_json(output/f'progress_case_{len(cases):02d}.json',
            {'case':cases[-1],'boundary':_boundary(runtime)})
    return dict(cases=cases,completed_controls=runtime.control_completed,
        full_comparison_executed=len(cases)==2,
        task_gates_pending_independent_readback=True,
        readonly_mj_objectVelocity_calls=sum(c['readonly_mj_objectVelocity_calls'] for c in cases))
