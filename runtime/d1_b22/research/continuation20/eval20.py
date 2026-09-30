"""Root-owned once-only C20 dev or sealed evaluation; no physics on import."""
from pathlib import Path
import json


def evaluate(session, runtime, guard, env, output, writer, counter):
    from rl16_learning_11 import load_and_verify_final
    from recipes20 import make_schedule, floor_schedule
    from heldout20 import record_case
    from floor_bridge20 import read_floor_isolated
    from run_rl16_training_08 import _boundary
    policies, hashes, paths = {}, {}, {}
    for actor in ('old', 'A', 'B'):
        manifest = session['eval_manifests'][actor]
        counter.phase = 'probe'
        policies[actor], report = load_and_verify_final(manifest['folder'], manifest, return_model=True)
        hashes[actor] = manifest['files']['final_model.zip']['sha256']
        paths[actor] = str(Path(manifest['folder'])/'final_model.zip')
        writer.commit_json(output/(actor+'_load.json'), report)
    construction = json.loads((output/'construction_receipt.json').read_text())
    writer.commit_json(output/'eval_identity.json', {
        'source_hashes': session['source_hashes'], 'execution_stage': 20,
        'controller_variant': 'combined', 'evaluation_split': session['evaluation_split'],
        'checkpoint_sha256_by_actor': hashes, 'checkpoint_paths_by_actor': paths,
        'actual_geometry_binding': construction['geometry_binding'],
        'compiled_geometry': construction['compiled_geometry'], 'warnings': []})
    floors, enabled, cases = {}, [], []
    if session['include_floors']:
        floorroot = output/'floor'
        floorroot.mkdir(exist_ok=False)
        for actor in ('A', 'B'):
            counter.phase = 'floor:'+actor
            receipt, _ = record_case(runtime, guard, env, policies[actor],
                floor_schedule(session['floor_seed']), actor, floorroot/actor,
                writer=writer, pair_initial=None, checkpoint_sha256=hashes[actor])
            if not receipt['record_valid']:
                raise RuntimeError('C20 floor record invalid')
            gate = read_floor_isolated(floorroot/actor,
                artifact_dir=output/'offline_floor_readers'/actor)
            floors[actor] = gate
            writer.commit_json(output/(actor+'_floor_gate.json'), gate)
            if gate['numeric_gate_passed']:
                enabled.append(actor)
    else:
        enabled = ['A', 'B']
    heldroot = output/'heldout'
    heldroot.mkdir(exist_ok=False)
    for spec in session['case_specs']:
        paired = None
        for actor in spec['actors']:
            if actor in ('A', 'B') and actor not in enabled:
                continue
            counter.phase = 'heldout:'+spec['case_id']+':'+actor
            receipt, initial = record_case(runtime, guard, env,
                policies.get(actor), make_schedule(spec['case_id']), actor,
                heldroot/(spec['case_id']+'_'+actor), writer=writer, pair_initial=paired,
                checkpoint_sha256=hashes.get(actor))
            if not receipt['record_valid']:
                raise RuntimeError('C20 task record invalid')
            if paired is None:
                paired = initial
            cases.append({k: receipt[k] for k in (
                'case_id', 'actor', 'seed', 'completed_controls', 'record_valid',
                'terminated', 'truncated', 'stop_reason', 'policy_predictions', 'checkpoint_sha256')})
            writer.commit_json(output/f'progress_case_{len(cases):02d}.json',
                {'case': cases[-1], 'boundary': _boundary(runtime)})
    return {'floors': floors, 'enabled_arms': enabled, 'cases': cases,
            'completed_controls': runtime.control_completed,
            'full_comparison_executed': len(cases) == sum(len(s['actors']) for s in session['case_specs']),
            'qualification_pending_independent_readback': True}
