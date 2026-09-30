"""Stage 17 fixed-controller evaluation with the unchanged final RL policy."""
from pathlib import Path
import json


def evaluate(session, runtime, guard, env, output, writer, counter):
    from rl16_learning_11 import load_and_verify_final
    from recipes18 import make_schedule, floor_schedule
    from heldout15 import record_case
    from floor_bridge18 import read_floor_isolated
    from run_rl16_training_08 import _boundary
    manifest = session['eval_manifests']['grouped']
    counter.phase = 'probe'
    policy, report = load_and_verify_final(manifest['folder'], manifest, return_model=True)
    sha = manifest['files']['final_model.zip']['sha256']
    writer.commit_json(output/'grouped_load.json', report)
    construction = json.loads((output/'construction_receipt.json').read_text())
    writer.commit_json(output/'eval_identity.json', {
        'source_hashes': session['source_hashes'], 'execution_stage': 18,
        'controller_variant': session['controller_variant'],
        'stage15_training_reused_without_further_updates': True,
        'checkpoint_sha256_by_actor': {'grouped_continue': sha},
        'checkpoint_paths_by_actor': {'grouped_continue': str(Path(manifest['folder'])/'final_model.zip')},
        'actual_geometry_binding': construction['geometry_binding'],
        'compiled_geometry': construction['compiled_geometry'], 'warnings': []})
    floorroot = output/'floor'
    floorroot.mkdir(exist_ok=False)
    counter.phase = 'floor:grouped'
    receipt, _ = record_case(runtime, guard, env, policy,
        floor_schedule(session['floor_seed']), 'grouped_continue',
        floorroot/'grouped_continue', writer=writer, pair_initial=None, checkpoint_sha256=sha)
    if not receipt['record_valid']:
        raise RuntimeError('stage18 floor record invalid')
    gate = read_floor_isolated(floorroot/'grouped_continue',
                              artifact_dir=output/'offline_floor_readers'/'grouped')
    writer.commit_json(output/'grouped_floor_gate.json', gate)
    writer.commit_json(output/'grouped_floor_progress.json', {
        'receipt': str(floorroot/'grouped_continue'/'case_receipt.json'),
        'boundary': _boundary(runtime), 'numeric_gate_passed': gate['numeric_gate_passed']})
    heldroot = output/'heldout'
    heldroot.mkdir(exist_ok=False)
    cases = []
    if gate['numeric_gate_passed']:
        for spec in session['case_specs']:
            paired = None
            for actor in spec['actors']:
                if actor not in ('zero', 'grouped_continue'):
                    raise RuntimeError('unexpected stage18 actor')
                counter.phase = 'heldout:'+spec['case_id']+':'+actor
                receipt, initial = record_case(runtime, guard, env,
                    None if actor == 'zero' else policy,
                    make_schedule(spec['case_id'], spec['seed'], mirror=spec['mirror']),
                    actor, heldroot/(spec['case_id']+'_'+actor), writer=writer,
                    pair_initial=paired, checkpoint_sha256=None if actor == 'zero' else sha)
                if not receipt['record_valid']:
                    raise RuntimeError('stage18 task record invalid')
                if actor == 'zero':
                    paired = initial
                cases.append({key: receipt[key] for key in (
                    'case_id', 'actor', 'seed', 'completed_controls', 'record_valid',
                    'terminated', 'truncated', 'stop_reason', 'policy_predictions', 'checkpoint_sha256')})
                writer.commit_json(output/f'progress_case_{len(cases):02d}.json',
                                   {'case': cases[-1], 'boundary': _boundary(runtime)})
    return {'floors': {'grouped': gate},
            'enabled_arms': ['grouped'] if gate['numeric_gate_passed'] else [],
            'cases': cases, 'completed_controls': runtime.control_completed,
            'full_comparison_executed': len(cases) == sum(len(s['actors']) for s in session['case_specs']),
            'qualification_pending_independent_readback': True}
