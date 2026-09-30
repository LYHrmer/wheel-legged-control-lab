"""Root's once-only warm-start training phase; no physics on import."""
from pathlib import Path
import numpy as np


def train(session, runtime, guard, env, output, writer, counter):
    from recipes22 import TRAIN_BUDGET as budget, build_plan, T, MEASUREMENT_SEED
    from curriculum22 import Curriculum20
    from learning21 import load_warm_start20, scoped_clip_audit20, file_identity, training_receipt20
    from rl16_learning_11 import make_rollout_callback, save_failure_checkpoint
    from run_short_rl16_training_11 import _training_metadata
    from run_rl16_training_08 import _audited_learning_origins, _boundary
    from checkpoint21 import save_final_and_verify
    from runtime_support22 import atomic_training_io

    folder = output / 'training'
    folder.mkdir(exist_ok=False)
    gradients = output / 'gradients'
    gradients.mkdir(exist_ok=False)
    plan = build_plan(source_sha256=session['short_plan_source_sha256'], arm=session['arm'], source_offset=session['source_episode_offset'])
    runtime.start_segment('training_' + session['arm'], budget.total_controls)
    runtime.bind(env)
    guard.start_segment(env.plant, folder, 'training_' + session['arm'], budget.total_controls, 'heldout')
    model = audit = clip = curriculum = None
    failure = None
    result = {}
    with atomic_training_io(writer):
        try:
            curriculum = Curriculum20(env, runtime, folder, budget=budget,
                plan=plan, caps=env.caps, measurement_seed=MEASUREMENT_SEED, writer=writer, source_offset=session['source_episode_offset'])
            counter.phase = 'warm_load'
            model, audit, clip, parentage = load_warm_start20(
                curriculum, budget, T['ppo_seed_by_stage'][session['stage']-1], session['arm'], session['parentpaths'], target_kl=T['target_kl'], stage=session['stage'])
            writer.commit_json(output / 'parentage.json', parentage)
            writer.commit_json(output / 'dependency_origins.json', _audited_learning_origins(session))
            parentfolder = Path(session['parentpaths']['model']).parent
            probes = np.fromfile(parentfolder / 'final_probe_observations_f32.bin', dtype=np.float32).reshape(32, 99)
            expected_actions = np.fromfile(parentfolder / 'final_probe_actions_f32.bin', dtype=np.float32).reshape(32,16)
            counter.phase = 'probe'
            actions, state = model.predict(probes, deterministic=True)
            if state is not None or np.asarray(actions,dtype=np.float32).tobytes() != expected_actions.tobytes():
                raise RuntimeError('warm-start deterministic probe differs from parent')
            writer.commit_json(output / 'warm_probe.json', {'rows': 32, 'actions_byte_exact': True,
                'source': str(parentfolder / 'final_probe_observations_f32.bin')})
            def persist_gradient(step, pre, post, ids):
                path = gradients / f'gradient_step_{step:04d}.npz'
                arrays = {'pre::' + name: value for name,value in pre.items()}
                arrays.update({'post::' + name: value for name,value in post.items()})
                writer.commit_npz(path, arrays)
                return {'file': str(path), 'identity': file_identity(path), 'ids': ids}
            clip.set_gradient_writer(persist_gradient)
            def progress(summary):
                index = summary['rollout_index']
                if curriculum.completed_controls != (index+1)*1024 or curriculum.gaussian_records != curriculum.completed_controls:
                    raise RuntimeError('rollout/control/Gaussian ledger mismatch')
                writer.commit_json(folder / f'progress_rollout_{index:04d}.json', {
                    'rollout': summary, 'completed_controls': curriculum.completed_controls,
                    'optimizer_steps_before_pending_update': audit.optimizer_steps,
                    'last_completed_update': audit.updates[-1] if audit.updates else None,
                    'boundary': _boundary(runtime)})
            callback = make_rollout_callback(audit, on_transition=curriculum.record_transition,
                                             on_rollout=progress)
            counter.phase = 'training'
            with scoped_clip_audit20(model, clip, soft_stop=lambda: writer.stop_requested):
                model.learn(total_timesteps=budget.total_controls, callback=callback, reset_num_timesteps=True)
            blocks = curriculum.finalize_training()
            if blocks['completed_controls'] != budget.total_controls or blocks['gaussian_records'] != budget.total_controls:
                raise RuntimeError('final training blocks do not close budget')
            learned = training_receipt20(model, audit, clip)
            writer.commit_json(folder / 'learning_receipt.json', learned)
            if not learned['qualified_for_final_checkpoint']:
                raise RuntimeError('training audit rejected final checkpoint')
            metadata = _training_metadata(session, env, runtime, curriculum, boundary=_boundary(runtime))
            metadata['execution_contract_id'] = session['execution_contract_id']
            metadata['numerical_protocol_schema'] = 'C20'
            metadata['stage20_parentage'] = parentage
            metadata['stage20_course_arm'] = session['arm']
            metadata['stage20_stage'] = session['stage']
            metadata['stage20_env_mode'] = 'train'
            metadata['stage20_archive_force_sampling_mode'] = 'heldout/full_contact'
            counter.phase = 'probe'
            manifest = save_final_and_verify(model, str(output / 'final_checkpoint'), metadata,
                                            probes, audit=audit, clip_audit=clip, receipt=learned)
            writer.commit_json(output / 'final_checkpoint_manifest.json', manifest)
            result = {'training': learned, 'checkpoint_manifest': manifest,
                      'completed_controls': curriculum.completed_controls}
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error)}
            # Archive failure remains latched. Never retry a failed payload path.
            if not writer.failed and curriculum is not None and not curriculum._finalized:
                try:
                    curriculum.seal_partial('stage20_failure')
                except BaseException as secondary:
                    failure['partial_seal_error'] = repr(secondary)
            if not writer.failed and model is not None and audit is not None:
                try:
                    failed = training_receipt20(model, audit, clip)
                    writer.commit_json(output / 'failure_learning_receipt.json', failed)
                    if failed['status'] != 'complete':
                        writer.commit_json(output / 'failure_checkpoint_manifest.json',
                            save_failure_checkpoint(model, str(output / 'failure_checkpoint'), failed))
                except BaseException as secondary:
                    failure['failure_checkpoint_error'] = repr(secondary)
            raise
        finally:
            cleanup = []
            try:
                runtime.unbind()
            except BaseException as secondary:
                cleanup.append(repr(secondary))
            try:
                segment = guard.finish_segment()
                if not writer.failed:
                    writer.commit_json(folder / 'training_segment_receipt.json', segment)
            except BaseException as secondary:
                cleanup.append(repr(secondary))
            if clip is not None and not writer.failed:
                try:
                    writer.commit_json(output / 'clip_audit.json', clip.as_dict())
                except BaseException as secondary:
                    cleanup.append(repr(secondary))
            if curriculum is not None and not curriculum._events.closed:
                try:
                    curriculum._events.close()
                except BaseException as secondary:
                    cleanup.append(repr(secondary))
            if cleanup and failure is not None:
                failure['cleanup_errors'] = cleanup
                error = __import__('sys').exception() if hasattr(__import__('sys'), 'exception') else __import__('sys').exc_info()[1]
                if error is not None:
                    setattr(error, 'stage20_cleanup_errors', cleanup)
            if cleanup and failure is None:
                raise RuntimeError('training closure failed: ' + repr(cleanup))
    return result
