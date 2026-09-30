"""Root-only bounded source binding. Does not construct any model or plant."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
from worker20 import identity, write

C = Path(__file__).resolve().parent
W = C.parent
R = Path('/home/lyh/wheel-legged-control-lab')


def training_model_limits():
    return dict(load=2, torch_load=6, save=2, learn=1, train=32, forward=32768,
                evaluate_actions=512, predict_values=64, predict=3, backward=512)


def eval_model_limits(max_predict):
    return dict(load=3, torch_load=9, save=0, learn=0, train=0, forward=0,
                evaluate_actions=0, predict_values=0, predict=max_predict, backward=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('training', 'development', 'final'), required=True)
    p.add_argument('--stage', type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    if args.phase == 'development' and args.stage != 1:
        raise ValueError('only one development phase is permitted; stage2 proceeds to sealed final')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('freeze hard600s')))
    signal.alarm(600)
    spec = json.loads((C/'spec20.json').read_text())
    review = json.loads((C/'source_review_20.json').read_text())
    if review['status'] != 'GO':
        raise ValueError('actual Astra source review GO required')
    for path, digest in review['source_sha256'].items():
        if identity(path)['sha256'] != digest:
            raise ValueError('reviewed source changed: '+path)
    checks = json.loads((C/'pure_tests_receipt_20.json').read_text())
    if not checks['passed'] or checks['changed_sources']:
        raise ValueError('necessary pure checks did not pass')
    for path, digest in checks['source_sha256'].items():
        if identity(path)['sha256'] != digest:
            raise ValueError('tested source changed: '+path)
    prior = W/'continuation18/plan_go_C18.json'
    frozen = json.loads(prior.read_text())['inputs']
    for path, expected in frozen.items():
        if identity(path) != expected:
            raise ValueError('historical frozen source changed: '+path)
    original = W.parent/'rl_improvement_20260914/frozen_source_before.json'
    old77 = json.loads(original.read_text())['sha256']
    if len(old77) != 77:
        raise ValueError('original frozen source count differs')
    for relative, digest in old77.items():
        if identity(R/relative)['sha256'] != digest:
            raise ValueError('original frozen source changed: '+relative)
    extras = [prior, original, C/'source_review_20.json', C/'pure_tests_receipt_20.json',
              C/'sealed_command_tables_20.json']
    extras += list(C.glob('*.py'))
    extras += [C/n for n in ('plan_20.md','training_contract_20.md','evaluation_contract_20.md','spec20.json')]
    for name in ('qualification_primary_01', 'qualification_mirror_01'):
        extras += [p for p in (W/'continuation18'/name).rglob('*') if p.is_file()]
    extras += [W/'continuation18'/n for n in (
        'qualification_primary_readback_18.json','qualification_mirror_readback_18.json',
        'qualification_summary_18.json','final_review_17_18.md')]
    parent = W/'continuation15/grouped_01'
    extras += [p for p in (parent/'final_checkpoint').iterdir() if p.is_file()]
    extras += [parent/'final_checkpoint_manifest.json']
    arms = {}
    if args.stage == 2 or args.phase == 'final':
        decision_path = C/'continuation_decision_20.json'
        from decision20 import validate_saved_decision
        decision = validate_saved_decision(decision_path)
        if args.stage == 2 and not decision['continue_training']:
            raise ValueError('stage2 continuation gate is not passed')
        if args.phase == 'final' and decision['candidate_stage'] != args.stage:
            raise ValueError('sealed final candidate differs from prior decision')
        extras.append(decision_path)
        extras += [Path(path) for path in decision['inputs']]
    reuse = {key: str(W/'continuation18'/name) for key,name in (
        ('primary_run','qualification_primary_01'),('mirror_run','qualification_mirror_01'),
        ('primary_readback','qualification_primary_readback_18.json'),
        ('mirror_readback','qualification_mirror_readback_18.json'))}
    if args.phase == 'training':
        # BudgetSpec is numeric metadata; no policy or plant imports here.
        budget = {'schema':'d1-course-rl16-learning-budget-v1', 'total_controls':32768,
                  'n_steps':1024, 'batch_size':256, 'n_epochs':4, 'rollouts':32,
                  'train_calls':32, 'epochs':128, 'minibatches_per_epoch':4, 'optimizer_steps':512}
        budget_sha = hashlib.sha256(json.dumps(budget,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        for arm in ('A','B'):
            selected = parent if args.stage == 1 else C/f'train_{arm}_1'
            offset = 0
            if args.stage == 2:
                reader = json.loads((C/f'train_{arm}_1_readback.json').read_text())
                if not reader['training_valid']:
                    raise ValueError('stage1 training not independently valid')
                offset = reader['next_source_episode_index']
                extras += [C/f'train_{arm}_1_readback.json']
                extras += [p for p in (selected/'final_checkpoint').iterdir() if p.is_file()]
                extras.append(selected/'final_checkpoint_manifest.json')
            model = selected/'final_checkpoint/final_model.zip'
            key = f'train_{arm}_{args.stage}'
            limits = spec['per_training_worker_limits']
            arms[key] = {**{k:limits[k] for k in ('controls','soft_s','close_s','hard_s')},
                'mode':'train','arm':arm,'stage':args.stage,'output_directory':key,
                'model_limits':training_model_limits(), 'source_episode_offset':offset,
                'budget_spec':budget,'budget_spec_sha256':budget_sha,
                'short_plan_schema':'d1-c18-command-coverage-short1000-c20-v1',
                'short_plan_source_sha256':identity(C/'recipes20.py')['sha256'],
                'ppo_seed':spec['training']['ppo_seed_by_stage'][args.stage-1],
                'command_seed':201002,'measurement_seed':201003,
                'seed':spec['training']['ppo_seed_by_stage'][args.stage-1],
                'segments':[32768],'training_control_limit':32768,'heldout_control_limit':0,
                'parentpaths':{'model':str(model),'metadata':str(model.with_name('final_metadata.json')),
                               'model_sha256':identity(model)['sha256']}}
    else:
        paired = C/'first1024_pair_20.json'
        pair_record = json.loads(paired.read_text())
        if not pair_record['passed']:
            raise ValueError('shared first1024 warmup must be byte-identical before evaluation')
        for path, expected in pair_record['inputs'].items():
            if identity(path) != expected:
                raise ValueError('paired warmup input changed')
        extras += [paired] + [Path(path) for path in pair_record['inputs']]
        for arm in ('A','B'):
            selected = C/f'train_{arm}_{args.stage}'
            reader_path = C/f'train_{arm}_{args.stage}_readback.json'
            reader = json.loads(reader_path.read_text())
            if not reader['training_valid']:
                raise ValueError('training record invalid: '+arm)
            extras.append(reader_path)
            extras += [p for p in (selected/'final_checkpoint').iterdir() if p.is_file()]
            extras.append(selected/'final_checkpoint_manifest.json')
        include_floor = args.phase == 'development' or args.stage == 2
        cases = ([{**s,'actors':['A','B']} for s in spec['evaluation']['regression']] if include_floor else [])
        group = 'development' if args.phase == 'development' else 'final_sealed'
        cases += [{**s,'actors':['zero','old','A','B']} for s in spec['evaluation'][group]]
        limits = spec['development_worker_limits'] if include_floor else spec['final_worker_limits_without_extension']
        key = f'{args.phase}_{args.stage}'
        arms[key] = {**{k:limits[k] for k in ('controls','soft_s','close_s','hard_s')},
            'mode':'eval','arm':key,'stage':args.stage,'output_directory':key,
            'evaluation_split':args.phase,'include_floors':include_floor,
            'floor_seed':201090 if args.stage == 1 else 201091,
            'case_specs':cases,'model_limits':eval_model_limits(limits['predict_api_max']),
            'training_control_limit':0,'heldout_control_limit':limits['controls'],
            'segments':([600,600] if include_floor else [])+[s['horizon'] for s in cases for _ in s['actors']],
            'seed':201090 if args.stage == 1 else 201091,'regression_reuse':reuse,
            'eval_manifest_paths':{'old':str(parent/'final_checkpoint_manifest.json'),
                **{a:str(C/f'train_{a}_{args.stage}/final_checkpoint_manifest.json') for a in ('A','B')}}}
        if args.phase == 'final' and args.stage == 1:
            development = C/'development_1_readback.json'
            extras += [development, C/'development_1/session.json',
                       C/'development_1_readback_host_receipt.json']
            arms[key]['development_readback_path'] = str(development)
    for path in extras:
        frozen[str(path)] = identity(path)
    plan = {'schema':'d1-coverage-source-go-20-v1','status':'GO','retry_permitted':False,
            'arms':arms,'inputs':dict(sorted(frozen.items())), 'source_review':identity(C/'source_review_20.json')}
    output = C/f'plan_go_{args.phase}_{args.stage}_20.json'
    write(output,plan)
    signal.alarm(0)
    print(json.dumps({'plan':str(output),'inputs':len(frozen),'controls':sum(s['controls'] for s in arms.values()),'identity':identity(output)}))


if __name__ == '__main__':
    main()
