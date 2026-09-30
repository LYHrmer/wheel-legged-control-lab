"""Prepare one additive C22 public snapshot; read-only unless --copy is set.

The full control/native archives stay local. Every local C22 file receives a
SHA256 inventory row, while one early block per observed run is copied as a
sample. A failed experiment is publishable after its final summary and review
are closed; this script does not turn a failed gate into a success claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01')
C22 = W/'continuation22'
C21 = W/'continuation21'
R = Path('/home/lyh/wheel-legged-control-lab')
DEST = R/'results'/'d1_main_route_execution_20260930_22'
SCHEMA = 'd1-main-route-c22-selected-publication-v1'
SUMMARY = C22/'execution_summary_22.json'
REVIEW = C22/'final_review_22.md'

# C21 stopped at its pure geometry gate. These files explain that failure and
# supply the frozen sources/templates reused by C22, not C21 runtime evidence.
PREDECESSOR_ROOT_FILES = frozenset({
    'execution_summary_21.json','failure_review_21.md','finite_geometry_21.json',
    'a_reuse_21.json','plan_21.md','training_contract_21.md',
    'evaluation_contract_21.md','spec21.json','geometry21.py',
    'run_preflight21.py','learning21.py','checkpoint21.py','state21.py',
    'finite_geometry_host_21/host_receipt.json',
    'finite_geometry_host_21/reservation.json',
    'finite_geometry_host_21/stdout.log',
})
SAMPLE_GZIP = frozenset({
    'train_B_1/training/full_control_records_0000.jsonl.gz',
    'train_B_1/training/native_block_0000.jsonl.gz',
    'development_1/heldout/dev_yaw_left_B/control_records_0000.jsonl.gz',
    'development_1/heldout/dev_yaw_left_B/native_block_0000.jsonl.gz',
    'final_1/heldout/final_yaw_left_B/control_records_0000.jsonl.gz',
    'final_1/heldout/final_yaw_left_B/native_block_0000.jsonl.gz',
})


def identity(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError('not a regular local file: '+str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
            size += len(chunk)
    return {'sha256':digest.hexdigest(),'bytes':size}


def document(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if type(value) is not dict:
        raise ValueError('expected JSON object: '+str(path))
    return value


def _inside(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise ValueError('required report is a symlink: '+str(path))
    actual = path.resolve(strict=True)
    if not actual.is_file() or not actual.is_relative_to(root):
        raise ValueError('required report is outside the C22 source: '+str(path))
    return actual


def _closed_host(run: Path) -> dict:
    host_path = run/'host_receipt.json'
    if not host_path.is_file():
        raise ValueError('reserved run lacks a closed host receipt: '+str(run))
    host = document(host_path)
    if (host.get('postcheck_complete') is not True
            or host.get('no_live_owned_processes') is not True
            or host.get('fully_reserved_budget_closed') is not True):
        raise ValueError('reserved run has not closed its budget: '+str(run))
    worker_path = run/'worker_receipt.json'
    worker = document(worker_path) if worker_path.is_file() else None
    return {'reservation_key':run.name,'host_identity':identity(host_path),
            'worker_identity':identity(worker_path) if worker is not None else None,
            'host_failure':host.get('failure'),
            'worker_failure':None if worker is None else worker.get('failure'),
            'worker_execution_complete':None if worker is None else worker.get('execution_complete'),
            'independent_readback_present':(C22/f'{run.name}_readback.json').is_file()}


def _sample_gzip(relative: str) -> bool:
    return relative in SAMPLE_GZIP


def _selected_c22(relative: str) -> bool:
    path = Path(relative)
    name = path.name
    if name.endswith('.manifest.json'):
        return _selected_c22(str(path.with_name(name.removesuffix('.manifest.json'))))
    if name.endswith(('.jsonl.gz','.gz')):
        return _sample_gzip(relative)
    if path.suffix=='.npz':
        if relative=='finite_geometry_preflight_22/four_revised_nominal_traces22.npz':
            return True
        if 'gradients' in path.parts:
            return True
        if name in ('controls_0000.npz','gaussian_0000.npz','initial_state.npz'):
            return True
        if name=='training_episode_000000_initial_state.npz':
            return True
        if name=='training_episode_000000_geometry21_nominal_path.npz':
            return True
        if name=='geometry21_nominal_path.npz':
            return True
        if name=='states.npz':
            return 'dev_yaw_left_B' in relative or 'final_yaw_left_B' in relative
        return False
    if path.suffix=='.zip':
        return name in ('final_model.zip','failure_model.zip')
    return True


def _selected_c21(relative: str) -> bool:
    return (relative in PREDECESSOR_ROOT_FILES
            or relative.startswith('finite_geometry_preflight_21/'))


def _inventory_root(root: Path, prefix: str, selector) -> tuple[dict,dict]:
    rows,chosen = {},{}
    for path in sorted(root.rglob('*')):
        relative_path = path.relative_to(root)
        if any(part in ('.ruff_cache','__pycache__') for part in relative_path.parts):
            continue
        if path.is_symlink():
            raise ValueError('publication source contains a symlink: '+str(path))
        if not path.is_file():
            continue
        relative = str(relative_path)
        key = prefix+relative
        included = selector(relative)
        rows[key] = {**identity(path),'local_source':str(path),
                     'included_in_public_package':included}
        if included:
            chosen[key] = path
    return rows,chosen


def _write_exclusive(path: Path, value: dict) -> None:
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def prepare(*, summary_path: Path = SUMMARY, review_path: Path = REVIEW,
            copy: bool = False) -> dict:
    summary_path = _inside(summary_path,C22)
    review_path = _inside(review_path,C22)
    if summary_path!=SUMMARY or review_path!=REVIEW:
        raise ValueError('publication requires the final C22 summary and review names')
    summary = document(summary_path)
    if not review_path.read_text(encoding='utf-8').strip():
        raise ValueError('C22 final review is empty')
    if DEST.exists():
        raise FileExistsError('additive C22 destination already exists: '+str(DEST))

    go_plans = sorted(C22.glob('plan_go_*_22.json'))
    if not go_plans:
        raise ValueError('C22 source GO is missing')
    source_go = {}
    for path in go_plans:
        plan = document(path)
        if plan.get('status')!='GO' or plan.get('retry_permitted') is not False:
            raise ValueError('non-GO source plan: '+str(path))
        for source,expected in plan['inputs'].items():
            if identity(Path(source))!=expected:
                raise ValueError('frozen C22 GO input changed: '+source)
        source_go[path.name]=identity(path)

    closed,offline = [],[]
    for reservation in sorted(C22.glob('*_reservation.json')):
        value = document(reservation)
        key = value.get('reservation_key')
        if key is None:
            offline.append(reservation.name)
            continue
        if reservation.name!=key+'_reservation.json':
            raise ValueError('worker reservation filename differs: '+str(reservation))
        run = C22/key
        if run.is_symlink() or not run.is_dir():
            raise ValueError('reserved worker folder missing: '+str(run))
        closed.append(_closed_host(run))

    current,chosen = _inventory_root(C22,'',_selected_c22)
    predecessor,old_chosen = _inventory_root(C21,'predecessor_c21/',_selected_c21)
    chosen.update(old_chosen)
    for required in (SUMMARY,REVIEW):
        if str(required.relative_to(C22)) not in chosen:
            raise ValueError('required final report omitted from public subset')
    for required in PREDECESSOR_ROOT_FILES:
        if 'predecessor_c21/'+required not in old_chosen:
            raise ValueError('required C21 failure/source evidence missing: '+required)
    inventory = {**current,**predecessor}
    report = {
        'schema':SCHEMA,
        'summary_identity':identity(summary_path),
        'final_review_identity':identity(review_path),
        'reported_outcome':{key:summary[key] for key in
            ('status','qualification_passed','training_valid','final_task_passed')
            if key in summary},
        'source_go_plans':source_go,
        'closed_reserved_runs':closed,
        'offline_reservations_preserved':offline,
        'preregistered_sample_gzip':sorted(SAMPLE_GZIP),
        'absent_preregistered_samples':sorted(SAMPLE_GZIP-set(current)),
        'predecessor_c21_status':'stopped_at_pure_geometry_gate_zero_new_robot_controls',
        'predecessor_c21_runtime_drafts_are_not_GO_evidence':True,
        'local_files':len(inventory),
        'local_bytes':sum(row['bytes'] for row in inventory.values()),
        'selected_files':len(chosen),
        'selected_bytes':sum(inventory[name]['bytes'] for name in chosen),
        'full_control_native_archives_in_public_subset':False,
        'full_independent_reader_requires_local_archives':True,
        'copied':False,
    }
    if not copy:
        return report

    DEST.mkdir(parents=False)
    payloads = {}
    for relative,source in chosen.items():
        target = DEST/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as origin,target.open('xb') as sink:
            shutil.copyfileobj(origin,sink,1024*1024)
        expected = {key:inventory[relative][key] for key in ('sha256','bytes')}
        if identity(target)!=expected:
            raise ValueError('public copied payload changed: '+relative)
        payloads[relative] = {**expected,'local_source':str(source)}
    _write_exclusive(DEST/'local_archive_inventory.json',{
        'schema':'d1-c22-complete-local-file-inventory-v1',
        'files':inventory,
        'snapshot_excludes_publication_outputs':True,
        'full_control_native_archives_remain_local':True,
        'public_subset_cannot_rerun_full_native_reader':True,
    })
    readme = ('# C22 main route execution\n\n'
              f'Reported outcome: `{summary.get("status","see execution summary")}`. '
              'The experiment result, including any failed gate, is recorded without filtering.\n\n'
              '- [Execution summary](execution_summary_22.json)\n'
              '- [Final review](final_review_22.md)\n'
              '- [C21 geometry-gate failure](predecessor_c21/failure_review_21.md)\n'
              '- [Local archive inventory](local_archive_inventory.json)\n\n'
              'The public files include selected control/native blocks and small arrays. '
              'The complete native/control archives remain at the paths and hashes in '
              'the inventory; the public subset alone cannot rerun the full independent reader.\n')
    with (DEST/'README.md').open('x',encoding='utf-8') as stream:
        stream.write(readme)
    report['copied']=True
    report['destination']=str(DEST)
    report['payloads']=payloads
    _write_exclusive(DEST/'publication_receipt.json',report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=SUMMARY)
    parser.add_argument('--final-review',type=Path,default=REVIEW)
    parser.add_argument('--copy',action='store_true')
    args = parser.parse_args()
    report = prepare(summary_path=args.summary,review_path=args.final_review,copy=args.copy)
    print(json.dumps({key:value for key,value in report.items() if key!='payloads'},
                     sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
