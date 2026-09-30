"""Copy reviewed runtime sources byte-for-byte into the repository, no imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

R=Path('/home/lyh/wheel-legged-control-lab')
W=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01')
B=R/'runtime/d1_b22'


def ident(path):
    data=path.read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    data=source.read_bytes()
    if target.exists():
        if target.read_bytes()!=data:
            raise ValueError('existing bundle source differs; choose a new revision: '+str(target))
    else:
        with target.open('xb') as stream:
            stream.write(data)
    return {'source':str(source),'target':str(target),**ident(target)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',action='store_true')
    parser.add_argument('--new-source',action='store_true')
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    copied=[]
    if args.base:
        previous=json.loads((W/'continuation22/final_1/session.json').read_text())
        candidates={Path(name) for name in previous['source_hashes']
                    if name.startswith(str(W)+'/') and name.endswith('.py')
                    and not Path(name).name.startswith('test_')}
        candidates.update(W/'gui12'/name for name in (
            'run_world_upright_gui_12.py','latest_frame_mailbox_12.py','async_course_renderer_12.py'))
        candidates.update(W/'continuation13/gui13_revision02'/name for name in (
            'gui13_bridge.py','gui13_contract.py'))
        for path in sorted(candidates):
            old=previous['source_hashes'].get(str(path))
            if old is not None and ident(path)!=old:
                raise ValueError('old source no longer matches frozen C22 bytes: '+str(path))
            copied.append(copy(path,B/'research'/path.relative_to(W)))
        folder=W/'continuation22/train_B_1/final_checkpoint'
        for path in sorted(folder.iterdir()):
            if path.is_file():
                copied.append(copy(path,B/'checkpoint'/path.name))
        copied.append(copy(folder.parent/'final_checkpoint_manifest.json',B/'checkpoint_manifest.json'))
    if args.new_source:
        for folder in (W/'continuation23/gui23',W/'continuation23/root23',W/'continuation24'):
            for path in sorted(folder.glob('*.py')):
                if path.name.startswith('test_') or path.name=='prepare_bundle23.py':
                    continue
                copied.append(copy(path,B/'research'/path.relative_to(W)))
    receipt={'schema':'d1-c23-runtime-bundle-copy-v1','copies':copied,
             'bytes':sum(row['bytes'] for row in copied),'physics':0,'model_calls':0}
    with args.receipt.open('x') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps({'files':len(copied),'bytes':receipt['bytes']}))


if __name__=='__main__':
    main()
