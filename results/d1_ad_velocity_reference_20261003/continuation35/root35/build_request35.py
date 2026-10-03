"""Seal one request after actual Astra exact-source GO."""
import argparse
import json
from pathlib import Path


def build(source_go, output):
    go = json.loads(source_go.read_text())
    if (go.get('decision') != 'GO' or go.get('execution_contract_id') !=
            'C35_continuous_diagonal_pair_fixed_feasibility_v1'
            or set(go['arms']) != {'development'}):
        raise RuntimeError('C35 exact-source GO required')
    spec = go['arms']['development']
    if (spec['control_limit'], spec['cycles_limit'], spec['normal_native_cap']) != (9000,5,45000):
        raise RuntimeError('C35 finite reservation differs')
    request = dict(schema='d1-c35-single-request-v1', arm='development',
        source_go_path=str(source_go.resolve(strict=True)), spec=spec, x11_events=None)
    with output.open('x') as stream:
        json.dump(request, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-go',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    build(a.source_go,a.output)
