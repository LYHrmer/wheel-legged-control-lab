"""Build one exclusive C34 host request from the reviewed, immutable source GO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONTRACT = 'C34_fixed_velocity_reference_and_40mm_v1'


def build(source_go: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    source_go = source_go.resolve(strict=True)
    go = json.loads(source_go.read_text())
    if (go.get('decision') != 'GO' or go.get('execution_contract_id') != CONTRACT
            or set(go.get('arms', {})) != {'development'}):
        raise RuntimeError('C34 requires reviewed single-arm source GO')
    spec = go['arms']['development']
    expected_output = Path(__file__).resolve().parents[1]/'development_01'
    if (spec.get('output_directory') != str(expected_output)
            or (spec.get('control_limit'), spec.get('cycles_limit'),
                spec.get('macros_limit')) != (22000, 10, 40)
            or (spec.get('soft_s'), spec.get('close_s'), spec.get('hard_s'),
                spec.get('outer_s')) != (900, 960, 1020, 1200)
            or spec.get('render') is not False
            or spec.get('retry_permitted') is not False):
        raise RuntimeError('C34 finite request differs from frozen campaign')
    request = {'schema': 'd1-c34-single-request-v1', 'arm': 'development',
               'source_go_path': str(source_go), 'spec': spec, 'x11_events': None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(request, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-go', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = build(args.source_go, args.output)
    print(json.dumps({'request': str(args.output), 'arm': request['arm'],
                      'output_directory': request['spec']['output_directory']}))


if __name__ == '__main__':
    main()
