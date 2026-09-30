"""Create a new archival runtime version; preserve all C17 inputs byte-for-byte."""
from pathlib import Path
import ast
import hashlib
import json
import re

C = Path(__file__).resolve().parent
OLD = C.parent / 'continuation17'
NAMES = ('controller17', 'residual17', 'verify_control17', 'read_eval17',
         'score17', 'recipes17', 'host17', 'worker17', 'eval17',
         'floor_bridge17', 'offline_floor17')
COPY = ('read_eval17', 'score17', 'recipes17', 'host17', 'worker17', 'eval17',
        'floor_bridge17', 'offline_floor17', 'readback_host17_v02')


def main():
    rows = {}
    for name in COPY:
        source = OLD / (name + '.py')
        target_name = name.replace('17', '18').replace('_v02', '')
        output = C / (target_name + '.py')
        raw = source.read_text()
        revised = raw
        for stem in sorted(NAMES, key=len, reverse=True):
            revised = revised.replace(stem, stem.replace('17', '18'))
        for old, new in (('C17', 'C18'), ('stage17', 'stage18'),
                         ('-17-v1', '-18-v1'), ('control17_', 'control18_'),
                         ('_17.json', '_18.json'), ('_17.md', '_18.md')):
            revised = revised.replace(old, new)
        revised = re.sub(r"(execution_stage['\"]\s*:\s*)17\b", r'\g<1>18', revised)
        revised = re.sub(r"(identity_doc\[['\"]execution_stage['\"]\]\s*==\s*)17\b",
                         r'\g<1>18', revised)
        ast.parse(revised)
        with output.open('x') as stream:
            stream.write(revised)
        rows[str(output)] = {
            'source': str(source), 'source_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'initial_port_sha256': hashlib.sha256(revised.encode()).hexdigest(),
            'reader_requires_new_controller_state_chain_review': name == 'read_eval17',
        }
    with (C / 'runtime_port_receipt_18.json').open('x') as stream:
        json.dump({'schema': 'd1-runtime-port-18-v1', 'files': rows,
                   'old_sources_modified': False, 'models': 0, 'physics': 0}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'new_files': len(rows), 'AST_valid': True}))


if __name__ == '__main__':
    main()
