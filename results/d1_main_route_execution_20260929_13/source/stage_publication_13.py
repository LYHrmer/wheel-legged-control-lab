"""Copy the explicitly selected completed stage13 artifacts without overwriting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = Path('/home/lyh/wheel-legged-control-lab/results/d1_main_route_execution_20260929_13')


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main() -> None:
    selection = json.loads((HERE / 'publication_selection_13.json').read_text())
    copied = []
    for entry in selection['files']:
        source = HERE / entry['source']
        target = DEST / entry['target']
        if not source.resolve().is_relative_to(HERE) or source.is_symlink():
            raise ValueError('source outside new continuation13')
        if not target.resolve().is_relative_to(DEST):
            raise ValueError('target outside new stage package')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        if identity(source) != identity(target):
            raise RuntimeError('publication copy identity differs')
        copied.append({'source': str(source), 'target': str(target.relative_to(DEST)),
                       **identity(target)})
    with (DEST / 'copy_provenance.json').open('x') as stream:
        json.dump({'schema': 'd1-stage13-copy-v1', 'files': copied}, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
