#!/usr/bin/env python3
"""Verify or restore lossless records, including index-bound shared blobs."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe record or blob path')
    return Path(*path.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-only', action='store_true')
    action.add_argument('--output', type=Path, help='New directory; full restoration needs about 1.02 GiB')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    index = json.loads((bundle/'evidence_index.json').read_text())
    if index['schema'] != 'd1-lossless-shared-evidence-bundle-v1':
        raise ValueError('unknown bundle schema')
    dependencies = {}
    for name, dependency in index['external_bundles'].items():
        # Only sibling bundles are accepted, never arbitrary upward traversal.
        parts = PurePosixPath(dependency['relative_path']).parts
        if len(parts) != 2 or parts[0] != '..' or parts[1] in ('.', '..'):
            raise ValueError('external bundle must be a named sibling')
        directory = bundle.parent/parts[1]
        if directory.is_symlink() or directory.resolve().parent != bundle.parent:
            raise ValueError('external bundle cannot escape through a symlink')
        raw = (directory/'evidence_index.json').read_bytes()
        if sha(raw) != dependency['index_sha256']:
            raise ValueError(f'external index changed: {name}')
        dependencies[name] = (directory, json.loads(raw))
    if args.output is not None:
        if args.output.exists() or any(p.is_symlink() for p in (args.output, *args.output.parents)):
            raise FileExistsError('output must be new and cannot use symlinks')
        args.output.mkdir(parents=True, exist_ok=False)
    cache, total, external_count = {}, 0, 0
    for name, record in index['files'].items():
        key = record['blob']
        if key not in cache:
            metadata = index['blobs'][key]
            if 'external_bundle' in metadata:
                directory, dependency = dependencies[metadata['external_bundle']]
                old_key = metadata['external_blob']
                expected_path = index['external_bundles'][metadata['external_bundle']]['relative_path']+'/'+old_key
                if key != expected_path:
                    raise ValueError('external reference path disagrees with bound dependency')
                base_metadata = {k: v for k, v in metadata.items() if not k.startswith('external_')}
                if dependency['blobs'].get(old_key) != base_metadata:
                    raise ValueError('external metadata differs from bound index')
                path = directory/relative(old_key)
                external_count += 1
            else:
                path = bundle/relative(key)
            packed = path.read_bytes()
            if len(packed) != metadata['stored_bytes'] or sha(packed) != metadata['stored_sha256']:
                raise ValueError(f'stored blob mismatch: {key}')
            payload = lzma.decompress(packed) if metadata['codec'] == 'xz' else packed
            if len(payload) != metadata['payload_bytes'] or sha(payload) != metadata['payload_sha256']:
                raise ValueError(f'decoded blob mismatch: {key}')
            cache[key] = payload
        raw = cache[key]
        if 'gzip_mtime_hex' in record:
            raw = raw[:4]+bytes.fromhex(record['gzip_mtime_hex'])+raw[8:]
        if len(raw) != record['bytes'] or sha(raw) != record['sha256']:
            raise ValueError(f'original record mismatch: {name}')
        if args.output is not None:
            destination = args.output/relative(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(raw)
        total += len(raw)
    print(json.dumps({'all_original_files_verified': True, 'files': len(index['files']),
                      'original_bytes': total, 'recorded_runs': index['recorded_runs'],
                      'verified_external_blob_count': external_count,
                      'all_external_index_bindings_verified': True,
                      'restored': args.output is not None}))


if __name__ == '__main__':
    main()
