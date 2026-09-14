"""Publish original side-input records losslessly, reusing immutable sibling blobs."""
import argparse
import hashlib
import json
import lzma
import shutil
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--restore-script', type=Path, required=True)
    args = parser.parse_args()
    root, output, shared = args.input.resolve(), args.output.resolve(), args.shared.resolve()
    if output.parent != shared.parent:
        raise ValueError('shared dependency must be a sibling of the new bundle')
    dependency_raw = (shared/'evidence_index.json').read_bytes()
    dependency = json.loads(dependency_raw)
    by_payload = {metadata['payload_sha256']: (name, metadata) for name, metadata in dependency['blobs'].items()}
    output.mkdir(parents=True, exist_ok=False)
    (output/'blobs').mkdir()
    original_files = [path for path in sorted(root.rglob('*')) if path.is_file() and '__pycache__' not in path.relative_to(root).parts]
    files, blobs, known = {}, {}, {}
    for number, path in enumerate(original_files):
        raw = path.read_bytes()
        record = {'sha256': sha(raw), 'bytes': len(raw)}
        payload = raw
        if path.name in ('source.tar.gz', 'work_source.tar.gz'):
            if raw[:3] != b'\x1f\x8b\x08':
                raise ValueError('expected a gzip source archive')
            record['gzip_mtime_hex'] = raw[4:8].hex()
            payload = raw[:4]+b'\0'*4+raw[8:]
        payload_hash = sha(payload)
        if payload_hash not in known:
            if payload_hash in by_payload:
                old_key, old_metadata = by_payload[payload_hash]
                key = '../'+shared.name+'/'+old_key
                blobs[key] = dict(old_metadata, external_bundle=shared.name, external_blob=old_key)
            else:
                stored, codec = payload, 'raw'
                if len(payload) > 256 and path.suffix not in ('.npz', '.gz'):
                    compressed = lzma.compress(payload, preset=6)
                    if len(compressed) < len(payload):
                        stored, codec = compressed, 'xz'
                key = f'blobs/{payload_hash}.{codec}'
                with (output/key).open('xb') as stream:
                    stream.write(stored)
                blobs[key] = {'codec': codec, 'stored_bytes': len(stored), 'stored_sha256': sha(stored),
                              'payload_bytes': len(payload), 'payload_sha256': payload_hash}
            known[payload_hash] = key
        record['blob'] = known[payload_hash]
        files[path.relative_to(root).as_posix()] = record
        if number % 50 == 0:
            print(json.dumps({'files_processed': number+1, 'files_total': len(original_files)}), flush=True)
    external = {key: value for key, value in blobs.items() if 'external_bundle' in value}
    local = {key: value for key, value in blobs.items() if 'external_bundle' not in value}
    result = {'schema': 'd1-lossless-shared-evidence-bundle-v1', 'files': files, 'blobs': blobs,
              'external_bundles': {shared.name: {'relative_path': '../'+shared.name, 'index_sha256': sha(dependency_raw)}},
              'recorded_runs': sum(path.name == 'states.npz' for path in original_files),
              'original_bytes': sum(row['bytes'] for row in files.values()),
              'local_blob_count': len(local), 'local_blob_bytes': sum(row['stored_bytes'] for row in local.values()),
              'external_blob_count': len(external), 'reused_stored_blob_bytes': sum(row['stored_bytes'] for row in external.values()),
              'normalization': 'Only source.tar.gz/work_source.tar.gz gzip mtime bytes 4:8 are normalized. Each exact original header is retained for reconstruction and original SHA verification. All other bytes are unchanged or losslessly compressed.',
              'excluded': 'Only Python __pycache__; all ten full physical run records and supplied provenance are retained.'}
    dump(output/'evidence_index.json', result)
    shutil.copyfile(args.restore_script, output/'restore_evidence.py')
    print(json.dumps({key: result[key] for key in ('recorded_runs', 'original_bytes', 'local_blob_count', 'local_blob_bytes', 'external_blob_count', 'reused_stored_blob_bytes')}), flush=True)


if __name__ == '__main__':
    main()
