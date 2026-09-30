"""One-shot saved-data comparison; no engine/model imports or physics."""
import gzip
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/home/lyh/wheel-legged-control-lab/runtime/d1_b22/research/continuation13/archive13/atomic_archive_13.py')
SAVED = ROOT.parent/'continuation23/keyboard_gui_01/episode_0'


def main():
    output = ROOT/'io_probe_01'
    output.mkdir(exist_ok=False)
    # Execute the unchanged pure-I/O prefix only; omit the later native-guard import.
    source = ARCHIVE.read_text().split('\nfrom course_impl08.course_native_guard_08 import (', 1)[0]
    namespace = {'__name__': 'frozen_archive_io_only'}
    exec(compile(source, str(ARCHIVE), 'exec'), namespace)
    ArchiveWriter = namespace['ArchiveWriter']

    class FastWriter(ArchiveWriter):
        def commit_native_block(self, path, rows, npz_path=None, arrays=None):
            if npz_path is not None or arrays is not None:
                raise ValueError('saved heldout-only benchmark')
            return self.commit_gzip_rows(path, rows)

        def commit_gzip_rows(self, path, rows):
            def write(stream):
                with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, compresslevel=1) as compressed:
                    for row in rows:
                        compressed.write((json.dumps(row, sort_keys=True, allow_nan=False,
                                                    separators=(',', ':'))+'\n').encode())
            return self._commit([(path, write)])

    payloads = []
    for filename, method in [('native_block_0000.jsonl.gz', 'commit_native_block'),
                             ('controls.jsonl.gz', 'commit_gzip_rows')]:
        content = gzip.decompress((SAVED/filename).read_bytes())
        payloads.append((filename, method, [json.loads(line) for line in content.splitlines()],
                         hashlib.sha256(content).hexdigest()))
    results = []
    for level, writer_type in [(9, ArchiveWriter), (1, FastWriter)]:
        folder = output/f'level{level}'
        folder.mkdir()
        writer = writer_type()
        writes = []
        start = time.monotonic()
        for filename, method, rows, expected in payloads:
            then = time.monotonic()
            getattr(writer, method)(folder/filename, rows)
            writes.append({'file': filename, 'rows': len(rows), 'write_s': time.monotonic()-then,
                           'bytes': (folder/filename).stat().st_size})
        duration = time.monotonic()-start
        for (filename, _, _, expected), row in zip(payloads, writes):
            actual = hashlib.sha256(gzip.decompress((folder/filename).read_bytes())).hexdigest()
            row['decoded_byte_exact'] = actual == expected
            row['manifest_valid'] = namespace['verify_manifest'](folder/(filename+'.manifest.json'))
            assert row['decoded_byte_exact'] and row['manifest_valid']
        results.append({'compression_level': level, 'total_write_s': duration, 'writes': writes})
    saving = results[0]['total_write_s']-results[1]['total_write_s']
    report = {'schema': 'd1-c25-full-saved-archive-probe-v1', 'source': str(ARCHIVE),
              'source_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              'controls_replayed': 723, 'native_rows_replayed': 3615,
              'new_model_calls': 0, 'new_physics_calls': 0,
              'results': results, 'measured_saving_s': saving,
              'required_saving_from_old_active_s': 3.110393989,
              'exceeds_required_saving': saving >= 3.110393989,
              'prediction_only_not_GUI_performance_qualification': True}
    (output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
