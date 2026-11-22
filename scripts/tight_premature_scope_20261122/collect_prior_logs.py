"""Losslessly export completed execution logs alongside the new scope audit."""
import gzip
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DEST = PROJECT/'research_docs/evidence/tight_premature_scope_20261122_01/prior_execution_logs'
SOURCES = {
    'final24': PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01',
    'same_history': PROJECT/'outputs/diagnostics/tight_entry_final24_20261122_01',
}


def main():
    records = []
    for label, source in SOURCES.items():
        assert (source/'project_exit.json').exists() or (source/'exit.json').exists()
        for original in sorted(source.rglob('*')):
            if not original.is_file() or original.suffix not in ('.log', '.jsonl'):
                continue
            raw = original.read_bytes()
            target = DEST/label/original.relative_to(source)
            target = target.with_name(target.name+'.gz')
            target.parent.mkdir(parents=True, exist_ok=True)
            compressed = gzip.compress(raw, compresslevel=6, mtime=0)
            target.write_bytes(compressed)
            assert gzip.decompress(compressed) == raw
            records.append(dict(source=str(original), source_bytes=len(raw),
                source_sha256=hashlib.sha256(raw).hexdigest(),
                published_path=str(target.relative_to(DEST)), published_bytes=len(compressed),
                published_sha256=hashlib.sha256(compressed).hexdigest()))
    (DEST/'manifest.json').write_text(json.dumps(dict(
        scope='All .log and .jsonl files in the two completed execution roots; byte-preserving gzip',
        records=records, files=len(records), source_bytes=sum(r['source_bytes'] for r in records),
        published_bytes=sum(r['published_bytes'] for r in records)), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(files=len(records), published_bytes=sum(r['published_bytes'] for r in records))))


if __name__ == '__main__':
    main()
