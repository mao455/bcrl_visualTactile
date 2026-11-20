"""Publish small exact cleanup/archival ledgers; no raw/model bytes are copied."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

PUBLIC=Path(__file__).resolve().parent
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def export():
    for name in ('attempt06_records','attempt07_records','attempt08_failure_archive','attempt09_nominal',
                 'attempt10_records','attempt11_nominal'):
        source=ROOT/'resource_cleanup_20261120'/name
        dest=PUBLIC/'resource_cleanup'/name
        dest.mkdir(parents=True,exist_ok=True)
        for filename in ('plan.json','result.json'):
            if (source/filename).exists():
                shutil.copy2(source/filename,dest/filename)
        files={str(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in source.iterdir() if p.is_file()}
        dump(dest/'ledger.json',dict(source=str(source),files=files,utc=(source/'result.json').stat().st_mtime if (source/'result.json').exists() else (source/'manifest.json').stat().st_mtime))
        journal=source/'archive.jsonl'
        if journal.exists():
            rows=[json.loads(line) for line in journal.read_text().splitlines()]
            columns=('path','archive','sha256','archive_sha256','original_bytes','compressed_bytes','saved_allocated_bytes','byte_roundtrip_verified','original_removed')
            with (dest/'archived_files.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
            dump(dest/'restore_manifest.json',dict(files=[dict(path=r['path']) for r in rows if r.get('original_removed')]))
        manifest=source/'manifest.json.gz'
        if manifest.exists():
            with gzip.open(manifest,'rt') as stream: document=json.load(stream)
            with (dest/'deleted_files.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=('path','sha256','size','blocks'),extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(document['files'])
        if name=='attempt06_records' and (source/'apply.log').exists():
            shutil.copy2(source/'apply.log',dest/'failure.log')
    for name in ('ram_recovery02_wait_retirement.json','ram_recovery03.json'):
        if (ROOT/'round02'/name).exists():
            shutil.copy2(ROOT/'round02'/name,PUBLIC/'round02'/name)
    for name in ('supervision_started.json','supervision_latest.json','round2_ram_recovery03_controller.json'):
        if (ROOT/name).exists():
            shutil.copy2(ROOT/name,PUBLIC/'round02'/name)
    print(json.dumps(dict(exported=True,utc=time.time())))


if __name__=='__main__':export()
