"""Bounded-memory audit of immutable baseline assets; no CUDA or mutation."""
import json,time
from pathlib import Path
from visuotactile.compact import sha256
from tools.tight_adaptive_dagger import BASE,ROOT,PUBLIC,read,write
started=time.time();data=read(BASE/'A/state_data/data.json');rows=[]
for identity,entry in data['entries'].items():
    path=Path(entry['path']);actual=sha256(path)
    assert actual==entry['source_sha256'],identity
    rows.append(dict(identity=identity,path=str(path),bytes=path.stat().st_size,sha256=actual,kind='raw'))
    write(ROOT/'asset_audit_progress.json',dict(verified=len(rows),identity=identity,utc=time.time()))
index=read(BASE/'A/cache_index.json')
seen=set()
for identity,entry in index['entries'].items():
    path=Path(entry['path']).resolve()
    if path in seen:continue
    seen.add(path);actual=sha256(path)
    assert actual==entry['sha256'],identity
    rows.append(dict(identity=identity,path=str(path),bytes=path.stat().st_size,sha256=actual,kind='frozen_visual'))
write(PUBLIC/'asset_audit.json',dict(passed=True,raw=sum(r['kind']=='raw' for r in rows),visual=len(seen),rows=rows,seconds=time.time()-started,cuda_initialized=False))
print('Verified',len(rows),'immutable raw/cache assets',flush=True)
