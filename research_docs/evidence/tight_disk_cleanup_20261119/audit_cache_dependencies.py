"""Resolve cold derived features to retained raw sources and pinned encoder."""
import json,collections,time
from pathlib import Path
P=Path.cwd();OUT=P/'research_docs/evidence/tight_disk_cleanup_20261119';RUN=P/'outputs/visuotactile/run'
def read(p):return json.loads(p.read_text())
current=read(RUN/'tight_exposure50_20261119_01/A/cache_index.json')
protected={str(Path(r['path']).resolve()) for r in current['entries'].values()}
references=collections.defaultdict(list);errors=[]
for index in RUN.rglob('cache_index.json'):
 data=index.parent/'state_data/data.json'
 if not data.is_file():continue
 try:
  idx=read(index);d=read(data)
  if 'dino_sha256' not in idx:continue
  for identity,row in idx['entries'].items():
   target=Path(row['path']).resolve();source=d['entries'].get(identity)
   if source is None:continue
   references[str(target)].append(dict(index=str(index),data=str(data),identity=identity,source=source,contract={k:v for k,v in idx.items() if k not in ['entries','preserved_orphans']},cache=row,session=str(index.parent/'session.json')))
 except Exception as e:errors.append(dict(index=str(index),error=repr(e)))
rows=[]
for archive in RUN.rglob('*.pt.zst'):
 if archive.parent.name!='perception' or archive.is_symlink():continue
 original=archive.with_suffix('');receipt=original.with_suffix('.pt.archive.json');refs=references.get(str(original.resolve()),[])
 reasons=[]
 if str(original.resolve()) in protected:reasons.append('current cache dependency')
 if not receipt.is_file():reasons.append('missing archive receipt')
 if not refs:reasons.append('missing index plus raw source mapping')
 chosen=None
 for ref in refs:
  source=ref['source'];contract=ref['contract']
  if not Path(source['path']).is_file():continue
  if source.get('source_sha256')!=ref['cache'].get('source_sha256'):continue
  if any(contract.get(k)!=current.get(k) for k in ['dino_revision','dino_sha256','cameras','resolution','patches','depth','stride','random_augmentation','trainable_projection_cached']):continue
  session_path=next((a/'session.json' for a in Path(ref['index']).parents if (a/'session.json').is_file() and read(a/'session.json').get('dino_sha256')==contract['dino_sha256']),None)
  if session_path is None:continue
  chosen=dict(ref,session=str(session_path));break
 if chosen is None:reasons.append('no complete compatible reconstruction inputs')
 r=read(receipt) if receipt.exists() else None;s=archive.stat()
 rows.append(dict(archive=str(archive),original=str(original),size=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,receipt=r,recipe=chosen,reference_count=len(refs),references=[r['index'] for r in refs],reasons=reasons,classification='pending_dependency_hash_and_rebuild_check' if not reasons else 'retain_uncertain'))
result=dict(utc=time.time(),current_paths=sorted(protected),archives=rows,errors=errors)
(OUT/'cold_cache_dependencies.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(archives=len(rows),compatible=sum(not r['reasons'] for r in rows),compatible_gib=sum(r['allocated'] for r in rows if not r['reasons'])/2**30,errors=len(errors),reasons=collections.Counter(x for r in rows for x in r['reasons'])),indent=2))
