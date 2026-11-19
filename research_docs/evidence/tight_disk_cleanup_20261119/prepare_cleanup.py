"""Prepare an exact-file manifest; refuse any protected/current artifact."""
import hashlib,json,time
from pathlib import Path
P=Path.cwd();OUT=P/'research_docs/evidence/tight_disk_cleanup_20261119'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*2**20):h.update(b)
 return h.hexdigest()
probe=read(OUT/'cache_rebuild_result.json');assert probe['bytewise_equal'] and probe['optimizer_updates']==0
assets=read(OUT/'protected_and_rebuild_hashes.json');assert assets['passed']
verified={r['path']:r for r in assets['rows']};dependencies=read(OUT/'cold_cache_dependencies.json');protected=set(dependencies['current_paths']);rows=[]
for i,r in enumerate(dependencies['archives']):
 assert not r['reasons'];p=Path(r['archive']);original=Path(r['original']);recipe=r['recipe'];s=p.lstat()
 assert p.resolve().is_relative_to(P/'outputs/visuotactile/run') and p.parent.name=='perception' and p.suffixes[-2:]==['.pt','.zst']
 assert not p.is_symlink() and not original.exists() and str(original.resolve()) not in protected
 assert s.st_nlink==1 and s.st_ino==r['inode'] and s.st_mtime_ns==r['mtime_ns'] and s.st_size==r['size']
 source=Path(recipe['source']['path']).resolve();v=verified[str(source)];assert v['passed']
 assert v['expected_sha256']==recipe['source']['source_sha256']==recipe['cache']['source_sha256']
 assert sha(p)==r['receipt']['archive_sha256']
 rows.append(dict(path=str(p),bytes=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,nlink=1,
  sha256=r['receipt']['archive_sha256'],category=1,type='retired_rebuildable_frozen_DINO_visual_cache_archive',action='unlink_exact_file_after_recheck',
  reason='No current136 dependency or trainable features; raw source full SHA verified; frozen encoder SHA and revision plus generator retained; full228-frame sample rebuilt with exact payload SHA',
  original_payload_sha256=r['receipt']['sha256'],source=str(source),source_sha256=v['expected_sha256'],recipe_row=i,retained_receipt=str(original.with_suffix('.pt.archive.json')),
  references=recipe['index'],historical_reference_count=r['reference_count'],historical_replay='Regenerate using rebuild_visual_cache.py before reusing old cache index; old Zstd restore no longer applies to this payload'))
# The successful audit generated this redundant cache; it is disposable and indexed.
p=Path(probe['destination']);s=p.lstat();assert sha(p)==probe['sha256'];rows.append(dict(path=str(p),bytes=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,nlink=1,sha256=probe['sha256'],category=1,type='completed_cache_rebuild_probe_output',action='unlink_exact_file_after_recheck',reason='One fully reproduced disposable feature payload; result/hash and recipe retained'))
code={str(P/x):sha(P/x) for x in ['visuotactile/native_data.py','visuotactile/models/native_vision.py','visuotactile/checkpointing.py','visuotactile/precision.py','research_docs/evidence/tight_disk_cleanup_20261119/rebuild_visual_cache.py']}
manifest=dict(version='exact_verified_derived_cache_retirement_v1',utc=time.time(),project=str(P),rows=rows,code_sha256=code,total_allocated=sum(r['allocated'] for r in rows),optimizer_updates=0,protected_paths=list(verified),current_visual_paths=sorted(protected),rebuild_probe=probe)
(OUT/'cleanup_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(files=len(rows),allocated_gib=manifest['total_allocated']/2**30)))
