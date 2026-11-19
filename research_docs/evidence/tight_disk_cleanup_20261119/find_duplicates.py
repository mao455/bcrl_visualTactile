"""Identify byte-identical independent files; no mutation."""
import hashlib,json,collections,time
from pathlib import Path
P=Path.cwd();OUT=P/'research_docs/evidence/tight_disk_cleanup_20261119'
groups=json.loads((P/'outputs/disk_audit_20261119/duplicate_size_candidates.json').read_text());result=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*2**20):h.update(b)
 return h.hexdigest()
for group in groups:
 sampled=collections.defaultdict(list)
 for row in group:
  p=P/row['path'];s=p.stat()
  assert s.st_size==row['size'] and s.st_mtime_ns==row['mtime_ns']
  h=hashlib.sha256()
  with p.open('rb') as f:
   for off in [0,max(0,s.st_size//2-32768),max(0,s.st_size-65536)]:f.seek(off);h.update(f.read(65536))
  sampled[h.hexdigest()].append(row)
 for rows in sampled.values():
  if len(rows)<2:continue
  full=collections.defaultdict(list)
  for row in rows:full[sha(P/row['path'])].append(row)
  for digest,rr in full.items():
   if len(rr)>1:result.append(dict(sha256=digest,files=rr,possible_saved=sum(r['allocated'] for r in rr[1:])))
(OUT/'byte_duplicate_groups.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(groups=len(result),possible_saved=sum(r['possible_saved'] for r in result),largest=[dict(sha256=r['sha256'],paths=[f['path'] for f in r['files']],possible_saved=r['possible_saved']) for r in sorted(result,key=lambda x:-x['possible_saved'])[:12]]),indent=2))
