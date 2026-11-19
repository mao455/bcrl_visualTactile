"""Full-byte read-only verification of protected assets and reconstruction inputs."""
import concurrent.futures,hashlib,json,time,argparse
parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["before","after"],default="before");phase=parser.parse_args().phase
from pathlib import Path
P=Path.cwd();OUT=P/'research_docs/evidence/tight_disk_cleanup_20261119';RUN=P/'outputs/visuotactile/run/tight_exposure50_20261119_01'
def read(p):return json.loads(p.read_text())
items={}
def add(path,digest,category):
 p=Path(path).resolve();s=p.stat();key=str(p)
 if key in items:
  assert items[key]['expected_sha256']==digest;items[key]['categories'].append(category);return
 items[key]=dict(path=key,expected_sha256=digest,categories=[category],size=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns)
for path,h in read(P/'research_docs/evidence/tight_exposure50_20261119_01/protected.json').items():add(path,h,'protected_model_statistics_manifest')
for r in read(P/'research_docs/evidence/tight_exposure50_20261119_01/interruption_checkpoint_audit.json')['checkpoints']:add(RUN/'A/training'/r['file'],r['sha256'],'current_A_DP_checkpoint')
for r in read(P/'research_docs/evidence/tight_exposure50_20261119_01/raw_integrity.json')['rows']:add(r['path'],r['source_sha256'],'current_raw130_plus_validation6')
for r in read(RUN/'A/cache_index.json')['entries'].values():add(r['path'],r['sha256'],'current_visual136')
manifest=P/'outputs/visuotactile/data/close_v4/manifest.json'
for r in read(manifest)['episodes']:
 for name,f in r['files'].items():add(manifest.parent/r['directory']/name,f['sha256'],'original400')
for row in read(OUT/'cold_cache_dependencies.json')['archives']:
 if row['reasons']:continue
 r=row['recipe']['source'];add(r['path'],r['source_sha256'],'cold_cache_retained_raw_source')
 s=read(Path(row['recipe']['session']));add(s['dino_checkpoint'],row['recipe']['contract']['dino_sha256'],'cold_cache_encoder')

def verify(row):
 p=Path(row['path']);h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*2**20):h.update(b)
 s=p.stat();row.update(actual_sha256=h.hexdigest(),passed=h.hexdigest()==row['expected_sha256'] and s.st_size==row['size'] and s.st_mtime_ns==row['mtime_ns'])
 return row
began=time.time();results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for row in pool.map(verify,items.values()):
  results.append(row)
  if len(results)%50==0:
   (OUT/('asset_verification_progress.json' if phase=='before' else 'asset_verification_after_progress.json')).write_text(json.dumps(dict(checked=len(results),total=len(items),bytes=sum(r['size'] for r in results),utc=time.time()))+'\n')
summary=dict(utc=time.time(),wall_seconds=time.time()-began,passed=all(r['passed'] for r in results),files=len(results),bytes=sum(r['size'] for r in results),rows=results)
(OUT/('protected_and_rebuild_hashes.json' if phase=='before' else 'protected_and_rebuild_hashes_after.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}));assert summary['passed']
