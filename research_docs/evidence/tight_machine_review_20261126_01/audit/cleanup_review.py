"""Dependency-scoped final retirement. Metadata accounting never follows symlinks."""
from pathlib import Path
import os,sqlite3,json,time,stat,collections,ctypes,struct,hashlib,gzip
R=Path.cwd();O=R/'outputs/diagnostics/tight_machine_review_20261126_01';W=R/'outputs/maintenance_20261123/round2';P=R/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
G=2**30
libc=ctypes.CDLL(None,use_errno=True)
def birth(p):
 b=ctypes.create_string_buffer(256)
 if libc.statx(-100,os.fsencode(p),0x100,0x800,b)==0 and struct.unpack_from('I',b,0)[0]&0x800:
  sec,nsec=struct.unpack_from('qI',b,80);return sec+nsec/1e9
 return None
def read(p):return json.loads(Path(p).read_text())
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def snap(name):
 c=sqlite3.connect(O/(name+'.sqlite'));c.execute('create table files(path text primary key,device int,inode int,mode int,nlink int,size int,allocated int,mtime_ns int,ctime_ns int,birth_utc real)')
 stack=[R];rows=[];seen=set();total=0;count=0
 while stack:
  d=stack.pop()
  for e in os.scandir(d):
   if e.path.startswith(str(O/(name+'.sqlite'))):continue
   try:s=e.stat(follow_symlinks=False)
   except FileNotFoundError:continue
   p=Path(e.path);rel=str(p.relative_to(R));rows.append((rel,s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_blocks*512,s.st_mtime_ns,s.st_ctime_ns,birth(p)));count+=1
   key=s.st_dev,s.st_ino
   if key not in seen:seen.add(key);total+=s.st_blocks*512
   if stat.S_ISDIR(s.st_mode):stack.append(p)
   if len(rows)>=3000:c.executemany('insert into files values(?,?,?,?,?,?,?,?,?,?)',rows);rows=[]
 if rows:c.executemany('insert into files values(?,?,?,?,?,?,?,?,?,?)',rows)
 c.commit();c.close();return dict(utc=time.time(),allocated_bytes=total,GiB=total/G,paths=count,unique_inodes=len(seen))
def prepare():
 current=sqlite3.connect(O/'disk_before.sqlite');current.row_factory=sqlite3.Row
 old=sqlite3.connect(W/'snapshot.sqlite');old.row_factory=sqlite3.Row
 oldrows={x['path']:dict(x) for x in old.execute('select * from final')}
 byrun=collections.defaultdict(lambda:dict(new_paths=0,removed_paths=0,new_inode_bytes=0,old_inode_bytes=0))
 oldinode={(x['device'],x['inode']):x['allocated'] for x in oldrows.values()}
 nowrows=[dict(x) for x in current.execute('select * from files')];nowset={x['path'] for x in nowrows};newinode={};seenold=set()
 def run(p):
  a=p.split('/');return a[3] if a[:3]==['outputs','visuotactile','run'] and len(a)>3 else '/'.join(a[:2])
 for x in nowrows:
  g=byrun[run(x['path'])];key=x['device'],x['inode']
  if x['path'] not in oldrows:g['new_paths']+=1
  if key not in newinode:
   newinode[key]=x['allocated'];g['new_inode_bytes']+=x['allocated']
 for p,x in oldrows.items():
  g=byrun[run(p)];key=x['device'],x['inode']
  if p not in nowset:g['removed_paths']+=1
  if key not in seenold:g['old_inode_bytes']+=x['allocated'];seenold.add(key)
 write('disk_since_round2_by_run.json',{k:dict(v,delta_GiB=(v['new_inode_bytes']-v['old_inode_bytes'])/G) for k,v in byrun.items()})
 # Protect runtime inputs and every complete scientific model/valid trajectory.
 # All candidates below are retired derived nominal tensors, unfinished writes, or redundant last weights.
 candidates=[]
 for x in nowrows:
  p=R/x['path'];rel=x['path']
  if not stat.S_ISREG(x['mode']):continue
  reason=None
  if rel.startswith('outputs/visuotactile/run/') and '/nominal/' in rel and p.suffix=='.pt':reason='Closed backend training; offline geometry/action diagnosis uses raw source and full policy, no old nominal consumer; regenerable from preserved pair'
  if rel.startswith('outputs/visuotactile/') and (p.name.endswith('.partial') or '.zst.partial-' in p.name):reason='Closed interrupted writer; no continuation and no accepted pool entry points to partial payload; preserve core numeric/result/seed/failure receipts'
  if rel.startswith(str(P.relative_to(R))) and (p.name in ('comparison_epoch_60.pt','validation_best.pt') or p.name=='last.pt'):
   if (p.parent/'best.pt').exists():reason='Closed training intermediate/redundant durable last; deployed best and actual resume boundary receipt preserved'
  if reason:candidates.append(dict(x,reason=reason,birth_utc=birth(p),mtime_utc=x['mtime_ns']/1e9))
 # Parse actual pools (including features) to forbid deleting referenced entries.
 protected=set()
 for role in 'AB':
  data=read(P/'learning'/role/'state_data/data.json')
  for entry in data['entries'].values():
   for k in ('path','perception_path','feature_path'):
    if entry.get(k):protected.add(str(Path(entry[k]).resolve()))
  for index in (P/'learning'/role/'state_data').rglob('*index*.json'):
   def paths(v):
    if isinstance(v,str) and v.startswith(str(R)) and Path(v).exists():protected.add(str(Path(v).resolve()))
    elif isinstance(v,dict):
     for y in v.values():paths(y)
    elif isinstance(v,list):
     for y in v:paths(y)
   paths(read(index))
 # Check actual formal pairs, which always name best.pt.
 for src in [P/'parent_pairs.json',P/'evaluation/test/registration.json',R/'outputs/visuotactile/run/tight_entry_final24_20261122_01/registration.json']:
  a=read(src);a=a.get('pairs',a)
  for v in a.values():
   if isinstance(v,dict):
    for k in ('frontend','backend'):
     if v.get(k):protected.add(str(Path(v[k]).resolve()))
 assert not any(str((R/x['path']).resolve()) in protected for x in candidates)
 # File-descriptor guard, without signalling any process.
 openinodes=set()
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   for fd in (proc/'fd').iterdir():
    try:s=fd.stat();openinodes.add((s.st_dev,s.st_ino))
    except OSError:pass
  except OSError:pass
 candidates=[x for x in candidates if (x['device'],x['inode']) not in openinodes]
 write('retirement_plan.json',dict(utc=time.time(),files=candidates,protected_input_paths=sorted(protected),
  excluded=['original400','current261 raw and frozen features','all complete scientific best endpoints','all success/failure core numeric and results','current diagnostic captures'],
  qualification='Nominal and partial retirement only after train/dev reconstruction retained; no classification by failed substring',
  previous_release_GiB_not_recounted=407.722))
 print('plan',len(candidates),'path GiB',sum(x['allocated'] for x in candidates)/G,flush=True)
def apply():
 plan=read(O/'retirement_plan.json');deleted=[];total=0
 for x in plan['files']:
  p=R/x['path'];s=p.lstat()
  assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(x['device'],x['inode'],x['mtime_ns'],x['size'])
  x['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();x['deleted_utc']=time.time()
  released=s.st_blocks*512 if s.st_nlink==1 else 0
  p.unlink();x['released_bytes']=released;total+=released;deleted.append(x)
  with (O/'retirement_deletions.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 write('retirement_result.json',dict(utc=time.time(),files=len(deleted),released_bytes=total,released_GiB=total/G))
 # Append this scope to the original ledger, never alter historic release records.
 c=sqlite3.connect(R/'outputs/maintenance_20261123/inventory.sqlite')
 c.execute('create table if not exists cleanup_machine_review_20261126(path text primary key,inode int,device int,allocated int,released int,birth_utc real,mtime_utc real,deleted_utc real,reason text,sha256 text)')
 c.executemany('insert into cleanup_machine_review_20261126 values(?,?,?,?,?,?,?,?,?,?)',[(x['path'],x['inode'],x['device'],x['allocated'],x['released_bytes'],x['birth_utc'],x['mtime_utc'],x['deleted_utc'],x['reason'],x['sha256']) for x in deleted]);c.commit();c.close()
 print('deleted',len(deleted),'released GiB',total/G,flush=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='apply':apply()
 elif sys.argv[1]=='snapshot':write('disk_after.json',snap('disk_after'))
