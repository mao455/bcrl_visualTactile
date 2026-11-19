"""Read-only, non-following inventory; allocation counts each inode once."""
import collections,gzip,json,os,stat,time
from pathlib import Path
P=Path.cwd();OUT=P/'research_docs/evidence/tight_disk_cleanup_20261119'
raw=P/'outputs/disk_audit_20261119';raw.mkdir(exist_ok=True)
seen=set();groups=collections.defaultdict(lambda:[0,0,0]);folders=collections.Counter();large=[];links=[];errors=[];by_size=collections.defaultdict(list);ptgroups=collections.Counter()
with gzip.open(raw/'project_files.jsonl.gz','wt') as f:
 for base,dirs,files in os.walk(P,followlinks=False):
  for name in list(dirs)+files:
   p=Path(base)/name
   try:s=p.lstat()
   except OSError as e:errors.append([str(p),str(e)]);continue
   rel=p.relative_to(P)
   if stat.S_ISLNK(s.st_mode):
    links.append(dict(path=str(rel),target=os.readlink(p),resolved=str(p.resolve()),exists=p.exists()));continue
   if not stat.S_ISREG(s.st_mode):continue
   key=(s.st_dev,s.st_ino);unique=key not in seen;seen.add(key)
   row=dict(path=str(rel),size=s.st_size,allocated=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,mode=stat.S_IMODE(s.st_mode),unique_inode=unique)
   f.write(json.dumps(row)+'\n')
   kind='other'
   if p.suffix=='.h5':kind='raw_hdf5'
   elif 'perception' in p.parts and p.suffix in ('.pt','.zst'):kind='visual_features'+p.suffix
   elif p.suffix=='.pt':kind='tensor_or_checkpoint'
   elif p.suffix=='.partial':kind='partial_evidence'
   elif p.suffix in ('.mp4','.png','.jpg'):kind='video_image'
   elif p.suffix in ('.jsonl','.csv','.log','.gz','.zst'):kind='logs_or_archives'
   if unique:
    groups[kind][0]+=1;groups[kind][1]+=s.st_size;groups[kind][2]+=s.st_blocks*512
    for n in range(1,min(len(rel.parts),5)):folders[str(Path(*rel.parts[:n]))]+=s.st_blocks*512
    if p.suffix=='.pt':ptgroups[str(rel.parent).replace('/A/','/{role}/').replace('/B/','/{role}/')]+=s.st_blocks*512
   if s.st_size>=100*2**20:large.append(row)
   if s.st_size>=2**20 and unique and p.suffix in ('.pt','.h5','.npz','.zst','.partial','.pth'):by_size[s.st_size].append(row)
summary=dict(utc=time.time(),project=str(P),unique_regular_files=len(seen),groups=groups,folders=folders.most_common(),tensor_groups=ptgroups.most_common(),errors=errors,symlinks=len(links),large_files=len(large))
(OUT/'project_inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'large_files.json').write_text(json.dumps(sorted(large,key=lambda x:-x['allocated']),indent=2)+'\n')
(raw/'symlinks.json').write_text(json.dumps(links,indent=2)+'\n')
(raw/'duplicate_size_candidates.json').write_text(json.dumps([v for v in by_size.values() if len(v)>1],indent=2)+'\n')
print(json.dumps(dict(groups=groups,tensor_groups=ptgroups.most_common(25),same_size_groups=sum(len(v)>1 for v in by_size.values()),errors=len(errors)),indent=2))
