"""Archive inactive rejected/intermediate checkpoints without their Adam state.

Preserves BOTH raw model and selected EMA tensors, all contracts/configuration,
training counters/RNG/curves and original checkpoint SHA. Only exact verified
archives allow retirement of the original full optimizer checkpoint. Never
applies to current, source A11/B09, rollback05/06, B13 or round14 weights.
"""
import argparse,json,os,time
from pathlib import Path
import torch
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
NAMES=['tight_fresh_20261114_01/round_02_v2','tight_fresh_20261114_01/round_03','tight_fresh_20261114_01/round_04',
 'tight_fresh_20261114_04/round05_visual_aux_study/control','tight_fresh_20261114_04/round05_visual_aux_study/auxiliary',
 'tight_fresh_20261114_05/paired_io_compare01/plain','tight_fresh_20261114_05/paired_io_recovery01/paired',
 'tight_fresh_20261115_02/round_12']
PATHS=[BASE/n/r/'training/best.pt' for n in NAMES for r in 'AB']
PATHS+=[BASE/'tight_fresh_20261115_03/round_13/A/training/best.pt']
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
if not a.apply:
 a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for f in PATHS:
  assert f.is_file() and not f.is_symlink();done=json.loads(f.with_name('finished.json').read_text());assert done['complete']
  digest=sha256(f);assert digest==done['checkpoint_sha256']
  s=f.stat();rows.append(dict(path=str(f),sha256=digest,bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
 atomic_json(a.output/'inventory.json',dict(utc=time.time(),rows=rows,scope=__doc__))
else:
 assert not (a.output/'applied.json').exists();inventory=json.loads((a.output/'inventory.json').read_text());rows=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except OSError:continue
  assert not any(str(Path(r['path']).parent) in cmd for r in inventory['rows'])
 try:
  for row in inventory['rows']:
   f=Path(row['path']);s=f.stat();assert (s.st_ino,s.st_mtime_ns,s.st_size)==(row['inode'],row['mtime_ns'],row['bytes'])
   archive=f.with_name('best.weights_only.pt');tmp=archive.with_suffix('.tmp');assert not archive.exists() and not tmp.exists()
   saved=torch.load(f,map_location='cpu',mmap=True,weights_only=False);assert 'optimizer' in saved
   reduced={k:v for k,v in saved.items() if k!='optimizer'}
   reduced['archival']=dict(original_sha256=row['sha256'],original_bytes=row['bytes'],optimizer_resume_available=False,
      preserved='All model and EMA tensors plus every non-optimizer field. New filename; not an ordinary resumable checkpoint.')
   torch.save(reduced,tmp)
   with tmp.open('rb') as stream:os.fsync(stream.fileno())
   restored=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False)
   for field in ('model','ema'):
    assert saved[field].keys()==restored[field].keys()
    assert all(torch.equal(v,restored[field][k]) for k,v in saved[field].items())
   os.replace(tmp,archive);digest=sha256(archive)
   proof=dict(**row,archive=str(archive),archive_sha256=digest,archive_bytes=archive.stat().st_size,
      preserved_model_and_ema_exact=True,removed_optimizer_only=True)
   atomic_json(f.with_name('best_archival.json'),proof)
   del restored,reduced,saved
   f.unlink();rows.append(proof);atomic_json(a.output/'progress.json',dict(rows=rows))
 finally:
  atomic_json(a.output/'applied.json',dict(utc=time.time(),complete=len(rows)==len(inventory['rows']),rows=rows,
    reduced_logical_bytes=sum(r['bytes']-r['archive_bytes'] for r in rows)))
print(json.dumps(dict(phase='apply' if a.apply else 'prepare',output=str(a.output))))
