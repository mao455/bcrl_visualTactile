"""Reconstruct a retired immutable visual cache from pinned original observations.

Never train or overwrite any original artifact. A destination in a fresh directory
is mandatory. Old indices must only be reused if the resulting SHA is identical;
otherwise validate the numerical payload and register a new cache identity.
"""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
P=Path(__file__).resolve().parents[3];PUBLIC=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(4*2**20):h.update(b)
 return h.hexdigest()

def rebuild(row,destination):
 import h5py,numpy as np,torch
 from visuotactile.models.native_vision import NativeDinoSpatial,DINO_REVISION
 from visuotactile.native_data import depth_grid
 from visuotactile.checkpointing import save
 from visuotactile.precision import state_fp32
 recipe=row['recipe'];source=recipe['source'];contract=recipe['contract']
 assert not destination.exists() and destination.name==Path(row['original']).name
 assert destination.resolve()!=Path(row['original']).resolve()
 assert sha(source['path'])==source['source_sha256']
 session=json.loads(Path(recipe['session']).read_text())
 assert sha(session['dino_checkpoint'])==contract['dino_sha256']
 assert contract['dino_revision']==DINO_REVISION and contract['stride']==1
 assert not contract['random_augmentation'] and not contract['trainable_projection_cached']
 state_fp32();torch.set_num_threads(2);began=time.time();model=NativeDinoSpatial().cuda().eval();patches=[];depth=[]
 telemetry=[]
 with h5py.File(source['path'],'r') as f:
  g=f['episodes'][source['episode']];meta=json.loads(f.attrs['metadata_json'])
  assert meta['camera_ids']==contract['cameras']
  ticks=np.arange(0,source['length'],contract['stride']);ct=torch.from_numpy(g['obs/camera_time_s'][ticks]);cv=torch.from_numpy(g['obs/camera_valid'][ticks])
  assert cv.all()
  for tick in ticks:
   if time.time()-began>300:raise TimeoutError('Bounded cache reconstruction audit')
   mem={a:b.split()[0] for a,b in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
   free,_=torch.cuda.mem_get_info()
   assert int(mem['MemAvailable'])*1024>10*2**30 and free>4*2**30
   if tick%30==0:telemetry.append(dict(tick=int(tick),utc=time.time(),gpu_free_bytes=free,ram_available_bytes=int(mem['MemAvailable'])*1024))
   with torch.no_grad():patches.append(model(torch.from_numpy(g['frames/rgb_u8'][int(tick)]).cuda()).cpu())
   depth.append(depth_grid(torch.from_numpy(g['frames/depth_m'][int(tick)]),torch.from_numpy(g['frames/depth_valid'][int(tick)])))
 destination.parent.mkdir(parents=True,exist_ok=True)
 save(destination,dict(patches=torch.stack(patches),depth=torch.stack(depth),camera_time=ct,camera_valid=cv))
 return dict(destination=str(destination),sha256=sha(destination),expected_sha256=row['receipt']['sha256'],bytes=destination.stat().st_size,frames=len(ticks),wall_seconds=time.time()-began,telemetry=telemetry,optimizer_updates=0)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--row',type=int,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
 rows=json.loads((PUBLIC/'cold_cache_dependencies.json').read_text())['archives']
 print(json.dumps(rebuild(rows[a.row],a.destination),indent=2))
if __name__=='__main__':main()
