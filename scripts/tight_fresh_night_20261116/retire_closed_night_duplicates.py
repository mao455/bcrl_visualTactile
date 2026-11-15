"""After main shutdown, retire only entire-payload-equal last/best snapshots."""
import importlib.util,json,os,shutil,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ROOT=BASE/'tight_fresh_20261116_02'
read=lambda p:json.loads(p.read_text())

def run():
 torch.set_num_threads(2)
 state=read(ROOT/'night/state.json')
 assert state.get('safe_stop') and (ROOT/'monitor_exit.json').exists()
 assert [r['index'] for r in state['rounds'] if r['complete']][:2]==[17,18]
 spec=importlib.util.spec_from_file_location('duplicate_equality',Path(__file__).with_name('retire_duplicate_last.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 output=ROOT/'closed_completed_night_duplicates';output.mkdir(exist_ok=False)
 rows=[];kept=[];before=shutil.disk_usage(ROOT).free
 candidates=[BASE/'tight_fresh_20261116_01/round_16',ROOT/'round_17',ROOT/'round_18']
 try:
  for candidate in candidates:
   for role in 'AB':
    directory=candidate/role/'training';last=directory/'last.pt';best=directory/'best.pt'
    done=read(directory/'finished.json');assert done['complete']
    assert last.is_file() and not last.is_symlink() and last.stat().st_nlink==1
    a=torch.load(last,map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(best,map_location='cpu',mmap=True,weights_only=False)
    same=(a['epoch']==b['epoch'] and a['updates']==b['updates'] and module.equal(a,b))
    counters=dict(last_epoch=a['epoch'],selected_epoch=b['epoch'],last_updates=a['updates'],selected_updates=b['updates'])
    del a,b
    if not same:
     kept.append(dict(path=str(last),reason='Distinct full training state preserved',**counters));continue
    assert sha256(best)==done['checkpoint_sha256']
    s=last.stat();rows.append(dict(path=str(last),retained_best=str(best),retained_sha256=done['checkpoint_sha256'],
     sha256=sha256(last),device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size,
     allocated_bytes=s.st_blocks*512,entire_payload_equal=True,**counters))
  atomic_json(output/'inventory.json',dict(utc=time.time(),rows=rows,preserved=kept,
   scope='Only exact redundant final serializations; full model/EMA/Adam/RNG/history/counters remain in selected best. Raw data and every distinct training state preserved.'))
  targets={r['path'] for r in rows};inodes={(r['device'],r['inode']) for r in rows}
  for process in Path('/proc').iterdir():
   if not process.name.isdigit() or int(process.name)==os.getpid():continue
   try:
    for fd in (process/'fd').iterdir():
     try:s=fd.stat()
     except OSError:continue
     assert (s.st_dev,s.st_ino) not in inodes,'Last snapshot is still open: '+process.name
    maps=(process/'maps').read_text()
    assert not any(path in maps for path in targets),'Last snapshot is still memory mapped: '+process.name
   except (OSError,PermissionError):continue
  removed=[]
  for row in rows:
   path=Path(row['path']);s=path.stat()
   assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(row['device'],row['inode'],row['mtime_ns'],row['bytes'])
   path.unlink();removed.append(row['path']);atomic_json(output/'progress.json',dict(removed=removed))
  atomic_json(output/'finished.json',dict(complete=True,utc=time.time(),removed=removed,preserved=kept,
   retired_allocated_bytes=sum(r['allocated_bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(ROOT).free))
 except BaseException as error:
  atomic_json(output/'failure.json',dict(error=repr(error),utc=time.time()));raise

if __name__=='__main__':run()
