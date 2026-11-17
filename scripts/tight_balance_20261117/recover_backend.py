"""One explicit disk-stop recovery: exact B state, original deadlines, no DP rerun."""
import json,os,shutil,sys,time,subprocess,traceback
from pathlib import Path
from visuotactile.compact import sha256
from scripts.tight_layout_20261116.publication import write
from scripts.tight_layout_20261116.pilot import stop
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
OLD=BASE/'tight_balance76_20261117_01_recovery01'
ROOT=BASE/'tight_balance76_20261117_01_recovery02'
def read(p):return json.loads(p.read_text())
def links(src,dst,exclude=()):
 dst.mkdir()
 for p in src.iterdir():
  if p.name not in exclude:(dst/p.name).symlink_to(p,target_is_directory=p.is_dir())
def prepare():
 import torch
 assert read(OLD/'owner_exit.json')['children_reaped'] and (OLD/'failure.json').exists()
 assert shutil.disk_usage(OLD).free>17*1024**3
 assert time.time()<read(OLD/'session.json')['training_deadline']-600
 links(OLD,ROOT,{'B','session.json','registration.json','prepared.json','logs','resources','followup','status.json','failure.json','STOP_MONITOR','monitor.json','monitor.log','monitor_exit.json','owner_started.json','owner_exit.json','owner.lock','shared_resources.json','resource_decisions.jsonl','allocator_2399626.jsonl','allocator_2410026.jsonl'})
 links(OLD/'B',ROOT/'B',{'backend'})
 links(OLD/'B/backend',ROOT/'B/backend',{'B'})
 links(OLD/'B/backend/B',ROOT/'B/backend/B',{'session.json','training'})
 src=OLD/'B/backend/B/training';dest=ROOT/'B/backend/B/training';dest.mkdir()
 for p in src.iterdir():
  if p.is_file():
   if p.suffix=='.pt':os.link(p,dest/p.name)
   else:shutil.copyfile(p,dest/p.name)
 session=read(OLD/'session.json');session.update(resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],recovery_from=str(OLD))
 write(ROOT/'session.json',session)
 s=read(OLD/'B/backend/B/session.json');s.update(resource_root=str(ROOT),owned_worker_roots=[str(ROOT)])
 write(ROOT/'B/backend/B/session.json',s)
 reg=read(OLD/'registration.json');reg.update(root=str(ROOT),recovery_from=str(OLD));write(ROOT/'registration.json',reg)
 (ROOT/'logs').mkdir()
 for p in (OLD/'logs').glob('*_complete.json'):shutil.copyfile(p,ROOT/'logs'/p.name)
 saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=False,mmap=True)
 write(ROOT/'recovery.json',dict(reason='Disk reserve stopped original B backend',old_failure=read(OLD/'failure.json'),checkpoint=str(src/'last.pt'),sha256=sha256(src/'last.pt'),updates=saved['updates'],epoch=saved['epoch'],next_episode_group=saved['next_episode_group'],fixed_deadline=session['deadline'],fixed_training_deadline=session['training_deadline'],preserve_original=True))
 write(ROOT/'prepared.json',dict(recovery=True,source=str(OLD),source_sha256={str(p):sha256(p) for folder in ['visuotactile','tools','scripts/tight_balance_20261117','scripts/tight_layout_20261116'] for p in Path(folder).rglob('*.py')}))
 print(ROOT)
def run():
 assert not (ROOT/'owner_started.json').exists()
 for p,h in read(ROOT/'prepared.json')['source_sha256'].items():assert sha256(p)==h,p
 write(ROOT/'owner_started.json',dict(pid=os.getpid(),utc=time.time()));monitor=child=None
 try:
  with (ROOT/'monitor.log').open('x') as log:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],stdout=log,stderr=subprocess.STDOUT)
  until=time.time()+30
  while not (ROOT/'shared_resources.json').exists():
   assert monitor.poll() is None and time.time()<until;time.sleep(.25)
  rear=ROOT/'B/backend/B'
  for name,mode,receipt in [('rear_train_B_recovery','residual-train',rear/'training/finished.json'),('rear_offline_B','residual-offline',rear/'offline/result.json')]:
   assert not receipt.exists();start=time.time()
   args=[sys.executable,'-m','visuotactile.anchor_continue',mode,'--root',str(rear),'--job','image','--output',str(rear/('training' if mode=='residual-train' else 'offline'))]
   with (ROOT/'logs'/(name+'.log')).open('x') as log:child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(ROOT/'logs'/(name+'_started.json'),dict(command=args,pid=child.pid,utc=start))
   while child.poll() is None:
    assert monitor.poll() is None
    if time.time()>read(ROOT/'session.json')['training_deadline']-30 or (ROOT/'STOP').exists():raise TimeoutError('Original deadline/STOP')
    write(ROOT/'status.json',dict(stage=name,pid=child.pid,utc=time.time()));time.sleep(3)
   assert child.returncode==0 and receipt.exists(),name
   write(ROOT/'logs'/(name+'_complete.json'),dict(seconds=time.time()-start,exit_code=0,receipt_sha256=sha256(receipt),utc=time.time()));child=None
  from tools.tight_fresh_rollouts import completed_pair
  pairs={r:completed_pair(ROOT,r) for r in 'AB'}
  write(ROOT/'complete.json',dict(complete=True,pairs=pairs,recovery=True,utc=time.time()))
 except BaseException:
  write(ROOT/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
 finally:
  stop(child);(ROOT/'STOP_MONITOR').touch()
  if monitor is not None:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:stop(monitor)
  write(ROOT/'owner_exit.json',dict(children_reaped=True,monitor_returncode=None if monitor is None else monitor.poll(),utc=time.time()))
if __name__=='__main__':globals()[sys.argv[1]]()
