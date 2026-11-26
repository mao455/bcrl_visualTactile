from pathlib import Path
import json,time
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127';D=P/'outputs/diagnostics/tight_correction_20261127_01';root=R/'short_old_only';read=lambda p:json.loads(p.read_text())
assert read(R/'owners/short_old_only_retry01/project_exit.json')['children_reaped']
archive=root/'attempt_preflight02';archive.mkdir();moved=[]
for role in 'AB':
 a=archive/role;a.mkdir()
 assert not (root/role/'training/learning.jsonl').exists() and not (root/role/'training/last.pt').exists()
 for name in ('cache_acceptance_index.json','cache_progress_index.json','cache_index.json'):
  p=root/role/name
  if p.exists():
   moved.append(dict(role=role,path=str(p),sha256=sha256(p),arithmetic=read(p).get('visual_arithmetic')));p.rename(a/name)
owner=R/'owners/short_old_only_retry02';owner.mkdir(parents=True);(owner/'resources').mkdir();s=read(root/'session.json');s.update(resource_root=str(owner),owned_worker_roots=[str(root),str(owner)],recovery_registered_utc=time.time());atomic_json(owner/'session.json',s)
for name in ('shared_resources.json','resources'):
 p=root/name;assert p.is_symlink();p.rename(archive/(name+'.prior_link'));p.symlink_to(owner/name,target_is_directory=name=='resources')
job=read(C/'short_old_only.json');job['resource_root']=str(owner);atomic_json(C/'short_old_only_retry02.json',job)
atomic_json(archive/'repair_receipt.json',dict(utc=time.time(),optimizer_updates=0,reason='Original zero-update attempt also left a cache progress index without arithmetic identity; all three canonical index names are now quarantined',moved=moved))
p=D/'continue_pipeline01.py';script=p.read_text().replace('short_old_only_retry01','short_old_only_retry02').replace("attempt='recovery01'","attempt='recovery02'");(D/'continue_pipeline02.py').write_text(script)
print('all three cache-index names checked; no optimizer updates; original deadline retained')
