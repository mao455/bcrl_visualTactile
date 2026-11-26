from pathlib import Path
import json,shutil,time
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';D=P/'outputs/diagnostics/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127';read=lambda p:json.loads(p.read_text());root=R/'short_old_only'
assert read(root/'project_exit.json')['children_reaped']
archive=root/'attempt_preflight01';archive.mkdir();receipt=dict(utc=time.time(),optimizer_updates=0,reason='Child visual arithmetic registration absent; all cached feature payloads matched trusted source, training rejected missing index contract before updates',roles={})
trusted=read(R/'reusable_visual_index.json')
for role in 'AB':
 child=root/role;index=read(child/'cache_index.json');assert all(v['sha256']==trusted['entries'][i]['sha256'] for i,v in index['entries'].items())
 assert not (child/'training/learning.jsonl').exists() and not (child/'training/last.pt').exists()
 a=archive/role;a.mkdir();receipt['roles'][role]=dict(cache_index_sha256=sha256(child/'cache_index.json'),trusted_payloads=len(index['entries']))
 (child/'cache_index.json').rename(a/'cache_index.json')
 if (child/'state_data/perception').exists():(child/'state_data/perception').rename(a/'perception')
 if (child/'training').exists():(child/'training').rename(a/'training_preflight')
for condition in ['old_only','with_corrections']:
 for role in 'AB':shutil.copyfile(R/('short_'+condition)/'visual_arithmetic.json',R/('short_'+condition)/role/'visual_arithmetic.json')
# Retain failed owner logs; resume preparation/training under a fresh bounded owner.
owner=R/'owners/short_old_only_retry01';owner.mkdir(parents=True);(owner/'resources').mkdir();s=read(root/'session.json');s.update(resource_root=str(owner),owned_worker_roots=[str(root),str(owner)],recovery_registered_utc=time.time());atomic_json(owner/'session.json',s)
(root/'shared_resources.json').rename(archive/'shared_resources.json');(root/'shared_resources.json').symlink_to(owner/'shared_resources.json')
(root/'resources').rename(archive/'resources');(root/'resources').symlink_to(owner/'resources',target_is_directory=True)
job=read(C/'short_old_only.json');job['resource_root']=str(owner);atomic_json(C/'short_old_only_retry01.json',job)
atomic_json(archive/'repair_receipt.json',receipt)
p=D/'prepare_short_learning.py';script=p.read_text();needle="atomic_json(child/'session.json',s);(child/'shared_resources.json').symlink_to(root/'shared_resources.json');";replacement="shutil.copyfile(root/'visual_arithmetic.json',child/'visual_arithmetic.json')\n  "+needle;assert needle in script;script=script.replace(needle,replacement);p.write_text(script)
p=D/'short_train.py';script=p.read_text();script=script.replace("state_fp32();torch.set_num_threads(2)\n for role in 'AB':", "state_fp32();torch.set_num_threads(2)\n from visuotactile.frozen_payload_audit import FrozenPayloadAudit\n audit=FrozenPayloadAudit()\n arithmetic=json.loads((root/'visual_arithmetic.json').read_text())\n for role in 'AB':\n  assert json.loads((root/role/'visual_arithmetic.json').read_text())==arithmetic, 'Explicit child cache arithmetic required before preparation'",1);script=script.replace("cache_visual_pair(root/role,reuse_index=json.loads((root/role/'continuation.json').read_text())['reuse_index'])", "cache_visual_pair(root/role,reuse_index=json.loads((root/role/'continuation.json').read_text())['reuse_index'],payload_audit=audit)");p.write_text(script)
print(receipt)
