from pathlib import Path
import json,copy,time
from tools.tight_machine_review import session_at,sources
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127';owner=R/'development_retry03';s=session_at(owner,duration=12*3600)
s['resources']['min_ram_gib']=6
s['authorization']='Technical-only same-family development retry;2complete policies perwave and6GiB measured RAM reserve,4GiB GPU reserve; no external process mutation or replacement of physical results'
atomic_json(owner/'session.json',s);stages=[]
for old in sorted((R/'development_retry02').glob('wave*')):
 p=json.loads((old/'plan.json').read_text());out=owner/old.name;out.mkdir();p.update(resource_root=str(owner),source_sha256=sources());p['batch']['writer_bytes']=64<<20
 atomic_json(out/'session.json',s);atomic_json(out/'plan.json',p)
 stages.append(dict(entry='machine_review',args=['worker','--output',str(out),'--mode','autonomous']))
atomic_json(C/'development_retry03.json',dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=stages))
atomic_json(owner/'registration.json',dict(utc=time.time(),technical_only_retries=True,completed_prior_physical_results_preserved=True,families=[28911005,28911006,28911007],ram_reserve_gib=6,old_ram_reserve_gib=10,lanes=2,source_sha256=sources()))
