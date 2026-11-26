from pathlib import Path
import json,copy,time
from tools.tight_machine_review import session_at,sources
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127'
owner=R/'development_retry02';s=session_at(owner,duration=12*3600);stages=[]
for i in range(5,8):
 for role in 'AB':
  p=json.loads((R/f'development_before/wave{i:02d}/plan.json').read_text());out=owner/f'wave{i:02d}_{role}';out.mkdir()
  p['cases']=[c for c in p['cases'] if c['role'].startswith(role+'_')];p['pairs']={r:v for r,v in p['pairs'].items() if r.startswith(role+'_')}
  assert len(p['cases'])==2
  p.update(resource_root=str(owner),source_sha256=sources());p['batch'].update(lanes=2)
  p['execution_backend'].update(kit_tasking_threads=4,physics_threads=2)
  atomic_json(out/'session.json',s);atomic_json(out/'plan.json',p)
  stages.append(dict(entry='machine_review',args=['worker','--output',str(out),'--mode','autonomous']))
atomic_json(C/'development_retry02.json',dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=stages))
atomic_json(owner/'registration.json',dict(utc=time.time(),reason='Four-policy initialization exceeded10GiB RAM reserve before actions; two-policy/family waves retain same frozen checkpoints and noise, actual layout verification',technical_only_retries=True,unchanged_families=[28911005,28911006,28911007],terminal64=False))
