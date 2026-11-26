from pathlib import Path
import json,time,copy
from tools.tight_machine_review import session_at,sources
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127'
owner=R/'grasp_expert_controls';s=session_at(owner,duration=12*3600)
s['authorization']='Corrected-layout original expert positive controls on two existing development failures; no terminal families or training admission'
atomic_json(owner/'session.json',s)
cases=[]
for i in (1,2):
 plan=json.loads((R/f'development_before/wave{i:02d}/plan.json').read_text())
 spec=copy.deepcopy(plan['cases'][0]['spec']);name=f'expert_{spec["seed"]}';spec.update(identity=name,diagnostic_only=True,split='diagnostic_development')
 cases.append(dict(name=name,spec=spec))
atomic_json(owner/'plan.json',dict(mode='expert',cases=cases,source_sha256=sources()))
atomic_json(C/'grasp_expert_controls.json',dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=[dict(entry='machine_review',args=['worker','--output',str(owner),'--mode','expert'])]))
atomic_json(owner/'registration.json',dict(utc=time.time(),families=[28911001,28911002],hypothesis='Compare expert approach/contact/lift with actual failed current students before choosing a correction',terminal64=False,training_admission=False))
print(owner)
