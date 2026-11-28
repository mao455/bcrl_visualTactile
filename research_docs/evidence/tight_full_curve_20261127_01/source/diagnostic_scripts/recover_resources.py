from pathlib import Path
import copy,json,time
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_full_curve_20261127_01').resolve();C=Path('configs/experiments/full_curve_20261127');read=lambda p:json.loads(p.read_text())
owner=R/'evaluation_e00_start_retry02';owner.mkdir();s=copy.deepcopy(read(R/'session.json'));s['resource_root']=str(owner);atomic_json(owner/'session.json',s)
out=owner/'rollouts';out.mkdir();previous=R/'evaluation_e00_start_retry01/rollouts';plan=read(previous/'plan.json');finished={p.parent.name for p in previous.glob('*/result.json') if read(p).get('software_complete')};cases=[c for c in plan['cases'] if c['name'] not in finished];assert len(cases)==21
plan.update(cases=cases,resource_root=str(owner));atomic_json(out/'plan.json',plan);atomic_json(out/'session.json',s)
atomic_json(owner/'retry.json',dict(utc=time.time(),reason='Measured concurrent twoDP+two-laneKit exceeded24GiB; retry only21unfinished cases after all trainers exit',original_completed_preserved=sorted(finished),original_denominator=32,technical_attempt_only=True))
atomic_json(C/'evaluation_e00_start_retry02.json',dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=[dict(entry='machine_review',args=['worker','--output',str(out),'--mode','autonomous'])]))
for c in ['old_only','plus_corrected24']:
 root=R/c;old=root/'A/training';assert not (old/'last.pt').exists();old.rename(root/'A/training_abandoned_before_first_update')
 atomic_json(root/'frontend_attempt00_audit.json',dict(utc=time.time(),successful_updates=0,epoch=0,reason='CUDA OOM before first successful optimizer update; immutable failed snapshot retained separately',restart='Own original parent EMA/fresh AdamW/EMA and original schedule/noise seeds; full registered budget unchanged'))
print(owner)
