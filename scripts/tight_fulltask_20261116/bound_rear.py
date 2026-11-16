"""Change only both unstarted rear training budgets; cached DP inputs unchanged."""
import time
from pathlib import Path
from scripts.tight_fulltask_20261116.run import ROOT,read,atomic_json
from visuotactile.compact import sha256
rows=[]
for role in 'AB':
    base=ROOT/'update'/role/'backend'/role
    assert not (base/'training/configuration.json').exists() and not (base/'training/last.pt').exists(),'Never change a started optimizer budget'
    p=base/'configuration.json';c=read(p);v=c['residual_pair']
    assert v['maximum_control_tick_epochs']==2 and v['minimum_control_tick_epochs']==2
    rows.append(dict(role=role,path=str(p),before=c,before_sha256=sha256(p)))
registration=dict(version='bounded_shared_rear_budget_v1',utc=time.time(),requested_epochs=2,effective_epochs=1,roles=['A','B'],
    reason='B DP I/O stall plus coordinator handoff consumed deadline reserve. Preserve equal completed DP2; reduce both unstarted rears to one full shared-pool epoch and reserve native paired evaluation.',
    unchanged='DP weights/cache generation, data, sampling, LR, optimizer, labels, architecture, execution and deadline',rows=rows)
atomic_json(ROOT/'rear_budget_adjustment.json',registration)
for row in rows:
    c=read(row['path']);v=c['residual_pair'];v['maximum_control_tick_epochs']=v['minimum_control_tick_epochs']=1
    v['budget_adjustment']=dict(version=registration['version'],requested_epochs=2,effective_epochs=1,registration=str(ROOT/'rear_budget_adjustment.json'))
    atomic_json(Path(row['path']),c);row['after_sha256']=sha256(row['path'])
atomic_json(ROOT/'rear_budget_adjustment.json',registration)
