import json,time,copy
from pathlib import Path
from tools.tight_paired_dagger import ROOT,BASE,AUDIT,CONFIG,read,session_at
from visuotactile.training_expansion import make_registry,registry_binding,read_registry
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
rows=read(ROOT/'experts/results.json');bad='expanded_paired20261122_s28910012'
rejected=[x for x in rows if x['family']==bad]
assert len(rejected)==3 and all(x['result'].get('not_started') and not x['result']['legal'] and not x['accepted'] for x in rejected)
assert not (ROOT/'collection').exists() and not (ROOT/'learning').exists()
assert not (ROOT/'active_training_expansion.json').exists()
session=read(ROOT/'session.json');seeds=[s for s in range(28910000,28910024) if s!=28910012]+[28910024]
registry=make_registry('paired20261122',session['domain'],seeds,{r:BASE/r/'state_data/data.json' for r in 'AB'},AUDIT/'all_historical_partitions.json',AUDIT/'all_frozen_families.json',created_utc=time.time())
registry['admission_amendment']=dict(reason='Seed28910012 rejected by collision gate before any action in all3 registered attempts; use next sequential unused seed28910024',performance_outcomes_used=False,original_registry=registry_binding(ROOT/'training_expansion_registry.json'),excluded_preaction_family=bad,original_expert_result_path=str(ROOT/'experts/results.json'))
p=ROOT/'training_expansion_registry_admission01.json';atomic_json(p,registry);binding=registry_binding(p);read_registry(binding)
atomic_json(ROOT/'active_training_expansion.json',binding)
amend=dict(utc=time.time(),version='preaction_legality_family_replacement_v1',original_registry=registry['admission_amendment']['original_registry'],active_registry=binding,excluded_family=bad,excluded_attempts=rejected,new_seed=28910024,rule='Replace only all-preaction collision-rejected family using next sequential unused seed; retain all attempts; post-action task failures never replace a family',students_started=0,new_optimizer_updates=0)
atomic_json(ROOT/'training_family_admission_amendment.json',amend)
owner=ROOT/'owners/experts_admission01';session_at(owner);output=ROOT/'experts_admission01';session_at(output)
(output/'shared_resources.json').symlink_to(owner/'shared_resources.json')
atomic_json(output/'family_subset.json',['expanded_paired20261122_s28910024'])
atomic_json(CONFIG/'experts_admission01.json',dict(version='tight_project_job_v1',operation='prepare',resource_root=str(owner),stages=[dict(entry='paired_experts',args=['--output',str(output)])]))
print(json.dumps(dict(amended=True,binding=binding,expert_job=str(CONFIG/'experts_admission01.json'))))
