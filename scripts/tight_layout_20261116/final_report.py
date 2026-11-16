"""Closed final-test results and measured resource evidence; never select models."""
import argparse,json,time
from pathlib import Path
from tools.tight_fresh_grasp_evidence import behavior
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def summarize(root,output):
 read=lambda p:json.loads(p.read_text())
 assert read(root/'complete.json')['complete'] and read(root/'owner_exit.json')['children_reaped']
 registration=read(root/'registration.json');rows=[];parts=[]
 assert len(registration['seeds']) in (8,20)
 for i in range(len(registration['seeds'])//4):
  folder=root/f'part{i}';plan=read(folder/'plan.json')
  assert plan['test_used'] and plan['holdout_contract']=='layout_final_test_v1'
  assert plan['pairs']==registration['pairs']
  assert all(case['spec']['split']=='test' for case in plan['cases'])
  completed=read(folder/'result.json');paired=read(folder/'paired_initial_check.json')
  assert completed['software_complete'] and completed['worker_error'] is None
  assert completed['cases']==8 and paired['passed']
  b=behavior(folder)
  for row in b['rows']:
   result=read(folder/row['case']/'result.json')
   assert result['software_complete']
   row.update(steps=result['steps'],reasons=result['reasons'])
  rows+=b['rows'];parts.append(dict(part=i,counts=b['counts'],
   paired_initial_check=read(folder/'paired_initial_check.json'),result=read(folder/'result.json'),
   result_sha256=sha256(folder/'result.json')))
 assert len(rows)==2*len(registration['seeds']) and sorted({r['seed'] for r in rows})==registration['seeds']
 assert sorted((r['role'],r['seed']) for r in rows)==sorted((role,seed) for role in 'AB' for seed in registration['seeds'])
 counts={}
 for role in 'AB':
  selected=[r for r in rows if r['role']==role];legal=[r for r in selected if r['legal']]
  counts[role]=dict(registered=len(selected),legal=len(legal),illegal=len(selected)-len(legal),
   successes=sum(r['insertion'] for r in legal),
   **{k:sum(bool(r[k]) for r in legal) for k in ['native_alignment','native_subgoals','native_lift']},
   fall=sum(bool(r['reasons']['fall']) for r in legal),timeout=sum(bool(r['reasons']['timeout']) for r in legal))
 for pair in registration['pairs'].values():
  for label in ['frontend','backend']:assert sha256(pair[label])==pair[label+'_sha256']
 sources=[root.parent/'tight_layout10_20261116_learning01',root.parent/'tight_layout10_20261116_dagger01',
  Path(registration['source']),root]
 if registration.get('resource_roots'):sources=[Path(p) for p in registration['resource_roots']]
 resources={str(p):read(p/'resources/resources.json') for p in sources}
 report=dict(utc=time.time(),final_test=counts,rows=rows,parts=parts,
  final_models=registration['pairs'],model_hashes_verified_after_test=True,
  registration_sha256=sha256(root/'registration.json'),resources=resources,
  owner_exit=read(root/'owner_exit.json'),software_interruptions=0,
  scope='Registered untouched paired test families, once per model. No test tuning; no matched old-layout training-budget control. Development success is not a held-out improvement claim.')
 output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,report);print(json.dumps(counts))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();summarize(a.root.resolve(),a.output.resolve())
