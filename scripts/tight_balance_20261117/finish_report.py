"""Summarize closed bounded candidate and paired development; never promote."""
import json,time,collections,shutil
from pathlib import Path
import torch,numpy as np
from visuotactile.compact import sha256
from tools.tight_fresh_rollouts import completed_pair
from scripts.tight_balance_20261117.offline_phases import ROOT,BASE,OUT
from scripts.tight_layout_20261116.publication import write,table
from visuotactile.anchor_sampling import supervision_rows
from visuotactile.target_windows import window_mask

def read(p):return json.loads(p.read_text())
def main():
 assert read(ROOT/'followup/complete.json')['complete']
 reg=read(ROOT/'registration.json');assert all(sha256(p)==h for p,h in reg['protected'].items())
 models={};budgets=[];exposures=[];timing=[]
 for role in 'AB':
  models[role]=completed_pair(ROOT,role);data=read(ROOT/role/'state_data/data.json');numeric=torch.load(ROOT/role/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records'];original=set(data['normalization_train_ids'])
  for component,folder in [('DP',ROOT/role/'training'),('Transformer',ROOT/role/'backend'/role/'training')]:
   best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False,mmap=True);finished=read(folder/'finished.json')
   budgets.append(dict(role=role,component=component,selected_epoch=best['epoch'],selected_updates=best['updates'],actual_updates=finished['updates'],finished=finished,checkpoint_sha256=sha256(folder/'best.pt')))
   if component=='DP':
    counts={scope:collections.Counter() for scope in ['actual','selected']};visits=collections.Counter();unique=collections.defaultdict(set)
    for line in (folder/'learning.jsonl').open():
     row=json.loads(line)
     for v in row['visits']:
      i=v['episode'];rec=numeric[i];mask,seg=supervision_rows(rec);first=int(mask.nonzero()[0]);a=torch.tensor(v['anchor_sampling']['selected_anchors']);valid=window_mask(mask,seg,a,24,'recovery_onset_commit12_v1');ticks=(a[:,None]+torch.arange(24)[None])[valid];assert len(ticks)==v['target_ticks']
      c=collections.Counter(windows=len(a),target_slots=len(ticks),original_early_slots=int((ticks<150).sum()) if i in original else 0,correction_onset_slots=int((ticks<first+24).sum()) if i not in original else 0)
      counts['actual'].update(c)
      if row['updates']<=best['updates']:counts['selected'].update(c)
      visits[i]+=1;unique[i].update(ticks.tolist())
    assert counts['selected']['windows']==best['anchors_seen'] and counts['selected']['target_slots']==best['target_ticks']
    assert counts['actual']['windows']==finished['anchors_seen'] and counts['actual']['target_slots']==finished['target_ticks']
    exposures.append(dict(role=role,counts={k:dict(v) for k,v in counts.items()},visits=dict(visits),unique_supervised_ticks=sum(map(len,unique.values()))))
   del best
 for p in sorted((ROOT/'logs').glob('*_complete.json')):timing.append(dict(stage=p.stem,**read(p)))
 for p in sorted((ROOT/'followup').glob('*_exit.json')):timing.append(dict(stage=p.stem,**read(p)))
 rows=[];pairs=[];profiles=[]
 for part in range(4):
  report=read(OUT/f'comparison{part:02d}.json');rows.extend(report['rows']);pairs.append(report['paired']);profiles.append(report['wave_profiles'])
 assert len(rows)==32 and all(p['passed'] for p in pairs)
 counts={}
 for role in ['A','A_baseline','B','B_baseline']:
  rr=[r for r in rows if r['role']==role];assert len(rr)==8 and len({r['seed'] for r in rr})==8
  counts[role]=dict(opportunities=8,legal=sum(r['legal'] for r in rr),success=sum(r['success'] for r in rr),hold120=sum(r.get('hold120',{}).get('sustained',False) for r in rr),subgoal=sum(bool(r.get('native_subgoals')) for r in rr),software_interruptions=sum(not r['software_complete'] for r in rr))
 telemetry=[]
 old=Path(read(ROOT/'recovery.json')['checkpoint']).parents[4] if (ROOT/'recovery.json').exists() else None
 monitors=list(ROOT.rglob('resources/resources.jsonl'))
 if old is not None:monitors+=list(old.rglob('resources/resources.jsonl'))
 for p in sorted(set(monitors)):
  samples=[json.loads(l) for l in p.open()];dt=np.diff([r['utc_epoch'] for r in samples]);telemetry.append(dict(path=str(p),samples=len(samples),maximum_interval_s=float(dt.max()) if len(dt) else None,min_ram_gib=min(r['ram_available_bytes'] for r in samples)/1024**3,min_gpu_free_mib=min(r['gpu_free_mib'] for r in samples if r.get('gpu_free_mib') is not None),peak_gpu_used_mib=max(r['gpu_used_mib'] for r in samples if r.get('gpu_used_mib') is not None)))
 table(OUT/'training_budget.csv',budgets);table(OUT/'actual_exposure.csv',exposures);table(OUT/'stage_timing.csv',timing);table(OUT/'paired32.csv',rows)
 write(OUT/'completed.json',dict(complete=True,models=models,counts=counts,budgets=budgets,exposures=exposures,timing=timing,telemetry=telemetry,paired_checks=pairs,wave_profiles=profiles,protected_hashes_pass=True,new_dagger=False,baseline_not_replaced=True,free_disk_bytes=shutil.disk_usage(ROOT).free,elapsed_since_registered_start_s=time.time()-read(ROOT/'session.json')['started_utc'],limitations='8 same-wave development pairs per role, not statistical proof or a new blind test. No automatic promotion.'))
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
