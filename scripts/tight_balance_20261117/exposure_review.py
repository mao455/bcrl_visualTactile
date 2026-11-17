"""Actual logged planning exposure, separated from independent legal supervision."""
import json,collections
import torch,h5py,numpy as np
from scripts.tight_balance_20261117.offline_phases import ROOT,OUT
from scripts.tight_balance_20261117.diagnose import phases
from scripts.tight_layout_20261116.publication import write,table
from visuotactile.anchor_sampling import supervision_rows
from visuotactile.target_windows import window_mask
from visuotactile.geometry_sampling import STAGES

def main():
 rows=[];byepisode=[]
 for role in 'AB':
  data=json.loads((ROOT/role/'state_data/data.json').read_text());records=torch.load(ROOT/role/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records'];best=torch.load(ROOT/role/'training/best.pt',map_location='cpu',weights_only=False,mmap=True)['updates']
  phase={};source={};onset={};acc=collections.defaultdict(collections.Counter);unique=collections.defaultdict(set)
  for i in data['splits']['train']:
   e=data['entries'][i];source[i]=e.get('source_group','original')
   with h5py.File(e['path']) as f:phase[i]=phases(f['episodes'][e['episode']]['transition'])
   mask,_=supervision_rows(records[i]);onset[i]=int(mask.nonzero()[0])
  for line in (ROOT/role/'training/learning.jsonl').open():
   row=json.loads(line)
   for v in row['visits']:
    i=v['episode'];mask,seg=supervision_rows(records[i]);anchors=torch.tensor(v['anchor_sampling']['selected_anchors']);valid=window_mask(mask,seg,anchors,24,'recovery_onset_commit12_v1');ticks=(anchors[:,None]+torch.arange(24)[None])[valid].numpy();assert len(ticks)==v['target_ticks']
    for scope in ['actual']+(['selected'] if row['updates']<=best else []):
     c=acc[scope,i];c.update(visits=1,windows=len(anchors),slots=len(ticks),onset24_slots=int((ticks<onset[i]+24).sum()) if source[i]!='original' else 0,original_early_slots=int((ticks<150).sum()) if source[i]=='original' else 0)
     for k,name in enumerate(STAGES):c[name]+=int((phase[i][ticks]==k).sum())
     unique[scope,i,'ticks'].update(ticks.tolist());unique[scope,i,'starts'].update(anchors.tolist())
  for (scope,i),c in acc.items():byepisode.append(dict(role=role,scope=scope,source=source[i],identity=i,family=data['entries'][i]['family'],unique_supervised_ticks=len(unique[scope,i,'ticks']),unique_window_starts=len(unique[scope,i,'starts']),**c))
  for scope in ['actual','selected']:
   for src in ['original','A','B']:
    ii=[i for i in data['splits']['train'] if source[i]==src];c=collections.Counter()
    for i in ii:c.update(acc[scope,i])
    rows.append(dict(role=role,scope=scope,source=src,trajectories=len(ii),families=len({data['entries'][i]['family'] for i in ii}),unique_supervised_ticks=sum(len(unique[scope,i,'ticks']) for i in ii),unique_window_starts=sum(len(unique[scope,i,'starts']) for i in ii),**c))
 table(OUT/'actual_phase_exposure.csv',rows);table(OUT/'actual_episode_exposure.csv',byepisode)
 write(OUT/'actual_phase_exposure.json',dict(rows=rows,scope='Actual logged optimizer trajectory visits. Windows/slots repeat; unique counts are per trajectory not new independent states. Geometry proxies exclusive; onset overlaps phase. DP A/B label spaces remain distinct.'))
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
