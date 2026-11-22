"""Read original HDF5 + historical production loaders, before changing the sampler."""
from pathlib import Path
import sys, json, csv, hashlib
import h5py, numpy as np, torch
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS')
SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS'
RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.native_data import visual_records
from visuotactile.target_windows import with_target_window
from visuotactile.anchor_sampling import labels,validate_training_anchors,sample_training_anchors,coverage_sampling_profile

def read(p):return json.loads(p.read_text())
def run():
 torch.set_num_threads(2)
 old=read(CAN/'outputs/visuotactile/run/tight_exposure50_20261119_01/A/state_data/data.json')
 previous=set(old['splits']['train']);rows=[];files={}
 for n in (1,2):
  root=RUN/f'round{n:02d}/learning';data=read(root/'A/state_data/data.json');new=sorted(set(data['splits']['train'])-previous);previous=set(data['splits']['train'])
  assert len(new)==(29 if n==1 else 21)
  for role in 'AB':
   saved=torch.load(root/role/'training/best.pt',map_location='cpu',weights_only=False,mmap=True)
   records=with_target_window(visual_records(root/role/'state_data',new,'G00'),'recovery_onset_commit12_v1')
   for identity in new:
    entry=data['entries'][identity];rec=records[identity]
    with h5py.File(entry['path']) as f:
     g=f['episodes'][entry['episode']];raw=g['transition/dagger/expert_label_valid'][:].astype(bool);legal=np.flatnonzero(raw)
     assert np.array_equal(raw,rec['supervision_mask'].numpy())
     switch=int(entry['takeover_tick']);first=int(legal[0]);anchor=first//12*12;slot=first-anchor
     assert first==switch+2
     target,mask=labels(rec,torch.tensor([anchor]),24)
     error=None
     try:validate_training_anchors(rec,torch.tensor([anchor]))
     except ValueError as e:error=str(e)
     sampled,receipt=sample_training_anchors(rec,coverage_sampling_profile(),identity=identity,epoch=1,visit_index=1)
     coverage=saved['visit_sampling_coverage'][identity]
     seen=coverage['anchor_ticks']
     source=g['transition/diagnostics_after/joint_position_rad'] if role=='A' else g['transition/target_after_rad']
     loss_ticks=(anchor+torch.where(mask[0])[0]).tolist()
     value_error=float(np.max(np.abs(target[0,slot].numpy()-source[first])))
     assert value_error==0 and not mask[0,:slot].any() and bool(mask[0,slot])
     row=dict(round=n,role=role,identity=identity,path=entry['path'],switch=switch,first_legal=first,native_anchor=anchor,commit_slot=slot,length=len(raw),legal_ticks=len(legal),raw_mask_around_switch=raw[max(0,switch-2):first+3].tolist(),target_window_mask=mask[0].tolist(),loss_absolute_ticks=loss_ticks,first_label_error=value_error,validator_error=error,selected_epoch=saved['epoch'],historical_visits=coverage['visits'],entry_in_historical_selected_anchors=anchor in seen,minimum_historical_anchor=min(map(int,seen)),epoch1_actual_sampler=receipt)
     rows.append(row)
   del records,saved
 for name in ('visuotactile/anchor_sampling.py','visuotactile/target_windows.py','visuotactile/effective_exposure.py','visuotactile/native_data.py'):
  files[name]=dict(path=str(SRC/name),sha256=hashlib.sha256((SRC/name).read_bytes()).hexdigest())
 unique=[r for r in rows if r['role']=='A'];result=dict(source_files=files,unique_corrections=len(unique),nonboundary_corrections=sum(r['commit_slot']>0 for r in unique),role_rows=len(rows),rejected_rows=sum(r['validator_error'] is not None for r in rows),rows=rows)
 (OUT/'independent_phase.json').write_text(json.dumps(result,indent=2)+'\n')
 with (OUT/'entry_coverage_before.csv').open('w') as f:
  fields=['round','role','identity','switch','first_legal','native_anchor','commit_slot','historical_visits','entry_in_historical_selected_anchors','validator_error'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_files')}))
 print(json.dumps(next(r for r in rows if r['identity']=='r10_B_layout10_train_28817004_contact' and r['role']=='B'),indent=2))
if __name__=='__main__':run()
