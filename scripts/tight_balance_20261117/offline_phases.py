"""Frozen-parent/candidate phase errors on the legal76, no optimizer or tests."""
import json,time,subprocess,sys,os,traceback
from pathlib import Path
import torch,numpy as np,h5py
from tools.tight_fresh_train import inputs,new_model,supervised_conditions
from visuotactile.anchor_sampling import labels
from visuotactile.offline_planning import predict_contiguous,profile
from visuotactile.frozen_visual_io import FrozenVisualHotset,mapped_profile
from visuotactile.position_control import bounded_targets
from visuotactile.bounded_extension import error_bins,summarize
from visuotactile.precision import state_fp32
from visuotactile.state_resources import lease_for
from scripts.tight_balance_20261117.diagnose import phases,BASE,OUT
from scripts.tight_layout_20261116.publication import write,table
ROOT=BASE.parent/'tight_balance76_20261117_01_recovery01'

def calculate(review):
 state_fp32();session=json.loads((ROOT/'session.json').read_text());lease=lease_for(review,deadline=session['deadline'])
 allrows=[];started=time.time()
 for role in 'AB':
  with lease.block():
   dr,data,records,stats=inputs(ROOT,role)
   loader=FrozenVisualHotset(dr,data['splits']['validation']+data['splits']['train'],max_bytes=4*1024**3)
   masks={}
   for i in data['splits']['train']:
    entry=data['entries'][i];valid=records[i]['supervision_mask'].cpu().numpy() if 'supervision_mask' in records[i] else np.ones(len(records[i]['time']),bool)
    ticks=np.flatnonzero(valid);onset=np.zeros(len(valid),bool);correction=entry.get('trajectory_kind')=='expert_takeover_v1'
    if correction:onset[ticks[:24]]=True
    with h5py.File(entry['path']) as f:phase=phases(f['episodes'][entry['episode']]['transition'])
    masks[i]=dict(original_early=(np.arange(len(valid))<150)&(not correction),correction_onset24=onset,near_alignment=phase==4,after_subgoal=phase==5,all=valid)
  for version,path in [('parent72',BASE/'round05/learning'/role/'training/best.pt'),('candidate76',ROOT/role/'training/best.pt')]:
   rows=[]
   with lease.block(),torch.no_grad():
    model=new_model(role,stats);saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(saved['ema']);model.eval()
    for i in data['splits']['train']:
     lease.check();rec=records[i];ep=loader(i);c,a=supervised_conditions(model,rec,episode=ep)
     pred=predict_contiguous(model,c,a,configuration=profile(32));pred=bounded_targets(pred.flatten(0,1),rec['joint_limits']).reshape_as(pred)
     truth,valid=labels(rec,a,24);mask=valid[:,:12];e=(pred[:,:12]-truth[:,:12])[mask].cpu();tick=(a[:,None]+torch.arange(12,device=a.device)[None])[mask].cpu().numpy()
     row=error_bins(e,tick,masks[i],model.action_scale);rows.append(row)
     allrows.append(dict(role=role,version=version,identity=i,bins=row))
     del ep,c,a,pred,truth,e
    report=dict(role=role,version=version,checkpoint=str(path),selected_updates=saved['updates'],phases=summarize(rows),cache=loader.summary(),elapsed=time.time()-started)
    write(OUT/f'offline_{role}_{version}.json',report);print(json.dumps(report),flush=True)
    del model,saved
   torch.cuda.empty_cache()
  del loader,records;torch.cuda.empty_cache()
 table(OUT/'offline_phase_errors.csv',allrows)
 from scripts.tight_balance_20261117.backend_phases import calculate as backend_phases
 backend_phases(ROOT,lease)
 write(OUT/'offline_complete.json',dict(complete=True,seconds=time.time()-started,scope='Teacher-forced legal76, identical fixed native planning and histories; no test used, no autonomy claim'))
def main():
 review=ROOT/'phase_review';review.mkdir(exist_ok=False)
 session=json.loads((ROOT/'session.json').read_text());session.update(resource_root=str(review),owned_worker_roots=[str(review)])
 write(review/'session.json',session)
 with (review/'monitor.log').open('x') as log:
  monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(review)],stdout=log,stderr=subprocess.STDOUT)
  try:
   until=time.time()+30
   while not (review/'shared_resources.json').exists():
    assert monitor.poll() is None and time.time()<until;time.sleep(.25)
   calculate(review)
  except BaseException:
   write(review/'failure.json',dict(error=traceback.format_exc()));raise
  finally:
   (review/'STOP_MONITOR').touch();monitor.wait(timeout=20)
   write(review/'owner_exit.json',dict(children_reaped=True,monitor_returncode=monitor.returncode,utc=time.time()))
if __name__=='__main__':main()
