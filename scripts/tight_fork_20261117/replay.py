"""Full-prefix same-sensor/actual-history inference; never a closed-loop score."""
import sys,time,subprocess,traceback
import numpy as np
import torch
from .common import *
from visuotactile.training_resources import atomic_json
from scripts.tight_layout_20261116.publication import table
REPLAY=ROOT/'replay_fp32'
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 return x
def error(a,b):
 if isinstance(a,dict):return max([error(a[k],b[k]) for k in a]+[0.])
 if a is None:return 0. if b is None else float('inf')
 return float((a.cpu().double()-b.cpu().double()).abs().max())
def acknowledge(c,command,applied,stamp):
 # Real external history, even when shadow predicted something different.
 c.base.position_previous=(command.clone(),applied.clone(),stamp.clone())
def worker():
 from visuotactile.gpu_batch_controller import GPUBatchedResidualController
 from visuotactile.state_resources import lease_for
 from visuotactile.precision import state_fp32
 torch.set_num_threads(2);state_fp32()
 lease=lease_for(REPLAY,deadline=read(ROOT/'session.json')['deadline'])
 meta=read(ROOT/'replay_metadata.json');pairs=read(ROOT/'capture/plan.json')['pairs']
 generators=torch.load(ROOT/'initial_generators.pt',weights_only=False)
 summaries=[];allrows=[];outputs={};start=time.time()
 for role,ids in [('A',[0,4]),('B',[2,6])]:
  for stream in ['candidate','baseline']:
   selected=[i+(stream=='baseline') for i in ids];active=[i in selected for i in range(8)]
   for model in ['candidate','baseline']:
    name=role+('_baseline' if model=='baseline' else '')
    with lease.block():
     c=GPUBatchedResidualController(torch.load(pairs[name]['backend'],map_location='cpu',weights_only=False),meta,None,8,neural_batch=True,scheduler_index_mode='host',vision_microbatch=4)
     for i in selected:c.lanes[i].base.generator.set_state(generators[name][i])
    maxima={k:0. for k in ['nominal','delta','plan','command','robot','native','visual_memory','visual_read','rear_state','generator']};rr=[]
    for t in range(121):
     snap=torch.load(ROOT/'inputs'/f'{t:03d}.pt',weights_only=False)
     p=snap['packet'];d=snap['diagnostics']
     with lease.block(),torch.no_grad():
      out=c.act(p,d,active,current_features=snap['features'][selected].cuda())
      for i in selected:
       lane=c.lanes[i];live=snap['controllers'][i]
       assert lane.base.memory.next_tick==t+1 and len(lane.base.memory.robot)==min(t+1,13) and len(lane.base.memory.frames)==min(t+1,3)
       values=dict(nominal=lane.nominal,delta=lane.delta,plan=lane.plan,command=out[i],robot=lane.base.last_robot,native=lane.base.memory.native,visual_memory=lane.base.memory.visual,visual_read=lane.visual,rear_state=lane.state,generator=lane.base.generator.get_state())
       if stream==model:
        for k,v in values.items():maxima[k]=max(maxima[k],error(v,snap['command'][i] if k=='command' else live[k]))
       row=dict(role=role,stream=stream,model=model,lane=i,seed=28824002 if i<4 else 28824006,tick=t,plan_cursor=t%12,robot_length=len(lane.base.memory.robot),frame_length=len(lane.base.memory.frames))
       rr.append({**row,**cpu(values)})
       acknowledge(lane,snap['command'][i],snap['command'][i],p['time_s'][i])
    outputs[f'{role}_{stream}_{model}']=rr
    summaries.append(dict(role=role,stream=stream,model=model,self_replay=stream==model,max_abs_error=maxima))
    atomic_json(REPLAY/'progress.json',dict(finished=summaries,seconds=time.time()-start))
    del c;torch.cuda.empty_cache()
 torch.save(outputs,REPLAY/'outputs.pt')
 # Decomposition along complete recorded histories; hidden states remain model-own.
 for role,seed,_ in CASES:
  indexed={ (s,m):{r['tick']:r for r in outputs[f'{role}_{s}_{m}'] if r['seed']==seed} for s in ['baseline','candidate'] for m in ['baseline','candidate']}
  for t in range(121):
   row=dict(role=role,seed=seed,tick=t,plan_cursor=t%12)
   for k in ['nominal','delta','command']:
    oo=indexed['baseline','baseline'][t][k].double();no=indexed['baseline','candidate'][t][k].double();nn=indexed['candidate','candidate'][t][k].double();on=indexed['candidate','baseline'][t][k].double()
    diffs=dict(model_on_baseline=no-oo,feedback_on_candidate=nn-no,total=nn-oo,model_on_candidate=nn-on,feedback_on_baseline=on-oo)
    for label,v in diffs.items():row[k+'_'+label+'_rms']=float(v.square().mean().sqrt())
    row[k+'_decomposition_max']=float(((no-oo)+(nn-no)-(nn-oo)).abs().max())
    row[k+'_model_feedback_cosine']=float(((no-oo)*(nn-no)).sum()/((no-oo).norm()*(nn-no).norm()).clamp_min(1e-30))
   allrows.append(row)
 table(OUT/'same_input_ticks.csv',allrows)
 atomic_json(OUT/'same_input.json',dict(self_replays=summaries,seconds=time.time()-start,scope='New121tick capture. Actual external command history forced; own hidden states, own previous nominal and own 24-step plan retained; replan every12. No teacher-forced task success claim.',applied_history='Capture numeric chain independently checked requested=applied before replay.',peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 print(summaries)
def run():
 from scripts.tight_layout_20261116.pilot import stop
 assert read(ROOT/'owner_exit.json')['children_reaped'];REPLAY.mkdir()
 session=read(ROOT/'session.json');session.update(resource_root=str(REPLAY),owned_worker_roots=[str(REPLAY)])
 atomic_json(REPLAY/'session.json',session);monitor=child=None
 try:
  with (REPLAY/'monitor.log').open('x') as f:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(REPLAY)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+30
  while not (REPLAY/'shared_resources.json').exists():
   assert monitor.poll() is None and time.time()<until;time.sleep(.2)
  with (REPLAY/'worker.log').open('x') as f:child=subprocess.Popen([sys.executable,'-m',__spec__.name,'worker'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll() is None:
   assert monitor.poll() is None and time.time()<session['deadline'];time.sleep(2)
  assert child.returncode==0
  atomic_json(REPLAY/'complete.json',dict(complete=True,utc=time.time()))
 except BaseException:
  atomic_json(REPLAY/'failure.json',dict(error=traceback.format_exc()));raise
 finally:
  stop(child);(REPLAY/'STOP_MONITOR').touch()
  if monitor:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:stop(monitor)
  atomic_json(REPLAY/'owner_exit.json',dict(children_reaped=True,utc=time.time()))
if __name__=='__main__':globals()[sys.argv[1]]()
