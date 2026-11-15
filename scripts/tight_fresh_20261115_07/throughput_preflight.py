"""Sequential real acceptance and scaling, before this session's formal cycle.

Never overlap a second Kit with another rollout. Each preserved attempt runs once.
"""
import argparse,json,subprocess,time,copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_expansion import prepare as collect_prepare
from tools.tight_fresh_batch_eval import prepare as probe_prepare
from tools.tight_fresh_batch_run import prepare as evaluation_prepare

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--loop',type=Path,required=True);a=p.parse_args()
read=lambda p:json.loads(p.read_text())
root=a.root;tree=Path.cwd();bundle=Path(__file__).resolve().parent;out=root/'throughput_preflight';out.mkdir(exist_ok=False)
loop=read(a.loop);deadline=read(root/'session.json')['deadline'];backend=read(bundle/'collection_backend.json')
state=dict(started=time.time(),complete=False,stages=[],selection=None)
def run(name,owner,seconds):
 directory=out/name
 cmd=[str(tree/'.venv_isaacsim/bin/python'),'-m','tools.'+owner,'--output',str(directory),'--worker-tree',str(tree),'--startup-seconds','180']
 if owner=='tight_batch_probe_owner':cmd+=['--wall-seconds',str(seconds)]
 # Contain startup peaks as well as the normal continuously monitored reserve.
 unit='tight-07-'+name.replace('_','-')
 cmd=['systemd-run','--user','--wait','--pipe','--service-type=exec','--unit='+unit,
  '--property=WorkingDirectory='+str(tree),'--property=KillMode=control-group',
  '--property=RuntimeMaxSec='+str(seconds+30),'--property=MemoryMax=28G','--property=MemorySwapMax=0',
  '--property=LimitNOFILE=65536','--setenv=OMP_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2']+cmd
 row=dict(name=name,command=cmd,started=time.time());state['stages'].append(row);atomic_json(out/'status.json',state)
 with (out/(name+'.log')).open('x') as log:
  child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=seconds+60)
 row.update(ended=time.time(),returncode=child.returncode)
 receipt=directory/'owner_finished.json';row['receipt']=read(receipt) if receipt.exists() else None
 atomic_json(out/'status.json',state)
 return child.returncode==0 and row['receipt'] is not None and row['receipt']['software_complete']
try:
 collect_prepare(root,out/'render_capture',session_root=root,registry_id='session07_render_acceptance',seeds=[],
  partition_manifest=loop['inputs']['partition_manifest'],frozen_families=loop['inputs']['frozen_families'],
  schedules=['grasp_early'],lanes=2,event_seed=29105000,round_index=15,
  revisit_families=['expanded_session20261115_03_prefetch13_s29000067','expanded_session20261115_03_prefetch13_s29000091'],
  execution_backend=backend,vision_microbatch=2,verification_workers=2,verification_max_pending=4,
  capture_audit_memo=True,queued_gpu=True)
 if not run('render_capture','tight_batch_owner',600):raise RuntimeError('Real capture acceptance failed; retain and inspect, no formal training')
 if read(out/'render_capture/result.json')['accepted_training_trajectories']<2:raise RuntimeError('Too few actual successful expert suffixes for collector acceptance')
 probes={}
 for name,n,tiled in [('tiled16',16,True),('standard32',32,False)]:
  if deadline-time.time()<loop['bounds']['minimum_round_remaining_seconds']+300:
   state['stages'].append(dict(name=name,not_started=True,reason='Preserve the registered full-cycle time budget'));break
  cfg=copy.deepcopy(backend)
  if tiled:cfg.update(version='tight_execution_backend_v3',camera_backend='tiled',tiled_fixed_camera_usd_sync=True)
  probe_prepare(root,out/name,n,29510000,neural_batch=True,max_control_ticks=96,roles=['A'],
   execution_backend=cfg,vision_microbatch=8,gpu_pipeline=True)
  probes[name]=run(name,'tight_batch_probe_owner',240)
 # Standard32 provides a headroom gate, not a claim that tiled or32 is optimal.
 lanes=16;reason='Standard16 retained pending demonstrated larger-lane headroom'
 if probes.get('standard32'):
  samples=read(out/'standard32/resource_samples.json')['samples']
  if samples and min(r['ram_available_bytes'] for r in samples)>=14*1024**3 and min(r['gpu_free_mib'] for r in samples)>=6144:
   lanes=24;reason='32-lane real load passed with reserves;24 formal opportunities fill24 lanes without idle allocation'
 # Recover unresolved round14 executions with its already completed weights.
 # Preserve the original interruption and its completed zero-action rejection.
 previous=Path(read(bundle/'launch_config.json')['recovery_previous_evaluation'])
 old=read(previous/'plan.json');resolved={}
 for case in old['cases']:
  result=previous/case['name']/'result.json'
  if result.exists() and read(result).get('software_complete') is True:resolved[case['name']]=str(result)
 recovered=out/'round14_recovery'
 evaluation_prepare(root,recovered,lanes=8,count=8,seed=28500000,execution_backend=backend,
  vision_microbatch=8,queued_gpu=True)
 new=read(recovered/'plan.json');new['cases']=[c for c in old['cases'] if c['name'] not in resolved]
 new['pairs']=old['pairs'];new['policy_source_root']=str(previous.parent)
 new['recovery']=dict(previous=str(previous),completed_cases_preserved=resolved,
  reason='Original parallel Kit RAM reserve interruption; same weights, unfinished registered families only',
  scope='Fresh real trajectories; no hidden-state restoration, new GPU queue/render execution; full native task horizons')
 atomic_json(recovered/'plan.json',new)
 if not run('round14_recovery','tight_batch_owner',1100):raise RuntimeError('Round14 recovery incomplete; inspect preserved results')
 state['round14_recovery']=dict(output=str(recovered),original_completed=resolved,
  recovered_cases=len(new['cases']),original_attempt_unchanged=True)
 loop['collection']['lanes']=lanes
 state['selection']=dict(formal_lanes=lanes,reason=reason,tiled_adopted=False,
  interpretation='Tiled short probe alone cannot establish complete expert/contact behavior; formal run uses standard cameras')
 atomic_json(a.loop,loop);state.update(complete=True,ended=time.time());atomic_json(out/'result.json',state)
except BaseException as error:
 state.update(error=repr(error),ended=time.time());atomic_json(out/'result.json',state);raise
finally:atomic_json(out/'status.json',state)
