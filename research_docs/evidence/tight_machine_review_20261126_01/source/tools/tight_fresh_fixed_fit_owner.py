"""One bounded train2 fitting diagnosis, serial GPU workers, preserved results."""
import argparse,copy,json,os,time,traceback
from pathlib import Path
from tools.tight_fresh_collect_cycle_owner import run_stage
from tools.tight_fresh_grasp_evidence import behavior
from visuotactile.training_resources import atomic_json
read=lambda p:json.loads(Path(p).read_text())

def prepare_plan(root,backend,output,*,fit=False,fit_root=None):
 import torch
 from tools.tight_fresh_original_probe import register
 from visuotactile.state_benchmark import dagger_policy_identity
 register(root,output,backend)
 plan=read(output/'plan.json')
 for case in plan['cases']:case['spec']['diagnostic_cohort']='fixed_original_train2_fitting_diagnostic'
 if fit:
  for role in 'AB':
   path=(root/'fixed_fit' if fit_root is None else fit_root)/role/'best.pt';saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
   assert saved['diagnostic_only'] and saved['updates']>0
   plan['pairs'][role]=dagger_policy_identity(saved,path)
 plan.update(cohort='Two original training families; '+('updated rear, same frozen paired DP' if fit else 'retained complete source pair'),
  diagnostic_only=True,formal_candidate=False)
 atomic_json(output/'plan.json',plan)

def compare_initials(baseline,candidate):
 import h5py,numpy as np
 from tools.tight_fresh_closed_gap import object_position_in_base
 rows=[]
 for case in read(baseline/'plan.json')['cases']:
  name=case['name'];a=read(baseline/name/'result.json');b=read(candidate/name/'result.json')
  if not(a['legal'] and b['legal']):rows.append(dict(case=name,both_legal=False));continue
  with h5py.File(baseline/name/'numeric.h5') as f,h5py.File(candidate/name/'numeric.h5') as g:
   first='episode/transition/diagnostics_before/'
   q=f[first+'joint_position_rad'][0]-g[first+'joint_position_rad'][0]
   obj=object_position_in_base(f[first+'object_pose_w'][:1],f[first+'base_pose_w'][:1])-object_position_in_base(g[first+'object_pose_w'][:1],g[first+'base_pose_w'][:1])
   rows.append(dict(case=name,both_legal=True,initial_joint_max_abs_rad=float(np.abs(q).max()),
    initial_object_position_gap_m=float(np.linalg.norm(obj)),hidden_state_exact_replay_claim=False))
 return rows

def run(root,configuration):
 cfg=read(configuration);tree=Path.cwd();python=str(tree/'.venv_isaacsim/bin/python')
 ledger=root/'fixed_fit_owner';ledger.mkdir(exist_ok=False)
 session=read(root/'session.json');deadline=min(session['deadline'],cfg['absolute_wall_deadline'])-120
 stops=(root/'STOP',ledger/'STOP');baseline=root/'fixed_fit_source_evaluation';candidate=root/'fixed_fit_evaluation'
 stages=[]
 def stage(name,argv,collection=baseline):
  atomic_json(ledger/'state.json',dict(stage=name,utc=time.time(),owner_pid=os.getpid(),deadline=deadline))
  receipt=run_stage(name,[python,'-m',*map(str,argv)],dict(ledger=ledger,paths=dict(worker_tree=tree,
   collection=collection,output=root)),deadline,stops)
  stages.append(receipt)
  if not receipt['complete']:raise RuntimeError('Bounded fitting stage failed: '+name)
 atomic_json(ledger/'started.json',dict(pid=os.getpid(),utc=time.time(),configuration=cfg,deadline=deadline))
 try:
  stage('prepare_source',['tools.tight_fresh_fixed_fit_owner','prepare','--root',root,'--backend',cfg['backend_json'],'--output',baseline])
  stage('source_autonomy',['tools.tight_batch_owner','--output',baseline,'--worker-tree',tree,'--startup-seconds',300])
  for role in (() if cfg.get('fit_root') else 'AB'):
   stage('fit_'+role,['tools.tight_fresh_fixed_rear_fit','--source',cfg['sources'][role],'--role',role,
    '--resource-root',root,'--output',root/'fixed_fit'/role,'--maximum-passes',80,'--wall-seconds',900])
  stage('prepare_fitted',['tools.tight_fresh_fixed_fit_owner','prepare','--root',root,'--backend',cfg['backend_json'],'--output',candidate,'--fit','--fit-root',cfg.get('fit_root',root/'fixed_fit')])
  stage('fitted_autonomy',['tools.tight_batch_owner','--output',candidate,'--worker-tree',tree,'--startup-seconds',300],candidate)
  result=dict(software_complete=True,stages=stages,utc=time.time(),baseline=behavior(baseline,horizon=300),
   fitted=behavior(candidate,horizon=300),initial_comparison=compare_initials(baseline,candidate),
   training={r:dict(finished=read(Path(cfg.get('fit_root',root/'fixed_fit'))/r/'finished.json'),metrics=read(Path(cfg.get('fit_root',root/'fixed_fit'))/r/'metrics.json')) for r in 'AB'},
   scope='Fixed train2 capacity and closed-loop diagnosis. DP unchanged and still active at deployment. No held-out tuning, no automatic candidate adoption or DAgger expansion.')
  atomic_json(ledger/'finished.json',result)
  atomic_json(ledger/'state.json',dict(stage='complete',software_complete=True,utc=time.time(),owner_pid=os.getpid()))
 except BaseException as e:
  atomic_json(ledger/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),stages=stages,utc=time.time()))
  atomic_json(ledger/'state.json',dict(stage='safe_failure',error=repr(e),utc=time.time(),owner_pid=os.getpid()))
  raise
 finally:(root/'STOP_MONITOR').touch(exist_ok=True)

def fit_only(root,configuration):
 cfg=read(configuration);tree=Path.cwd();python=str(tree/'.venv_isaacsim/bin/python')
 ledger=root/'fixed_fit_stages';ledger.mkdir(exist_ok=False)
 deadline=min(read(root/'session.json')['deadline'],cfg['absolute_wall_deadline'])-120
 atomic_json(ledger/'started.json',dict(pid=os.getpid(),utc=time.time(),configuration=cfg,deadline=deadline))
 stages=[]
 try:
  for role in 'AB':
   command=[python,'-m','tools.tight_fresh_fixed_rear_fit','--source',cfg['sources'][role],'--role',role,
    '--resource-root',str(root),'--output',str(root/'fixed_fit'/role),'--maximum-passes','80','--wall-seconds','900']
   context=dict(ledger=ledger,paths=dict(worker_tree=tree,collection=root/'fixed_fit',output=root/'fixed_fit'))
   row=run_stage('fit_'+role,command,context,deadline,(root/'STOP',ledger/'STOP'))
   stages.append(row)
   if not row['complete']:raise RuntimeError('Bounded fixed-history fit failed: '+role)
  atomic_json(ledger/'finished.json',dict(software_complete=True,stages=stages,utc=time.time(),
   scope='Independent bounded offline diagnosis, no simulation. Shared main resource monitor retained. Autonomous comparison still required.'))
 except BaseException as error:
  atomic_json(ledger/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),stages=stages,utc=time.time()));raise


if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['prepare','run','fit-only']);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--configuration',type=Path);p.add_argument('--backend',type=Path);p.add_argument('--output',type=Path)
 p.add_argument('--fit',action='store_true');p.add_argument('--fit-root',type=Path);a=p.parse_args()
 if a.mode=='prepare':prepare_plan(a.root.resolve(),a.backend.resolve(),a.output.resolve(),fit=a.fit,fit_root=a.fit_root)
 elif a.mode=='fit-only':fit_only(a.root.resolve(),a.configuration.resolve())
 else:run(a.root.resolve(),a.configuration.resolve())
