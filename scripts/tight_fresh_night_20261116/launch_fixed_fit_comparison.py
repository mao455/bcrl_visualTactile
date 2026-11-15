"""Serial, bounded closed-loop comparison after the two supervised night cycles.

Reuses the already completed train2 fits. Never resumes or repeats their updates,
and never starts a simulator while the main night owner is still active.
"""
import argparse,json,math,subprocess,sys,time
from pathlib import Path

TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.training_resources import atomic_json
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
OLD=BASE/'tight_fresh_20261116_02';ROOT=BASE/'tight_fresh_20261116_03'
read=lambda p:json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);a=p.parse_args()
 assert not ROOT.exists()
 state=read(OLD/'night/state.json');cfg=read(OLD/'fixed_fit_configuration.json')
 assert state.get('safe_stop') and [r['index'] for r in state['rounds'] if r['complete']][:2]==[17,18]
 assert (OLD/'monitor_exit.json').exists() and read(OLD/'fixed_fit_stages/finished.json')['software_complete']
 for role in 'AB':assert read(OLD/'fixed_fit'/role/'finished.json')['complete']
 deadline=cfg['absolute_wall_deadline'];assert deadline-time.time()>1200
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()==a.expected_commit
 assert not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','scripts'],cwd=TREE,text=True).strip()
 ledger=ROOT.with_name(ROOT.name+'_launch');ledger.mkdir();python=str(TREE/'.venv_isaacsim/bin/python')
 source=Path(state['data_root'])
 cmd=[python,'-m','tools.tight_fresh_followup_session','--source-root',str(source),
  '--data-source-root',str(source),'--evaluation',str(source/'development'),'--output',str(ROOT),
  '--hours',str((deadline-time.time())/3600),'--allow-illegal-starts','--reserve-gib','16']
 for role in 'AB':cmd+=['--role-source',role+'='+cfg['sources'][role]]
 atomic_json(ledger/'registration.json',dict(utc=time.time(),source_commit=a.expected_commit,command=cmd,
  absolute_deadline=deadline,fit_root=str(OLD/'fixed_fit'),
  scope='Completed A11/B15 train2 fits, four source and four fitted full-policy grasp rollouts; no new optimizer updates or automatic adoption.'))
 with (ledger/'prepare.log').open('x') as stream:
  subprocess.run(cmd,cwd=TREE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
 cfg.update(source_commit=a.expected_commit,fit_root=str(OLD/'fixed_fit'),
  backend_json=str(TREE/'scripts/tight_fresh_night_20261116/backend.json'))
 atomic_json(ROOT/'fixed_fit_configuration.json',cfg)
 common=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(TREE),
  '--property=Restart=no','--property=KillMode=control-group',
  '--property=RuntimeMaxSec='+str(math.floor(deadline-time.time()))]
 subprocess.run(common+['--unit=tight-fresh-fixed-fit-monitor-20261116-03',python,'-m',
  'visuotactile.state_resources','--root',str(ROOT)],check=True)
 end=time.time()+40
 while not (ROOT/'shared_resources.json').exists() or time.time()-read(ROOT/'shared_resources.json')['utc_epoch']>8:
  if time.time()>end:raise RuntimeError('Diagnostic monitor did not start')
  time.sleep(.5)
 subprocess.run(common+['--unit=tight-fresh-fixed-fit-20261116-03','--property=MemoryHigh=18G',
  '--property=MemoryMax=24G','--property=ExecStopPost=/usr/bin/touch '+str(ROOT/'STOP_MONITOR'),
  python,'-m','tools.tight_fresh_fixed_fit_owner','run','--root',str(ROOT),
  '--configuration',str(ROOT/'fixed_fit_configuration.json')],check=True)
 atomic_json(ledger/'launched.json',dict(utc=time.time(),absolute_deadline=deadline,
  repeated_fit_updates=0,source_commit=a.expected_commit))

if __name__=='__main__':main()
