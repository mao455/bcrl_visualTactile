"""New identity after the preserved round16 coordinator and camera failures.

Requires the actual repaired evaluation. Counts zero prior clean supervised
cycles; rounds17/18 must both finish before the existing release barrier opens.
"""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.training_resources import atomic_json
BUNDLE=Path(__file__).resolve().parent
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ROOT=BASE/'tight_fresh_20261116_02';OLD=BASE/'tight_fresh_20261116_01'
DEADLINE=datetime(2026,11,16,9,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
read=lambda p:json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);a=p.parse_args()
 assert not ROOT.exists() and (OLD/'monitor_exit.json').exists()
 evaluation=OLD/'round_16/development_camera_repaired_01'
 assert read(evaluation/'result.json')['software_complete']
 decision=read(OLD/'night/repaired16_decision.json')
 state=read(OLD/'night/state.json');sources=state['sources']
 assert set(sources)=={'A','B'} and all(not row['use_candidate'] for row in decision['decisions'].values())
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()==a.expected_commit
 assert not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','scripts'],cwd=TREE,text=True).strip()
 assert DEADLINE-time.time()>7200
 ledger=ROOT.with_name(ROOT.name+'_launch');ledger.mkdir();python=str(TREE/'.venv_isaacsim/bin/python')
 cmd=[python,'-m','tools.tight_fresh_followup_session','--source-root',str(OLD/'round_16'),
  '--data-source-root',str(OLD/'round_16'),'--evaluation',str(evaluation),'--output',str(ROOT),
  '--hours',str((DEADLINE-time.time())/3600),'--allow-illegal-starts','--reserve-gib','16']
 for role in 'AB':cmd+=['--role-source',role+'='+sources[role]]
 atomic_json(ledger/'registration.json',dict(utc=time.time(),expected_commit=a.expected_commit,old_root=str(OLD),
  old_failure_state=state['stage'],actual_repaired_evaluation=str(evaluation),command=cmd,
  absolute_deadline=DEADLINE,clean_supervised_cycles=0,
  reason='Preserve both failed attempts and all completed16 training; use retained A11/B15 plus cumulative16 data. New early-phase learning profile, same morning deadline.'))
 with (ledger/'prepare.log').open('x') as stream:
  subprocess.run(cmd,cwd=TREE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
 actual=read(ROOT/'session.json')['deadline'];assert abs(actual-DEADLINE)<600
 config=read(BUNDLE/'configuration.template.json')
 config.update(source_commit=a.expected_commit,absolute_wall_deadline=DEADLINE,sources=sources,
  source_evaluations=state['evaluations'],data_root=str(OLD/'round_16'),refresh_source_backend_comparison=False,
  training_sampling_json='training_sampling_grasp.json',backend_phase_weighting_json='backend_grasp_weighting.json')
 config['rounds']=[row for row in config['rounds'] if row['index']>=17]
 assert [row['index'] for row in config['rounds'][:2]]==[17,18]
 for key in ('backend_json','training_sampling_json','frozen_visual_io_json','backend_phase_weighting_json'):
  config[key]=str(BUNDLE/config[key])
 atomic_json(ROOT/'night_configuration.json',config)
 base=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(TREE),'--property=Restart=no',
  '--property=KillMode=control-group','--property=RuntimeMaxSec='+str(math.floor(DEADLINE-time.time()))]
 subprocess.run(base+['--unit=tight-fresh-night-monitor-20261116-02',python,'-m','visuotactile.state_resources','--root',str(ROOT)],check=True)
 end=time.time()+40
 while not (ROOT/'shared_resources.json').exists() or time.time()-read(ROOT/'shared_resources.json')['utc_epoch']>8:
  if time.time()>end:raise RuntimeError('Night resource monitor unavailable')
  time.sleep(.5)
 subprocess.run(base+['--unit=tight-fresh-night-20261116-02','--property=MemoryHigh=28G','--property=MemoryMax=34G',
  '--property=ExecStopPost=/usr/bin/touch '+str(ROOT/'STOP_MONITOR'),python,'-m','tools.tight_fresh_night_owner',
  '--root',str(ROOT),'--configuration',str(ROOT/'night_configuration.json')],check=True)
 atomic_json(ledger/'launched.json',dict(utc=time.time(),actual_deadline=actual,supervised_cycles_passed=0,unattended_ready=False))

if __name__=='__main__':main()
