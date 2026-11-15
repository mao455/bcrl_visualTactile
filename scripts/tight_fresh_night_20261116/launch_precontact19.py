"""One evidence-based pre-contact cycle after two supervised clean loops.

Uses new real prefixes, both-network exposure12, eight epochs each. Retains the
09:00 absolute deadline and16GiB runtime reserve. No automatic blind next cycle.
"""
import argparse,json,math,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
BUNDLE=Path(__file__).resolve().parent
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
OLD=BASE/'tight_fresh_20261116_02';DIAG=BASE/'tight_fresh_20261116_03';ROOT=BASE/'tight_fresh_20261116_04'
DEADLINE=datetime(2026,11,16,9,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
read=lambda p:json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--expected-commit',required=True);a=p.parse_args()
 assert not ROOT.exists()
 state=read(OLD/'night/state.json')
 assert state['safe_stop'] and [r['index'] for r in state['rounds']]==[17,18]
 assert all(r['complete'] for r in state['rounds'])
 assert (OLD/'monitor_exit.json').exists() and (DIAG/'monitor_exit.json').exists()
 assert read(DIAG/'fixed_fit_owner/finished.json')['software_complete']
 assert read(DIAG/'closed_grasp_archives/finished.json')['complete']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()==a.expected_commit
 assert not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','scripts'],cwd=TREE,text=True).strip()
 assert DEADLINE-time.time()>5400
 # Measured closed four-opportunity footprint + checkpoint temporary +16GiB.
 assert shutil.disk_usage(BASE).free>=int(27.5*2**30)
 ledger=ROOT.with_name(ROOT.name+'_launch');ledger.mkdir()
 python=str(TREE/'.venv_isaacsim/bin/python');source=OLD/'round_18'
 cmd=[python,'-m','tools.tight_fresh_followup_session','--source-root',str(source),
  '--data-source-root',str(source),'--evaluation',str(source/'development'),'--output',str(ROOT),
  '--hours',str((DEADLINE-time.time())/3600),'--allow-illegal-starts','--reserve-gib','16']
 for role in 'AB':cmd+=['--role-source',role+'='+state['sources'][role]]
 atomic_json(ledger/'registration.json',dict(utc=time.time(),source_commit=a.expected_commit,command=cmd,
  absolute_deadline=DEADLINE,prior_two_clean_cycles=[17,18],prior_root=str(OLD),
  diagnosis=str(DIAG/'fixed_fit_owner/finished.json'),
  reason='Contact-onset data missed approach labels; new true pre-contact opportunities plus more fitting in BOTH networks. Limited to one new cycle, no claim of component causality.'))
 with (ledger/'prepare.log').open('x') as stream:
  subprocess.run(cmd,cwd=TREE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
 assert abs(read(ROOT/'session.json')['deadline']-DEADLINE)<600
 cfg=read(OLD/'night_configuration.json')
 cfg.update(source_commit=a.expected_commit,absolute_wall_deadline=DEADLINE,
  sources=state['sources'],source_evaluations=state['evaluations'],data_root=str(source),
  refresh_source_backend_comparison=False,latest_repeat=12,backend_latest_repeat=12,
  minimum_round_seconds=4500,minimum_free_bytes=int(27.5*2**30),
  rounds=[dict(index=19,families=['expanded_session20261115_03_prefetch13_s29000076',
   'expanded_session20261115_03_prefetch13_s29000086'],schedules=['grasp_approach'],
   event_seed=29203000,minimum_accepted=3,epochs=8,rear_epochs=8)])
 for key in ('backend_json','training_sampling_json','frozen_visual_io_json','backend_phase_weighting_json'):
  cfg[key]=str(BUNDLE/Path(cfg[key]).name)
 atomic_json(ROOT/'night_configuration.json',cfg)
 atomic_json(ROOT/'prior_supervised_review.json',dict(utc=time.time(),reviewed_rounds=[17,18],
  two_real_complete_cycles_observed=True,source_state_sha256=sha256(OLD/'night/state.json'),
  actual_new_trigger_live_verified=False,unattended_release=False,
  conclusion='Sequential real cycles, admission, both optimizers, checkpoints, autonomous queues and automatic handoff passed; no development gain. Review first new pre-contact capture before detaching.'))
 base=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(TREE),
  '--property=Restart=no','--property=KillMode=control-group','--property=RuntimeMaxSec='+str(math.floor(DEADLINE-time.time()))]
 subprocess.run(base+['--unit=tight-fresh-night-monitor-20261116-04',python,'-m','visuotactile.state_resources','--root',str(ROOT)],check=True)
 end=time.time()+40
 while not (ROOT/'shared_resources.json').exists() or time.time()-read(ROOT/'shared_resources.json')['utc_epoch']>8:
  if time.time()>end:raise RuntimeError('Resource monitor unavailable')
  time.sleep(.5)
 subprocess.run(base+['--unit=tight-fresh-night-20261116-04','--property=MemoryHigh=28G','--property=MemoryMax=34G',
  '--property=ExecStopPost=/usr/bin/touch '+str(ROOT/'STOP_MONITOR'),python,'-m','tools.tight_fresh_night_owner',
  '--root',str(ROOT),'--configuration',str(ROOT/'night_configuration.json')],check=True)
 atomic_json(ledger/'launched.json',dict(utc=time.time(),source_commit=a.expected_commit,deadline=DEADLINE,unattended_release=False))

if __name__=='__main__':main()
