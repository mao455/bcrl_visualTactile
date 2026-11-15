"""Start one new bounded night session only after the previous owner has closed."""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.training_resources import atomic_json
BUNDLE=Path(__file__).resolve().parent
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ROOT=BASE/'tight_fresh_20261116_01';OLD=BASE/'tight_fresh_20261116_00'
DEADLINE=datetime(2026,11,16,9,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
read=lambda p:json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--role-source',action='append',required=True)
a=p.parse_args();sources=dict(v.split('=',1) for v in a.role_source);assert set(sources)=={'A','B'}
assert not ROOT.exists() and (OLD/'monitor_exit.json').exists() and read(OLD/'owner_finished.json')['software_complete']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()==a.expected_commit
assert not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','scripts'],cwd=TREE,text=True).strip()
assert DEADLINE-time.time()>7200
ledger=ROOT.with_name(ROOT.name+'_launch');ledger.mkdir();python=str(TREE/'.venv_isaacsim/bin/python')
cmd=[python,'-m','tools.tight_fresh_followup_session','--source-root',str(OLD/'round_15'),
 '--data-source-root',str(OLD/'round_15'),'--evaluation',str(OLD/'round_15/development'),
 '--output',str(ROOT),'--hours',str((DEADLINE-time.time())/3600),'--allow-illegal-starts','--reserve-gib','16']
for role in 'AB':cmd+=['--role-source',role+'='+sources[role]]
atomic_json(ledger/'registration.json',dict(utc=time.time(),expected_commit=a.expected_commit,old_deadline=read(OLD/'session.json')['deadline'],
 new_deadline_target=DEADLINE,command=cmd,authorization='Nov16 latest user permits a new bounded night window until next morning; old six-hour temporary deadline is preserved'))
with (ledger/'prepare.log').open('x') as stream:subprocess.run(cmd,cwd=TREE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
actual=read(ROOT/'session.json')['deadline'];assert abs(actual-DEADLINE)<600
config=read(BUNDLE/'configuration.template.json');config.update(source_commit=a.expected_commit,absolute_wall_deadline=DEADLINE,sources=sources,
 source_evaluations={role:str(Path(path)/'development') for role,path in sources.items()},data_root=str(OLD/'round_15'))
for key in ('backend_json','training_sampling_json','frozen_visual_io_json'):config[key]=str(BUNDLE/config[key])
atomic_json(ROOT/'night_configuration.json',config)
base=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(TREE),'--property=Restart=no',
 '--property=KillMode=control-group','--property=RuntimeMaxSec='+str(math.floor(DEADLINE-time.time()))]
subprocess.run(base+['--unit=tight-fresh-night-monitor-20261116-01',python,'-m','visuotactile.state_resources','--root',str(ROOT)],check=True)
end=time.time()+40
while not (ROOT/'shared_resources.json').exists() or time.time()-read(ROOT/'shared_resources.json')['utc_epoch']>8:
 if time.time()>end:raise RuntimeError('Night resource monitor unavailable')
 time.sleep(.5)
subprocess.run(base+['--unit=tight-fresh-night-20261116-01','--property=MemoryHigh=28G','--property=MemoryMax=34G',
 '--property=ExecStopPost=/usr/bin/touch '+str(ROOT/'STOP_MONITOR'),python,'-m','tools.tight_fresh_night_owner',
 '--root',str(ROOT),'--configuration',str(ROOT/'night_configuration.json')],check=True)
atomic_json(ledger/'launched.json',dict(utc=time.time(),actual_deadline=actual,supervised_cycles_passed=0,unattended_ready=False))
