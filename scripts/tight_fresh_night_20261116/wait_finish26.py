"""One-shot persistent transition after the supervised DP-boundary request."""
import argparse,json,subprocess,time,sys
from pathlib import Path
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.training_resources import atomic_json
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
OLD=BASE/'tight_fresh_20261115_13';LEDGER=BASE/'tight_fresh_20261116_00_handover'
p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);a=p.parse_args()
LEDGER.mkdir(exist_ok=False);deadline=json.loads((OLD/'session.json').read_text())['deadline']+30
try:
 while time.time()<deadline:
  requested=(OLD/'night_boundary_request.json').exists()
  closed=(OLD/'continuation/owner_finished.json').exists() and (OLD/'monitor_exit.json').exists()
  atomic_json(LEDGER/'status.json',dict(stage='waiting_for_completed_DP_boundary',heartbeat=time.time(),
   requested=requested,old_owner_and_monitor_closed=closed,deadline=deadline,expected_commit=a.expected_commit))
  if closed:
   if not requested:raise RuntimeError('Old owner ended without the completed-DP handover request')
   break
  time.sleep(5)
 else:raise TimeoutError('Old bounded session did not close at the requested boundary')
 cmd=[str(TREE/'.venv_isaacsim/bin/python'),str(TREE/'scripts/tight_fresh_night_20261116/finish26.py'),
  'launch','--expected-commit',a.expected_commit]
 atomic_json(LEDGER/'launch_started.json',dict(utc=time.time(),command=cmd))
 with (LEDGER/'launch.log').open('x') as stream:subprocess.run(cmd,cwd=TREE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
 atomic_json(LEDGER/'status.json',dict(stage='new_tail_owner_launched',heartbeat=time.time(),unattended_validated=False))
except BaseException as e:
 atomic_json(LEDGER/'failure.json',dict(utc=time.time(),error=repr(e)));raise
