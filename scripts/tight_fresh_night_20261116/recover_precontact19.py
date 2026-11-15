"""Explicit once-only repair of the pre-optimizer repeat12 CLI rejection.

The successful collection and its original frozen source stay unchanged. Only
unstarted training/evaluation moves to the new frozen tree; original failure,
configuration, monitor receipts and operational state remain in recovery ledger.
"""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
read=lambda p:json.loads(p.read_text())

def run(root,expected):
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()==expected
 assert not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','scripts'],cwd=TREE,text=True).strip()
 state=read(root/'night/state.json');cfg=read(root/'night_configuration.json')
 assert state['safe_stop'] and state['stage']=='safe_failure' and not state['rounds']
 assert cfg['latest_repeat']==cfg['backend_latest_repeat']==12 and len(cfg['rounds'])==1
 assert not (root/'round_19').exists() and not (root/'round_19_stages').exists()
 assert (root/'monitor_exit.json').exists()
 collection=root/'new_family_19';assert read(collection/'result.json')['software_complete']
 assert read(root/'night/collection_19_audit.json')['admitted']
 failures=list((root/'night').glob('cycle_19_*/cycle_19_finished.json'));assert len(failures)==1
 prior=read(failures[0]);assert prior['returncode']==2 and not prior['complete']
 assert "invalid choice: 12" in failures[0].with_name('cycle_19.log').read_text()
 for name in ('tight-fresh-night-20261116-04','tight-fresh-night-monitor-20261116-04'):
  assert subprocess.check_output(['systemctl','--user','show',name,'--property=MainPID','--value'],text=True).strip()=='0'
 ledger=root/'night/repeat12_recovery';ledger.mkdir(exist_ok=False)
 for name in ('monitor.json','monitor_exit.json','shared_resources.json','night/state.json','night_configuration.json'):
  atomic_json(ledger/('prior_'+name.replace('/','_')),read(root/name))
 deadline=min(read(root/'session.json')['deadline'],cfg['absolute_wall_deadline'])
 assert deadline-time.time()>4500
 bindings={name:sha256(collection/name) for name in ('plan.json','result.json','results.json','correction_entries.json','owner_finished.json')}
 cfg.update(source_commit=expected,minimum_free_bytes=int(26.5*2**30))
 path=root/'night_configuration_recovery19.json';atomic_json(path,cfg)
 state.update(configuration_sha256=sha256(path),stage='explicit_repeat12_recovery',safe_stop=False,complete=False,error=None)
 state.pop('traceback',None)
 atomic_json(ledger/'registration.json',dict(utc=time.time(),source_commit=expected,worker_tree=str(TREE),
  original_collection_commit=read(collection/'plan.json')['source_commit'],collection_bindings=bindings,
  zero_prior_optimizer_updates=True,zero_repeated_collection=True,deadline=deadline,
  admission_explanation='26.5GiB remaining-phase admission:1GiB capture already spent and permanently reused; unchanged16GiB runtime reserve. No new opportunities or budget extension.'))
 atomic_json(root/'night/state.json',state)
 (root/'STOP_MONITOR').unlink()
 python=str(TREE/'.venv_isaacsim/bin/python')
 subprocess.run(['systemd-run','--user','--unit=tight-fresh-repeat12-monitor-20261116-04',
  '--service-type=exec','--property=WorkingDirectory='+str(TREE),'--property=Restart=no',
  '--property=RuntimeMaxSec='+str(math.floor(deadline-time.time())),python,'-m','visuotactile.state_resources','--root',str(root)],check=True)
 until=time.time()+35
 while time.time()-read(root/'shared_resources.json')['utc_epoch']>8:
  if time.time()>until:raise RuntimeError('Recovery monitor did not start')
  time.sleep(.5)
 assert all(sha256(collection/name)==digest for name,digest in bindings.items())
 atomic_json(ledger/'launched.json',dict(utc=time.time(),pid=os.getpid(),configuration=str(path)))
 os.chdir(TREE)
 os.execv(python,[python,'-m','tools.tight_fresh_night_owner','--root',str(root),'--configuration',str(path)])

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-commit',required=True);a=p.parse_args()
 try:run(a.root.resolve(),a.expected_commit)
 except BaseException:
  (a.root/'STOP_MONITOR').touch(exist_ok=True);raise
