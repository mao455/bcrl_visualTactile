"""Same-deadline cache continuation: no new collection or repeated optimizer work."""
import argparse,json,math,subprocess,time
from pathlib import Path
from visuotactile.training_resources import atomic_json

TREE=Path(__file__).resolve().parents[2];BUNDLE=Path(__file__).resolve().parent
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ROOT=BASE/'tight_fresh_20261115_10';OLD=BASE/'tight_fresh_20261115_09'
read=lambda p:json.loads(p.read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);a=p.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()==a.expected_commit
    assert not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','scripts'],cwd=TREE,text=True).strip()
    assert not ROOT.exists() and (OLD/'STOP_MONITOR').exists() and (OLD/'monitor_exit.json').exists()
    assert read(OLD/'new_family_15/owner_finished.json')['software_complete']
    assert 'visual_cache' in read(OLD/'round_15_stages/failure.json')['error']
    deadline=read(OLD/'session.json')['deadline'];assert deadline-time.time()>5400
    ledger=ROOT.with_name(ROOT.name+'_launch');ledger.mkdir();python=str(TREE/'.venv_isaacsim/bin/python')
    cmd=[python,'-m','tools.tight_fresh_followup_session','--source-root',str(BASE/'tight_fresh_20261115_01/round_11'),
        '--data-source-root',str(BASE/'tight_fresh_20261115_04/round_14'),
        '--evaluation',str(BASE/'tight_fresh_20261115_01/round_11/development'),'--output',str(ROOT),'--hours','6',
        '--allow-illegal-starts','--role-source','A='+str(BASE/'tight_fresh_20261115_01/round_11'),
        '--role-source','B='+str(BASE/'tight_fresh_20261114_06/round_09'),
        '--absolute-deadline',str(deadline),'--resource-parent',str(OLD),'--reserve-gib','16']
    atomic_json(ledger/'registration.json',dict(commit=a.expected_commit,deadline=deadline,command=cmd,
        source_collection=str(OLD/'new_family_15'),prepared_from=str(OLD/'round_15'),reserve_gib=16,
        reason='32GiB reserve interrupted frozen feature cache before any optimizer update; retain16GiB with measured remaining growth. Existing committed cache reused; parallel independent frontends use original joint headroom gate.'))
    with (ledger/'prepare.log').open('x') as log:subprocess.run(cmd,cwd=TREE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assert read(ROOT/'session.json')['deadline']==deadline
    base=['systemd-run','--user','--service-type=exec','--property=WorkingDirectory='+str(TREE),
        '--property=Restart=no','--property=KillMode=control-group','--property=RuntimeMaxSec='+str(math.floor(deadline-time.time()))]
    subprocess.run(base+['--unit=tight-fresh-monitor-20261115-10',python,'-m','visuotactile.state_resources','--root',str(ROOT)],check=True)
    end=time.time()+40
    while not (ROOT/'shared_resources.json').exists() or time.time()-read(ROOT/'shared_resources.json')['utc_epoch']>8:
        if time.time()>end:raise RuntimeError('New monitor did not become live')
        time.sleep(.5)
    subprocess.run(base+['--unit=tight-fresh-cache-continuation-20261115-10',
        '--property=ExecStopPost=/usr/bin/touch '+str(ROOT/'STOP_MONITOR'),python,str(BUNDLE/'owner.py')],check=True)
    atomic_json(ledger/'launched.json',dict(utc=time.time(),deadline=deadline,startup_only=True))
