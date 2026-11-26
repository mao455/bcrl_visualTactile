"""Pause only this run's full-GPU lease blocks when another GPU test appears."""
import json,time,hashlib,signal,os
from pathlib import Path
from tools.tight_paired_dagger import ROOT,AUDIT,read
from visuotactile.training_resources import atomic_json
PAUSES=[None,'G00','G10','G01','G11']

def decision(rows,now):
    fresh=[r for r in rows if now-r.get('utc_epoch',0)<8]
    if not fresh:return True,[], 'no_fresh_owned_telemetry'
    newest=max(fresh,key=lambda r:r['utc_epoch'])
    external=newest.get('external_gpu_processes',[])
    return bool(external),external,'external_gpu_present' if external else 'no_external_gpu'

def digest(content):return hashlib.sha256(content).hexdigest()

def interrupted(signum,frame):raise KeyboardInterrupt(signum)

def main():
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    end=read(ROOT/'session.json')['deadline'];saved={};changed={};last=None
    backup=AUDIT/'resource_watch_backups';backup.mkdir(exist_ok=False)
    log=AUDIT/'resource_watch_decisions.jsonl'
    try:
        while time.time()<end and not (ROOT/'STOP_RESOURCE_WATCH').exists():
            rows=[]
            for p in (ROOT/'owners').glob('*/shared_resources.json'):
                try:rows.append(read(p))
                except (OSError,ValueError):pass
            pause,external,reason=decision(rows,time.time())
            if pause:
                for session in ROOT.rglob('session.json'):
                    try:
                        if read(session).get('resource_policy')!='full_gpu_v1':continue
                    except (OSError,ValueError):continue
                    p=session.parent/'concurrency.json'
                    if str(p) not in saved:
                        content=p.read_bytes() if p.exists() else None
                        saved[str(p)]=content
                        key=digest(str(p).encode())
                        atomic_json(backup/(key+'.json'),dict(path=str(p),existed=content is not None,content=None if content is None else content.decode()))
                    else:content=saved[str(p)]
                    current=p.read_bytes() if p.exists() else None
                    if current is not None and digest(current)!=changed.get(str(p)) and current!=content:continue
                    original={} if content is None else json.loads(content)
                    groups=list(original.get('pause_groups',[]))
                    for group in PAUSES:
                        if group not in groups:groups.append(group)
                    row=dict(original,pause_groups=groups,paired_resource_watch_pid=os.getpid())
                    atomic_json(p,row);changed[str(p)]=digest(p.read_bytes())
            else:
                for key,expected in list(changed.items()):
                    p=Path(key)
                    if p.exists() and digest(p.read_bytes())==expected:
                        content=saved[key]
                        if content is None:p.unlink()
                        else:p.write_bytes(content)
                    del changed[key]
            state=dict(paused=pause,external_gpu_processes=external,reason=reason)
            atomic_json(AUDIT/'resource_watch_status.json',dict(state,pid=os.getpid(),utc=time.time(),modified_schedules=len(changed)))
            if state!=last:
                with log.open('a') as stream:stream.write(json.dumps(dict(state,utc=time.time()))+'\n')
                last=state
            time.sleep(2)
    finally:
        for key,expected in changed.items():
            p=Path(key)
            if p.exists() and digest(p.read_bytes())==expected:
                if saved[key] is None:p.unlink()
                else:p.write_bytes(saved[key])
        atomic_json(AUDIT/'resource_watch_exit.json',dict(utc=time.time(),only_owned_schedules_restored=True))

if __name__=='__main__':main()
