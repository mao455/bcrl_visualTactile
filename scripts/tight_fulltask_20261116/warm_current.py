"""Bounded immutable current/next B2 OS-page warming; no extra disk cache."""
import json,time,os,signal
from pathlib import Path
from scripts.tight_fulltask_20261116.run import ROOT,DEADLINE,read,atomic_json
stop=False
signal.signal(signal.SIGTERM,lambda *a:globals().__setitem__('stop',True))
buffer=bytearray(16*2**20);seen=set();started=time.time();amount=0;rows=[]
try:
    while not stop and time.time()<DEADLINE-900:
        active=[]
        for role in 'AB':
            worker=ROOT/('frontend_'+role+'_worker.json')
            if not worker.exists():continue
            w=read(worker)
            try:cmd=Path('/proc',str(w['pid']),'cmdline').read_bytes()
            except FileNotFoundError:continue
            if b'tools.tight_fresh_train' in cmd:active.append(role)
        if not active:
            if (ROOT/'update/B/training/finished.json').exists():break
            time.sleep(.5);continue
        assert len(active)==1
        role=active[0];base=ROOT/'update'/role;p=base/'training/progress.json'
        if not p.exists():time.sleep(.3);continue
        r=read(p);cursor=(role,r['epoch'],r['batch_index'])
        if cursor in seen:time.sleep(.2);continue
        deck=read(base/'training'/f"deck_{r['epoch']:03d}.json")['order']
        # The last committed batch is finished; warm the next two visits only.
        identities=deck[r['batch_index']*2:(r['batch_index']+1)*2]
        began=time.time();count=0
        for identity in identities:
            file=(base/'state_data/perception'/(identity+'.pt')).resolve()
            with file.open('rb',buffering=0) as f:
                while not stop:
                    n=f.readinto(buffer)
                    if not n:break
                    count+=n
        amount+=count;seen.add(cursor);rows.append(dict(role=role,epoch=r['epoch'],after_update=r['updates'],bytes=count,seconds=time.time()-began))
        atomic_json(ROOT/'read_warm_progress.json',dict(utc=time.time(),pid=os.getpid(),total_bytes=amount,last=rows[-1],extra_disk_bytes=0,buffer_bytes=len(buffer)))
finally:atomic_json(ROOT/'read_warm_complete.json',dict(utc=time.time(),seconds=time.time()-started,rows=rows,total_bytes=amount,extra_disk_bytes=0))
