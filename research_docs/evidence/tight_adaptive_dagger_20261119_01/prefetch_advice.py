"""Bounded read-only OS readahead for the already registered next DP batch.

No tensor cache, optimizer call, sampling draw, file write, or training-source edit.
Only output receipts are written. WILLNEED is advisory; benefit is not guaranteed.
"""
from pathlib import Path
import json,os,time
import psutil
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')
TARGET=ROOT/'round01/learning/A'
OUT=ROOT/'readonly_prefetch_advice01'
OUT.mkdir(exist_ok=False)

def read(path):return json.loads(path.read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2))

index=read(TARGET/'cache_index.json')['entries'];initial=read(TARGET/'training/progress.json');start=time.time();seen=set();rows=[];epoch=None;deck=None
write('registration.json',dict(started_utc=start,initial_update=initial['updates'],maximum_seconds=600,maximum_new_updates=80,
    memory_admission_gib=16,next_batches=2,maximum_advised_bytes_per_poll=2*2**30,source='Immutable current cache_index; registered deck only',
    inputs_unchanged=True,optimizer_state_unchanged=True,no_extra_optimizer_updates=True,
    method='POSIX_FADV_WILLNEED in64MiB ranges; kernel may ignore; no file-content writes',
    initial_progress={k:initial[k] for k in ('epoch','batch_index','updates','utc')}))
with (OUT/'advice.jsonl').open('x') as log:
 while time.time()-start<600:
    if (ROOT/'owners/r1_train/project_failure.json').exists():break
    p=read(TARGET/'training/progress.json')
    if p['updates']>=initial['updates']+80:break
    if psutil.virtual_memory().available<16*2**30:
        time.sleep(2);continue
    if epoch!=p['epoch']:
        epoch=p['epoch'];deck=read(TARGET/f'training/deck_{epoch:03d}.json')['order'];seen=set()
    maximum=2*2**30;advised=[]
    for identity in dict.fromkeys(deck[2*p['batch_index']:2*(p['batch_index']+2)]):
        key=(p['updates']//4,identity)
        if key in seen:continue
        path=Path(index[identity]['path']);size=path.stat().st_size
        if size>maximum:continue
        maximum-=size
        with path.open('rb') as stream:
            for offset in range(0,size,64*2**20):os.posix_fadvise(stream.fileno(),offset,min(64*2**20,size-offset),os.POSIX_FADV_WILLNEED)
        seen.add(key);advised.append(dict(identity=identity,path=str(path),source_sha256=index[identity]['sha256'],file_bytes=size))
    if advised:
        row=dict(epoch=epoch,observed_update=p['updates'],files=advised,utc=time.time());rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    time.sleep(.5)
last=read(TARGET/'training/progress.json')
write('complete.json',dict(started_utc=start,ended_utc=time.time(),first_update=initial['updates'],last_update=last['updates'],advice_events=len(rows),
    scope='Existing authorized training continued; no additional optimizer work. Advisory cache hints only, no data/source/RNG changes; timing is confounded by other workloads.'))
print(json.dumps(read(OUT/'complete.json')),flush=True)
