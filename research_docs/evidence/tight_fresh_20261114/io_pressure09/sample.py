"""One finite read-only proc sample; no torch, CUDA, directory workload or tuning."""
import hashlib,json,os,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TARGETS=(4030721,4030720)
CLK=os.sysconf('SC_CLK_TCK');UID=os.getuid()

def psi(kind):
    result={}
    for line in Path('/proc/pressure',kind).read_text().splitlines():
        k,*values=line.split();result[k]={a:float(b) for a,b in (v.split('=') for v in values)}
    return result

def disks():
    result={}
    for line in Path('/proc/diskstats').read_text().splitlines():
        s=line.split();name=s[2]
        if re.fullmatch(r'nvme\d+n\d+',name):
            v=list(map(int,s[3:]));names=('reads','reads_merged','read_sectors','read_ms','writes','writes_merged','write_sectors','write_ms','inflight','busy_ms','weighted_ms')
            result[name]=dict(zip(names,v))
    return result

def process(directory):
    if directory.stat().st_uid!=UID:return None
    fields=directory.joinpath('stat').read_text().rsplit(')',1)[1].split()
    values={k:int(v.strip()) for k,v in (row.split(':',1) for row in directory.joinpath('io').read_text().splitlines())}
    return dict(pid=int(directory.name),ppid=int(fields[1]),start_ticks=int(fields[19]),state=fields[0],
        utime_ticks=int(fields[11]),stime_ticks=int(fields[12]),delayacct_blkio_ticks=int(fields[39]),
        rss_bytes=int(fields[21])*os.sysconf('SC_PAGESIZE'),wchan=directory.joinpath('wchan').read_text().strip(),io=values)

def extra(pid):
    directory=Path('/proc',str(pid));fds=[];threads={};errors=[]
    try:
        for p in directory.joinpath('fd').iterdir():
            try:
                target=os.readlink(p)
                if target.startswith('/home/') or target.endswith('.pt') or '.pt.' in target:fds.append(target)
            except OSError:pass
        for p in directory.joinpath('task').iterdir():
            try:
                fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split();key=fields[0]+':'+p.joinpath('wchan').read_text().strip();threads[key]=threads.get(key,0)+1
            except OSError:pass
    except OSError as e:errors.append(repr(e))
    return dict(open_home_files=sorted(set(fds))[:40],open_home_file_count=len(set(fds)),thread_states=threads,errors=errors)

start=time.monotonic();rows=[];metadata={}
for sample in range(13):
    delay=start+sample*5-time.monotonic()
    if delay>0:time.sleep(delay)
    began=time.monotonic();processes={};errors=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            value=process(p)
            if value is not None:
                key=str(value['pid'])+':'+str(value['start_ticks']);processes[key]=value
                if key not in metadata:
                    metadata[key]=dict(pid=value['pid'],ppid=value['ppid'],argv=p.joinpath('cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0'))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        except Exception as e:errors.append(dict(pid=p.name,error=repr(e)))
    row=dict(utc=time.time(),elapsed=time.monotonic()-start,io_pressure=psi('io'),memory_pressure=psi('memory'),disks=disks(),processes=processes,targets={str(pid):extra(pid) for pid in TARGETS},errors=errors)
    row['sample_overhead_s']=time.monotonic()-began;rows.append(row)
    with (ROOT/'samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    if sample in (0,6,12):print(json.dumps(dict(sample=sample,elapsed=row['elapsed'],io_psi=row['io_pressure'],target_states={str(v['pid']):v['state']+':'+v['wchan'] for v in processes.values() if v['pid'] in TARGETS})),flush=True)

byproc={}
for before,after in zip(rows,rows[1:]):
    for key,new in after['processes'].items():
        if key not in before['processes']:continue
        old=before['processes'][key]
        out=byproc.setdefault(key,dict(**metadata[key],observed_seconds=0,read_bytes=0,write_bytes=0,cancelled_write_bytes=0,rchar=0,wchar=0,delayacct_blkio_seconds=0,cpu_seconds=0))
        out['observed_seconds']+=after['elapsed']-before['elapsed']
        for name in ('read_bytes','write_bytes','cancelled_write_bytes','rchar','wchar'):out[name]+=max(0,new['io'][name]-old['io'][name])
        out['delayacct_blkio_seconds']+=max(0,new['delayacct_blkio_ticks']-old['delayacct_blkio_ticks'])/CLK
        out['cpu_seconds']+=max(0,new['utime_ticks']+new['stime_ticks']-old['utime_ticks']-old['stime_ticks'])/CLK
elapsed=rows[-1]['elapsed']-rows[0]['elapsed'];disk={}
for name,after in rows[-1]['disks'].items():
    before=rows[0]['disks'][name];delta={k:after[k]-before[k] for k in after if k!='inflight'}
    disk[name]=dict(delta,read_MiB_s=delta['read_sectors']*512/2**20/elapsed,write_MiB_s=delta['write_sectors']*512/2**20/elapsed,
        busy_fraction=delta['busy_ms']/1000/elapsed,mean_queue_depth=delta['weighted_ms']/1000/elapsed,
        average_read_ms=delta['read_ms']/max(1,delta['reads']),average_write_ms=delta['write_ms']/max(1,delta['writes']))
for v in byproc.values():
    v['read_MiB_s']=v['read_bytes']/2**20/elapsed;v['write_MiB_s']=v['write_bytes']/2**20/elapsed
pressure={kind:{scope:dict(stalled_s=(rows[-1][kind][scope]['total']-rows[0][kind][scope]['total'])/1e6,
        window_fraction=(rows[-1][kind][scope]['total']-rows[0][kind][scope]['total'])/1e6/elapsed,
        start=rows[0][kind][scope],end=rows[-1][kind][scope]) for scope in ('some','full')} for kind in ('io_pressure','memory_pressure')}
report=dict(version='read_only_proc_io_sample_v1',start_utc=rows[0]['utc'],end_utc=rows[-1]['utc'],seconds=elapsed,sample_count=len(rows),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),target_pids=TARGETS,disks=disk,pressure=pressure,
    targets=[v for v in byproc.values() if v['pid'] in TARGETS],
    top_user_io=sorted(byproc.values(),key=lambda v:v['read_bytes']+v['write_bytes'],reverse=True)[:20],
    maximum_sample_overhead_s=max(r['sample_overhead_s'] for r in rows),
    limits=['Process read/write_bytes are kernel process accounting, not independently additive physical disk attribution; parents may inherit waited-child counters.',
    'Short-lived processes between samples and bytes before first/after last observation may be missed. No unrelated process is signalled or changed.',
    'Thread wchan is a sampled wait site, not elapsed wait attribution. delayacct zero may mean disabled accounting.',
    'No torch imports, CUDA, synthetic disk benchmark, project directory enumeration, payload hashing or reads.'])
(ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'process_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(dict(complete=True,report=str(ROOT/'report.json'),seconds=elapsed,disks=disk,targets=report['targets'],top_user_io=report['top_user_io'][:6])),flush=True)
