"""Compare actual inode blocks with the published round2 boundary, once."""
from pathlib import Path
import sqlite3,json,collections,stat,gzip,csv,time
R=Path.cwd();A=R/'outputs/diagnostics/tight_machine_review_20261126_01';W=R/'outputs/maintenance_20261123/round2'
def read(p):return json.loads(Path(p).read_text())
def table(path,name):
    c=sqlite3.connect(path);c.row_factory=sqlite3.Row
    return {r['path']:dict(r) for r in c.execute('select * from '+name)}
old=table(W/'snapshot.sqlite','final');now=table(A/'disk_after.sqlite','files')
before=table(A/'disk_before.sqlite','files');summary=read(W/'final_summary.json')
cache={}
def generated(path):
    p=(R/path).parent
    while p!=R and R in p.parents:
        for name in ('run_started.json','started.json','project_job_started.json','registration.json'):
            q=p/name
            if q not in cache:
                try:
                    x=read(q);v=next((x[k] for k in ('utc','started_utc','registered_utc','created_utc') if isinstance(x.get(k),(float,int))),None)
                    cache[q]=v
                except (FileNotFoundError,ValueError,TypeError):cache[q]=None
            if cache[q] is not None:return cache[q],str(q.relative_to(R))
        p=p.parent
    return None,'unknown'
fields=['path','status','device','inode','nlink','size','allocated','birth_utc','birth_time_status','mtime_utc','run_metadata_utc','run_metadata_source','old_allocated','old_inode']
counts=collections.Counter()
with gzip.open(A/'file_changes_since_round2.csv.gz','wt',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for p in sorted(set(old)|set(now)):
        x=now.get(p);o=old.get(p)
        if x is not None and o is not None and all(x.get(k)==o.get(k) for k in ('device','inode','size','allocated','mtime_ns')):continue
        state='removed' if x is None else 'added' if o is None else 'changed'
        v=x or o;t,source=generated(p);counts[state]+=1
        w.writerow(dict(path=p,status=state,device=v['device'],inode=v['inode'],nlink=v['nlink'],size=v['size'],allocated=v['allocated'],
            birth_utc=x.get('birth_utc') if x else None,birth_time_status='known' if x and x.get('birth_utc') is not None else 'unknown',
            mtime_utc=v['mtime_ns']/1e9,run_metadata_utc=t,run_metadata_source=source,
            old_allocated=o['allocated'] if o else None,old_inode=o['inode'] if o else None))
def group(p):
    a=p.split('/')
    return a[3] if a[:3]==['outputs','visuotactile','run'] and len(a)>3 else '/'.join(a[:2])
def aggregate(rows):
    by=collections.defaultdict(lambda:dict(paths=0,regular_paths=0,unique_inodes=0,allocated_bytes=0));seen=set()
    for p,x in rows.items():
        g=by[group(p)];g['paths']+=1;g['regular_paths']+=int(stat.S_ISREG(x['mode']));key=x['device'],x['inode']
        if key in seen:continue
        seen.add(key);g['unique_inodes']+=1;g['allocated_bytes']+=x['allocated']
    return by
was,mid,after=map(aggregate,(old,before,now));runs=[]
for key,g in after.items():
    runs.append(dict(run=key,**g,GiB=g['allocated_bytes']/2**30,
        round2_counts=was.get(key,{}),before_review_counts=mid.get(key,{}),
        delta_since_round2_GiB=(g['allocated_bytes']-was.get(key,{}).get('allocated_bytes',0))/2**30,
        delta_during_review_GiB=(g['allocated_bytes']-mid.get(key,{}).get('allocated_bytes',0))/2**30))
runs.sort(key=lambda x:x['allocated_bytes'],reverse=True)
raw=read(A/'disk_after.json');pre=read(A/'disk_before.json');deleted=read(A/'retirement_result.json')
baseline=summary['after_project_including_round2_allocated_bytes']
net=raw['allocated_bytes']-baseline
result=dict(snapshot_utc=raw['utc'],round2_snapshot_as_of=summary['as_of'],
    published_round2_baseline_bytes=baseline,published_round2_baseline_GiB=baseline/2**30,
    baseline_snapshot_excludes_round2_audit_bytes=summary['round2_audit_allocated_bytes'],
    before_review=pre,after_review=raw,net_since_published_round2_bytes=net,net_since_published_round2_GiB=net/2**30,
    this_review_deleted=deleted,net_during_review_bytes=raw['allocated_bytes']-pre['allocated_bytes'],
    added_plus_growth_minus_other_removals_during_review_bytes=raw['allocated_bytes']-pre['allocated_bytes']+deleted['released_bytes'],
    changed_paths=counts,top_runs=runs[:25],all_runs=runs,
    previous_407_722_GiB_release_excluded=True,scan_scope='Allocated blocks once per dev/inode, no symlink following. Final scanner SQLite itself excluded; report/export created after snapshot listed separately.',
    generation_time_rule='Nearest actual run metadata timestamp; unknown if absent. Neither mtime nor birth is treated as numeric generation time.',
    consumers=dict(original400='Protected raw expert corpus',current261_raw='Actual A/B replay pools and raw-label/fitting readers',
        current261_features='Actual selected model fixed-history and future authorized training; retained',
        formal_models='Historical130-50/entry/B72 and paired parent/new complete pairs; retained',
        machine_review='Two true rescue inputs/features, interrupted39-label branch, all numeric/outcomes/provenance retained'))
(A/'disk_final_accounting.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(R/'outputs/maintenance_20261123/README_MACHINE_REVIEW_20261126.md').write_text(
    '# 原机复核清理追加账本\n\n'
    f"round2 发布基线 {baseline/2**30:.6f} GiB；本轮快照 {raw['GiB']:.6f} GiB。"
    f"本轮删除 {deleted['files']} 个路径，按最后硬链接实际释放 {deleted['released_GiB']:.6f} GiB；不重复计上次 407.722 GiB。\n\n"
    'inventory.sqlite 的 cleanup_machine_review_20261126 表为追加记录；逐文件 birth（不可得为 unknown）、mtime、SHA、inode、理由和实际释放见 '
    '../../outputs/diagnostics/tight_machine_review_20261126_01/retirement_deletions.jsonl。完整差异与剩余大户见同目录 disk_final_accounting.json、file_changes_since_round2.csv.gz 和 disk_after.sqlite。\n\n'
    '400 原始专家、261 唯一实际回放输入与特征、正式完整模型、成功/失败核心数值均保留。诊断新救回 2 条及特征、中断但有 39 合法专家标签的分支保留；无标签的独立传感副本已退役。\n')
print(json.dumps({k:result[k] for k in ('published_round2_baseline_GiB','net_since_published_round2_GiB','changed_paths')}));print('after',raw['GiB'],'freed',deleted['released_GiB']);print([(x['run'],round(x['GiB'],3)) for x in runs[:8]])
