"""Retire diagnostic captures with no legal expert suffix; preserve core traces.

The two successful real branches and their new features are explicit live inputs.
Never uses failed/success in a path as a dependency decision.
"""
from pathlib import Path
import json,time,sqlite3,hashlib,os
import h5py
from cleanup_review import birth,R,O
ROOT=R/'outputs/visuotactile/run/tight_machine_review_20261126_01'
keep={'B72_28817004_expert','B72_28817008_expert'}
openinodes=set()
for proc in Path('/proc').glob('[0-9]*'):
    try:
        for fd in (proc/'fd').iterdir():
            try:s=fd.stat();openinodes.add((s.st_dev,s.st_ino))
            except OSError:pass
    except OSError:pass
deleted=[];blocked=[]
for p in sorted(ROOT.glob('never_gate_pilot*/rollouts/*/demonstration/data.h5*')):
    case=p.parent.parent
    if case.name in keep:continue
    s=p.lstat()
    if (s.st_dev,s.st_ino) in openinodes:blocked.append(str(p));continue
    numeric=case/'numeric.h5';partial=p.name!='data.h5'
    try:
        with h5py.File(p) as f:
            g=next(iter(f['episodes'].values()));tr=g['transition'];n=len(tr['step_id'])
            if 'dagger/expert_label_valid' in tr and tr['dagger/expert_label_valid'][:].any():
                blocked.append(str(p));continue
            if not numeric.exists():
                # A closed interrupted sensor writer can still have a complete
                # prefix numeric dataset. Keep every recorded nonimage field.
                tmp=case/'numeric_retained.h5'
                with h5py.File(tmp,'x') as out:
                    e=out.create_group('episode')
                    for key in ('obs','transition','terminated','truncated'):
                        if key in g:g.copy(key,e)
                    for k,v in f.attrs.items():out.attrs[k]=v
                numeric=tmp
            with h5py.File(numeric) as f2:
                core=f2['episode/transition']
                assert len(core['step_id'])==n
                for key in ('target_after_rad','diagnostics_before/joint_position_rad','diagnostics_after/object_pose_w','native_keypoints_max_dist'):
                    assert key in core
    except (OSError,KeyError) as exc:
        # No readable physics frames means no reconstructable trajectory;
        # retain the exact initialization/failure identity, not corrupt bytes.
        n=0
        assert (case/'capture_error.json').exists() or (case.parent/'worker_failure.json').exists()
        if not partial:blocked.append(str(p));continue
    reason='Closed diagnostic capture with zero legal expert labels; full core numeric, result or failure identity, seed/time and pair retained; no fitting/training/branch consumer'
    x=dict(path=str(p.relative_to(R)),device=s.st_dev,inode=s.st_ino,allocated=s.st_blocks*512,
        size=s.st_size,nlink=s.st_nlink,birth_utc=birth(p),birth_time_status='known' if birth(p) is not None else 'unknown',
        mtime_utc=s.st_mtime,numeric_generated_time_source=str(case/'started.json'),
        preserved_numeric=str(numeric) if numeric.exists() else None,recorded_ticks=n,reason=reason,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),deleted_utc=time.time(),
        released_bytes=s.st_blocks*512 if s.st_nlink==1 else 0)
    p.unlink();deleted.append(x)
    with (O/'retirement_deletions.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
c=sqlite3.connect(R/'outputs/maintenance_20261123/inventory.sqlite')
c.executemany('insert into cleanup_machine_review_20261126 values(?,?,?,?,?,?,?,?,?,?)',
    [(x['path'],x['inode'],x['device'],x['allocated'],x['released_bytes'],x['birth_utc'],x['mtime_utc'],x['deleted_utc'],x['reason'],x['sha256']) for x in deleted]);c.commit();c.close()
before=json.loads((O/'retirement_result.json').read_text())
before.update(files=before['files']+len(deleted),released_bytes=before['released_bytes']+sum(x['released_bytes'] for x in deleted),
    diagnostic_retired=len(deleted),utc=time.time())
before['released_GiB']=before['released_bytes']/2**30
(O/'retirement_result.json').write_text(json.dumps(before,indent=2)+'\n')
(O/'diagnostic_retirement.json').write_text(json.dumps(dict(deleted=deleted,blocked=blocked,retained_inputs=sorted(keep)),indent=2)+'\n')
print('retired',len(deleted),'released GiB',sum(x['released_bytes'] for x in deleted)/2**30,'blocked',blocked)
