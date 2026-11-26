"""Final small-metadata accounting; no GPU, tensors, HDF5 or log scans."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'


def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (start,end)) or end<start:
            raise ValueError('Invalid recorded interval')
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return dict(intervals=merged,seconds=sum(b-a for a,b in merged))


def inventory():
    sources={}
    def read(path,required=True):
        if not path.exists() and not required:return None
        raw=path.read_bytes()
        if len(raw)>8*2**20:raise ValueError('Large metadata refused')
        sources[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    main=read(HERE/'driver_main_recovery05/status.json')
    waiter=read(HERE/'wait_main_shared06/status.json')
    if main.get('stage')!='main_stages_complete' or waiter.get('stage')!='reviewed_main_closed':
        raise ValueError('Real main and waiter closure required')
    watcher=read(HERE/'watch_shared_capacity06/resource_watch_v4_exit.json')
    if watcher.get('modified_schedules')!=0 or watcher.get('conflicts')!=[]:
        raise ValueError('Current watcher must restore and close')
    session=read(ROOT/'session.json');now=time.time()
    components=[]
    for phase,directory in [('parents','common180_parents'),('new','learning')]:
        for role in 'AB':
            for component,sub in [('DP','training'),('Transformer',f'backend/{role}/training')]:
                d=ROOT/directory/role/sub
                deployment=read(d/'deployment.json');finished=read(d/'finished.json')
                completed=(finished.get('complete') is True if component=='DP' else
                    finished.get('target_reached') is True and finished.get('control_tick_epochs',0)>=60)
                if deployment.get('epoch')!=60 or not completed:
                    raise ValueError('All eight real epoch60 components required')
                updates=finished.get('updates');deployed_updates=deployment.get('update')
                if (type(updates) is not int or updates<0 or type(deployed_updates) is not int
                        or deployed_updates<0 or deployed_updates!=updates):
                    raise ValueError('Deployment and finished committed update counts differ or are invalid: '+str(d))
                components.append(dict(phase=phase,role=role,component=component,
                    deployment=deployment,actual_committed_updates=updates,
                    timer_fields={k:v for k,v in finished.items() if k in ('wall_seconds','wall_s','training_wall_s','setup_wall_s','wait_s','work_s')},
                    timer_limit='Host work and resource waits included; recovered DP timer may reset; these are not GPU-active hours'))
    owners=[];intervals=[]
    required_owners={ROOT/'owners'/name for name in ('main_cache','main_train')}
    required_owners.update(ROOT/'evaluation'/c for c in ('development','test'))
    owner_paths=sorted(set((ROOT/'owners').glob('*'))|required_owners)
    for d in owner_paths:
        started=read(d/'project_job_started.json',False)
        if not started:
            if d in required_owners:raise ValueError('Current owner start receipt missing: '+str(d))
            continue
        a=started['utc'];union([[a,a]])
        historical_collection=d==ROOT/'owners/collection'
        closed=read(d/'project_exit.json',False)
        if closed is None:
            if (d/'project_exit.json').exists():raise ValueError('Invalid owner exit receipt: '+str(d))
            if not historical_collection:raise ValueError('Owner exit receipt missing: '+str(d))
            owners.append(dict(path=str(d),start_utc=a,end_utc=None,elapsed_s=None,
                failure=read(d/'project_failure.json',False),children_reaped=None,
                closure_status='historical_exit_missing',wall_time_certified=False,
                start_receipt=str(d/'project_job_started.json'),exit_receipt=None,
                limitation='Original collection lost its process without an exit receipt. End time and closed wall duration are unknown; excluded from the certified interval union.'))
            continue
        if closed.get('children_reaped') is not True:raise ValueError('Unreaped owner '+str(d))
        b=closed['utc'];union([[a,b]])
        intervals.append([a,b]);owners.append(dict(path=str(d),start_utc=a,end_utc=b,
            elapsed_s=b-a,failure=read(d/'project_failure.json',False),children_reaped=True,
            closure_status='closed',wall_time_certified=True,
            start_receipt=str(d/'project_job_started.json'),exit_receipt=str(d/'project_exit.json')))
    standalone=[]
    for stage in ('summarize_development','summarize_test','audit_collection_cuda_noise_recovery05'):
        admission=read(HERE/'driver_main_recovery05'/(stage+'_shared_capacity06_admission.json'))
        end=read(ROOT/'cycle_logs'/(stage+'_exit.json'))
        if end.get('returncode')!=0:raise ValueError('Standalone audit not complete')
        a,b=admission['utc'],end['utc'];intervals.append([a,b])
        standalone.append(dict(stage=stage,start_admission_utc=a,end_utc=b,elapsed_s=b-a,
            scope='Admission-to-exit wall time; includes Python/CUDA setup and host audit, not measured GPU activity'))
    discarded=[]
    for path in sorted((ROOT/'common180_parents').glob('*/training/recovery.*.json')):
        row=read(path);ids=row.get('known_discarded_update_ids',[])
        discarded.append(dict(path=str(path),update_ids=ids,age_and_exposure_credit=0))
    archives=[]
    for name in ('leased_archive_capacity01','leased_archive_capacity02','leased_archive_shared06','leased_archive_shared07'):
        d=HERE/name;registration=read(d/'registration.json',False)
        if not registration:continue
        end=read(d/'complete.json',False) or read(d/'failure.json',False)
        if not end:raise ValueError('Cold archive still live '+name)
        rows=[read(f) for f in sorted(d.glob('file_*.json'))]
        archives.append(dict(attempt=name,registration=registration.get('binding'),
            start_utc=registration['utc'],end=end,completed_files=len(rows),
            saved_bytes=sum(r['saved_bytes'] for r in rows),
            scope='CPU-only cold auxiliary archives; interrupted originals/failures preserved'))
    collection=read(ROOT/'collection/collection_summary.json')
    final=read(ROOT/'evaluation/test/summary/primary_summary.json')
    parent_updates=sum(r['actual_committed_updates'] for r in components if r['phase']=='parents')
    new_updates=sum(r['actual_committed_updates'] for r in components if r['phase']=='new')
    return dict(version='paired_final_compute_shared06_v1',final=True,cutoff_utc=now,
        cutoff_iso=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
        session_started_utc=session['started_utc'],wall_elapsed_to_accounting_cutoff_s=now-session['started_utc'],
        original_deadline=session['deadline'],components=components,
        committed_parent_updates=parent_updates,committed_main_updates=new_updates,
        discarded_parent_updates=discarded,discarded_parent_update_count=sum(len(r['update_ids']) for r in discarded),
        discarded_update_limit='Discarded attempts are recorded separately and never added to deployed age/exposure.',
        owners=owners,standalone_audits=standalone,
        owner_and_standalone_interval_union=union(intervals),
        interval_scope='Union of certified closed project owners including evaluation owners and standalone CUDA audit admission-to-exit. Includes waits/setup/failures. A historical owner without an exit remains explicit unknown and is excluded. Excludes unrecorded activity and is not GPU-active compute.',
        cold_archives=archives,collection=collection,
        test_summary_path=str(ROOT/'evaluation/test/summary/primary_summary.json'),
        test_summary_fields=list(final),gpu_hours=None,
        gpu_hours_reason='No per-job GPU-active integration exists; wall intervals and CUDA residency cannot establish GPU-hours.',
        source_small_metadata_sha256=sources)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=inventory()
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':main()
