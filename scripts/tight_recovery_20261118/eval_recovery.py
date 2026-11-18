"""Finish the registered evaluation at eight lanes; preserve RAM-aborted attempt."""
import copy, os, time, traceback
from scripts.tight_recovery_20261118.run import ROOT, OUT, read, write
from scripts.tight_recovery_20261118.review import evaluation_prepare, evaluation
from scripts.tight_cycle_20261118.cycle import run_wave


def main():
    import sys
    recovery = "--recover-import" in sys.argv
    if recovery:
        assert read(ROOT/"eval_recovery_exit.json")["children_reaped"]
        assert "ImportError: cannot import name" in (ROOT/"comparison01_8lane0/worker.log").read_text()
        assert not (ROOT/"comparison01_8lane0/batch_progress.json").exists()
        for name in ["started", "status", "failure", "exit"]:
            p=ROOT/f"eval_recovery_{name}.json"
            if p.exists(): p.rename(ROOT/f"eval_recovery_import_failure_{name}.json")
        (ROOT/"comparison01_8lane0").rename(ROOT/"comparison01_8lane0_import_failure")
    assert not (ROOT/'eval_recovery_started.json').exists()
    assert read(ROOT/'tail_recovery02_exit.json')['children_reaped']
    assert read(ROOT/'comparison00/result.json')['software_complete']
    assert 'RAM reserve' in (ROOT/('comparison01_failed_ram' if recovery else 'comparison01')/'worker.log').read_text()
    write(ROOT/'eval_recovery_started.json', dict(utc=time.time(), pid=os.getpid()))
    failed=ROOT/'comparison01_failed_ram'
    if not recovery:(ROOT/'comparison01').rename(failed)
    write(OUT/'evaluation_recovery.json', dict(
        failed=str(failed), original_deadline=read(ROOT/'session.json')['deadline'],
        remaining_cases=48, physical_waves=6, lanes=8,
        scope='Same registered families and complete pairs. Four models remain within one physical wave per family; native batch context changes, no hidden-state replay claim. Failed partial cases excluded, preserved. No training, new cases or deadline reset.'))
    try:
        for part in range(1,4):
            if not (recovery and part==1):evaluation_prepare(part)
            dest=ROOT/f'comparison{part:02d}'
            plan=read(dest/'plan.json'); seeds=sorted({c['spec']['seed'] for c in plan['cases']})
            waves=[]
            for sub in range(2):
                d=ROOT/f'comparison{part:02d}_8lane{sub}';d.mkdir()
                p=copy.deepcopy(plan);selected={seeds[sub],seeds[sub+2]}
                p['cases']=[c for c in p['cases'] if c['spec']['seed'] in selected]
                assert len(p['cases'])==8
                p['batch']['lanes']=8;p['resource_root']=str(d)
                s=read(dest/'session.json');s.update(resource_root=str(d),owned_worker_roots=[str(d)])
                write(d/'session.json',s);write(d/'plan.json',p)
                write(ROOT/'eval_recovery_status.json',dict(stage=d.name,utc=time.time()))
                run_wave(d);waves.append(d)
            results=[];profiles=[];checks=[];videos=[];seconds=0.;init=0.
            for d in waves:
                results.extend(read(d/'results.json'));profiles.extend(read(d/'wave_profiles.json'))
                checks.extend(dict(physical_wave=d.name,**r) for r in read(d/'paired_initial_check.json')['rows'])
                assert read(d/'paired_initial_check.json')['passed']
                videos.extend(read(d/'videos.json')['rows'])
                seconds+=read(d/'owner_complete.json')['seconds'];init+=read(d/'initialization.json')['seconds']
                for c in read(d/'plan.json')['cases']:(dest/c['name']).symlink_to(d/c['name'],target_is_directory=True)
            plan.update(intake_only=True,physical_waves=[d.name for d in waves]);plan['batch']['lanes']=8
            write(dest/'plan.json',plan);write(dest/'results.json',results);write(dest/'wave_profiles.json',profiles)
            write(dest/'videos.json',dict(complete=True,rows=videos))
            write(dest/'paired_initial_check.json',dict(passed=True,rows=checks,scope='Within physical8-lane waves, four models per family.'))
            write(dest/'initialization.json',dict(seconds=init,lanes=8,waves=2))
            write(dest/'result.json',dict(software_complete=True,cases=16,lanes=8,waves=2,worker_error=None,elapsed_seconds=seconds))
            write(dest/'owner_complete.json',dict(seconds=seconds,utc=time.time(),aggregate=True))
            write(dest/'owner_exit.json',dict(children_reaped=True,utc=time.time(),aggregate=True))
        evaluation()
        write(ROOT/'eval_recovery_complete.json',dict(complete=True,utc=time.time()))
        write(ROOT/'tail_recovery02_complete.json',dict(complete=True,utc=time.time(),completed_by='eval_recovery',scientific_review_required=True,no_automatic_promotion=True))
    except BaseException:
        write(ROOT/'eval_recovery_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
    finally:write(ROOT/'eval_recovery_exit.json',dict(utc=time.time(),children_reaped=True))

if __name__=='__main__':main()
