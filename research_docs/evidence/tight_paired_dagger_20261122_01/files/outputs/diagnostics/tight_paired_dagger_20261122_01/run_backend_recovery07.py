"""Explicit continuation of never-started backends; no DP or physical retry.

Preparation writes new AUDIT registration/job files only. Root alone activates
the new finite owner after reviewing the source amendment and watcher handoff.
Frozen failed owners, sources, completed DPs, caches and receipts are preserved.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import parallel_rear_proposal as pipeline
import parallel_rear_shared06 as shared
import run_main_recovery05 as original
import shared_capacity06 as capacity

ROOT, PROJECT, CONFIG = original.ROOT, original.PROJECT, original.CONFIG
MODULE = original.MODULE + 'run_backend_recovery07'
WORKER_MODULE = original.MODULE + 'parallel_rear_recovery07'
OUT = HERE / 'driver_backend_recovery07'
OWNER = ROOT / 'owners/backend_recovery07'
JOB = HERE / 'proposed_backend_recovery07.json'
REGISTRATION = HERE / 'BACKEND_RECOVERY07_REGISTRATION.json'
RUNTIME = HERE / 'parallel_rear_recovery07_runtime'
LOGS = ROOT / 'cycle_logs_recovery07'
WATCH = HERE / 'watch_shared_capacity07'
WATCH_MODULE = original.MODULE + 'shared_capacity07'
ALLOWED_AMENDMENT_PATHS = frozenset((
    'tools/tight_fresh_backend.py', 'visuotactile/fresh_contracts.py',
    'visuotactile/anchor_continue.py'))


def backend_job():
    return dict(version='tight_project_job_v1', operation='train',
        resource_root=str(OWNER), stages=[dict(entry='paired_backend_recovery07',
        args=['--root', str(ROOT / 'learning'), '--registration', str(REGISTRATION)])])


def stages():
    yield 'train_backend_recovery07', ['-m', WORKER_MODULE, 'project', 'train', '--job', str(JOB)], True
    found = False
    for name, args, gpu in original.stages():
        if name == 'verify_final_actual_training':
            found = True
        if found:
            # The old main never reached this guard; preserve its absent output.
            if name == 'verify_final_actual_training':
                args = [*args[:-1], str(HERE / 'presealed_guard_main_recovery07.json')]
            yield name, args, gpu


def terminal_paths():
    return [
        HERE / 'driver_main_recovery05/failure.json',
        HERE / 'driver_main_recovery05/status.json',
        HERE / 'driver_main_recovery05/registration.json',
        HERE / 'wait_main_shared06/failure.json',
        HERE / 'wait_main_shared06/reviewed_main_exit.json',
        ROOT / 'owners/main_train/project_failure.json',
        ROOT / 'owners/main_train/project_exit.json',
        ROOT / 'cycle_logs/train_four_components_parallel_rears_exit.json',
        HERE / 'parallel_rear_shared06_runtime/serial_caches/pipeline_events.json',
        HERE / 'parallel_rear_shared06_runtime/serial_caches/A_stage_00.log',
        HERE / 'PAIRED_CONTRACT07_ACTUAL_PREFLIGHT.json',
        HERE / 'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json',
        HERE / 'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json',
        HERE / 'PAIRED_CONTRACT07.patch',
    ]


def preserved_dp(root=ROOT, *, verify_payload=False):
    result = {}
    hashed = {}
    for role in 'AB':
        train = Path(root) / 'learning' / role / 'training'
        deployment, finished = original.read(train / 'deployment.json'), original.read(train / 'finished.json')
        original.require(deployment['epoch'] == 60 and deployment['update'] == 10200,
            'Only completed registered epoch60 DPs may continue')
        original.require(finished.get('complete') is True and finished['epochs'] == 60
            and finished['updates'] == 10200, 'DP completion differs')
        original.require(deployment['selection'] == 'preregistered_epoch60_only'
            and deployment['validation_used_for_selection'] is False, 'DP selection changed')
        expected = deployment['checkpoint_sha256']
        original.require(deployment['final_sha256'] == expected, 'DP final identity differs')
        paths = [train / 'best.pt', train / 'last.pt']
        for path in paths:
            original.require(path.is_file() and not path.is_symlink(), 'Completed DP payload absent or aliased')
            if verify_payload:
                st = path.stat(); key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                if key not in hashed:
                    hashed[key] = original.digest(path)
                original.require(hashed[key] == expected, 'Completed DP SHA changed')
        result[role] = dict(checkpoint_sha256=expected,
            deployment_sha256=original.digest(train / 'deployment.json'),
            finished_sha256=original.digest(train / 'finished.json'),
            deployment=str(train / 'deployment.json'), finished=str(train / 'finished.json'),
            checkpoints=[str(path) for path in paths], epoch=60, updates=10200)
    return result


def expected_registration(*, now=None):
    from tools.tight_paired_dagger import sources
    old = original.read(HERE / 'driver_main_recovery05/registration.json')
    current = sources()
    original.require(current.keys() == old['source_sha256'].keys(), 'Production source population changed')
    changed = {path: dict(old_sha256=old['source_sha256'][path], new_sha256=sha)
        for path, sha in current.items() if sha != old['source_sha256'][path]}
    original.require(set(changed) == ALLOWED_AMENDMENT_PATHS,
        'Only explicitly reviewed paired backend contract paths may differ')
    session = original.read(ROOT / 'session.json')
    original.require(time.time() < session['deadline'] == old['deadline'], 'Original finite deadline changed or expired')
    original.require(original.read(ROOT / 'owners/main_train/project_exit.json').get('children_reaped') is True,
        'Prior training owner must be closed and reaped')
    original.require(original.read(HERE / 'wait_main_shared06/reviewed_main_exit.json')['returncode'] == 1,
        'Prior failed main terminal identity differs')
    for role in 'AB':
        original.require(not (ROOT / 'learning' / role / 'backend').exists(),
            'A backend attempt already exists; never replace it')
    for path in (OWNER, OUT, RUNTIME, LOGS, ROOT / 'evaluation/development', ROOT / 'evaluation/test'):
        original.require(not path.exists(), 'Preserve attempted continuation/evaluation identity: ' + str(path))
    snapshot_names = ('run_backend_recovery07.py', 'parallel_rear_recovery07.py', 'shared_capacity07.py',
        'parallel_rear_proposal.py', 'parallel_rear_shared06.py', 'run_main_recovery05.py',
        'shared_capacity06.py', 'resource_watch_v4.py', 'presealed_guard.py',
        'audit_collection_recovery05.py', 'driver_support.py', 'driver_support_v4.py')
    return dict(version='paired_backend_only_recovery07_v1', root=str(ROOT),
        created_utc=time.time() if now is None else now, original_deadline=session['deadline'],
        source_sha256=current, old_production_sha256=old['source_sha256'],
        source_amendment=dict(changed_paths=sorted(changed), changes=changed,
            receipt=str(HERE/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json'),
            receipt_sha256=original.digest(HERE/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json'),
            patch=str(HERE/'PAIRED_CONTRACT07.patch'),
            patch_sha256=original.digest(HERE/'PAIRED_CONTRACT07.patch'),
            scope='Explicit paired complete-demo plus own expert-suffix contract; no DP replay, new data, optimizer/sampling/target/native inference changes'),
        old_watcher_closure=dict(path=str(HERE/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json'),
            sha256=original.digest(HERE/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json')),
        old_terminal_sha256={str(path): original.digest(path) for path in terminal_paths()},
        preserved_dp=preserved_dp(), owner=str(OWNER), job=str(JOB), logs=str(LOGS),
        job_sha256=pipeline.digest(JOB), watcher=str(WATCH), watcher_module=WATCH_MODULE,
        watcher_registration='Exact old watcher closure and original schedule restoration required before one new watcher starts',
        stages=[dict(name=n, command=a, gpu=g) for n,a,g in stages()],
        orchestration_sha256={str(HERE / name): original.digest(HERE / name) for name in snapshot_names},
        prior_registration_sha256=original.digest(HERE / 'driver_main_recovery05/registration.json'),
        backend_schedule=deepcopy(original.read(HERE / 'parallel_rear_shared06_registration.json')),
        scientific_invariants=dict(dp_new_updates=0, dp_reruns=0, collection_retries=0,
            new_opportunities=0, optimizer_epochs=60, deploy_epoch=60,
            rear_updates_per_role=38400, caches_own_dp=True, encoder_frozen=True,
            development_descriptive_only=True, sealed_test_families=64,
            software_failures='Primary failures with no outcome-dependent retries',
            external_presence_is_veto=False), activated=False)


def prepare():
    original.require(not JOB.exists() and not REGISTRATION.exists(), 'Preserve prepared continuation identity')
    pipeline.write_new(JOB, backend_job())
    try:
        registration = expected_registration()
    except BaseException:
        # The new job is preparation evidence and is deliberately retained.
        raise
    pipeline.write_new(REGISTRATION, registration)
    print(json.dumps(dict(status='prepared_no_activation', registration=str(REGISTRATION),
        registration_sha256=original.digest(REGISTRATION), owner=str(OWNER), stages=registration['stages'])))


def validate_registration(*, payload=False, require_absent_owner=False):
    from tools.tight_paired_dagger import sources
    reg = original.read(REGISTRATION)
    original.require(reg['version'] == 'paired_backend_only_recovery07_v1' and reg['root'] == str(ROOT), 'Recovery registry identity differs')
    original.require(reg['original_deadline'] == original.read(ROOT / 'session.json')['deadline']
        and time.time() < reg['original_deadline'], 'Original deadline changed or expired')
    original.require(reg['source_sha256'] == sources(), 'Registered amended production sources changed')
    original.require(reg['old_production_sha256'] == original.read(HERE / 'driver_main_recovery05/registration.json')['source_sha256'], 'Historical source chronology changed')
    actual_changes = {path: dict(old_sha256=reg['old_production_sha256'][path], new_sha256=sha)
        for path,sha in reg['source_sha256'].items() if sha != reg['old_production_sha256'][path]}
    original.require(actual_changes == reg['source_amendment']['changes']
        and sorted(actual_changes) == reg['source_amendment']['changed_paths']
        and set(actual_changes) == ALLOWED_AMENDMENT_PATHS, 'Unexpected production amendment')
    for mapping in ('orchestration_sha256', 'old_terminal_sha256'):
        for path, sha in reg[mapping].items():
            original.require(original.digest(path) == sha, 'Pinned historical/orchestration file changed: ' + path)
    original.require(pipeline.digest(JOB) == reg['job_sha256'] and original.read(JOB) == backend_job(), 'Backend-only job changed')
    original.require(reg.get('logs') == str(LOGS), 'Recovery logs path differs')
    original.require(preserved_dp(verify_payload=payload) == reg['preserved_dp'], 'Completed DP identities changed')
    original.require(reg['stages'] == [dict(name=n,command=a,gpu=g) for n,a,g in stages()], 'Recovery stage list changed')
    if require_absent_owner:
        original.require(not OWNER.exists() and not OUT.exists() and not RUNTIME.exists() and not LOGS.exists(), 'Recovery already attempted')
        for role in 'AB':
            original.require(not (ROOT / 'learning' / role / 'backend').exists(), 'Previous backend attempt must be preserved')
        for cohort in ('development','test'):
            original.require(not (ROOT / 'evaluation' / cohort).exists(), 'Evaluation already attempted')
    return reg


def rear_pair(root, registration):
    original.require(Path(root).resolve() == ROOT / 'learning' and Path(registration).resolve() == REGISTRATION, 'Exact backend root/registry required')
    reg = validate_registration(payload=True)
    for role in 'AB':
        original.require(not (ROOT / 'learning' / role / 'backend').exists(), 'Never restart a backend attempt')
    launch = deepcopy(reg['backend_schedule'])
    launch.update(owner=str(OWNER), root=str(ROOT / 'learning'), deadline=reg['original_deadline'])
    affinity = set(launch['cpu_affinity'])
    original.require(affinity <= os.sched_getaffinity(0), 'Registered CPU8-15 not available')
    os.sched_setaffinity(0, affinity)
    from resource_watch_v4 import external_test_processes
    deadline = reg['original_deadline']
    stops = [ROOT/'STOP', ROOT/'STOP_CYCLE', ROOT/'learning/STOP', OWNER/'STOP']
    RUNTIME.mkdir(exist_ok=False)
    allowance = dict(launch, vram_per_worker_mib=launch['cache_and_offline_vram_per_worker_mib'])
    def ready():
        try:
            row = original.read(OWNER/'shared_resources.json')
            decision = shared.readiness(row,external_test_processes(),launch,time.time())
            admission = shared.admission(row,allowance,time.time(),1)
            decision.update(allowed=decision['allowed'] and admission['allowed'], launch_capacity=admission, raw_monitor_row=row)
        except (OSError,KeyError,TypeError,ValueError) as exc:
            decision=dict(allowed=False,reasons=[repr(exc)])
        with (RUNTIME/'readiness.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(utc=time.time(),decision=decision))+'\n')
        return decision['allowed']
    def await_ready():
        while not ready():
            if time.time()>=deadline or any(path.exists() for path in stops):
                raise TimeoutError('Original deadline or STOP during backend capacity wait')
            time.sleep(.5)
    with pipeline.termination_signals():
        await_ready()
        commands={role:pipeline.stage_commands(ROOT/'learning',role) for role in 'AB'}
        prep=RUNTIME/'serial_caches';prep.mkdir()
        pipeline.run_roles({r:commands[r][:2] for r in 'AB'},False,deadline,stops,ready,prep)
        sizes={r:sum(path.stat().st_size for path in (ROOT/'learning'/r/'backend'/r/'nominal').glob('*.pt')) for r in 'AB'}
        actual=dict(launch,ram_per_worker_bytes=max(launch['ram_per_worker_bytes'],2*2**30+3*max(sizes.values())))
        allowance=actual
        await_ready(); snapshot=original.read(OWNER/'shared_resources.json')
        pair=shared.admission(snapshot,actual,time.time(),2);single=shared.admission(snapshot,actual,time.time(),1)
        original.require(pair['allowed'] or single['allowed'], 'No actual remaining capacity for a backend trainer')
        pipeline.write_new(RUNTIME/'admission.json',dict(parallel=pair['allowed'],pair=pair,single=single,
            actual_nominal_cache_bytes=sizes,admission_registration=actual,raw_monitor_row=snapshot,utc=time.time()))
        training=RUNTIME/'rear_training';training.mkdir()
        events=pipeline.run_roles({r:[commands[r][2]] for r in 'AB'},pair['allowed'],deadline,stops,ready,training)
        offline=RUNTIME/'serial_offline';offline.mkdir()
        allowance=dict(actual,vram_per_worker_mib=launch['cache_and_offline_vram_per_worker_mib'])
        pipeline.run_roles({r:[commands[r][3]] for r in 'AB'},False,deadline,stops,ready,offline)
        pipeline.write_new(RUNTIME/'complete.json',dict(utc=time.time(),events=len(events),parallel=pair['allowed'],caches_serial=True,offline_serial=True))


class Driver(original.Driver):
    def gpu_gate(self,name):
        import psutil
        from shared_capacity07 import startup_row
        gpu,ram=(2048,6*2**30) if name=='train_backend_recovery07' else ((11264,16*2**30) if name.startswith('evaluate_') else (1024,2*2**30))
        while True:
            self.finite()
            try:
                row=original.read(WATCH/'resource_watch_v4_status.json')
                decision=startup_row(row,psutil.Process(row['pid']),time.time(),allow_gap=True)
                live=capacity.capacity_decision(capacity.live_capacity(),gpu_resident_mib=gpu,ram_resident_bytes=ram)
                original.require(live['ready'],'Actual remaining capacity below stage allowance plus reserve')
                fresh=original.read(WATCH/'resource_watch_v4_status.json')
                startup_row(fresh,psutil.Process(fresh['pid']),time.time(),allow_gap=True)
                self.write(OUT/(name+'_shared_capacity06_admission.json'),dict(utc=time.time(),watcher=decision,watcher_status=fresh,capacity=live))
                return
            except (OSError,KeyError,TypeError,ValueError,psutil.Error,original.subprocess.SubprocessError) as exc:
                self.note('waiting_actual_shared_capacity_before_'+name,reason=repr(exc));time.sleep(5)

    def run(self,name,args,gpu):
        reg=validate_registration()
        original.require(self.source_identity==reg['source_sha256'],'Recovery source binding changed')
        if name=='train_backend_recovery07':
            path=ROOT/'learning/shared_resources.json'
            original.require(path.is_symlink(),'Only existing training telemetry symlink may be rebound')
            old=os.readlink(path)
            self.write(OUT/'training_telemetry_transition.json',dict(utc=time.time(),old=old,new=str(OWNER/'shared_resources.json')))
            path.unlink();path.symlink_to(OWNER/'shared_resources.json')
            self.write(OUT/'backend_source_identity.json',dict(utc=time.time(),sources=self.source_identity,
                completed_dp_sources_preserved=str(ROOT/'learning/training_source_identity.json')))
        return super().run(name,args,gpu)

    def main(self):
        reg=validate_registration(payload=True,require_absent_owner=True)
        OUT.mkdir(exist_ok=False);self.logs=LOGS;self.logs.mkdir(exist_ok=False)
        self.source_identity=reg['source_sha256']
        OWNER.mkdir(exist_ok=False)
        session=deepcopy(original.read(ROOT/'session.json'));session['resource_root']=str(OWNER)
        self.write(OWNER/'session.json',session)
        self.write(OWNER/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
        self.write(OUT/'registration.json',dict(reg,activated=True,activated_utc=time.time(),registry_sha256=original.digest(REGISTRATION)))
        for name,args,gpu in stages():
            self.run(name,args,gpu)
        self.note('main_stages_complete',report_required=True,dp_reruns=0,collection_retries=0)
        self.write(OUT/'closure.json',dict(utc=time.time(),stage='main_stages_complete',status='backend_and_remaining_main_stages_complete',report_required=True))


def project():
    from tools import tight_project
    tight_project.ENTRIES['paired_backend_recovery07']=('train',WORKER_MODULE,['rear-pair'])
    sys.argv.pop(1);tight_project.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':
        return project()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','rear-pair'])
    parser.add_argument('--root',type=Path);parser.add_argument('--registration',type=Path,default=REGISTRATION)
    args=parser.parse_args()
    if args.mode=='prepare':return prepare()
    if args.mode=='rear-pair':return rear_pair(args.root,args.registration)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def interrupted(signum,frame):raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    original.OUT=OUT
    driver=Driver()
    try:driver.main()
    except BaseException as exc:
        if OUT.exists():
            driver.write(OUT/'failure.json',dict(utc=time.time(),error=repr(exc),traceback=traceback.format_exc()))
            driver.note('stopped_with_evidence',error=repr(exc))
        raise
    finally:
        if OUT.exists():
            driver.write(OUT/'driver_exit.json',dict(utc=time.time(),pid=os.getpid(),returned_after_owned_helpers=True))


if __name__=='__main__':main()
