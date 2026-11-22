"""Run the requested existing-history comparison only after primary delivery."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.tight_entry_final24_20261122.summarize import read, sha, write

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
EVAL = PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01'
OLD = PROJECT/'outputs/diagnostics/tight_entryfix_20261121_01'
ROOT = PROJECT/'outputs/diagnostics/tight_entry_final24_20261122_01'
PUBLIC = PROJECT/'research_docs/evidence/tight_entry_final24_20261122_01'


def main():
    assert read(PUBLIC/'primary_summary.json')['complete']
    assert read(EVAL/'project_exit.json')['children_reaped']
    assert not ROOT.exists(), 'Keep every diagnostic attempt'
    ROOT.mkdir()
    session = read(EVAL/'session.json')
    session.update(resource_root=str(ROOT), owned_worker_roots=[str(ROOT)],
        diagnostic_started_utc=time.time(),
        authorization='Frozen existing new50 real-history inference only; no optimizer or simulator')
    session['deadline'] = session['training_deadline'] = min(session['deadline'], time.time()+2*3600)
    write(ROOT/'session.json', session)
    for name in ('entry_same_history_plan.json', 'A_new50_entry_selected_20261122.csv'):
        shutil.copy2(OLD/name, ROOT/name)
    plan = read(ROOT/'entry_same_history_plan.json')
    cases = [c for c in plan['cases'] if c['new50']]
    assert len(cases)==50 and sum(len(c['ticks']) for c in cases)==334
    old_registration = read(PROJECT/'research_docs/evidence/tight_entry_phase_20261121_01/development/registration.json')
    bindings = {}
    for group in 'AB':
        for tag, key in [('original_R2',group+'_R2'),('entry_repaired',group+'_entry')]:
            pair = plan['pairs'][group][tag]
            reference = old_registration['pairs'][key]
            for name in ('frontend','backend'):
                assert pair[name]==reference[name]
                assert sha(pair[name])==pair[name+'_sha256']==reference[name+'_sha256']
                bindings[pair[name]] = pair[name+'_sha256']
    verified = []
    for case in cases:
        assert sha(case['raw_path'])==case['raw_source_sha256']
        assert sha(case['visual_cache_path'])==case['visual_cache_sha256']
        for group in 'AB':
            data_root = PROJECT/f'outputs/visuotactile/run/tight_entry_phase_20261121_01/{group}/state_data'
            data = read(data_root/'data.json')['entries'][case['identity']]
            assert data['path']==case['raw_path'] and data['source_sha256']==case['raw_source_sha256']
            assert sha(data_root/'perception'/(case['identity']+'.pt'))==case['visual_cache_sha256']
        verified.append(dict(identity=case['identity'],raw_sha256=case['raw_source_sha256'],
            visual_sha256=case['visual_cache_sha256'],family=data['family'],both_role_inputs_identical=True))
    write(ROOT/'input_verification.json', dict(cases=verified, pairs=bindings,
        original_script_sha256=sha(OLD/'entry_same_history.py'),
        executed_script_sha256=sha(PROJECT/'scripts/tight_entry_final24_20261122/entry_history.py'),
        original_plan_sha256=sha(OLD/'entry_same_history_plan.json')))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    env['PYTHONPATH'] = str(PROJECT)
    command = [sys.executable,'-m','scripts.tight_entry_final24_20261122.entry_history','run',
        '--roles','AB','--tags','original_R2,entry_repaired','--population','new50','--output','history_new50']
    write(ROOT/'job.json', dict(version='tight_entry_history_job_v1', command=command,
        deadline=session['deadline'], optimizer_updates=0,new_dagger_trajectories=0,physics_steps=0))
    monitor = child = None
    started = time.time()
    try:
        with (ROOT/'monitor.log').open('x') as log:
            monitor = subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],
                cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        until = time.time()+30
        while not (ROOT/'shared_resources.json').exists():
            assert monitor.poll() is None and time.time()<until
            time.sleep(.2)
        with (ROOT/'console.log').open('x') as log:
            child = subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        write(ROOT/'job_started.json',dict(pid=child.pid,utc=time.time(),command=command))
        while child.poll() is None:
            assert monitor.poll() is None and time.time()<session['deadline']
            time.sleep(1)
        assert child.returncode==0, f'Diagnostic worker failed: {child.returncode}'
        for path,digest in bindings.items():
            assert sha(path)==digest
        write(ROOT/'complete.json',dict(wall_s=time.time()-started,weights_unchanged=True,
            optimizer_updates=0,new_dagger_trajectories=0,physics_steps=0))
    except BaseException as exc:
        write(ROOT/'failure.json',dict(error=repr(exc),utc=time.time()))
        raise
    finally:
        from tools.tight_project import stop
        stop(child)
        (ROOT/'STOP_MONITOR').touch()
        if monitor:
            try:
                monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop(monitor)
        write(ROOT/'exit.json',dict(children_reaped=True,utc=time.time()))


if __name__=='__main__':
    main()
