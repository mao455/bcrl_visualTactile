"""One explicit pre-action recorder repair; retain identity, deadline and budget."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools import h24_diagnosis_run as base
from tools.h24_diagnosis_common import ROOT, read, sha
from visuotactile.training_resources import atomic_json

RECOVERY='recorder_recovery_01'

def register(root):
    folder=root/RECOVERY
    if (folder/'registration.json').exists():raise RuntimeError('Already registered')
    assert time.time()<read(root/'plan.json')['deadline']
    error=read(root/'stages/V1/worker_error.json')
    assert 'name parameter cannot be an empty string' in error['error']
    assert not list((root/'stages/V1').rglob('trace.npz'))
    assert not list((root/'stages/V1').rglob('progress.json'))
    exited=read(root/'stages/V1_process_exit.json')
    assert exited['code']==1 and exited['reaped'] and not Path('/proc',str(exited['pid'])).exists()
    overrides={name:sha(base.CODE/name) for name in ['tools/h24_diagnosis_worker.py']}
    for name,digest in read(root/'frozen.json')['source_sha256'].items():
        if name not in overrides:assert sha(base.CODE/name)==digest
    files={str(p.relative_to(root)):sha(p) for p in (root/'stages/V1').rglob('*') if p.is_file()}
    files.update({str(p.relative_to(root)):sha(p) for p in folder.rglob('*') if p.is_file()})
    atomic_json(folder/'registration.json',dict(overrides=overrides,protected=files,
        launcher_sha256=sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        original_plan_sha256=sha(root/'plan.json'),original_session_sha256=sha(root/'session.json'),
        original_deadline=read(root/'plan.json')['deadline'],utc=time.time(),
        reason='Flat policy recorder passed empty group parent; failure before first env.step. Fix only flat numeric writing.',
        startup_budget='Original failed start counts; V1 recovery is its second and final attempt, no V1_attempt2 extra'))

def verify(root):
    registration=read(root/RECOVERY/'registration.json')
    assert sha(Path(__file__))==registration['launcher_sha256']
    assert sha(root/'plan.json')==registration['original_plan_sha256']
    assert sha(root/'session.json')==registration['original_session_sha256']
    frozen=read(root/'frozen.json')
    for name,digest in frozen['source_sha256'].items():
        assert sha(base.CODE/name)==registration['overrides'].get(name,digest),name
    for name,digest in registration['protected'].items():assert sha(root/name)==digest,name

class RecoveryOwner(base.Owner):
    def stage(self,name,*args,**kwargs):
        if name=='V1_attempt2':
            atomic_json(self.root/RECOVERY/'v1_budget_used.json',dict(reason='First pre-action software failure and registered recovery use both V1 starts; no extra repeat',utc=time.time()))
            return None
        if name=='V1' and name not in self.results:
            result=super().stage('V1_recording_recovery',*args,**kwargs)
            if result:
                self.results.pop('V1_recording_recovery')
                self.results['V1']=result
                atomic_json(self.root/'results.json',self.results)
            return result
        return super().stage(name,*args,**kwargs)

def run(root):
    verify(root)
    folder=root/RECOVERY
    if (folder/'started.json').exists():raise RuntimeError('Recovery already started; no automatic repeat')
    previous=read(root/'results.json')
    assert set(previous)=={'V1'} and previous['V1']['state']=='failed'
    previous['V1_software_attempt1']=previous.pop('V1')
    atomic_json(root/'results.json',previous)
    # Original bytes are already archived and hashed. These are current lifecycle
    # markers, not failed trajectories, logs or experimental result files.
    (root/'owner_exit.json').rename(folder/'prior_owner_exit.json')
    (root/'STOP_MONITOR').rename(folder/'prior_STOP_MONITOR')
    atomic_json(folder/'started.json',dict(pid=os.getpid(),utc=time.time(),registration_sha256=sha(folder/'registration.json')))
    base.verify=verify
    base.Owner=RecoveryOwner
    base.run(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','run'])
    parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    os.chdir(base.CODE)
    (register if args.command=='register' else run)(args.root.resolve())
