"""Run-local exact identity adapter; original v4 pause and restore logic retained."""
from pathlib import Path
from outputs.diagnostics.tight_paired_dagger_20261122_01 import resource_watch_v4 as base
from outputs.diagnostics.tight_paired_dagger_20261122_01 import resource_watch_parallel_proposal as parallel
MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.collection_recovery05'
ADAPTER=Path(__file__).resolve().with_name('collection_recovery05.py')
DEFAULT_ROOT=base.DEFAULT_ROOT
JOB=DEFAULT_ROOT/'collection_recovery05_job.json'
OWNER=DEFAULT_ROOT/'owners/collection_recovery05'
OUTPUT=DEFAULT_ROOT/'collection_recovery05'
original_worker=base.registered_worker

def specs(root, config):
    rows=parallel.specs(root,config)
    rows.append(dict(name='collection_recovery05',owner=Path(root)/'owners/collection_recovery05',
        module=MODULE,operation='dagger',job=Path(root)/'collection_recovery05_job.json'))
    return rows

def registered_worker(pid, registrations, root, lookup=base.process):
    proc=lookup(pid)
    if proc:
        for reg in registrations:
            spec=reg['spec']
            if spec['name']!='collection_recovery05':continue
            try:
                root=Path(root).resolve();owner=root/'owners/collection_recovery05';out=root/'collection_recovery05'
                exact_spec=dict(name='collection_recovery05',owner=owner,module=MODULE,operation='dagger',job=root/'collection_recovery05_job.json')
                if spec!=exact_spec or owner.is_symlink() or out.is_symlink():continue
                if reg['receipt'].get('job')!=reg['job']:continue
                status=base.read(owner/'project_status.json');index=status['stage']
                if type(index) is not int or index!=0:continue
                stage=reg['job']['stages'][index]
                if stage!=dict(entry='paired_collection_recovery05',args=['--output',str(out)]):continue
                expected=['python','-m',MODULE,'worker','--output',str(out)]
                if (status.get('pid')!=pid or not base.born_before_receipt(proc,status.get('utc'))
                    or not base.descends(proc,reg['owner_process'],lookup)
                    or not base.matching_command(proc['command'],expected,proc['cwd'])
                    or not base.matching_command(status['command'],expected,proc['cwd'])
                    or not base.matching_command(reg['receipt']['commands'][index],expected,proc['cwd'])):continue
                binding=base.read(out/'registration.json');plan=base.read(out/'plan.json')
                checks=[Path(binding.get('root','/')).resolve()==Path(root).resolve(),
                    Path(binding.get('owner','/')).resolve()==owner.resolve(),
                    Path(binding.get('output','/')).resolve()==out,
                    Path(binding.get('job','/')).resolve()==Path(spec['job']).resolve(),
                    binding.get('plan_sha256')==base.sha((out/'plan.json').read_bytes()),
                    Path(binding.get('recovery_adapter','/')).resolve()==ADAPTER,
                    binding.get('recovery_adapter_sha256')==base.sha(ADAPTER.read_bytes()),
                    Path(plan.get('resource_root','/')).resolve()==Path(root).resolve()]
                if not all(checks):continue
                return dict(pid=pid,created=proc['created'],name=spec['name'],output=str(out),
                    owner_pid=reg['owner_process']['pid'],owner_created=reg['owner_process']['created'])
            except (OSError,KeyError,IndexError,TypeError,ValueError):continue
    return original_worker(pid,registrations,root,lookup)

def main():
    base.specs=specs;base.registered_worker=registered_worker;base.main()
if __name__=='__main__':main()
