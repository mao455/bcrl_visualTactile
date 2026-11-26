"""Bounded train-only full-cache gate for opt-in offline nominal-plan batching.

Prepare on CPU, then run owner in a fresh frozen candidate tree. The owner has a
hard wall bound and never retries. Frozen selected-checkpoint dependencies build
the serial reference; only the registered cache function/planning helper differ.
No optimizer, rear training, validation/test or autonomous execution is performed.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.tight_fresh_selected_fit import read, sha256, checked_hash, exclusive_json, import_frozen_trainer
from tools.tight_offline_planning_gate import admit, train_projection, validate_checkpoint, require

VERSION = 'complete_residual_cache_offline_planning_gate_v1'
TARGET_ATOL = 2e-5
PLAN_KEYS = {'plans', 'base', 'previous_nominal'}
PROFILE_KEY = 'cache_offline_planning_profile'


def source_hashes(tree):
    return {str(p.relative_to(tree)):sha256(p) for folder in ('tools', 'visuotactile')
        for p in sorted((tree/folder).rglob('*.py'))}


def compare_caches(reference, candidate, profile):
    """Only plan-derived float fields have tolerance; every other field is exact."""
    import torch
    require(PROFILE_KEY not in reference and candidate.get(PROFILE_KEY) == profile,
        'Explicit candidate profile and unmodified serial identity required')
    require(set(candidate)-{PROFILE_KEY} == set(reference), 'Cache field population differs')
    rows = {}
    def visit(a, b, path, tolerant=False):
        require(type(a) is type(b), 'Cache type differs: '+path)
        if isinstance(a, torch.Tensor):
            require(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device,
                'Cache tensor schema differs: '+path)
            require(not a.is_floating_point() or (torch.isfinite(a).all() and torch.isfinite(b).all()),
                'Nonfinite cache tensor: '+path)
            if tolerant:
                require(a.is_floating_point() and a.numel() > 0, 'Nonempty floating plans required')
                delta = float((a.double()-b.double()).abs().max())
                passed = delta <= TARGET_ATOL
            else:
                passed = bool(torch.equal(a, b)); delta = None
            rows[path] = dict(passed=passed, exact_required=not tolerant, maximum_absolute_difference=delta)
        elif isinstance(a, dict):
            require(a.keys() == b.keys(), 'Nested cache field population differs: '+path)
            for key in a: visit(a[key], b[key], path+'.'+key, tolerant)
        else:
            rows[path] = dict(passed=a == b, exact_required=True, maximum_absolute_difference=None)
    for key in reference: visit(reference[key], candidate[key], key, key in PLAN_KEYS)
    return dict(passed=all(row['passed'] for row in rows.values()), fixed_target_atol_rad=TARGET_ATOL,
        plan_float_keys=sorted(PLAN_KEYS), fields=rows)


def prepare(root, role, frozen_root, resource_root, output, *, anchor_batch=32, max_seconds=600):
    # Importing the small pure helper on CPU is safe here; the worker is a fresh
    # process and imports the frozen model before its explicit candidate modules.
    from visuotactile.offline_planning import profile
    import torch
    require(type(max_seconds) is int and 60 <= max_seconds <= 900, 'Finite 60..900 second owner bound required')
    config = profile(anchor_batch)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    try:
        receipt = admit(root, role, frozen_root)
        data = read(receipt['paths']['data'])
        ids = receipt['identities']
        require(len(ids) == 2 and data['entries'][ids[0]].get('trajectory_kind') is None
            and data['entries'][ids[1]].get('trajectory_kind') == 'expert_takeover_v1',
            'Gate requires one original and one latest real train takeover')
        saved = torch.load(receipt['paths']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        validate_checkpoint(saved, receipt)
        numeric = torch.load(receipt['paths']['numeric'], map_location='cpu', weights_only=False, mmap=True)
        receipt['projection'] = train_projection(output, data, numeric, receipt)
        del numeric
        receipt.update(epoch=saved['epoch'], updates=saved['updates'], checkpoint_weights='ema')
        residual_stats = Path(root).resolve()/'B/state_data/statistics.pt'
        receipt['residual_statistics'] = dict(path=str(residual_stats), sha256=sha256(residual_stats))
        receipt['original_sources'] = {i:dict(path=data['entries'][i]['path'],
            sha256=checked_hash(data['entries'][i]['path'], data['entries'][i]['source_sha256'])) for i in ids}
        resource_root = Path(resource_root).resolve(strict=True)
        session = read(resource_root/'session.json')
        require(time.time() < session['deadline'], 'Original resource session expired')
        tree = Path(__file__).resolve().parents[1]
        plan = dict(version=VERSION, receipt=receipt, resource_root=str(resource_root),
            session_sha256=sha256(resource_root/'session.json'), original_session_deadline=session['deadline'],
            max_seconds=max_seconds, profile=config, source_tree=str(tree), source_sha256=source_hashes(tree),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),
            prepared_utc=time.time(), preparation_seconds=time.time()-began,
            checkpoint_weights='selected best EMA', optimizer_updates=0, validation_evaluated=False,
            test_evaluated=False, cuda_initialized=torch.cuda.is_initialized(), automatic_retry=False)
        exclusive_json(output/'plan.json', plan)
        return plan
    except BaseException as error:
        exclusive_json(output/'prepare_error.json', dict(error=repr(error), utc=time.time(), automatic_retry=False))
        raise


def verify_sources(plan, tree):
    require(plan['version'] == VERSION, 'Unknown gate identity')
    require(type(plan['max_seconds']) is int and 60 <= plan['max_seconds'] <= 900,
        'Registered finite gate bound differs')
    require(source_hashes(tree) == plan['source_sha256'], 'Candidate source bytes/population changed')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip() == plan['source_commit'],
        'Candidate source commit changed')
    checked_hash(Path(plan['resource_root'])/'session.json', plan['session_sha256'])
    require(read(Path(plan['resource_root'])/'session.json')['deadline'] == plan['original_session_deadline'],
        'Original session deadline changed')


def check_bounds(plan, output, owner):
    if time.time() >= owner['work_deadline']:
        raise TimeoutError('Finite cache gate work deadline reached')
    if (Path(output)/'STOP').exists() or (Path(plan['resource_root'])/'STOP').exists():
        raise InterruptedError('Owned cache gate STOP requested')


def load_candidate(plan, tree):
    # Frozen NativePolicy/vision/position_control and cache readers have already
    # been imported. Only these two explicit modules are the versioned candidate.
    for name, relative in [('visuotactile.offline_planning','visuotactile/offline_planning.py'),
            ('visuotactile._cache_planning_candidate','visuotactile/native_data.py')]:
        path = tree/relative; checked_hash(path, plan['source_sha256'][relative])
        require(name not in sys.modules, 'Candidate module was imported before frozen dependency admission: '+name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec);sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def io_bytes():
    return {key:int(value) for key,value in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def worker(output):
    import torch
    output = Path(output).resolve(); plan = read(output/'plan.json'); owner = read(output/'owner_started.json')
    tree = Path(__file__).resolve().parents[1]
    verify_sources(plan, tree);check_bounds(plan, output, owner)
    require(owner['plan_sha256'] == sha256(output/'plan.json'), 'Owner/worker plan identity differs')
    exclusive_json(output/'worker_started.json', dict(pid=os.getpid(), utc=time.time(), plan_sha256=owner['plan_sha256']))
    began = time.time()
    try:
        receipt = plan['receipt']
        # Recheck every bound source before touching the GPU. Only selected
        # train raw/visual files are opened; whole numeric integrity bytes were
        # read during preparation, but held-out raw/visual files are excluded.
        for relative, expected in receipt['configuration']['source_sha256'].items():
            checked_hash(Path(receipt['frozen_source_root'])/relative, expected)
        for key, expected in receipt['sha256'].items(): checked_hash(receipt['paths'][key], expected)
        for info in receipt['selected_visual_cache'].values():checked_hash(info['path'], info['sha256'])
        for info in receipt['original_sources'].values():checked_hash(info['path'], info['sha256'])
        checked_hash(receipt['residual_statistics']['path'], receipt['residual_statistics']['sha256'])
        projection = receipt['projection']; projected_root = Path(projection['root'])
        checked_hash(projected_root/'data.json', projection['data_sha256'])
        checked_hash(projected_root/'numeric.pt', projection['numeric_sha256'])
        for identity, info in receipt['selected_visual_cache'].items():
            require((projected_root/'perception'/(identity+'.pt')).resolve() == Path(info['path']).resolve(),
                'Train projection visual path changed')
        check_bounds(plan, output, owner)
        import_frozen_trainer(receipt)
        from visuotactile import native_data as frozen
        from visuotactile.models.native_dp import load_policy
        from visuotactile.precision import state_fp32
        from visuotactile.state_resources import lease_for
        candidate = load_candidate(plan, tree)
        sys.modules['visuotactile.offline_planning'].validate_profile(plan['profile'])
        state_fp32()
        real_lease = lease_for(Path(plan['resource_root']), deadline=owner['work_deadline'])
        class BoundedLease:
            def check(self):
                check_bounds(plan, output, owner)
                real_lease.check()
            @contextmanager
            def block(self):
                check_bounds(plan, output, owner)
                with real_lease.block():yield
                check_bounds(plan, output, owner)
        lease = BoundedLease()
        saved = torch.load(receipt['paths']['checkpoint'], map_location='cpu', weights_only=False)
        validate_checkpoint(saved, receipt)
        with lease.block():model = load_policy(saved, 'cuda:0').requires_grad_(False).eval()
        del saved
        records = frozen.visual_records(projected_root, receipt['identities'], 'G00')
        data = read(projected_root/'data.json')
        binding = dict(base_checkpoint_sha256=receipt['sha256']['checkpoint'],
            statistics_sha256=receipt['sha256']['statistics'], data_sha256=projection['data_sha256'],
            residual_statistics_sha256=receipt['residual_statistics']['sha256'])
        rows = []
        for identity in receipt['identities']:
            record = records[identity];entry = data['entries'][identity]
            def measured(function, options, label):
                check_bounds(plan, output, owner);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter();io_start = io_bytes()
                def progress(**value):
                    # CPU receipt only: cache's normal cadence, no per-tick CUDA synchronization.
                    from visuotactile.training_resources import atomic_json
                    atomic_json(output/'progress.json', dict(identity=identity, arm=label, utc=time.time(), **value))
                value = function(model, record, entry, lease, cache_identity=binding, progress=progress,
                    transfer_mode='episode', scheduler_index_mode='host', **options)
                torch.cuda.synchronize();ended = time.perf_counter();io_end = io_bytes()
                return value, dict(seconds=ended-start, io_delta={k:io_end[k]-io_start[k] for k in io_start},
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            reference, serial_time = measured(frozen.cache_residual_episode, {}, 'serial')
            result, candidate_time = measured(candidate.cache_residual_episode,
                dict(offline_planning_profile=plan['profile']), 'candidate')
            compared = compare_caches(reference, result, plan['profile'])
            row = dict(identity=identity, frames=len(record['time']),
                real_student_prefix_ticks=int((~record['supervision_mask']).sum()) if 'supervision_mask' in record else 0,
                all_grid_anchors=list(range(0, len(record['time']), 12)),
                serial=serial_time, candidate=candidate_time, **compared)
            exclusive_json(output/('episode_%02d.json'%(len(rows)+1)), row);rows.append(row)
            del reference,result
        result = dict(complete=True, passed=all(row['passed'] for row in rows), rows=rows,
            profile=plan['profile'], checkpoint_sha256=receipt['sha256']['checkpoint'], epoch=receipt['epoch'],
            wall_seconds=time.time()-began, optimizer_updates=0, validation_evaluated=False, test_evaluated=False,
            scope='Complete train-only nominal cache field parity; no policy success or whole-training speed claim',
            timing_limitations='Per episode serial first then candidate; page-cache warming, allocator/kernel warming and external work confound timing. Process read_bytes is not an independently attributed physical-disk total.',
            binding='Frozen selected model/dependencies and serial cache; explicit candidate native_data/offline_planning only. Same train-only projection, original raw records, checkpoint, and normalizers.',
            sampling='Full recorded 60Hz history and per-tick visual readout; all 5Hz anchors from tick0, including student prefix, seed7091 individual (1,24,29) draws; no plan feedback')
        exclusive_json(output/'result.json', result)
        require(result['passed'], 'Complete cache gate failed; retain serial cache')
        return result
    except BaseException as error:
        exclusive_json(output/'worker_error.json', dict(error=repr(error), utc=time.time(), automatic_retry=False))
        raise


def owner(output, worker_tree):
    # Imports only in owner process: the fresh worker must import its frozen
    # research dependency tree first.
    from tools.tight_batch_owner import bind_worker_group, cleanup_owned_group
    from tools.tight_fresh_eval_recovery import stop_owned
    from visuotactile.training_resources import atomic_json
    output = Path(output).resolve();worker_tree = Path(worker_tree).resolve()
    plan = read(output/'plan.json');verify_sources(plan,worker_tree)
    start = time.time();deadline = min(plan['original_session_deadline'],start+plan['max_seconds'])
    require(deadline-start >= 60, 'Insufficient original session time for finite gate')
    command = [sys.executable,'-m','tools.tight_residual_cache_gate','worker','--output',str(output)]
    started = dict(pid=os.getpid(),utc=start,deadline=deadline,work_deadline=deadline-30,
        plan_sha256=sha256(output/'plan.json'),command=command,worker_tree=str(worker_tree),automatic_retry=False)
    exclusive_json(output/'owner_started.json',started)
    process = binding = None;complete=False;error=None;cleanup=None;handlers={}
    def interrupted(signum,frame):raise SystemExit(128+signum)
    try:
        check_bounds(plan, output, started)
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        with (output/'worker.log').open('x') as stream:
            process=subprocess.Popen(command,cwd=worker_tree,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1'))
            binding=bind_worker_group(process.pid)
            atomic_json(output/'owner_worker_started.json',dict(binding=binding,utc=time.time()))
            while process.poll() is None:
                check_bounds(plan,output,started);time.sleep(.25)
        result = read(output/'result.json') if (output/'result.json').is_file() else {}
        complete = process.returncode == 0 and result.get('complete') is True and result.get('passed') is True
        require(sha256(output/'plan.json')==started['plan_sha256'], 'Registered plan changed during gate')
        require(complete,'Worker failed or cache comparison did not pass')
    except BaseException as caught:error=repr(caught);complete=False
    finally:
        if process is not None:
            try:
                stop_owned(process)
                cleanup=cleanup_owned_group(binding) if binding else dict(live_descendants_exited=False,error='Missing binding')
            except BaseException as caught:cleanup=dict(live_descendants_exited=False,error=repr(caught))
            if not cleanup['live_descendants_exited']:complete=False
            atomic_json(output/'owner_cleanup.json',cleanup)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        atomic_json(output/'owner_finished.json',dict(complete=complete,passed=complete,error=error,
            returncode=process.returncode if process else None,seconds=time.time()-start,deadline=deadline,
            original_session_deadline=plan['original_session_deadline'],automatic_retry=False))
    require(complete,'Bounded cache gate failed; evidence preserved, no automatic retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    prep=sub.add_parser('prepare')
    for name in ('root','frozen-source-root','resource-root','output'):prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--role',choices=('A','B'),required=True)
    prep.add_argument('--anchor-batch',type=int,default=32);prep.add_argument('--max-seconds',type=int,default=600)
    run=sub.add_parser('owner');run.add_argument('--output',type=Path,required=True);run.add_argument('--worker-tree',type=Path,required=True)
    work=sub.add_parser('worker');work.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();values=vars(args).copy();stage=values.pop('stage')
    if stage=='prepare':
        values['frozen_root']=values.pop('frozen_source_root');plan=prepare(**values)
        print(json.dumps(dict(prepared=True,cuda_initialized=plan['cuda_initialized'],identities=plan['receipt']['identities'])),flush=True)
    elif stage=='owner':owner(**values)
    else:worker(**values)


if __name__=='__main__':main()
