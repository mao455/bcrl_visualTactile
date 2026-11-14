"""One bounded preflight for random-time planning and rotating geometry.

Reuse existing source/contract checks, run four real two-trajectory updates, then
choose geometry cost explicitly. No validation/test or autonomous trial is run.
"""
import argparse,copy,json,time,traceback
from pathlib import Path
import torch
from tools.tight_offline_planning_gate import admit,validate_checkpoint,train_projection
from tools.tight_fresh_rollouts import read
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256


def feature_io_check(root,role,identity,profile_path,resource_root,output):
    """One training shard, exact tensor bytes; no model, CUDA or held-out input.

    Eager loading necessarily warms the mapped comparison. Its elapsed time is
    descriptive, never a controlled speed ratio against the lazy open call.
    """
    import resource,subprocess
    from visuotactile.native_data import Episode
    output.mkdir(parents=True,exist_ok=False)
    started=time.time();session=read(resource_root/'session.json')
    deadline=min(started+120,session['deadline']-30)
    def check():
        if time.time()>=deadline or any((p/'STOP').exists() for p in (resource_root,output)):
            raise InterruptedError('Registered feature check deadline or STOP')
    def available_ram():
        rows=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(rows['MemAvailable'].split()[0])*1024
    state=root/role/'state_data';data=read(state/'data.json');cache=read(root/role/'cache_index.json')
    if identity not in data['splits']['train'] or cache.get('complete') is not True:
        raise ValueError('A completed frozen cache and an explicit training identity are required')
    entry=cache['entries'][identity];path=(state/'perception'/(identity+'.pt')).resolve(strict=True)
    if path!=Path(entry['path']).resolve(strict=True) or entry['source_sha256']!=data['entries'][identity]['source_sha256']:
        raise ValueError('Training source and frozen cache binding differ')
    profile=read(profile_path);before=path.stat();ram=available_ram()
    if ram<10*1024**3+2*before.st_size+1024**3:
        raise RuntimeError('Insufficient RAM for two current copies plus original reserve')
    if torch.cuda.is_initialized():raise ValueError('This check must start without CUDA')
    torch.set_num_threads(2)
    launch=dict(version='unified_frozen_feature_io_check_v1',started=started,deadline=deadline,
        original_deadline=session['deadline'],source_root=str(root),role=role,identity=identity,
        worker_tree=str(Path.cwd()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cache_index_sha256=sha256(root/role/'cache_index.json'),data_sha256=sha256(state/'data.json'),
        profile_sha256=sha256(profile_path),profile=profile,feature_path=str(path),
        feature_declared_sha256=entry['sha256'],feature_bytes=before.st_size,available_ram_before=ram,
        optimizer_updates=0,models_loaded=0,validation_used=False,test_used=False)
    atomic_json(output/'started.json',launch)
    try:
        check();began=time.perf_counter()
        eager=Episode(state,identity,'I-HistoryDP',frozen_visual_io=profile)
        eager_load_s=time.perf_counter()-began
        check();began=time.perf_counter();mapped=Episode(state,identity,'I-HistoryDP')
        mmap_open_s=time.perf_counter()-began
        rows=[];began=time.perf_counter()
        for section in ('record','perception'):
            left,right=getattr(mapped,section),getattr(eager,section)
            if left.keys()!=right.keys():raise ValueError('Frozen feature fields differ')
            for key in left:
                a,b=left[key],right[key]
                if a.dtype!=b.dtype or a.shape!=b.shape or a.stride()!=b.stride():
                    raise ValueError('Tensor layout differs: '+section+'/'+key)
                # Bounded copies compare actual bits, including signed zero;
                # no extra full-trajectory materialization or learned cache.
                for offset in range(0,len(a),32):
                    check()
                    if not torch.equal(a[offset:offset+32].contiguous().view(torch.uint8),
                                       b[offset:offset+32].contiguous().view(torch.uint8)):
                        raise ValueError('Tensor bytes differ: '+section+'/'+key)
                rows.append(dict(field=section+'/'+key,dtype=str(a.dtype),shape=list(a.shape),
                    strides=list(a.stride()),bytes=a.numel()*a.element_size(),all_bytes_equal=True))
        after=path.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Frozen feature file changed during comparison')
        if torch.cuda.is_initialized():raise ValueError('CPU-only check initialized CUDA')
        atomic_json(output/'result.json',dict(complete=True,**launch,ended=time.time(),seconds=time.time()-started,
            eager_load_seconds=eager_load_s,lazy_mmap_open_seconds=mmap_open_s,
            byte_comparison_seconds=time.perf_counter()-began,tensors=rows,
            tensor_bytes=sum(v['bytes'] for v in rows),cuda_initialized=False,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            available_ram_after=available_ram(),eager_receipt=getattr(eager,'frozen_visual_io_receipt',None),
            timing_scope='One sequential read under concurrent formal training. Lazy open does not read all pages; comparison is warmed. No cold-cache or whole-cycle speed claim.',
            payload_rehashed=False))
        del eager,mapped
    except BaseException as error:
        atomic_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=time.time()))
        raise


def run(root,frozen_tree,resource_root,output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.time();deadline=min(started+570,read(resource_root/'session.json')['deadline']-30)
    rows=[]
    def check():
        if time.time()>=deadline or (resource_root/'STOP').exists() or (output/'STOP').exists():
            raise InterruptedError('Registered deadline or STOP')
    atomic_json(output/'started.json',dict(version='unified_random_visit_training_check_v1',started=started,deadline=deadline,
        original_deadline=read(resource_root/'session.json')['deadline'],maximum_updates=4,source_root=str(root),
        worker_tree=str(Path.cwd()),source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        diagnostic_only=True,validation_used=False,test_used=False))
    try:
        from tools import tight_fresh_train as trainer
        from visuotactile.trajectory_training import execute_update
        from visuotactile.fresh_visit_sampling import VERSION,visit
        from visuotactile.anchor_sampling import sampling_profile
        from visuotactile.geometry_sampling import profile as geometry_profile
        from visuotactile.models.vendor.ema_model import EMAModel
        from visuotactile.native_data import Episode
        from visuotactile.target_windows import with_target_window
        from visuotactile.precision import state_fp32
        from visuotactile.state_resources import lease_for
        state_fp32();torch.cuda.set_per_process_memory_fraction(.45)
        lease=lease_for(resource_root,deadline=deadline)
        for role in ('A','B'):
            receipt=admit(root,role,frozen_tree,None)
            saved=torch.load(receipt['paths']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
            validate_checkpoint(saved,receipt)
            numeric=torch.load(receipt['paths']['numeric'],map_location='cpu',weights_only=False,mmap=True)
            projection_root=output/role;projection_root.mkdir()
            projection=train_projection(projection_root,read(receipt['paths']['data']),numeric,receipt);del numeric
            atomic_json(projection_root/'inputs.json',dict(receipt=receipt,projection=projection))
            identities=receipt['identities'];assert len(identities)==2
            statistics=torch.load(receipt['paths']['statistics'],map_location='cpu',weights_only=False)
            with lease.block():
                records=with_target_window(trainer.visual_records(projection['root'],identities,'G00',device='cuda:0'),receipt['target_window'])
                io_start=time.perf_counter();episodes=[Episode(Path(projection['root']),i,'I-HistoryDP') for i in identities]
                episode_open_s=time.perf_counter()-io_start
                variants=[('original_two',None),('random_half',None),('random_cap32',32)] if role=='A' else [('random_starts',32)]
                for name,cap in variants:
                    check();model=trainer.new_model(role,statistics).train();model.load_state_dict(saved['ema'],strict=True)
                    averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
                    config=receipt['configuration'];opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],betas=(.95,.999),weight_decay=config['weight_decay'])
                    new=name!='original_two'
                    profile=dict(version=VERSION,anchors=sampling_profile(),geometry=geometry_profile(cap)) if new else None
                    objective,geometry,proof=trainer.fresh_objective(root,role,read(receipt['paths']['data']),records,
                        **(dict(geometry_sampling_profile=profile['geometry']) if new and role=='A' else {}))
                    options={}
                    if new:
                        selections=[visit(records[i],profile,identity=i,epoch=1,visit_index=j+1) for j,i in enumerate(identities)]
                        options=dict(selected_anchors=[x[0] for x in selections],anchor_receipts=[x[1] for x in selections],
                            geometry_generators=[x[2] for x in selections],geometry_sampling_profile=profile['geometry'])
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
                    result=execute_update(model,opt,ema,objective,identities,[records[i] for i in identities],episodes,
                        torch.Generator(device='cuda').manual_seed(990031),check=check,diagnostic_gradients=role=='A',**options)
                    torch.cuda.synchronize()
                    row=dict(role=role,variant=name,profile=profile,seconds=time.perf_counter()-began,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),episode_open_s_shared=episode_open_s,
                        result=result,geometry=geometry,source_checkpoint_sha256=sha256(receipt['paths']['checkpoint']))
                    rows.append(row);atomic_json(output/'progress.json',dict(completed_updates=len(rows),rows=rows))
                    del model,averaged,ema,opt,objective;torch.cuda.empty_cache()
                del records,episodes
        check();atomic_json(output/'result.json',dict(complete=True,updates=len(rows),rows=rows,seconds=time.time()-started,
            timing_scope='Full update including optional gradient influence reverse passes, excludes shared episode open; sequential warm variants; no whole-cycle speed claim'))
    except BaseException as error:
        atomic_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_updates=len(rows),rows=rows,utc=time.time()))
        raise


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('root','frozen-tree','resource-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--feature-io-profile',type=Path,help='CPU-only one-shard loading check instead of the four GPU batches')
    p.add_argument('--feature-identity',help='Explicit training identity; required for the CPU feature check')
    p.add_argument('--role',choices=('A','B'),default='A')
    a=p.parse_args()
    if bool(a.feature_io_profile)!=bool(a.feature_identity):p.error('Feature profile and training identity must be provided together')
    if a.feature_io_profile:
        feature_io_check(a.root.resolve(),a.role,a.feature_identity,a.feature_io_profile.resolve(),a.resource_root.resolve(),a.output.resolve())
    else:run(a.root.resolve(),a.frozen_tree.resolve(),a.resource_root.resolve(),a.output.resolve())
if __name__=='__main__':main()
