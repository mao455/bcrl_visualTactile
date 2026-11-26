"""Bounded real-input throughput probes; never writes formal training outputs."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import torch

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


class Samples:
    """Whole-card samples: competing production work is explicitly retained."""
    def __init__(self):
        self.rows=[];self.done=threading.Event()

    def __enter__(self):
        def sample():
            while not self.done.is_set():
                result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                gpu,used,free=map(float,result.stdout.splitlines()[0].split(','))
                self.rows.append(dict(utc=time.time(),gpu_percent=gpu,memory_used_mib=used,memory_free_mib=free))
                self.done.wait(.5)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self

    def __exit__(self,*_):
        self.done.set();self.thread.join()

    def summary(self):
        return dict(samples=len(self.rows),gpu_mean_percent=statistics.mean(r['gpu_percent'] for r in self.rows),
            gpu_peak_percent=max(r['gpu_percent'] for r in self.rows),
            card_memory_peak_mib=max(r['memory_used_mib'] for r in self.rows),
            card_memory_free_min_mib=min(r['memory_free_mib'] for r in self.rows),
            attribution='whole card including unchanged concurrent production worker')


def resources():
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=float(gpu.stdout.splitlines()[0]);mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if free < 6*1024 or mem['MemAvailable'] < 14*1024**2:
        raise RuntimeError('Probe requires 6GiB free VRAM and 14GiB available RAM')
    return dict(free_vram_mib=free,available_ram_gib=mem['MemAvailable']/1024**2,
        active_gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip())


def dp_fixture(root, checkpoint, device):
    import h5py
    from visuotactile.native_data import visual_records
    from visuotactile.state_history import PersistentImage
    from visuotactile.state_image_loss import GeometryObjective
    from visuotactile.anchor_sampling import AnchorSampler
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['configuration'];session=read(root/'session.json');data_root=root.parent/'state_data'
    if not data_root.exists():data_root=root.parent.parent/'state_data'
    data=read(data_root/'data.json')
    model=PersistentImage(saved['group'],saved['contract']);model.load_state_dict(saved['model']);model.to(device)
    ids=saved['train_ids'];records=visual_records(data_root,ids,cfg['modality'],device)
    with h5py.File(data['entries'][ids[0]]['path']) as stream:metadata=json.loads(stream.attrs['metadata_json'])
    objective=GeometryObjective(metadata,mode=cfg['loss_id'],commit=12);objective.weights=dict(cfg['loss_weights']);objective.gain=1.
    geometry=cfg.get('hand_geometry')
    if geometry:
        gate=read(geometry['acceptance']);live=read(geometry['live'])
        objective.enable_hand_geometry(live['inventory'],gate,loss_kind=geometry['loss_kind'],**geometry['options'])
        objective.weights=dict(cfg['loss_weights'])
        objective.loss_context=torch.load(data_root/'loss_context.pt',map_location='cpu',weights_only=False)
    objective.supervision_records=records
    generator=torch.Generator();generator.set_state(saved['schedule_rng'])
    sampler=AnchorSampler({i:len(records[i]['time']) for i in ids},generator,cfg['effective_batch'],records=records,
        strata=cfg.get('sampling_strata'),weights=cfg.get('sampling_weights'),recovery_prefix=cfg.get('recovery_prefix'))
    sampler.load_state_dict(saved['anchor_sampler']);selected=sampler.draw()
    return saved,cfg,model,records,objective,selected


def dp_benchmark(args):
    from visuotactile.anchor_continue import diffusion_draws,geometry_selection,projection_batch
    from visuotactile.anchor_sampling import anchor_batch,mask_totals,weighted_parts
    from visuotactile.models.vendor.ema_model import EMAModel
    root=Path(args.root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint or root/'last.pt');frozen=out/'source.pt'
    if not frozen.exists():
        import shutil
        shutil.copyfile(checkpoint,frozen)
    source_hash=sha256(frozen);device='cuda:0';initial_resources=resources()
    saved,cfg,model,records,objective,selected=dp_fixture(root,frozen,device)
    totals=mask_totals({i:len(r['time']) for i,r in records.items()},selected,12,records=records)
    direct=objective.hand_geometry is not None
    selected_geometry,geo=(geometry_selection(objective,selected,{i:len(r['time']) for i,r in records.items()},
        cfg['seed']+1000003*(saved['updates']+1)) if direct else (None,{}))
    initial={k:v.clone() for k,v in model.state_dict().items()}
    averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr_floor'],betas=(.95,.999),weight_decay=1e-6)
    noise=torch.Generator(device=device)
    rows=[];reference=None
    variants=[(4,2,0),(8,2,0),(16,2,0),(4,4,0),(8,4,0),(16,4,0)]
    if args.variants:variants=[tuple(map(int,v.split(':'))) for v in args.variants.split(',')]
    def run(micro,threads,frame_cache,repetition,warmup=False):
        nonlocal reference
        resources();torch.set_num_threads(threads);model.load_state_dict(initial);model.train()
        model._episodes.clear();model.raw_frame_device_cache=bool(frame_cache)
        opt.load_state_dict(copy.deepcopy(saved['optimizer']));averaged.load_state_dict(saved['ema']);ema.optimization_step=saved['ema_optimization_step']
        noise.set_state(saved['noise_rng']);opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter();cpu_began=time.process_time()
        timings=dict(anchor_s=0.,objective_s=0.,backward_s=0.,optimizer_ema_s=0.);loss_sum=0.;parts_sum={};geometry_s=0.
        with Samples() as samples:
            draws=diffusion_draws(len(selected),24,noise,device)
            for start in range(0,len(selected),micro):
                chunk=selected[start:start+micro];stamp=time.perf_counter()
                c,y,m=anchor_batch(model,records,chunk);torch.cuda.synchronize();timings['anchor_s']+=time.perf_counter()-stamp;stamp=time.perf_counter()
                context=projection_batch(objective,chunk,device,selected_mask=selected_geometry[start:start+micro] if direct else None,
                    update_selected_count=geo.get('selected_count'),active=True)
                _,parts,_=objective.forward(model,c,y,m,noise,geometry_context=context,
                    diffusion_inputs=tuple(x[start:start+micro] for x in draws))
                loss,scaled=weighted_parts(parts,m,totals,objective.weights,12,global_hand_geometry=direct)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real-input probe loss')
                geometry_s+=(objective.last_geometry_metrics or {}).get('elapsed_s',0.)
                torch.cuda.synchronize();timings['objective_s']+=time.perf_counter()-stamp;stamp=time.perf_counter()
                loss.backward();torch.cuda.synchronize();timings['backward_s']+=time.perf_counter()-stamp
                loss_sum+=float(loss.detach())
                for key,value in scaled.items():parts_sum[key]=parts_sum.get(key,0.)+float(value.detach())
            gradient=torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);stamp=time.perf_counter()
            opt.step();ema.step(model);torch.cuda.synchronize();timings['optimizer_ema_s']+=time.perf_counter()-stamp
            elapsed=time.perf_counter()-began;cpu=time.process_time()-cpu_began
        params=torch.cat([p.detach().flatten() for p in model.parameters()]);grad=gradient.cpu();param=params.cpu()
        if reference is None:reference=(loss_sum,grad,param)
        row=dict(microbatch=micro,threads=threads,raw_frame_device_cache=bool(frame_cache),repetition=repetition,warmup=warmup,seconds_per_update=elapsed,
            anchors_per_second=len(selected)/elapsed,valid_target_ticks=totals[0],loss=loss_sum,components=parts_sum,
            gradient_norm=float(norm),geometry_seconds=geometry_s,geometry_selected_count=geo.get('selected_count',0),
            loss_abs_difference=abs(loss_sum-reference[0]),gradient_max_abs_difference=float((grad-reference[1]).abs().max()),
            gradient_relative_l2=float((grad-reference[1]).norm()/reference[1].norm().clamp_min(1e-30)),
            updated_parameter_max_abs_difference=float((param-reference[2]).abs().max()),
            allocated_peak_mib=torch.cuda.max_memory_allocated()/1024**2,reserved_peak_mib=torch.cuda.max_memory_reserved()/1024**2,
            cpu_seconds=cpu,cpu_over_wall=cpu/elapsed,timings=timings,resources=samples.summary(),
            generator_state_sha256=__import__('hashlib').sha256(noise.get_state().numpy().tobytes()).hexdigest())
        rows.append(row);atomic_json(out/'dp_progress.json',dict(last=row,completed=len(rows)))
        del gradient,params,grad,param
    run(4,2,0,-1,True)
    for repetition in range(args.repeats):
        for micro,threads,frame_cache in variants if repetition%2==0 else list(reversed(variants)):run(micro,threads,frame_cache,repetition)
    summaries=[]
    for micro,threads,frame_cache in variants:
        values=[r for r in rows if not r['warmup'] and (r['microbatch'],r['threads'],r['raw_frame_device_cache'])==(micro,threads,bool(frame_cache))]
        summaries.append(dict(microbatch=micro,threads=threads,raw_frame_device_cache=bool(frame_cache),updates=len(values),
            median_seconds_per_update=statistics.median(r['seconds_per_update'] for r in values),
            mean_seconds_per_update=statistics.mean(r['seconds_per_update'] for r in values)))
    reference_time=next(r['median_seconds_per_update'] for r in summaries if (r['microbatch'],r['threads'])==(4,2))
    for row in summaries:row['speedup_vs_micro4_threads2']=reference_time/row['median_seconds_per_update']
    report=dict(version='actual_dagger_dp_microbatch_v1',formal_optimizer_updates=0,disposable=True,
        source_checkpoint=str(checkpoint),source_sha256=source_hash,source_update=saved['updates'],
        source_configuration=cfg,effective_batch=len(selected),selected=selected,valid_target_ticks=totals[0],
        variants=summaries,measurements=rows,initial_resources=initial_resources,
        affinity=sorted(os.sched_getaffinity(0)),tf32=False,amp=False,compile=False,
        notes=['One real optimizer update per repeated trial from identical model/optimizer/EMA/RNG.',
            'Unchanged full causal prefixes and geometry: two selected configurations per complete optimizer update.',
            'Legacy micro4 interleaved noise/timestep draws are concatenated then sliced; final RNG identical.',
            'Synchronized component timings intentionally instrument every phase; short shared-card probe is not full-round speedup.'])
    atomic_json(out/'dp_benchmark.json',report)


def residual_benchmark(args):
    from visuotactile.models.native_dp import TargetResidual
    from visuotactile.native_data import residual_batch,residual_supervision_mask
    from visuotactile.anchor_continue import residual_aux
    from visuotactile.anchor_sampling import recovery_prefix_rows,residual_recovery_weights
    def residual_episode_device(records,device):
        # Deliberately benchmark-only: measured gain was too small for deployment.
        def move(value):
            if isinstance(value,torch.Tensor):return value.to(device)
            if isinstance(value,dict):return {key:move(item) for key,item in value.items()}
            return value
        return [move(record) for record in records]
    root=Path(args.root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint or root/'training/last.pt');saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['configuration'];data=read(root/'state_data/data.json');stats=torch.load(cfg['residual_statistics_path'],map_location='cpu',weights_only=False)
    model=TargetResidual(stats,saved['model']['scales'],execution_adapter=cfg.get('execution_adapter')).cuda()
    model.load_state_dict(saved['model']);model.train();opt=torch.optim.AdamW(model.parameters(),lr=cfg.get('learning_rate',1e-3),weight_decay=cfg.get('weight_decay',1e-4))
    strata=data.get('sampling_strata');ids=[]
    if strata:
        for name in ('current_correction','historical_correction','core'):ids+=strata[name][:1]
    ids += [i for i in data['splits']['train'] if i not in ids][:cfg['microbatch']-len(ids)]
    assert len(ids)==cfg['microbatch']
    records=[torch.load(root/'nominal'/(i+'.pt'),map_location='cpu',weights_only=False) for i in ids]
    prefixes=[torch.zeros(len(r['robot']),dtype=torch.bool) if i in (strata or {}).get('core',[]) else recovery_prefix_rows(r,cfg['recovery_prefix']['ticks']) for i,r in zip(ids,records)]
    rows=[];reference=None
    variants=[(2,False),(2,True),(4,False),(4,True)]
    for repetition in range(-1,args.repeats):
        current=variants[:1] if repetition==-1 else variants if repetition%2==0 else list(reversed(variants))
        for threads,resident in current:
            resources();torch.set_num_threads(threads);model.load_state_dict(saved['model']);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
            state=None;updates=ticks=0;losses=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter();cpu=time.process_time()
            with Samples() as samples:
                selected=residual_episode_device(records,'cuda:0') if resident else records
                for start in range(0,max(len(r['robot']) for r in records),32):
                    stop=min(start+32,max(len(r['robot']) for r in records));inputs=residual_batch(selected,start,stop,'cuda:0')
                    raw,state=model(inputs,state);supervised=residual_supervision_mask(records,start,stop,'cuda:0')&inputs['valid']
                    gate=residual_aux(selected,start,stop,'gate','cuda:0')&inputs['valid']
                    action,delta=model.compose(residual_aux(selected,start,stop,'base','cuda:0'),raw,gate,
                        torch.stack([r['limits'] for r in selected]).to('cuda:0')[:,None],previous=inputs['robot'][...,58:87])
                    mask,_=residual_recovery_weights(prefixes,start,stop,supervised,cfg['recovery_prefix']['backend_weight'])
                    loss=model.objective(action,residual_aux(selected,start,stop,'target','cuda:0'),delta,mask)
                    if not torch.isfinite(loss):raise FloatingPointError('Residual performance probe nonfinite loss')
                    if supervised.any():
                        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(norm):raise FloatingPointError('Residual performance probe nonfinite gradient')
                        opt.step();updates+=1;ticks+=int(supervised.sum());losses.append(float(loss.detach()))
                    def detach(value):return {k:detach(v) for k,v in value.items()} if isinstance(value,dict) else value.detach()
                    state=detach(state)
                torch.cuda.synchronize();elapsed=time.perf_counter()-began;cpu_elapsed=time.process_time()-cpu
            params=torch.cat([p.detach().cpu().flatten() for p in model.parameters()]);values=torch.tensor(losses)
            if reference is None:reference=(params.clone(),values.clone())
            row=dict(threads=threads,episode_device_cache=resident,repetition=repetition,warmup=repetition<0,wall_s=elapsed,
                optimizer_updates=updates,supervised_ticks=ticks,seconds_per_update=elapsed/updates,
                ticks_per_second=ticks/elapsed,loss_max_abs_difference=float((values-reference[1]).abs().max()),
                updated_parameter_max_abs_difference=float((params-reference[0]).abs().max()),
                allocated_peak_mib=torch.cuda.max_memory_allocated()/1024**2,reserved_peak_mib=torch.cuda.max_memory_reserved()/1024**2,
                cpu_seconds=cpu_elapsed,cpu_over_wall=cpu_elapsed/elapsed,resources=samples.summary())
            rows.append(row);atomic_json(out/'residual_progress.json',dict(last=row))
    summaries=[]
    for threads,resident in variants:
        values=[r for r in rows if not r['warmup'] and r['threads']==threads and r['episode_device_cache']==resident]
        summaries.append(dict(threads=threads,episode_device_cache=resident,
            median_wall_s=statistics.median(r['wall_s'] for r in values),mean_wall_s=statistics.mean(r['wall_s'] for r in values)))
    baseline=summaries[0]['median_wall_s']
    for row in summaries:row['speedup']=baseline/row['median_wall_s']
    atomic_json(out/'residual_benchmark.json',dict(version='actual_dagger_residual_episode_cache_v1',
        formal_optimizer_updates=0,disposable=True,source_checkpoint=str(checkpoint),source_sha256=sha256(checkpoint),
        actual_complete_episode_ids=ids,episode_lengths=[len(r['robot']) for r in records],effective_batch=cfg['microbatch'],
        measurements=rows,variants=summaries,affinity=sorted(os.sched_getaffinity(0)),
        scope='Same complete real four-episode group,32tick detach and actual supervised updates; model/RNG/optimizer reset each trial; includes group upload but excludes checkpoint/log IO.',
        amp=False,tf32=False,compile=False))


def paired_worker(args):
    """Two independent real A/B learners, controlled only by disposable files."""
    from visuotactile.anchor_continue import diffusion_draws,geometry_selection,projection_batch
    from visuotactile.anchor_sampling import anchor_batch,mask_totals,weighted_parts
    from visuotactile.models.vendor.ema_model import EMAModel
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);root=Path(args.root).resolve()
    torch.set_num_threads(2);saved,cfg,model,records,obj,selected=dp_fixture(root,Path(args.checkpoint),'cuda:0')
    model.raw_frame_device_cache=True;model.train();averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr_floor'],betas=(.95,.999),weight_decay=1e-6)
    noise=torch.Generator(device='cuda:0');direct=obj.hand_geometry is not None
    totals=mask_totals({i:len(r['time']) for i,r in records.items()},selected,12,records=records)
    geometry,report=geometry_selection(obj,selected,{i:len(r['time']) for i,r in records.items()},cfg['seed']+1000003*(saved['updates']+1)) if direct else (None,{})
    def reset():
        model.load_state_dict(saved['model']);averaged.load_state_dict(saved['ema']);ema.optimization_step=saved['ema_optimization_step']
        opt.load_state_dict(copy.deepcopy(saved['optimizer']));noise.set_state(saved['noise_rng']);model._episodes.clear()
    def update():
        opt.zero_grad(set_to_none=True);draws=diffusion_draws(len(selected),24,noise,'cuda:0');loss_sum=0.
        for start in range(0,len(selected),16):
            chunk=selected[start:start+16];c,y,m=anchor_batch(model,records,chunk)
            context=projection_batch(obj,chunk,'cuda:0',selected_mask=geometry[start:start+16] if direct else None,
                update_selected_count=report.get('selected_count'),active=True)
            _,parts,_=obj.forward(model,c,y,m,noise,geometry_context=context,diffusion_inputs=tuple(x[start:start+16] for x in draws))
            loss,_=weighted_parts(parts,m,totals,obj.weights,12,global_hand_geometry=direct)
            if not torch.isfinite(loss):raise FloatingPointError('Pair probe nonfinite loss')
            loss.backward();loss_sum+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Pair probe nonfinite gradient')
        opt.step();ema.step(model);return loss_sum
    reset();update();torch.cuda.synchronize()
    for cycle in ('sequential_0','parallel_0','parallel_1','sequential_1'):
        reset();atomic_json(out/(cycle+'.ready.json'),dict(pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0))))
        deadline=time.time()+600
        while not (out/(cycle+'.start')).exists():
            if time.time()>deadline:raise TimeoutError('Bounded pair probe barrier')
            time.sleep(.05)
        torch.cuda.synchronize();began=time.perf_counter();losses=[]
        for _ in range(2):losses.append(update())
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        atomic_json(out/(cycle+'.done.json'),dict(wall_s=elapsed,updates=2,selected=selected,
            valid_target_ticks_per_update=totals[0],losses=losses,source_sha256=sha256(args.checkpoint),source_updates=saved['updates'],
            gradient_geometry=direct,allocated_peak_mib=torch.cuda.max_memory_allocated()/1024**2))


def paired_benchmark(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);resources();children=[]
    # These are actual current A and previous-round B checkpoint/data identities,
    # used only as a throughput comparison. No new formal pair is constructed.
    jobs=[('A',args.root,args.checkpoint,'16,17'),('B',args.other_root,args.other_checkpoint,'18,19')]
    def wait_file(path,child):
        deadline=time.time()+600
        while not path.exists():
            if child.poll() is not None:raise RuntimeError('Pair worker exited: '+str(path))
            if time.time()>deadline:raise TimeoutError('Bounded pair benchmark')
            time.sleep(.1)
    try:
        for name,root,checkpoint,cpus in jobs:
            directory=out/name;directory.mkdir(exist_ok=False)
            log=(directory/'worker.log').open('w')
            command=['taskset','-c',cpus,__import__('sys').executable,'-m','tools.tight_performance','paired_worker',
                '--root',str(root),'--checkpoint',str(checkpoint),'--output',str(directory)]
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);children.append((name,directory,child,log))
            wait_file(directory/'sequential_0.ready.json',child)
        rows=[]
        for cycle in ('sequential_0','parallel_0','parallel_1','sequential_1'):
            for name,directory,child,log in children:wait_file(directory/(cycle+'.ready.json'),child)
            resources();began=time.perf_counter()
            with Samples() as samples:
                if cycle.startswith('parallel'):
                    for _,directory,_,_ in children:(directory/(cycle+'.start')).touch()
                    for _,directory,child,_ in children:wait_file(directory/(cycle+'.done.json'),child)
                else:
                    for _,directory,child,_ in children:
                        (directory/(cycle+'.start')).touch();wait_file(directory/(cycle+'.done.json'),child)
                elapsed=time.perf_counter()-began
            rows.append(dict(cycle=cycle,wall_s=elapsed,total_optimizer_updates=4,
                per_group={name:read(directory/(cycle+'.done.json')) for name,directory,_,_ in children},resources=samples.summary()))
            atomic_json(out/'paired_progress.json',dict(measurements=rows))
        sequential=statistics.mean(r['wall_s'] for r in rows if r['cycle'].startswith('sequential'))
        parallel=statistics.mean(r['wall_s'] for r in rows if r['cycle'].startswith('parallel'))
        atomic_json(out/'paired_benchmark.json',dict(version='actual_ab_concurrency_v1',disposable=True,formal_optimizer_updates=0,
            measurements=rows,sequential_mean_wall_s=sequential,parallel_mean_wall_s=parallel,speedup=sequential/parallel,
            scope='Same fixed two updates each actual A/B learner per trial; models, optimizer, EMA, sampler labels and noise independently restored.',
            limitation='Real A current-round and B previous-round datasets; tests two-model compute contention, not a matched scientific comparison or full DP/cache/rear/eval pipeline.',
            microbatch=16,effective_batch=64,threads_each=2,raw_frame_device_cache=True))
    finally:
        for _,_,child,log in children:
            if child.poll() is None:child.terminate()
            child.wait(timeout=30);log.close()



def benchmark_cache(root,checkpoint,output,repeats=2,modes=('tick','episode','episode_host')):
    """Compare full real Core/recovery caches with current-weight causal replay."""
    import argparse,fcntl,json,os,subprocess,threading,time
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    import psutil
    import torch
    from visuotactile.compact import sha256
    from visuotactile.native_data import cache_residual_episode,visual_records
    from visuotactile.models.native_dp import load_policy
    from visuotactile.precision import state_fp32

    def compare(left,right,prefix=''):
        rows=[]
        if isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:rows.extend(compare(left[key],right[key],prefix+'.'+key))
        elif isinstance(left,torch.Tensor):
            assert left.shape==right.shape and left.dtype==right.dtype
            exact=torch.equal(left,right)
            error=float((left.double()-right.double()).abs().max()) if left.numel() else 0.
            rows.append(dict(field=prefix,bitwise_equal=exact,maximum_absolute_error=error,shape=list(left.shape),dtype=str(left.dtype)))
        else:rows.append(dict(field=prefix,bitwise_equal=left==right))
        return rows

    def gpu():
        text=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        return dict(gpu_utilization=float(text[0]),gpu_used_mib=float(text[1]),gpu_free_mib=float(text[2]),ram_available_gib=psutil.virtual_memory().available/2**30)

    a=SimpleNamespace(root=root,checkpoint=checkpoint,output=output,repeats=repeats,modes=list(modes))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);root=Path(a.root);data=root/'state_data';info=json.loads((data/'data.json').read_text());ids=info['splits']['train']
    correction=[i for i in ids if i.startswith('tight_dagger_matched_') and info['entries'][i].get('trajectory_kind')=='expert_takeover_v1']
    selected=[ids[0],max(correction,key=lambda i:info['entries'][i]['length'])]
    ck=Path(a.checkpoint);checkpoint_sha=sha256(ck);saved=torch.load(ck,map_location='cpu',weights_only=False)
    assert checkpoint_sha==sha256(ck),'Mutable checkpoint changed'
    state_fp32();torch.set_num_threads(2)
    records=visual_records(data,selected,saved['contract']['modality'])
    command_statistics=root.parent/'command_statistics/statistics.pt'
    if not command_statistics.is_file():raise FileNotFoundError('Benchmark requires actual paired command statistics: '+str(command_statistics))
    binding=dict(base_checkpoint_sha256=checkpoint_sha,statistics_sha256=sha256(data/'statistics.pt'),data_sha256=sha256(data/'data.json'),residual_statistics_sha256=sha256(command_statistics))
    # Pin the actual frontend and command normalizers even though nominal
    # cache generation does not itself apply residual command normalization.
    report=dict(checkpoint=str(ck),checkpoint_sha256=checkpoint_sha,data_root=str(data),data_sha256=binding['data_sha256'],identities=selected,
        threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),source_sha256=sha256(Path('visuotactile/native_data.py')),policy_source_sha256=sha256(Path('visuotactile/models/native_dp.py')),rows=[],parity=[],
        scope='same frozen actual EMA model, two complete real train episodes, FP32 DDIM16 H24 commit12; no optimizer updates; legacy/episode interleaved; production worker may compete',
        source_normalization='actual paired frontend and expert-command statistics; no residual model or training',command_statistics_path=str(command_statistics),command_statistics_sha256=binding['residual_statistics_sha256'],started=time.time())
    proc=psutil.Process();lock=open('/tmp/tight_dagger_perf_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        before=gpu();assert before['gpu_free_mib']>=6144 and before['ram_available_gib']>=14,before
        model=load_policy(saved,'cuda:0').requires_grad_(False).eval();del saved
        report['baseline_resources']=before
        report['competing_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
        lease=SimpleNamespace(block=nullcontext)
        actual_observe=model.observe;hidden=[]
        def observe(*args,**kwargs):
            memory=actual_observe(*args,**kwargs)
            hidden.append((memory.native,memory.visual))
            return memory
        model.observe=observe
        # The first full Core episode warms model kernels; no result or timing is
        # admitted. Every measured call still starts fresh recurrent memory/RNG.
        cache_residual_episode(model,records[selected[0]],info['entries'][selected[0]],lease,cache_identity=binding,transfer_mode='tick')
        torch.cuda.synchronize();hidden.clear();print('CACHE_WARMED',flush=True)
        for repeat in range(a.repeats):
            for index,identity in enumerate(selected):
                outputs={};order=a.modes if (repeat+index)%2==0 else list(reversed(a.modes))
                for mode in order:
                    samples=[];stop=threading.Event()
                    def collect():
                        while not stop.is_set():
                            try:samples.append(dict(utc=time.time(),**gpu(),cpu_percent=proc.cpu_percent()))
                            except Exception as e:samples.append(dict(error=str(e)))
                            stop.wait(.5)
                    thread=threading.Thread(target=collect,daemon=True);thread.start()
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();cpu=proc.cpu_times()
                    hidden.clear()
                    outputs[mode]=cache_residual_episode(model,records[identity],info['entries'][identity],lease,cache_identity=binding,transfer_mode='episode' if mode=='episode_host' else mode,scheduler_index_mode='host' if mode=='episode_host' else 'legacy')
                    torch.cuda.synchronize();seconds=time.perf_counter()-start;after=proc.cpu_times();stop.set();thread.join()
                    valid=[x for x in samples if 'gpu_utilization'in x]
                    assert all(x['gpu_free_mib']>=6144 and x['ram_available_gib']>=14 for x in valid),'Reserve fell below bound'
                    row=dict(repeat=repeat,identity=identity,mode=mode,ticks=len(records[identity]['time']),seconds=seconds,ticks_per_second=len(records[identity]['time'])/seconds,
                        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        cpu_seconds=after.user+after.system-cpu.user-cpu.system,gpu_utilization_mean=sum(x['gpu_utilization']for x in valid)/len(valid),gpu_utilization_peak=max(x['gpu_utilization']for x in valid),
                        whole_card_vram_peak_mib=max(x['gpu_used_mib']for x in valid),resource_samples=samples)
                    outputs[mode]['_benchmark_native_hidden']=torch.stack([h[0]for h in hidden]).cpu()
                    outputs[mode]['_benchmark_visual_hidden']=torch.stack([h[1]for h in hidden]).cpu()
                    report['rows'].append(row);(out/'cache_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in row.items()if k!='resource_samples'}),flush=True)
                for mode in a.modes:
                    if mode=='tick':continue
                    parity=compare(outputs['tick'],outputs[mode]);report['parity'].append(dict(repeat=repeat,identity=identity,mode=mode,fields=parity,bitwise_equal=all(r['bitwise_equal']for r in parity)))
                    assert all(r['bitwise_equal']for r in parity),'Cache tensor, hidden state or semantic metadata changed'
                (out/'cache_benchmark.json').write_text(json.dumps(report,indent=2))
        report.update(complete=True,ended=time.time(),all_fields_bitwise_equal=all(x['bitwise_equal']for x in report['parity']))
        seconds={mode:sum(r['seconds']for r in report['rows']if r['mode']==mode)for mode in a.modes}
        report['paired_total_seconds']=seconds;report['speedup']={mode:seconds['tick']/seconds[mode]for mode in a.modes if mode!='tick'};(out/'cache_benchmark.json').write_text(json.dumps(report,indent=2));print('CACHE_COMPLETE',report['speedup'],flush=True)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('dp','residual','paired','paired_worker','cache'));parser.add_argument('--root',required=True)
    parser.add_argument('--checkpoint');parser.add_argument('--output',required=True);parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--variants');parser.add_argument('--other-root');parser.add_argument('--other-checkpoint');args=parser.parse_args()
    if not 1<=args.repeats<=4:raise ValueError('Short bounded benchmark: 1 to 4 repetitions')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage=='cache':
        modes=tuple(args.variants.split(',')) if args.variants else ('tick','episode','episode_host')
        benchmark_cache(args.root,args.checkpoint,args.output,args.repeats,modes=modes);return
    if args.stage=='paired_worker':paired_worker(args);return
    with open('/tmp/tight_dagger_perf_gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        dict(dp=dp_benchmark,residual=residual_benchmark,paired=paired_benchmark)[args.stage](args)


if __name__=='__main__':main()
