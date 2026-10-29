"""Bounded real-production acceptance; writes local evidence, never full training."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch

from .compact import EpisodeFile
from .manifest import ManifestReader, ManifestSequenceDataset
from .models.nominal import PolicyConfig
from .train_dp import _make_model, _move, _condition, _batch_episode_chunks, _save, _fit_proprio_stats, _split, _run_epoch
from .history import detach_state
from .rollout import load_command_controller
from .training_resources import atomic_json, ResourceGuard


def _slice(data, rows, steps):
    return {k:({n:x[rows,steps] for n,x in v.items()} if isinstance(v,dict) else v[rows,steps])
            for k,v in data.items() if k in ('obs','tactile','tactile_valid')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.36)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda'); reader=ManifestReader(args.manifest);splits=_split(reader,41)
    all_ds=ManifestSequenceDataset(args.manifest,stage='production',include_tactile=True,require_command=True)
    ordered=sorted(splits[0],key=lambda k:len(all_ds.datasets[k]))
    ids=[ordered[0],ordered[-1]]
    report=dict(scope='real production data; bounded diagnostic updates, not formal epochs',ids=ids,
                lengths=[len(all_ds.datasets[k]) for k in ids], variants={})
    mean,std,rows=_fit_proprio_stats(ManifestSequenceDataset(args.manifest,ids=splits[0],stage='production'))
    for variant in ('b0','b1'):
        torch.manual_seed(41)
        cfg=PolicyConfig(views=3,own_plan_mode='none',vision_checkpoint_images=12)
        model=_make_model(variant,cfg).to(device)
        encoder=model.encoder if variant=='b0' else model.visual
        encoder.proprio_mean.copy_(mean.to(device));encoder.proprio_std.copy_(std.to(device))
        ds=ManifestSequenceDataset(args.manifest,ids=ids,stage='production',include_tactile=variant=='b1',require_command=True)
        batch=next(_batch_episode_chunks(ds,ids,32,2))
        for j,key in enumerate(ids):
            row=reader.entries[key]
            with EpisodeFile(reader.path.parent/row['directory']/'data.h5') as f:
                g=f['episodes'][row['episode']]
                commands=g['transition/command_clipped_pre_delay'][:]
                for t in range(32):
                    np.testing.assert_array_equal(batch['action_target'][j,t],commands[t:t+12])
                assert np.abs(commands).max()<=1
                valid=np.asarray(g['obs/contact_valid'][:]).astype(bool)
                clock=g['obs/time_s'][:]
                times=g['obs/contact_time_s'][:]
                assert np.all(times[valid] <= clock[valid,None]+1e-6)
                assert np.all(np.diff(times[valid].reshape(-1))>0)
        data=_move(batch,device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
        submodules=dict(rgb=encoder.vision.rgb,depth=encoder.vision.depth,history=encoder.temporal,denoiser=model.denoiser)
        if variant=='b1':submodules.update(tactile=model.tactile,fusion=model.fusion)
        before={k:[p.detach().clone() for p in m.parameters()] for k,m in submodules.items()}
        def loss_fixed():
            # Fixed diffusion RNG solely for the diagnostic trend.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(91);torch.cuda.manual_seed_all(91)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    cond,_=_condition(model,variant,data,None)
                    return model.loss_from_condition(cond.flatten(0,1),data['action_target'].flatten(0,1),
                        (data['action_valid']&data['obs']['valid'][...,None]).flatten(0,1))
        torch.cuda.reset_peak_memory_stats()
        model.train();trend=[];grads={}
        for step in range(8):
            optimizer.zero_grad(set_to_none=True);loss=loss_fixed();trend.append(float(loss.detach()))
            loss.backward()
            grads={k:sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None) for k,m in submodules.items()}
            assert all(np.isfinite(v) and v>0 for v in grads.values())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(norm)
            optimizer.step()
        del loss
        delta={k:sum(float((p-q).abs().sum()) for p,q in zip(m.parameters(),before[k])) for k,m in submodules.items()}
        assert all(v>0 and np.isfinite(v) for v in delta.values())
        model.eval()
        with torch.no_grad():
            final=float(loss_fixed())
        peak=torch.cuda.max_memory_allocated();del before,data,batch
        # Fixed weights, every time in both episodes, full versus singleton,
        # swap, causal future edits, and branch states through masked tails.
        state=None;single_states=[None,None];swapped_state=None;error=swaperr=0.;valid_steps=0
        def state_error(a,b,row):
            if isinstance(a,dict):return max(state_error(a[k],b[k],row) for k in a)
            if a.dtype==torch.bool:return float((a[row:row+1]!=b).any())
            return float((a[row:row+1]-b).abs().max())
        stateerr=0.
        with torch.no_grad():
            for batch in _batch_episode_chunks(ds,ids,32,2):
                data=_move(batch,device);cond,state=_condition(model,variant,data,state)
                swapped,swapped_state=_condition(model,variant,_slice(data,[1,0],slice(None)),swapped_state)
                swaperr=max(swaperr,float((cond-swapped.flip(0)).abs().max()))
                for j in range(2):
                    n=int(data['obs']['valid'][j].sum())
                    if n:
                        one,single_states[j]=_condition(model,variant,_slice(data,slice(j,j+1),slice(0,n)),single_states[j])
                        error=max(error,float((cond[j,:n]-one[0]).abs().max()));valid_steps+=n
                    stateerr=max(stateerr,state_error(state,single_states[j],j))
                del data,cond,swapped
        print('fixed_weight_errors',variant,error,swaperr,stateerr,flush=True)
        assert max(error,swaperr,stateerr)<2e-5,(error,swaperr,stateerr)
        # Actual saved loader and every-step online replay on the real prefix.
        settings=SimpleNamespace(variant=variant,seed=41,chunk_size=32,batch_size=2,amp='bf16')
        checkpoint=args.output/f'{variant}_diagnostic.pt'
        _save(checkpoint,model,optimizer,cfg,settings,'diagnostic',splits,0,[],mean,std,
              metadata=ds.datasets[ids[0]].metadata,optimizer_updates=8)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);saved['diagnostic_only']=True;torch.save(saved,checkpoint)
        restored=load_command_controller(checkpoint,2)
        batch=next(_batch_episode_chunks(ds,ids,32,2));cpu_data=_slice(batch,slice(None),slice(0,8))
        with torch.no_grad():expected,_=_condition(restored.model,variant,cpu_data,None)
        rng=torch.get_rng_state().clone();online_error=0
        for t in range(8):
            packet={k:v[:,t] for k,v in cpu_data['obs'].items() if k!='valid'}
            packet.update(episode_id=torch.zeros(2,dtype=torch.long),step_id=torch.full((2,),t))
            if variant=='b1':packet.update({k:v[:,t] for k,v in cpu_data['tactile'].items()},contact_valid=cpu_data['tactile_valid'][:,t])
            action=restored.act(packet);assert action.shape==(2,29) and action.abs().max()<=1
            online_error=max(online_error,float((restored.condition_rows(torch.ones(2,dtype=torch.bool))-expected[:,t]).abs().max()))
            assert not restored.last_packet['prior_plan_valid'].any()
        assert torch.equal(rng,torch.get_rng_state())
        assert online_error<2e-5
        result=dict(label_alignment='obs[t] -> clipped command[t:t+12] exact',
            tactile_causality='all valid stored physical timestamps <= observation time, strictly ordered',
            gradient_l1=grads,parameter_delta_l1=delta,fixed_noise_losses=trend+[final],
            peak_cuda_allocated_bytes=peak,batch_singleton_max_error=error,batch_swap_max_error=swaperr,
            branch_state_max_error=stateerr,compared_valid_steps=valid_steps,
            actual_controller_max_error=online_error,updates=8,diagnostic_checkpoint=str(checkpoint))
        report['variants'][variant]=result
        atomic_json(args.output/'report.json',report)
        print(variant,json.dumps(result),flush=True)
        del model,optimizer,restored,encoder,submodules,batch,cpu_data,expected
        torch.cuda.empty_cache()
    if args.benchmark:
        report['benchmarks']=[]
        for variant,batch_size,prefetch in (('b0',1,0),('b0',2,1),('b1',2,1)):
            ds=ManifestSequenceDataset(args.manifest,ids=splits[0][:12],stage='production',include_tactile=variant=='b1')
            model=_make_model(variant,cfg).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            guard=ResourceGuard(args.output/f'bench_{variant}_{batch_size}',min_ram_available_gib=8)
            torch.cuda.reset_peak_memory_stats()
            result=_run_epoch(model,ds,splits[0][:12],variant,device,32,optimizer,batch_size=batch_size,
                amp='bf16',resource_guard=guard,prefetch=prefetch)
            guard.sample();result.update(variant=variant,batch_size=batch_size,prefetch=prefetch,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),resources=guard.summary())
            report['benchmarks'].append(result);atomic_json(args.output/'report.json',report)
            print('benchmark',json.dumps(result),flush=True)
            del model,optimizer;torch.cuda.empty_cache()

if __name__=='__main__':main()
