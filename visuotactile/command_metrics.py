"""Offline sampled-command error with one fixed DDIM draw per replan condition."""
import time
import torch


@torch.no_grad()
def sampled_command_metrics(model, dataset, ids, variant, device, chunk_size=32, batch_size=2,
                            resource_guard=None):
    from .train_dp import _batch_episode_chunks, _condition, _move
    model.eval();state=None;zero_state=None
    generator=torch.Generator(device=device).manual_seed(101)
    error=torch.zeros(model.cfg.horizon,29,device=device,dtype=torch.float64)
    count=torch.zeros(model.cfg.horizon,device=device,dtype=torch.float64)
    zero_error=0.;zero_count=0
    memory_s=sample_s=0.;replans=0
    started=time.perf_counter()
    for chunk in _batch_episode_chunks(dataset,ids,chunk_size,batch_size):
        if resource_guard:
            resource_guard.before_chunk()
        if chunk['episode_start']:state=zero_state=None
        data=_move(chunk,device)
        if device.type=='cuda':torch.cuda.synchronize()
        tick=time.perf_counter();cond,state=_condition(model,variant,data,state)
        if device.type=='cuda':torch.cuda.synchronize()
        memory_s+=time.perf_counter()-tick
        due=(torch.arange(cond.shape[1],device=device)+chunk['start'])%model.cfg.replan_steps==0
        selected=data['obs']['valid'] & due[None]
        if selected.any():
            tick=time.perf_counter();rng=generator.get_state()
            prediction=model.predict_from_condition(cond[selected],generator)
            if device.type=='cuda':torch.cuda.synchronize()
            sample_s+=time.perf_counter()-tick
            labels=data['action_target'][selected];valid=data['action_valid'][selected]
            squared=(prediction-labels).double().square()
            error+=(squared*valid[...,None]).sum(0);count+=valid.sum(0);replans+=len(prediction)
        if variant=='b1':
            tactile={**data['tactile']}
            for key in ('contact_substeps_w_N','contact_substeps_local_N'):tactile[key]=torch.zeros_like(tactile[key])
            zero_cond,zero_state=_condition(model,variant,{**data,'tactile':tactile},zero_state)
            if selected.any():
                after=generator.get_state();generator.set_state(rng)
                prediction_zero=model.predict_from_condition(zero_cond[selected],generator)
                generator.set_state(after)
                zero_error+=float(((prediction_zero-labels).square()*valid[...,None]).sum())
                zero_count+=int(valid.sum())*29
        del data,cond
    mse=error/count.clamp_min(1)[:,None]
    return dict(protocol='one DDIM8 draw at every sixth control observation, seed101; full causal history',
        replans=replans,arm_rmse=float((error[:,:7].sum()/(count.sum()*7)).sqrt()),
        hand_rmse=float((error[:,7:].sum()/(count.sum()*22)).sqrt()),
        horizon_rmse=mse.mean(-1).sqrt().cpu().tolist(),
        horizon_valid_targets=count.cpu().tolist(),memory_seconds=memory_s,sampling_seconds=sample_s,
        wall_seconds=time.perf_counter()-started,
        zero_force_rmse=(zero_error/max(zero_count,1))**.5 if variant=='b1' else None,
        ablation='B1 zero world/local force causal replay; FK and times retained; same noise' if variant=='b1' else None,
        units='native clipped commands, not joint-angle errors')
