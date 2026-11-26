"""Isolated real-gradient gate, one-trajectory fit and fixed-model noise audit."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time
import traceback

import h5py
import numpy as np
import torch

from tools.runtime import read,write,sha
from tools.narrow_repair import beat
from tools.narrow_repair_audit import stages,motion
from tools.state_mechanism_audit import WindowSampler
from visuotactile.anchor_sampling import AnchorSampler,anchor_batch,labels,mask_totals,weighted_parts,continuation_lr
from visuotactile.checkpointing import save,training_state
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.position_control import contract,bounded_targets
from visuotactile.state_history import PersistentState,HistoryBank
from visuotactile.state_image_loss import GeometryObjective,LOSS_ID
from visuotactile.state_motion import target_metrics,plan_metrics
from visuotactile.state_resources import lease_for
from visuotactile.precision import state_fp32


def context(root,ids=None,fresh=False,train_weights=False):
    state_fp32();torch.set_num_threads(2)
    s=read(root/'session.json');parent=Path(s['parent']);data=read(parent/'state_data/data.json')
    if ids is None:ids=data['splits']['train']
    assert set(ids)<=set(data['splits']['train']),'Mechanism inputs are TRAIN only'
    cache=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    assert cache['data_sha256']==sha(parent/'state_data/data.json')
    records={i:cache['records'][i] for i in ids};del cache
    with h5py.File(data['entries'][ids[0]]['path']) as f:objective=GeometryObjective(json.loads(f.attrs['metadata_json']))
    objective.weights=s['loss_weights'];lease=lease_for(root,deadline=s['deadline'])
    torch.manual_seed(s['random_seed']);model=PersistentState('dp',contract(24,True))
    saved=None
    if fresh:
        stats=torch.load(parent/'state_data/statistics.pt',map_location='cpu',weights_only=False)
        expert=torch.load(s['expert_checkpoint'],map_location='cpu',weights_only=False)[0]['model']
        model.set_statistics(stats,expert);del expert
    else:
        saved=torch.load(parent/('expanded_training/last.pt' if train_weights else 'expanded_training/best.pt'),map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model' if train_weights else 'ema'])
    with lease.block():
        model.cuda();records={i:{k:v.cuda() for k,v in rec.items()} for i,rec in records.items()}
    return s,data,records,model,objective,lease,saved


def gradient(root):
    s,data,records,model,obj,lease,saved=context(root,train_weights=True)
    rng=torch.Generator();rng.set_state(saved['schedule_rng'])
    sampler=AnchorSampler({i:len(records[i]['time']) for i in saved['train_ids']},rng,64);sampler.load_state_dict(copy.deepcopy(saved['anchor_sampler']))
    selected=sampler.draw();lengths=sampler.lengths;totals=mask_totals(lengths,selected)
    noise=torch.Generator(device='cuda');noise.set_state(saved['noise_rng'])
    opt=torch.optim.AdamW(model.parameters(),lr=3e-6,betas=(.95,.999),weight_decay=1e-6)
    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    lr=continuation_lr(saved['updates']+1,saved['configuration']['target'],saved['configuration']['warmup'])
    for g in opt.param_groups:g['lr']=lr
    model.train();opt.zero_grad(set_to_none=True);captured=[];micro=[];loss_sum=0.
    gradients_none_before=all(p.grad is None for p in model.parameters())
    def hook(module,args,kwargs):captured.append((args[0].detach().clone(),args[1].detach().clone()))
    handle=model.denoiser.register_forward_pre_hook(hook,with_kwargs=True)
    for k in range(0,len(selected),4):
        with lease.block():
            c,y,m=anchor_batch(model,records,selected[k:k+4]);_,parts,_=obj.forward(model,c,y,m,noise)
            loss,_=weighted_parts(parts,m,totals,obj.weights);loss.backward();loss_sum+=float(loss)
            micro.append(dict(samples=selected[k:k+4],condition_shape=list(c.shape),target_shape=list(y.shape),mask_shape=list(m.shape),
                valid_rows_per_sample=m.sum(1).tolist(),valid_adjacent_pairs=int((m[:,1:]&m[:,:-1]).sum()),
                loss=float(loss),history_gradient_norm=float(model.history.weight_hh_l0.grad.norm()),optimizer_steps_before=saved['updates']))
        beat(root,stage='gradient_gate',microbatches=len(micro))
    handle.remove()
    with lease.block():
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
        updated={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        model.load_state_dict(saved['model']);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True)
        c,y,m=anchor_batch(model,records,selected)
        prediction=model.denoiser(torch.cat([x[0] for x in captured]),torch.cat([x[1] for x in captured]),global_cond=c)
        values=obj.components(prediction,y,m,model.action_scale,model.action_offset)
        full=values['base']+sum(obj.weights[k]*values[k] for k in obj.weights);full.backward()
        denominator=sum(float(g.double().square().sum()) for g in grads.values())
        difference=sum(float((p.grad.detach().cpu().double()-grads[n].double()).square().sum()) for n,p in model.named_parameters() if p.grad is not None)
        rel=math.sqrt(difference/max(denominator,1e-30))
        maximum=max(float((p.grad.detach().cpu()-grads[n]).abs().max()) for n,p in model.named_parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        param_max=max(float((p.detach().cpu()-updated[n]).abs().max()) for n,p in model.named_parameters())
        update_max=max(float((updated[n]-saved['model'][n]).abs().max()) for n in updated)
    loss_error=abs(loss_sum-float(full))/max(abs(loss_sum),1e-12)
    optimizer_steps=sorted({int(v['step']) for v in opt.state.values()})
    passed=(rel<s['gates']['gradient_relative_l2'] and param_max<s['gates']['parameter_update_max_abs'] and loss_error<s['gates']['loss_relative_error']
        and gradients_none_before and update_max>0 and optimizer_steps==[saved['updates']+1])
    result=dict(passed=passed,formal_model_unchanged=True,diagnostic_optimizer_steps=2,source_update=saved['updates'],
        effective_batch=len(selected),distinct_episodes=len(set(i for i,t in selected)),selected=selected,microbatches=micro,
        input_shapes={i:{k:list(records[i][k].shape) for k in ('actor','robot','geometry','command','time')} for i,t in selected},
        prefix_and_gradient=[dict(identity=i,anchor=t,prefix_ticks=t+1,detached_prefix=t//96*96,gradient_ticks=t%96+1) for i,t in selected],
        total_valid_targets=totals[0],total_adjacent_pairs=totals[1],micro_loss=loss_sum,full_loss=float(full),loss_relative_error=loss_error,
        gradient_norm_before_clip=norm,gradient_relative_l2=rel,gradient_max_abs=maximum,parameter_update_max_difference=param_max,
        actual_parameter_max_change=update_max,lr=lr,thresholds=s['gates'],
        gradients_none_before=gradients_none_before,optimizer_steps_after=optimizer_steps,
        deterministic_noise='Captured actual objective.forward noised inputs/timesteps from the16 microbatches, reused identically for reference full batch; no objective/weights change')
    write(root/'gradient_gate.json',result)
    if not passed:raise RuntimeError('Real full-model gradient/update equivalence failed')


@torch.no_grad()
def predictions(model,rec,lease,seed,common_noise=False):
    model.eval()
    with lease.block():condition,a,_=model.encode(rec)
    generator=torch.Generator(device='cuda').manual_seed(seed);plans=[]
    for j in range(len(a)):
        with lease.block():
            if common_noise:generator.manual_seed(seed)
            plans.append(model.predict(condition[j:j+1],generator)[0])
    return torch.stack(plans)


@torch.no_grad()
def metrics(plans,rec,obj,phase):
    n=len(rec['time']);raw=plans[:,:12].reshape(-1,29)[:n];bounded=bounded_targets(raw,rec['joint_limits']);truth=rec['command']
    result={};labels_np=np.asarray(phase);indexes=torch.arange(n,device='cuda');previous=rec['robot'][:,58:87]
    xyz,_=obj.fk(bounded);gt,_=obj.fk(truth);fk=(xyz[:,0]-gt[:,0]).norm(dim=-1)*1000
    for label,idx in [('all',torch.ones(n,dtype=torch.bool,device='cuda')),('first12',indexes<12)]+[(p,torch.tensor(labels_np==p,device='cuda')) for p in sorted(set(phase))]:
        error=bounded[idx]-truth[idx];rawerr=raw[idx]-truth[idx];diff=bounded[idx,:7]-previous[idx,:7];exact=truth[idx,:7]-previous[idx,:7]
        active=exact.norm(dim=-1)>1e-6
        row=dict(ticks=int(idx.sum()),arm_rmse=float(error[:,:7].square().mean().sqrt()),hand_rmse=float(error[:,7:].square().mean().sqrt()),
            fk_mm=float(fk[idx].square().mean().sqrt()),raw_arm_rmse=float(rawerr[:,:7].square().mean().sqrt()),
            direction_cos=float(torch.nn.functional.cosine_similarity(diff[active],exact[active]).mean()) if active.any() else None,
            one_tick_amplitude_ratio=float((diff[active].norm(dim=-1)/exact[active].norm(dim=-1)).median()) if active.any() else None,
            expert_actual_arm_speed_rms=float(rec['robot'][idx,29:36].square().mean().sqrt()))
        # Boundary/internal changes are conditioned on this tick's stage.
        delta=bounded[1:]-bounded[:-1];mask=idx[1:]
        for name,b in [('boundary',indexes[1:]%12==0),('internal',indexes[1:]%12!=0)]:
            use=mask&b;row[name+'_arm_jump_rms']=float(delta[use,:7].square().mean().sqrt()) if use.any() else None
        result[label]=row
    valid=torch.arange(0,n,12,device='cuda')[:,None]+torch.arange(24,device='cuda')[None]<n
    result['raw_plan_continuity']=plan_metrics(plans.cpu().numpy(),valid.cpu().numpy())
    result['submitted_continuity']=target_metrics(bounded.cpu().numpy());result['expert_continuity']=target_metrics(truth.cpu().numpy())
    result['clipping_max_rad']=float((raw-bounded).abs().max())
    # Raw overlap disagreement by stage, aligned to identical future control ticks.
    overlap=plans[:-1,12:]-plans[1:,:12];positions=torch.arange(12,len(plans)*12,12,device='cuda')[:,None]+torch.arange(12,device='cuda')[None]
    good=positions<n
    result['overlap_by_stage']={}
    for label in sorted(set(phase)):
        mask=good & torch.tensor(labels_np[np.minimum(positions.cpu().numpy(),n-1)]==label,device='cuda')
        if mask.any():result['overlap_by_stage'][label]=dict(arm_rms_rad=float(overlap[mask][:,:7].square().mean().sqrt()),hand_rms_rad=float(overlap[mask][:,7:].square().mean().sqrt()))
    return result


def stochasticity(root):
    plan=read(root/'plan.json');parent=Path(read(root/'session.json')['parent']);entries=read(parent/'manifest.json')['episodes']
    ids=[plan['tiny_entry']['id'],next(e['id'] for e in entries if e['family']==plan['developer_specs'][1]['family'] and e['split']=='train')]
    s,data,records,model,obj,lease,saved=context(root,ids=ids);phases=read(root/'phases_by_episode.json');rows=[]
    for identity in ids:
        samples=[]
        for seed in (7091,8091,9091,10091):
            pred=predictions(model,records[identity],lease,seed);samples.append(pred)
            with lease.block():row=metrics(pred,records[identity],obj,phases[identity])
            rows.append(dict(identity=identity,seed=seed,noise_mode='sequential_deployment_noise',metrics=row))
            beat(root,stage='stochasticity',identity=identity,seed=seed)
        pred=predictions(model,records[identity],lease,7091,common_noise=True)
        with lease.block():
            row=metrics(pred,records[identity],obj,phases[identity]);stack=torch.stack(samples)
            # Identical state/history, only DDIM initial noise differs.
            sd=stack.std(dim=0,unbiased=False);n=len(records[identity]['time']);valid=torch.arange(0,n,12,device='cuda')[:,None]+torch.arange(24,device='cuda')[None]<n
            variation=dict(arm_std_rms_rad=float(sd[valid][:,:7].square().mean().sqrt()),hand_std_rms_rad=float(sd[valid][:,7:].square().mean().sqrt()),
                first12_arm_std_rms_rad=float(sd[0,:12,:7].square().mean().sqrt()),first12_hand_std_rms_rad=float(sd[0,:12,7:].square().mean().sqrt()))
            phase_variation={}
            for label in sorted(set(phases[identity])):
                mask=np.asarray(phases[identity])==label;prefix=sd[:,:12].reshape(-1,29)[:n]
                phase_variation[label]=dict(arm_std_rms=float(prefix[torch.tensor(mask,device='cuda'),:7].square().mean().sqrt()),hand_std_rms=float(prefix[torch.tensor(mask,device='cuda'),7:].square().mean().sqrt()))
        rows.append(dict(identity=identity,seed=7091,noise_mode='same_initial_noise_each_replan_diagnostic_only',metrics=row,noise_variation=variation,variation_by_stage=phase_variation))
        write(root/'stochasticity.json',dict(complete=False,rows=rows,checkpoint_sha256=sha(parent/'expanded_training/best.pt'),weight_updates=0))
    write(root/'stochasticity.json',dict(complete=True,rows=rows,checkpoint_sha256=sha(parent/'expanded_training/best.pt'),weight_updates=0))


def rollout_motion(out):
    with h5py.File(out/'numeric.h5') as f:result=motion(f['episode'])
    p=np.load(out/'plans.npz');result['raw_plan_continuity']=plan_metrics(p['raw_plans'])
    result['after_overlap_plan_continuity']=plan_metrics(p['plans'])
    trace=np.load(out/'trace.npz');n=len(trace['joint']);raw=p['raw_plans'][:,:12].reshape(-1,29)[:n]
    result['raw_to_requested_max_rad']=float(abs(raw-trace['command']).max())
    result['requested_to_actuator_max_rad']=float(abs(trace['command'][:,None]-trace['actuator_targets']).max())
    result['actual_velocity_arm_rms']=float(np.sqrt(np.mean(trace['velocity'][:,:7]**2)))
    result['actual_velocity_hand_rms']=float(np.sqrt(np.mean(trace['velocity'][:,7:]**2)))
    return result


def train(root):
    identity=read(root/'plan.json')['tiny_entry']['id'];s,data,records,model,obj,lease,_=context(root,ids=[identity],fresh=True)
    directory=root/'tiny_training';directory.mkdir(exist_ok=False);budget=read(root/'tiny_budget.json');lease.deadline=budget['deadline']
    config=dict(contract=model.contract,seed=s['random_seed'],loss_weights=obj.weights,loss_id=LOSS_ID,
        effective_batch=s['tiny_batch'],microbatch=4,sampler='diagnostic_uniform_windows_short_tail',train_ids=[identity],
        statistics='Unchanged formal275 TRAIN-only normalizers and frozen expert RMS/embedding; no test statistics',
        initialization='fresh random model, optimizer, EMA, RNG and cosine schedule',target=s['tiny_max_updates'])
    write(directory/'configuration.json',config);phases=read(root/'phases_by_episode.json')[identity]
    with lease.block():
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-6,betas=(.95,.999),weight_decay=1e-6)
    generator=torch.Generator().manual_seed(s['random_seed']);noise=torch.Generator(device='cuda').manual_seed(s['random_seed']+1)
    lengths={identity:len(records[identity]['time'])};sampler=WindowSampler(lengths,generator,s['tiny_batch'])
    step=0;best=float('inf');history=[];stop=[False];began=time.time();reason='update_budget';fit_streak=0
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    def checkpoint(path):
        save(path,training_state(model,averaged,opt,ema,generator,noise,kind='dp',group=model.group,updates=step,
            diagnostic=True,train_ids=[identity],configuration=config,diagnostic_window_sampler=sampler.state_dict(),best=best,loss_id=LOSS_ID))
    while step<s['tiny_max_updates']:
        if time.time()>budget['deadline']-240 or (root/'STOP').exists() or stop[0]:reason='fixed_wall_or_stop';break
        selected=sampler.draw();totals=mask_totals(lengths,selected);model.train();opt.zero_grad(set_to_none=True);loss_sum=0.
        for j in range(0,len(selected),4):
            with lease.block():
                c,y,m=anchor_batch(model,records,selected[j:j+4]);_,parts,_=obj.forward(model,c,y,m,noise)
                loss,_=weighted_parts(parts,m,totals,obj.weights);assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss)
        with lease.block():
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm)
            step+=1;lr=continuation_lr(step,s['tiny_max_updates'],200)
            for g in opt.param_groups:g['lr']=lr
            opt.step();ema.step(model)
        progress=dict(update=step,loss=loss_sum,actual_batch=len(selected),selected=selected,valid_rows=totals[0],valid_pairs=totals[1],
            gradient_norm_before_clip=norm,lr=lr,draws=sampler.draws,unique_anchors=sum(v>0 for v in sampler.counts),
            anchor_equivalent_epochs=sampler.draws/len(sampler.population),updates_per_hour=step*3600/(time.time()-began),utc=time.time())
        write(directory/'progress.json',progress);beat(root,stage='tiny_training',update=step)
        with (directory/'training.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
        if step==1 or step%100==0:checkpoint(directory/'last.pt')
        if step==1 or step%200==0 or step==s['tiny_max_updates']:
            scores=[]
            for seed in (7091,8091,9091):
                pred=predictions(averaged,records[identity],lease,seed)
                with lease.block():scores.append(dict(seed=seed,metrics=metrics(pred,records[identity],obj,phases)))
            score=float(np.mean([.5*(x['metrics']['all']['arm_rmse']**2+x['metrics']['all']['hand_rmse']**2) for x in scores]))
            row=dict(update=step,selection_mse=score,seeds=scores,utc=time.time());history.append(row)
            if score<best:best=score;checkpoint(directory/'best.pt')
            checkpoint(directory/'last.pt');write(directory/'fit_history.json',history)
            fit=all(x['metrics']['all']['arm_rmse']<s['tiny_fit']['arm_rmse'] and x['metrics']['all']['hand_rmse']<s['tiny_fit']['hand_rmse']
                and x['metrics']['all']['fk_mm']<s['tiny_fit']['fk_mm'] and x['metrics']['first12']['arm_rmse']<s['tiny_fit']['first_arm_rmse'] for x in scores)
            fit_streak=fit_streak+1 if fit else 0
            if step>=s['tiny_fit']['minimum_updates'] and fit_streak>=2:reason='small_dataset_fit_gate_two_checks';break
            if step>=1200 and len(history)>=4 and 1-min(x['selection_mse'] for x in history[-3:])/history[-4]['selection_mse']<.01:
                reason='small_dataset_loss_plateau';break
    checkpoint(directory/'last.pt')
    assert (directory/'best.pt').exists(),'No assessed diagnostic model'
    cp=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    assert cp['contract']==model.contract and cp['train_ids']==[identity]
    assert cp['updates']==cp['ema_optimization_step'] and all(int(v['step'])==cp['updates'] for v in cp['optimizer']['state'].values())
    write(directory/'finished.json',dict(complete=True,updates=step,best_update=cp['updates'],stop_reason=reason,elapsed_s=time.time()-began,
        best_sha256=sha(directory/'best.pt'),checkpoint_readback=True,normalization_unchanged=True,generalization_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gradient','stochasticity','train'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    try:globals()[a.mode](root)
    except BaseException:
        write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc(),utc=time.time()));raise
