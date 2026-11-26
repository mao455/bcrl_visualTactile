"""Opt-in fixed-distribution budget extension; diagnostics never change sampling."""
import copy
import json
import math
from pathlib import Path


def profile():
    return dict(version='fixed_distribution_extension_v1', maximum_epochs=24,
                minimum_epochs=16, patience=4, relative_improvement=.005,
                lr_factor=.5, maximum_reductions=1, activate_after_epoch=8)



def fresh_layout_profile(kind='frontend'):
    if kind not in ('frontend','backend'):raise ValueError('Unknown fresh layout budget')
    return dict(version='fresh_layout_budget_v1',kind=kind,
        maximum_epochs=80 if kind=='frontend' else 40,
        minimum_epochs=40 if kind=='frontend' else 20,
        patience=8 if kind=='frontend' else 6,relative_improvement=.005,
        lr_factor=.5,maximum_reductions=1,activate_after_epoch=20 if kind=='frontend' else 8)


def phase_dagger_profile(kind='frontend'):
    if kind not in ('frontend','backend'):raise ValueError('Unknown continuation kind')
    return dict(version='phase_dagger_budget_v1',kind=kind,maximum_epochs=40 if kind=='frontend' else 24,
        minimum_epochs=16 if kind=='frontend' else 12,patience=5,relative_improvement=.005,
        lr_factor=.5,maximum_reductions=1,activate_after_epoch=8)


def long_fixed_profile(kind='frontend', maximum_epochs=192):
    if kind not in ('frontend','backend'):raise ValueError('Unknown long-training component')
    if maximum_epochs not in ((192,360) if kind=='frontend' else (64,)):
        raise ValueError('Unregistered long-training budget')
    return dict(version='fixed56_long_budget_v1',kind=kind,maximum_epochs=maximum_epochs,
        minimum_epochs=maximum_epochs,patience=24 if kind=='frontend' else 12,
        relative_improvement=.005,lr_factor=.5,maximum_reductions=1,
        activate_after_epoch=96 if kind=='frontend' else 24)


def matched_update_profile(kind, maximum_epochs):
    """Finite constant-LR, final-checkpoint diagnostic; no adaptive selection."""
    if kind not in ('frontend','backend') or type(maximum_epochs) is not int or not 1<=maximum_epochs<=128:
        raise ValueError('Invalid finite matched-update budget')
    return dict(version='matched_update_diagnostic_v1',kind=kind,maximum_epochs=maximum_epochs,
        minimum_epochs=maximum_epochs,patience=maximum_epochs+1,relative_improvement=.005,
        lr_factor=.5,maximum_reductions=0,activate_after_epoch=maximum_epochs,
        selection='registered_final_update_only')


def rear_group_boundaries(masks, *, maximum_epochs=64, seed=2026111404):
    """Predict exact existing 32-tick/128-valid-tick updates, whole-group stops."""
    import torch
    ids=list(masks);generator=torch.Generator().manual_seed(seed);updates=0;result={}
    if not ids or any(not torch.as_tensor(m).bool().any() for m in masks.values()):
        raise ValueError('Each episode requires legal expert supervision')
    for epoch in range(1,maximum_epochs+1):
        order=[ids[i] for i in torch.randperm(len(ids),generator=generator).tolist()]
        for offset in range(0,len(order),4):
            group=order[offset:offset+4];length=max(len(masks[i]) for i in group);pending=0
            for start in range(0,length,32):
                pending+=sum(int(torch.as_tensor(masks[i][start:start+32]).bool().sum()) for i in group)
                if pending and (pending>=128 or start+32>=length):updates+=1;pending=0
            result[updates]=dict(epoch=epoch,next_group=offset+len(group))
    return result


def exposure50_profile(kind='frontend'):
    if kind not in ('frontend','backend'):raise ValueError('Unknown component')
    return dict(version='effective50_budget_v1',kind=kind,maximum_epochs=60,
        minimum_epochs=60,patience=8,relative_improvement=.005,lr_factor=.5,
        maximum_reductions=1,activate_after_epoch=40)


def valid_policy(value):
    if isinstance(value,dict) and value.get('version')=='matched_update_diagnostic_v1':
        try:return value==matched_update_profile(value['kind'],value['maximum_epochs'])
        except (KeyError,ValueError):return False
    return value in (exposure50_profile(),exposure50_profile('backend'),long_fixed_profile(),long_fixed_profile(maximum_epochs=360),long_fixed_profile('backend',64),profile(),fresh_layout_profile('frontend'),fresh_layout_profile('backend'),phase_dagger_profile(),phase_dagger_profile('backend'))


def load(root):
    path=Path(root)/'bounded_extension.json'
    if not path.exists():return None
    value=json.loads(path.read_text())
    if not valid_policy(value['policy']):raise ValueError('Unknown bounded extension policy')
    return value


class Plateau:
    def __init__(self, policy, state=None, baseline=None):
        if not valid_policy(policy):raise ValueError('Unknown bounded extension policy')
        self.policy=policy
        self.state=copy.deepcopy(state or dict(best=baseline, bad=0, reductions=0, stop=False, events=[]))

    def observe(self, epoch, score, optimizer):
        if not math.isfinite(score):raise FloatingPointError('Nonfinite independent validation')
        p=self.policy;s=self.state;event='observe'
        if s['best'] is None or score < s['best']*(1-p['relative_improvement']):
            s.update(best=score,bad=0);event='material_improvement'
        elif epoch>p['activate_after_epoch']:s['bad']+=1
        if s['bad']>=p['patience']:
            if s['reductions']<p['maximum_reductions']:
                for group in optimizer.param_groups:group['lr']*=p['lr_factor']
                s.update(bad=0,reductions=s['reductions']+1);event='reduce_lr'
            elif epoch>=p['minimum_epochs']:
                s['stop']=True;event='stop_validation_plateau'
        row=dict(epoch=epoch,score=score,event=event,bad=s['bad'],
                 next_learning_rates=[g['lr'] for g in optimizer.param_groups])
        s['events'].append(row)
        return row


def admit_dp_restore(old,new,registration,checkpoint):
    """Only the registered old boundary may cross this explicit identity change."""
    from .compact import sha256
    if sha256(checkpoint)!=registration['last_sha256']:
        raise ValueError('Extension must start from registered complete last checkpoint')
    allowed={'source_sha256','frozen_visual_io','frozen_visual_io_sha256',
             'planned_episode_visits','planned_optimizer_updates','planned_ema_updates','bounded_extension'}
    differences={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    if not differences<=allowed:raise ValueError('Extension changes training semantics: '+str(differences-allowed))
    if new['bounded_extension']['version']=='fixed56_long_budget_v1':
        if registration.get('version')!='fixed56_full_state_extension_v1':
            raise ValueError('Missing explicit full-state budget registration')
        if registration['old_policy']!=old['bounded_extension'] or registration['new_policy']!=new['bounded_extension']:
            raise ValueError('Registered continuation policies differ')
        for key in ('planned_episode_visits','planned_optimizer_updates','planned_ema_updates'):
            if new[key]<=old[key] or new[key]!=registration['new_budget'][key]:
                raise ValueError('Registered long budget mismatch')
    else:
        for key in ('planned_episode_visits','planned_optimizer_updates','planned_ema_updates'):
            if new[key]!=old[key]*3:raise ValueError('Expected registered 8 to 24 maximum budget')
    return dict(changed_configuration_fields=sorted(differences),optimizer_restored=True,
                ema_restored=True,all_rng_restored=True,old_scheduler='constant_lr_no_scheduler',
                scope='Same tensors and training state, new finite budget/diagnostic/I-O identity; not bitwise GPU replay')


def cohorts(data):
    train=data['splits']['train'];original=data['normalization_train_ids'];latest=data['aggregation']['new_training_ids']
    historical=[i for i in train if i not in set(original+latest)]
    # Fixed, predeclared coverage across the historical pool, not selected by loss.
    history=[historical[j*(len(historical)-1)//3] for j in range(4)]
    return dict(original=list(original[:4]),historical=history,latest=list(latest),validation=list(data['splits']['validation']))


def split_audit(data):
    groups={k:{data['entries'][i]['family'] for i in ids} for k,ids in data['splits'].items()}
    if groups['train']&groups['validation']:raise ValueError('Training/validation family leak')
    paths={k:{(data['entries'][i]['path'],data['entries'][i]['episode']) for i in ids} for k,ids in data['splits'].items()}
    if paths['train']&paths['validation']:raise ValueError('Repeated raw validation trajectory')
    return dict(train_trajectories=len(data['splits']['train']),train_families=len(groups['train']),
                validation_trajectories=len(data['splits']['validation']),validation_families=len(groups['validation']),
                family_intersection=[],raw_trajectory_intersection=[],cohorts=cohorts(data),
                scope='Registered initial-state families and exact raw episode identities; narrow-domain geometric similarity remains')


def phase_masks(entry):
    import h5py
    import numpy as np
    with h5py.File(entry['path']) as f:
        t=f['episodes'][entry['episode']]['transition'];force=t['contact_substeps_after_w_N'][:]
        pose=t['diagnostics_after/object_pose_w'][:];lift=t['diagnostics_after/lifted_object'][:].reshape(-1).astype(bool)
    touch=(np.linalg.norm(force,axis=-1)>.1).any(axis=(1,2));two=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)>=2
    n=len(touch);first=int(np.flatnonzero(touch)[0]) if touch.any() else n
    height=pose[:,2]>=np.median(pose[24:48,2])+.02
    return dict(all=np.ones(n,bool),before150=np.arange(n)<150,pre_contact=np.arange(n)<first,
                first_contact12=(np.arange(n)>=first)&(np.arange(n)<first+12),grasp_contact=two,
                native_lift=lift,contact_height=two&height,later=np.arange(n)>=150)


def error_bins(errors, ticks, masks, scale):
    import torch
    result={}
    for name,mask in masks.items():
        e=errors[torch.as_tensor(mask[ticks],dtype=torch.bool)]
        z=e*scale.cpu()
        result[name]=dict(ticks=len(e),arm_squared_sum=float(e[:,:7].square().sum()),
            hand_squared_sum=float(e[:,7:].square().sum()),
            arm_normalized_squared_sum=float(z[:,:7].square().sum()),hand_normalized_squared_sum=float(z[:,7:].square().sum()))
    return result


def summarize(rows):
    import math
    result={}
    for name in rows[0]:
        n=sum(r[name]['ticks'] for r in rows);a=sum(r[name]['arm_squared_sum'] for r in rows);h=sum(r[name]['hand_squared_sum'] for r in rows)
        an=sum(r[name]['arm_normalized_squared_sum'] for r in rows);hn=sum(r[name]['hand_normalized_squared_sum'] for r in rows)
        result[name]=dict(ticks=n,arm_rmse_rad=math.sqrt(a/(n*7)) if n else None,
            hand_rmse_rad=math.sqrt(h/(n*22)) if n else None,
            normalized_equal_mse=.5*(an/(n*7)+hn/(n*22)) if n else None)
    return result


def dp_diagnostics(model,records,data,objective,visual_io,offline,lease,*,cohort_ids=None):
    import torch
    from tools.tight_fresh_train import supervised_conditions
    from .native_data import Episode
    from .anchor_sampling import labels
    from .offline_planning import predict_contiguous
    from .position_control import bounded_targets
    from .anchor_continue import projection_batch
    result={}
    # Validation has no train-only rotating phase sampler. Use the accepted
    # fixed two-configuration estimator for this separately named diagnostic.
    diagnostic_objective=copy.copy(objective)
    diagnostic_objective.geometry_sampling_profile=None
    # Fork global RNG and use separate objective/noise generators: no change to training stream.
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        torch.manual_seed(87102);model.eval()
        for name,ids in (cohorts(data) if cohort_ids is None else cohort_ids).items():
            rows=[];losses=[];parts_rows=[]
            for identity in ids:
                lease.check();rec=records[identity]
                ep=Episode(rec['source_root'],identity,'I-HistoryDP',frozen_visual_io=visual_io)
                c,a=supervised_conditions(model,rec,episode=ep)
                truth,valid=labels(rec,a,24)
                pred=predict_contiguous(model,c,a,configuration=offline)
                pred=bounded_targets(pred.flatten(0,1),rec['joint_limits']).reshape_as(pred)
                mask=valid[:,:12];e=(pred[:,:12]-truth[:,:12])[mask].cpu()
                ticks=(a[:,None]+torch.arange(12,device=a.device)[None])[mask].cpu().numpy()
                rows.append(error_bins(e,ticks,phase_masks(data['entries'][identity]),model.action_scale))
                noise=torch.Generator(device=c.device).manual_seed(87103)
                context=projection_batch(diagnostic_objective,[(identity,int(t)) for t in a],'cuda:0',
                    geometry_generator=torch.Generator().manual_seed(87104))
                loss,parts,_=objective.forward(model,c,truth,valid,noise,geometry_context=context)
                losses.append(float(loss));parts_rows.append({k:float(v) for k,v in parts.items()})
                del ep
            result[name]=dict(ids=ids,phases=summarize(rows),fixed_noise_objective=sum(losses)/len(losses),
                geometry_diagnostic='fixed accepted two-configuration estimator; training rotating-half sampler unchanged',
                loss_components={k:sum(r[k] for r in parts_rows)/len(parts_rows) for k in parts_rows[0]})
    return result


def rear_diagnostics(model,records,data,lease,*,cohort_ids=None):
    import torch
    from .anchor_continue import residual_prediction
    result={}
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        for name,ids in (cohorts(data) if cohort_ids is None else cohort_ids).items():
            rows=[]
            for identity in ids:
                rec=records[identity];p=residual_prediction(model,rec,lease).cpu()
                valid=rec['supervision_mask'].cpu().bool();ticks=valid.nonzero().flatten().numpy()
                rows.append(error_bins((p-rec['target'].cpu())[valid],ticks,phase_masks(data['entries'][identity]),model.action_scale))
            result[name]=dict(ids=ids,phases=summarize(rows))
    return result
