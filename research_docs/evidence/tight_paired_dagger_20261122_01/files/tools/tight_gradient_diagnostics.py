"""Bounded paired residual diagnosis on hash-bound, causally replayed records.

This tool never modifies formal runs, DP weights, caches, or test data. Probe
weights are disposable and cannot be selected by the ordinary DAgger owner.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from visuotactile.anchor_sampling import recovery_prefix_rows, supervision_rows
from visuotactile.compact import sha256
from visuotactile.models.native_dp import load_residual
from visuotactile.native_data import residual_batch
from visuotactile.position_control import project_processed_position_targets
from visuotactile.training_resources import atomic_json

GROUPS = ('QG01', 'G01')
THRESHOLDS = dict(arm_error_rad=1e-4, hand_error_rad=1e-3,
    local_fit_gradient_per_rad=1e-10, clip_change_rad=1e-7,
    tanh_saturation_abs=.95, label_feasibility_rad=1e-6,
    arm_movement_rad=1e-4, hand_movement_rad=1e-3)


def read(path):
    return json.loads(Path(path).read_text())


def group_root(root, group):
    return root/'round_01'/group/'backend'/group


def manifest(root, out):
    paths = {g: group_root(root, g) for g in GROUPS}
    data = {g: read(paths[g]/'state_data/data.json') for g in GROUPS}
    if data['QG01']['splits'] != data['G01']['splits']:
        raise ValueError('Paired data identities differ')
    d = data['QG01']; selected = []
    for identity in sorted(d['sampling_strata']['core'])[:2]:
        selected.append(dict(identity=identity, partition='train', kind='core'))
    for source in GROUPS:
        train = sorted(i for i in d['sampling_strata']['current_correction'] if f'__{source}__r1__' in i)
        dev = sorted(i for i in d['splits']['correction_dev'] if i.startswith(root.name+'__') and f'__{source}__r1__' in i)
        for identity, partition in ((train[0], 'train'), (dev[0], 'dev')):
            selected.append(dict(identity=identity, partition=partition, kind='correction', source=source))
    identities = {}; common_records = {}
    for g in GROUPS:
        p = paths[g]; checkpoint = p/'training/best.pt'; payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = payload['configuration']; index = read(p/'nominal/index.json')
        if sha256(cfg['base_checkpoint']) != cfg['base_sha256'] or index['base_sha256'] != cfg['base_sha256']:
            raise ValueError('Current DP/checkpoint/cache binding differs')
        if sha256(p/'nominal/index.json') != cfg['cache_index_sha256']:
            raise ValueError('Residual was not trained on this cache index')
        identities[g] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            dp_checkpoint=cfg['base_checkpoint'], dp_sha256=cfg['base_sha256'],
            cache_index_sha256=sha256(p/'nominal/index.json'), data_sha256=sha256(p/'state_data/data.json'),
            command_statistics_sha256=cfg['residual_statistics_sha256'],
            execution_adapter=cfg.get('execution_adapter'), original_loss_id=cfg['loss_id'],
            dp_supervision='q_reached' if g=='QG01' else 'u_expert', backend_supervision='u_expert')
        for s in selected:
            identity=s['identity']; cache=p/'nominal'/f'{identity}.pt'
            if sha256(cache)!=index['episodes'][identity]['sha256']:
                raise ValueError('Selected cache payload changed')
            record=torch.load(cache,map_location='cpu',weights_only=False)
            binding=record['cache_identity']
            if binding['base_checkpoint_sha256']!=cfg['base_sha256'] or binding['residual_statistics_sha256']!=cfg['residual_statistics_sha256']:
                raise ValueError('Selected cache model/normalizer mismatch')
            s.setdefault('cache',{})[g]=dict(path=str(cache),sha256=sha256(cache))
            if g==GROUPS[0]:
                mask,segment=supervision_rows(record); early=recovery_prefix_rows(record,48) if s['kind']=='correction' else torch.zeros_like(mask)
                s.update(ticks=len(mask),valid_supervised_ticks=int(mask.sum()),recovery_initial48_ticks=int(early.sum()),
                    first_supervised_tick=int(mask.nonzero()[0]),last_supervised_tick=int(mask.nonzero()[-1]),
                    source_entry=d['entries'][identity])
                common_records[identity]=record
            else:
                a=common_records[identity]
                for key in ('robot','target','time','supervision_mask','control_segment','plan_cursor'):
                    if not torch.equal(a[key],record[key]):raise ValueError('A/B factual history or labels differ: '+key)
    result=dict(version='tight_gradient_diagnosis_v1',created_utc=time.time(),root=str(root),thresholds=THRESHOLDS,
        sample_rule='Lexicographically first 2 Core training IDs; first current-round correction training ID and first held-out current-round correction-dev ID from each collection source. Entire episodes retained. No model error used for selection.',
        selection=selected,models=identities,test_episodes_read=0,formal_optimizer_updates=0,
        history='Every current model replays the complete actual prefix from tick0; detached history is rebuilt after every optimizer update.',
        gradient_definition='Autograd of command-scale-normalized equal Arm/Hand MSE only, before residual regularizer. Local gradients remove masked-weight/sum-weight reduction, so the threshold does not depend on episode length. Lost fraction denominator is explicitly all valid joint coordinates and separately errorful coordinates.',
        probe=dict(maximum_updates_per_group_per_loss=200,lambda_pre=1.,learning_rate=3e-4,weight_decay=1e-4,
            optimizer='fresh identical AdamW states per paired candidate',chunk_ticks=32,train_ids=[s['identity'] for s in selected if s['partition']=='train'],
            selection='fixed final update, never dev-selected',disposable=True))
    atomic_json(out/'manifest.json',result)
    return result


def model_and_records(m, group, device='cpu'):
    payload=torch.load(m['models'][group]['checkpoint'],map_location='cpu',weights_only=False)
    model=load_residual(payload,device=device)
    records={s['identity']:torch.load(s['cache'][group]['path'],map_location='cpu',weights_only=False) for s in m['selection']}
    return model,records,payload


def chain(model,base,raw,limits,previous):
    """Decompose actual compose; exact parity checked on every audited sample."""
    if hasattr(model,'compose_chain'):
        result=model.compose_chain(base,raw,None,limits,previous=previous)
    else:
        delta=model.scales*raw.tanh();pre=base+delta/model.action_scale
        joint=torch.maximum(torch.minimum(pre,limits[...,1]),limits[...,0])
        execute,_=model.compose(base,raw,None,limits,previous=previous)
        result=dict(raw=raw,delta=delta,u_pre=pre,u_joint=joint,u_exec=execute)
    actual,delta=model.compose(base,raw,None,limits,previous=previous)
    if not torch.equal(actual,result['u_exec']) or not torch.equal(delta,result['delta']):
        raise ValueError('Diagnostic chain differs from actual execution')
    return result


def fit(error):
    return .5*error[...,:7].square().mean(-1)+.5*error[...,7:].square().mean(-1)


def evaluate(model,records,m,group,out,tag):
    device=next(model.parameters()).device;model.eval();summaries=[];arrays=[]
    for selection in m['selection']:
        identity=selection['identity'];record=records[identity];state=None;raws=[]
        with torch.no_grad():
            for start in range(0,len(record['robot']),32):
                raw,state=model(residual_batch([record],start,min(start+32,len(record['robot'])),device),state)
                raws.append(raw[0].cpu())
        raw=torch.cat(raws).to(device).detach().requires_grad_()
        base=record['base'].to(device);target=record['target'].to(device);previous=record['robot'][:,58:87].to(device)
        mask,_=supervision_rows(record);early=recovery_prefix_rows(record,48) if selection['kind']=='correction' else torch.zeros_like(mask)
        weight=mask.to(device).float()*(1+2*early.to(device).float())
        c=chain(model,base,raw,record['limits'].to(device),previous)
        main=(fit((c['u_exec']-target)*model.action_scale)*weight).sum()/weight.sum().clamp_min(1)
        preloss=(fit((c['u_pre']-target)*model.action_scale)*weight).sum()/weight.sum().clamp_min(1)
        regularizer=(fit(c['delta'])*weight).sum()/weight.sum().clamp_min(1)*1e-4
        grad_pre,grad_raw=torch.autograd.grad(main,(c['u_pre'],raw),retain_graph=True)
        grad_pre_aux,grad_raw_aux=torch.autograd.grad(preloss,(c['u_pre'],raw),retain_graph=True)
        grad_regularizer=torch.autograd.grad(regularizer,raw)[0]
        reduction=(weight/weight.sum().clamp_min(1))[:,None]
        local_grad=torch.where(reduction>0,grad_pre/reduction.clamp_min(1e-30),0.)
        local_grad_new=torch.where(reduction>0,(grad_pre+grad_pre_aux)/reduction.clamp_min(1e-30),0.)
        expert_projected=project_processed_position_targets(target,previous,record['limits'].to(device),model.execution_adapter)
        tensors=dict(base=base,raw=raw,delta=c['delta'],residual_rad=c['delta']/model.action_scale,
            plans=record['plans'],plan_valid=record['plan_valid'],u_pre=c['u_pre'],u_joint=c['u_joint'],u_exec=c['u_exec'],
            target=target,previous=previous,fit_grad_u_pre=grad_pre,local_fit_grad_u_pre=local_grad,
            local_new_fit_grad_u_pre=local_grad_new,fit_grad_raw=grad_raw,pre_fit_grad_raw=grad_raw_aux,
            regularizer_grad_raw=grad_regularizer,expert_projection_delta=expert_projected-target,
            needed_normalized_delta=(target-base)*model.action_scale,action_scale=model.action_scale,
            scales=model.scales,mask=mask,weight=weight,early=early,plan_cursor=record['plan_cursor'])
        values={k:v.detach().cpu().numpy() for k,v in tensors.items()};arrays.append((selection,values))
        strata={'core':mask} if selection['kind']=='core' else {'recovery_initial48':mask&early,'recovery_later':mask&~early}
        for stratum,sel in strata.items():
            for part,a,b,tol in [('Arm',0,7,THRESHOLDS['arm_error_rad']),('Hand',7,29,THRESHOLDS['hand_error_rad'])]:
                selected=sel.numpy();err=values['u_exec'][selected,a:b]-values['target'][selected,a:b]
                pre_err=values['u_pre'][selected,a:b]-values['target'][selected,a:b];hold=values['previous'][selected,a:b]-values['target'][selected,a:b]
                errorful=np.abs(err)>tol;gradient=values['local_fit_grad_u_pre'][selected,a:b]
                lost=errorful&(np.abs(gradient)<=THRESHOLDS['local_fit_gradient_per_rad'])
                lost_new=errorful&(np.abs(values['local_new_fit_grad_u_pre'][selected,a:b])<=THRESHOLDS['local_fit_gradient_per_rad'])
                movement=values['target'][selected,a:b]-values['previous'][selected,a:b];pred_move=values['u_exec'][selected,a:b]-values['previous'][selected,a:b]
                moving=np.abs(movement)>tol;n=err.size;scale=values['action_scale'][a:b]
                ws=values['weight'][selected,None];mass=(err*scale)**2*ws
                summaries.append(dict(group=group,candidate=tag,identity=identity,partition=selection['partition'],stratum=stratum,part=part,
                    ticks=int(selected.sum()),joint_coordinates=n,errorful_coordinates=int(errorful.sum()),
                    lost_coordinates=int(lost.sum()),lost_fraction_all=float(lost.sum()/max(n,1)),
                    lost_fraction_errorful=float(lost.sum()/max(errorful.sum(),1)),
                    new_loss_lost_coordinates=int(lost_new.sum()),
                    lost_weighted_error_fraction=float(mass[lost].sum()/max(mass.sum(),1e-30)),
                    exec_rmse_rad=float(np.sqrt(np.mean(err**2))) if n else None,
                    pre_rmse_rad=float(np.sqrt(np.mean(pre_err**2))) if n else None,
                    hold_rmse_rad=float(np.sqrt(np.mean(hold**2))) if n else None,
                    exec_sse_rad=float((err**2).sum()),pre_sse_rad=float((pre_err**2).sum()),hold_sse_rad=float((hold**2).sum()),
                    joint_clip_coordinates=int((np.abs(values['u_joint'][selected,a:b]-values['u_pre'][selected,a:b])>THRESHOLDS['clip_change_rad']).sum()),
                    projection_coordinates=int((np.abs(values['u_exec'][selected,a:b]-values['u_joint'][selected,a:b])>THRESHOLDS['clip_change_rad']).sum()),
                    saturated_coordinates=int((np.abs(np.tanh(values['raw'][selected,a:b]))>=.95).sum()),
                    theoretical_range_exceeded_coordinates=int((np.abs(values['needed_normalized_delta'][selected,a:b])>values['scales'][a:b]+1e-7).sum()),
                    label_projection_changed_coordinates=int((np.abs(values['expert_projection_delta'][selected,a:b])>1e-6).sum()),
                    label_projection_max_rad=float(np.abs(values['expert_projection_delta'][selected,a:b]).max()) if n else None,
                    moving_coordinates=int(moving.sum()),direction_correct_coordinates=int(((movement*pred_move>0)&moving).sum()),
                    normalized_weighted_exec_error_mass=float(mass.sum()),lost_normalized_weighted_exec_error_mass=float(mass[lost].sum()),
                    normalized_weighted_pre_error_mass=float(((pre_err*scale)**2*ws).sum()),
                    normalized_weighted_hold_error_mass=float(((hold*scale)**2*ws).sum()),
                    fit_raw_gradient_l2=float(np.linalg.norm(values['fit_grad_raw'][selected,a:b])),
                    pre_fit_raw_gradient_l2=float(np.linalg.norm(values['pre_fit_grad_raw'][selected,a:b])),
                    regularizer_raw_gradient_l2=float(np.linalg.norm(values['regularizer_grad_raw'][selected,a:b]))))
    target=out/group/tag;target.mkdir(parents=True,exist_ok=True)
    with (target/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    # Full numeric output chain, masks and plan are local; scalar metrics are public-safe.
    for selection,values in arrays:
        np.savez_compressed(target/(selection['identity']+'.npz'),**values)
    atomic_json(target/'summary.json',dict(group=group,candidate=tag,rows=summaries,
        history='Full observation prefix replayed using this exact candidate; no stale residual hidden states.',
        test_episodes_read=0,weights_updated=False))
    return summaries


def diagnose(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);m=manifest(Path(args.root).resolve(),out)
    atomic_json(out/'progress.json',dict(stage='diagnosis',state='RUNNING',started_utc=time.time()))
    allrows=[]
    for group in GROUPS:
        model,records,_=model_and_records(m,group,args.device)
        rows=evaluate(model,records,m,group,out,'initial');allrows+=rows
        atomic_json(out/'diagnosis.json',dict(version=m['version'],completed_groups=[r for r in GROUPS if r==group] if not allrows else sorted(set(r['group'] for r in allrows)),rows=allrows,thresholds=THRESHOLDS))
        print(json.dumps(dict(group=group,stage='initial_complete',rows=len(rows))),flush=True)
    atomic_json(out/'progress.json',dict(stage='diagnosis',state='COMPLETE',finished_utc=time.time()))


def probe_sequence(m,records,updates):
    banks=[]
    for selection in m['selection']:
        if selection['partition']!='train':continue
        record=records[selection['identity']];mask,_=supervision_rows(record)
        early=recovery_prefix_rows(record,48)
        masks=[('core',mask)] if selection['kind']=='core' else [('early',mask&early),('later',mask&~early)]
        for kind,eligible in masks:
            windows=[]
            for start in range(0,len(mask),32):
                stop=min(start+32,len(mask))
                if eligible[start:stop].any():windows.append(dict(identity=selection['identity'],start=start,stop=stop,bank=kind))
            if windows:banks.append(windows)
    counts=[0]*len(banks);sequence=[]
    for update in range(updates):
        b=update%len(banks);sequence.append(banks[b][counts[b]%len(banks[b])]);counts[b]+=1
    return sequence


def probe(args):
    from visuotactile.models.native_dp import TargetResidual
    from tools.tight_performance import Samples,resources
    if not 1<=args.updates<=200:raise ValueError('Bounded probe allows 1..200 updates only')
    out=Path(args.output).resolve();m=read(out/'manifest.json');root=Path(args.root).resolve()
    if args.device!='cuda:0':raise ValueError('Use explicitly scheduled small CUDA probe')
    if (out/'probe_preregistration.json').exists():raise ValueError('No automatic rerun or changed probe budget')
    _,reference,_=model_and_records(m,GROUPS[0]);sequence=probe_sequence(m,reference,args.updates)
    prereg=dict(created_utc=time.time(),updates=args.updates,sequence=sequence,lambda_pre=1.,
        learning_rate=3e-4,weight_decay=1e-4,grad_norm_clip=1.,seed=20261112,
        history='Rebuild full prefix with current weights before every optimizer update, then 32tick truncated gradients. No saved stale Transformer state.',
        data='4 fixed training episodes only; 2 correction-dev episodes never optimized; recovery first48 weight3 in both losses.',
        formal_switch_gates=dict(train_pre_mse_ratio_vs_initial_max=.99,dev_exec_mse_ratio_vs_old_max=1.02,
            core_exec_mse_ratio_vs_old_max=1.05,all_losses_finite=True,u_pre_dead_gradient_restored=True,
            applies_to_both_groups=True),no_automatic_probe_extension=True)
    atomic_json(out/'probe_preregistration.json',prereg);all_candidates={};started=time.time()
    for group in GROUPS:
        _,records,payload=model_and_records(m,group)
        cfg=payload['configuration'];all_candidates[group]={}
        for candidate in ('old_loss','preclip_loss'):
            contract=None if candidate=='old_loss' else dict(version='exec_preclip_equal_arm_hand_v1',lambda_pre=1.)
            torch.manual_seed(prereg['seed']);torch.cuda.manual_seed_all(prereg['seed'])
            model=TargetResidual(payload['model'],payload['model']['scales'],execution_adapter=cfg['execution_adapter'],loss_contract=contract).to(args.device)
            model.load_state_dict(payload['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=prereg['learning_rate'],weight_decay=prereg['weight_decay'])
            if candidate=='old_loss':evaluate(model,records,m,group,out,'probe_initial')
            rows=[];began=time.time();torch.cuda.reset_peak_memory_stats()
            with Samples() as telemetry:
                for update,item in enumerate(sequence,1):
                    if update%10==1:
                        resources()
                        formal=read(root/'status.json')
                        formal_stage=formal.get('stage','')
                        if 'parallel_frontends' in formal_stage or ('parallel' in formal_stage and 'train' in formal_stage):
                            raise RuntimeError('Formal parallel training started: stop diagnostic before contention')
                    record=records[item['identity']];state=None;start=item['start'];stop=item['stop'];model.train()
                    # Prefix histories depend on trainable recurrent parameters and must never survive opt.step().
                    with torch.no_grad():
                        for prefix in range(0,start,32):
                            _,state=model(residual_batch([record],prefix,min(prefix+32,start),args.device),state)
                    inputs=residual_batch([record],start,stop,args.device);raw,_=model(inputs,state)
                    c=chain(model,record['base'][None,start:stop].to(args.device),raw,record['limits'].to(args.device),inputs['robot'][...,58:87])
                    mask,_=supervision_rows(record)
                    selection=next(s for s in m['selection'] if s['identity']==item['identity'])
                    early=recovery_prefix_rows(record,48) if selection['kind']=='correction' else torch.zeros_like(mask)
                    weight=mask[start:stop].to(args.device).float()*(1+2*early[start:stop].to(args.device).float())
                    target=record['target'][None,start:stop].to(args.device)
                    pieces=model.objective_components(c['u_exec'],target,c['delta'],weight[None],u_pre=c['u_pre'])
                    loss=pieces['total'];optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite paired probe')
                    optimizer.step()
                    rows.append(dict(update=update,**item,supervised_ticks=int(mask[start:stop].sum()),recovery_ticks=int(early[start:stop].sum()),
                        loss=float(loss.detach()),exec_fit=float(pieces['exec_fit'].detach()),pre_fit=float(pieces['pre_fit'].detach()),
                        regularizer=float(pieces['regularizer'].detach()),gradient_norm=float(norm),elapsed_s=time.time()-began))
                    if update%20==0:
                        atomic_json(out/'progress.json',dict(stage='probe',group=group,candidate=candidate,update=update,maximum_updates=args.updates,utc=time.time()))
                        print(json.dumps(dict(group=group,candidate=candidate,update=update,elapsed_s=time.time()-began)),flush=True)
            elapsed=time.time()-began;destination=out/group/candidate;destination.mkdir(parents=True,exist_ok=True)
            saved=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},updates=args.updates,
                residual_version=model.version,residual_loss_version=(contract or {}).get('version'),
                configuration=dict(cfg,loss_contract=contract,disposable=True,diagnostic_source_checkpoint_sha256=m['models'][group]['checkpoint_sha256']),
                disposable=True,formal_optimizer_updates=0,probe_optimizer_updates=args.updates)
            torch.save(saved,destination/'probe.pt')
            with (destination/'training.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            metrics=evaluate(model,records,m,group,out,candidate)
            result=dict(updates=len(rows),wall_s=elapsed,supervised_ticks=sum(r['supervised_ticks']for r in rows),
                recovery_initial48_ticks=sum(r['recovery_ticks']for r in rows),resources=telemetry.summary(),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,checkpoint=str(destination/'probe.pt'),checkpoint_sha256=sha256(destination/'probe.pt'),
                all_losses_finite=True,metrics=metrics)
            all_candidates[group][candidate]=result
            atomic_json(out/'probe_results.json',dict(started_utc=started,updated_utc=time.time(),groups=all_candidates,formal_optimizer_updates=0,disposable=True))
            del model,optimizer;torch.cuda.empty_cache()
    gates=[]
    def mass(rows,metric,predicate):
        return sum(r[metric]/(7 if r['part']=='Arm' else 22)*.5 for r in rows if predicate(r))
    for group in GROUPS:
        original=read(out/group/'probe_initial/summary.json')['rows'];old=all_candidates[group]['old_loss']['metrics'];new=all_candidates[group]['preclip_loss']['metrics']
        train=lambda r:r['partition']=='train';dev=lambda r:r['partition']=='dev';core=lambda r:r['stratum']=='core'
        pre=mass(new,'normalized_weighted_pre_error_mass',train)/mass(original,'normalized_weighted_pre_error_mass',train)
        dev_ratio=mass(new,'normalized_weighted_exec_error_mass',dev)/mass(old,'normalized_weighted_exec_error_mass',dev)
        core_ratio=mass(new,'normalized_weighted_exec_error_mass',core)/mass(old,'normalized_weighted_exec_error_mass',core)
        restored=sum(r['new_loss_lost_coordinates']for r in new)==0
        gates.append(dict(group=group,train_pre_mse_ratio_vs_initial=pre,dev_exec_mse_ratio_vs_old=dev_ratio,
            core_exec_mse_ratio_vs_old=core_ratio,u_pre_dead_gradient_restored=restored,
            pass_all=pre<=.99 and dev_ratio<=1.02 and core_ratio<=1.05 and restored))
    atomic_json(out/'probe_gates.json',dict(gates=gates,eligible_for_reviewed_formal_switch=all(g['pass_all']for g in gates),
        no_closed_loop_improvement_claim=True,remaining_issue='Formal rear training still uses parameter-stale detached TBPTT histories; both probes use rebuilt current-model prefixes.'))
    summarize(args)
    atomic_json(out/'progress.json',dict(stage='probe',state='COMPLETE',finished_utc=time.time()))

def audit(args):
    """CPU readback of real transition provenance and current-weight history.

    Uses the diagnosis manifest without resampling, rebuilding caches or touching
    formal checkpoints. Two disposable single-update probes quantify carried
    TBPTT state drift; they are not a formal round or recovery-success test.
    """
    import importlib.util
    from types import SimpleNamespace as NS
    import h5py
    from visuotactile.position_control import processed_expert_target
    from visuotactile.narrow_data import record, actual_configuration_labels

    if args.device != 'cpu':
        raise ValueError('History/control audit is a bounded CPU-only readback')
    ROOT=Path(args.output).resolve()
    M=read(ROOT/'manifest.json')
    if Path(M['root']).resolve()!=Path(args.root).resolve():
        raise ValueError('Audit manifest belongs to a different formal run')
    def check(eq,label):
     if not bool(eq): raise AssertionError(label)
    def tensor(x):return torch.from_numpy(x[:])
    def maxerr(a,b):return float((a-b).abs().max())
    def flat(x):
     if isinstance(x,dict):
      for k,v in x.items():
       for sub,val in flat(v):yield (k+'/'+sub).rstrip('/'),val
     else:yield '',x
    spec=importlib.util.spec_from_file_location('_diagnostic_native_action',Path(__file__).resolve().parents[1]/'isaacsimenvs/tasks/play/utils/action_utils.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    report=dict(implementation_sha256=sha256(__file__), version='current_checkpoint_history_control_audit_v1',created_utc=time.time(),manifest_sha256=sha256(ROOT/'manifest.json'),source_scope='Same fixed six complete episodes; no test22 reads; CPU only',transitions=[],models={},parity=[],stale_state_probe=[],formal_updates=0,kit_starts=0)
    for item in M['selection']:
     e=item['source_entry'];key,meta,actual=record(e['path'],episode=e['episode'],privileged=False);q,qreport=actual_configuration_labels(e)
     a=dict(identity=item['identity'],partition=item['partition'],kind=item['kind'],ticks=len(q),q_after_before_exact=True,pre_action_target_observation_exact=True,issued_history_exact=True,physics_substeps_exact=True,u_expert_q_reached_distinct_max_rad=qreport['command_reached_max_abs_rad'])
     with h5py.File(e['path']) as f:
      g=f['episodes'][e['episode']];tr=g['transition'];obs=g['obs'];n=len(q)
      check(torch.equal(tensor(tr['time_before_s']),tensor(obs['time_s'])),'clock')
      if item['kind']=='correction':
       d=tr['dagger'];mask=tensor(d['expert_label_valid']).bool();src=tensor(d['control_source']);indices=tensor(d['expert_delay_index']).long();steps=tensor(d['expert_source_step']).long();qlen=tensor(d['expert_queue_length']);take=tensor(d['takeover_tick']);expected=(src==1)&(torch.arange(n)>=take+qlen-1)&(steps>=take)
       check(torch.equal(mask,expected),'takeover validity');check(not mask[:int(take[0])+int(qlen[0])-1].any(),'switch masked')
       clipped=tensor(d['expert_clipped_action']);raw=tensor(d['expert_raw_action']);delay=tensor(d['expert_delayed_action']);before=tensor(tr['target_before_rad']);target=tensor(tr['target_after_rad']);native=tensor(d['expert_native_target_rad'])
       check(torch.equal(raw.clamp(-1,1),clipped),'native raw mapped');check(torch.equal(steps,(torch.arange(n)-indices).clamp_min(0)),'delay source index');check(torch.equal(delay,clipped[steps]),'queue action')
       check(torch.equal(indices,tensor(tr['expert_bridge_delay_rng_draw'])),'one delay draw')
       check(torch.equal(tensor(d['previous_issued']),before),'previous target');check(torch.equal(tensor(d['u_expert'])[mask],target[mask]),'expert command label');check(torch.equal(tensor(d['q_reached'])[mask],q[mask]),'reached label')
       for name in ['position_requested_rad','position_applied_rad']:check(torch.equal(tensor(tr[name]),target),name)
       for name in ['actual_issued','actual_applied']:check(torch.equal(tensor(d[name]),target),name)
       check(torch.equal(tensor(tr['control_trace/applied_position_target_rad']),target),'executor control trace')
       check(not tensor(tr['control_trace/delay_enabled']).any(),'executor duplicate delay')
       limits=actual['joint_limits'];act=meta['config']['action'];p=NS(device='cpu',num_envs=n,step_dt=meta['control_dt'],_perm_canon_to_lab=torch.arange(29),_arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),_arm_lower=limits[:7,0],_arm_upper=limits[:7,1],_hand_lower=limits[7:,0],_hand_upper=limits[7:,1],_prev_targets=before.clone(),_cur_targets=before.clone(),cfg=NS(action=NS(**{k:act[k] for k in ['dof_speed_scale','arm_moving_average','hand_moving_average']}),domain_randomization=NS(use_action_delay=False,action_delay_max=0)))
       mod.apply_action_pipeline(p,delay);native_error=maxerr(p._cur_targets,native);check(native_error<=1e-6,'native mapped rad math')
       progressed,mapped=processed_expert_target(before,native,dict(arm=.5,hand=1.,hand_target_cap_rad_s=2.),1/60)
       check(torch.equal(mapped,tensor(d['expert_mapped_target_rad'])),'once speed');check(torch.equal(mapped[src.bool()],target[src.bool()]),'actual expert mapping');projected=project_processed_position_targets(target,before,limits,meta['execution_adapter']);check(torch.equal(projected,target),'shared execution feasible')
       a.update(valid_expert_ticks=int(mask.sum()),switch_tick=int(take[0]),first_valid_tick=int(mask.nonzero()[0]),delay_queue=int(qlen[0]),raw_to_native_mapping_max_rad=native_error,source_delay_exact=True,position_executor_second_delay=False,expert_postprocessing_once_exact=True,expert_labels_feasible=True,physical_submission_matches_recorded_target=True)
     for group,ident in M['models'].items():
      cp=item['cache'][group];r=torch.load(cp['path'],map_location='cpu',weights_only=False);check(sha256(cp['path'])==cp['sha256'],'cache SHA');check(torch.equal(r['robot'],actual['robot']),'cache factual robot');check(torch.equal(r['time'],actual['time']),'cache clock');check(torch.equal(r['target'][r['supervision_mask']],actual['command'][r['supervision_mask']]),'cache command target')
      if group=='QG01':check(torch.equal(r['q_reached'][r['supervision_mask']],q[r['supervision_mask']]),'Q reached clean label')
      check(r['cache_identity']['base_checkpoint_sha256']==ident['dp_sha256'],'base paired');check(r['cache_identity']['data_sha256']==ident['data_sha256'],'data paired');check(r['cache_identity']['residual_statistics_sha256']==ident['command_statistics_sha256'],'command stats')
      cur=torch.arange(len(q))%12;check(torch.equal(r['plan_cursor'],cur),'cursor');check(torch.equal(r['plan_valid'],cur[:,None]+torch.arange(24)[None]<24),'plan mask');check(torch.equal(r['base'],r['plans'][:,0]),'nominal firstplan');check(torch.equal(r['previous_nominal'][1:],r['base'][:-1]),'prev nominal');check(not r['previous_nominal_valid'][0] and not r['robot'][0,176] and torch.equal(r['robot'][1:,116:145],actual['command'][:-1]),'t0/issued history')
     a['cache_actual_history_and_label_checks_both_groups']=True;report['transitions'].append(a)
     print('transition',a['identity'],flush=True)
    atomic_json(ROOT/'history_control_audit.json',report)
    for group,ident in M['models'].items():
     check(sha256(ident['checkpoint'])==ident['checkpoint_sha256'],'rear SHA');check(sha256(ident['dp_checkpoint'])==ident['dp_sha256'],'DP SHA')
     payload=torch.load(ident['checkpoint'],map_location='cpu',weights_only=False);c=payload['configuration'];check(c['base_sha256']==ident['dp_sha256'],'rear dp binding');check(c['execution_adapter']==ident['execution_adapter'],'adapter');check(sha256(c['residual_statistics_path'])==ident['command_statistics_sha256'],'stats sha')
     model=load_residual(payload,'cpu').eval();stats=torch.load(c['residual_statistics_path'],map_location='cpu',weights_only=False)
     for k in ['robot_mean','robot_std','action_scale','action_offset']:check(torch.equal(getattr(model,k),stats[k]),'buffer '+k)
     report['models'][group]=dict(frontend_sha256=ident['dp_sha256'],backend_sha256=ident['checkpoint_sha256'],statistics_sha256=ident['command_statistics_sha256'],execution_adapter=ident['execution_adapter'],binding_passed=True)
     for item in M['selection']:
      r=torch.load(item['cache'][group]['path'],map_location='cpu',weights_only=False);n=len(r['robot']);start=time.time()
      with torch.no_grad():
       full,fullstate=model(residual_batch([r],0,n,'cpu'));parts=[];state=None
       for t in range(n):
        raw,state=model(residual_batch([r],t,t+1,'cpu'),state);parts.append(raw)
       single=torch.cat(parts,dim=1);chunks=[];cs=None
       for t in range(0,n,32):
        v,cs=model(residual_batch([r],t,min(n,t+32),'cpu'),cs);chunks.append(v)
       chunk=torch.cat(chunks,dim=1)
       action,_=model.compose(r['base'][None],full,r['gate'][None],r['limits'],previous=r['robot'][None,:,58:87]);single_action,_=model.compose(r['base'][None],single,r['gate'][None],r['limits'],previous=r['robot'][None,:,58:87])
      er=maxerr(full,single);ce=maxerr(full,chunk);check(er<2e-5 and ce<2e-5,'real full/chunk/tick parity')
      state_errors={k:maxerr(v,dict(flat(state))[k]) for k,v in flat(fullstate) if v.is_floating_point()};check(max(state_errors.values())<2e-5,'state parity')
      row=dict(group=group,identity=item['identity'],partition=item['partition'],ticks=n,raw_full_vs_tick_max_abs=er,raw_full_vs_32chunk_max_abs=ce,exec_command_full_vs_tick_max_rad=maxerr(action,single_action),state_max_abs=max(state_errors.values()),t0_included=True,replan_boundaries=n//12,whole_prefix_replayed=True,wall_s=time.time()-start)
      report['parity'].append(row);print('parity',group,item['identity'],er,flush=True)
     # Quantify the exact training approximation on the first selected training correction.
     item=next(x for x in M['selection'] if x['kind']=='correction' and x['partition']=='train');r=torch.load(item['cache'][group]['path'],map_location='cpu',weights_only=False);model=load_residual(payload,'cpu').train();start=(item['first_supervised_tick']//32)*32;end=start+32
     with torch.no_grad():
      state=None
      for t in range(0,start,32):_,state=model(residual_batch([r],t,min(start,t+32),'cpu'),state)
     raw,old_state=model(residual_batch([r],start,end,'cpu'),state);action,delta=model.compose(r['base'][None,start:end],raw,r['gate'][None,start:end],r['limits'],previous=r['robot'][None,start:end,58:87]);mask=r['supervision_mask'][None,start:end].float();early=torch.arange(start,end)<item['first_supervised_tick']+48;mask=mask*torch.where(early[None],3.,1.);loss=model.objective(action,r['target'][None,start:end],delta,mask)
     opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();detach=lambda x:{k:detach(v) for k,v in x.items()} if isinstance(x,dict) else x.detach()
     with torch.no_grad():
      stale,_=model(residual_batch([r],end,end+32,'cpu'),detach(old_state));freshstate=None
      for t in range(0,end,32):_,freshstate=model(residual_batch([r],t,min(end,t+32),'cpu'),freshstate)
      fresh,_=model(residual_batch([r],end,end+32,'cpu'),freshstate)
      sa,_=model.compose(r['base'][None,end:end+32],stale,r['gate'][None,end:end+32],r['limits'],previous=r['robot'][None,end:end+32,58:87]);fa,_=model.compose(r['base'][None,end:end+32],fresh,r['gate'][None,end:end+32],r['limits'],previous=r['robot'][None,end:end+32,58:87])
     report['stale_state_probe'].append(dict(group=group,identity=item['identity'],single_disposable_updates=1,update_window=[start,end],comparison_window=[end,end+32],raw_max_abs=maxerr(stale,fresh),raw_rms=float((stale-fresh).square().mean().sqrt()),exec_command_max_rad=maxerr(sa,fa),scope='One fresh-optimizer exact formal LR/update on real current checkpoint. State carried from pre-update model vs rebuilt current-weight prefix. This quantifies mismatch, not task-failure causality.'))
     atomic_json(ROOT/'history_control_audit.json',report)
    report.update(completed_utc=time.time(),passed=True,disposable_updates=2,recovery_closed_loop=dict(completed=False,kit_starts=0,reason='No validated complete switch-state restore in existing collector. Initial export explicitly lacks PhysX solver/contact/broadphase state; observable restore also lacks sensor/history/observation-delay state. RecordedReplayEnv is native-event replay, not validated complete-policy takeover-state reconstruction. Prefix replay state drift was not measured in this CPU audit; active formal Kit collection not interrupted.',conclusion='Offline history parity and real labels do not prove physical recovery ability.'))
    atomic_json(ROOT/'history_control_audit.json',report);print('COMPLETE',flush=True)


def summarize(args):
    """Aggregate existing numerical diagnostics; performs no inference/updates."""
    out=Path(args.output).resolve();rows=[]
    for group in GROUPS:
        for candidate in ('probe_initial','old_loss','preclip_loss'):
            values=read(out/group/candidate/'summary.json')['rows']
            strata=sorted(set((r['partition'],r['stratum'],r['part'])for r in values))
            for partition,stratum,part in strata:
                selected=[r for r in values if (r['partition'],r['stratum'],r['part'])==(partition,stratum,part)]
                total=lambda key:sum(r[key]for r in selected)
                n=total('joint_coordinates');errorful=total('errorful_coordinates');moving=total('moving_coordinates')
                rows.append(dict(group=group,candidate=candidate,partition=partition,stratum=stratum,part=part,
                    ticks=total('ticks'),joint_coordinates=n,
                    pre_rmse_rad=(total('pre_sse_rad')/n)**.5,exec_rmse_rad=(total('exec_sse_rad')/n)**.5,
                    hold_rmse_rad=(total('hold_sse_rad')/n)**.5,errorful_coordinates=errorful,
                    exec_only_lost_coordinates=total('lost_coordinates'),
                    combined_fit_lost_coordinates=total('new_loss_lost_coordinates' if candidate=='preclip_loss' else 'lost_coordinates'),
                    exec_only_lost_fraction_errorful=total('lost_coordinates')/max(errorful,1),
                    moving_coordinates=moving,direction_correct_fraction=total('direction_correct_coordinates')/max(moving,1),
                    label_unreachable_coordinates=total('label_projection_changed_coordinates'),
                    range_exceeded_coordinates=total('theoretical_range_exceeded_coordinates'),saturated_coordinates=total('saturated_coordinates'),
                    weighted_normalized_exec_error_mass=total('normalized_weighted_exec_error_mass'),
                    weighted_normalized_pre_error_mass=total('normalized_weighted_pre_error_mass')))
    with (out/'probe_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def publish(args):
    """Publish numerical evidence and one figure, never weights or raw caches."""
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(args.output).resolve()
    destination=Path(args.public_output).resolve();destination.mkdir(parents=True,exist_ok=True)
    required=('manifest.json','diagnosis.json','probe_results.json','probe_gates.json',
        'probe_preregistration.json','probe_comparison.csv','history_control_audit.json','state_only_replay_check.json')
    optional=('focused_tests.json','runtime_snapshot.json','test_results.json')
    published=[]
    for name in required+optional:
        path=source/name
        if not path.exists():
            if name in required:raise FileNotFoundError(path)
            continue
        shutil.copyfile(path,destination/name)
        published.append(dict(path=name,sha256=sha256(path),bytes=path.stat().st_size))
    for group in GROUPS:
        for candidate in ('initial','probe_initial','old_loss','preclip_loss'):
            folder=destination/group/candidate;folder.mkdir(parents=True,exist_ok=True)
            for name in ('metrics.csv','training.csv'):
                path=source/group/candidate/name
                if path.exists():
                    target=folder/name;shutil.copyfile(path,target)
                    published.append(dict(path=str(target.relative_to(destination)),sha256=sha256(path),bytes=path.stat().st_size))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    gates={g['group']:g for g in read(source/'probe_gates.json')['gates']}
    for col,group in enumerate(GROUPS):
        ax=axes[0,col]
        for candidate,label,color in [('old_loss','Exec-only','#3366aa'),('preclip_loss','Exec + preclip','#cc6633')]:
            with (source/group/candidate/'training.csv').open() as stream:rows=list(csv.DictReader(stream))
            updates=np.array([int(r['update'])for r in rows]);loss=np.array([float(r['exec_fit'])for r in rows])
            ax.plot(updates[5:],np.convolve(loss,np.ones(6)/6,mode='valid'),label=label,color=color)
        ax.set_title(('A (QG01)' if group=='QG01' else 'B (G01)')+' paired 200-update probe')
        ax.set_xlabel('Diagnostic optimizer updates');ax.set_ylabel('Execution fit (6-update moving mean)');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
        ax=axes[1,col];gate=gates[group]
        ratios=[gate['dev_exec_mse_ratio_vs_old'],gate['core_exec_mse_ratio_vs_old']]
        bars=ax.bar(['Correction dev','Core train'],ratios,color=['#cc6633','#b78b54'])
        ax.axhline(1,color='black',lw=1,label='Old-loss endpoint')
        ax.scatter([0,1],[1.02,1.05],marker='_',s=350,color='#3366aa',label='Pre-registered maximum')
        for bar,ratio in zip(bars,ratios):ax.text(bar.get_x()+bar.get_width()/2,ratio+.025,f'{ratio:.3f}x',ha='center')
        ax.set_ylim(0,max(ratios)*1.2);ax.set_ylabel('Execution MSE: new / old');ax.legend(loc='lower right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same source checkpoint, order, optimizer and current-weight causal prefixes\nTraining windows vary; endpoint comparisons use the same fixed full episodes. No closed-loop claim.',fontsize=10)
    figure=destination/'probe_comparison.png';fig.savefig(figure,dpi=160);plt.close(fig)
    published.append(dict(path=figure.name,sha256=sha256(figure),bytes=figure.stat().st_size))
    atomic_json(destination/'publication.json',dict(published_utc=time.time(),implementation_sha256=sha256(Path(__file__)),
        source=str(source),files=published,excluded=['checkpoints','raw HDF5','full per-tick NPZ plans/command chains','cache payloads','videos'],
        full_numeric_chain_local_index=[dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)for path in sorted(source.glob('*/*/*.npz'))],
        figure_definition='Top: six-update moving means of actual exec_fit, matched prescribed variable windows. Bottom: final fixed-episode command-normalized recovery-weighted equal Arm/Hand MSE new/old ratios. The legacy pre_fit=0 log placeholder is never plotted.'))
    print(json.dumps(dict(published=str(destination),files=len(published))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['diagnose','probe','audit','summarize','publish']);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--updates',type=int,default=200);parser.add_argument('--public-output',default='research_docs/evidence/tight_gradient_20261112')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.mode=='diagnose':diagnose(args)
    elif args.mode=='audit':audit(args)
    elif args.mode=='summarize':summarize(args)
    elif args.mode=='publish':publish(args)
    else:probe(args)


if __name__=='__main__':main()
