"""Bounded execution-learning repair; immutable parents and no test22 access."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.execution_learning import (PROJECTION_FY_LOSS, ideal_execution_bounds,
    projection_fy_elements, execution_expressibility, actual_execution_expressibility)
from visuotactile.models.native_dp import TargetResidual
from visuotactile.native_data import residual_batch
from visuotactile.anchor_sampling import supervision_rows,recovery_prefix_rows
from visuotactile.anchor_continue import residual_prefix_state
from tools.tight_gradient_diagnostics import (GROUPS,read,model_and_records,probe_sequence,evaluate,fit)

SOURCE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_gradient_diagnostic_20261112_01')
PARENT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_dagger_matched_20261112_01')
HISTORY=dict(version='current_weight_prefix_v1')


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bind(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'registration.json').exists():raise ValueError('Registered repair identity already exists; never reset deadline')
    m=read(SOURCE/'manifest.json')
    for g,item in m['models'].items():
        for key,digest in [('checkpoint','checkpoint_sha256'),('dp_checkpoint','dp_sha256')]:
            if sha256(item[key])!=item[digest]:raise ValueError('Model SHA drift')
        backend=Path(item['checkpoint']).parents[1]
        for rel,digest in [('nominal/index.json','cache_index_sha256'),('state_data/data.json','data_sha256')]:
            if sha256(backend/rel)!=item[digest]:raise ValueError('Data/cache identity drift')
        for s in m['selection']:
            c=s['cache'][g]
            if sha256(c['path'])!=c['sha256']:raise ValueError('Fixed sample cache drift')
    now=time.time()
    atomic_json(out/'manifest.json',m)
    atomic_json(out/'registration.json',dict(version='execution_learning_repair_v1',started_utc=now,
        absolute_deadline=now+86400,compute_deadline=now+85800,parent_run=str(PARENT),source_manifest_sha256=sha256(SOURCE/'manifest.json'),
        parent_deadline=read(PARENT/'session.json')['deadline'],loss_contract=PROJECTION_FY_LOSS,history_contract=HISTORY,
        output_version=TargetResidual.version,execution_adapter=m['models']['QG01']['execution_adapter'],
        maximum_probe_updates=200,maximum_new_dagger_rounds=0,maximum_collection_opportunities=0,
        proposed_backend_tick_epochs=6,maximum_autonomous_development_per_group=10,
        resource_policy='full_gpu_v1',min_gpu_free_mib=6144,min_ram_gib=14,min_disk_gib=60,
        test22_reads=0,frontends_frozen=True,formal_retraining_requires_both_probe_gates=True,
        budget='Preparation/waits included; no new GPU work while parent workers/owner active. Bound full backend work by measured replay throughput; never extend 24h.'))
    return m


def active_parent_processes():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
            if '-m' not in args or '--root' not in args:continue
            module=args[args.index('-m')+1];root=Path(args[args.index('--root')+1]).resolve()
            if not (root==PARENT or PARENT in root.parents):continue
            if module not in {'tools.tight_finish','visuotactile.anchor_continue','visuotactile.state_resources','tools.tight_dagger'}:continue
            rows.append(dict(pid=int(p.name),module=module,cwd=os.readlink(p/'cwd'),root=str(root)))
        except (OSError,ValueError,IndexError):pass
    return rows


def runtime_snapshot(out):
    session=read(PARENT/'session.json');freeze=read(PARENT/'source_freeze.json')
    # session.cwd was inherited from palm_eval; use the actual owner's cwd.
    matches=active_parent_processes()
    owners=[p for p in matches if p['module']=='tools.tight_finish']
    tree=Path(owners[0]['cwd']) if owners else Path('/home/CNS2025331827/docs/bc+rl_RSS_dagger_perf')
    mismatches=[p for p,h in freeze.items() if not (tree/p).exists() or sha256(tree/p)!=h]
    status=read(PARENT/'status.json')
    atomic_json(out/'runtime_snapshot.json',dict(utc=time.time(),parent_state=status['state'],parent_stage=status['stage'],
        runtime_source=session['source_head'],runtime_tree=str(tree),recorded_session_cwd=session['cwd'],live_processes=matches,freeze_count=len(freeze),source_mismatches=mismatches,
        runtime_tree_head=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip(),
        editing_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        remote_head=subprocess.check_output(['git','ls-remote','origin','refs/heads/feature/tight-state-foundation'],text=True).split()[0],
        parent_deadline=session['deadline'],compute_deadline=session['compute_deadline']))
    if mismatches:raise ValueError('Parent frozen source drift')


def audit(out):
    m=read(out/'manifest.json');rows=[];parameter_rows=[]
    for group in GROUPS:
        model,records,payload=model_and_records(m,group)
        for selection in m['selection']:
            identity=selection['identity'];record=records[identity]
            # Reuse recorded actual forward results; no whole-dataset inference.
            values=np.load(SOURCE/group/'initial'/(identity+'.npz'))
            raw=torch.from_numpy(values['raw']).requires_grad_();base=torch.from_numpy(values['base'])
            target=record['target'];valid,_=supervision_rows(record);early=recovery_prefix_rows(record,48) if selection['kind']=='correction' else torch.zeros_like(valid)
            weight=valid.float()*(1+2*early.float());p=record['robot'][:,58:87]
            bounds=ideal_execution_bounds(p,record['limits'],model.execution_adapter)
            chain=model.compose_chain(base,raw,None,record['limits'],previous=p)
            torch.testing.assert_close(chain['u_exec'],torch.from_numpy(values['u_exec']),atol=0,rtol=0)
            fy,roundoff=projection_fy_elements(chain['u_pre'],target,bounds,weight)
            terms=dict(old=(fit((chain['u_exec']-target)*model.action_scale)*weight).sum()/weight.sum(),
                failed_pre=(fit((chain['u_pre']-target)*model.action_scale)*weight).sum()/weight.sum(),
                fy=((.5*(fy*model.action_scale.square())[...,:7].mean(-1)+.5*(fy*model.action_scale.square())[...,7:].mean(-1))*weight).sum()/weight.sum(),
                regularizer=(fit(chain['delta'])*weight).sum()/weight.sum()*1e-4)
            gradients={k:torch.autograd.grad(v,(chain['u_pre'],raw),retain_graph=True,allow_unused=True) for k,v in terms.items()}
            categories=execution_expressibility(base,model.scales,model.action_scale,target,bounds)
            exact_categories=actual_execution_expressibility(base,model.scales,model.action_scale,target,bounds,record['limits'])
            ideal=torch.maximum(torch.minimum(chain['u_pre'],bounds[...,1]),bounds[...,0])
            assert (ideal-chain['u_exec']).abs().max()<=1.01e-6
            strata={'core':valid} if selection['kind']=='core' else {'recovery_initial48':valid&early,'recovery_later':valid&~early}
            for stratum,mask in strata.items():
                for part,sl,tol in [('Arm',slice(0,7),1e-4),('Hand',slice(7,29),1e-3)]:
                    error=(chain['u_exec']-target)[mask,sl];gpre,gz=gradients['fy'];oldpre=gradients['old'][0]
                    local=lambda g:g[mask,sl]/(weight[mask,None]/weight.sum()).clamp_min(1e-30)
                    wrong=error.abs()>tol;correct=~wrong;exact=error==0
                    row=dict(group=group,identity=identity,partition=selection['partition'],stratum=stratum,part=part,
                        coordinates=error.numel(),errorful=int(wrong.sum()),old_flat_errorful=int((wrong&(local(oldpre).abs()<=1e-10)).sum()),
                        fy_flat_errorful=int((wrong&(local(gpre).abs()<=1e-10)).sum()),
                        wrong_direction=int((wrong&(local(gpre)*error<0)).sum()),
                        raw_tanh_saturated=int((raw[mask,sl].tanh().abs()>=.95).sum()),
                        label_roundoff_max_rad=float(roundoff[mask,sl].max()) if mask.any() else 0,
                        actual_vs_ideal_max_rad=float((ideal-chain['u_exec'])[mask,sl].abs().max()) if mask.any() else 0)
                    for k,v in categories.items():row[k]=int(v[mask,sl].sum())
                    for k,v in exact_categories.items():row['actual_'+k]=int(v[mask,sl].sum())
                    row['fy_flat_logit_errorful']=int((wrong&(gz[mask,sl].abs()<=1e-10)).sum())
                    row['legacy_exceeded_but_stable']=int((categories['legacy_pre_range_exceeded']&categories['stable'])[mask,sl].sum())
                    for name,(gp,glogit) in gradients.items():
                        row[name+'_logit_l2']=float(glogit[mask,sl].norm())
                        row[name+'_logit_l2_correct_within_tol']=float(glogit[mask,sl][correct].norm())
                        row[name+'_logit_l2_exact_exec']=float(glogit[mask,sl][exact].norm())
                    rows.append(row)
        # Actual shared parameter gradients, with identical current-weight prefixes.
        for selected in [s for s in m['selection'] if s['partition']=='train'][:3:2]:
            r=records[selected['identity']];mask,_=supervision_rows(r);start=0 if selected['kind']=='core' else int(torch.where(mask)[0][0])
            stop=min(start+32,len(mask));weight=mask[start:stop].float()
            if selected['kind']!='core':weight*=1+2*recovery_prefix_rows(r,48)[start:stop].float()
            model.loss_contract=PROJECTION_FY_LOSS;model.train()
            prefix=[residual_batch([r],t,min(t+32,start),'cpu') for t in range(0,start,32)]
            state=residual_prefix_state(model,prefix);inp=residual_batch([r],start,stop,'cpu');raw,_=model(inp,state)
            c=model.compose_chain(r['base'][None,start:stop],raw,None,r['limits'],previous=inp['robot'][...,58:87])
            target=r['target'][None,start:stop];pieces=model.objective_components(c['u_exec'],target,c['delta'],weight[None],u_pre=c['u_pre'],execution_bounds=c['execution_bounds'])
            pre=(fit((c['u_pre']-target)*model.action_scale)*weight[None]).sum()/weight.sum()
            terms={'old':pieces['exec_fit'],'failed_pre':pre,'fy':pieces['fy_fit'],'regularizer':pieces['regularizer']}
            params=[p for p in model.parameters() if p.requires_grad]
            vectors={}
            for name,loss in terms.items():
                grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                vectors[name]=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(params,grad)])
            report=dict(group=group,identity=selected['identity'],start=start,stop=stop,parameters=sum(p.numel() for p in params))
            for name,v in vectors.items():report[name+'_parameter_l2']=float(v.norm())
            for a,b in [('old','failed_pre'),('old','fy'),('failed_pre','fy'),('regularizer','fy')]:
                report[a+'_'+b+'_cosine']=float(torch.nn.functional.cosine_similarity(vectors[a][None],vectors[b][None]))
            # Same initial parameters, fresh matching optimizer, one actual update.
            for name in ['old','failed_pre','fy']:
                model.load_state_dict(payload['model']);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                vector=vectors['old']+vectors['failed_pre']+vectors['regularizer'] if name=='failed_pre' else vectors[name]+vectors['regularizer']
                cursor=0
                for p in params:p.grad=vector[cursor:cursor+p.numel()].reshape_as(p).clone();cursor+=p.numel()
                norm=torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                with torch.no_grad():
                    state=residual_prefix_state(model,prefix);z,_=model(inp,state);cc=model.compose_chain(r['base'][None,start:stop],z,None,r['limits'],previous=inp['robot'][...,58:87])
                    error=(fit((cc['u_exec']-target)*model.action_scale)*weight[None]).sum()/weight.sum()
                report[name+'_preclip_parameter_norm']=float(norm);report[name+'_one_step_exec_mse']=float(error)
            report['initial_exec_mse']=float(terms['old'].detach());parameter_rows.append(report)
            model.load_state_dict(payload['model'])
    write_csv(out/'execution_diagnostic.csv',rows);atomic_json(out/'parameter_gradients.json',parameter_rows)
    summary=[]
    for group in GROUPS:
        for part in ['Arm','Hand']:
            rr=[r for r in rows if r['group']==group and r['part']==part]
            keys=['coordinates','errorful','old_flat_errorful','fy_flat_errorful','wrong_direction','unreachable','limit_only','finite_saturated','stable','legacy_pre_range_exceeded','legacy_exceeded_but_stable','raw_tanh_saturated','actual_unreachable','actual_limit_only','actual_finite_saturated','actual_stable','fy_flat_logit_errorful']
            summary.append(dict(group=group,part=part,**{k:sum(r[k] for r in rr) for k in keys},label_roundoff_max_rad=max(r['label_roundoff_max_rad'] for r in rr),actual_vs_ideal_max_rad=max(r['actual_vs_ideal_max_rad'] for r in rr)))
    atomic_json(out/'execution_summary.json',dict(rows=summary,test22_reads=0,scope='Same six fixed complete trajectories; ideal-box classification, actual v1 execution reported separately',shared_parameters=parameter_rows))
    print(json.dumps(summary,indent=2),flush=True)


def paired_probe(out):
    from visuotactile.state_resources import lease_for
    m=read(out/'manifest.json');registration=read(out/'registration.json')
    if (out/'probe_preregistration.json').exists():raise ValueError('No automatic repeat of a partially/completely used probe budget')
    _,reference,_=model_and_records(m,GROUPS[0]);sequence=probe_sequence(m,reference,200)
    prereg=dict(updates_each=200,sequence=sequence,seed=2026111301,learning_rate=3e-4,weight_decay=1e-4,
        gradient_norm_clip=1.,optimizer='fresh AdamW for both arms; no formal optimizer state inherited',
        loss_candidates=['old_loss','projection_fy'],history_contract=HISTORY,selection='fixed update200; no dev selection',
        gates=dict(dev_exec_ratio_max=1.02,core_exec_ratio_max=1.05,recovery_hand_direction_decrease_max=.01,
            saturation_fraction_increase_max=.01),test22_reads=0,started_utc=time.time())
    atomic_json(out/'probe_preregistration.json',prereg);results={};lease=lease_for(out,registration['compute_deadline'])
    for group in GROUPS:
        _,records,payload=model_and_records(m,group);results[group]={}
        for candidate in ['old_loss','projection_fy']:
            torch.manual_seed(prereg['seed']);torch.cuda.manual_seed_all(prereg['seed'])
            contract=None if candidate=='old_loss' else PROJECTION_FY_LOSS
            with lease.block():
                model=TargetResidual(payload['model'],payload['model']['scales'],execution_adapter=payload['configuration']['execution_adapter'],
                    loss_contract=contract,history_contract=HISTORY).cuda();model.load_state_dict(payload['model'])
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rows=[];began=time.time()
            for update,item in enumerate(sequence,1):
                with lease.block():
                    if active_parent_processes():raise RuntimeError('Parent process returned during repair GPU stage')
                    r=records[item['identity']];a,b=item['start'],item['stop'];model.train()
                    prefix=[residual_batch([r],t,min(t+32,a),'cuda:0') for t in range(0,a,32)]
                    state=residual_prefix_state(model,prefix);inp=residual_batch([r],a,b,'cuda:0');z,_=model(inp,state)
                    c=model.compose_chain(r['base'][None,a:b].cuda(),z,None,r['limits'].cuda(),previous=inp['robot'][...,58:87])
                    mask,_=supervision_rows(r);early=recovery_prefix_rows(r,48) if item['bank']!='core' else torch.zeros_like(mask)
                    w=mask[a:b].float()*(1+2*early[a:b].float())
                    kw={'execution_bounds':c['execution_bounds']} if contract else {}
                    parts=model.objective_components(c['u_exec'],r['target'][None,a:b].cuda(),c['delta'],w[None].cuda(),u_pre=c['u_pre'],**kw)
                    opt.zero_grad(set_to_none=True);parts['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm) or not torch.isfinite(parts['total']):raise ValueError('Nonfinite short optimization')
                    opt.step()
                rows.append(dict(update=update,**item,supervised_ticks=int(mask[a:b].sum()),recovery_ticks=int((early&mask)[a:b].sum()),
                    exec_fit=float(parts['exec_fit'].detach()),main_fit=float(parts.get('fy_fit',parts['exec_fit']).detach()),
                    regularizer=float(parts['regularizer'].detach()),parameter_gradient_norm=float(norm),elapsed_s=time.time()-began))
                if update%20==0:atomic_json(out/'status.json',dict(state='RUNNING',stage='paired_probe',group=group,candidate=candidate,updates=update,utc=time.time()))
            destination=out/group/candidate;destination.mkdir(parents=True,exist_ok=True)
            cfg=dict(payload['configuration'],loss_contract=contract,history_contract=HISTORY,disposable=True)
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},updates=200,residual_version=model.version,
                residual_loss_version=(contract or {}).get('version'),residual_history_version=HISTORY['version'],configuration=cfg,
                contract=payload['contract'],disposable=True),destination/'probe.pt')
            write_csv(destination/'training.csv',rows)
            # Existing evaluator reports actual v1 execution, never the FY surrogate.
            with lease.block():metrics=evaluate(model,records,m,group,out,candidate)
            results[group][candidate]=dict(updates=200,wall_s=time.time()-began,optimization_wall_s=rows[-1]['elapsed_s'],
                supervised_ticks=sum(r['supervised_ticks'] for r in rows),recovery_ticks=sum(r['recovery_ticks'] for r in rows),
                checkpoint_sha256=sha256(destination/'probe.pt'),metrics=metrics)
            atomic_json(out/'probe_results.json',dict(groups=results,utc=time.time(),disposable=True))
            del model,opt;torch.cuda.empty_cache()
    def mass(rows,predicate):return sum(r['normalized_weighted_exec_error_mass']*.5/(7 if r['part']=='Arm' else 22) for r in rows if predicate(r))
    def direction(rows):
        r=[x for x in rows if x['partition']=='train' and x['stratum']=='recovery_initial48' and x['part']=='Hand']
        return sum(x['direction_correct_coordinates'] for x in r)/max(1,sum(x['moving_coordinates'] for x in r))
    gates=[]
    for group in GROUPS:
        old=results[group]['old_loss']['metrics'];new=results[group]['projection_fy']['metrics']
        dev=mass(new,lambda r:r['partition']=='dev')/mass(old,lambda r:r['partition']=='dev')
        core=mass(new,lambda r:r['stratum']=='core')/mass(old,lambda r:r['stratum']=='core')
        movement=direction(new)-direction(old)
        saturation=(sum(r['saturated_coordinates'] for r in new)-sum(r['saturated_coordinates'] for r in old))/sum(r['joint_coordinates'] for r in old)
        gates.append(dict(group=group,dev_exec_ratio=dev,core_exec_ratio=core,recovery_hand_direction_difference=movement,
            saturation_fraction_difference=saturation,pass_all=dev<=1.02 and core<=1.05 and movement>=-.01 and saturation<=.01))
    atomic_json(out/'probe_gates.json',dict(gates=gates,passed=all(r['pass_all'] for r in gates),
        interpretation='Finite optimization diagnostic; no independent generalization or autonomous success claim'))


def prepare_formal(out):
    from tools.tight_finish import residual_prepare
    from tools.tight_dagger import reuse_nominal_cache, share_monitor
    m=read(out/'manifest.json');reg=read(out/'registration.json');probe=read(out/'probe_results.json')
    if not read(out/'probe_gates.json')['passed']:raise ValueError('Both candidate optimization gates required')
    # Measured per-tick throughput includes current-parameter prefix replay.
    # Double it for full-episode length/batching uncertainty; reserve 2h eval+I/O.
    rates=[r['optimization_wall_s']/r['supervised_ticks'] for g in probe['groups'].values() for r in g.values()]
    remaining=reg['compute_deadline']-time.time()-7200
    # Use the latest completed joint parent pair and its complete immutable pool.
    # Never mix a newer frontend with a previous-round rear/cache.
    candidates=[]
    for directory in sorted(PARENT.glob('round_[0-9][0-9]')):
        if all((directory/g/'backend'/g/'training/exposure_endpoint.json').exists() and
               (directory/g/'backend/evaluation_complete.json').exists() and
               read(directory/g/'backend/evaluation_complete.json').get('complete') for g in GROUPS):
            candidates.append(directory)
    if not candidates:raise ValueError('No complete parent A/B pair for formal continuation')
    selected=candidates[-1];formal_models={}
    for group in GROUPS:
        ck=selected/group/'backend'/group/'training/best.pt'
        payload=torch.load(ck,map_location='cpu',weights_only=False);cfg=payload['configuration']
        formal_models[group]=dict(checkpoint=str(ck),checkpoint_sha256=sha256(ck),
            dp_checkpoint=cfg['base_checkpoint'],dp_sha256=cfg['base_sha256'],command_statistics_sha256=cfg['residual_statistics_sha256'])
        if sha256(cfg['base_checkpoint'])!=cfg['base_sha256']:raise ValueError('Formal DP source SHA drift')
    atomic_json(out/'formal_sources.json',dict(parent_round=str(selected),models=formal_models,
        diagnostic_round=str(PARENT/'round_01'),same_probe_initial_pair=selected.name=='round_01',
        interpretation='Short probe tests the loss on pinned round01; formal continuation reuses the latest complete pair, with its own baseline validation.'))
    # Recompute population/time admission against the actual formal caches.
    population={g:read(Path(formal_models[g]['checkpoint']).parents[1]/'nominal/scales.json')['control_tick_population'] for g in GROUPS}
    choices=[e for e in [6,3,1] if sum(population.values())*e*max(rates)*2<remaining]
    if not choices:raise TimeoutError('Latest pool cannot fit matched exposure and reserved evaluation')
    epochs=choices[0]
    atomic_json(out/'formal_budget.json',dict(registered_utc=time.time(),epochs_each=epochs,population=population,
        conservative_training_estimate_seconds=sum(population.values())*epochs*max(rates)*2,
        seconds_per_supervised_tick=rates,evaluation_io_reserve_seconds=7200,
        compute_deadline=reg['compute_deadline'],source_pool=str(selected/'snapshot.json'),frontend_updates=0,
        selection='common full exposure endpoint; validation best retained; no selection by precommand MSE'))
    shared=None
    for group in GROUPS:
        item=formal_models[group];old=Path(item['checkpoint']).parents[2]
        cfg=copy.deepcopy(read(old/'configuration.json'));v=cfg['residual_pair']
        data=read(old/'state_data/data.json');index=read(old/group/'nominal/index.json')
        if set(index['episodes'])&set(data['splits']['test']):raise ValueError('Test payloads cannot enter reuse')
        split_identity=(data['snapshot_id'],data['splits'])
        if shared is None:shared=split_identity
        elif shared!=split_identity:raise ValueError('A/B immutable source snapshots differ')
        v.update(loss_contract=PROJECTION_FY_LOSS,history_contract=HISTORY,
            maximum_control_tick_epochs=epochs,minimum_control_tick_epochs=epochs,
            comparison_control_tick_epochs=[epochs],validation_epoch_interval=epochs,
            initialization_seed=2026111301,videos_per_group=1,
            continuation_from=dict(v['continuation_from'],path=item['checkpoint'],sha256=item['checkpoint_sha256']),
            dagger_evaluation=dict(initial_trials=10,max_trials=10,expand_if_success=1,expand_if_subgoal=1,expand_if_native_lift=2))
        v.pop('residual_scale_adaptation',None)
        dest=out/'formal'/group/'backend';residual_prepare(dest,configuration=cfg)
        share_monitor(out,dest)
        reuse_nominal_cache(old/group/'nominal',dest/group/'nominal',v['bases'][group],dest/'state_data',item['command_statistics_sha256'])
        shutil.copy2(old/group/'nominal/scales.json',dest/group/'nominal/scales.json')
        shutil.copy2(PARENT/'evaluation_plan.json',dest/'evaluation_plan.json')
        for path in [dest/'session.json',dest/group/'session.json']:
            session=read(path);session.update(resource_root=str(out),matrix_owner=str(out),matrix_deadline=reg['compute_deadline'],
                started=reg['started_utc'],deadline=reg['absolute_deadline'],training_deadline=reg['compute_deadline']-7200,
                offline_deadline=reg['compute_deadline']-3600,evaluation_deadline=reg['compute_deadline'],
                source_head=read(out/'launch.json')['source_head'],cwd=str(Path.cwd()))
            atomic_json(path,session)
        atomic_json(dest/'cache_complete.json',dict(complete=True,exact_reuse=True,source=str(old/group/'nominal'),new_frontend_updates=0))


def source_valid(out):
    freeze=read(out/'source_freeze.json')
    if any(sha256(p)!=h for p,h in freeze.items()):raise ValueError('Repair runtime source drift')


def run_child(out,name,args,deadline):
    from tools.runtime import RUN,kill_owned,proc
    source_valid(out)
    receipt=out/(name+'_receipt.json')
    if receipt.exists():raise ValueError('Stage already attempted; no silent repeat: '+name)
    if time.time()>=deadline:raise TimeoutError('No stage budget remaining')
    with (out/(name+'.log')).open('a') as log:
        child=subprocess.Popen(RUN+args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(out/'active_stage.json',dict(name=name,pid=child.pid,process=proc(child.pid),deadline=deadline,args=args))
        code=None
        try:code=child.wait(timeout=deadline-time.time())
        finally:
            kill_owned(child)
            atomic_json(receipt,dict(name=name,exit_code=child.returncode,finished_utc=time.time(),args=args,deadline=deadline))
    if proc(child.pid) is not None:raise RuntimeError('Owned child exit not verified: '+name)
    if code!=0:raise RuntimeError('Bounded stage failed: '+name)


def export_small_artifacts(out,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gradients=read(out/'parameter_gradients.json')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    x=np.linspace(-.5,1.5,1001);y=.99;p=np.clip(x,0,1)
    for values,label in [(2*(p-y)*((x>0)&(x<1)),'Executed MSE'),
                         (2*(x-y),'Precommand MSE'),(2*(p-y),'2 x projection FY')]:
        axes[0].plot(x,values,label=label)
    axes[0].set(xlabel='Precommand x (rad)',ylabel='Main loss derivative',title='Near-boundary label y=0.99; box [0,1]')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    ticks=np.arange(len(gradients))
    for shift,key,label in [(-.25,'failed_pre','Precommand term'),(0,'fy','FY'),(.25,'regularizer','Regularizer')]:
        axes[1].bar(ticks+shift,[r[key+'_parameter_l2']/r['old_parameter_l2'] for r in gradients],width=.25,label=label)
    axes[1].set(yscale='log',ylabel='Parameter gradient norm / old main norm',title='Four real blocks; matched current-weight prefixes')
    axes[1].set_xticks(ticks,[r['group']+' '+('Core' if r['start']==0 else 'Recovery') for r in gradients],rotation=15,fontsize=8)
    axes[1].axhline(1,color='black',lw=.7);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(target/'loss_and_parameter_gradients.png',dpi=150);plt.close(fig)
    artifacts=[]
    for group in GROUPS:
        root=out/'formal'/group/'backend'/group
        for relative in ['training/finished.json','training/validation.json','training/curves.png',
                         'training/exposure_endpoint.json','offline/stage_errors.csv','offline/stage_errors.png',
                         'evaluation_summary.json']:
            source=root/relative
            if source.is_file():
                if source.stat().st_size>2*1024**2:raise ValueError('Public report too large: '+str(source))
                name=group+'_'+relative.replace('/','_');shutil.copy2(source,target/name)
                artifacts.append(dict(local=str(source),public=name,sha256=sha256(source)))
        for source in sorted((root/'evaluation').glob('**/*')):
            if source.is_file() and source.suffix in {'.mp4','.npz'}:
                artifacts.append(dict(local=str(source),public=None,bytes=source.stat().st_size,sha256=sha256(source)))
    atomic_json(target/'artifact_index.json',dict(artifacts=artifacts,large_artifacts_remain_local=True))


def update_public(out):
    """Small public evidence only; source models/NPZ/video never enter Git."""
    target=Path('research_docs/evidence/tight_execution_learning_20261113');target.mkdir(exist_ok=True)
    names=['registration.json','runtime_snapshot.json','execution_summary.json','execution_diagnostic.csv',
        'parameter_gradients.json','cpu_tests.json','launch.json','status.json','formal_sources.json','formal_budget.json','probe_gates.json','results.json','monitor_exit.json']
    for name in names:
        p=out/name
        if p.exists():
            if p.is_symlink() or p.stat().st_size>2*1024**2:raise ValueError('Unexpected public evidence')
            shutil.copy2(p,target/name)
    export_small_artifacts(out,target)
    if (out/'probe_results.json').exists():
        r=read(out/'probe_results.json');small={g:{k:{key:value for key,value in v.items() if key!='metrics'} for k,v in arms.items()} for g,arms in r['groups'].items()}
        atomic_json(target/'probe_exposure.json',small)
    state=read(out/'status.json') if (out/'status.json').exists() else {'state':'PREPARED','stage':'CPU_complete_GPU_not_started'}
    reg=read(out/'registration.json')
    from datetime import datetime,timezone,timedelta
    deadline=datetime.fromtimestamp(reg['absolute_deadline'],timezone(timedelta(hours=8))).isoformat()
    text=('<!-- EXECUTION_LEARNING_STATUS_START -->\n'
        '## 2026-11-13 执行学习修复：机器状态\n\n'
        f"状态 **{state['state']} / {state['stage']}**；新 run `tight_execution_learning_20261113_01`，绝对截止 **{deadline}**。\n"
        '旧 matched run 的冻结源码及原截止不变；本轮只有一个FY候选，先配对200更新，双组门槛通过后才做共同预算后端重训与每组10次完整学生开发验证。\n'
        '短探针固定第一轮；后端正式适配复用旧run最新完整配套前端/缓存与共享数据快照，A仍学习q_reached、B仍学习u_expert。test22隔离；等待/启动不是训练或闭环完成。\n'
        '[数学、执行链与边界](DAGGER_METHOD.md#execution-learning-repair) · [实际状态](evidence/tight_execution_learning_20261113/status.json) · [诊断数值](evidence/tight_execution_learning_20261113/execution_summary.json) · [梯度图](evidence/tight_execution_learning_20261113/loss_and_parameter_gradients.png)。\n'
        '<!-- EXECUTION_LEARNING_STATUS_END -->\n\n')
    for name in ['STATE_DP_RESULTS.md','SESSION_HANDOFF.md','README.md']:
        p=Path('research_docs')/name;old=p.read_text();start=old.find('<!-- EXECUTION_LEARNING_STATUS_START -->');end=old.find('<!-- EXECUTION_LEARNING_STATUS_END -->')
        if start>=0 and end>=start:old=old[:start]+old[end+len('<!-- EXECUTION_LEARNING_STATUS_END -->'):].lstrip('\n')
        p.write_text(text+old)


def publish_after_parent(out):
    """Merge only the parent's bounded report before the authorized feature push."""
    target='feature/tight-state-foundation'
    def git(*args):return subprocess.check_output(['git',*args],text=True,stderr=subprocess.STDOUT,timeout=90).strip()
    if git('status','--porcelain'):raise ValueError('Publication requires clean reviewed source tree')
    git('fetch','origin',target);remote=git('rev-parse','FETCH_HEAD');base=git('merge-base','HEAD',remote)
    changes=git('diff','--name-only',base,remote).splitlines()
    allowed=lambda p:p in {'research_docs/STATE_DP_RESULTS.md','research_docs/CLEANUP_AUDIT.md'} or p.startswith('research_docs/evidence/'+PARENT.name+'/')
    if not all(allowed(p) for p in changes):raise ValueError('Remote changes outside parent publication scope; review needed')
    git('merge','--no-edit',remote)
    git('push','origin','HEAD:'+target)
    atomic_json(out/'source_publication.json',dict(head=git('rev-parse','HEAD'),remote_head=git('ls-remote','origin','refs/heads/'+target).split()[0],utc=time.time()))


def publish_results(out):
    update_public(out)
    paths=['research_docs/evidence/tight_execution_learning_20261113','research_docs/STATE_DP_RESULTS.md','research_docs/SESSION_HANDOFF.md','research_docs/README.md']
    subprocess.run(['git','add','--',*paths],check=True)
    diff=subprocess.run(['git','diff','--cached','--quiet'])
    if diff.returncode==1:subprocess.run(['git','commit','-m','Record bounded execution-learning repair results'],check=True)
    elif diff.returncode!=0:raise RuntimeError('Could not inspect publication index')
    subprocess.run(['git','push','origin','HEAD:feature/tight-state-foundation'],check=True,timeout=90)
    atomic_json(out/'result_publication.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),utc=time.time()))


def run(out):
    import fcntl,traceback
    from tools.runtime import RUN,kill_owned,supervised_owner
    return supervised_owner(_run)(out)


def _run(out):
    import fcntl,traceback
    from tools.runtime import RUN,kill_owned
    lock=(out/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'launch.json').exists():raise ValueError('No implicit restart/deadline reset')
    reg=read(out/'registration.json');deadline=reg['compute_deadline']
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit reviewed repair tree before launch')
    files=subprocess.check_output(['git','ls-files','*.py','visuotactile/run_cpu.sh','configs/tight_state.json'],text=True).splitlines()
    freeze={str(Path(p).resolve()):sha256(p) for p in files};atomic_json(out/'source_freeze.json',freeze)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    atomic_json(out/'launch.json',dict(pid=os.getpid(),source_head=head,cwd=str(Path.cwd()),started_utc=time.time(),absolute_deadline=reg['absolute_deadline']))
    roots=[str(out),*[str(out/'formal'/g/'backend'/g) for g in GROUPS],*[str(out/'formal'/g/'backend') for g in GROUPS]]
    session=dict(resource_policy='full_gpu_v1',resource_target_percent=None,resources=dict(min_gpu_free_mib=6144,min_ram_gib=14,min_disk_gib=60),
        reserve_bytes=60*1024**3,training_deadline=deadline,deadline=reg['absolute_deadline'],owned_worker_roots=roots)
    atomic_json(out/'session.json',session)
    def status(state,stage,**kw):atomic_json(out/'status.json',dict(state=state,stage=stage,utc=time.time(),pid=os.getpid(),**kw))
    monitor=None;published=False
    try:
        status('WAITING','parent_workers_and_terminal_publication',gpu_started=False)
        while True:
            if time.time()>=deadline-3600:raise TimeoutError('Insufficient remaining bounded repair window')
            active=active_parent_processes()
            publication=read(PARENT/'terminal_publication_status.json')
            parent=read(PARENT/'status.json')
            if not active and parent.get('owned_workers_exited') is True and publication['state']=='COMPLETE':break
            if not active and publication['state']=='FAILED':raise RuntimeError('Parent terminal publication failed; preserve and review safe merge')
            time.sleep(15)
        source_valid(out);publish_after_parent(out);published=True
        with (out/'monitor.log').open('a') as log:monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while not (out/'shared_resources.json').exists():
            if monitor.poll() is not None:raise RuntimeError('Resource monitor startup failed')
            if time.time()>=deadline:raise TimeoutError('Monitor startup deadline')
            time.sleep(.2)
        status('RUNNING','paired_probe',gpu_started=True)
        run_child(out,'paired_probe',['-m','tools.tight_execution_learning','probe','--root',str(out)],min(deadline-3600,time.time()+1800))
        if not read(out/'probe_gates.json')['passed']:
            status('COMPLETE','candidate_not_adopted',formal_training=False,closed_loop=False)
            return
        status('RUNNING','formal_preparation');prepare_formal(out);budget=read(out/'formal_budget.json')
        # Both new rear models finish before either enters autonomous evaluation.
        for group in GROUPS:
            dest=out/'formal'/group/'backend';status('RUNNING','backend_training',group=group)
            run_child(out,group+'_training',['-m','visuotactile.anchor_continue','residual-train','--root',str(dest/group)],deadline-7200)
            from tools.tight_dagger import select_exposure_endpoint
            select_exposure_endpoint(dest/group/'training','residual')
            run_child(out,group+'_offline',['-m','visuotactile.anchor_continue','residual-offline','--root',str(dest/group)],deadline-3600)
        # Source populations and sampling schedule are identical; verify actual exposure.
        exposure={g:read(out/'formal'/g/'backend'/g/'training/finished.json') for g in GROUPS}
        for key in ['supervised_ticks','updates']:
            if exposure['QG01'][key]!=exposure['G01'][key]:raise ValueError('Paired actual backend exposure differs: '+key)
        for group in GROUPS:
            dest=out/'formal'/group/'backend';status('RUNNING','autonomous_development',group=group)
            run_child(out,group+'_evaluation',['-m','tools.tight_finish','residual_evaluate','--root',str(dest)],deadline)
            if not read(dest/'evaluation_complete.json')['complete']:raise ValueError('Incomplete fixed evaluation cohort')
        atomic_json(out/'results.json',dict(groups={g:dict(exposure={k:exposure[g].get(k) for k in ['updates','supervised_ticks','control_tick_epochs','target_reached']},
            evaluation=read(out/'formal'/g/'backend'/g/'evaluation_summary.json')) for g in GROUPS},
            test22_used=False,frontend_updates=0,new_dagger_rounds=0,bitwise_physical_pairing_claimed=False))
        status('COMPLETE','bounded_backend_and_autonomous_evaluation',gpu_started=True)
    except BaseException as error:
        status('FAILED','bounded_stage_stopped',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if monitor is not None:
            (out/'STOP_MONITOR').touch()
            try:monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:kill_owned(monitor)
        current=read(out/'status.json');current['owned_workers_exited']=monitor is None or monitor.poll() is not None;atomic_json(out/'status.json',current)
        if published:
            try:publish_results(out)
            except BaseException:atomic_json(out/'publication_error.json',dict(traceback=traceback.format_exc(),utc=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','audit','probe','run','publish-local']);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);args.root=args.root.resolve()
    if args.command=='prepare':bind(args.root);runtime_snapshot(args.root)
    elif args.command=='audit':audit(args.root)
    elif args.command=='probe':paired_probe(args.root)
    elif args.command=='publish-local':update_public(args.root)
    else:run(args.root)

if __name__=='__main__':main()
