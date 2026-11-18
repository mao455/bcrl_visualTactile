"""Small current-checkpoint gradient attribution and actual-history shift diagnostics.

No optimizer is constructed. These gradients cannot reconstruct old training
phase gradients: Adam, clipping and EMA are nonlinear across historical updates.
"""
import argparse, collections, time
import numpy as np
import torch
from tools.tight_representation_audit import RUN,PUBLIC,ROOTS,read,write,sha,load,table,quant,check_resources


def gradients():
    from visuotactile.models.native_dp import load_residual
    from visuotactile.native_data import residual_batch
    from visuotactile.geometry_sampling import STAGES
    from tools.tight_representation_probe import source
    from tools.tight_fresh_visual_prefix_probe import cache_binding,admit_population
    began=time.time();rows=[];selections=[]
    for cohort,root in ROOTS.items():
        data=read(root/'A/state_data/data.json');pop=admit_population(data,read(root/'B/state_data/data.json'))
        groups={'original':sorted(pop['original'])[:1],'correction':sorted(pop['corrections'])[:1],'validation':sorted(pop['validation'])[:1]}
        for role in 'AB':
            model=load_residual(load(root/role/f'backend/{role}/training/best.pt'));model.requires_grad_(True);model.eval()
            index_path,_,_=cache_binding(root/role,sha(root/'B/state_data/statistics.pt'))
            parameters=list(model.named_parameters())
            for kind,ids in groups.items():
                for identity in ids:
                    check_resources();r=load(index_path.parent/(identity+'.pt'));s=source(data['entries'][identity]);seen=set();state=None
                    selections.append(dict(model=cohort+'_'+role,identity=identity,kind=kind,selection='lexicographic first, before metrics'))
                    for start in range(0,len(r['robot']),32):
                        inputs=residual_batch([r],start,min(start+32,len(r['robot'])),'cpu');n=inputs['robot'].shape[1]
                        masks={stage:torch.from_numpy(s['legal'][start:start+n]&(s['phase'][start:start+n]==k))[None] for k,stage in enumerate(STAGES) if stage not in seen}
                        masks={k:v for k,v in masks.items() if v.any()}
                        with torch.set_grad_enabled(bool(masks)):
                            raw,next_state=model(inputs,state)
                            if masks:
                                chain=model.compose_chain(r['base'][None,start:start+n],raw,r['gate'][None,start:start+n],r['limits'],previous=inputs['robot'][...,58:87])
                                for stage,mask in masks.items():
                                    components=model.objective_components(chain['u_exec'],r['target'][None,start:start+n],chain['delta'],mask,u_pre=chain['u_pre'],execution_bounds=chain['execution_bounds'])
                                    # Exact fixed128 scalar measure, one observed segment per phase.
                                    loss=components['total']*mask.sum()/128
                                    gs=torch.autograd.grad(loss,[p for _,p in parameters],retain_graph=True,allow_unused=True)
                                    squared=collections.Counter()
                                    for (name,_),g in zip(parameters,gs):
                                        if g is not None:squared[name.split('.')[0]]+=float(g.square().sum())
                                    total=sum(squared.values());row=dict(model=cohort+'_'+role,kind=kind,identity=identity,phase=stage,start=start,ticks=int(mask.sum()),loss_fixed128=float(loss.detach()),gradient_norm=total**.5)
                                    row.update({k+'_gradient_norm':squared[k]**.5 for k in ('visual','tactile','motion','memory','transformer','policy')});rows.append(row);seen.add(stage)
                        def detach(x):return {k:detach(v) for k,v in x.items()} if isinstance(x,dict) else x.detach()
                        state=detach(next_state)
            print('gradient',cohort,role,len(rows),flush=True)
    table(PUBLIC/'current_phase_gradients.csv',rows)
    write(PUBLIC/'current_gradient_scope.json',dict(seconds=time.time()-began,segments=len(rows),selections=selections,policy_updates=0,
        scope='Current selected rear FY training objective, exact fixed128 scaling, first32tick block containing each phase of first original/correction/validation record. Current-weight complete prefix, gradient32tick only, no optimizer/clipping. Module norms not additive attribution fractions; not historical training gradients or a causal capability test.'))


def shift():
    from sklearn.neighbors import NearestNeighbors
    from visuotactile.geometry_sampling import STAGES
    rows=[]
    for cohort in ROOTS:
        for role in 'AB':
            name=cohort+'_'+role;meta=read(RUN/f'features_{name}.json');z=dict(np.load(RUN/f'features_{name}.npz'));n=len(z['truth'])
            family=np.empty(n,object);kind=np.empty(n,object);split=np.empty(n,object)
            for e in meta['episodes']:
                sl=slice(e['offset'],e['offset']+e['count']);family[sl]=e['family'];kind[sl]=e['kind'];split[sl]=e['split']
            original=(kind=='original')&(split=='train')
            for method in ('proprio','controller_feedback178','visual','sensor_history','decoder_history'):
                x=z[method];sd=x[original].std(0);active=sd>1e-6;xn=(x[:,active]-x[original][:,active].mean(0))/sd[active]
                for k,stage in enumerate(STAGES):
                    base=original&(z['phase']==k)
                    if base.sum()<10:continue
                    for pop,query in [('expert_original',base),('student_prefix',(split=='train')&z['student']&(z['phase']==k)),('correction_legal',(split=='train')&(kind=='correction')&z['legal']&(z['phase']==k)),('independent_expert',(split=='validation')&(z['phase']==k))]:
                        distances=[]
                        for f in sorted(set(family[query])):
                            tr=base&(family!=f);va=query&(family==f)
                            if not tr.any():continue
                            d,_=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(xn[tr]).kneighbors(xn[va]);distances.extend((d[:,0]/np.sqrt(active.sum())).tolist())
                        if distances:rows.append(dict(model=name,method=method,phase=stage,population=pop,frames=len(distances),rms_standardized_distance_median=float(np.median(distances)),rms_standardized_distance_p95=float(np.percentile(distances,95))))
    table(PUBLIC/'history_shift.csv',rows)
    write(PUBLIC/'history_shift_scope.json',dict(scope='Recorded genuine student prefixes versus original expert histories, same phase, nearest other family. Original-only normalization, remove constant channels; unitless RMS feature distance. Different prefix lengths/phase occupancy remain; no expert counterfactual labels before takeover. Not freshly generated closed-loop histories or proof of irreducible aliasing.',policy_updates=0))

@torch.no_grad()
def dp_interventions():
    from visuotactile.models.native_dp import load_policy
    from visuotactile.native_data import Episode
    rows=[]
    for cohort,root in ROOTS.items():
        for role in 'AB':
            check_resources();folder=root/role;saved=load(folder/'training/best.pt');model=load_policy(saved)
            assert model.modality=='G00' and model.horizon==24 and model.contract['visual_stride']==1
            data=read(folder/'state_data/data.json');numeric=load(folder/'state_data/numeric.pt')['records']
            ids=[sorted(data['normalization_train_ids'])[0],sorted(data['splits']['validation'])[0]]
            for identity in ids:
                e=Episode(folder/'state_data',identity,model.group);r=numeric[identity];normal=None
                for intervention in ('normal','rgb_patches_zero','depth_and_validity_zero','both_zero'):
                    memory=None;conditions=[]
                    for t in range(25):
                        frame=e.frame(t,'cpu')
                        if intervention in ('rgb_patches_zero','both_zero'):frame['patches']=torch.zeros_like(frame['patches'])
                        if intervention in ('depth_and_validity_zero','both_zero'):frame['depth']=torch.zeros_like(frame['depth'])
                        memory=model.observe(r['robot'][t],r['time'][t],t,frame,memory)
                        if t%12==0:conditions.append(model.condition(memory,r['time'][t]))
                    conditions=torch.cat(conditions)
                    plans=model.predict(conditions,torch.Generator().manual_seed(20261119),inference_steps=16)
                    if intervention=='normal':normal=(conditions,plans);continue
                    rows.append(dict(model=cohort+'_'+role,identity=identity,intervention=intervention,observed_ticks=25,planning_ticks='0,12,24',condition_rms_delta=float((conditions-normal[0]).square().mean().sqrt()),unbounded_plan_rms_delta_rad=float((plans-normal[1]).square().mean().sqrt())))
            print('DP intervention',cohort,role,flush=True)
    table(PUBLIC/'dp_interventions.csv',rows)
    write(PUBLIC/'dp_intervention_scope.json',dict(policy_updates=0,seed=20261119,scope='First25 actual ticks of first original and first validation trajectory, identical DDIM16 noise in every arm, zeroing actual cached RGB patch/depth features. Demonstrates numerical DP dependence only; off-manifold interventions and short prefix do not identify causal task benefit. G00 DP has no tactile input; tactile is in rear.'))


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['gradients','shift','dp_interventions']);a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    {'gradients':gradients,'shift':shift,'dp_interventions':dp_interventions}[a.stage]()
