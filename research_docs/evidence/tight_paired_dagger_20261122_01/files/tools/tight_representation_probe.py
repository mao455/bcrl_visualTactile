"""Bounded CPU readouts of actual frozen policy features, with family-held-out probes.

Privileged states occur only in diagnostic labels or explicitly named oracle
readouts. Formal student weights, inputs, data and manifests are never written.
"""
import argparse, collections, contextlib, time
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation
from tools.tight_representation_audit import RUN,PUBLIC,ROOTS,read,write,sha,load,table,quant,check_resources
from tools.tight_fresh_position_probe import fit as _fit,predict

def fit(x,y):
    parameters,receipt=_fit(x,y)
    receipt["feature_standardization"]="Only the current fold training families; mean and population std; constant scale1; float64 ridge solve."
    return parameters,receipt


def relative(a,b):
    ra=Rotation.from_quat(a[:,[4,5,6,3]]);rb=Rotation.from_quat(b[:,[4,5,6,3]])
    p=ra.inv().apply(b[:,:3]-a[:,:3]);mat=(ra.inv()*rb).as_matrix()
    return np.concatenate((p,mat[:,:,:2].reshape(-1,6)),1)


def source(entry):
    from scripts.tight_balance_20261117.diagnose import phases
    with h5py.File(entry['path']) as f:
        g=f['episodes'][entry['episode']];t=g['transition'];obs=g['obs']
        assert np.array_equal(obs['time_s'][:],t['time_before_s'][:])
        b=t['diagnostics_before'];palm=b['palm_body_pose_w'][:];obj=b['object_pose_w'][:];hole=b['hole_pose_w'][:]
        po=relative(palm,obj);ho=relative(hole,obj);n=len(obj)
        # Read previous completed object-filtered forces; t0 has no recorded prior transition.
        force=np.concatenate((np.zeros((1,5)),np.linalg.norm(t['object_contact_force_w'][:-1,-1],axis=-1)),0)
        contacts=force>.1
        trend=np.zeros((n,3));trend[:-12]=(po[12:,:3]-po[:-12,:3])/.2
        future_loss=np.zeros(n)
        for j in range(n-12):future_loss[j]=float(contacts[j].sum()>=2 and (contacts[j+1:j+13].sum(-1)==0).any())
        y=np.concatenate((po,ho,trend,contacts.astype(float),np.log1p(force),future_loss[:,None]),1)
        valid=np.arange(n)<n-12;valid[:1]=False
        legal=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool) if 'dagger' in t else np.ones(n,bool)
        student=t['dagger/control_source'][:].reshape(-1)==0 if 'dagger' in t else np.zeros(n,bool)
        onset=np.zeros(n,bool)
        if 'dagger' in t:onset[np.flatnonzero(legal)[:24]]=True
        return dict(y=y.astype(np.float32),phase=phases(t),legal=legal,onset=onset,future_valid=valid,
                    support_eligible=contacts.sum(-1)>=2,student=student,command=t['target_after_rad'][:],q_reached=t['diagnostics_after/joint_position_rad'][:],
                    state_hash=__import__('hashlib').sha256(y.tobytes()).hexdigest())


@torch.no_grad()
def extract_forward(model,r,*,capture=False,intervention=None):
    from visuotactile.native_data import residual_batch
    chunks=[];features=collections.defaultdict(list);state=None;handles=[]
    if capture:
        for name in ('visual','motion','memory'):
            handles.append(getattr(model,name).register_forward_hook(lambda m,a,o,n=name:features[n].append(o.detach().cpu())))
        handles.append(model.tactile.register_forward_hook(lambda m,a,o:features['touch'].append(o[0][:,1::2].detach().cpu().flatten(0,1))))
        handles.append(model.transformer.register_forward_hook(lambda m,a,o:features['decoder'].append(o.detach().cpu().flatten(0,1))))
    try:
        for start in range(0,len(r['robot']),32):
            stop=min(start+32,len(r['robot']));inputs=residual_batch([r],start,stop,'cpu')
            if intervention=='visual_zero':inputs['visual']=torch.zeros_like(inputs['visual'])
            if intervention=='visual_reverse':inputs['visual']=r['visual'].flip(0)[None,start:stop]
            raw,state=model(inputs,state,mask_force=intervention=='force_zero')
            action,_=model.compose(r['base'][None,start:stop],raw,r['gate'][None,start:stop],r['limits'],previous=inputs['robot'][...,58:87]);chunks.append(action[0])
    finally:
        for h in handles:h.remove()
    return torch.cat(chunks),{k:torch.cat(v) for k,v in features.items()}


def extract():
    from visuotactile.models.native_dp import load_residual
    from tools.tight_fresh_visual_prefix_probe import cache_binding,admit_population
    began=time.time();deadline=began+45*60;receipts=[];errors=[];norms=[];ablations=[]
    # First two sorted originals, first two corrections and all six independent validation trajectories.
    # Selection is registered before inspecting metrics; all other records still receive normal inference.
    for cohort,root in ROOTS.items():
        data=read(root/'A/state_data/data.json');pop=admit_population(data,read(root/'B/state_data/data.json'))
        ids=data['splits']['train']+data['splits']['validation'];original=set(pop['original']);validation=set(pop['validation'])
        raw={i:source(data['entries'][i]) for i in ids}
        ablate=set(sorted(original)[:2]+sorted(pop['corrections'])[:2]+sorted(validation))
        for role in 'AB':
            check_resources();p=root/role/f'backend/{role}/training/best.pt';model=load_residual(load(p));index_path,index,binding=cache_binding(root/role,sha(root/'B/state_data/statistics.pt'))
            allfeatures=[];alltruth=[];allmeta=[];offset=0;chosen=[]
            for j,i in enumerate(ids):
                if time.time()>deadline:raise TimeoutError('Fixed45min CPU extraction window')
                check_resources();path=index_path.parent/(i+'.pt');assert sha(path)==index['episodes'][i]['sha256'];r=load(path);assert r['cache_identity']==binding
                s=raw[i];n=len(r['robot']);assert n==len(s['y']) and np.array_equal(r['supervision_mask'].numpy(),s['legal'])
                action,f=extract_forward(model,r,capture=True)
                force=r['tactile']['force_w_N'].reshape(n,2,5,3)[:,-1].flatten(1)
                local=r['tactile']['force_local_N'].reshape(n,2,5,3)[:,-1].flatten(1)
                touch=torch.cat((force,local),-1);touch=touch.sign()*torch.log1p(touch.abs())
                robot=(r['robot']-model.robot_mean)/model.robot_std
                robot[:,116:176]=torch.where(r['robot'][:,176:177].bool(),robot[:,116:176],0.);robot[:,176]=r['robot'][:,176]
                proprio=robot[:,:58] # q and qdot only: exclude recorded expert target/action history shortcuts.
                modalities=dict(proprio=proprio,controller_feedback178=robot,visual=r['visual'],touch_proprio=torch.cat((proprio,touch),-1),
                    vision_touch_proprio=torch.cat((r['visual'],proprio,touch),-1),
                    vision_touch_feedback=torch.cat((r['visual'],robot,touch),-1),
                    learned_touch_proprio=torch.cat((proprio,f['touch']),-1),
                    sensor_history=torch.cat((f['visual'],f['touch'],f['motion'],f['memory']),-1),decoder_history=f['decoder'])
                # Fixed6tick stride bounds probe solve cost; full-history inference is still every tick.
                take=np.arange(0,n,6);take=take[s['future_valid'][take]]
                chosen.append(dict(identity=i,family=data['entries'][i]['family'],split='validation' if i in validation else 'train',
                    kind='original' if i in original else 'validation' if i in validation else 'correction',offset=offset,count=len(take),ticks=take.tolist(),
                    source_sha256=data['entries'][i]['source_sha256'],cache_sha256=index['episodes'][i]['sha256'],label_sha256=s['state_hash']))
                offset+=len(take);allfeatures.append({k:v[take].float().numpy() for k,v in modalities.items()});alltruth.append(s['y'][take]);allmeta.append({k:s[k][take] for k in ('phase','legal','student','onset','support_eligible','command')})
                masks=dict(legal=s['legal'],first24=s['onset'],student_prefix=s['student'])
                from visuotactile.geometry_sampling import STAGES
                for k,stage in enumerate(STAGES):masks[stage]=s['legal']&(s['phase']==k)
                from visuotactile.position_control import reachable_position_bounds
                bounds=reachable_position_bounds(r['robot'][:,58:87],r['limits'],model.execution_adapter)
                span=(bounds[...,1]-bounds[...,0]).clamp_min(1e-6)
                truth=torch.from_numpy(s['command']);target=torch.from_numpy(s['q_reached'] if role=='A' else s['command'])
                for metric,pred,ref in [('DP_own_target',r['base'],target),('DP_command',r['base'],truth),('composed_command',action,truth)]:
                    for stage,m in masks.items():
                        if not m.any():continue
                        if stage=='student_prefix':continue # no counterfactual expert label here
                        err=(pred-ref)[m];errors.append(dict(model=cohort+'_'+role,identity=i,family=data['entries'][i]['family'],split='validation' if i in validation else 'train',kind='original' if i in original else 'correction' if i not in validation else 'validation',stage=stage,metric=metric,ticks=len(err),arm_squared_sum=float(err[:,:7].square().sum()),hand_squared_sum=float(err[:,7:].square().sum()),arm_span_squared_sum=float((err[:,:7]/span[m,:7]).square().sum()),hand_span_squared_sum=float((err[:,7:]/span[m,7:]).square().sum())))
                norms.append(dict(model=cohort+'_'+role,identity=i,robot_abs_p99=float(torch.quantile(robot.abs().flatten(),.99)),robot_abs_max=float(robot.abs().max()),over5_fraction=float((robot.abs()>5).float().mean()),visual_norm=float(r['visual'].norm(dim=-1).mean()),touch_norm=float(f['touch'].norm(dim=-1).mean())))
                if i in ablate:
                    for intervention in ('force_zero','visual_zero','visual_reverse'):
                        other,_=extract_forward(model,r,intervention=intervention);delta=other-action
                        m=s['legal'];normal_error=(action-truth)[m];changed_error=(other-truth)[m]
                        ablations.append(dict(model=cohort+'_'+role,identity=i,split='validation' if i in validation else 'train',intervention=intervention,
                            ticks=n,command_delta_rms_rad=float(delta.square().mean().sqrt()),command_delta_max_rad=float(delta.abs().max()),
                            legal_ticks=int(m.sum()),normal_arm_mse=float(normal_error[:,:7].square().mean()),changed_arm_mse=float(changed_error[:,:7].square().mean()),normal_hand_mse=float(normal_error[:,7:].square().mean()),changed_hand_mse=float(changed_error[:,7:].square().mean())))
                if j%20==0:print('extract',cohort,role,j,len(ids),round(time.time()-began,1),flush=True)
            arrays={k:np.concatenate([x[k] for x in allfeatures]) for k in allfeatures[0]};arrays['truth']=np.concatenate(alltruth)
            arrays.update({k:np.concatenate([x[k] for x in allmeta]) for k in allmeta[0]});np.savez_compressed(RUN/f'features_{cohort}_{role}.npz',**arrays)
            write(RUN/f'features_{cohort}_{role}.json',dict(episodes=chosen,binding=binding,rear_checkpoint_sha256=sha(p),dimensions={k:v.shape[1] for k,v in arrays.items() if v.ndim==2},sample_stride=6))
            receipts.append(dict(model=cohort+'_'+role,episodes=len(ids),probe_ticks=len(arrays['truth']),binding=binding,rear_sha256=sha(p)))
    table(PUBLIC/'offline_errors.csv',errors);table(PUBLIC/'normalization.csv',norms);table(PUBLIC/'interventions.csv',ablations)
    write(PUBLIC/'feature_extraction.json',dict(models=receipts,seconds=time.time()-began,policy_updates=0,cuda_initialized=torch.cuda.is_initialized(),
        scope='Every native tick advances actual learned sensor/Transformer states; fixed stride6 only for probes. Offline DP plans are matched frozen cached predictions. Force-zero and visual-zero/reversal are fixed recorded-history interventions, not autonomous benefits. Prefix expert labels unavailable: excluded from action error.'))


def rotation_error(pred,true):
    def matrix(x):
        a=x.reshape(-1,3,2);u=a[:,:,0];u=u/np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-12);v=a[:,:,1]-(u*a[:,:,1]).sum(-1,keepdims=True)*u;v=v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12);return np.stack((u,v,np.cross(u,v)),axis=-1)
    a=matrix(pred);b=matrix(true);cos=np.clip((np.einsum('nij,nij->n',a,b)-1)/2,-1,1);return np.degrees(np.arccos(cos))


def scores(y,p,eligible):
    from sklearn.metrics import roc_auc_score,average_precision_score
    result={}
    for name,sl,mult in [('object_palm_mm',slice(0,3),1000),('object_hole_mm',slice(9,12),1000),('relative_motion_m_s',slice(18,21),1)]:
        result[name]=float(np.sqrt(np.mean(np.sum((y[:,sl]-p[:,sl])**2,axis=-1)))*mult)
    for name,sl in [('object_palm_deg',slice(3,9)),('object_hole_deg',slice(12,18))]:result[name]=float(np.mean(rotation_error(p[:,sl],y[:,sl])))
    result['finger_force_log1p_rmse']=float(np.sqrt(np.mean((p[:,26:31]-y[:,26:31])**2)))
    for label,yt,pt in [('finger_contact',y[:,21:26].flatten(),p[:,21:26].flatten()),('support_loss12',y[eligible,31],p[eligible,31])]:
        result[label+'_n']=len(yt);result[label+'_positives']=int(yt.sum());result[label+'_brier']=float(np.mean((np.clip(pt,0,1)-yt)**2)) if len(yt) else None
        result[label+'_auc']=float(roc_auc_score(yt,pt)) if len(np.unique(yt))==2 else None
        result[label+'_ap']=float(average_precision_score(yt,pt)) if len(np.unique(yt))==2 else None
    return result


def probes():
    began=time.time();rows=[];fold_receipts=[];ambiguity=[]
    for cohort in ROOTS:
        for role in 'AB':
            name=cohort+'_'+role;meta=read(RUN/f'features_{name}.json');z=dict(np.load(RUN/f'features_{name}.npz'));n=len(z['truth'])
            family=np.empty(n,object);kind=np.empty(n,object);split=np.empty(n,object)
            for e in meta['episodes']:
                s=slice(e['offset'],e['offset']+e['count']);family[s]=e['family'];kind[s]=e['kind'];split[s]=e['split']
            train=split=='train';validation=~train;families=sorted(set(family[train]));assert len(families)==30 and not set(family[train])&set(family[validation])
            # Six untouched-by-probe validation families, plus fixed five out-of-family correction folds.
            folds=[('independent6',train,validation)]+[(f'family_fold{k}',train&~np.isin(family,families[k::5]),train&np.isin(family,families[k::5])) for k in range(5)]
            y=z['truth'];methods=['proprio','controller_feedback178','visual','touch_proprio','vision_touch_proprio','vision_touch_feedback','learned_touch_proprio','sensor_history','decoder_history']
            for method in methods:
                x=z[method];oof=np.zeros_like(y);oof_valid=np.zeros(n,bool)
                for fold,tr,va in folds:
                    assert not set(family[tr])&set(family[va]);param,info=fit(torch.from_numpy(x[tr]),torch.from_numpy(y[tr]));p=predict(param,torch.from_numpy(x[va])).numpy()
                    oof[va]=p;oof_valid[va]=True
                    if fold=='independent6':
                        rows.append(dict(model=name,method=method,population='independent6_all',frames=int(va.sum()),**scores(y[va],p,z['support_eligible'][va])))
                        # Explicit in-sample error separates probe-fit failure from new-family gap.
                        trainpred=predict(param,torch.from_numpy(x[tr])).numpy();rows.append(dict(model=name,method=method,population='train_fit',frames=int(tr.sum()),**scores(y[tr],trainpred,z['support_eligible'][tr])))
                    fold_receipts.append(dict(model=name,method=method,fold=fold,train_families=sorted(set(family[tr])),validation_families=sorted(set(family[va])),train_frames=int(tr.sum()),validation_frames=int(va.sum()),**info))
                from visuotactile.geometry_sampling import STAGES
                for k,stage in enumerate(STAGES):
                    mask=validation&(z['phase']==k)
                    if mask.any():rows.append(dict(model=name,method=method,population='independent6_'+stage,frames=int(mask.sum()),**scores(y[mask],oof[mask],z['support_eligible'][mask])))
                for pop,mask in [('family_oof_original',train&(kind=='original')),('family_oof_correction_legal',train&(kind=='correction')&z['legal']),('family_oof_student_prefix',train&z['student']),('family_oof_first24',train&z['onset'])]:
                    mask=mask&oof_valid
                    if mask.any():rows.append(dict(model=name,method=method,population=pop,frames=int(mask.sum()),**scores(y[mask],oof[mask],z['support_eligible'][mask])))
                print('probe',name,method,round(time.time()-began,1),flush=True)
            # Privileged current-state action readout is a diagnostic comparator, not a proven upper bound.
            mtrain=train&z['legal'];mval=validation&z['legal'];target=z['command']
            for method in ('proprio','controller_feedback178','vision_touch_proprio','decoder_history','privileged_current_pose'):
                x=(np.concatenate((z['proprio'],y[:,:18],y[:,21:31]),-1) if method=='privileged_current_pose' else z[method])
                par,info=fit(torch.from_numpy(x[mtrain]),torch.from_numpy(target[mtrain]));pred=predict(par,torch.from_numpy(x[mval])).numpy();e=pred-target[mval]
                write(RUN/f'oracle_{name}_{method}.json',dict(arm_rmse_rad=float(np.sqrt(np.mean(e[:,:7]**2))),hand_rmse_rad=float(np.sqrt(np.mean(e[:,7:]**2))),**info,scope='Linear one-step expert-command readout; current exact poses/contact labels only, no future trend/slip labels, no deployment or task performance upper-bound claim.'))
            # Nearest cross-family features with unit-standardized distances; similarity is empirical, not true aliasing.
            from sklearn.neighbors import NearestNeighbors
            for method in ('proprio','visual','vision_touch_proprio','decoder_history'):
                x=z[method];mu=x[mtrain].mean(0);sd=x[mtrain].std(0);sd[sd<1e-6]=1;xn=(x-mu)/sd
                nn=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(xn[mtrain]);dist,ix=nn.kneighbors(xn[mval]);ref=np.flatnonzero(mtrain)[ix[:,0]];q=np.flatnonzero(mval)
                threshold=np.percentile(dist[:,0],25);near=dist[:,0]<=threshold;delta=target[q]-target[ref]
                ambiguity.append(dict(model=name,method=method,comparison='nearest training family for independent validation6 legal ticks',queries=len(q),nearest_quartile_queries=int(near.sum()),distance_threshold=float(threshold),arm_rmse_near=float(np.sqrt(np.mean(delta[near,:7]**2))),hand_rmse_near=float(np.sqrt(np.mean(delta[near,7:]**2))),palm_distance_near_mm=float(np.mean(np.linalg.norm(y[q[near],:3]-y[ref[near],:3],axis=-1))*1000),limitation='Nearest finite observations are not identical histories; cannot prove irreducible perceptual ambiguity.'))
    table(PUBLIC/'probe_metrics.csv',rows);write(PUBLIC/'probe_folds.json',fold_receipts);write(PUBLIC/'nearest_observation.json',ambiguity)
    oracle={p.stem:read(p) for p in RUN.glob('oracle_*.json')};write(PUBLIC/'privileged_readouts.json',oracle)
    write(PUBLIC/'probe_scope.json',dict(seconds=time.time()-began,ridge=.01,family_folds=5,heldout_validation_families=6,hyperparameter_search=False,policy_updates=0,
        labels='Current object-root pose relative palm and hole; rotation6. Future12tick relative palm displacement/.2s. Object-filtered previous completed fingertip force/contact. Future loss means currently>=2 object-contact fingertips then zero within12ticks, not true support/slip ground truth.',
        limitations='Validation6 already selected policy checkpoints historically, not untouched final test. Probe folds exclude families but encoders trained on the30 train families. Native timestep autocorrelation persists; no random-tick split. Readout failure does not prove information absence; contact readout from force is largely a sensor sanity check. Hole/object root relative pose is not a calibrated peg-tip/insertion clearance estimate.'))

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    RUN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','probes']);args=p.parse_args()
    (extract if args.stage=='extract' else probes)()
