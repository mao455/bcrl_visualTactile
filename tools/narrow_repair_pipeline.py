"""One slow-data expansion and one random StateDP contrast after repair gates.

The finite graph has no edge from training/evaluation back to collection.
Old test families are never training; final seeds were frozen before this file ran.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from tools.runtime import CODE, read, write, sha
from tools.narrow_repair import execute, beat, status, protected


def select_speed(rows):
    """Adjacent tested endpoints, real slowing, all three starts successful."""
    import numpy as np
    native={r['family']:r for r in rows if r['alpha']==1.}
    candidates=[]
    for low,high in ((.55,.60),(.60,.65)):
        chosen=[r for r in rows if r['alpha'] in (low,high)]
        if len(chosen)!=6 or not all(r['success'] and r['visual_passed'] for r in chosen):continue
        ratios=[r['duration_s']/native[r['family']]['duration_s'] for r in chosen]
        # Contact motion is from actual measured velocity, not timestamps or a
        # trailing idle interval. Proxy phases are disclosed in each report.
        speed_ratios=[];contact_ratios=[]
        for r in chosen:
            for phase in ('contact_low_object','native_alignment','second_insertion'):
                a=r['motion']['phases'].get(phase);b=native[r['family']]['motion']['phases'].get(phase)
                if a and b and b['arm_actual_speed_rms']>1e-4:
                    speed_ratios.append(a['arm_actual_speed_rms']/b['arm_actual_speed_rms'])
                    contact_ratios.append(a['seconds']/b['seconds'])
        if np.median(ratios)<1.35 or not speed_ratios or np.median(speed_ratios)>.95:continue
        candidates.append(dict(interval=[low,high],successes=6,attempts=6,
            median_duration_s=float(np.median([r['duration_s'] for r in chosen])),
            median_duration_ratio=float(np.median(ratios)),median_contact_arm_speed_ratio=float(np.median(speed_ratios)),
            median_contact_duration_ratio=float(np.median(contact_ratios)),
            scope='Exploratory 3-family paired screen, not a population reliability estimate'))
    if not candidates:return dict(passed=False,reason='No adjacent alpha endpoints show reliable real slowing in this bounded pilot')
    chosen=min(candidates,key=lambda c:abs(c['median_duration_s']-6.))
    return dict(passed=True,selected=chosen,candidates=candidates,policy='One stable alpha per episode, continuous uniform between tested endpoints')


def cohort_results(rows):
    from visuotactile.narrow_decision import failure_stage
    return {split:dict(attempts=len(rr),valid=sum(r['legal'] and r['software_complete'] for r in rr),successes=sum(r['success'] for r in rr),
        software_failures=sum(not r['software_complete'] for r in rr),
        initial_rejections=sum(not r['legal'] for r in rr),failures=dict(Counter(failure_stage(r) for r in rr if r['software_complete'] and not r['success'])))
        for split in ('train','unseen') for rr in [[r for r in rows if r['split']==split]]}


def capture_classification(result, visual):
    """Sensor correctness and natural task outcome are separate gates.

    A falling object may leave a camera. Keep that failed trajectory as failure
    evidence, never as a successful demonstration, without calling it a broken
    sensor. Successful demonstrations retain the same target-visibility gate.
    """
    if not result['software_complete'] or not result['legal']:
        raise RuntimeError('Incomplete capture or initial rejection requires review')
    if not visual['passed']:
        raise RuntimeError('RGB-D content gate failed: '+result['identity'])
    visible=all(all(v['in_frame']) for sample in visual['sampled_target_projection'] for v in sample['views'][:2])
    if result['success'] and not visible:
        raise RuntimeError('Successful capture target region left fixed frame: '+result['identity'])
    return dict(task_failure=not result['success'],fixed_target_visible=visible,
        successful_demonstration=result['success'],sensor_content_passed=True)


def pending_production(jobs, results):
    """Continue exactly the unexecuted suffix, including failures in the prefix."""
    planned=[j['identity'] for j in jobs]
    done=[r['identity'] for r in results if r['identity'] in set(planned)]
    if done!=planned[:len(done)]:raise ValueError('Production results are not the frozen plan prefix')
    return jobs[len(done):]


def collect(root, resume=False):
    import numpy as np
    from tools.narrow_collection import environment
    from tools.narrow_followup import dataset_union
    from tools.narrow_repair_audit import motion
    from visuotactile.narrow_data import initial_row,near,record,build_cache,admission_conflict,quantiles
    from visuotactile.camera_validity import inspect_capture
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.state_benchmark import prepare,trial
    from visuotactile.recording import model_fingerprint
    import h5py
    s=read(root/'session.json');parent=Path(s['parent']);plan=read(root/'plan.json')
    app,env,player,lease=environment(root);fingerprint=model_fingerprint(player.model)
    original=[dict(e,state_valid=True,multi_view_valid=False,
        modality_reason='Preserved200: fixed RGB blank; low-dimensional State acceptance only') for e in read(parent/'manifest.json')['episodes']]
    accepted=[];results=[];pilot=[]
    if resume:
        receipt=read(root/'reviewed_collection_resume.json')
        for p,h in receipt['input_sha256'].items():
            if sha(p)!=h:raise RuntimeError('Reviewed continuation input changed: '+p)
        accepted=[e for e in read(root/'manifest.json')['episodes'] if e.get('multi_view_valid')]
        results=read(root/'collection_results.json');pilot=read(root/'pilot_results.json')
    def ingest(spec,out,result):
        if not result['legal']:raise RuntimeError('Frozen admitted family rejected; no outcome replacement')
        path=Path(result['dataset_capture']['path'])
        visual=inspect_capture(path,preview=out/'three_view_review.jpg' if len(results)<12 else None)
        classification=capture_classification(result,visual)
        key,meta,rec=record(path)
        source=parent/'initials'/spec['family']/'initial_observable.json'
        if not source.exists():source=root/'initials'/spec['family']/'initial_observable.json'
        difference=admission_conflict(read(source),read(out/'initial_observable.json'),True,True)
        if max(difference['observable_max_abs'].values())>1e-5:raise RuntimeError('Frozen family initial state differs')
        with h5py.File(path) as f:metrics=motion(f['episodes'][key])
        row=dict(identity=spec['identity'],family=spec['family'],alpha=spec['alpha'],success=result['success'],
            duration_s=result['steps']/60,motion=metrics,visual_passed=True,classification=classification)
        results.append(row)
        if result['success']:
            accepted.append(dict(id=spec['identity'],family=spec['family'],split=spec['split'],alpha=spec['alpha'],
                path=str(path),steps=result['steps'],sha256=sha(path),size=path.stat().st_size,
                state_valid=True,multi_view_valid=True,visual_acceptance_sha256=sha(path.parent/'visual_acceptance.json')))
        write(root/'manifest.json',dict(version=3,collection_finished=False,episodes=dataset_union(original,accepted),
            parent_manifest_sha256=sha(parent/'manifest.json'),original400_included=False,quarantined=False,
            visual_subset='Only entries with multi_view_valid=true; never splice rerun images into old200'))
        write(root/'collection_counts.json',dict(attempts=len(results),successes=len(accepted),total_successes=200+len(accepted),
            task_failures=sum(not r['success'] for r in results),initial_rejections=0,software_failures=0,
            new_multimodal_valid=len(accepted),utc=time.time()))
        write(root/'collection_results.json',results)
        return row
    if resume:
        old_ids={f['identity'] for f in read(parent/'families.json')}
        chosen=[f for f in read(root/'families.json') if f['identity'] not in old_ids]
        jobs=read(root/'production_plan.json')
        if not read(root/'speed_decision.json')['passed']:raise RuntimeError('Original speed gate not passed')
    else:
        # Count the actual camera-acceptance expert once; do not simulate it again.
        first=read(root/'camera_expert.json')['result'];first_spec=read(Path(first['dataset_capture']['path']).parents[1]/'started.json')['spec']
        pilot.append(ingest(first_spec,Path(first['dataset_capture']['path']).parents[1],first))
        for alpha in plan['alphas']:
            for f in plan['pilot_families']:
                if alpha==1. and f['identity']==plan['pilot_families'][0]['identity']:continue
                spec=dict(f['spec'],identity=f['identity']+f'_repair_a{round(alpha*100):03}',family=f['identity'],alpha=alpha,split=f['split'])
                speed=dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all');env.speed_config=speed
                out=root/'episodes'/spec['identity'];beat(root,stage='slow_pilot',identity=spec['identity'])
                r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],expert_speed=speed,capture_dataset=True,
                    video=alpha==.60,video_title=f"Frozen RL expert | alpha={alpha} | {f['identity']} | 1x")
                pilot.append(ingest(spec,out,r));write(root/'pilot_results.json',pilot)
        decision=select_speed(pilot);write(root/'speed_decision.json',decision)
        if not decision['passed']:raise RuntimeError(decision['reason'])
        # Initial states, geometry selection, and split are fixed before production
        # outcomes, and no student/test result is used to choose coverage.
        old=read(parent/'families.json');pool=[];initials=root/'initials';initials.mkdir();rejections=0
        for spec in plan['candidates']:
            if time.time()>s['collection_deadline']-900:raise TimeoutError('No time for new coverage')
            out=initials/spec['identity'];out.mkdir()
            with lease.block():
                _,legal=prepare(env,IsolatedEnvironment(env,spec['seed']),spec,{},out)
                row=initial_row(spec['identity'],{k:v[0].cpu().numpy() for k,v in env._diagnostics().items()},env.scene.env_origins[0].cpu().numpy())
            duplicate=any(near(row,f) for f in old+pool)
            if legal['legal'] and not duplicate:pool.append(dict(row,spec=spec))
            else:rejections+=1
            beat(root,stage='family_admission',accepted=len(pool),rejected=rejections)
        if len(pool)<32:raise RuntimeError('Fewer than32 new legal distinct families, no widening')
        keys=('object_xyz','relative_xyz','quaternion','arm','hand','velocity')
        old_train=[f for f in old if f['split']=='train']
        scales={k:np.maximum(np.std([f[k] for f in old_train],axis=0),1e-3) for k in keys}
        def feature(f):return np.concatenate([np.asarray(f[k])/scales[k]/math.sqrt(len(f[k])) for k in keys])
        ref=np.stack([feature(f) for f in old_train]);chosen=[]
        while len(chosen)<32:
            j=int(np.argmax([np.linalg.norm(ref-feature(f),axis=1).min() for f in pool]));f=pool.pop(j)
            chosen.append(f);ref=np.vstack([ref,feature(f)])
        rng=np.random.default_rng(2026110604)
        for j,i in enumerate(rng.permutation(32)):chosen[int(i)]['split']='validation' if j<4 else 'test' if j<8 else 'train'
        write(root/'families.json',old+chosen)
        write(root/'coverage.json',dict(new_families=32,new_splits=dict(Counter(f['split'] for f in chosen)),
            candidate_initial_rejections=rejections,selection='geometry max-min from fixed80 within same narrow domain',
            new_quantiles=quantiles(chosen,s['domain']['center'])))
        jobs=[];low,high=decision['selected']['interval']
        for wave in range(2):
            for i in rng.permutation(32):
                f=chosen[int(i)];alpha=1. if wave==1 and int(i)%4==0 else float(rng.uniform(low,high))
                jobs.append(dict(family=f['identity'],identity=f['identity']+f'_v{wave}',alpha=alpha,split=f['split']))
        write(root/'production_plan.json',jobs)
    byid={f['identity']:f for f in chosen}
    jobs=pending_production(jobs,results)
    consecutive=0
    for row in reversed(results):
        if row['success']:break
        consecutive+=1
    for job in jobs:
        if len(results)>=s['maximum_capture_attempts']:raise RuntimeError('Capture attempt cap')
        if (root/'STOP_AFTER_EPISODE').exists():raise InterruptedError('Stopped at complete episode boundary')
        if time.time()>s['collection_deadline']-600:
            if len(accepted)>=60 and len({e['family'] for e in accepted if e['family'] in byid})>=28:break
            raise TimeoutError('Insufficient data before fixed collection deadline')
        spec=dict(byid[job['family']]['spec'],**job)
        speed=dict(kind='closed_loop',arm=job['alpha'],hand=job['alpha'],phase='all');env.speed_config=speed
        beat(root,stage='slow_production',identity=spec['identity'],successes=len(accepted))
        out=root/'episodes'/spec['identity']
        r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],expert_speed=speed,capture_dataset=True)
        ingest(spec,out,r)
        consecutive=0 if r['success'] else consecutive+1
        if consecutive>=3 or (len(results)>=20 and sum(r['success'] for r in results)/len(results)<.8):
            raise RuntimeError('Slow expert reliability below bounded production policy')
    new_families={e['family'] for e in accepted if e['family'] in byid}
    if len(accepted)<60 or len(new_families)<28:raise RuntimeError('Insufficient accepted new data/coverage')
    if model_fingerprint(player.model)!=fingerprint:raise RuntimeError('Frozen expert changed')
    manifest=read(root/'manifest.json');manifest['collection_finished']=True;write(root/'manifest.json',manifest)
    beat(root,stage='merged_numeric_acceptance');data=build_cache(root)
    modal={split:dict(episodes=sum(e['split']==split for e in accepted),families=len({e['family'] for e in accepted if e['split']==split}))
        for split in ('train','validation','test')}
    ready=dict(data_ready=True,state_valid=200+len(accepted),multi_view_valid=len(accepted),new_successes=len(accepted),
        new_families=len(new_families),state_data=str(data),modality_splits=modal,
        state_splits={split:dict(episodes=sum(e['split']==split for e in manifest['episodes']),
            families=len({e['family'] for e in manifest['episodes'] if e['split']==split})) for split in modal},
        all_new_frames_content_checked=True,old200_visual_invalid_preserved=True,
        future_image_tactile_comparison='Identical accepted multi_view_valid subset and family splits for both modalities; cannot compare differing data amounts as architecture effects')
    write(root/'DATA_READY.json',ready);write(root/'COMPLETE.json',ready)


def diagnose(root):
    """Recorded control/reader audit plus stage-resolved offline prediction errors."""
    import h5py
    import numpy as np
    import torch
    from visuotactile.precision import state_fp32
    from visuotactile.state_resources import lease_for
    from visuotactile.state_history import PersistentState
    from visuotactile.state_image_loss import GeometryObjective
    from visuotactile.position_control import bounded_targets
    from visuotactile.native_data import proprio_radians,robot_rows
    from tools.narrow_repair_audit import stages
    from visuotactile.state_motion import plan_metrics
    state_fp32();s=read(root/'session.json');parent=Path(s['parent']);lease=lease_for(root,deadline=s['gate_deadline'])
    data=read(parent/'state_data/data.json');cache=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    saved=torch.load(parent/'training/best.pt',map_location='cpu',weights_only=False)
    ids=data['splits']['train'][:4]+data['splits']['validation'][:4]
    with h5py.File(data['entries'][ids[0]]['path']) as f:objective=GeometryObjective(json.loads(f.attrs['metadata_json']))
    report=dict(passed=False,reader_max_abs=0.,execution_max_abs=0.,rows=[],no_optimizer_updates=True)
    # Every legal historical rollout: H24/replan12, no second delay/EMA, actual
    # issued targets equal both actuator submissions; packet q inversion agrees.
    for rp in sorted((parent/'evaluation').glob('*/result.json')):
        r=read(rp)
        if not r['legal']:continue
        trace=np.load(rp.parent/'trace.npz');plans=np.load(rp.parent/'plans.npz')['plans']
        limits=np.asarray(read(rp.parent/'metadata.json')['joint_limits_rad'][0])
        issued=np.stack([np.clip(plans[t//12,t%12],limits[:,0],limits[:,1]) for t in range(r['steps'])])
        error=max(float(abs(issued-trace['command']).max()),float(abs(trace['target'][:,None]-trace['actuator_targets']).max()),
            float(abs(trace['target']-trace['command']).max()))
        report['execution_max_abs']=max(error,report['execution_max_abs'])
        with h5py.File(rp.parent/'numeric.h5') as f:
            g=f['episode'];q=proprio_radians(torch.from_numpy(g['obs/proprio'][:]),torch.from_numpy(limits))
            actual=torch.cat([torch.from_numpy(g['transition/diagnostics_before/'+k][:]) for k in ('joint_position_rad','joint_velocity_rad_s')],-1)
            report['reader_max_abs']=max(report['reader_max_abs'],float((q-actual).abs().max()))
    with lease.block():model=PersistentState('dp',saved['contract']).cuda().eval();model.load_state_dict(saved['ema'])
    with torch.no_grad():
        for identity in ids:
            with lease.block():
                rec={k:v.cuda() for k,v in cache['records'][identity].items()};conditions,anchors,_=model.encode(rec)
            predictions=[]
            for i in range(len(conditions)):
                with lease.block():
                    predictions.append(bounded_targets(model.predict(conditions[i:i+1],torch.Generator(device='cuda').manual_seed(7091+i))[0],rec['joint_limits']))
            pred=torch.stack(predictions);index=anchors[:,None]+torch.arange(12,device='cuda');mask=index<len(rec['time']);index=index.clamp_max(len(rec['time'])-1)
            with lease.block():
                truth=rec['command'][index];p=pred[:,:12];xyz,_=objective.fk(p[mask]);gt,_=objective.fk(truth[mask])
                err=(p-truth).cpu().numpy();fk=(xyz[:,0]-gt[:,0]).norm(dim=-1).cpu().numpy()*1000
            with h5py.File(data['entries'][identity]['path']) as f:phase=stages(f['episodes'][data['entries'][identity]['episode']])
            phases=phase[index.cpu().numpy()][mask.cpu().numpy()];err=err[mask.cpu().numpy()]
            per={k:dict(ticks=int((phases==k).sum()),arm_rmse_rad=float(np.sqrt(np.mean(err[phases==k,:7]**2))),
                hand_rmse_rad=float(np.sqrt(np.mean(err[phases==k,7:]**2))),fk_mm=float(np.sqrt(np.mean(fk[phases==k]**2)))) for k in sorted(set(phases))}
            valid=(anchors[:,None]+torch.arange(24,device='cuda')<len(rec['time'])).cpu().numpy()
            report['rows'].append(dict(identity=identity,split='train' if identity in data['splits']['train'] else 'validation',phases=per,
                predicted_continuity=plan_metrics(pred.cpu().numpy(),valid)))
            write(root/'motion_diagnosis.json',report);beat(root,stage='offline_motion',identity=identity)
    report['passed']=report['reader_max_abs']<2e-5 and report['execution_max_abs']<1e-6
    report['scope']='Teacher-forced observed trajectories, deterministic per-anchor noise; contact phases are proxies, not stable-grasp labels'
    write(root/'motion_diagnosis.json',report)
    if not report['passed']:raise RuntimeError('Reader/execution mapping mismatch; expansion blocked')


def evaluate(root):
    import torch
    import numpy as np
    from tools.narrow_collection import environment
    from visuotactile.state_benchmark import trial
    from visuotactile.video import verify_video
    s=read(root/'session.json');job=read(root/'final_job.json');app,env,player,lease=environment(root,True)
    results=[];initials={};pairs={}
    for model in job['models']:
        assert sha(model['checkpoint'])==model['sha256']
        with lease.block():saved=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
        for j,spec in enumerate(job['plan']):
            identity=model['name']+'__'+spec['identity'];out=root/'final_evaluation'/identity
            beat(root,stage='final_evaluation',identity=identity)
            r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],saved=saved,position=True,video=j in (0,4,5),
                inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091),
                video_title=f"{model['name']} | update {saved['updates']} | {spec['identity']} | final cohort | 1x")
            if r['legal'] and j in (0,4,5):r['video']=verify_video(out/'rollout.mp4',r['steps'])
            initial=read(out/'initial_observable.json')
            if spec['identity'] in initials:
                pairs[identity]={k:float(np.abs(np.asarray(initial[k])-initials[spec['identity']][k]).max()) for k in
                    ('joint_position_rad','joint_velocity_rad_s','object_pose_w','hole_pose_w','table_pose_w','palm_body_pose_w','finger_position_w','finger_quat_wxyz')}
            else:initials[spec['identity']]=initial
            results.append(dict(r,model=model['name']));write(root/'final_evaluation/results.json',results)
    write(root/'final_evaluation/result.json',dict(complete=True,models={m['name']:cohort_results([r for r in results if r['model']==m['name']]) for m in job['models']},
        final_test=True,selection_allowed=False,observable_pair_max_errors=pairs,hidden_physics_exact=False))


def queue(root, resume=False):
    from tools.narrow_followup import train_session
    s=read(root/'session.json');parent=Path(s['parent']);plan=read(root/'plan.json')
    assert read(root/'gpu_parity.json')['passed'] and read(root/'camera_expert.json')['visual']['passed']
    review=read(root/'camera_visual_review.json')
    assert review['accepted'] and review['camera_expert_sha256']==sha(root/'camera_expert.json')
    if resume:
        receipt=read(root/'reviewed_collection_resume.json')
        assert read(root/'status.json')['owned_workers_exited']
        assert receipt['completed_physical_attempts']==50 and receipt['pending_production_attempts']==26
        assert receipt['automatic_retry'] is False and time.time()<s['collection_deadline']-600
        for p,h in receipt['input_sha256'].items():
            if sha(p)!=h:raise RuntimeError('Reviewed continuation input changed: '+p)
        assert not (root/'expanded_training').exists(),'Only before the one fresh training'
    def model(name,p):return dict(name=name,checkpoint=str(p),sha256=sha(p))
    models=[model('state200_fp32',parent/'training/best.pt')]
    def jobs(run):
        if not resume:
            write(root/'startup_gpu/session.json',dict(resource_policy='shared_recent_v1',resource_target_percent=40.,
                training_deadline=s['gate_deadline'],external_monitor_root=str(root),random_seed=s['fresh_seed']))
            run('startup_gpu_smoke',['-m','visuotactile.anchor_continue','fresh-smoke','--smoke-device','cuda:0','--job','state',
                '--data',str(parent/'state_data'),'--root',str(root/'startup_gpu'),'--target','1','--output',str(root/'startup_gpu/check')],limit=s['gate_deadline'])
            run('old_model_recheck',['-m','tools.narrow_repair','recheck-worker','--root',str(root)],limit=s['gate_deadline'],kit=True)
        else:
            assert read(root/'startup_gpu/check/result.json')['passed'] and read(root/'motion_diagnosis.json')['passed']
        assert read(root/'recheck_complete.json')['passed']
        recheck=read(root/'recheck_results.json')
        counts={mode:sum(r['success'] for r in recheck if r['precision_name']==mode) for mode in ('legacy_default','unified_fp32')}
        improved=counts['unified_fp32']>=2 and counts['unified_fp32']-counts['legacy_default']>=2
        decision=dict(recheck_successes=counts,action='independent_old_model_validation' if improved else 'one_slow_coverage_expansion',
            recheck_counts={mode:cohort_results([r for r in recheck if r['precision_name']==mode]) for mode in counts},
            rationale='Precision improvement reviewed separately; otherwise measured validation plateau, fitted train motion and weak unseen motion justify one bounded data contrast, not proof of data insufficiency')
        if resume:
            assert decision==read(root/'decision.json'),'Do not change the original research decision'
        else:
            write(root/'decision.json',decision)
            run('motion_audit',['-m','tools.narrow_repair_pipeline','diagnose','--root',str(root)],limit=s['gate_deadline'])
        if not improved:
            run('slow_collection',['-m','tools.narrow_repair_pipeline','collect-remaining' if resume else 'collect','--root',str(root)],limit=s['collection_deadline'],kit=True)
            data=read(root/'state_data/data.json');ids=data['splits']['train'];anchors=sum(math.ceil(data['entries'][i]['length']/12) for i in ids)
            average=len(ids)/math.ceil(len(ids)/64);old_budget=read(parent/'training/budget.json');rate=read(parent/'training/progress.json')['updates_per_hour']
            # Plateau starts around4000 in the original curve; allow500-update
            # headroom and equivalent anchor exposure, capped by measured speed.
            exposure=4500*old_budget['mean_batch']/old_budget['anchors']
            target=min(s['maximum_training_updates'],math.ceil(exposure*anchors/average),int(rate*2.4))
            deadline=min(s['deadline']-3600,time.time()+s['training_hours']*3600)
            if deadline-time.time()<1800:raise TimeoutError('Too little remaining training time')
            args=train_session(root,root/'expanded_training',root/'state_data',target,deadline,minimum=max(2000,int(target*.6)))
            write(root/'expanded_training/exposure_budget.json',dict(anchors=anchors,mean_batch=average,predecessor_plateau_update=4000,
                exposure_with_headroom=exposure,target=target,throughput_updates_per_hour=rate,random_seed=s['fresh_seed'],fresh_random=True))
            run('fresh_gpu_smoke',['-m','visuotactile.anchor_continue','fresh-smoke','--smoke-device','cuda:0','--job','state',
                '--data',str(root/'state_data'),'--root',str(root/'disposable_gpu'),'--target','1','--output',str(root/'disposable_gpu/check')],limit=deadline)
            run('fresh_training',args,limit=deadline)
            models.append(model('state_expanded_fp32',root/'expanded_training/best.pt'))
        write(root/'offline_job.json',dict(models=models))
        run('offline_comparison',['-m','tools.narrow_repair_pipeline','offline','--root',str(root)],limit=s['deadline']-1800)
        final=plan['final_train']+plan['final_unseen']
        assert len(final)*len(models)+len(recheck)<=s['maximum_evaluation_starts']
        write(root/'final_job.json',dict(models=models,plan=final,precision='unified_fp32',selection_allowed=False))
        run('final_evaluation',['-m','tools.narrow_repair_pipeline','evaluate','--root',str(root)],kit=True)
        protected(root)
        write(root/'FINAL_REPORT.json',dict(complete=True,decision=decision,models=models,
            expansion=read(root/'DATA_READY.json') if (root/'DATA_READY.json').exists() else None,
            training=read(root/'expanded_training/finished.json') if (root/'expanded_training/finished.json').exists() else None,
            final_evaluation=read(root/'final_evaluation/result.json'),old200_preserved=True,
            automatic_second_expansion=False,automatic_image_or_tactile_training=False))
    execute(root,'reviewed_collection_completion' if resume else 'production_and_training',jobs,s['deadline'])
    finished=root/'expanded_training/finished.json'
    status(root,state='complete',stage='complete',formal_updates=read(finished)['updates'] if finished.exists() else 0)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','continue-reviewed','collect','collect-remaining','diagnose','evaluate','offline'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.mode in ('run','continue-reviewed'):
        try:queue(root,resume=a.mode=='continue-reviewed')
        finally:
            # All owned GPU processes are released by execute before publication.
            # Conflicts retain results locally and never force-push or overwrite.
            if (root/'repair_publication_request.json').exists() and read(root/'status.json').get('owned_workers_exited'):
                try:
                    from tools.narrow_repair_delivery import finish
                    finish(root)
                except BaseException:
                    write(root/'publication_error.json',dict(traceback=traceback.format_exc(),utc=time.time()))
        return
    kit=a.mode in ('collect','collect-remaining','evaluate');code=0
    try:
        if a.mode in ('collect','collect-remaining'):collect(root,resume=a.mode=='collect-remaining')
        elif a.mode=='diagnose':diagnose(root)
        elif a.mode=='evaluate':evaluate(root)
        else:
            from visuotactile.precision import state_fp32
            from tools.narrow_followup import offline
            state_fp32();offline(root,root/'offline_job.json')
    except BaseException:
        write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc(),utc=time.time()))
        if not kit:raise
        code=1
    if kit:sys.stdout.flush();sys.stderr.flush();os._exit(code)


if __name__=='__main__':main()
