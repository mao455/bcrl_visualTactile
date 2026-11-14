"""Train-only stage error of source/selected DP EMA, in separate frozen processes.

prepare is CPU-only. run-arm executes exactly one admitted frontend with complete
real histories. compare never runs a model. No rear plan cache is reused.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

from tools.tight_fresh_selected_fit import (read,sha256,checked_hash,exclusive_json,
    validate_inputs,validate_checkpoint,import_frozen_trainer)

LEGACY_VERSION='train_only_selected_frontend_stage_fit_v1'
VERSION='train_only_selected_frontend_stage_fit_v2'
STAGES=('grasp_before_first_hold','holding','hand_rotation','near_hole60',
    'after_first_native_subgoal','native_alignment','native_subgoal_increment')
BASELINE_COLUMNS=dict(hold_anchor_q=(0,29),hold_anchor_actuator_target=(58,87))


def baseline_names(role):
    require(role in ('A','B'),'Unknown frontend role')
    return ['hold_anchor_q']+(['hold_anchor_actuator_target'] if role=='B' else [])


def require(value,message):
    if not value:raise ValueError(message)


def stage_masks(pose,palm,force,goal_before,goal_after,stage_before,stage_after,keypoint,valid):
    """Privileged, overlapping *diagnostic* masks, never neural inputs or targets."""
    import numpy as np
    from visuotactile.takeover_trigger import contact_height_windows
    from tools.tight_fresh_takeover_pose_audit import window_components
    n=len(valid)
    require(n>=48 and pose.shape==(n,7) and palm.shape==(n,7) and force.shape==(n,2,5,3),
        'Complete diagnostic geometry and 48-transition height reference required')
    require(valid.dtype==bool and valid.any(),'Real legal expert actions required')
    height=float(np.median(pose[24:48,2]))
    held=contact_height_windows(pose,force,height);held[:47]=False
    rotating=np.zeros(n,bool)
    for tick in np.flatnonzero(held):
        rotating[tick]=window_components(pose,palm,force,height,int(tick)+1)['palm_relative_max_rotation_rad']>.2
    same=(goal_before==goal_after).all(-1)&(stage_before==stage_after)
    found=np.flatnonzero(held);first=int(found[0]) if len(found) else n
    masks=dict(grasp_before_first_hold=(np.arange(n)<first)&(stage_before==0),
        holding=held,hand_rotation=held&rotating,near_hole60=held&same&(keypoint<=.06),
        after_first_native_subgoal=stage_before>0,
        native_subgoal_increment=stage_after>stage_before)
    return {key:mask&valid for key,mask in masks.items()},height


def bind_source(receipt,frozen_root,selected):
    """The source is the exact continuation EMA, not an arbitrary earlier model."""
    import torch
    continuation=receipt['configuration']['continuation']
    path=Path(continuation['path']).resolve(strict=True)
    digest=checked_hash(path,continuation['sha256'])
    require(path.name=='best.pt' and read(path.parent/'finished.json')['checkpoint_sha256']==digest,
        'Continuation must bind the actual selected source best checkpoint')
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    require(saved.get('kind')=='dp' and saved.get('group')=='I-HistoryDP' and
        saved['contract']==selected['contract'] and saved.get('epoch',0)>0 and
        saved['configuration']['statistics_sha256']==receipt['sha256']['statistics'],
        'Source and selected frontend contracts/normalization differ')
    root=Path(frozen_root).resolve(strict=True)
    sources=saved['configuration']['source_sha256']
    require('tools/tight_fresh_train.py' in sources,'Source frozen trainer identity missing')
    for relative,digest in sources.items():
        source=(root/relative).resolve(strict=True)
        require(source.is_relative_to(root),'Frozen source escapes its tree')
        checked_hash(source,digest)
    return dict(path=str(path),sha256=sha256(path),epoch=saved['epoch'],updates=saved['updates'],
        frozen_source_root=str(root),configuration=saved['configuration'],source_files_verified=len(sources),
        checkpoint_weights='ema'),saved


def project_training_records(output,data,container,identities,visual_cache):
    """Copy selected tensors only; frozen readers never load held-out payloads."""
    import torch
    ids=set(identities)
    require(ids<=set(data['splits']['train']) and not ids &
        (set(data['splits'].get('validation',[]))|set(data['splits'].get('test',[]))),
        'Only training identities may enter the isolated reader projection')
    size=sum(t.numel()*t.element_size() for identity in identities for t in container['records'][identity].values())
    require(size<=512*1024**2,'Train-only numeric projection exceeds 512MiB bound')
    root=Path(output)/'train_only';root.mkdir();(root/'perception').mkdir()
    projected=copy.deepcopy(data)
    projected['entries']={identity:projected['entries'][identity] for identity in identities}
    projected['splits']=dict(train=list(identities),validation=[],test=[])
    projected['diagnostic_train_projection']=True
    exclusive_json(root/'data.json',projected)
    records={identity:{key:value.clone() for key,value in container['records'][identity].items()} for identity in identities}
    torch.save(dict(data_sha256=sha256(root/'data.json'),records=records),root/'numeric.pt')
    for identity in identities:(root/'perception'/(identity+'.pt')).symlink_to(Path(visual_cache[identity]['path']).resolve())
    return dict(root=str(root),data_sha256=sha256(root/'data.json'),numeric_sha256=sha256(root/'numeric.pt'),
        numeric_tensor_bytes=size,identities=list(identities),
        source_container_access='Original mixed numeric file integrity hash and mmap metadata only; only selected train tensors are dereferenced/copied. No held-out raw or visual shard is opened.')


def prepare(root,role,collection,selected_frozen,source_frozen,output):
    import numpy as np
    import torch
    import h5py
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    try:
        receipt=validate_inputs(root,role,selected_frozen)
        selected=torch.load(receipt['paths']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        validate_checkpoint(selected,receipt)
        source,source_saved=bind_source(receipt,source_frozen,selected)
        collection=Path(collection).resolve(strict=True)
        require(read(collection/'result.json').get('software_complete'), 'Completed collection required')
        accepted=read(collection/'correction_entries.json')
        indexed={row['case']:row['result'] for row in read(collection/'results.json')}
        data=read(receipt['paths']['data'])
        # The reader owns a monolithic numeric container; no validation/test
        # record is selected or evaluated, and no held-out raw/cache file opens.
        numeric=torch.load(Path(root)/role/'state_data/numeric.pt',map_location='cpu',weights_only=False,mmap=True)
        require(numeric['data_sha256']==receipt['sha256']['data'],'Numeric container binding differs')
        masks={};episodes={}
        for identity in receipt['identities']:
            entry=data['entries'][identity]
            require(identity in accepted and accepted[identity]==entry,'Current correction differs from accepted collection entry')
            result=indexed[identity]
            require(result.get('legal') and result.get('accepted_for_training') and
                result['dataset_capture']['sha256']==entry['source_sha256'], 'Accepted capture binding differs')
            path=collection/identity/'numeric.h5'
            with h5py.File(path,'r') as stream:
                t=stream['episode/transition'];before=t['diagnostics_before'];after=t['diagnostics_after']
                valid=t['dagger/expert_label_valid'][:].astype(bool)
                record=numeric['records'][identity]
                require(len(valid)==entry['length'] and np.array_equal(valid,record['supervision_mask'].numpy()) and
                    np.array_equal(t['dagger/control_source'][:]==0,np.arange(len(valid))<entry['takeover_tick']) and
                    np.array_equal(t['time_before_s'][:],record['time'].numpy()),'Stage and training history/label masks differ')
                key='q_reached' if role=='A' else 'u_expert'
                require(np.array_equal(t['dagger/'+key][:][valid],record[key].numpy()[valid]),
                    'Stage diagnostics are not the actual supervised target trajectory')
                values,height=stage_masks(after['object_pose_w'][:],after['palm_body_pose_w'][:],
                    t['object_contact_force_w'][:],before['goal_pose_w'][:],after['goal_pose_w'][:],
                    before['successes'][:],after['successes'][:],t['native_keypoints_max_dist'][:],valid)
                values['native_alignment']=t['native_alignment_condition'][:].astype(bool)&valid
                values['legal_expert']=valid
                matrix=np.stack([values[key] for key in (*STAGES,'legal_expert')])
                masks[identity]=matrix
                visits=int(selected['sampling_exposure'][identity])
                episodes[identity]=dict(family=entry['family'],history_frames=len(valid),
                    legal_expert_actions=int(valid.sum()),legal_actions_by_stage={key:int(values[key].sum()) for key in STAGES},
                    selected_full_history_visits=visits,
                    selected_legal_action_visible_count={key:int(values[key].sum())*visits for key in STAGES},
                    source_prior_full_history_visits=source_saved.get('sampling_exposure',{}).get(identity,0)
                        if identity in source_saved['train_ids'] else 0,
                    actual_training_anchor_target_slots_by_stage=None,
                    unknown_slots_reason='Per-stage optimizer target-slot receipts were not recorded; action visibility is not target-slot exposure.',
                    settled_height_m=height,settled_reference_authority='student_only' if entry['takeover_tick']>=48 else 'includes_expert_ticks',
                    numeric_path=str(path),numeric_sha256=sha256(path),capture_sha256=entry['source_sha256'])
        np.savez(output/'stage_masks.npz',**masks)
        projection=project_training_records(output,data,numeric,receipt['identities'],receipt['selected_visual_cache'])
        registered=dict(version=VERSION,root=str(Path(root).resolve()),role=role,collection=str(collection),
            train_projection=projection,
            selected_receipt=receipt,source=source,
            selected=dict(path=receipt['paths']['checkpoint'],sha256=receipt['sha256']['checkpoint'],
                epoch=selected['epoch'],updates=selected['updates'],checkpoint_weights='ema',
                frozen_source_root=receipt['frozen_source_root'],configuration=receipt['configuration']),
            stages=list(STAGES),identities=receipt['identities'],episodes=episodes,
            baseline_names=baseline_names(role),
            masks_path=str(output/'stage_masks.npz'),masks_sha256=sha256(output/'stage_masks.npz'),
            collection_entries_sha256=sha256(collection/'correction_entries.json'),
            evaluator_sha256=sha256(__file__),cuda_initialized=torch.cuda.is_initialized(),
            scope='DP frontend target error only; source and selected EMA, full real history, train-only latest accepted corrections.',
            definitions=dict(grasp_before_first_hold='Original first-goal ticks before first complete contact-height window; includes approach/initial contact, not intent.',
                holding='12 actual transitions: >=2 fingers >0.1N in both physics samples, >=2cm above settled reference; rotation allowed.',
                hand_rotation='Holding with past12 palm-relative rotation >0.2rad; does not distinguish intended rotation from slip.',
                near_hole60='Holding, unchanged current goal and stage, native max keypoint distance <=60mm; native success threshold unchanged.',
                after_first_native_subgoal='Pre-action native stage>0; later-goal/insertion-stage proxy, not proof of actual insertion.',
                overlap='Masks overlap and are retrospective diagnostics only. Privileged geometry never enters model inputs.',
                teacher_forcing='Recorded actual observations/actions including complete student prefix; not autonomous execution.'),
            baseline_definitions=dict(hold_anchor_q='Actual pre-action q at each admitted 5Hz/onset planning anchor, held through its commit12. Same legal target slots as both DPs; no target-tick q.',
                hold_anchor_actuator_target='B only: actual pre-action actuator target at the planning anchor, held through commit12. This matches the earlier frontend-baseline B input semantics; it is not the current expert target.',
                limits='Forecast-error diagnostics only, no executed hold policy. No 60Hz observation refresh inside a commitment. Different A/B labels must not be ranked by raw RMSE.'),
            rear_evaluated=False,validation_evaluated=False,test_evaluated=False,optimizer_updates=0)
        exclusive_json(output/'registration.json',registered)
        return registered
    except BaseException as error:
        exclusive_json(output/'prepare_error.json',dict(error=repr(error),utc=time.time()))
        raise


def error_totals(errors,target_ticks,masks):
    """CPU reductions of actual commit12 slots; never mix student/invalid labels."""
    import numpy as np
    require(errors.shape==(len(target_ticks),29),'Expected canonical29 errors for each evaluated slot')
    require(masks.shape[0]==len(STAGES)+1 and masks[-1,target_ticks].all(),
        'A student or invalid expert action entered stage error')
    rows={}
    for index,key in enumerate((*STAGES,'legal_expert')):
        chosen=errors[masks[index,target_ticks]]
        rows[key]=dict(evaluated_commit_slots=len(chosen),
            arm_squared_sum=float(np.square(chosen[:,:7].astype(np.float64)).sum()),
            hand_squared_sum=float(np.square(chosen[:,7:].astype(np.float64)).sum()))
    return rows


def finalized(rows):
    import math
    return {key:dict(value,arm_rmse_rad=math.sqrt(value['arm_squared_sum']/(7*value['evaluated_commit_slots']))
        if value['evaluated_commit_slots'] else None,
        hand_rmse_rad=math.sqrt(value['hand_squared_sum']/(22*value['evaluated_commit_slots']))
        if value['evaluated_commit_slots'] else None) for key,value in rows.items()}


def anchor_hold_totals(record,anchors,truth,valid,masks,names):
    """Only anchor observations; even a legal suffix uses its earlier anchor q."""
    import torch
    require(names and len(set(names))==len(names) and all(name in BASELINE_COLUMNS for name in names),
        'Unknown or duplicate causal hold baseline')
    require(truth.shape==(len(anchors),24,29) and valid.shape==(len(anchors),24) and
        valid.dtype==torch.bool,'Expected actual horizon24 label windows')
    robot=record['robot'];mask=valid[:,:12]
    require(robot.ndim==2 and robot.shape[1]>=max(BASELINE_COLUMNS[name][1] for name in names),
        'Canonical actual joint/actuator observations required')
    require(anchors.ndim==1 and anchors.dtype==torch.int64 and (anchors>=0).all() and
        (anchors<len(robot)).all(),'Invalid real planning anchors')
    ticks=(anchors[:,None]+torch.arange(12,device=anchors.device))[mask].cpu().numpy()
    results={}
    for name in names:
        start,stop=BASELINE_COLUMNS[name]
        observed=robot[anchors,start:stop]
        require(torch.isfinite(observed).all(),'Nonfinite anchor observation')
        errors=(observed[:,None,:]-truth[:,:12])[mask].cpu().numpy()
        results[name]=error_totals(errors,ticks,masks)
    return results


def frontend_episode(trainer,model,record,masks,*,baselines=()):
    """One independent full-prefix encoding under this arm's actual parameters."""
    import torch
    conditions,anchors=trainer.supervised_conditions(model,record)
    predictions=trainer.predict_on_native_clock(model,conditions,anchors)
    predictions=trainer.bounded_targets(predictions.flatten(0,1),record['joint_limits']).reshape_as(predictions)
    truth,valid=trainer.labels(record,anchors,24);mask=valid[:,:12]
    ticks=(anchors[:,None]+torch.arange(12,device=anchors.device))[mask].cpu().numpy()
    errors=(predictions[:,:12]-truth[:,:12])[mask].cpu().numpy()
    result=error_totals(errors,ticks,masks)
    if baselines:return result,anchor_hold_totals(record,anchors,truth,valid,masks,baselines)
    return result


def sum_episode_totals(episodes):
    return {key:{field:sum(row[key][field] for row in episodes.values()) for field in
        ('evaluated_commit_slots','arm_squared_sum','hand_squared_sum')} for key in (*STAGES,'legal_expert')}


def run_arm(registration,arm):
    import numpy as np
    import torch
    path=Path(registration).resolve(strict=True);plan=read(path);out=path.parent/arm
    out.mkdir(exist_ok=False);start=time.perf_counter()
    exclusive_json(out/'started.json',dict(pid=os.getpid(),utc=time.time(),arm=arm,registration_sha256=sha256(path)))
    try:
        require(plan['version']==VERSION and arm in ('source','selected'),'Unknown stage-fit arm')
        require(plan['baseline_names']==baseline_names(plan['role']),'Registered baseline semantics changed')
        checked_hash(__file__,plan['evaluator_sha256'])
        receipt=plan['selected_receipt'];weights=plan[arm]
        checked_hash(receipt['evaluator_source'],receipt['evaluator_sha256'])
        checked_hash(weights['path'],weights['sha256'])
        for key in ('data','statistics','cache_index'):
            checked_hash(receipt['paths'][key],receipt['sha256'][key])
        checked_hash(Path(plan['root'])/plan['role']/'state_data/numeric.pt',receipt['sha256']['numeric'])
        projection=plan['train_projection'];projected_root=Path(projection['root'])
        checked_hash(projected_root/'data.json',projection['data_sha256'])
        checked_hash(projected_root/'numeric.pt',projection['numeric_sha256'])
        checked_hash(plan['masks_path'],plan['masks_sha256'])
        frozen=Path(weights['frozen_source_root'])
        for relative,digest in weights['configuration']['source_sha256'].items():
            checked_hash(frozen/relative,digest)
        for item in receipt['selected_visual_cache'].values():checked_hash(item['path'],item['sha256'])
        for identity,item in receipt['selected_visual_cache'].items():
            require((projected_root/'perception'/(identity+'.pt')).resolve(strict=True)==Path(item['path']).resolve(strict=True),
                'Projected frozen visual shard binding changed')
        saved=torch.load(weights['path'],map_location='cpu',weights_only=False,mmap=True)
        require(saved['epoch']==weights['epoch'] and saved['configuration']==weights['configuration'],
            'Admitted EMA epoch/configuration changed')
        trainer=import_frozen_trainer(dict(frozen_source_root=str(frozen)))
        from visuotactile.precision import state_fp32
        from visuotactile.state_resources import lease_for
        from visuotactile.target_windows import with_target_window
        state_fp32();root=Path(plan['root'])
        with lease_for(root,deadline=read(root/'session.json')['deadline']).block(),torch.no_grad():
            records=trainer.visual_records(projected_root,plan['identities'],'G00',device='cuda:0')
            records=with_target_window(records,receipt['target_window'])
            statistics=torch.load(receipt['paths']['statistics'],map_location='cpu',weights_only=False)
            model=trainer.new_model(plan['role'],statistics)
            require(model.contract==saved['contract'],'Frozen constructor/checkpoint contract differs')
            model.load_state_dict(saved['ema'],strict=True);model.eval().requires_grad_(False)
            archive=np.load(plan['masks_path'],allow_pickle=False);episodes={}
            baseline_episodes={name:{} for name in plan['baseline_names']}
            for identity in plan['identities']:
                record=records[identity]
                require(len(record['time'])==plan['episodes'][identity]['history_frames'],'Full-history length changed')
                episodes[identity],held=frontend_episode(trainer,model,record,archive[identity],
                    baselines=plan['baseline_names'])
                for name,values in held.items():baseline_episodes[name][identity]=values
            archive.close();torch.cuda.synchronize()
        totals=sum_episode_totals(episodes)
        result=dict(complete=True,arm=arm,epoch=weights['epoch'],updates=weights['updates'],
            checkpoint_sha256=weights['sha256'],registration_sha256=sha256(path),
            stages=finalized(totals),episodes={identity:finalized(row) for identity,row in episodes.items()},
            baselines={name:dict(stages=finalized(sum_episode_totals(values)),
                episodes={identity:finalized(row) for identity,row in values.items()})
                for name,values in baseline_episodes.items()},
            full_history_recomputed_each_episode=True,learned_hidden_state_cached=False,
            target_window=receipt['target_window'],seconds=time.perf_counter()-start,
            optimizer_updates=0,rear_evaluated=False,validation_evaluated=False,test_evaluated=False,
            scope='Teacher-forced DP frontend commit12 target error, not complete-student or autonomous performance.')
        exclusive_json(out/'result.json',result);return result
    except BaseException as error:
        exclusive_json(out/'error.json',dict(error=repr(error),utc=time.time()));raise


def compare(registration):
    path=Path(registration);plan=read(path);arms={arm:read(path.parent/arm/'result.json') for arm in ('source','selected')}
    version=plan.get('version',LEGACY_VERSION)
    require(version in (VERSION,LEGACY_VERSION),'Unknown stage-fit comparison version')
    names=baseline_names(plan['role']) if version==VERSION else []
    if names:require(plan['baseline_names']==names,'Registered baseline semantics changed')
    for arm,result in arms.items():
        require(result['complete'] and result['registration_sha256']==sha256(path) and
            result['checkpoint_sha256']==plan[arm]['sha256'],'Stage-fit result binding differs')
        if names:require(set(result.get('baselines',{}))==set(names),'Causal hold baseline result missing')
    rows={}
    for key in (*STAGES,'legal_expert'):
        source,selected=(arms[arm]['stages'][key] for arm in ('source','selected'))
        require(source['evaluated_commit_slots']==selected['evaluated_commit_slots'],'Arm evaluated populations differ')
        rows[key]=dict(evaluated_commit_slots=source['evaluated_commit_slots'],source=source,selected=selected,
            delta_rad={metric:selected[metric]-source[metric] if source[metric] is not None else None
                for metric in ('arm_rmse_rad','hand_rmse_rad')})
        if names:
            rows[key]['baselines']={};rows[key]['model_minus_hold_rmse_rad']={}
            for name in names:
                held=arms['source']['baselines'][name]['stages'][key]
                require(held==arms['selected']['baselines'][name]['stages'][key] and
                    held['evaluated_commit_slots']==source['evaluated_commit_slots'],
                    'Causal hold inputs/target populations differ between arms')
                rows[key]['baselines'][name]=held
                rows[key]['model_minus_hold_rmse_rad'][name]={arm:{metric:arms[arm]['stages'][key][metric]-held[metric]
                    if held[metric] is not None else None for metric in ('arm_rmse_rad','hand_rmse_rad')}
                    for arm in ('source','selected')}
    result=dict(version=version,registration_sha256=sha256(path),stages=rows,
        source_epoch=plan['source']['epoch'],selected_epoch=plan['selected']['epoch'],episodes=plan['episodes'],
        scope=plan['scope'],rear_evaluated=False,optimizer_updates=0,validation_evaluated=False,test_evaluated=False)
    exclusive_json(path.parent/'comparison.json',result);return result


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for option in ('root','collection','selected-frozen','source-frozen','output'):prep.add_argument('--'+option,type=Path,required=True)
    prep.add_argument('--role',choices=['A','B'],required=True)
    run=sub.add_parser('run-arm');run.add_argument('--registration',type=Path,required=True);run.add_argument('--arm',choices=['source','selected'],required=True)
    comp=sub.add_parser('compare');comp.add_argument('--registration',type=Path,required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    result=prepare(**args) if command=='prepare' else run_arm(**args) if command=='run-arm' else compare(**args)
    print(json.dumps(dict(command=command,complete=True,scope=result.get('scope'))))


if __name__=='__main__':main()
