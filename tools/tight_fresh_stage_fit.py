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
UNION_VERSION='train_only_selected_frontend_stage_fit_v3'
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


def bind_collection_union(data,latest_collection,include_collections):
    """Match accepted captures by immutable content, independent of pool ID names."""
    paths=[Path(p).resolve(strict=True) for p in [latest_collection,*include_collections]]
    require(len(paths)==len(set(paths)),'Duplicate collection in stage-fit union')
    train=data['splits']['train'];heldout=set(data['splits'].get('validation',[]))|set(data['splits'].get('test',[]))
    heldout_families={data['entries'][i]['family'] for i in heldout}
    by_sha={}
    for identity in train:
        digest=data['entries'][identity]['source_sha256']
        require(digest not in by_sha,'Ambiguous current training source SHA')
        by_sha[digest]=identity
    cases={};collections=[]
    for index,path in enumerate(paths):
        require(read(path/'result.json').get('software_complete'), 'Completed collection required')
        plan=read(path/'plan.json');accepted=read(path/'correction_entries.json')
        require(plan.get('kind')=='takeover' and type(plan.get('round_index')) is int,
            'Registered takeover collection round required')
        planned={case['name']:case for case in plan['cases']}
        rows=read(path/'results.json');indexed={row['case']:row['result'] for row in rows}
        require(len(planned)==len(plan['cases']) and len(indexed)==len(rows),'Duplicate collection case identity')
        require(accepted,'Empty collection cannot contribute stage-fit union data')
        matched=[]
        for case_name,entry in accepted.items():
            identity=by_sha.get(entry['source_sha256'])
            require(identity is not None and identity not in heldout and entry==data['entries'][identity],
                'Accepted union capture differs from current training content')
            require(identity not in cases,'Duplicate capture across stage-fit collections')
            require(entry.get('split')=='train' and entry.get('trajectory_kind')=='expert_takeover_v1' and
                entry['family'] not in heldout_families,'Only training-family expert corrections may enter union')
            require(case_name in planned and case_name in indexed,'Accepted case is absent from its registered collection')
            case=planned[case_name];result=indexed[case_name]
            require(case['spec'].get('split')=='train' and case['spec'].get('family')==entry['family'] and
                case['spec'].get('identity')==case_name,'Collection case training identity differs')
            require(result.get('legal') and result.get('success') and result.get('accepted_for_training') and
                result['dataset_capture']['sha256']==entry['source_sha256'],'Accepted union recovery binding differs')
            cases[identity]=dict(collection=str(path),collection_index=index,collection_round=plan['round_index'],
                case=case_name,numeric_path=str(path/case_name/'numeric.h5'),result=result)
            matched.append(identity)
        if index==0:
            require(set(matched)==set(data['aggregation']['new_training_ids']),
                'Primary collection must bind exactly the latest accepted corrections')
        collections.append(dict(path=str(path),round_index=plan['round_index'],identities=matched,
            files={name:sha256(path/name) for name in ('plan.json','result.json','results.json','correction_entries.json')}))
    identities=[i for i in train if i in cases]
    require(identities and len(identities)==len(cases),'Incomplete union training identity binding')
    return identities,cases,collections


def extend_union_receipt(receipt,data,identities):
    """Reuse ordinary admission; additionally verify only the extra train shards."""
    root=Path(receipt['root'])/receipt['role'];cache=read(receipt['paths']['cache_index'])
    for identity in identities:
        if identity in receipt['selected_visual_cache']:continue
        entry=cache['entries'][identity];path=root/'state_data/perception'/(identity+'.pt')
        require(entry['source_sha256']==data['entries'][identity]['source_sha256'] and
            entry['frames']==data['entries'][identity]['length'],'Union visual cache source/frame identity differs')
        digest=checked_hash(path,entry['sha256'])
        receipt['selected_visual_cache'][identity]=dict(path=str(path.resolve()),sha256=digest,frames=entry['frames'])
    require(set(receipt['selected_visual_cache'])==set(identities),'Union visual population differs')
    receipt['identities']=list(identities)


def validate_batch_checkpoint_visits(saved):
    """B2 checkpoints count complete visits separately from optimizer updates."""
    from visuotactile.trajectory_training import validate_profile,validate_exposure
    config=saved['configuration'];profile=validate_profile(config.get('full_trajectory_batch'))
    if profile is None:return False
    multiplicities=config['episode_multiplicities']
    require(set(multiplicities)==set(saved['train_ids']) and
        all(type(n) is int and n>0 for n in multiplicities.values()) and
        config['episodes_per_epoch']==sum(multiplicities.values()) and
        config['optimizer_updates_per_epoch']==(sum(multiplicities.values())+1)//2,
        'B2 checkpoint registered visit deck differs')
    validate_exposure(saved.get('sampling_exposure'),multiplicities,completed_epochs=saved['epoch'],
        updates=saved['updates'],visits_seen=saved.get('visits_seen'))
    return True


def source_training_exposure(source,saved,data,identities):
    """Source-stage visits by raw-content identity, never by possibly renamed IDs."""
    path=Path(source['path']).parent.parent/'state_data/data.json'
    digest=checked_hash(path,saved['configuration']['data_sha256']);prior=read(path)
    require(saved['train_ids']==prior['splits']['train'],'Source checkpoint training manifest differs')
    by_sha={}
    for identity in saved['train_ids']:
        entry=prior['entries'][identity]
        require(entry['source_sha256'] not in by_sha,'Ambiguous source training SHA')
        by_sha[entry['source_sha256']]=identity
    batched=validate_batch_checkpoint_visits(saved)
    visits=saved.get('sampling_exposure')
    if visits is None:
        require(saved['configuration'].get('version')=='fresh30_continuous_episode_v1' and
            not saved['configuration'].get('sampling_contract') and
            saved['updates']==saved['epoch']*len(saved['train_ids']),
            'Cannot infer source-stage visits without an explicit complete uniform schedule')
        visits={identity:saved['epoch'] for identity in saved['train_ids']}
        authority='Completed uniform source epochs; checkpoint updates equal epochs times train episodes'
    else:
        require(set(visits)==set(saved['train_ids']) and all(type(v) is int and v>0 for v in visits.values()) and
            sum(visits.values())==(saved['visits_seen'] if batched else saved['updates']),
            'Source checkpoint visit receipts differ')
        authority='Source selected checkpoint sampling_exposure; stage-local, not inherited cumulative dose'
        if batched:authority+='; B2 full-history visits are distinct from optimizer/EMA updates'
    rows={}
    for identity in identities:
        entry=data['entries'][identity];old=by_sha.get(entry['source_sha256'])
        if old is not None:
            require(prior['entries'][old]['family']==entry['family'],'Same content source/current family differs')
        rows[identity]=dict(source_training_id=old,source_seen=old is not None,
            full_history_visits=visits[old] if old is not None else 0)
    return rows,dict(path=str(path),sha256=digest,authority=authority,
        identity='immutable source_sha256',inherited_earlier_stage_visits_included=False)


def union_strata(identities,data,cases,exposure):
    result=dict(all=list(identities),latest=[i for i in identities if i in data['aggregation']['new_training_ids']],
        source_seen=[i for i in identities if exposure[i]['source_seen']],
        source_unseen=[i for i in identities if not exposure[i]['source_seen']])
    for round_index in sorted({cases[i]['collection_round'] for i in identities}):
        result[f'collection_{round_index:02d}']=[i for i in identities if cases[i]['collection_round']==round_index]
    return result


def prepare(root,role,collection,selected_frozen,source_frozen,output,include_collection=None):
    import numpy as np
    import torch
    import h5py
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    try:
        receipt=validate_inputs(root,role,selected_frozen)
        data=read(receipt['paths']['data']);union=include_collection is not None
        if union:
            require(include_collection,'Explicit union requires at least one additional collection')
            identities,union_cases,collections=bind_collection_union(data,collection,include_collection)
            extend_union_receipt(receipt,data,identities)
        selected=torch.load(receipt['paths']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        validate_checkpoint(selected,receipt)
        validate_batch_checkpoint_visits(selected)
        source,source_saved=bind_source(receipt,source_frozen,selected)
        collection=Path(collection).resolve(strict=True)
        require(read(collection/'result.json').get('software_complete'), 'Completed collection required')
        accepted=read(collection/'correction_entries.json')
        indexed={row['case']:row['result'] for row in read(collection/'results.json')}
        if union:source_exposure,source_manifest=source_training_exposure(source,source_saved,data,receipt['identities'])
        # The reader owns a monolithic numeric container; no validation/test
        # record is selected or evaluated, and no held-out raw/cache file opens.
        numeric=torch.load(Path(root)/role/'state_data/numeric.pt',map_location='cpu',weights_only=False,mmap=True)
        require(numeric['data_sha256']==receipt['sha256']['data'],'Numeric container binding differs')
        masks={};episodes={}
        for identity in receipt['identities']:
            entry=data['entries'][identity]
            if union:
                bound=union_cases[identity];result=bound['result'];path=Path(bound['numeric_path'])
            else:
                require(identity in accepted and accepted[identity]==entry,'Current correction differs from accepted collection entry')
                result=indexed[identity];path=collection/identity/'numeric.h5'
            require(result.get('legal') and result.get('accepted_for_training') and
                result['dataset_capture']['sha256']==entry['source_sha256'], 'Accepted capture binding differs')
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
                if union:
                    episodes[identity].update(collection_index=bound['collection_index'],collection_round=bound['collection_round'],
                        collection_case=bound['case'],latest=identity in data['aggregation']['new_training_ids'],
                        source_seen=source_exposure[identity]['source_seen'],
                        source_training_id=source_exposure[identity]['source_training_id'],
                        source_prior_full_history_visits=source_exposure[identity]['full_history_visits'])
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
        if union:
            strata=union_strata(receipt['identities'],data,union_cases,source_exposure)
            registered.update(version=UNION_VERSION,collections=collections,source_training_manifest=source_manifest,
                strata=strata,stratum_counts={name:dict(trajectories=len(ids),families=len({episodes[i]['family'] for i in ids}),
                    history_frames=sum(episodes[i]['history_frames'] for i in ids),
                    legal_expert_actions=sum(episodes[i]['legal_expert_actions'] for i in ids),
                    selected_full_history_visits=sum(episodes[i]['selected_full_history_visits'] for i in ids),
                    source_prior_full_history_visits=sum(episodes[i]['source_prior_full_history_visits'] for i in ids))
                    for name,ids in strata.items()},
                scope='DP frontend target error only; explicit train-only accepted collection union, latest and source-seen/unseen strata; full real histories separately rebuilt per source/selected EMA.')
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


def stratified_results(episodes,baseline_episodes,strata):
    result={}
    for name,identities in strata.items():
        require(len(identities)==len(set(identities)) and set(identities)<=set(episodes),
            'Registered stratum is not a unique evaluated episode subset')
        result[name]=dict(identities=identities,
            stages=finalized(sum_episode_totals({i:episodes[i] for i in identities})),
            baselines={key:dict(stages=finalized(sum_episode_totals({i:rows[i] for i in identities})))
                for key,rows in baseline_episodes.items()})
    return result


def run_arm(registration,arm):
    import numpy as np
    import torch
    path=Path(registration).resolve(strict=True);plan=read(path);out=path.parent/arm
    out.mkdir(exist_ok=False);start=time.perf_counter()
    exclusive_json(out/'started.json',dict(pid=os.getpid(),utc=time.time(),arm=arm,registration_sha256=sha256(path)))
    try:
        require(plan['version'] in (VERSION,UNION_VERSION) and arm in ('source','selected'),'Unknown stage-fit arm')
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
        if plan['version']==UNION_VERSION:
            for collection in plan['collections']:
                for name,digest in collection['files'].items():checked_hash(Path(collection['path'])/name,digest)
            checked_hash(plan['source_training_manifest']['path'],plan['source_training_manifest']['sha256'])
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
        if plan['version']==UNION_VERSION:
            result['strata']=stratified_results(episodes,baseline_episodes,plan['strata'])
        exclusive_json(out/'result.json',result);return result
    except BaseException as error:
        exclusive_json(out/'error.json',dict(error=repr(error),utc=time.time()));raise


def compare_stage_rows(arms,names):
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
    return rows


def compare(registration):
    path=Path(registration);plan=read(path);arms={arm:read(path.parent/arm/'result.json') for arm in ('source','selected')}
    version=plan.get('version',LEGACY_VERSION)
    require(version in (VERSION,LEGACY_VERSION,UNION_VERSION),'Unknown stage-fit comparison version')
    names=baseline_names(plan['role']) if version in (VERSION,UNION_VERSION) else []
    if names:require(plan['baseline_names']==names,'Registered baseline semantics changed')
    for arm,result in arms.items():
        require(result['complete'] and result['registration_sha256']==sha256(path) and
            result['checkpoint_sha256']==plan[arm]['sha256'],'Stage-fit result binding differs')
        if names:require(set(result.get('baselines',{}))==set(names),'Causal hold baseline result missing')
    result=dict(version=version,registration_sha256=sha256(path),stages=compare_stage_rows(arms,names),
        source_epoch=plan['source']['epoch'],selected_epoch=plan['selected']['epoch'],episodes=plan['episodes'],
        scope=plan['scope'],rear_evaluated=False,optimizer_updates=0,validation_evaluated=False,test_evaluated=False)
    if version==UNION_VERSION:
        for arm,value in arms.items():
            require(set(value['episodes'])==set(plan['identities']) and
                all(set(row['episodes'])==set(plan['identities']) for row in value['baselines'].values()),
                'Union evaluated episode population differs')
            rebuilt=stratified_results(value['episodes'],{key:row['episodes'] for key,row in value['baselines'].items()},plan['strata'])
            require(value.get('strata')==rebuilt,'Union stratum result differs from its bound episode totals')
            require(value['stages']==rebuilt['all']['stages'],'Union total differs from registered all stratum')
            require(all(value['baselines'][name]['stages']==rebuilt['all']['baselines'][name]['stages'] for name in names),
                'Union hold total differs from its bound episode totals')
        result['strata']={key:dict(identities=ids,counts=plan['stratum_counts'][key],
            stages=compare_stage_rows({arm:value['strata'][key] for arm,value in arms.items()},names))
            for key,ids in plan['strata'].items()}
        result['stratum_overlap']='Latest, per-collection and source-seen/unseen are overlapping views; only seen/unseen and distinct collections partition all.'
    exclusive_json(path.parent/'comparison.json',result);return result


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for option in ('root','collection','selected-frozen','source-frozen','output'):prep.add_argument('--'+option,type=Path,required=True)
    prep.add_argument('--role',choices=['A','B'],required=True)
    prep.add_argument('--include-collection',type=Path,action='append',
        help='Additional completed training collection; primary --collection remains exactly the latest cohort')
    run=sub.add_parser('run-arm');run.add_argument('--registration',type=Path,required=True);run.add_argument('--arm',choices=['source','selected'],required=True)
    comp=sub.add_parser('compare');comp.add_argument('--registration',type=Path,required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    result=prepare(**args) if command=='prepare' else run_arm(**args) if command=='run-arm' else compare(**args)
    print(json.dumps(dict(command=command,complete=True,scope=result.get('scope'))))


if __name__=='__main__':main()
