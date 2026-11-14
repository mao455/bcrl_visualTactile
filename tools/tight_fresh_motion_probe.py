"""CPU full-history increment diagnostic on fixed original train2 and validation6.

No training, rollout, threshold selection, or held-out test access. Prediction
increments are evaluated on recorded histories, never accumulated as a rollout.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tools.narrow_repair_audit import stages
from tools.tight_fresh_onset_report import array_sha
from tools.tight_fresh_study_onset import CPUAnalysis
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

EFFECTIVE_GT_THRESHOLD_RAD = 1e-4


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def phase_masks(contact, native_phases):
    contact=np.asarray(contact,dtype=bool)
    native_phases=np.asarray(native_phases)
    require(contact.ndim==1 and native_phases.shape==contact.shape, 'Aligned recorded diagnostic phases required')
    first=np.flatnonzero(contact);first=int(first[0]) if len(first) else len(contact)
    tick=np.arange(len(contact))
    masks=dict(all=np.ones(len(contact),bool),before_first_object_contact=tick<first,
        first_object_contact12=(tick>=first)&(tick<first+12),
        next_object_contact12=(tick>=first+12)&(tick<first+24),
        first_object_contact24=(tick>=first)&(tick<first+24),
        after_first_object_contact24=tick>=first+24)
    masks.update({'native_'+phase:native_phases==phase for phase in sorted(set(native_phases))})
    return masks


def plan_start_observations(previous, cursor, commit):
    require(previous.ndim==2 and commit==12 and cursor.dtype==torch.int64
        and torch.equal(cursor,torch.arange(len(previous))%commit),
        'Expected recorded contiguous 5Hz plan cursors at every 60Hz tick')
    return previous[torch.arange(len(previous))-cursor]


def motion_sums(prediction, target, previous, mask, plan_start=None):
    """Same-sign includes nearzero predictions; those fractions are nonexclusive."""
    require(prediction.shape==target.shape==previous.shape and prediction.ndim==2
        and mask.shape==prediction.shape[:1] and mask.dtype==torch.bool,
        'Matching complete predictions, targets, causal observations and bool mask required')
    require(all(torch.isfinite(x[mask]).all() for x in (prediction,target,previous)), 'Finite selected values required')
    p=(prediction.double()-previous.double())[mask]
    actual=(target.double()-previous.double())[mask]
    error=p-actual
    effective=actual.abs()>EFFECTIVE_GT_THRESHOLD_RAD
    same=(p*actual)>0
    nearzero=p.abs()<=EFFECTIVE_GT_THRESHOLD_RAD
    sums=dict(ticks=len(actual),dofs=prediction.shape[1],components=actual.numel(),
        error_squared_rad2=float(error.square().sum()),predicted_delta_squared_rad2=float(p.square().sum()),
        target_delta_squared_rad2=float(actual.square().sum()),dot_rad2=float((p*actual).sum()),
        effective_gt_components=int(effective.sum()),effective_gt_squared_rad2=float(actual[effective].square().sum()),
        same_direction_components=int((same&effective).sum()),
        same_direction_gt_squared_rad2=float(actual[same&effective].square().sum()),
        nearzero_prediction_on_effective_gt_components=int((nearzero&effective).sum()),
        opposite_prediction_on_effective_gt_components=int(((p*actual<0)&~nearzero&effective).sum()))
    if plan_start is not None:
        require(plan_start.shape==target.shape and torch.isfinite(plan_start[mask]).all(),
            'Finite aligned planning-start observations required')
        sums['plan_start_hold_error_squared_rad2']=float((target.double()-plan_start.double())[mask].square().sum())
    return sums


def summarize(rows):
    if not rows:return dict(ticks=0,components=0)
    dofs={r['dofs'] for r in rows};require(len(dofs)==1, 'Cannot mix arm/hand component denominators')
    keys=set(rows[0])-{'dofs'}
    require(all(set(row)-{'dofs'}==keys for row in rows), 'Cannot aggregate different baseline definitions')
    total={key:sum(row[key] for row in rows) for key in keys}
    n=total['components'];active=total['effective_gt_components'];actual=total['target_delta_squared_rad2']
    predicted=total['predicted_delta_squared_rad2'];effective=total['effective_gt_squared_rad2']
    def ratio(a,b):return a/b if b else None
    result=dict(total,dofs=next(iter(dofs)),prediction_rmse_rad=math.sqrt(total['error_squared_rad2']/n) if n else None,
        hold_current_observation_rmse_rad=math.sqrt(actual/n) if n else None,
        predicted_increment_rms_rad=math.sqrt(predicted/n) if n else None,
        actual_increment_rms_rad=math.sqrt(actual/n) if n else None,
        error_to_hold_ratio=math.sqrt(total['error_squared_rad2']/actual) if actual else None,
        predicted_to_actual_increment_rms_ratio=math.sqrt(predicted/actual) if actual else None,
        regression_gain=ratio(total['dot_rad2'],actual),
        increment_cosine=ratio(total['dot_rad2'],math.sqrt(actual*predicted)),
        effective_gt_component_fraction=ratio(active,n),
        effective_gt_motion_energy_fraction=ratio(effective,actual),
        same_direction_fraction=ratio(total['same_direction_components'],active),
        same_direction_gt_motion_energy_fraction=ratio(total['same_direction_gt_squared_rad2'],effective),
        nearzero_prediction_fraction_on_effective_gt=ratio(total['nearzero_prediction_on_effective_gt_components'],active),
        opposite_prediction_fraction_on_effective_gt=ratio(total['opposite_prediction_on_effective_gt_components'],active))
    if 'plan_start_hold_error_squared_rad2' in total:
        plan_hold=total['plan_start_hold_error_squared_rad2']
        result.update(hold_plan_start_observation_rmse_rad=math.sqrt(plan_hold/n) if n else None,
            error_to_plan_start_hold_ratio=math.sqrt(total['error_squared_rad2']/plan_hold) if plan_hold else None)
    return result


def check_population(data):
    original=data['normalization_train_ids']
    ids=dict(train=list(data['splits']['train'][:2]),validation=list(data['splits']['validation']))
    require(len(original)==30 and len(set(original))==30 and len(ids['train'])==2 and len(ids['validation'])==6
        and set(ids['train'])<=set(original), 'Fixed original train2 and validation6 required')
    require(len(set(ids['train']))==2 and len(set(ids['validation']))==6,
        'Unique train2 and validation6 identities required')
    require(ids['train']==list(original[:2]), 'Fixed original first2 training identities changed')
    families={split:{data['entries'][i]['family'] for i in selected} for split,selected in ids.items()}
    require(len(families['train'])==2, 'Unique train2 initial-state families required')
    all_train={data['entries'][i]['family'] for i in data['splits']['train']}
    require(len(families['validation'])==6 and not all_train&families['validation'], 'Validation family overlaps training')
    selected=set(ids['train']+ids['validation'])
    require(not selected&set(data['splits'].get('test',[])), 'Test trajectories must never enter this probe')
    return ids


def check_normalization_bindings(front, rear, index, statistics, residual_statistics, binding):
    config=rear['configuration']
    require(config['base_sha256']==index['base_sha256']==binding['base_checkpoint_sha256']
        and config['statistics_sha256']==front['configuration']['statistics_sha256']==index['statistics_sha256']==binding['statistics_sha256']
        and config['residual_statistics_sha256']==index['residual_statistics_sha256']==binding['residual_statistics_sha256']
        and config['data_sha256']==front['configuration']['data_sha256']==index['data_sha256']==binding['data_sha256'],
        'Checkpoint configuration or cache statistics binding differs')
    require(config['residual_normalization_identity']==index['residual_normalization_identity'],
        'Residual normalization identity differs')
    keys=('robot_mean','robot_std','action_scale','action_offset')
    require(all(torch.equal(state[key],statistics[key]) for state in (front['model'],front['ema']) for key in keys)
        and all(torch.equal(rear['model'][key],residual_statistics[key]) for key in keys)
        and all(torch.equal(statistics[key],residual_statistics[key]) for key in ('robot_mean','robot_std')),
        'Checkpoint normalization buffers differ from bound statistics')


def check_transition_timing(actual):
    before=actual['time_before'];after=actual['time_after'];observed=actual['time']
    require(observed.ndim==1 and before.shape==after.shape==observed.shape
        and torch.equal(before,observed) and torch.equal(after[:-1],before[1:])
        and torch.isfinite(before).all() and torch.isfinite(after).all()
        and torch.allclose(after-before,torch.full_like(before,1/60),atol=1e-9,rtol=0),
        'Recorded observation/transition timestamps do not align at60Hz')


def check_cache_alignment(record, numeric, actual, identity, label, binding):
    require(record['cache_identity']==binding, 'Exact per-episode cache identity differs')
    require(record['identity']==identity and record['nominal_supervision']==label
        and torch.equal(record['robot'],numeric['robot'])
        and torch.equal(record['time'],numeric['time']) and torch.equal(record['time'],actual['time'])
        and torch.equal(record['target'],actual['command'])
        and torch.equal(record['robot'][:,58:87],actual['previous'])
        and torch.equal(record['robot'][:,:29],actual['q']), 'Actual full-history cache or labels differ')
    n=len(actual['time'])
    require(all(value.dtype==torch.bool and value.shape==(n,) and value.all()
        for value in (record['supervision_mask'],numeric['supervision_mask']))
        and torch.equal(record['control_segment'],numeric['control_segment']),
        'Complete legal expert supervision and control segments required')
    require(torch.equal(record['base'],record['plans'][:,0]), 'Committed base differs from first remaining plan target')
    require(torch.equal(numeric['q_reached'],actual['q_after'])
        and torch.equal(numeric['u_expert'],actual['command'])
        and torch.equal(numeric['command'],actual['command']), 'Numeric labels differ from same recorded transition')
    require((label=='q_reached' and torch.equal(record['q_reached'],actual['q_after']))
        or (label=='u_expert' and record['q_reached'] is None), 'Cached q_reached differs from same recorded transition')


def report(root):
    from visuotactile.anchor_continue import residual_prediction
    from visuotactile.models.native_dp import load_residual
    torch.set_num_threads(2);began=time.monotonic()
    data_by_role={r:read(root/r/'state_data/data.json') for r in ('A','B')}
    ids=check_population(data_by_role['A'])
    require(check_population(data_by_role['B'])==ids, 'A/B selected population differs')
    raw={};source_hashes={};joint_names=None
    for identity in ids['train']+ids['validation']:
        entry=data_by_role['A']['entries'][identity];other=data_by_role['B']['entries'][identity]
        require(all(entry[k]==other[k] for k in ('source_sha256','path','episode','family')), 'A/B raw source differs')
        digest=sha256(entry['path']);require(digest==entry['source_sha256'], 'Actual demonstration bytes changed')
        source_hashes[identity]=digest
        with h5py.File(entry['path']) as f:
            group=f['episodes'][entry['episode']];metadata=json.loads(f.attrs['metadata_json'])
            native=np.asarray(stages(group));force=group['transition/object_contact_force_w'][:]
            n=len(native);contact=np.linalg.norm(force,axis=-1).reshape(n,-1).max(-1)>.1
            command=torch.from_numpy(group['transition/target_after_rad'][:])
            previous=torch.from_numpy(group['obs/actuator_target_rad'][:])
            q=torch.from_numpy(group['transition/diagnostics_before/joint_position_rad'][:])
            q_after=torch.from_numpy(group['transition/diagnostics_after/joint_position_rad'][:])
            require(torch.equal(command[:-1],previous[1:]) and torch.equal(q_after[:-1],q[1:]),
                'Recorded actual target/state does not continue across transitions')
            names=metadata['joint_order'];require(len(names)==29, 'Canonical29 joint names required')
            require(joint_names is None or names==joint_names, 'Source canonical joint order differs')
            joint_names=names
            raw[identity]=dict(command=command,previous=previous,q=q,q_after=q_after,
                time=torch.from_numpy(group['obs/time_s'][:]),
                time_before=torch.from_numpy(group['transition/time_before_s'][:]),
                time_after=torch.from_numpy(group['transition/time_after_s'][:]),
                phase_masks=phase_masks(contact,native),native_phases=native.tolist())
            check_transition_timing(raw[identity])
    roles={}
    for role in ('A','B'):
        directory=root/role;data=data_by_role[role];data_path=directory/'state_data/data.json'
        numeric_path=directory/'state_data/numeric.pt';numeric=torch.load(numeric_path,map_location='cpu',weights_only=False)
        require(numeric['data_sha256']==sha256(data_path), 'Numeric input binding differs')
        index_path=directory/'backend'/role/'nominal/index.json';index=read(index_path)
        rear=directory/'backend'/role/'training';saved=torch.load(rear/'best.pt',map_location='cpu',weights_only=False)
        config=saved['configuration'];selected=read(rear/'selected.json');digest=sha256(rear/'best.pt')
        require(read(rear/'finished.json')['target_reached'] and selected['checkpoint_sha256']==digest
            and selected['base_checkpoint_sha256']==index['base_sha256']==sha256(directory/'training/best.pt')
            and config['cache_index_sha256']==sha256(index_path)
            and config['data_sha256']==index['data_sha256']==sha256(data_path), 'Selected complete pair/cache/data binding differs')
        front=torch.load(directory/'training/best.pt',map_location='cpu',weights_only=False)
        label='q_reached' if role=='A' else 'u_expert'
        require(front['contract']==saved['contract'] and front['contract']['supervision']==label
            and front['contract']['visual_stride']==1 and front['contract']['commit']==index['sampler']['commit']==12,
            'Complete pair frontend contract differs')
        binding=dict(base_checkpoint_sha256=sha256(directory/'training/best.pt'),
            statistics_sha256=sha256(directory/'state_data/statistics.pt'),
            data_sha256=sha256(data_path),residual_statistics_sha256=sha256(config['residual_statistics_path']))
        require(sha256(config['base_checkpoint'])==binding['base_checkpoint_sha256'], 'Configured frontend checkpoint changed')
        statistics=torch.load(directory/'state_data/statistics.pt',map_location='cpu',weights_only=False)
        residual_statistics=torch.load(config['residual_statistics_path'],map_location='cpu',weights_only=False)
        check_normalization_bindings(front,saved,index,statistics,residual_statistics,binding)
        model=load_residual(saved,'cpu').eval().requires_grad_(False)
        buckets=defaultdict(list);joint_buckets=defaultdict(list);cursor_buckets=defaultdict(list);cases=[];case_joints=[]
        for split,identities in ids.items():
            for identity in identities:
                path=index_path.parent/(identity+'.pt');cache_sha=sha256(path)
                require(cache_sha==index['episodes'][identity]['sha256'], 'Exact current cache shard changed')
                record=torch.load(path,map_location='cpu',weights_only=False);actual=raw[identity];num=numeric['records'][identity]
                check_cache_alignment(record,num,actual,identity,label,binding)
                prediction=residual_prediction(model,record,CPUAnalysis())
                require(prediction.shape==actual['command'].shape, 'Incomplete causal history prediction')
                forecasts=dict(backend=(prediction,actual['command'],actual['previous']),
                    dp=(record['base'],actual['q_after'] if role=='A' else actual['command'],
                        actual['q'] if role=='A' else actual['previous']))
                item=dict(identity=identity,split=split,family=data['entries'][identity]['family'],
                    ticks=len(prediction),cache_sha256=cache_sha,prediction_sha256=array_sha(prediction.numpy()),metrics={})
                for predictor,(pred,target,previous) in forecasts.items():
                    plan_start=plan_start_observations(previous,record['plan_cursor'],12) if predictor=='dp' else None
                    item['metrics'][predictor]={}
                    for phase,mask in actual['phase_masks'].items():
                        mask=torch.from_numpy(mask)
                        item['metrics'][predictor][phase]={}
                        for part,section in [('arm',slice(0,7)),('hand',slice(7,29))]:
                            sums=motion_sums(pred[:,section],target[:,section],previous[:,section],mask,
                                plan_start[:,section] if plan_start is not None else None)
                            buckets[split,predictor,phase,part].append(sums)
                            item['metrics'][predictor][phase][part]=summarize([sums])
                        if predictor=='backend' and phase in ('all','first_object_contact12','next_object_contact12','first_object_contact24'):
                            for joint,name in enumerate(joint_names):
                                sums=motion_sums(pred[:,joint:joint+1],target[:,joint:joint+1],previous[:,joint:joint+1],mask)
                                joint_buckets[split,phase,name].append(sums)
                                if split=='validation' and phase=='first_object_contact24' and any(finger in name for finger in ('index','thumb','ring')):
                                    case_joints.append(dict(identity=identity,family=item['family'],joint=name,**summarize([sums])))
                    if predictor=='dp':
                        for cursor in range(12):
                            mask=record['plan_cursor']==cursor
                            for part,section in [('arm',slice(0,7)),('hand',slice(7,29))]:
                                cursor_buckets[split,cursor,part].append(motion_sums(
                                    pred[:,section],target[:,section],previous[:,section],mask,plan_start[:,section]))
                cases.append(item)
                print(json.dumps(dict(role=role,split=split,identity=identity,ticks=len(prediction))),flush=True)
        roles[role]=dict(frontend_label=label,frontend_sha256=index['base_sha256'],backend_sha256=digest,
            verified_cache_identity=binding,
            binding_checks='Exact cache identity; config base/statistics hashes; frontend model and EMA/backend normalization buffers; observation/transition/cache timestamps; next-step q_reached; first remaining plan; complete legal masks and control segments.',
            selected=selected,data_sha256=sha256(data_path),numeric_sha256=sha256(numeric_path),
            cache_index_sha256=sha256(index_path),cases=cases,
            validation_contact24_case_joint_metrics=case_joints,
            metrics=[dict(split=split,predictor=predictor,phase=phase,part=part,**summarize(values))
                for (split,predictor,phase,part),values in sorted(buckets.items())],
            dp_cursor_metrics=[dict(split=split,cursor=cursor,part=part,**summarize(values))
                for (split,cursor,part),values in sorted(cursor_buckets.items())],
            joint_metrics=[dict(split=split,phase=phase,joint=joint,**summarize(values))
                for (split,phase,joint),values in sorted(joint_buckets.items())])
    require(not torch.cuda.is_initialized(), 'CPU diagnostic initialized CUDA')
    return dict(version='fixed_train2_validation6_motion_probe_v4',roles=roles,ids=ids,
        raw_source_sha256=source_hashes,source_sha256=sha256(__file__),
        dependencies_sha256={name:sha256(Path(__file__).resolve().parents[1]/name) for name in
            ('visuotactile/anchor_continue.py','visuotactile/models/native_dp.py','visuotactile/native_data.py',
             'tools/narrow_repair_audit.py','tools/tight_fresh_study_onset.py','tools/tight_fresh_onset_report.py')},
        effective_gt_threshold_rad=EFFECTIVE_GT_THRESHOLD_RAD,
        threshold_rule='One fixed threshold before analysis: abs(GT increment)>1e-4 rad. Report component and motion-energy coverage plus continuous cosine/gain/RMS.',
        semantics=dict(A_dp='q(t+1)-observed q(t)',B_dp='expert absolute target(t)-actual pre-action target(t)',
            backend='composed expert-command prediction(t)-actual pre-action target(t), same command space for A/B',
            hold='Hold the actual current observation relevant to that predictor, not a simulated autonomous policy',
            plan_start_hold='DP-only fair 5Hz information baseline: hold observed q for A or actual pre-action target for B from that chunk planning tick. Verified recorded cursor0..11; all12 cursors reported. Current-observation hold additionally uses fresh60Hz observations unavailable to the frozen chunk.',
            cursor0='First predicted A target is actual configuration after the next60Hz control transition; B is the expert absolute command target after the current action. Neither first target is required to equal the current q or pre-action target. Hold comparisons measure prediction accuracy, not an equality contract.',
            contact='Real same-transition object-contact force>0.1N at either physics sample; diagnostic phase labels only, never student inputs',
            masks='Before contact/first12/next12/after24 partition each episode. First24 combines first12+next12; native stages and all are separate overlapping summaries.',
            ratios='Gain and cosine use all components without the threshold. Direction and nearzero fractions use effective GT components. Nearzero prediction means abs(delta)<=1e-4 rad. Same-direction counts any same-sign prediction, including nearzero predictions; these fractions are nonexclusive and must not be summed as a partition. Opposite-direction excludes nearzero predictions. These original metric definitions are unchanged.'),
        scope='Original fixed first2 training episodes and same6 validation families used for model selection, not untouched test. '
            'Each selected backend recomputes complete recorded history with current parameters. DP commitments come from hash-verified current full-frame caches. '
            'Offline increments and their sums do not execute a rollout or identify causal task failure. A/B DP label spaces differ; compare DP only within each role.',
        device='cpu',threads=2,cuda_initialized=False,optimizer_updates=0,simulator_starts=0,test_used=False,
        backend_episode_forward_calls=16,seconds=time.monotonic()-began)


def compact_report(full, full_sha256):
    """Lossless columnar selected tables; full report remains a local artifact."""
    def table(rows):
        columns=sorted(set().union(*(set(row) for row in rows)))
        return dict(columns=columns,rows=[[row.get(key) for key in columns] for row in rows])
    output={key:value for key,value in full.items() if key!='roles'}
    output.update(full_report_sha256=full_sha256,
        compact_scope='All aggregate metrics and all12 DP cursor metrics; per-family arm/hand metrics for all/first12/next12/first24 contact; validation all29 joint metrics and first24 per-family index/thumb/ring joints. Other case phases and train2 joint detail remain in the original full report. Tables map columns to each row without numeric rounding.',roles={})
    phases={'all','first_object_contact12','next_object_contact12','first_object_contact24'}
    for role,data in full['roles'].items():
        copied={key:value for key,value in data.items() if key not in ('cases','metrics','joint_metrics','dp_cursor_metrics','validation_contact24_case_joint_metrics')}
        copied['metrics']=table(data['metrics']);copied['dp_cursor_metrics']=table(data['dp_cursor_metrics'])
        copied['joint_metrics']=table([row for row in data['joint_metrics'] if row['split']=='validation'])
        copied['validation_contact24_case_joint_metrics']=table(data.get('validation_contact24_case_joint_metrics',[]))
        copied['cases']=[{key:value for key,value in case.items() if key!='metrics'} for case in data['cases']]
        copied['case_metrics']=table([dict(identity=case['identity'],predictor=predictor,phase=phase,part=part,**values)
            for case in data['cases'] for predictor,by_phase in case['metrics'].items()
            for phase,by_part in by_phase.items() if phase in phases for part,values in by_part.items()])
        output['roles'][role]=copied
    return output


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--compact-output',type=Path);args=parser.parse_args()
    if args.output.exists() or (args.compact_output and args.compact_output.exists()):
        raise FileExistsError('Preserve earlier diagnostic result')
    full=report(args.root.resolve());atomic_json(args.output,full)
    if args.compact_output:
        atomic_json(args.compact_output,compact_report(full,sha256(args.output)))


if __name__=='__main__':main()
