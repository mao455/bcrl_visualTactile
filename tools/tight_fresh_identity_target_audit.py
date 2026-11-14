"""CPU-only physical target increments on train99; no model/DINO/raw-image reads."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from visuotactile.compact import sha256
from visuotactile.anchor_sampling import planning_anchors,labels
from visuotactile.takeover_trigger import contact_height_windows
from visuotactile.training_resources import atomic_json

QUANTILES=(0.,.1,.5,.9,.99,1.)
THRESHOLDS=(.0001,.001,.005,.01)
DIRECTION_EPS_RAD=1e-5


def read(path):return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise ValueError(message)


def quantiles(value):
    return {str(q):float(v) for q,v in zip(QUANTILES,np.quantile(value,QUANTILES))} if value.size else None


def differences(value):
    require(value.ndim==2,'Expected tick x joint difference')
    if not len(value):return dict(ticks=0,joint_values=0,rmse_rad=None)
    absolute=np.abs(value);rms=np.sqrt(np.mean(value**2,axis=1))
    return dict(ticks=len(value),joint_values=value.size,rmse_rad=float(np.sqrt(np.mean(value**2))),
        mean_signed_rad=float(value.mean()),signed_joint_quantiles_rad=quantiles(value),
        absolute_joint_quantiles_rad=quantiles(absolute),tick_rms_quantiles_rad=quantiles(rms),
        joint_fraction_below_rad={str(t):float((absolute<=t).mean()) for t in THRESHOLDS},
        tick_rms_fraction_below_rad={str(t):float((rms<=t).mean()) for t in THRESHOLDS})


def directions(actual,command):
    require(actual.shape==command.shape and actual.ndim==2,'Aligned actual/command increments required')
    a_rms=np.sqrt(np.mean(actual**2,axis=1));u_rms=np.sqrt(np.mean(command**2,axis=1))
    moving=(a_rms>DIRECTION_EPS_RAD)&(u_rms>DIRECTION_EPS_RAD)
    a,u=actual[moving],command[moving]
    cosine=np.sum(a*u,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(u,axis=1)) if len(a) else np.array([])
    joint=(np.abs(actual)>DIRECTION_EPS_RAD)&(np.abs(command)>DIRECTION_EPS_RAD)
    dot=float(np.sum(actual*command));energy=float(np.sum(command**2))
    return dict(ticks=len(actual),direction_rms_floor_rad=DIRECTION_EPS_RAD,
        both_directions_resolvable=int(moving.sum()),actual_near_zero_ticks=int((a_rms<=DIRECTION_EPS_RAD).sum()),
        command_near_zero_ticks=int((u_rms<=DIRECTION_EPS_RAD).sum()),
        both_near_zero_ticks=int(((a_rms<=DIRECTION_EPS_RAD)&(u_rms<=DIRECTION_EPS_RAD)).sum()),
        vector_cosine_quantiles=quantiles(cosine),
        vector_positive_fraction=float((cosine>0).mean()) if len(cosine) else None,
        resolvable_joint_values=int(joint.sum()),
        joint_sign_agreement_fraction=float((np.sign(actual[joint])==np.sign(command[joint])).mean()) if joint.any() else None,
        least_squares_immediate_response_along_command=dot/energy if energy else None,
        interpretation='Descriptive one-tick motion direction/projection; not pressure, stiffness identification or causal contact attribution.')


def summarize(q0,q1,u):
    require(q0.shape==q1.shape==u.shape and q0.ndim==2 and q0.shape[1]==29,'Finite canonical29 states required')
    require(all(np.isfinite(v).all() for v in (q0,q1,u)),'Nonfinite target/state')
    result={}
    for part,index in (('arm',slice(0,7)),('hand',slice(7,29))):
        motion=q1[:,index]-q0[:,index];request=u[:,index]-q0[:,index]
        result[part]=dict(A_q_next_minus_q_before=differences(motion),
            B_u_minus_q_before=differences(request),u_minus_q_next=differences(u[:,index]-q1[:,index]),
            increment_direction=directions(motion,request))
    return result


def cached_arrays(record):
    valid=record['supervision_mask'].numpy()
    q0=record['robot'][:,:29].numpy();q1=record['q_reached'].numpy();u=record['u_expert'].numpy()
    require(valid.dtype==bool and valid.shape==(len(q0),) and q0.shape==q1.shape==u.shape,
        'Complete real supervised record required')
    require(np.array_equal(u[valid],record['command'].numpy()[valid]),'Expert command differs from actual issued target')
    # q_reached is intentionally zero on masked student prefixes. Validate only
    # legal nonterminal labels against the next actual pre-action observation.
    require(np.array_equal(q1[:-1][valid[:-1]],q0[1:][valid[:-1]]),
        'q_next is not the next actual observation on legal nonterminal transitions')
    clocks=record['time'].numpy()
    require(np.allclose(np.diff(clocks),1/60,rtol=0,atol=1e-8),'Control interval is not1/60second')
    return q0.astype(np.float64),q1.astype(np.float64),u.astype(np.float64),valid


def anchor_samples(record,scope,contract):
    """Exactly the trainer's onset-aware anchors and contiguous legal windows."""
    configured=dict(record,target_window_contract=contract)
    anchors=planning_anchors(configured)
    _,valid=labels(configured,anchors,24)
    a=anchors.numpy();v=valid.numpy()
    samples={}
    for offset in (0,5,11,23):
        target=a+offset
        inside=target<len(scope)
        keep=inside&v[:,offset]&scope[np.minimum(target,len(scope)-1)]
        samples[str(offset+1)]=(a[keep],target[keep])
    return samples


def phase_masks(path,record,entry):
    """Only completed accepted round06/07 small numeric.h5 diagnostics."""
    q0,q1,u,valid=cached_arrays(record)
    with h5py.File(path,'r') as stream:
        metadata=json.loads(stream.attrs['metadata_json'])
        require(abs(metadata['physics_dt']-1/120)<1e-8 and abs(metadata['control_dt']-1/60)<1e-8 and
            len(metadata['joint_order'])==29 and metadata['joint_order'][:7]==[f'iiwa14_joint_{i}' for i in range(1,8)],
            'Actual metadata timing or canonical joint ordering differs')
        t=stream['episode/transition'];before=t['diagnostics_before'];after=t['diagnostics_after']
        require(np.array_equal(t['dagger/expert_label_valid'][:].astype(bool),valid), 'Stage expert mask differs')
        require(np.array_equal(before['joint_position_rad'][:],q0) and
            np.array_equal(after['joint_position_rad'][:][valid],q1[valid]) and
            np.array_equal(t['target_after_rad'][:][valid],u[valid]),'Physical field/cache mapping differs')
        require(np.array_equal(t['time_before_s'][:],record['time'].numpy()) and
            np.allclose(t['time_after_s'][:]-t['time_before_s'][:],1/60,rtol=0,atol=1e-8),
            'Stage physical/control clock differs')
        pose=after['object_pose_w'][:];force=t['object_contact_force_w'][:]
        require(len(pose)>=48 and entry['takeover_tick']>=48,'Student-only settled reference is unavailable')
        height=float(np.median(pose[24:48,2]));held=contact_height_windows(pose,force,height);held[:47]=False
        lifted=after['lifted_object'][:].astype(bool);first=np.flatnonzero(lifted)
        prior=np.arange(len(valid))<(int(first[0]) if len(first) else len(valid))
        same=(before['goal_pose_w'][:]==after['goal_pose_w'][:]).all(-1)&(before['successes'][:]==after['successes'][:])
        near=held&same&(t['native_keypoints_max_dist'][:]<=.06)
        return (dict(legal_expert=valid,before_first_native_lift=valid&prior,
            sustained_contact_height12=valid&held,held_near_hole60=valid&near),
            dict(settled_height_m=height,numeric_sha256=sha256(path),metadata_control_dt_s=metadata['control_dt'],
                metadata_physics_dt_s=metadata['physics_dt'],joint_order=metadata['joint_order'],
                native_first_lift_completed_tick=int(first[0])+1 if len(first) else None))


def collect_summary(records,ids,scopes,contract):
    arrays=[cached_arrays(records[i]) for i in ids]
    selected=[(a[scope],b[scope],u[scope]) for (a,b,u,valid),scope in zip(arrays,scopes)]
    q0,q1,u=(np.concatenate([row[j] for row in selected]) for j in range(3))
    one_step=summarize(q0,q1,u)
    anchors={str(offset):[[],[],[]] for offset in (1,6,12,24)}
    for identity,(a,b,u,_),scope in zip(ids,arrays,scopes):
        for offset,(start,target) in anchor_samples(records[identity],scope,contract).items():
            for bucket,values in zip(anchors[offset],(a[start],b[target],u[target])):bucket.append(values)
    hold={}
    for offset,buckets in anchors.items():
        a,b,u=(np.concatenate(bucket) for bucket in buckets)
        hold[offset]=dict(legal_target_slots=len(a),target_step_offset=int(offset)-1,
            q_after_seconds_from_anchor=int(offset)/60,u_issue_seconds_from_anchor=(int(offset)-1)/60,
            **{part:dict(A_q_target_minus_anchor_q=differences((b-a)[:,index]),
                B_u_target_minus_anchor_q=differences((u-a)[:,index]))
                for part,index in [('arm',slice(0,7)),('hand',slice(7,29))]})
    return dict(episodes_with_actions=sum(bool(scope.any()) for scope in scopes),legal_expert_ticks=len(q0),
        one_control_step=one_step,native_5hz_anchor_hold_current=hold)


def run(pool,collections,output):
    output=Path(output);require(not output.exists(),'Preserve earlier CPU evidence')
    began=time.perf_counter();pool=Path(pool).resolve();data={};containers={}
    for role in ('A','B'):
        root=pool/role/'state_data';data[role]=read(root/'data.json')
        require(data[role]['supervision']==('q_reached' if role=='A' else 'u_expert'),'Unexpected target semantics')
        containers[role]=torch.load(root/'numeric.pt',map_location='cpu',weights_only=False,mmap=True)
        require(containers[role]['data_sha256']==sha256(root/'data.json'),'Cached data binding differs')
    require(data['A']['splits']==data['B']['splits'] and data['A']['entries']==data['B']['entries'], 'A/B populations differ')
    identities=data['A']['splits']['train']
    require(len(set(identities))==len(identities) and not set(identities)&set(data['A']['splits'].get('validation',[])+data['A']['splits'].get('test',[])),
        'Training population is not unique/disjoint')
    records=containers['A']['records']
    for identity in identities:
        require(set(records[identity])==set(containers['B']['records'][identity]) and all(
            torch.equal(value,containers['B']['records'][identity][key]) for key,value in records[identity].items()),
            'A/B training tensor values differ: '+identity)
        cached_arrays(records[identity])
    contracts={role:read(pool/role/'continuation.json').get('target_window_contract') for role in ('A','B')}
    require(contracts['A']==contracts['B'],'A/B native target-window contracts differ')
    contract=contracts['A'];staged={};stage_proofs={};bindings=[]
    for collection in map(Path,collections):
        require(read(collection/'result.json')['software_complete'],'Stage collection is incomplete')
        entries=read(collection/'correction_entries.json');indexed={r['case']:r['result'] for r in read(collection/'results.json')}
        for identity,entry in entries.items():
            require(identity in identities and entry==data['A']['entries'][identity] and entry['round_index'] in (6,7),
                'Stage audit permits only accepted round06/07 train entries')
            require(identity not in staged,'Repeated phase trajectory')
            result=indexed[identity]
            require(result.get('legal') and result.get('accepted_for_training') and result['dagger']['expert_takeover_success'] and
                result['dataset_capture']['sha256']==entry['source_sha256'],'Stage capture provenance differs')
            staged[identity],stage_proofs[identity]=phase_masks(collection/identity/'numeric.h5',records[identity],entry)
        bindings.append(dict(root=str(collection.resolve()),entries_sha256=sha256(collection/'correction_entries.json'),
            results_sha256=sha256(collection/'results.json')))
    require(staged,'Explicit accepted round06/07 phase cohort required')
    stage_ids=list(staged)
    summaries=dict(train_pool_legal_expert=collect_summary(records,identities,
        [records[i]['supervision_mask'].numpy() for i in identities],contract))
    for name in next(iter(staged.values())):
        summaries['round06_07_'+name]=collect_summary(records,stage_ids,[staged[i][name] for i in stage_ids],contract)
    result=dict(version='train_only_q_next_identity_target_audit_v1',pool=str(pool),
        dataset=dict(train_episodes=len(identities),original_demonstrations=sum(data['A']['entries'][i].get('round_index',0)==0 for i in identities),
            train_families=len({data['A']['entries'][i]['family'] for i in identities}),phase_episodes=len(stage_ids)),
        fields=dict(q_before='robot[:,0:29] = diagnostics_before/joint_position_rad; position history leaves these columns unchanged',
            q_next='q_reached = same transition diagnostics_after/joint_position_rad, only legal mask; adjacent q_before verified',
            u='u_expert = actual target_after_rad on legal expert ticks',units='radian; arm0:7, hand7:29',
            control_dt_s=1/60,physics_dt_s=1/120,planning_hz=5,anchor_target_window=contract),
        definitions=dict(before_first_native_lift='Expert ticks before the first recorded native lift, including any student prefix lift; distinct from2cm contact-height window.',
            sustained_contact_height12='Past12 actual transitions with >=2 fingers >0.1N in each physics sample and height>=median(student25:48)+2cm; rotation allowed.',
            held_near_hole60='Sustained holding, current goal/stage unchanged, native keypoint max distance<=60mm; success threshold unchanged.',
            direction='Near-zero vectors are counted separately and excluded from cosine/sign agreement.',
            anchor='Actual onset-aware planning_anchors/labels from trainer; only valid target slots. Offsets1/6/12/24 denote q(t+offset) relative to anchor q(t).',
            one_step='Copy-current at each actual tick is an offline baseline, not a trained or deployed policy.',
            caution='u-q_next combines tracking dynamics, limits and contact; it is not a pressure label. Small one-step motion alone does not make the24-step target trivial.'),
        summaries=summaries,phase_provenance=stage_proofs,collections=bindings,
        data_sha256={role:sha256(pool/role/'state_data/data.json') for role in ('A','B')},
        numeric_container_metadata='mmap parses shared train/validation container; only train tensors are dereferenced. No held-out trajectory/raw/DINO is selected or evaluated.',
        source_sha256={name:sha256(name) for name in ('tools/tight_fresh_identity_target_audit.py',
            'visuotactile/narrow_data.py','visuotactile/native_data.py','visuotactile/position_control.py',
            'visuotactile/anchor_sampling.py','visuotactile/target_windows.py')},
        seconds=time.perf_counter()-began,optimizer_updates=0,simulator_starts=0,cuda_initialized=torch.cuda.is_initialized(),
        dino_shards_opened=0,raw_image_files_opened=0,validation_evaluated=False,test_evaluated=False)
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--stage-collections',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=run(args.pool,args.stage_collections,args.output)
    print(json.dumps(dict(output=str(args.output),dataset=result['dataset'],seconds=result['seconds'])))


if __name__=='__main__':main()
