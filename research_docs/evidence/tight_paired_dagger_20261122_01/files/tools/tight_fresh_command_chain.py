"""CPU-only execution-chain and grasp-motion decomposition of saved autonomous runs.

No expert counterfactual or model forward is used. Direction comparisons describe
how the recorded stages differ; they cannot identify the correct expert action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch

from tools.tight_fresh_phases import case as phase_case
from visuotactile.compact import sha256
from visuotactile.execution_learning import ideal_execution_bounds
from visuotactile.position_control import project_processed_position_targets
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
from visuotactile.takeover_trigger import stable_windows
from visuotactile.training_resources import atomic_json

VERSION = 'recorded_autonomous_command_chain_v1'
SEMANTIC_REVISION = 2
TOLERANCE_RAD = 2e-6
DIRECTION_EPSILON_RAD = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def quat_mul(a, b):
    return np.concatenate((a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1, keepdims=True),
        a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+np.cross(a[..., 1:], b[..., 1:])), -1)


def quat_inverse(q):
    result = q.copy(); result[..., 1:] *= -1
    return result


def rotate(q, v):
    cross = 2*np.cross(q[..., 1:], v)
    return v+q[..., :1]*cross+np.cross(q[..., 1:], cross)


def relative_pose(obj, palm):
    inverse = quat_inverse(palm[..., 3:])
    return np.concatenate((rotate(inverse, obj[..., :3]-palm[..., :3]),
        quat_mul(inverse, obj[..., 3:])), -1)


def compose_pose(parent, local):
    return np.concatenate((parent[..., :3]+rotate(parent[..., 3:], local[..., :3]),
        quat_mul(parent[..., 3:], local[..., 3:])), -1)


def angle(a, b):
    # Stored simulator quaternions may differ from unit norm by FP32 roundoff.
    a = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b/np.linalg.norm(b, axis=-1, keepdims=True)
    return 2*np.arccos(np.clip(abs((a*b).sum(-1)), 0, 1))


def direction(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    require(reference.shape == actual.shape and reference.size > 0, 'Matching nonempty motion arrays required')
    active = (abs(reference) > DIRECTION_EPSILON_RAD) & (abs(actual) > DIRECTION_EPSILON_RAD)
    dot = float((reference*actual).sum()); energy = float((reference**2).sum())
    norm = float(np.sqrt(energy*(actual**2).sum()))
    return dict(components=reference.size, reference_rms_rad=rms(reference), actual_rms_rad=rms(actual),
        difference_rms_rad=rms(actual-reference), cosine=dot/norm if norm else None,
        regression_gain=dot/energy if energy else None, both_active_components=int(active.sum()),
        opposite_active_components=int(((reference*actual < 0)&active).sum()),
        opposite_fraction_among_both_active=float((reference[active]*actual[active] < 0).mean()) if active.any() else None,
        active_reference_energy_fraction=float((reference[active]**2).sum()/energy) if energy else None)


def grasp_motion(before_obj, before_palm, after_obj, after_palm, goal):
    start_relative = relative_pose(before_obj, before_palm)
    end_relative = relative_pose(after_obj, after_palm)
    rigid = compose_pose(after_palm, start_relative)
    return dict(goal_reference='window_start_preaction_goal_pose',
        palm_translation_m=float(np.linalg.norm(after_palm[:3]-before_palm[:3])),
        palm_rotation_rad=float(angle(after_palm[3:], before_palm[3:])),
        object_translation_m=float(np.linalg.norm(after_obj[:3]-before_obj[:3])),
        object_rotation_rad=float(angle(after_obj[3:], before_obj[3:])),
        relative_palm_translation_m=float(np.linalg.norm(end_relative[:3]-start_relative[:3])),
        relative_palm_rotation_rad=float(angle(end_relative[3:], start_relative[3:])),
        actual_window_start_goal_root_distance_m=[float(np.linalg.norm(p[:3]-goal[:3])) for p in (before_obj, after_obj)],
        actual_window_start_goal_root_angle_rad=[float(angle(p[3:], goal[3:])) for p in (before_obj, after_obj)],
        rigid_attachment_endpoint_window_start_goal_root_distance_m=float(np.linalg.norm(rigid[:3]-goal[:3])),
        rigid_attachment_endpoint_window_start_goal_root_angle_rad=float(angle(rigid[3:], goal[3:])))


def window_goal_identity(before, after, start, stop, *, subgoals_before=None, subgoals_after=None):
    require(before.shape==after.shape and before.ndim==2 and before.shape[1]==7
        and 0<=start<stop<=len(before) and np.isfinite(before).all() and np.isfinite(after).all(),
        'Finite matching before/after goal poses and valid window required')
    b,a=before[start:stop],after[start:stop]
    def digest(value):
        value=np.ascontiguousarray(value)
        return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
    prechange=(np.flatnonzero(np.any(b[1:]!=b[:-1],axis=1))+start+2).tolist()
    postchange=(np.flatnonzero(np.any(a!=b,axis=1))+start+1).tolist()
    result=dict(goal_reference='window_start_preaction_goal_pose',window_start_goal_pose=before[start].tolist(),
        goal_before_pose_sha256=digest(b),goal_after_pose_sha256=digest(a),
        recorded_goal_pose_changed_within_window=bool(prechange or postchange),
        preaction_goal_pose_change_completed_ticks=prechange,
        posttransition_goal_pose_change_completed_ticks=postchange,
        maximum_same_transition_goal_root_translation_m=float(np.linalg.norm(a[:,:3]-b[:,:3],axis=-1).max()),
        maximum_same_transition_goal_root_rotation_rad=float(angle(a[:,3:],b[:,3:]).max()),
        change_rule='Exact recorded pose components, not quaternion-sign-adjusted geometry; q versus -q can flag change without a new physical goal. Tick identifies its control transition.',
        native_keypoint_goal_reference='Each recorded transition native target; not recomputed against the frozen window-start target.',
        native_keypoint_start_end_same_preaction_goal=bool(np.array_equal(b[0],b[-1])))
    if subgoals_before is not None or subgoals_after is not None:
        sb,sa=np.asarray(subgoals_before).reshape(-1),np.asarray(subgoals_after).reshape(-1)
        require(sb.shape==sa.shape==(len(before),),'Matching before/after native subgoal counters required')
        result.update(native_subgoal_transition_completed_ticks=(np.flatnonzero(sa[start:stop]!=sb[start:stop])+start+1).tolist(),
            native_subgoal_counter_start_before_end_after=[int(sb[start]),int(sa[stop-1])])
    return result


def admitted_case(evaluation, registered):
    """Full provenance admission allowing recorded native goal transitions."""
    plan=read(evaluation/'plan.json');directory=evaluation/registered['name']
    require(plan.get('kind')=='autonomous' and plan.get('test_used') is False
        and registered in plan['cases'] and registered['spec'].get('split')=='development'
        and read(evaluation/'result.json').get('software_complete'), 'Completed registered autonomous cohort required')
    pair=plan['pairs'][registered['role']];result=read(directory/'result.json')
    indexed=[r['result'] for r in read(evaluation/'results.json') if r['case']==registered['name']]
    require(indexed==[result] and result.get('legal') and result.get('software_complete')
        and result.get('dagger') is None and not result.get('censored') and not result.get('worker_error')
        and result['residual_ticks']==result['steps'] and result['checkpoint_sha256']==pair['backend_sha256']
        and result['simulation_step_hz']==60 and result['plan_hz']==5
        and all(result[k]==v for k,v in registered['spec'].items()),'Actual full-student case identity differs')
    for component in ('frontend','backend'):
        require(sha256(pair[component])==pair[component+'_sha256'],'Registered selected model changed')
    phase=phase_case(directory)
    require(phase['steps']==result['steps'] and phase['authority']['expert_ticks']==0,
        'Actual complete student authority required')
    return dict(pair=pair,phase=phase)


def first_lost_window(stable, pose, palm, force, height):
    found = np.flatnonzero(stable)
    if not len(found):
        return None
    after = np.flatnonzero((np.arange(len(stable)) > found[0]) & ~stable)
    if not len(after):
        return None
    stop = int(after[0])+1; start = stop-12
    local = relative_pose(pose[start:stop], palm[start:stop])
    drift = np.linalg.norm(local[:, :3]-local[0, :3], axis=-1).max()
    # Match stable_windows exactly, including its historical FP32 quaternion
    # arithmetic; descriptive motion angles elsewhere normalize the quaternion.
    rotation = (2*np.arccos(np.clip(abs((local[:, 3:]*local[0, 3:]).sum(-1)),0,1))).max()
    two = (np.linalg.norm(force[start:stop], axis=-1) > .1).sum(-1) >= 2
    return dict(completed_tick=stop, first_stable_completed_tick=int(found[0])+1,
        all_two_contact_fingers_both_substeps=bool(two.all()),
        all_above_settled_plus_2cm=bool((pose[start:stop, 2] >= height+.02).all()),
        max_relative_translation_m=float(drift), max_relative_rotation_rad=float(rotation),
        translation_condition=bool(drift <= .01), rotation_condition=bool(rotation <= .2),
        interpretation='Failure of a past12 diagnostic window is not proof of physical dropping.')


def compare_recorded(a, b, label, exact=False):
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label+' shape/finite mismatch')
    error = float(np.max(abs(a.astype(np.float64)-b.astype(np.float64))))
    require(np.array_equal(a, b) if exact else error <= TOLERANCE_RAD, label+' differs')
    return error


def analyze_case(evaluation, registered, backend):
    name = registered['name']; directory = evaluation/name
    admitted = admitted_case(evaluation, registered)
    phase = admitted['phase']; n = phase['steps']
    metadata = read(directory/'metadata.json'); config = backend['configuration']
    require(config['output_contract'] == REACHABLE_CENTERED_OUTPUT,
        'This decomposition requires the nominal-centered output contract')
    require(backend['contract']['supervision'] in ('q_reached', 'u_expert'), 'Explicit DP label semantics required')
    files = [directory/p for p in ('numeric.h5','trace.npz','plans.npz','metadata.json','result.json')]
    binding = {p.name: sha256(p) for p in files}
    with np.load(directory/'trace.npz') as z:
        t = {k: z[k].copy() for k in z.files}
    with np.load(directory/'plans.npz') as z:
        plans = z['plans'].copy()
    with h5py.File(directory/'numeric.h5') as f:
        e=f['episode']; transition=e['transition']; before=transition['diagnostics_before']; after=transition['diagnostics_after']
        a = dict(previous=e['obs/actuator_target_rad'][:], q_before=before['joint_position_rad'][:],
            q_after=after['joint_position_rad'][:], requested=transition['position_requested_rad'][:],
            applied=transition['position_applied_rad'][:], target=transition['target_after_rad'][:],
            actuator=transition['actuator_substep_target_rad'][:], obj_before=before['object_pose_w'][:],
            obj=after['object_pose_w'][:], palm_before=before['palm_body_pose_w'][:], palm=after['palm_body_pose_w'][:],
            goal=before['goal_pose_w'][:],goal_after=after['goal_pose_w'][:],
            subgoals_before=before['successes'][:],subgoals_after=after['successes'][:],force=transition['object_contact_force_w'][:],
            time=e['obs/time_s'][:], time_before=transition['time_before_s'][:], time_after=transition['time_after_s'][:],
            position_clip=transition['position_clip_mask'][:], delay=transition['control_trace/delay_enabled'][:],
            keypoint=transition['native_keypoints_max_dist'][:])
    require(plans.shape == ((n+11)//12, 24, 29), 'Recorded5Hz H24 plan count differs')
    require(np.array_equal(a['time'], a['time_before']) and np.array_equal(a['time_after'][:-1], a['time_before'][1:])
        and np.allclose(a['time_after']-a['time_before'], 1/60, atol=1e-9, rtol=0), 'Native observation/transition clock mismatch')
    checks = {}
    for left, right in [('requested','command'),('q_after','joint'),('target','target'),('actuator','actuator_targets')]:
        checks[left+'_trace'] = compare_recorded(a[left], t[right], left+' trace', exact=True)
    for label, first, second in [('issued_applied',a['requested'],a['applied']),('applied_target',a['applied'],a['target']),
            ('substep0_target',a['actuator'][:,0],a['target']),('substep1_target',a['actuator'][:,1],a['target']),
            ('previous_acknowledgement',a['previous'][1:],a['target'][:-1]),
            ('joint_continuity',a['q_before'][1:],a['q_after'][:-1]),
            ('object_continuity',a['obj_before'][1:],a['obj'][:-1]),
            ('palm_continuity',a['palm_before'][1:],a['palm'][:-1]),
            ('goal_continuity',a['goal'][1:],a['goal_after'][:-1]),
            ('execution_previous',a['previous'],t['execution_previous_target_rad'])]:
        checks[label] = compare_recorded(first, second, label, exact=True)
    require(not a['delay'].any() and not a['position_clip'].any(), 'Unexpected post-student delay or joint clip')
    limits = torch.tensor(metadata['joint_limits_rad'],dtype=torch.float32)
    if limits.shape == (1,29,2):
        limits = limits[0]
    require(limits.shape == (29,2), 'One canonical29-joint limit array required')
    previous = torch.from_numpy(a['previous'])
    bounds = ideal_execution_bounds(previous, limits, config['execution_adapter']).numpy()
    plan = plans[np.arange(n)//12, np.arange(n)%12]
    nominal = np.maximum(np.minimum(plan, limits[:,1].numpy()), limits[:,0].numpy())
    checks['plan_to_nominal'] = compare_recorded(nominal, t['nominal'], 'DP-to-nominal', exact=True)
    anchor = np.maximum(np.minimum(nominal, bounds[...,1]), bounds[...,0])
    proposal = nominal+t['residual_proposed_rad']
    joint = np.maximum(np.minimum(proposal,limits[:,1].numpy()),limits[:,0].numpy())
    checks['recorded_joint_projection'] = compare_recorded(joint,t['execution_preprojection_target_rad'],'joint projection')
    final = project_processed_position_targets(torch.from_numpy(joint),previous,limits,config['execution_adapter']).numpy()
    checks['reconstructed_final'] = compare_recorded(final,a['requested'],'final reachable projection')
    checks['actual_backend_effect'] = compare_recorded(a['requested']-nominal,t['residual_applied_rad'],'backend effect')
    checks['recorded_final_projection'] = compare_recorded(a['requested']-joint,t['execution_projection_rad'],'projection diagnostic')
    scale=backend['model']['action_scale'].numpy()
    checks['normalized_delta_units'] = compare_recorded(t['residual_proposed_rad'],t['residual_normalized']/scale,'delta units')
    raw = (proposal-anchor)/(bounds[...,1]-bounds[...,0])
    require(np.isfinite(raw).all(), 'Finite recovered linear head output required')
    stable, contact = stable_windows(a['obj'],a['palm'],a['force'],float(np.median(a['obj'][24:48,2])))
    require(int(stable.sum()) == phase['grasp_progress']['stable_ticks'], 'Existing stable diagnostic differs')
    first_contact = np.flatnonzero(contact.any((1,2)))
    windows = [('all',0,n)]
    if len(first_contact):
        start=int(first_contact[0]); windows.append(('first_contact12',start,min(n,start+12)))
    held=np.flatnonzero(stable)
    if len(held):
        first,last=int(held[0]),int(held[-1]); windows.append(('first_stable_window',first-11,first+1))
        for duration in (12,24,48):
            windows.append(('after_first_stable'+str(duration),first+1,min(n,first+1+duration)))
        windows.append(('last_stable_window',last-11,last+1))
        for duration in (12,24,48):
            windows.append(('after_last_stable'+str(duration),last+1,min(n,last+1+duration)))
    best=int(np.argmin(a['keypoint'])); windows.append(('native_closest12',max(0,best-11),best+1))
    summaries=[]
    for label,start,stop in windows:
        if stop <= start:continue
        sl=slice(start,stop); parts={}
        for part,indices in [('arm',slice(0,7)),('hand',slice(7,29))]:
            def v(x):return x[sl,indices]
            target_delta=a['requested']-a['previous']; actual_delta=a['q_after']-a['q_before']
            parts[part]=dict(ticks=stop-start,
                dp_joint_clip_rms_rad=rms(v(nominal-plan)),
                nominal_reachable_clip_rms_rad=rms(v(anchor-nominal)),
                nominal_reachable_changed_fraction=float((abs(v(anchor-nominal))>TOLERANCE_RAD).mean()),
                linear_head_raw_rms=rms(v(raw)),
                backend_from_reachable_anchor_rms_rad=rms(v(proposal-anchor)),
                proposal_joint_clip_rms_rad=rms(v(joint-proposal)),
                final_reachable_clip_rms_rad=rms(v(a['requested']-joint)),
                final_reachable_changed_fraction=float((abs(v(a['requested']-joint))>TOLERANCE_RAD).mean()),
                final_at_reachable_boundary_fraction=float(((abs(v(a['requested']-bounds[...,0]))<=TOLERANCE_RAD)
                    |(abs(v(a['requested']-bounds[...,1]))<=TOLERANCE_RAD)).mean()),
                issued_increment_rms_rad=rms(v(target_delta)),actual_increment_rms_rad=rms(v(actual_delta)),
                existing_tracking_error_rms_rad=rms(v(a['q_before']-a['previous'])),
                after_tracking_error_rms_rad=rms(v(a['q_after']-a['target'])),
                requested_gap_before_rms_rad=rms(v(a['requested']-a['q_before'])),
                nominal_vs_final_increment=direction(v(nominal-a['previous']),v(target_delta)),
                reachable_anchor_vs_final_increment=direction(v(anchor-a['previous']),v(target_delta)),
                proposal_vs_final_increment=direction(v(proposal-a['previous']),v(target_delta)),
                desired_joint_motion_vs_actual=direction(v(a['requested']-a['q_before']),v(actual_delta)),
                dp_vs_reached_rms_rad=rms(v(nominal-a['q_after'])))
            error=(a['q_before']-a['previous'])-target_delta+actual_delta
            require(np.max(abs(v(error-(a['q_after']-a['target']))))<2e-6, 'Tracking decomposition identity failed')
        motion=grasp_motion(a['obj_before'][start],a['palm_before'][start],a['obj'][stop-1],a['palm'][stop-1],a['goal'][start])
        motion.update(window_goal_identity(a['goal'],a['goal_after'],start,stop,
            subgoals_before=a['subgoals_before'],subgoals_after=a['subgoals_after']))
        local=relative_pose(a['obj'][sl],a['palm'][sl]); initial=relative_pose(a['obj_before'][start],a['palm_before'][start])
        motion.update(max_relative_translation_m=float(np.linalg.norm(local[:,:3]-initial[:3],axis=-1).max()),
            max_relative_rotation_rad=float(angle(local[:,3:],initial[3:]).max()),
            native_keypoint_start_end_m=[float(a['keypoint'][start]),float(a['keypoint'][stop-1])],
            both_two_contact_ticks=int(((contact[sl].sum(-1)>=2).all(-1)).sum()),stable_window_endpoints=int(stable[sl].sum()))
        tracking=np.sqrt(np.mean((a['q_after'][sl]-a['target'][sl]).astype(np.float64)**2,axis=0))
        top=np.argsort(-tracking[7:],kind='stable')[:6]+7
        joints=[dict(joint=metadata['joint_order'][j],joint_index=int(j),tracking_rmse_rad=float(tracking[j]),
            previous_target_rad=float(a['previous'][start,j]),last_issued_target_rad=float(a['target'][stop-1,j]),
            target_change_rad=float(a['target'][stop-1,j]-a['previous'][start,j]),
            actual_change_rad=float(a['q_after'][stop-1,j]-a['q_before'][start,j]),
            last_tracking_error_rad=float(a['q_after'][stop-1,j]-a['target'][stop-1,j])) for j in top]
        summaries.append(dict(window=label,start_completed_tick=start+1,end_completed_tick=stop,parts=parts,motion=motion,
            largest_hand_tracking_errors=joints))
    require(all(sha256(p)==binding[p.name] for p in files), 'Source changed during analysis')
    return dict(case=name,role=registered['role'],spec=registered['spec'],input_sha256=binding,pair=admitted['pair'],
        phase=phase,chain_alignment_max_abs_rad=checks,windows=summaries,
        first_lost_stable_window=first_lost_window(stable,a['obj'],a['palm'],a['force'],float(np.median(a['obj'][24:48,2]))),
        dp_supervision=backend['contract']['supervision'],steps=n,
        plan_replanning_tick_count=int((np.arange(n)%12==0).sum()),
        issued_equal_applied_equal_both_actuator_substeps=True,post_student_delay=False,
        raw_head_scope='Reconstructed from recorded normalized delta and nominal-centered output formula, not a separately logged network tensor.')


def run(evaluation, output, supersedes_report=None):
    evaluation,output=Path(evaluation).resolve(),Path(output).absolute()
    require(not output.exists(), 'Preserve previous command-chain evidence')
    plan=read(evaluation/'plan.json');rows=[];loaded={}
    torch.set_num_threads(2)
    for c in plan['cases']:
        role=c['role']
        if role not in loaded:
            pair=plan['pairs'][role]
            require(sha256(pair['backend'])==pair['backend_sha256'],'Selected backend changed')
            loaded[role]=torch.load(pair['backend'],map_location='cpu',weights_only=False)
        rows.append(analyze_case(evaluation,c,loaded[role]))
    code=['tools/tight_fresh_command_chain.py','tools/tight_fresh_phases.py',
        'visuotactile/native_controller.py','visuotactile/state_benchmark.py','visuotactile/position_control.py',
        'visuotactile/residual_output.py','visuotactile/execution_learning.py','visuotactile/takeover_trigger.py']
    result=dict(version=VERSION,semantic_revision=SEMANTIC_REVISION,complete=True,evaluation=str(evaluation),cases=rows,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:sha256(p) for p in code},plan_sha256=sha256(evaluation/'plan.json'),
        results_sha256=sha256(evaluation/'results.json'),device='cpu',model_forward_calls=0,optimizer_updates=0,simulator_starts=0,test_used=False,
        tolerances=dict(chain_reconstruction_rad=TOLERANCE_RAD,direction_active_rad=DIRECTION_EPSILON_RAD),
        semantics=dict(ticks='1-based completed control transitions; window motion starts at its first transition BEFORE pose.',
            dp='A predicts same-transition q_after. Its disagreement here is on its own altered closed-loop state, not expert validation error. B predicts expert target u; autonomous actual q is not its expert label.',
            chain='raw plan -> joint-limited nominal -> reachable nominal anchor -> linear raw correction -> joint clip -> shared reachable projection -> issued/applied/actuator -> actual q.',
            direction='Opposition to nominal is a descriptive backend intervention, not a claim of wrong expert direction. No legal expert labels exist on these autonomous states.',
            rigid_attachment='Kinematic diagnostic: observed endpoint palm pose with the object-to-palm transform frozen at window start. Not a simulated intervention or predicted task outcome.',
            hold='Stable entries are endpoints of existing past12 diagnostic windows; leaving that set does not itself mean physical instability or a drop. Relative motion can be intentional in-hand manipulation, and a short-window worsening of goal angle can be an intermediate path. Actual native completion is authoritative; legal expert recovery actions are needed to evaluate policy direction.',
            window_motion='Motion intervals include the first transition BEFORE pose; this spans12 transitions. The existing stable-window rule uses12 AFTER poses (11 intervals), so first_stable_window motion can exceed its threshold without contradiction.',
            goal='All motion goal distances/angles use the window-start pre-action goal, including endpoints after a native goal switch; they are not distances/angles to the current endpoint goal. Before/after goal hashes and switch ticks are explicit. Native keypoint endpoints use their own actual transition targets and may refer to different goals.',
            angle='Normalized wxyz quaternion geodesic without shape symmetries; original native keypoint criteria remain authoritative.',
            limits='Observational decomposition cannot isolate causal effects, friction, contact-model error or visual state error; correlated changes do not prove a cause. Reused development starts are not independent test.'))
    if supersedes_report is not None:
        prior=read(supersedes_report)
        require(prior['version']==VERSION and all(prior[k]==result[k] for k in
            ('evaluation','plan_sha256','results_sha256')), 'Superseded receipt must bind the identical recorded cohort')
        result['supersedes_report']=dict(path=str(Path(supersedes_report).absolute()),sha256=sha256(supersedes_report),
            semantic_revision=prior.get('semantic_revision',1),reason='Explicit frozen window-start goal and per-transition native-goal semantics; goal transition admission.')
    atomic_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--supersedes-report',type=Path)
    args=p.parse_args();run(args.evaluation,args.output,args.supersedes_report)
