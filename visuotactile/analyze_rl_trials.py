"""Audit and compare bounded, single-environment RL trials, without training.

HDF5 committed rows are authoritative, including incomplete episodes. Missing
telemetry stays missing; process/audit errors never become robot failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from .compact import EpisodeFile
import numpy as np

from .trial_utils import write_json

CONDITIONS = ("original", "force_1p5")


def episode_outcome(complete, terminated, truncated, achieved):
    if bool(complete) != bool(terminated or truncated):
        raise ValueError("Committed completion flag disagrees with native terminal row")
    return ("success" if achieved else "failure") if complete else "censored"


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return None
    p, z2 = successes/total, z*z
    centre = (p+z2/(2*total))/(1+z2/total)
    half = z*np.sqrt(p*(1-p)/total+z2/(4*total*total))/(1+z2/total)
    return [float(max(0, centre-half)), float(min(1, centre+half))]


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float)), initial=0))


def exposure(vector, dt):
    vector = vector.reshape(len(vector), 3)
    norm = np.linalg.norm(vector, axis=1)
    active = norm > 0
    return dict(sample_dt_s=float(dt),samples=len(norm),nonzero_steps=int(active.sum()),
                nonzero_runs=int(np.count_nonzero(active & ~np.r_[False, active[:-1]])),
                duration_s=float(active.sum()*dt), peak=float(norm.max(initial=0)),
                nonzero_quantiles=np.quantile(norm[active], [.5,.9,.95,1]).tolist() if active.any() else None,
                integral_norm=float(norm.sum()*dt), vector_integral=(vector.sum(0)*dt).tolist(),
                first_step=int(np.flatnonzero(active)[0]) if active.any() else None)


def load_trial(path):
    """Return public-size summary and small state arrays; never load all images."""
    path = Path(path)
    data_path = path/"trajectories.h5" if (path/"trajectories.h5").exists() else path/"data.h5"
    with EpisodeFile(data_path) as f:
        metadata = json.loads(f.attrs["metadata_json"])
        names = list(f.get("episodes", {}))
        if len(names) != 1:
            raise ValueError(f"Expected one recorded episode, found {len(names)}")
        g = f["episodes"][names[0]]
        n = int(g.attrs["length"])
        if n < 1:
            raise ValueError("No committed control steps")
        errors, missing = [], []
        def read(key, optional=False):
            if key not in g:
                if optional:
                    missing.append(key)
                    return None
                raise ValueError(f"Missing required field {key}")
            if len(g[key]) < n:
                raise ValueError(f"Incomplete committed dataset {key}")
            return g[key][:n]
        def check(ok, message):
            if not bool(ok):
                errors.append(message)
        tr = lambda key: read("transition/"+key)
        dt = float(metadata["control_dt"])
        done = read("terminated") | read("truncated")
        complete = bool(g.attrs.get("complete", False))
        check(complete == bool(done[-1]) and not done[:-1].any(), "episode boundary")
        step = read("obs/step_id")
        check(np.array_equal(step, np.arange(n)), "non-contiguous episode steps")
        check(np.array_equal(step, tr("step_id")), "observation/action step mismatch")
        check(np.all(read("obs/episode_id")==tr("episode_id")), "observation/action episode mismatch")
        before, after = tr("time_before_s"), tr("time_after_s")
        check(np.allclose(after-before,dt,atol=1e-9,rtol=0), "control dt")
        check(np.allclose(before[1:],after[:-1],atol=1e-9,rtol=0), "control time continuity")
        post_valid = tr("observation_after/available")
        check(np.array_equal(post_valid, ~done), "post-reset observation exposed as terminal")
        for key in ("image", "proprio", "camera_valid"):
            check(not np.any(g["transition/observation_after/"+key][n-1]) if done[-1] else True,
                  "terminal placeholder not masked: "+key)
        source = tr("provenance/selected_source")
        check(np.all(source==1) and np.all(tr("provenance/expert_computed")), "not pure expert")
        for field in ("nominal_computed", "correction_computed", "correction_executed"):
            check(not tr("provenance/"+field).any(), "unexpected computed/executed branch: "+field)
        for field in ("previous_nominal_valid", "prior_plan_valid"):
            check(not read("obs/"+field).any(), "fabricated nominal plan: "+field)
        submitted = tr("submitted_action")
        check(np.array_equal(submitted,tr("provenance/expert_command")), "expert/selected command differs")
        check(np.array_equal(np.clip(submitted,-1,1),tr("command_clipped_pre_delay")), "clipping")
        delay_steps = tr("control_trace/selected_command_step")
        check(np.all((delay_steps>=0)&(delay_steps<=step)), "non-causal delay selection")
        delayed = tr("control_trace/delayed_command")
        if np.all((delay_steps>=0)&(delay_steps<n)):
            check(np.array_equal(delayed,np.clip(submitted[delay_steps],-1,1)), "delay command provenance")
        check(np.array_equal(step-delay_steps,tr("control_trace/effective_lag_steps")), "delay lag")
        target0, target = tr("target_before_rad"), tr("target_after_rad")
        limits = np.array(metadata["joint_limits_rad"])[0]
        a = metadata["config"]["action"]
        expected = target0.copy()
        # Canonical first seven are arm, remaining 22 hand, as runtime asserts.
        raw = np.clip(target0[:,:7]+a["dof_speed_scale"]*dt*delayed[:,:7],limits[:7,0],limits[:7,1])
        expected[:,:7] = a["arm_moving_average"]*raw+(1-a["arm_moving_average"])*target0[:,:7]
        raw = limits[7:,0]+.5*(delayed[:,7:]+1)*(limits[7:,1]-limits[7:,0])
        expected[:,7:] = a["hand_moving_average"]*raw+(1-a["hand_moving_average"])*target0[:,7:]
        expected = np.clip(expected,limits[:,0],limits[:,1])
        check(np.allclose(target,expected,atol=3e-7,rtol=0), "native target math")
        check(np.array_equal(target0[1:],target[:-1]), "target continuity")
        state = {}
        for name in ("joint_position_rad", "joint_velocity_rad_s", "object_pose_w", "goal_pose_w", "hole_pose_w", "table_pose_w"):
            state[name] = tr("diagnostics_after/"+name)
            check(np.isfinite(state[name]).all(), "nonfinite state: "+name)
        check(np.array_equal(tr("diagnostics_before/joint_position_rad")[1:],state["joint_position_rad"][:-1]),
              "pre/post state continuity")
        for name in ("table_pose_w", "hole_pose_w"):
            check(np.allclose(state[name],tr("diagnostics_before/"+name)[0],atol=1e-5,rtol=0), "invalid CPU scene: "+name)
        initial = {name:tr("diagnostics_before/"+name)[0] for name in state}
        for name in ("object_velocity_w_m_s_rad_s","base_pose_w","palm_body_pose_w","object_initial_z_m"):
            value=read("transition/diagnostics_before/"+name,optional=True)
            if value is not None:
                initial[name]=value[0]
        initial["target_rad"] = target0[0]
        for name in ("external_force_probability", "external_torque_probability", "external_object_mass_kg"):
            initial[name] = tr(name)[0]
        gate = read("transition/external_wrench_lift_gate",optional=True)
        task_gate = read("transition/external_wrench_task_gate",optional=True)
        force, torque = tr("external_force_after_dr_w_N"), tr("external_torque_after_dr_w_Nm")
        applied_force = read("transition/applied_wrench/force_w_N",optional=True)
        applied_torque = read("transition/applied_wrench/torque_w_Nm",optional=True)
        if applied_force is not None:
            check(applied_force.shape==(n,2,1,3), "applied physics wrench shape")
            check(np.allclose(tr("applied_wrench/time_s"),before[:,None]+np.arange(2)*metadata["physics_dt"],atol=1e-9,rtol=0), "applied wrench timing")
            q=tr("applied_wrench/object_quat_wxyz");v=tr("applied_wrench/force_b_N")
            rotated=v+2*np.cross(q[...,1:],np.cross(q[...,1:],v)+q[...,:1]*v)
            check(np.allclose(rotated,applied_force,atol=2e-6,rtol=2e-6), "applied world force rotation")
            check(np.allclose(np.linalg.norm(applied_force,axis=-1),np.linalg.norm(force[:,None],axis=-1),atol=2e-5,rtol=2e-5), "requested/applied force norm")
        dr = metadata["config"]["domain_randomization"]
        if gate is not None and dr["force_only_when_lifted"]:
            check(not force[~gate].any(), "force while lift gate closed")
        if task_gate is not None:
            check(not force[~task_gate].any() and not torque[~task_gate].any(), "wrench during retract")
        check(np.all(tr("external_force_scale")==dr["force_scale"]), "force scale provenance")
        contact = tr("contact_substeps_after_w_N")
        contact_time = tr("contact_time_after_s")
        offsets = np.arange(1-contact.shape[1],1)*metadata["physics_dt"]
        check(np.allclose(contact_time,after[:,None]+offsets,atol=1e-9,rtol=0), "force/FK interval timing")
        rotations = tr("kinematics_after/finger_rotation_base")
        local = tr("kinematics_after/contact_substeps_local_N")
        check(np.allclose(local,np.einsum('...ji,...j->...i',rotations,contact),atol=2e-5,rtol=1e-5), "local force rotation")
        check(np.allclose(np.linalg.norm(local,axis=-1),np.linalg.norm(contact,axis=-1),atol=2e-5,rtol=1e-5), "local force norm")
        check(np.isfinite(local).all() and np.isfinite(submitted).all() and np.isfinite(force).all(), "nonfinite force/command")
        check(np.allclose(read("obs/camera_time_s"),before[:,None],atol=1e-9,rtol=0), "camera time")
        check(read("obs/camera_valid").all(), "missing camera")
        # Inspect EVERY stored image/depth row in bounded chunks, preserving precision.
        views=len(metadata['camera_ids']);height,width=metadata['cameras'][0]['resolution']
        valid_counts = np.zeros(views); pixels = 0
        for start in range(0,n,16):
            image = g["obs/image"][start:min(start+16,n)]
            valid = g["obs/depth_valid"][start:min(start+16,n)]
            depth = g["obs/depth_m"][start:min(start+16,n)]
            check(image.shape[1:]==(views,4,height,width), "image schema")
            check(np.isfinite(image).all() and image[:,:,:3].min()>=0 and image[:,:,:3].max()<=1, "RGB-D numerical values")
            check(np.array_equal(image[:,:,3:4],depth), "metric depth/image mismatch")
            check(np.all(depth[valid]>0) and np.all(depth[~valid]==0), "depth validity")
            valid_counts += valid.sum(axis=(0,2,3,4)); pixels += len(image)*height*width
        if metadata.get('camera_rig',{}).get('version')==3:
            pose=read('obs/camera_pose_w')
            check(pose.shape==(n,views,7) and np.isfinite(pose).all(),'camera pose shape/finite')
            check(np.allclose(np.linalg.norm(pose[:,:,3:],axis=-1),1,atol=2e-5),'camera rotation norm')
            check(np.allclose(pose[:,:2],pose[0,:2],atol=2e-5),'fixed cameras moved')
            mount=metadata['camera_rig'].get('wrist')
            if mount:
                parent=read('obs/camera_parent_pose_w');q=parent[:,3:];v=np.broadcast_to(mount['pos_parent_m'],(n,3))
                expected=parent[:,:3]+v+2*np.cross(q[:,1:],np.cross(q[:,1:],v)+q[:,:1]*v)
                check(np.allclose(pose[:,2,:3],expected,atol=2e-5),'wrist image/parent pose alignment')
                from scipy.spatial.transform import Rotation
                parent_rot=Rotation.from_quat(parent[:,[4,5,6,3]])
                local_q=np.asarray(mount['quat_parent_ros_wxyz'])
                expected_rot=parent_rot*Rotation.from_quat(local_q[[1,2,3,0]])
                observed_rot=Rotation.from_quat(pose[:,2][:,[4,5,6,3]])
                check(np.max((expected_rot.inv()*observed_rot).magnitude())<2e-5,'wrist image/parent rotation alignment')
        check(np.all(valid_counts>0), "all-invalid depth")
        sample_indices = sorted(set((0,n//2,n-1)))
        rgb_std = [g["obs/image"][i,:,:3].std(axis=(1,2,3)).tolist() for i in sample_indices]
        check(np.min(rgb_std)>0.01, "uniform camera")
        success_count = tr("diagnostics_after/successes")
        max_goals = tr("diagnostics_after/max_goals")
        retract = tr("diagnostics_after/retract_succeeded")
        random_goal = read("transition/diagnostics_after/is_random_goal",optional=True)
        achieved = bool(success_count[-1]>=max_goals[-1])
        if metadata["config"]["precise_assembly"]["enable_retract"]:
            achieved &= bool(retract[-1]) or bool(random_goal[-1] if random_goal is not None else False)
        outcome = episode_outcome(complete,read("terminated")[-1],read("truncated")[-1],achieved)
        reasons = [key for key,ds in g["transition/termination_reasons"].items() if ds[n-1]]
        norm = np.linalg.norm(contact,axis=-1).reshape(-1,5)
        force_active=np.linalg.norm(force.reshape(n,3),axis=1)>0
        prior_contact=(np.linalg.norm(read("obs/contact_w_N"),axis=-1).max(1)>.01) & read("obs/contact_valid")
        after_contact=(norm.reshape(n,-1,5)>.01).any(axis=(1,2))
        tracking = target-state["joint_position_rad"]
        progress = success_count/max_goals
        phase = tr("diagnostics_after/retract_phase")
        fk_error = tr("diagnostics_after/fk_position_error_m")
        summary = dict(path=str(path), seed=metadata["seed"], pair_id=metadata.get("pair_id"),
            condition=metadata["condition"], episode=names[0], control_steps=n,
            outcome=outcome, native_reasons=reasons, simulated_seconds=float(n*dt),
            progress_final=float(progress[-1]), success_goals_final=int(success_count[-1]),
            retract_final=bool(phase[-1]), retract_succeeded=bool(retract[-1]),
            phase_change_steps=(np.flatnonzero((success_count[1:]!=success_count[:-1])|(phase[1:]!=phase[:-1]))+1).tolist(),
            force=exposure(force,dt), torque=exposure(torque,dt),
            force_steps_with_finger_contact_before=int((force_active & prior_contact).sum()),
            force_steps_with_finger_contact_after=int((force_active & after_contact).sum()),
            applied_force=exposure(applied_force.reshape(-1,3),metadata["physics_dt"]) if applied_force is not None else None,
            applied_torque=exposure(applied_torque.reshape(-1,3),metadata["physics_dt"]) if applied_torque is not None else None,
            requested_vs_applied_peak_vector_error_N=float(np.linalg.norm(applied_force-force[:,None],axis=-1).max()) if applied_force is not None else None,
            lift_gate_steps=int(gate.sum()) if gate is not None else None,
            task_gate_steps=int(task_gate.sum()) if task_gate is not None else None,
            eligible_force_steps=int((gate & task_gate).sum()) if gate is not None and task_gate is not None else None,
            contact=dict(peak_N_by_finger=norm.max(0).tolist(),
                quantiles_N_by_finger=np.quantile(norm,[.5,.9,.95,.99],axis=0).tolist(),
                fraction_gt_0p01N_by_finger=(norm>.01).mean(0).tolist(),
                max_world_force_rate_N_s_by_finger=(np.linalg.norm(np.diff(contact.reshape(-1,5,3),axis=0),axis=-1)/metadata["physics_dt"]).max(0).tolist()),
            tracking=dict(rmse_rad=float(np.sqrt(np.mean(tracking**2))),peak_abs_rad=float(np.abs(tracking).max())),
            camera=dict(valid_fraction=(valid_counts/pixels).tolist(),rgb_std_samples=rgb_std),
            fk_max_position_error_m=float(fk_error.max()),
            delay_index_counts=np.bincount(tr("control_trace/delay_index").astype(int)).tolist(),
            audit_errors=errors, missing_telemetry=missing,
            trace_version=int(f.attrs.get("trace_version",1)), wrench_trace_version=metadata.get("wrench_trace_version",1),
            hdf5_bytes=data_path.stat().st_size)
        arrays = dict(initial=initial,object=state["object_pose_w"],joint=state["joint_position_rad"],target=target,
                      command=submitted, force=force,torque=torque,contact=norm.reshape(n,-1,5),
                      tracking=tracking,time=after,progress=progress,delay=tr("control_trace/delay_index"),
                      applied_force=applied_force,applied_torque=applied_torque)
        for key in ("torch_rng_before_inference","torch_rng_before_step","torch_rng_after_preprocess"):
            arrays[key] = read("transition/"+key,optional=True)
    return dict(summary=summary,metadata=metadata,arrays=arrays)


def config_differences(a,b,prefix=""):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for key in sorted(set(a)|set(b)):
            result += config_differences(a.get(key),b.get(key),prefix+"/"+key)
        return result
    return [] if a==b else [prefix.lstrip("/")]


def compare_pair(a,b):
    sa,sb,ra,rb=a["summary"],b["summary"],a["arrays"],b["arrays"]
    n=min(sa["control_steps"],sb["control_steps"])
    initial={k:maxdiff(v,rb["initial"][k]) for k,v in ra["initial"].items()}
    different=config_differences(a["metadata"]["config"],b["metadata"]["config"])
    valid=max(initial.values())<=1e-6 and different==["domain_randomization/force_scale"]
    provenance_match=all(a["metadata"].get(k)==b["metadata"].get(k) for k in (
        "expert_model_sha256","teacher_group","expert_inference_mode","initial_torch_rng_sha256",
        "cameras","joint_limits_rad","physics_dt","control_dt"))
    physical_source_paths=("visuotactile/collect.py","visuotactile/sim/environment.py",
        "isaacsimenvs/tasks/play/utils/action_utils.py","isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py")
    hashes=[t["metadata"].get("code_identity",{}).get("source_sha256",{}) for t in (a,b)]
    source_match=(all(hashes[0].get(p)==hashes[1].get(p) for p in physical_source_paths)
                  if all(all(p in h for p in physical_source_paths) for h in hashes) else None)
    scale_ratio=b["metadata"]["applied_force_scale"]/a["metadata"]["applied_force_scale"]
    valid &= provenance_match and np.isclose(scale_ratio,1.5)
    valid &= source_match is not False
    valid &= not sa["audit_errors"] and not sb["audit_errors"]
    translation=np.linalg.norm(ra["object"][:n,:3]-rb["object"][:n,:3],axis=1)
    qa,qb=ra["object"][:n,3:7].astype(float),rb["object"][:n,3:7].astype(float)
    dot=np.abs((qa*qb).sum(1))/(np.linalg.norm(qa,axis=1)*np.linalg.norm(qb,axis=1))
    rotation=2*np.arccos(np.clip(dot,0,1))
    changed=np.flatnonzero(np.max(np.abs(ra["joint"][:n]-rb["joint"][:n]),axis=1)>1e-6)
    force_change=np.flatnonzero(np.max(np.abs(ra["force"][:n]-rb["force"][:n]),axis=(1,2))>1e-8)
    pre_force_prefix=int(force_change[0]) if len(force_change) else n
    outcome="incomparable" if not valid else "censored" if "censored" in (sa["outcome"],sb["outcome"]) else {
        ("success","success"):"both_success", ("success","failure"):"original_only_success",
        ("failure","success"):"force_1p5_only_success", ("failure","failure"):"both_failure"}[(sa["outcome"],sb["outcome"])]
    rng={key:(bool(np.array_equal(ra[key][:n],rb[key][:n])) if ra[key] is not None and rb[key] is not None else None)
         for key in ("torch_rng_before_inference","torch_rng_before_step","torch_rng_after_preprocess")}
    first=int(force_change[0]) if len(force_change) else None
    return dict(seed=sa["seed"],pair_id=sa["pair_id"],category=outcome,initial_pass=bool(valid),
                initial_max_abs_diff=initial,config_differences=different,common_prefix_steps=n,
                policy_and_initial_rng_metadata_match=bool(provenance_match),configured_scale_ratio=float(scale_ratio),
                collection_control_source_hashes_match=source_match,
                object_translation_peak_m=float(translation.max()),object_rotation_peak_rad=float(rotation.max()),
                max_abs_diff={k:maxdiff(ra[k][:n],rb[k][:n]) for k in ("joint","target","command","force","torque")},
                applied_world_force_max_abs_diff_N=(maxdiff(ra["applied_force"][:n],rb["applied_force"][:n])
                    if ra["applied_force"] is not None and rb["applied_force"] is not None else None),
                applied_world_torque_max_abs_diff_Nm=(maxdiff(ra["applied_torque"][:n],rb["applied_torque"][:n])
                    if ra["applied_torque"] is not None and rb["applied_torque"] is not None else None),
                delay_sequence_equal=bool(np.array_equal(ra["delay"][:n],rb["delay"][:n])), rng_equal_common_prefix=rng,
                first_joint_divergence_step=int(changed[0]) if len(changed) else None,
                first_object_translation_divergence_step=(int(np.flatnonzero(translation>1e-6)[0])
                    if np.any(translation>1e-6) else None),
                first_force_difference_step=first,
                pre_force_max_abs_diff={k:maxdiff(ra[k][:pre_force_prefix],rb[k][:pre_force_prefix])
                    for k in ("object","joint","command","target")},
                translation_at_prefix_end_m=float(translation[-1]),
                last_10_steps_within_5mm_after_force=(bool((translation[-10:]<.005).all()) if first is not None and n-first>=10 else None),
                recovery_note="Proximity to paired reference only; different lengths/phases are not aligned or causal evidence")


def aggregate(rows):
    stats={}
    for condition in CONDITIONS:
        selected=[r for r in rows if r["condition"]==condition]
        counts={key:sum(r.get("outcome")==key for r in selected) for key in ("success","failure","censored")}
        counts.update(planned=len(selected), started=sum(r.get("started",False) for r in selected),
                      runtime_errors=sum(r.get("runtime_error_attempts",int(r.get("runtime_error",False))) for r in selected),
                      audit_errors=sum(bool(r.get("audit_errors")) for r in selected))
        completed=counts["success"]+counts["failure"]
        unknown=len(selected)-completed
        counts.update(completed=completed,success_rate_completed=counts["success"]/completed if completed else None,
                      wilson95_completed=wilson(counts["success"],completed),
                      planned_success_bounds=[counts["success"]/len(selected),(counts["success"]+unknown)/len(selected)] if selected else None,
                      zero_force_episodes=sum(r.get("force",{}).get("nonzero_steps")==0 for r in selected),
                      nonzero_force_steps=sum(r.get("force",{}).get("nonzero_steps",0) for r in selected),
                      force_peak_N=max((r.get("force",{}).get("peak",0) for r in selected),default=0))
        stats[condition]=counts
    return stats


def condition_metrics(loaded):
    """Descriptive exposure/state metrics; no causal or robustness inference."""
    result={}
    for condition in CONDITIONS:
        trials=[v for (seed,c),v in loaded.items() if c==condition]
        if not trials:
            result[condition]=None;continue
        summaries=[t["summary"] for t in trials]
        force_values=[];torque_values=[]
        for t in trials:
            for key,values in (("applied_force",force_values),("applied_torque",torque_values)):
                if t["arrays"][key] is not None:
                    norm=np.linalg.norm(t["arrays"][key].reshape(-1,3),axis=1)
                    values.extend(norm[norm>0].tolist())
        def quantiles(values):
            return np.quantile(values,[0,.5,.9,.95,1]).tolist() if len(values) else None
        contacts=np.concatenate([t["arrays"]["contact"].reshape(-1,5) for t in trials])
        result[condition]=dict(
            duration_s_quantiles=quantiles([s["simulated_seconds"] for s in summaries]),
            successful_duration_s_quantiles=quantiles([s["simulated_seconds"] for s in summaries if s["outcome"]=="success"]),
            native_reason_counts={reason:sum(reason in s["native_reasons"] for s in summaries)
                for reason in ("fall","max_successes","hand_far","dropped","timeout")},
            force_exposed_episodes=sum(s["force"]["nonzero_steps"]>0 for s in summaries),
            force_contact_before_steps=sum(s["force_steps_with_finger_contact_before"] for s in summaries),
            force_contact_after_steps=sum(s["force_steps_with_finger_contact_after"] for s in summaries),
            applied_nonzero_force_N_quantiles=quantiles(force_values),
            applied_nonzero_torque_Nm_quantiles=quantiles(torque_values),
            total_applied_force_norm_integral_Ns=sum(s["applied_force"]["integral_norm"] for s in summaries if s["applied_force"]),
            total_applied_force_duration_s=sum(s["applied_force"]["duration_s"] for s in summaries if s["applied_force"]),
            total_applied_torque_norm_integral_Nms=sum(s["applied_torque"]["integral_norm"] for s in summaries if s["applied_torque"]),
            contact_N_quantiles_by_finger=np.quantile(contacts,[.5,.9,.95,.99,1],axis=0).tolist(),
            contact_fraction_gt_0p01N_by_finger=(contacts>.01).mean(0).tolist(),
            episode_tracking_rmse_rad_quantiles=quantiles([s["tracking"]["rmse_rad"] for s in summaries]),
            requested_applied_world_force_error_peak_N=max((s["requested_vs_applied_peak_vector_error_N"] or 0 for s in summaries)),
            quantile_levels=[0,.5,.9,.95,1], contact_quantile_levels=[.5,.9,.95,.99,1],
            note="Contact statistics weight audited recorded samples, including censored prefixes if present; successful durations use successes only")
    return result


def plot_pair(pair,a,b,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,11))
    for t,label in ((a,"original"),(b,"force_1p5")):
        r=t["arrays"];time=r["time"]
        axes[0,0].plot(time,r["object"][:,2],label=label)
        axes[0,1].plot(time,r["progress"],label=label)
        for column,kind in enumerate(("force","torque")):
            applied=r["applied_"+kind]
            if applied is not None:
                sample_time=(time[:,None]-t["metadata"]["control_dt"]+np.arange(2)*t["metadata"]["physics_dt"]).reshape(-1)
                axes[1,column].plot(sample_time,np.linalg.norm(applied.reshape(-1,3),axis=1),label=label+" applied")
            else:
                axes[1,column].plot(time,np.linalg.norm(r[kind].reshape(-1,3),axis=1),label=label+" requested only")
        axes[2,0].plot(time,r["contact"].max((1,2)),label=label)
        axes[2,1].plot(time,np.sqrt((r["tracking"]**2).mean(1)),label=label)
    ra,rb=a["arrays"],b["arrays"];n=pair["common_prefix_steps"]
    axes[3,0].plot(ra["time"][:n],np.linalg.norm(ra["object"][:n,:3]-rb["object"][:n,:3],axis=1),label="paired common prefix")
    axes[3,1].plot(ra["time"][:n],np.max(np.abs(ra["command"][:n]-rb["command"][:n]),axis=1),label="paired common prefix")
    for ax,label in zip(axes.flat,("object z (m)","task progress","external force norm (N)","external torque norm (Nm)","max finger normal force (N)","target tracking RMS (rad)","object position difference (m)","command max absolute difference")):
        ax.set(xlabel="simulation time (s)",ylabel=label);ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f"seed {pair['seed']}: {pair['category']}; no trajectory time warping")
    fig.tight_layout();fig.savefig(output,dpi=130);plt.close(fig)


def analyze(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if (root/"plan.json").exists():
        plan=json.loads((root/"plan.json").read_text())
        entries=plan["trials"]
    else:
        plan=dict(note="Legacy discovery; absent launches require separate inventory")
        entries=[]
        for path in sorted(root.glob("pair*")):
            if path.is_dir() and (path/"metadata.json").exists():
                m=json.loads((path/"metadata.json").read_text())
                entries.append(dict(name=path.name,seed=m["seed"],condition=m["condition"]))
    rows,loaded=[],{}
    for entry in entries:
        path=root/entry["name"]
        row=dict(**entry,started=False,outcome="not_started",runtime_error=False,audit_errors=[])
        attempts=[json.loads(p.read_text()) for p in sorted(root.glob(entry["name"]+"*.status.json"))]
        good=[s for s in attempts if s.get("returncode")==0 and s.get("episode_started") and not s.get("audit_errors")]
        if len(good)>1:
            row.update(outcome="audit_error",audit_errors=["Multiple valid attempts; no outcome selection allowed"])
            rows.append(row);continue
        status=good[0] if good else attempts[-1] if attempts else {}
        row["runtime_error_attempts"]=sum(s.get("returncode",0)!=0 for s in attempts)
        row["attempts"]=[dict(name=s.get("attempt_name",entry["name"]),returncode=s.get("returncode"),
                              episode_started=s.get("episode_started")) for s in attempts]
        path=root/status.get("attempt_name",entry["name"])
        row["runtime_error"]=status.get("returncode",0)!=0 or (path/"FAILED.txt").exists()
        if (path/"trajectories.h5").exists():
            try:
                trial=load_trial(path);row.update(trial["summary"],started=True)
                if row["audit_errors"]:
                    row["recorded_outcome"]=row["outcome"];row["outcome"]="audit_error"
                elif row["runtime_error"]:
                    row["recorded_outcome"]=row["outcome"];row["outcome"]="runtime_error"
                else:
                    loaded[(entry["seed"],entry["condition"])]=trial
            except Exception as exc:
                row.update(outcome="audit_error",audit_errors=[repr(exc)])
        elif status.get("state")=="finished" or row["runtime_error"]:
            row.update(outcome="runtime_error",runtime_error=True)
        rows.append(row)
    pairs=[]
    for seed in sorted({r["seed"] for r in rows}):
        a,b=loaded.get((seed,"original")),loaded.get((seed,"force_1p5"))
        if a and b:
            pair=compare_pair(a,b);pairs.append(pair)
            plot_pair(pair,a,b,output/f"pair_{seed}.png")
        else:
            pairs.append(dict(seed=seed,category="unavailable",reason="missing, process error, or audit error"))
    report=dict(root=str(root),conditions=aggregate(rows),trials=rows,pairs=pairs,
                metrics=condition_metrics(loaded),
                paired_categories={key:sum(p["category"]==key for p in pairs) for key in (
                    "both_success","original_only_success","force_1p5_only_success","both_failure","censored","incomparable","unavailable")},
                caveat="Complete episodes use native success plus retract. Errors and external budget censoring are separate. Zero-exposure episodes remain in denominator. CI is descriptive and does not model paired dependence or systematic simulator errors.")
    write_json(output/"report.json",report)
    lines=["# Released tight-insertion expert trials","",report["caveat"],"",
           "| condition | planned | started | success | failure | censored | runtime errors | audit errors | completed rate | Wilson 95% |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key,v in report["conditions"].items():
        lines.append(f"| {key} | {v['planned']} | {v['started']} | {v['success']} | {v['failure']} | {v['censored']} | {v['runtime_errors']} | {v['audit_errors']} | {v['success_rate_completed']} | {v['wilson95_completed']} |")
    lines += ["","Paired categories:","","```json",json.dumps(report["paired_categories"],indent=2),"```","",
              "Per-trial exposure, contact, tracking, initial checks, differences, missing fields and audit errors are in report.json. Plots use native simulation times; terminal images are absent by design."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=analyze(args.root,args.output)
    print(json.dumps({k:report[k] for k in ("conditions","paired_categories")},indent=2))


if __name__=="__main__":
    main()
