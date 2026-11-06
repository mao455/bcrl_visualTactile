"""Fail-closed admission for the current command StateDP dataset.

No simulator import or dependence on a previous run. Motion stages are measured
proxies; they do not replace native task success. All thresholds are preregistered.
"""
import json
import math
from pathlib import Path

import h5py
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    from .narrow_data import direct_configuration
    require(config['version']=='tight_state_foundation_v1', 'Unknown current configuration')
    require(config['camera_content_validation'] is True, 'Explicit RGB-D content validation required')
    require(config['precision_policy']=='unified_fp32', 'GPU FP32 required')
    require(config['evaluation']['overlap_steps']==0, 'Formal overlap must be zero')
    require(config['training']['effective_batch']==64 and config['training']['microbatch']==4, 'Frozen sampler sizes')
    for key in ('table_z_m','object_height_above_table_origin_m','joint_velocity_halfwidth'):
        require(key in config['domain'], 'Explicit sampling range required: '+key)
    direct_configuration(config['domain'], 1)
    gate=config['development']
    require(gate['minimum_duration_ratio']==1.5 and gate['maximum_duration_ratio']==3., 'Duration gate is fixed at1.5–3')
    require(0<gate['maximum_phase_speed_ratio']<1, 'Measured phase speed must decrease')
    require(len(set(gate['seeds']))==len(gate['seeds']), 'Development seeds must be distinct')
    require(all(0<a<1 for a in gate['alphas']), 'Slow candidate alpha must be within(0,1)')
    require(sum(config['collection'][k] for k in ('train_families','validation_families','test_families'))==config['collection']['families'], 'Family split sizes')
    return config


def motion_stages(group):
    """Every moving phase uses measured q velocities and object velocities.

    Grasp=contact before lift; lift/transport=raised object before alignment;
    insertion=original native alignment or second-subgoal observations. No idle
    padding, target speed, or alpha enters the actual-motion measurements.
    """
    n=len(group['obs/time_s'])
    before=group['transition/diagnostics_before']
    qd=before['joint_velocity_rad_s'][:]
    obj=before['object_pose_w'][:]
    force=group['transition/object_contact_force_w'][:]
    contact=np.linalg.norm(force,axis=-1).reshape(n,-1).max(-1)>.1
    sub=before['successes'][:].reshape(n,-1).max(-1)
    align=group['transition/native_alignment_condition'][:].reshape(n,-1).any(-1)
    insertion=(align | (sub>=1)) & (sub<2)
    lifted=(obj[:,2]-obj[0,2])>.03
    masks=dict(grasp=contact & ~lifted & ~insertion & (sub<1),
               lift_transport=lifted & ~insertion & (sub<1), insertion=insertion)
    out={}
    for name,mask in masks.items():
        count=int(mask.sum())
        # Equal arm/hand groups, unchanged radians and seconds.
        speed=np.sqrt(.5*(np.mean(qd[mask,:7]**2)+np.mean(qd[mask,7:]**2))) if count else None
        ov=before['object_velocity_w_m_s_rad_s'][:][mask,:3]
        out[name]=dict(ticks=count,seconds=count/60,joint_speed_rms_rad_s=float(speed) if count else None,
            object_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(ov**2,axis=-1)))) if count else None)
    return out


def paired_admission(slow, reference, gate):
    errors=[];ratios={}
    for name,row in (('slow',slow),('reference',reference)):
        if not row.get('success'):errors.append(name+'_task_not_successful')
        if not row.get('numeric_valid'):errors.append(name+'_numeric_invalid')
        if not row.get('visual_passed'):errors.append(name+'_rgbd_invalid')
        if not row.get('tactile_valid'):errors.append(name+'_tactile_invalid')
        if row.get('post_success_actions')!=0:errors.append(name+'_post_success_padding')
    for key in ('family','configuration_sha256','reset_sha256','expert_sha256'):
        if not slow.get(key) or slow.get(key)!=reference.get(key):errors.append('reference_'+key+'_mismatch')
    if reference.get('alpha')!=1 or not 0<slow.get('alpha',1)<1:errors.append('invalid_speed_roles')
    duration=slow.get('duration_s',0);base=reference.get('duration_s',0)
    ratio=duration/base if base>0 else None
    if ratio is None or not math.isfinite(ratio) or not gate['minimum_duration_ratio']<=ratio<=gate['maximum_duration_ratio']:
        errors.append('measured_duration_outside_1.5_3')
    for phase in ('grasp','lift_transport','insertion'):
        a=slow.get('motion_stages',{}).get(phase,{});b=reference.get('motion_stages',{}).get(phase,{})
        av=a.get('joint_speed_rms_rad_s');bv=b.get('joint_speed_rms_rad_s')
        if min(a.get('ticks',0),b.get('ticks',0))<2 or av is None or bv is None or bv<=1e-5:
            errors.append(phase+'_insufficient_measured_motion');continue
        ratios[phase]=av/bv
        if not math.isfinite(ratios[phase]) or ratios[phase]>gate['maximum_phase_speed_ratio']:
            errors.append(phase+'_not_slower')
    return dict(passed=not errors,errors=errors,duration_ratio=ratio,actual_phase_speed_ratios=ratios,
                scope='Native success and measured simulation time/actual joint motion; no replay-success requirement')


def sensor_clock_admission(g):
    """Observed forces belong to the completed120Hz interval, never future ticks."""
    stamp=g['obs/time_s'][:];n=len(stamp)
    expected_before=stamp[:,None]+np.array([-1/120,0.])
    expected_after=stamp[:,None]+np.array([1/120,1/60])
    require(np.allclose(g['obs/contact_time_s'][:],expected_before,atol=1e-8,rtol=0),'Tactile pre-action timestamps')
    require(np.allclose(g['transition/contact_time_after_s'][:],expected_after,atol=1e-8,rtol=0),'Tactile120Hz transition timestamps')
    require(np.allclose(g['obs/camera_time_s'][:],stamp[:,None],atol=1e-8,rtol=0),'RGB-D observation timestamps')
    valid=g['obs/contact_valid'][:]
    require(valid.shape==(n,) and not valid[0] and valid[1:].all(),'Tactile reset validity')
    observed=g['obs/contact_substeps_w_N'][:];after=g['transition/contact_substeps_after_w_N'][:]
    require(observed.shape==after.shape==(n,2,5,3),'Two120Hz five-finger samples per control tick')
    require(np.array_equal(observed[1:],after[:-1]),'Tactile observation must equal prior completed substeps')
    require(np.array_equal(g['obs/contact_w_N'][:],observed[:,-1]),'Current force must equal final observed substep')
    return True


def numeric_admission(path):
    """Full native-rate numeric, label, physical target, sensor clock contract."""
    from .narrow_data import record
    key,meta,rec=record(path)
    with h5py.File(path,'r') as f:
        g=f['episodes'][key];n=len(rec['time'])
        sensor_clock_admission(g)
        success=g['transition/termination_reasons/max_successes'][:].reshape(n,-1).any(-1)
        before=g['transition/diagnostics_before/joint_position_rad'][:]
        after=g['transition/diagnostics_after/joint_position_rad'][:]
        require(np.array_equal(after[:-1],before[1:]), 'Actual state not aligned with next tick')
        require(np.isfinite(g['obs/contact_w_N'][:]).all(), 'Nonfinite tactile readings')
        require(np.isfinite(g['transition/contact_substeps_after_w_N'][:]).all(), 'Nonfinite120Hz tactile')
        # Check all recorded time vectors rather than fabricating sensor stamps.
        for section in ('obs','transition'):
            def check(name,value):
                if isinstance(value,h5py.Dataset) and value.dtype.kind=='f':
                    if 'depth' not in name:require(np.isfinite(value[:]).all(), 'Nonfinite numeric field '+section+'/'+name)
            g[section].visititems(check)
        result=dict(numeric_valid=True,tactile_valid=True,steps=n,duration_s=n/60,
            success=bool(success[-1] and success.sum()==1),post_success_actions=int(n-1-np.flatnonzero(success)[0]) if success.any() else 0,
            motion_stages=motion_stages(g),joint_names=meta.get('joint_names'),physics_hz=120,control_hz=60)
    return key,meta,rec,result


def validate_formal_manifest(manifest, config):
    require(manifest.get('version')=='tight_slow_multimodal_v1','Historical datasets cannot enter current formal training')
    require(manifest.get('collection_finished') is True,'Collection has not finished')
    require(manifest.get('development_passed') is True,'Development gate did not pass')
    entries=manifest['episodes'];families={};ids=set()
    for entry in entries:
        require(entry['id'] not in ids, 'Duplicate episode identity');ids.add(entry['id'])
        require(entry['split'] in ('train','validation','test'), 'Invalid split')
        require(families.setdefault(entry['family'],entry['split'])==entry['split'], 'Family split leakage')
        require(entry.get('alpha')==manifest['frozen_alpha'] and 0<entry['alpha']<1,'Native or mixed speed in slow dataset')
        require(all(entry.get(k) is True for k in ('state_valid','multi_view_valid','tactile_valid')),'Invalid formal modality')
        require(entry.get('admission',{}).get('passed') is True,'Missing individual admission')
        admission=entry['admission'];ratio=admission.get('duration_ratio')
        require(isinstance(ratio,(float,int)) and math.isfinite(ratio) and 1.5<=ratio<=3.,'Measured duration ratio is invalid despite passed flag')
        speeds=admission.get('actual_phase_speed_ratios',{})
        for phase in ('grasp','lift_transport','insertion'):
            value=speeds.get(phase)
            require(isinstance(value,(float,int)) and math.isfinite(value) and 0<=value<=config['development']['maximum_phase_speed_ratio'],
                'Actual phase speed missing or not slower: '+phase)
        require(entry.get('reference_sha256') and entry.get('visual_acceptance_sha256'),'Missing reference/content evidence')
    require(len(entries)>=config['collection']['minimum_accepted_episodes'],'Insufficient accepted episodes, no quantity backfill')
    require(len(families)>=config['collection']['minimum_accepted_families'],'Insufficient accepted families')
    require(set(families.values())=={'train','validation','test'},'Missing split')
    return True


def validate_training_gates(collection, data_root):
    """A passing historical/fixture receipt cannot admit a different dataset."""
    import hashlib
    def read(path):return json.loads(Path(path).read_text())
    def fingerprint(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    expected=dict(source_data_sha256=fingerprint(data_root/'data.json'),statistics_sha256=fingerprint(data_root/'statistics.pt'))
    cpu=read(collection/'cpu_audit.json')
    require(cpu.get('passed') and cpu.get('source_data_sha256')==expected['source_data_sha256'],'Formal CPU audit dataset mismatch')
    for name in ('gpu_parity','gradient_gate'):
        gate=read(collection/(name+'.json'))
        require(gate.get('passed') and all(gate.get(k)==v for k,v in expected.items()),'Formal GPU gate dataset/statistics mismatch: '+name)
    for name in ('restore_gate','actuator_gate'):
        require(read(collection/(name+'.json')).get('passed') is True,'Missing completed '+name)
    return True
