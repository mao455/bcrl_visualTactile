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
    if 'data_composition' in config:
        d=config['data_composition']
        require(d['version']=='core_auxiliary_v1' and d['auxiliary_bounds']==[1.35,3.3], 'Frozen Auxiliary bounds required')
        require(d['core_target']==200 and d['auxiliary_maximum']==100 and d['auxiliary_per_family_maximum']==2,
            'Frozen composition targets required')
        require(d['auxiliary_global_ratio_maximum']==.5 and d['auxiliary_per_family_not_above_core'] is True,
            'Frozen Auxiliary family balance required')
        require(d['training_subset']=='core_only', 'Auxiliary training requires a separate controlled experiment')
    cap=config.get('speed_scheme',{}).get('hand_target_cap_rad_s')
    if config.get('palm_training'):
        p=config['palm_training'];r=config['resources'];t=config['training']
        require(0<p['additional_target']<=p['additional_maximum']<=60 and 0<p['maximum_attempts']<=120,'Finite train-only collection cap')
        require(r['collection_hours']<=2 and r['training_hours']<=9 and p['offline_hours']<=1,'Finite stage wall budgets')
        require(t['maximum_anchor_epochs']==300 and t['minimum_anchor_epochs']>=200,'Explicit anchor exposure budget')
        require(r['resource_policy']=='bounded_mean_v1' and r['cap_percent']==50 and r['admission_percent']<=45
                and r['min_gpu_free_mib']>=6144 and r['min_ram_gib']>=14 and r['min_disk_gib']>=60,'Current resource caps/reserves')
        require(p['final_closed_loop'] is False and p['loss_id']=='palm_command_sample_commit12','Single palm model/offline-only scope')
    if cap is not None:
        values=np.asarray(cap,dtype=float)
        require(values.shape in ((),(22,)) and np.isfinite(values).all() and (values>0).all(),
            'Hand cap must be finite positive scalar or22 canonical rad/s values')
        physical=config.get('physical_motion_gate',{})
        speed=physical.get('maximum_mean_speed_ratio',float('nan'));path=physical.get('maximum_insertion_path_ratio',float('nan'))
        require(math.isfinite(speed) and 0<speed<1 and math.isfinite(path) and path>=1,
            'Capped production requires explicit physical speed/path gates in current configuration')
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
    def factors(row):
        scheme=row.get('speed_scheme')
        return (scheme['arm'],scheme['hand']) if scheme else (row.get('alpha',1),)*2
    native=factors(reference);limited=factors(slow)
    if native!=(1.,1.) or reference.get('speed_scheme',{}).get('hand_target_cap_rad_s') is not None or not all(isinstance(v,(int,float)) and 0<v<=1 for v in limited) or min(limited)>=1:
        errors.append('invalid_speed_roles')
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


def physical_motion_errors(comparison, rules):
    errors=[]
    for phase,groups in [('whole',('hand',)),('insertion',('hand','end_effector','object'))]:
        for group in groups:
            item=comparison.get(phase,{}).get(group,{})
            if not item.get('decomposition_valid'):
                errors.append(phase+'_'+group+'_missing_physical_motion');continue
            if not all(math.isfinite(item.get(k,float('nan'))) and item[k]>=0 for k in ('mean_speed_ratio','path_ratio')):
                errors.append(phase+'_'+group+'_invalid_physical_motion');continue
            if item['mean_speed_ratio']>rules['maximum_mean_speed_ratio']:
                errors.append(phase+'_'+group+'_actual_speed_not_reduced')
            if phase=='insertion' and item['path_ratio']>rules['maximum_insertion_path_ratio']:
                errors.append(phase+'_'+group+'_path_excess')
    return errors


def capture_configuration(config):
    """Composition/queue changes cannot masquerade as physical capture changes."""
    return {k:v for k,v in config.items() if k not in
            ('collection','data_composition','training','loss_weights','palm_training','resources','reference_commit')}


def composition_admission(slow, reference, config, reference_config):
    """New classification, retaining the historical strict decision verbatim.

    The only removable admission error is the strict duration bound. Native
    settings, reset, physical motion, full success and sensor gates remain hard.
    """
    require(capture_configuration(config)==capture_configuration(reference_config),
        'Reference physical capture configuration differs')
    a=dict(slow);b=dict(reference)
    # Full experiment hashes legitimately differ when only the queue/contract
    # changes; additionally require the actually recorded native configuration.
    native=a.get('actual_native_configuration_sha256')
    compatible=bool(native) and native==b.get('actual_native_configuration_sha256')
    if compatible:
        a['configuration_sha256']=native;b['configuration_sha256']=native
    strict=paired_admission(a,b,config['development'])
    physical=slow.get('physical_motion',{})
    errors=list(strict['errors'])+list(physical.get('errors',['missing_physical_motion']))
    errors+=physical_motion_errors(physical.get('comparison',{}),config['physical_motion_gate'])
    if not compatible:errors.append('recorded_native_configuration_mismatch')
    strict=dict(strict,errors=sorted(set(errors)),passed=not errors,
        physical_motion_comparison=physical.get('comparison',{}))
    errors=[e for e in strict['errors'] if e!='measured_duration_outside_1.5_3']
    ratio=strict['duration_ratio'];low,high=config['data_composition']['auxiliary_bounds']
    if ratio is None or not math.isfinite(ratio) or not low<=ratio<=high:
        errors.append('outside_frozen_auxiliary_bounds')
    kind='rejected' if errors else 'Core' if 1.5<=ratio<=3. else 'Auxiliary'
    return dict(strict,passed=not errors,errors=sorted(set(errors)),admission_class=kind,
        strict_admission=strict,source_configuration_sha256=slow.get('configuration_sha256'),
        reference_configuration_sha256=reference.get('configuration_sha256'),
        comparison_native_configuration_sha256=native if compatible else None)


def select_composition(entries, config):
    """Deterministic family rounds, never outcome-dependent recollection."""
    from collections import Counter,defaultdict
    core=[e for e in entries if e['admission']['admission_class']=='Core']
    counts=Counter(e['family'] for e in core);groups=defaultdict(list)
    for e in entries:
        if e['admission']['admission_class']=='Auxiliary':groups[e['family']].append(e)
    maximum=min(config['data_composition']['auxiliary_maximum'],len(core)//2)
    auxiliary=[]
    for index in range(config['data_composition']['auxiliary_per_family_maximum']):
        for family in sorted(groups):
            rows=sorted(groups[family],key=lambda e:e['id'])
            if len(auxiliary)<maximum and index<min(len(rows),counts[family]):auxiliary.append(rows[index])
    return core,auxiliary


def validate_formal_manifest(manifest, config):
    composition=manifest.get('version')=='tight_slow_multimodal_composition_v2'
    require(composition or manifest.get('version')=='tight_slow_multimodal_v1','Historical datasets cannot enter current formal training')
    if composition:require(manifest.get('data_composition')==config.get('data_composition') and config.get('data_composition'), 'Composition contract mismatch')
    require(manifest.get('collection_finished') is True,'Collection has not finished')
    require(manifest.get('development_passed') is True,'Development gate did not pass')
    entries=manifest['episodes'];families={};ids=set()
    for entry in entries:
        require(entry['id'] not in ids, 'Duplicate episode identity');ids.add(entry['id'])
        require(entry['split'] in ('train','validation','test'), 'Invalid split')
        require(families.setdefault(entry['family'],entry['split'])==entry['split'], 'Family split leakage')
        if 'frozen_speed_scheme' in manifest:
            scheme=manifest['frozen_speed_scheme']
            require(entry.get('speed_scheme')==scheme and 0<scheme['arm']<=1 and 0<scheme['hand']<=1
                and min(scheme['arm'],scheme['hand'])<1,'Native or mixed speed in slow dataset')
        else:
            require(entry.get('alpha')==manifest['frozen_alpha'] and 0<entry['alpha']<1,'Native or mixed speed in slow dataset')
        require(all(entry.get(k) is True for k in ('state_valid','multi_view_valid','tactile_valid')),'Invalid formal modality')
        require(entry.get('admission',{}).get('passed') is True,'Missing individual admission')
        admission=entry['admission'];ratio=admission.get('duration_ratio')
        bounds=(1.5,3.)
        if composition and admission.get('admission_class')=='Auxiliary':bounds=config['data_composition']['auxiliary_bounds']
        require(isinstance(ratio,(float,int)) and math.isfinite(ratio) and bounds[0]<=ratio<=bounds[1],'Measured duration ratio is invalid despite passed flag')
        if composition:
            require(admission.get('admission_class')==('Core' if 1.5<=ratio<=3. else 'Auxiliary') and not admission.get('errors'), 'Incorrect composition classification')
        speeds=admission.get('actual_phase_speed_ratios',{})
        for phase in ('grasp','lift_transport','insertion'):
            value=speeds.get(phase)
            require(isinstance(value,(float,int)) and math.isfinite(value) and 0<=value<=config['development']['maximum_phase_speed_ratio'],
                'Actual phase speed missing or not slower: '+phase)
        if config.get('speed_scheme',{}).get('hand_target_cap_rad_s') is not None:
            require(manifest.get('frozen_speed_scheme')==config['speed_scheme'],
                'Native or mixed speed does not match current validated scheme')
            require(not physical_motion_errors(admission.get('physical_motion_comparison',{}),config['physical_motion_gate']),
                'Missing or invalid physical speed/path evidence for capped production')
        require(entry.get('reference_sha256') and entry.get('visual_acceptance_sha256'),'Missing reference/content evidence')
    if composition:
        core,auxiliary=select_composition(entries,config)
        require(len(core)+len(auxiliary)==len(entries),'Auxiliary count/family balance exceeded')
        entries=core;families={e['family']:e['split'] for e in core}
    require(len(entries)>=config['collection']['minimum_accepted_episodes'],'Insufficient accepted Core episodes, no quantity backfill')
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
    if (collection/'accepted_foundation.json').exists():
        accepted_foundation(collection)
        return True
    for name in ('gpu_parity','gradient_gate'):
        gate=read(collection/(name+'.json'))
        require(gate.get('passed') and all(gate.get(k)==v for k,v in expected.items()),'Formal GPU gate dataset/statistics mismatch: '+name)
    for name in ('restore_gate','actuator_gate'):
        require(read(collection/(name+'.json')).get('passed') is True,'Missing completed '+name)
    return True


def accepted_foundation(root):
    """Reuse user-accepted engineering evidence only for unchanged implementations."""
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    receipt=json.loads((root/'accepted_foundation.json').read_text())
    require(receipt['baseline_commit']=='e67bfc6fa1b66359b55e019ae9899240625d5295','Accepted baseline changed')
    for path,expected in receipt['unchanged_implementations'].items():
        require(sha(path)==expected,'Accepted implementation changed: '+path)
    for name,item in receipt['gates'].items():
        require(sha(item['path'])==item['sha256'],'Accepted receipt changed: '+name)
        require(json.loads(Path(item['path']).read_text()).get('passed') is True,'Accepted gate did not pass: '+name)
    require(set(receipt['gates'])=={'gpu_parity','gradient_gate','restore_gate','actuator_gate'},'Missing accepted engineering evidence')
    return True


def scheme_reliability(rows, plan):
    """Separate attempt reliability, individual admission and family coverage."""
    scheme=plan['speed_scheme'];families=plan['families'];rules=plan['reliability']
    require(all(r.get('speed_scheme')==scheme for r in rows),'Mixed scheme cannot estimate reliability')
    require(len({r['identity'] for r in rows})==len(rows),'Duplicate validation attempt')
    require(all(r['family'] in families for r in rows),'Unregistered validation family')
    success=sum(bool(r.get('legal') and r.get('software_complete') and r.get('success')) for r in rows)
    admitted=sum(bool(r.get('admission',{}).get('passed')) for r in rows)
    coverage={f:dict(attempts=len(rr),successes=sum(bool(r.get('success')) for r in rr),
        admitted=sum(bool(r.get('admission',{}).get('passed')) for r in rr))
        for f in families for rr in [[r for r in rows if r['family']==f]]}
    complete=len(rows)==plan['slow_attempts']
    task_ok=complete and success>=rules['minimum_task_successes']
    yield_ok=complete and admitted>=rules['minimum_admitted']
    coverage_ok=complete and all(c['attempts']==plan['repeats_per_family'] and c['successes']>=rules['minimum_family_successes']
        and c['admitted']>=rules['minimum_family_admitted'] for c in coverage.values())
    remaining=plan['slow_attempts']-len(rows)
    impossible=(success+remaining<rules['minimum_task_successes'] or admitted+remaining<rules['minimum_admitted']
        or any(c['successes']+plan['repeats_per_family']-c['attempts']<rules['minimum_family_successes']
        or c['admitted']+plan['repeats_per_family']-c['attempts']<rules['minimum_family_admitted'] for c in coverage.values()))
    return dict(passed=task_ok and yield_ok and coverage_ok,complete=complete,attempts=len(rows),successes=success,admitted=admitted,
        task_reliability_passed=task_ok,admission_yield_passed=yield_ok,coverage_passed=coverage_ok,coverage=coverage,
        preregistered_threshold_unreachable=impossible,scope='Finite pilot evidence; individual acceptance is not attempt success; no population success guarantee')
