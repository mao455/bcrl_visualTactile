"""Finite remaining stages of the current run, with evidence-based transitions."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE,RUN,read,write,supervised_owner,sha,kill_owned,proc
from tools.tight_state import capture,digest,beat,check_protected
from visuotactile.tight_gate import paired_admission,validate_formal_manifest,require,scheme_reliability,accepted_foundation,physical_motion_errors,composition_admission,select_composition,capture_configuration


def cap_motion_admission(row, reference, rules):
    """Supplement unchanged per-record admission with real motion and no detour."""
    import h5py
    import numpy as np
    from tools.narrow_repair_audit import path_speed_metrics,path_speed_comparison
    from visuotactile.expert_speed import cap_hand_target,progress_target
    import torch
    if not row.get('path') or not reference.get('path'):
        return dict(passed=False,errors=['missing_physical_capture'])
    path=Path(row['path']).parents[1]/'numeric.h5'
    other=Path(reference['path']).parents[1]/'numeric.h5'
    metrics=path_speed_metrics(path);comparison=path_speed_comparison(metrics,path_speed_metrics(other))
    errors=physical_motion_errors(comparison,rules)
    with h5py.File(path) as f:
        g=f['episode'];prev=torch.from_numpy(g['transition/target_before_rad'][:])
        native=torch.from_numpy(g['transition/speed_native_target_rad'][:]);scheme=row['speed_scheme']
        expected=cap_hand_target(prev,progress_target(prev,native,scheme['arm'],scheme['hand']),scheme['hand_target_cap_rad_s'])
        actual=torch.from_numpy(g['transition/target_after_rad'][:])
        error=float((expected-actual).abs().max())
        maximum=float((actual[:,7:]-prev[:,7:]).abs().max()*60)
        requested=float((native[:,7:]-prev[:,7:]).abs().max()*60)
        before=g['transition/diagnostics_before'];after=g['transition/diagnostics_after']
        actual_hand_rms=float(np.sqrt(np.mean(before['joint_velocity_rad_s'][:,7:]**2)))
        sub=after['successes'][:].reshape(len(actual),-1).max(-1)
        transitions={str(k):int(np.flatnonzero(sub>=k)[0]+1) if (sub>=k).any() else None for k in (1,2)}
    if error>1e-6:errors.append('cap_applied_target_formula_mismatch')
    return dict(passed=not errors,errors=errors,metrics=metrics,comparison=comparison,
        target_formula_max_error_rad=error,maximum_hand_target_speed_rad_s=maximum,
        maximum_native_hand_target_request_rad_s=requested,actual_hand_rms_rad_s=actual_hand_rms,
        native_subgoal_completed_ticks=transitions)


def hand_cap_pilot(root):
    from tools.tight_state import environment
    s=read(root/'session.json');plan=read(root/'hand_cap_plan.json')
    require(not (root/'hand_cap_results.json').exists(),'No cap pilot restart/replacement')
    for path,expected in plan['protected_references'].items():require(sha(path)==expected,'Reference changed: '+path)
    app,env,player,lease=environment(root);lease.deadline=plan['deadline'];rows=[]
    for pair in plan['pairs']:
        ref=pair['reference'];spec=dict(pair['spec'],identity=pair['pair_id']+'_cap')
        row=capture(root,env,player,lease,spec,None,speed_scheme=plan['speed_scheme'])
        row['admission']=paired_admission(row,ref,s['development'])
        for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
            if row.get(key)!=ref.get(key):row['admission']['errors'].append(key+'_mismatch');row['admission']['passed']=False
        row['physical_motion']=cap_motion_admission(row,ref,plan['physical_rules'])
        row['admission']['errors'].extend(row['physical_motion']['errors'])
        row['admission']['passed']=not row['admission']['errors']
        rows.append(row);write(root/'hand_cap_results.json',rows)
        require(row['classification'] not in ('sensor_failure','software_or_sensor_failure'),'Software/sensor fault; no replacement')
    coverage={family:dict(attempts=len(rr),successes=sum(r['success'] for r in rr),admitted=sum(r['admission']['passed'] for r in rr))
        for family in plan['families'] for rr in [[r for r in rows if r['family']==family]]}
    successes=sum(r['success'] for r in rows);admitted=sum(r['admission']['passed'] for r in rows)
    rules=plan['pilot_improvement']
    passed=(len(rows)==rules['complete_attempts'] and successes>=rules['minimum_task_successes'] and
        admitted>=rules['minimum_admitted'] and all(x['admitted']>=rules['minimum_family_admitted'] for x in coverage.values()))
    write(root/'hand_cap_summary.json',dict(complete=True,passed=passed,attempts=len(rows),successes=successes,admitted=admitted,
        coverage=coverage,scope='Pilot improvement gate only; independent reliability required before production',
        parameters=plan['speed_scheme'],automatic_parameter_search=False))


def validate_speed(root):
    """One preregistered fixed scheme, distinct post-reset event seeds, no search."""
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json');plan=read(root/'speed_validation_plan.json')
    require(digest(s['domain'])==plan['domain_sha256'],'Registered domain changed')
    require(s['speed_scheme']==plan['speed_scheme'],'Registered scheme changed')
    require(not (root/'validation_results.json').exists(),'No blind validation restart')
    app,env,player,lease=environment(root);lease.deadline=plan['deadline']
    refs=[];slow=[];halt=None
    for pair in plan['pairs']:
        spec=dict(pair,split='development',cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=direct_configuration(s['domain'],pair['seed']))
        spec['identity']=pair['pair_id']+'_reference';reference=capture(root,env,player,lease,spec,1.)
        refs.append(reference)
        spec['identity']=pair['pair_id']+'_slow'
        row=capture(root,env,player,lease,spec,None,speed_scheme=plan['speed_scheme'])
        row['admission']=paired_admission(row,reference,s['development'])
        for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
            if row.get(key)!=reference.get(key):
                row['admission']['errors'].append(key+'_mismatch');row['admission']['passed']=False
        if 'physical_rules' in plan:
            row['physical_motion']=cap_motion_admission(row,reference,plan['physical_rules'])
            row['admission']['errors'].extend(row['physical_motion']['errors'])
            row['admission']['passed']=not row['admission']['errors']
        slow.append(row);gate=scheme_reliability(slow,plan)
        gate.update(reference_attempts=len(refs),reference_successes=sum(r['success'] for r in refs))
        write(root/'validation_results.json',dict(references=refs,slow=slow,gate=gate))
        if row['classification'] in ('sensor_failure','software_or_sensor_failure') or reference['classification'] in ('sensor_failure','software_or_sensor_failure'):
            halt='Sensor/software fault; no replacement attempt';break
        if gate['preregistered_threshold_unreachable'] or sum(not r['success'] for r in refs)>plan['slow_attempts']-plan['reliability']['minimum_reference_successes']:
            halt='Preregistered reliability/yield/coverage threshold is mathematically unreachable';break
    gate=scheme_reliability(slow,plan)
    reference_ok=len(refs)==plan['slow_attempts'] and sum(r['success'] for r in refs)>=plan['reliability']['minimum_reference_successes']
    gate.update(reference_reliability_passed=reference_ok,reference_attempts=len(refs),reference_successes=sum(r['success'] for r in refs),
        passed=gate['passed'] and reference_ok,stop_reason=halt,remaining_unstarted_pairs=plan['slow_attempts']-len(slow))
    write(root/'scheme_reliability.json',gate)
    write(root/'selected_development_gate.json',dict(passed=gate['passed'],alpha=None,speed_scheme=plan['speed_scheme'],
        references=refs,slow=slow,reliability=gate,reason=halt or 'Completed registered fixed-scheme validation'))


def refine(root):
    """The one registered narrower development domain, no final-test feedback."""
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json');plan=read(root/'development_refinement_plan.json')
    require(s.get('development_revision')==2 and s['domain']==plan['domain'] and s['development']['seeds']==plan['seeds'],
        'Refinement configuration must be applied before any Kit start')
    app,env,player,lease=environment(root);lease.deadline=s['development_deadline']
    refs=[];slow=[]
    for j,seed in enumerate(s['development']['seeds']):
        family=f'development_refined_{j:02d}'
        spec=dict(identity=family+'_native',family=family,seed=seed,split='development',cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(s['domain'],seed))
        refs.append(capture(root,env,player,lease,spec,1.))
        spec['identity']=family+'_a055'
        row=capture(root,env,player,lease,spec,.55)
        row['admission']=paired_admission(row,refs[-1],s['development'])
        for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
            if row.get(key)!=refs[-1].get(key):row['admission']['errors'].append(key+'_mismatch');row['admission']['passed']=False
        slow.append(row)
        write(root/'refined_development_results.json',dict(references=refs,slow=slow))
    write(root/'selected_development_gate.json',dict(passed=all(r['admission']['passed'] for r in slow),alpha=.55,
        references=refs,slow=slow,protocol='One geometry narrowing using development evidence; original12 attempts remain in denominator'))
    actuator(root,env,player,lease,slow,.55)


def actuator(root,env,player,lease,slow,alpha):
    s=read(root/'session.json')
    # One processed-target H24 execution gate with alpha deliberately still .55
    # in the wrapper. Task success of this open-loop replay is not an admission
    # requirement; only exact issued/applied/two-physics-substep targets are.
    import h5py
    from visuotactile.state_benchmark import trial
    source=next((r for r in slow if r.get('numeric_valid')),None)
    require(source is not None,'No valid numeric source for actuator gate')
    path=Path(source['path']);original=read(path.parents[1]/'started.json')['spec']
    with h5py.File(path) as f:targets=next(iter(f['episodes'].values()))['transition/target_after_rad'][:]
    spec=dict(original,identity='processed_target_oracle',role='actuator_gate')
    with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(spec,started=time.time(),classification='started',alpha_reapplication_allowed=False))+'\n')
    result=trial(env,lease,root/'processed_target_oracle',spec,{},player,s['expert_checkpoint'],replay=targets,position=True,replay_horizon=24)
    passed=result.get('legal') and result.get('software_complete') and result.get('actuator_target_max_error',1)<1e-6 and result.get('submitted_target_max_error',1)<1e-6
    write(root/'actuator_gate.json',dict(passed=bool(passed),result=result,retained_expert_alpha=alpha,
        replay_success_required=False,source_sha256=source['source_sha256'],source_length=len(targets)))
    with (root/'completed_attempts.jsonl').open('a') as f:f.write(json.dumps(dict(result,classification='actuator_gate',admission_passed=bool(passed)))+'\n')
    require(passed,'Actual actuator target mapping failed')


def actuator_worker(root):
    from tools.tight_state import environment
    gate=read(root/'selected_development_gate.json')
    app,env,player,lease=environment(root)
    alpha=gate['alpha'];env.speed_config=dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all')
    actuator(root,env,player,lease,gate['slow'],alpha)


def prepare_extension(root):
    """Freeze a train-family-only finite extension of the accepted Core dataset."""
    cfg=read(root/'configuration.json');p=cfg['palm_training'];parent=(CODE/p['source_root']).resolve()
    original=read(parent/'configuration.json');manifest=read(parent/'manifest.json')
    require(capture_configuration(cfg)==capture_configuration(original),'Capture physics/speed/domain changed')
    require(sha(parent/'training/best.pt')==p['baseline_checkpoint_sha256'],'Protected baseline identity')
    data=read(parent/'state_data/data.json')
    require([len(data['splits'][s]) for s in ('train','validation','test')]==[156,30,22],'Accepted split identity')
    used={e['id'] for e in manifest['episodes']};digests={e['sha256'] for e in manifest['episodes']}
    train={e['family'] for e in manifest['episodes'] if e['split']=='train'}
    rows=read(parent/'formal_results.json');refs={r['family']:r for r in rows if r['alpha']==1.}
    reusable=[r for r in rows if r['identity'] not in used and r['family'] in train
        and r.get('composition_admission',{}).get('admission_class')=='Core'
        and r.get('source_sha256') not in digests and Path(r.get('path','__missing__')).is_file()]
    # The current accepted source has no unused Core; fail explicitly if that
    # inventory changes instead of silently collecting over reusable evidence.
    require(not reusable,'Unused Core discovered: incorporate exact entries before physical collection')
    families=[f for f in read(parent/'frozen_families.json') if f['family'] in train]
    require(len(families)==36 and all(f['split']=='train' and refs[f['family']]['success'] for f in families),'Frozen training families/reference mismatch')
    jobs=[];variation=p['first_variation']
    while len(jobs)<p['maximum_attempts']:
        for f in families:
            fid=int(f['family'].split('_')[-1])
            job=dict(f,identity=f['family']+f'_slow_{variation}',seed=f['seed']+variation*100000,
                event_seed=original['collection']['event_seed_start']+variation*original['collection']['families']+fid)
            require(job['identity'] not in {r['identity'] for r in rows},'Physical attempt identity already used')
            jobs.append(job)
            if len(jobs)==p['maximum_attempts']:break
        variation+=1
    write(root/'extension_plan.json',dict(parent=str(parent),unused_core=0,train_families=sorted(train),jobs=jobs,
        references={f:refs[f] for f in train},maximum_new=p['additional_maximum'],target_new=p['additional_target'],
        validation_ids=data['splits']['validation'],test_ids=data['splits']['test'],base_manifest_sha256=sha(parent/'manifest.json')))
    protected=read(root/'protected.json')
    for name in ('manifest.json','composition_manifest.json','state_data/data.json','state_data/numeric.pt',
                 'state_data/statistics.pt','training/best.pt','training/configuration.json'):
        path=parent/name;protected[str(path)]=sha(path)
    write(root/'protected.json',protected)
    write(root/'extension_results.json',[]);write(root/'extension_entries.json',[])
    write(root/'status.json',dict(state='prepared',stage='palm_acceptance',formal_updates=0,base_core=208,
        base_train=156,new_core=0,attempts=0,no_new_task_success_evidence=True))


def extend_core(root):
    from tools.tight_state import environment
    s=read(root/'session.json');cfg=read(root/'configuration.json');plan=read(root/'extension_plan.json')
    parent=Path(plan['parent']);reference_config=read(parent/'configuration.json')
    rows=[];entries=[];stop_reason='attempt_budget'
    try:
        app,env,player,lease=environment(root)
        for job in plan['jobs']:
            if len(entries)>=plan['target_new']:stop_reason='additional_target';break
            if time.time()>=s['collection_deadline']-60:stop_reason='collection_wall_budget';break
            # The admission is the existing strict Core rule, including physical
            # speed/path, full success and all recorded sensor content.
            row=capture(root,env,player,lease,job,None,speed_scheme=s['speed_scheme'])
            ref=plan['references'][job['family']]
            row['physical_motion']=cap_motion_admission(row,ref,s['physical_motion_gate'])
            admission=composition_admission(row,ref,cfg,reference_config)
            row['composition_admission']=admission;rows.append(row)
            if admission['admission_class']=='Core':
                entries.append(dict(id=row['identity'],family=row['family'],split='train',alpha=None,speed_scheme=s['speed_scheme'],
                    path=row['path'],sha256=row['source_sha256'],steps=row['steps'],size=Path(row['path']).stat().st_size,
                    state_valid=True,multi_view_valid=True,tactile_valid=True,admission=admission,
                    reference_path=ref['path'],reference_sha256=ref['source_sha256'],visual_acceptance_sha256=row['visual_acceptance_sha256']))
            write(root/'extension_results.json',rows);write(root/'extension_entries.json',entries)
            write(root/'collection_progress.json',dict(attempts=len(rows),new_core=len(entries),
                task_successes=sum(bool(r.get('success')) for r in rows),
                task_failures=sum(r['classification']=='task_failure' for r in rows),
                initial_rejections=sum(not r.get('legal',False) for r in rows),
                core_rejections=sum(r['composition_admission']['admission_class']!='Core' for r in rows),utc=time.time()))
            if row['classification'] in ('sensor_failure','software_or_sensor_failure'):
                raise RuntimeError('Collection sensor/software failure; subsequent training stopped')
            require(len(entries)<=plan['maximum_new'],'Additional Core cap')
    except TimeoutError:
        if time.time()<s['collection_deadline']:raise
        stop_reason='collection_wall_budget_interrupted_attempt'
    write(root/'collection_finished.json',dict(complete=True,stop_reason=stop_reason,attempts_completed=len(rows),new_core=len(entries)))


def freeze_extension(root):
    from visuotactile.narrow_data import build_cache
    plan=read(root/'extension_plan.json');parent=Path(plan['parent']);cfg=read(root/'configuration.json')
    original=read(parent/'manifest.json');added=read(root/'extension_entries.json')
    require(sha(parent/'manifest.json')==plan['base_manifest_sha256'],'Protected manifest changed')
    require(len(added)<=plan['maximum_new'] and all(e['split']=='train' and e['family'] in plan['train_families'] for e in added),'Extension split or count violation')
    combined=original['episodes']+added
    require(len({e['id'] for e in combined})==len(combined) and len({e['sha256'] for e in combined})==len(combined),'Duplicate demonstration')
    manifest=dict(original,episodes=combined,collection_finished=True,base_manifest_sha256=plan['base_manifest_sha256'],
        additional_core=len(added),configuration_sha256=read(root/'session.json')['configuration_sha256'],
        scope='Accepted Core plus finite new train-only successes; Auxiliary excluded')
    validate_formal_manifest(manifest,cfg);write(root/'manifest.json',manifest)
    data_root=build_cache(root,reuse_source=parent);data=read(data_root/'data.json')
    require(data['splits']['validation']==plan['validation_ids'] and data['splits']['test']==plan['test_ids'],'Validation/test identities changed')
    attempts=[json.loads(line) for line in (root/'attempts.jsonl').read_text().splitlines()] if (root/'attempts.jsonl').exists() else []
    completed=read(root/'extension_results.json')
    require(len(attempts)<=cfg['palm_training']['maximum_attempts'],'Physical attempts exceeded bound')
    evidence=dict(passed=True,base_core=208,base_train=156,new_core=len(added),total_core=len(combined),
        split_counts={k:len(v) for k,v in data['splits'].items()},physical_attempts=len(attempts),
        completed_attempts=len(completed),interrupted_attempts=len(attempts)-len(completed),
        task_failures=sum(r['classification']=='task_failure' for r in completed),initial_rejections=sum(not r['legal'] for r in completed),
        core_rejections=sum(r['composition_admission']['admission_class']!='Core' for r in completed),
        manifest_sha256=sha(root/'manifest.json'),auxiliary_added=0,validation_test_preserved=True)
    write(root/'DATA_READY.json',evidence);write(root/'COMPLETE.json',evidence)


def palm_report(root):
    status=read(root/'status.json');data=read(root/'DATA_READY.json') if (root/'DATA_READY.json').exists() else {}
    progress=read(root/'training/progress.json') if (root/'training/progress.json').exists() else {}
    weights=read(root/'configuration.json')['loss_weights']
    start,end='<!-- tight-palm-training:start -->','<!-- tight-palm-training:end -->'
    lines=[start,'## 掌部几何损失与有限训练数据扩充（当前任务）','',
        f"状态 **{status['state']}**，阶段 `{status['stage']}`；{status.get('reason','')}。",
        f"基础208 Core/train156；新增{data.get('new_core','待冻结')}，物理尝试{data.get('physical_attempts','见collection_progress.json')}。validation30/test22及固定诊断身份不变，Aux不加入。",
        f"全新训练更新{progress.get('update',0)}，anchor等效epoch {progress.get('anchor_equivalent_epochs',0):.3f}；目标300，平台停止最早200，训练墙钟9h优先。",
        f"loss = 原完整29D扩散Loss B + {weights['position']:.10g} × 平均掌部位置平方距离(m²) + {weights['rotation']:.10g} × 平均0.5旋转矩阵Frobenius平方；仅有效commit12，500更新渐入。",
        '实际掌部/关节/决策前执行器目标和严格过去已施加目标已有60Hz通路，复用178/85维，不重复追加；GRU256不变。',
        '本轮尚无新的任务成功证据；完成训练/离线验证后停止，不自动物理闭环。',
        f'run：`{root}`；`status.json`、`training/progress.json`、`training/validation.jsonl`、`training/curves.png`、`offline/stage_metrics.json`。',
        '已有旧诊断38条/1103次完整采样/13000tick仅为离线参照，不是任务成功数。','',
        '|实际字段/切片|动作前时间语义与来源|每tick网络路径|重置与反传|',
        '|---|---|---|---|',
        '|robot[0:58]：q29/qd29，rad/rad·s⁻¹|diagnostics_before实测canonical状态，时刻t|60Hz native_encoder→GRU；当前/过去13帧另直达condition|真实episode重置；完整0…t前向，当前权重重算前缀|',
        '|geometry[28:35]：掌部xyz3+wxyz4|iiwa14_link_7实测刚体，robot base坐标，m/单位四元数；非控制目标FK|60Hz geometry_encoder→GRU；t−24/t−12/t另直达condition|同上，四元数w≥0；train-only mean/std|',
        '|robot[58:116]：目标29/目标−q29|target_before_rad＝动作前已生效执行器目标，rad|同robot两条路径|不读取本tick target_after标签|',
        '|robot[116:174]：过去目标29×2|严格t−1实际施加target_after；保留既有两个兼容槽，两者相等；不是专家滤波前request|每60Hz tick反馈，commit内部仍更新GRU|t=0置零，valid=0；重规划不清空|',
        '|robot[174:178]：age2/valid/elapsed|前次目标距当前时间，s/bool/s|同robot路径；valid不标准化|真实episode重置，缺失窗口全零且valid=false|',
        '|Actor141与geometry其余78维|既有Actor噪声/延迟、物体/目标/孔/桌/指尖几何；不新引入传感器|Actor/robot/geometry每tick拼384→GRU256；条件仍896→512|anchor t的反传区间[96⌊t/96⌋,t]，≤96tick（当前12tick anchors为1–85）；更早历史no_grad重算，不缓存旧权重hidden|','',
        '可丢弃验收：checkpoint/EMA/normalizer完全匹配，109tick实际在线controller与完整prefix输入完全相同，condition max1.90735e−6；FK真实位姿与梯度验收见palm_source_check.json，最终新增数据统计下再验收。',end]
    if (root/'offline/stage_metrics.json').exists():
        metrics=read(root/'offline/stage_metrics.json')
        summary=['','|固定专家历史诊断|Arm/hold (rad)|Hand/hold (rad)|掌部FK (mm/deg)|tick|','|---|---:|---:|---:|---:|']
        for a in (r for r in metrics if r['group']=='arm' and r['motion']=='all'):
            b=next(r for r in metrics if r['split']==a['split'] and r['phase']==a['phase'] and r['group']=='hand' and r['motion']=='all')
            fmt=lambda r:'未测' if not r['ticks'] else f"{r['rmse_rad']:.6f}/{r['hold_rmse_rad']:.6f}"
            geom='未测' if not a['ticks'] else f"{a['ee_rmse_mm']:.4f}/{a['palm_angle_mean_deg']:.4f}"
            summary.append(f"|{a['split']}/{a['phase']}|{fmt(a)}|{fmt(b)}|{geom}|{a['ticks']}|")
        lines[-1:-1]=summary
    path=CODE/'research_docs/STATE_DP_RESULTS.md';old=path.read_text();content='\n'.join(lines)
    if start in old:old=old[:old.index(start)]+content+old[old.index(end)+len(end):]
    else:old=old.replace('\n','\n\n'+content+'\n',1)
    tmp=path.with_suffix('.md.tmp');tmp.write_text(old);tmp.replace(path)


def collect(root):
    if read(root/'configuration.json').get('palm_training'):
        return extend_core(root)
    import numpy as np
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration,build_cache
    s=read(root/'session.json');cfg=read(root/'configuration.json');gate=read(root/'selected_development_gate.json')
    require(gate['passed'],'Development did not pass');c=s['collection'];alpha=gate['alpha']
    scheme=gate.get('speed_scheme',dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all'))
    require(scheme==s.get('speed_scheme',scheme),'Validated speed scheme differs from production configuration')
    if scheme.get('hand_target_cap_rad_s') is not None:
        require('physical_motion_gate' in s,'Extract physical admission into standalone current configuration before production')
    families=[];order=np.random.default_rng(c['split_seed']).permutation(c['families'])
    splits={int(i):'train' if j<c['train_families'] else 'validation' if j<c['train_families']+c['validation_families'] else 'test' for j,i in enumerate(order)}
    for i in range(c['families']):
        seed=c['seed_start']+i;family=f'formal_{i:03d}'
        families.append(dict(family=family,seed=seed,cohort='fresh',split=splits[i],collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=direct_configuration(s['domain'],seed)))
    write(root/'frozen_families.json',families)
    jobs=[]
    for f in families:jobs.append(dict(f,identity=f['family']+'_reference',alpha=1.,speed_scheme=dict(kind='closed_loop',arm=1.,hand=1.,phase='all')))
    for j in range(c['variations_per_family']):
        for f in families:
            # Same fixed-pose family/configuration; independent native joint,
            # observation/delay/wrench RNG draws. No alteration of DR settings.
            jobs.append(dict(f,identity=f['family']+f'_slow_{j}',seed=f['seed']+j*100000,
                event_seed=c['event_seed_start']+j*c['families']+int(f['family'].split('_')[-1]),alpha=alpha,speed_scheme=scheme))
    require(len(jobs)<=c['maximum_attempts'],'Predetermined attempt cap')
    write(root/'frozen_collection_plan.json',dict(jobs=jobs,configuration_sha256=s['configuration_sha256'],alpha=alpha,speed_scheme=scheme,
        reference_scope='Same fixed-pose family and native reset/DR configuration; variation seeds change joint/noise/event draws, not an exact hidden-state pair'))
    refs={};entries=[];results=[];consecutive=0;skipped=[];old_ids=set()
    composition='data_composition' in s
    reference_config=cfg
    def entry_for(row,ref,admission):
        return dict(id=row['identity'],family=row['family'],split=row['split'],alpha=alpha,speed_scheme=scheme,
            path=row['path'],sha256=row['source_sha256'],steps=row['steps'],size=Path(row['path']).stat().st_size,
            state_valid=True,multi_view_valid=True,tactile_valid=True,admission=admission,
            reference_path=ref['path'],reference_sha256=ref['source_sha256'],visual_acceptance_sha256=row['visual_acceptance_sha256'])
    if (root/'production_reuse.json').exists():
        require(composition,'Explicit composition contract required for historical reuse')
        reuse=read(root/'production_reuse.json')
        for path,expected in reuse['immutable_files'].items():require(sha(path)==expected,'Reuse evidence drift: '+path)
        prior=Path(reuse['root']);reference_config=read(prior/'configuration.json')
        require(capture_configuration(cfg)==capture_configuration(reference_config),'Reference physical capture configuration differs')
        require(read(prior/'frozen_families.json')==families,'Reuse family/split identity differs')
        results=read(prior/'formal_results.json');old_ids={r['identity'] for r in results}
        require(old_ids<={j['identity'] for j in jobs},'Reuse contains unknown planned starts')
        refs={r['family']:r for r in results if r['alpha']==1.}
        for row in results:
            if row['alpha']==1.:continue
            admission=composition_admission(row,refs[row['family']],cfg,reference_config)
            row['composition_admission']=admission
            available=Path(row['path']).is_file() if row.get('path') else False
            row['composition_payload_available']=available
            if admission['passed'] and available:entries.append(entry_for(row,refs[row['family']],admission))
        historical_slow=[r for r in results if r['alpha']!=1.]
        write(root/'reuse_classification.json',dict(historical_strict_successes=sum(r.get('admission',{}).get('passed',False) for r in historical_slow),historical_slow_attempts=len(historical_slow),
            source=str(prior),rows=[r for r in results if r['alpha']!=1.],reusable=len(entries)))
    app,env,player,lease=environment(root)
    budget_stop=None
    for index,job in enumerate(jobs):
        if job['identity'] in old_ids:continue
        if composition and job['alpha']!=1. and not refs[job['family']].get('success'):
            skipped.append(dict(identity=job['identity'],reason='known_unavailable_successful_reference',
                seed=job['seed'],event_seed=job.get('event_seed'),family=job['family']))
            write(root/'unstarted_dependencies.json',skipped);continue
        import shutil
        if shutil.disk_usage(root).free<s['reserve_bytes']+5*1024**3 or (composition and time.time()>=s['collection_deadline']-1800):
            core=select_composition(entries,cfg)[0] if composition else entries
            candidate=dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,frozen_alpha=alpha,frozen_speed_scheme=scheme,episodes=core)
            validate_formal_manifest(candidate,cfg)
            budget_stop=dict(reason='Fixed reserve or inherited collection deadline with30min freeze allowance',
                unstarted=len(jobs)-index,completed=len(results),accepted=len(entries));break
        row=capture(root,env,player,lease,job,job['alpha'],speed_scheme=job['speed_scheme']);results.append(row)
        if job['alpha']==1.:refs[job['family']]=row
        else:
            ref=refs[job['family']];admission=paired_admission(row,ref,s['development']);row['admission']=admission
            if scheme.get('hand_target_cap_rad_s') is not None:
                row['physical_motion']=cap_motion_admission(row,ref,s['physical_motion_gate'])
                admission['physical_motion_comparison']=row['physical_motion'].get('comparison',{})
                admission['errors'].extend(row['physical_motion']['errors']);admission['passed']=not admission['errors']
            if composition:
                admission=composition_admission(row,ref,cfg,reference_config)
                row['composition_admission']=admission
                row['admission']=admission['strict_admission']
            if admission['passed']:
                entries.append(entry_for(row,ref,admission))
            elif not composition and row.get('numeric_valid') and row.get('visual_passed') and row.get('software_complete'):
                # Classified, complete rejected NEW captures keep full numeric
                # traces, initial/config evidence, sensor hashes and attempt row.
                # Never remove protected400/275 or unknown partial/sensor faults.
                path=Path(row['path']);require(root/'episodes' in path.parents,'New own capture only')
                require(sha(path)==row['source_sha256'],'Rejected payload changed before cleanup')
                numeric=path.parents[1]/'numeric.h5';require(numeric.is_file(),'Numeric failure evidence required')
                st=path.stat();path.unlink()
                row['payload_retention']=dict(mode='classified_rejected_rgbd_removed_numeric_and_small_evidence_retained',
                    numeric_path=str(numeric),numeric_sha256=sha(numeric),deleted_source_sha256=row['source_sha256'],
                    released_bytes=st.st_blocks*512 if st.st_nlink==1 else 0)
                with (root/'collection_cleanup.jsonl').open('a') as stream:stream.write(json.dumps(dict(identity=row['identity'],**row['payload_retention']))+'\n')
        if row['classification']=='task_failure':consecutive+=1
        elif row['classification']=='task_success':consecutive=0
        write(root/'formal_results.json',results)
        core,auxiliary=select_composition(entries,cfg) if composition else (entries,[])
        write(root/'manifest.pending.json',dict(version='tight_slow_multimodal_v1',collection_finished=False,
            development_passed=True,frozen_alpha=alpha,frozen_speed_scheme=scheme,episodes=core))
        if composition:
            write(root/'composition.pending.json',dict(core=len(core),auxiliary=len(auxiliary),
                candidate_episodes=entries,all_slow_attempts=sum(r['alpha']!=1. for r in results),
                core_families=len({e['family'] for e in core}),auxiliary_families=len({e['family'] for e in auxiliary})))
        if row.get('classification') in ('sensor_failure','software_or_sensor_failure'):raise RuntimeError('Sensor/software failure; no promotion')
        if consecutive>=c['maximum_consecutive_task_failures']:raise RuntimeError('Preregistered consecutive task-failure stop')
        completed_slow=[r for r in results if r['alpha']!=1.]
        if c.get('reliability_window') and len(completed_slow) and len(completed_slow)%c['reliability_window']==0:
            legal_slow=[r for r in completed_slow if r.get('legal')]
            task_rate=sum(r['success'] for r in legal_slow)/max(1,len(legal_slow))
            yield_rate=sum(r.get('admission',{}).get('passed',False) for r in completed_slow)/len(completed_slow)
            require(task_rate>=c['minimum_production_task_rate'] and (composition or yield_rate>=c['minimum_production_yield']),
                'Preregistered cumulative production reliability/yield stop')
        if composition and job['alpha']!=1. and job['family']==families[-1]['family'] and len(core)>=s['data_composition']['core_target']:
            check=dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,
                frozen_alpha=alpha,frozen_speed_scheme=scheme,episodes=core)
            validate_formal_manifest(check,cfg)
            budget_stop=dict(reason='Core soft target reached at complete predetermined family round',unstarted=len(jobs)-index-1);break
    if composition:entries,auxiliary=select_composition(entries,cfg)
    manifest=dict(version='tight_slow_multimodal_v1',collection_finished=True,development_passed=True,
        frozen_alpha=alpha,frozen_speed_scheme=scheme,episodes=entries,all_attempts=len(results),configuration_sha256=s['configuration_sha256'],
        original400_included=False,original275_included=False,quarantined=False,budget_stop=budget_stop)
    validate_formal_manifest(manifest,cfg)
    if composition:
        combined=dict(manifest,version='tight_slow_multimodal_composition_v2',data_composition=s['data_composition'],
            episodes=entries+auxiliary,core_count=len(entries),auxiliary_count=len(auxiliary))
        validate_formal_manifest(combined,cfg)
    # Recheck every accepted payload and both modalities at freeze, never trust
    # flags copied from the old legacy camera path.
    from visuotactile.camera_validity import inspect_capture
    from visuotactile.tight_gate import numeric_admission
    for e in entries+(auxiliary if composition else []):
        require(sha(e['path'])==e['sha256'] and sha(e['reference_path'])==e['reference_sha256'],'Frozen pair source changed')
        require(numeric_admission(e['path'])[3]['success'],'Native complete success missing')
        require(inspect_capture(e['path'])['passed'],'RGB-D content failed at freeze')
    if composition:write(root/'composition_manifest.json',combined)
    write(root/'manifest.json',manifest);data=build_cache(root)
    write(root/'DATA_READY.json',dict(passed=True,episodes=len(entries),families=len({e['family'] for e in entries}),
        auxiliary_episodes=len(auxiliary) if composition else 0,training_subset='core_only',manifest_sha256=sha(root/'manifest.json')))
    write(root/'COMPLETE.json',read(root/'DATA_READY.json'))


def baseline_checkpoint(saved, input_builder):
    """Explicit evaluation-only I/O intervention; preserve source weights/state."""
    from visuotactile.position_control import contract
    require(saved.get('contract') == contract(24, True), 'Baseline requires frozen H24 command-target StateDP')
    require(saved.get('kind') == 'dp' and saved.get('updates', 0) > 0, 'Trained StateDP required')
    require(input_builder in ('legacy_packet_inverse_v1', 'canonical_joint_state_v1'), 'Unknown baseline input intervention')
    result = dict(saved)
    result['configuration'] = dict(saved.get('configuration', {}), input_builder=input_builder)
    return result


def baseline_evaluate(root):
    """Paired old275 diagnostic, separate from formal slow-data admission."""
    import torch
    from visuotactile.state_benchmark import trial
    from tools.tight_state import environment
    s=read(root/'session.json');plan=read(root/'baseline_plan.json')
    require(plan['dataset_scope']=='original275_state_valid_mixed_speed', 'Explicit historical baseline scope required')
    require(not (root/'baseline_results.json').exists(), 'No baseline restart/replacement')
    for path,expected in plan['protected_inputs'].items():require(sha(path)==expected, 'Baseline source drift: '+path)
    saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
    require(len(plan['input_builders']) in (1,2) and len(set(plan['input_builders']))==len(plan['input_builders']),'One independent variant or two distinct paired variants required')
    variants={key:baseline_checkpoint(saved,key) for key in plan['input_builders']}
    app,env,player,lease=environment(root,True);rows=[];pairs=[]
    for index,pair in enumerate(plan['pairs']):
        current=[]
        order=plan['input_builders'][::1 if index%2==0 else -1]
        for builder in order:
            identity=pair['pair_id']+'_'+('legacy' if builder.startswith('legacy') else 'canonical')
            spec=dict(pair['spec'],identity=identity)
            with (root/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(identity=identity,pair_id=pair['pair_id'],input_builder=builder,started=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
            output=root/'baseline_evaluation'/identity
            result=trial(env,lease,output,spec,{},player,s['expert_checkpoint'],saved=variants[builder],position=True,inference=plan['inference'])
            result.update(pair_id=pair['pair_id'],input_builder=builder,source_student_sha256=plan['protected_inputs'][plan['checkpoint']],
                evaluation_only_input_override=True,dataset_scope=plan['dataset_scope'])
            if result['legal']:
                require(result['actuator_target_max_error']<1e-6 and result['submitted_target_max_error']<1e-6,'Baseline actuator mismatch')
            rows.append(result);current.append(result);write(root/'baseline_results.json',rows)
            require(result['software_complete'],'Baseline software fault; no replacement')
            beat(root,stage='baseline_evaluation',identity=identity,completed=len(rows))
        initial=[root/'baseline_evaluation'/x['identity']/'initial_observable.json' for x in current]
        pairs.append(dict(pair_id=pair['pair_id'],split=pair['spec']['split'],initial_observable_exact=sha(initial[0])==sha(initial[1]) if len(initial)==2 else None,
            members=[x['identity'] for x in current],successes={x['input_builder']:x['success'] for x in current},
            scope='Independent frozen canonical rollout' if len(current)==1 else 'Same native reset/config/event and diffusion seeds; observable equality checked, hidden PhysX state not asserted'))
        write(root/'baseline_pairs.json',pairs)
    summary={builder:{split:dict(attempts=len(rr),legal=sum(r['legal'] for r in rr),successes=sum(r['success'] for r in rr),
        initial_rejections=sum(not r['legal'] for r in rr),software_failures=sum(not r['software_complete'] for r in rr))
        for split in ('train','unseen') for rr in [[r for r in rows if r['input_builder']==builder and r['split']==split]]} for builder in plan['input_builders']}
    for path,expected in plan['protected_inputs'].items():require(sha(path)==expected,'Baseline source changed during evaluation')
    write(root/'baseline_summary.json',dict(complete=True,cohorts=summary,physical_attempts=len(rows),pairs=len(pairs),
        paired_comparisons=sum(len(p['members'])==2 for p in pairs),
        observable_exact_pairs=sum(p['initial_observable_exact'] is True for p in pairs),optimizer_updates=0,
        dataset_scope=plan['dataset_scope'],formal_slow_data_admitted=False,final_test_for_selection=False))


def evaluation_family_plan(families, manifest, evaluation):
    """A planned family without an admitted Core is not a training family."""
    available={f['family']:f for f in families};admitted={}
    for entry in manifest['episodes']:
        family=entry['family'];split=entry['split']
        require(family in available and available[family]['split']==split,'Evaluation family/source split mismatch')
        require(admitted.setdefault(family,split)==split,'Evaluation family leakage')
    plan=[]
    for split,key in [('train','train_families'),('validation','validation_families')]:
        selected=sorted(f for f,s in admitted.items() if s==split)[:evaluation[key]]
        plan.extend(dict(available[f],identity='evaluation_'+f) for f in selected)
    return plan


def evaluate(root):
    import torch
    from visuotactile.state_benchmark import trial
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json');app,env,player,lease=environment(root,True)
    with lease.block():saved=torch.load(root/'training/best.pt',map_location='cpu',weights_only=False)
    families=read(root/'frozen_families.json');e=s['evaluation']
    # Select by admitted dataset membership before any student result is known.
    plan=evaluation_family_plan(families,read(root/'manifest.json'),e)
    for j in range(e['new_initial_attempts']):
        seed=e['new_seed_start']+j
        plan.append(dict(identity=f'new_{j:03d}',family=f'new_{j:03d}',split='new',seed=seed,cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(s['domain'],seed)))
    write(root/'frozen_evaluation_plan.json',plan);results=[]
    for spec in plan:
        previous=[r for r in results if r['split']=='new']
        if spec['split']=='new' and len(previous)>=e['initial_probe'] and not any(r['success'] for r in previous):break
        if spec['split']=='new' and len(previous)>=e['zero_success_stop_at'] and not any(r['success'] for r in previous[:e['zero_success_stop_at']]):break
        result=trial(env,lease,root/'evaluation'/spec['identity'],spec,{},player,s['expert_checkpoint'],saved=saved,position=True,
            inference=dict(inference_steps=16,clip_sample=False,overlap_steps=0,sampler_seed=e['sampler_seed']))
        if result['legal']:
            require(result['actuator_target_max_error']<1e-6 and result['submitted_target_max_error']<1e-6,'Actual actuator mismatch')
        results.append(result);write(root/'evaluation_results.json',results)
        beat(root,stage='evaluation',identity=spec['identity'])
    cohorts={k:dict(attempts=len(rr),legal=sum(r['legal'] for r in rr),successes=sum(r['success'] for r in rr),
        initial_rejections=sum(not r['legal'] for r in rr)) for k in ('train','validation','new') for rr in [[r for r in results if r['split']==k]]}
    write(root/'evaluation_summary.json',dict(complete=True,cohorts=cohorts,checkpoint_sha256=sha(root/'training/best.pt'),
        final_test_for_selection=False,new_denominator_not_padded=True))


@supervised_owner
@supervised_owner
def pipeline(root,refinement=False,speed_validation=False,baseline=False,mechanism=False):
    import fcntl
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=read(root/'session.json');monitor=None;child=None
        palm=s.get('palm_training')
        if palm:
            require(read(root/'status.json')['state']=='prepared' and not (root/'execution_source_freeze.json').exists(),'No duplicate/restarted palm run')
            check=read(root/'palm_source_check.json');require(check['passed'],'Focused source/gradient check required')
            for path,expected in check['source_sha256'].items():require(sha(CODE/path)==expected,'Accepted implementation drift: '+path)
            require(not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','configs','tests'],cwd=CODE,text=True).strip(),'Commit runtime before launch')
            require(read(root/'configuration.json')==read(CODE/'configs/tight_state.json'),'Prepared configuration differs from committed configuration')
            now=time.time();s.update(started=now,deadline=now+s['maximum_hours']*3600,
                collection_deadline=now+s['collection_hours']*3600,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
                cwd=str(CODE),python=str(Path(sys.executable).resolve()))
            write(root/'session.json',s);write(root/'gpu_schedule.json',read(root/'configuration.json')['resources'])
            write(root/'execution_source_freeze.json',{str(p):sha(p) for folder in ('tools','visuotactile','isaacsimenvs','evaluation','rl_games') for p in (CODE/folder).rglob('*.py')})
            write(root/'source_freeze.json',read(root/'execution_source_freeze.json'))
        (root/'STOP_MONITOR').unlink(missing_ok=True)
        def worker(stage,args,deadline,bounded_collection=False):
            nonlocal child
            if (root/'execution_source_freeze.json').exists():
                for path,expected in read(root/'execution_source_freeze.json').items():require(sha(path)==expected,'Execution source drift: '+path)
            path=root/'stages'/stage;path.mkdir(parents=True,exist_ok=False)
            write(root/'status.json',dict(state='running',stage=stage,utc=time.time()))
            with (path/'worker.log').open('w') as log:child=subprocess.Popen(RUN+args,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'active_owner.json',dict(owner=proc(os.getpid()),worker=proc(child.pid),monitor=proc(monitor.pid)))
            while child.poll() is None:
                if monitor.poll() is not None:raise RuntimeError('Monitor stopped')
                if time.time()>=deadline:
                    if bounded_collection:
                        require(not (root/'collect_error.json').exists(),'Software error at collection deadline; do not promote to budget stop')
                        kill_owned(child)
                        write(root/'collection_finished.json',dict(complete=True,stop_reason='collection_wall_budget_owned_worker_stopped',
                            completed=len(read(root/'extension_results.json')),new_core=len(read(root/'extension_entries.json'))))
                        write(path/'exit.json',dict(returncode=child.returncode,budget_stop=True,utc=time.time()))
                        return
                    raise TimeoutError('Fixed deadline')
                if (root/'STOP').exists():raise InterruptedError('Owned STOP')
                if palm:
                    status=dict(state='running',stage=stage,pid=os.getpid(),worker_pid=child.pid,monitor_pid=monitor.pid,
                        utc=time.time(),hard_stop_utc=deadline,run=str(root),formal_updates=0,no_new_task_success_evidence=True)
                    for field in ('collection_progress.json','training/progress.json','offline/progress.json'):
                        if (root/field).exists():status.update(read(root/field))
                    if (root/'training/progress.json').exists():status['formal_updates']=status.get('update',0)
                    status.update(last_progress_utc=status.get('utc'),utc=time.time(),hard_stop_utc=deadline)
                    write(root/'status.json',status)
                time.sleep(2)
            write(path/'exit.json',dict(returncode=child.returncode,utc=time.time()))
            if child.returncode:raise RuntimeError(stage+' failed; no blind retry')
        try:
            with (root/'monitor.log').open('a') as log:monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if palm:
                palm_report(root)
                worker('collection',['-m','tools.tight_finish','collect','--root',str(root)],s['collection_deadline']-45,bounded_collection=True)
                worker('freeze',['-m','tools.tight_finish','freeze_extension','--root',str(root)],min(s['deadline']-10*3600,time.time()+1800))
                worker('palm_acceptance',['-m','tools.tight_audit','palm','--root',str(root),'--data',str(root/'state_data')],time.time()+1200)
                data=read(root/'state_data/data.json');ids=data['splits']['train'];anchors=sum(math.ceil(data['entries'][i]['length']/12) for i in ids)
                t=s['training'];batch=t['effective_batch'];groups=math.ceil(len(ids)/batch)
                def updates_for(draws):
                    rounds,left=divmod(math.ceil(draws),len(ids));return rounds*groups+math.ceil(left/batch)
                total=updates_for(300*anchors);train=root/'training';train.mkdir(exist_ok=False)
                old_rows=[json.loads(line) for line in (Path(palm['source_root'])/'training/training.jsonl').read_text().splitlines()]
                old_seconds=(old_rows[-1]['utc']-old_rows[0]['utc'])/(old_rows[-1]['update']-old_rows[0]['update'])
                budget=dict(anchor_population=anchors,train_episodes=len(ids),effective_batch_max=batch,short_tail=len(ids)%batch,
                    mean_batch=len(ids)/groups,total_updates=total,target_anchor_epochs=300,early_stop_minimum_update=updates_for(200*anchors),
                    comparison_updates=sorted(set([8500,updates_for(100*anchors)])),
                    baseline_seconds_per_update=old_seconds,initial_eta_seconds_range=[.6*old_seconds*total,1.3*old_seconds*total],
                    eta_basis='Old measured throughput; new palm-only FK may be faster; real worker updates this estimate',
                    selection=palm['selection'],wall_hours=9,validation_interval=t['validation_interval'])
                write(train/'budget.json',budget)
                deadline=min(s['deadline']-3600,time.time()+9*3600)
                runtime=dict(resource_policy=s['resource_policy'],external_monitor_root=str(root),training_deadline=deadline,
                    random_seed=t['random_seed'],validation_interval=t['validation_interval'],
                    early_stop_minimum_update=budget['early_stop_minimum_update'],comparison_updates=budget['comparison_updates'],
                    deterministic_validation_loss=True,stage_stop_update=total)
                write(train/'session.json',runtime);palm_report(root)
                worker('training',['-m','visuotactile.anchor_continue','fresh-train','--job','state','--data',str(root/'state_data'),
                    '--root',str(train),'--target',str(total),'--output',str(train/'launch')],deadline)
                require((train/'best.pt').exists(),'No validation-selected model: stop without final diagnostic')
                s['offline_deadline']=min(s['deadline'],time.time()+palm['offline_hours']*3600);write(root/'session.json',s)
                worker('offline',['-m','tools.tight_diagnostics','final','--root',str(root)],s['offline_deadline'])
                finished=read(train/'finished.json')
                write(root/'status.json',dict(state='BUDGET_STOP' if finished['stop_reason']=='fixed_wall_budget' else 'COMPLETE',stage='trained_and_offline_evaluated',
                    formal_updates=finished['updates'],training_stop_reason=finished['stop_reason'],data=read(root/'DATA_READY.json'),
                    no_new_task_success_evidence=True,exit_code=0,utc=time.time()))
                return
            inherited=(root/'accepted_foundation.json').exists()
            if inherited:accepted_foundation(root)
            if mechanism:
                require(inherited,'Reuse accepted foundation with affected regressions')
                worker('hand_cap_pilot',['-m','tools.tight_finish','hand_cap_pilot','--root',str(root)],read(root/'hand_cap_plan.json')['deadline'])
                # A negative mechanism result does not cancel independent B.
                worker('baseline_evaluation',['-m','tools.tight_finish','baseline_evaluate','--root',str(root)],read(root/'baseline_plan.json')['deadline'])
                if not read(root/'hand_cap_summary.json')['passed']:
                    write(root/'status.json',dict(state='complete',stage='bounded_mechanism_and_baseline_evaluated',
                        mechanism=read(root/'hand_cap_summary.json'),baseline=read(root/'baseline_summary.json'),
                        formal_updates=0,formal_slow_episodes=0,production_blocked=True))
                    return
                worker('independent_cap_validation',['-m','tools.tight_finish','validate_speed','--root',str(root)],s['development_deadline'])
                if not read(root/'scheme_reliability.json')['passed']:
                    write(root/'status.json',dict(state='complete',stage='independent_validation_rejected',
                        mechanism=read(root/'scheme_reliability.json'),baseline=read(root/'baseline_summary.json'),
                        formal_updates=0,formal_slow_episodes=0,production_blocked=True))
                    return
            if baseline:
                require(inherited,'Baseline must explicitly retain accepted engineering evidence')
                worker('baseline_evaluation',['-m','tools.tight_finish','baseline_evaluate','--root',str(root)],s['deadline'])
                write(root/'status.json',dict(state='complete',stage='baseline_evaluated',formal_updates=0,formal_slow_episodes=0,
                    baseline=read(root/'baseline_summary.json')))
                return
            elif not inherited and not (root/'gpu_parity.json').exists():
                worker('foundation_fixture',['-m','tools.tight_audit','fixture','--root',str(root)],s['development_deadline'])
                worker('foundation_gpu',['-m','tools.tight_audit','gpu','--root',str(root),'--data',str(root/'diagnostic_data')],s['development_deadline'])
            if not inherited:require(read(root/'gpu_parity.json')['passed'],'Foundation full trajectory GPU gate must pass')
            if speed_validation:
                worker('speed_validation',['-m','tools.tight_finish','validate_speed','--root',str(root)],s['development_deadline'])
            if refinement:
                require(s.get('development_revision')==2,'Refinement revision missing; refuse startup')
                worker('refined_development_v2',['-m','tools.tight_finish','refine','--root',str(root)],s['development_deadline'])
            gate=read(root/'selected_development_gate.json')
            if not gate['passed']:raise ValueError('Fixed-scheme gate failed: '+json.dumps(gate.get('reliability',gate.get('reason'))))
            if not inherited and not (root/'actuator_gate.json').exists():
                worker('actuator_gate',['-m','tools.tight_finish','actuator_worker','--root',str(root)],s['development_deadline'])
            if not inherited:require(read(root/'actuator_gate.json')['passed'],'Actual processed-target execution gate must pass')
            worker('collection',['-m','tools.tight_finish','collect','--root',str(root)],s['collection_deadline'])
            worker('formal_cpu_audit',['-m','tools.tight_audit','cpu','--root',str(root),'--data',str(root/'state_data')],s['deadline'])
            if not inherited:worker('formal_gpu_audit',['-m','tools.tight_audit','gpu','--root',str(root),'--data',str(root/'state_data')],s['deadline'])
            data=read(root/'state_data/data.json');ids=data['splits']['train'];anchors=sum(math.ceil(data['entries'][i]['length']/12) for i in ids)
            batch=len(ids)/math.ceil(len(ids)/64);t=s['training'];train=root/'training';train.mkdir()
            total=math.ceil(t['maximum_anchor_epochs']*anchors/batch);boundary=math.ceil(t['initial_anchor_epochs']*anchors/batch)
            write(train/'budget.json',dict(anchors=anchors,mean_batch=batch,total_updates=total,initial_updates=boundary,
                selection='validation only',sufficient_training_not_inferred_from_budget=True))
            runtime=dict(resource_policy=s['resource_policy'],resource_target_percent=s['resource_target_percent'],external_monitor_root=str(root),
                training_deadline=min(s['deadline']-3600,time.time()+s['training_hours']*3600),random_seed=t['random_seed'],
                validation_interval=t['validation_interval'],early_stop_minimum_update=math.ceil(t['minimum_anchor_epochs']*anchors/batch),
                deterministic_validation_loss=True)
            stage=0
            while True:
                stage+=1;runtime['stage_stop_update']=boundary;write(train/'session.json',runtime)
                worker('training_'+str(stage),['-m','visuotactile.anchor_continue','fresh-train','--job','state','--data',str(root/'state_data'),
                    '--root',str(train),'--target',str(total),'--output',str(train/f'launch_{stage}')],runtime['training_deadline'])
                end=read(train/'finished.json');rows=[json.loads(x) for x in (train/'validation.jsonl').read_text().splitlines()]
                recent=rows[-t['plateau_validations']:];gain=(recent[0]['metrics']['selection_mse']-min(r['metrics']['selection_mse'] for r in recent))/max(recent[0]['metrics']['selection_mse'],1e-12)
                improving=len(recent)<t['plateau_validations'] or gain>=t['plateau_relative_improvement']
                if end['updates']<boundary or not improving or boundary==total:break
                boundary=min(total,boundary+math.ceil(t['extension_anchor_epochs']*anchors/batch))
            write(train/'sufficiency.json',dict(updates=end['updates'],stop_reason=end['stop_reason'],recent_validation_gain=gain,
                still_improving=improving,sufficient=not improving and end['updates']>=runtime['early_stop_minimum_update'],
                final_test_used=False,scope='Finite preregistered exposure and validation trend; no claim of global convergence'))
            worker('evaluation',['-m','tools.tight_finish','evaluate','--root',str(root)],s['deadline'])
            write(root/'status.json',dict(state='complete',stage='evaluated',formal_updates=end['updates'],data=read(root/'DATA_READY.json'),evaluation=read(root/'evaluation_summary.json')))
        except BaseException as error:
            write(root/'finish_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            previous=read(root/'status.json');write(root/'status.json',dict(previous,state=('BUDGET_STOP' if isinstance(error,TimeoutError) else 'FAILED') if palm else 'blocked',reason=repr(error),exit_code=1,utc=time.time()))
            raise
        finally:
            if child is not None:kill_owned(child)
            (root/'STOP_MONITOR').touch()
            if monitor is not None:
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
            check_protected(root)
            status=read(root/'status.json');status['owned_workers_exited']=True;write(root/'status.json',status)
            if palm:
                if (root/'resources/resources.json').exists():write(root/'resource_summary.json',dict(read(root/'resources/resources.json'),
                    configured_limits=read(root/'configuration.json')['resources'],hard_gpu_quota_guaranteed=False))
                palm_report(root)


def vision_prepare(root):
    """Freeze the accepted 218 records, lawful inputs and one shared initializer."""
    import torch
    import h5py
    from visuotactile.position_control import contract,image_contract
    from visuotactile.state_history import PersistentImage
    from visuotactile.native_data import tactile_values
    from visuotactile.checkpointing import save
    cfg=read(CODE/'configs/tight_state.json');v=cfg['vision_pair'];base=CODE/v['source_root']
    require(not (root/'configuration.json').exists(),'Already prepared; no overwrite')
    source=read(base/'session.json');data=read(base/'state_data/data.json')
    require([len(data['splits'][k]) for k in ('train','validation','test')]==[166,30,22],'Frozen 218 Core identity required')
    require(data['manifest_sha256']==sha(base/'manifest.json'),'Accepted manifest identity changed')
    cfg['loss_weights']=source['loss_weights'];cfg['training']=source['training'];cfg['palm_training']=source['palm_training']
    cfg['palm_training']['previous_diagnostics']=str(base/'offline')
    cfg['resources']={k:v[k] for k in ('min_gpu_free_mib','min_ram_gib','min_disk_gib')}
    cfg['resources'].update(resource_policy='full_gpu_v1',resource_target_percent=None)
    root.mkdir(parents=True,exist_ok=True);out=root/'state_data';out.mkdir()
    write(root/'manifest.json',read(base/'manifest.json'))
    data=dict(data,visual_pair=True,manifest_sha256=sha(root/'manifest.json'),
        policy_inputs=['robot178','RGB_DINO_patches','deterministic_depth_grid'],tactile_separate=True,
        privileged_inputs=False,source_data_sha256=sha(base/'state_data/data.json'))
    write(out/'data.json',data)
    raw=torch.load(base/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    require(raw['data_sha256']==data['source_data_sha256'],'Accepted numeric identity changed')
    records={i:{k:r[k] for k in ('robot','command','time','joint_limits')} for i,r in raw['records'].items()}
    save(out/'numeric.pt',dict(contract=contract(24,True),data_sha256=sha(out/'data.json'),records=records));del raw
    stats=torch.load(base/'state_data/statistics.pt',map_location='cpu',weights_only=False)
    legal={k:stats[k] for k in ('robot_mean','robot_std','action_scale','action_offset')};save(out/'statistics.pt',legal)
    touch={};training=[]
    for identity in data['splits']['train']+data['splits']['validation']:
        entry=data['entries'][identity]
        with h5py.File(entry['path']) as f:
            g=f['episodes'][entry['episode']]['obs']
            values=[torch.from_numpy(g[k][:entry['length']]) for k in
                ('contact_substeps_w_N','contact_substeps_local_N','contact_time_s','time_s','contact_valid')]
            touch[identity]=tactile_values(*values)
            require(len(touch[identity])==len(records[identity]['time']),'Tactile/robot ticks differ')
            require(torch.equal(values[3],records[identity]['time']),'Tactile acquisition tick differs')
        if identity in data['splits']['train']:
            t=touch[identity].flatten(0,1);training.append(t[t[:,-1].bool(),:30])
    save(out/'tactile.pt',touch);all_touch=torch.cat(training)
    save(out/'tactile_statistics.pt',dict(tactile_mean=all_touch.mean(0),tactile_std=all_touch.std(0).clamp_min(1e-4)))
    torch.manual_seed(v['initialization_seed']);model=PersistentImage('I-HistoryDP',image_contract('I-HistoryDP',24,'G00'))
    for k,value in legal.items():getattr(model,k).copy_(value)
    save(root/'common_initial.pt',model.state_dict())
    dino=Path.home()/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
    session=dict(source,version='tight_visual_pair_v1',vision_pair=v,resources=cfg['resources'],
        resource_policy='full_gpu_v1',resource_target_percent=None,reserve_bytes=v['min_disk_gib']*1024**3,
        dino_sha256=sha(dino),dino_checkpoint=str(dino),cache_deadline=time.time()+1200,
        training=cfg['training'],palm_training=cfg['palm_training'],cwd=str(CODE),source_head=None)
    write(root/'configuration.json',cfg);write(root/'session.json',session)
    budget=read(base/'training/budget.json');budget={k:budget[k] for k in
        ('anchor_population','train_episodes','effective_batch_max','short_tail','mean_batch','total_updates',
         'target_anchor_epochs','early_stop_minimum_update','comparison_updates','selection','wall_hours','validation_interval')}
    write(root/'budget.json',budget)
    for group in v['groups']:
        dest=root/group;dest.mkdir();(dest/'training').mkdir();(dest/'state_data').symlink_to(out,target_is_directory=True)
        write(dest/'configuration.json',cfg);write(dest/'session.json',dict(session,resource_root=str(root),modality=group))
    write(root/'evaluation_plan.json',read(CODE/v['evaluation_plan'])[:v['evaluation_attempts_per_group']])
    write(root/'COMPLETE.json',dict(dataset_prepared=True,core=218,train=166,validation=30,test=22,
        source=str(base),source_manifest_sha256=sha(base/'manifest.json'),source_statistics_sha256=sha(base/'state_data/statistics.pt'),
        legal_statistics_identical=True,trainable_initial_sha256=sha(root/'common_initial.pt'),
        estimated_cache_gib=sum(math.ceil(data['entries'][i]['length']/12) for i in data['splits']['train']+data['splits']['validation'])*3*386*18*32*2/1024**3,
        stopped_datagen_service='fingereye-async-production.service',completed_datagen_preserved=True))
    write(root/'status.json',dict(state='PREPARED',stage='modality_acceptance_pending',utc=time.time(),formal_updates=0))


def vision_check(root):
    """Two real prefixes, disposable updates; no formal model is reused from gates."""
    import torch
    import h5py
    from visuotactile.native_data import cache_visual_pair,visual_records,Episode
    from visuotactile.state_history import PersistentImage
    from visuotactile.position_control import image_contract
    from visuotactile.precision import state_fp32
    from visuotactile.anchor_sampling import anchor_batch,mask_totals,weighted_parts
    from visuotactile.state_image_loss import GeometryObjective,PALM_LOSS_ID
    from visuotactile.checkpointing import save,training_state,restore_training
    from visuotactile.models.vendor.ema_model import EMAModel
    state_fp32();s=read(root/'session.json');data=read(root/'state_data/data.json')
    ids=s['palm_training']['diagnostic_cohorts']['train'][:2]
    cache_visual_pair(root,ids)
    with h5py.File(data['entries'][ids[0]]['path']) as f:metadata=json.loads(f.attrs['metadata_json'])
    from visuotactile.models.native_vision import NativeDinoSpatial
    live_dino=NativeDinoSpatial().cuda().eval();cached_episode=Episode(root/'state_data',ids[0],'I-HistoryDP')
    pixel_error=0.
    with h5py.File(data['entries'][ids[0]]['path']) as f:
        for tick in (0,108):
            rgb=torch.from_numpy(f['episodes'][data['entries'][ids[0]]['episode']]['frames/rgb_u8'][tick]).cuda()
            pixel_error=max(pixel_error,float((live_dino(rgb.float()/255)-cached_episode.frame(tick,'cuda')['patches']).abs().max()))
    require(pixel_error==0 and not any(p.requires_grad for p in live_dino.parameters()),'Live/cached frozen DINO differs')
    del live_dino,rgb;torch.cuda.empty_cache()
    obj=GeometryObjective(metadata,mode=PALM_LOSS_ID,commit=12);obj.weights=s['loss_weights'];obj.gain=1/500
    initial=torch.load(root/'common_initial.pt',map_location='cpu',weights_only=False);checks={}
    for modality in ('G00','G10'):
        torch.manual_seed(s['vision_pair']['initialization_seed'])
        model=PersistentImage('I-HistoryDP',image_contract('I-HistoryDP',24,modality))
        state=model.state_dict()
        for k,value in initial.items():state[k].copy_(value)
        if modality=='G10':
            for k,value in torch.load(root/'state_data/tactile_statistics.pt',weights_only=False).items():getattr(model,k).copy_(value)
        require(all(torch.equal(state[k],v) for k,v in initial.items()),'Shared initialization differs')
        model.cuda();records=visual_records(root/'state_data',ids,modality,'cuda');rec=records[ids[0]]
        rejected=[]
        for forbidden in ('actor','geometry')+(('tactile',) if modality=='G00' else ()):
            try:model.encode(dict(rec,**{forbidden:torch.zeros(1)}),0,1)
            except ValueError:rejected.append(forbidden)
        require(len(rejected)==(3 if modality=='G00' else 2),'Input allowlist failed')
        model.eval();episode=Episode(root/'state_data',ids[0],'I-HistoryDP');memory=None
        with torch.no_grad():
            for tick in range(109):
                memory=model.observe(rec['robot'][tick],rec['time'][tick],tick,episode.frame(tick,'cuda') if tick%12==0 else None,
                    memory,tactile=rec['tactile'][tick] if modality=='G10' else None)
            direct=model.condition(memory,rec['time'][108]);rebuilt=anchor_batch(model,records,[(ids[0],108)])[0]
            parity=float((direct-rebuilt).abs().max());require(parity<2e-5,'Full causal prefix differs')
            if modality=='G10':
                altered=copy.copy(memory);altered.tactile=[x.clone() for x in memory.tactile];altered.tactile[-1][:,:30]+=.01
                impact=float((model.condition(altered,rec['time'][108])-direct).abs().max())
            else:
                altered=copy.copy(memory);altered.frames=[(x+0.01,t,v) for x,t,v in memory.frames]
                impact=float((model.condition(altered,rec['time'][108])-direct).abs().max())
            require(impact>1e-7,'Modality does not affect condition')
        averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6,betas=(.95,.999),weight_decay=1e-6)
        generator=torch.Generator().manual_seed(7091);noise=torch.Generator(device='cuda').manual_seed(7092)
        selected=[(i,108) for i in ids];model.train();optimizer.zero_grad(set_to_none=True)
        c,y,m=anchor_batch(model,records,selected);_,parts,_=obj.forward(model,c,y,m,noise)
        loss,_=weighted_parts(parts,m,mask_totals({i:len(r['time']) for i,r in records.items()},selected,12),
            {k:v*obj.gain for k,v in obj.weights.items()},12)
        require(torch.isfinite(loss),'Nonfinite gate loss');loss.backward()
        modules=['robot_encoder','vision','native_input','native_gru','visual_gru','condition_projection','denoiser']
        if modality=='G10':modules+=['tactile_history','tactile_fusion']
        evidence={}
        for name in modules:
            params=list(getattr(model,name).parameters());grad=sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
            require(grad>0 and math.isfinite(grad),'Missing gradient: '+name)
            p=next(p for p in params if p.grad is not None and p.grad.abs().sum()>0);evidence[name]=(p,p.detach().clone(),grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ema.step(model)
        updated={name:dict(gradient_sum=g,changed=bool((p!=old).any())) for name,(p,old,g) in evidence.items()}
        require(all(x['changed'] for x in updated.values()),'Trainable module did not update')
        saved=training_state(model,averaged,optimizer,ema,generator,noise,group=model.group,updates=1,configuration=dict(modality=modality))
        path=root/('gate_'+modality+'.pt');save(path,saved)
        loaded=torch.load(path,map_location='cpu',weights_only=False);restore_training(loaded,model,averaged,optimizer,ema,generator,noise)
        require(all(torch.equal(v.cpu(),loaded['ema'][k]) for k,v in averaged.state_dict().items()),'EMA restore differs')
        with torch.no_grad():prediction=averaged.predict(c.detach(),torch.Generator(device='cuda').manual_seed(7091))
        require(prediction.shape==(2,24,29) and torch.isfinite(prediction).all(),'DDIM prediction failed')
        from visuotactile.native_controller import NativeController
        # Replay only legal sensor fields and past command acknowledgments through
        # the deployment controller. Predictions are bypassed for this history
        # check only; the full DDIM call above independently exercises sampling.
        current_tick=[0]
        controller=NativeController(loaded,metadata,dino=lambda rgb:cached_episode.frame(current_tick[0],'cuda')['patches'])
        controller.model.predict=lambda condition,generator:torch.zeros(1,24,29,device='cuda')
        with h5py.File(data['entries'][ids[0]]['path']) as f:
            episode_data=f['episodes'][data['entries'][ids[0]]['episode']];obs=episode_data['obs']
            for tick in range(109):
                current_tick[0]=tick
                packet={k:torch.from_numpy(obs[k][tick:tick+1]) for k in
                    ('episode_id','step_id','time_s','task_id','goal_valid','actuator_target_rad','camera_time_s','camera_valid')}
                if modality=='G10':packet.update({k:torch.from_numpy(obs[k][tick:tick+1]) for k in
                    ('contact_substeps_w_N','contact_substeps_local_N','contact_time_s','contact_valid')})
                if tick%12==0:
                    frame=episode_data['frames'];rgb=torch.from_numpy(frame['rgb_u8'][tick]).float()/255
                    packet['image']=torch.cat((rgb,torch.from_numpy(frame['depth_m'][tick])),1)[None]
                    packet['depth_valid']=torch.from_numpy(frame['depth_valid'][tick:tick+1])
                diagnostics={k:torch.from_numpy(episode_data['transition/diagnostics_before'][k][tick:tick+1]) for k in
                    ('joint_position_rad','joint_velocity_rad_s')}
                controller.act(packet,diagnostics)
                controller.command=rec['command'][tick:tick+1].cpu()
                controller.observe_execution({'position_applied_rad':controller.command})
        with torch.no_grad():expected=anchor_batch(averaged,records,[(ids[0],108)])[0]
        deployed_error=float((controller.last_condition-expected).abs().max())
        require(deployed_error<2e-5,'Deployment sensor/history condition differs')
        checks[modality]=dict(passed=True,prefix_max_abs=parity,condition_effect=impact,rejected=rejected,
            modules=updated,loss=float(loss),full_prediction_shape=list(prediction.shape),ema_step=ema.optimization_step,
            saved_sha256=sha(path),gate_updates_not_formal_training=True,deployment_prefix_max_abs=deployed_error,
            live_cached_dino_max_abs=pixel_error)
        del model,averaged,optimizer,ema,saved,loaded,evidence,state,records,rec,c,y,m,parts,loss,memory,altered,p,prediction
        del controller,expected;path.unlink();torch.cuda.empty_cache()
    write(root/'vision_acceptance.json',dict(passed=True,checks=checks,data_sha256=sha(root/'state_data/data.json'),
        statistics_sha256=sha(root/'state_data/statistics.pt'),dino_sha256=s['dino_sha256'],
        cache=read(root/'cache_acceptance_index.json'),utc=time.time()))


def vision_cache(root):
    from visuotactile.native_data import cache_visual_pair
    cache_visual_pair(root)


def vision_report(root):
    """Ordinary owner publishes measured results, including partial failures."""
    status=read(root/'status.json');session=read(root/'session.json');rows=[];text_rows=[]
    for group in ('G00','G10'):
        g=root/group;p=g/'training/progress.json';end=g/'training/finished.json';offline=g/'offline/stage_metrics.json'
        progress=read(p) if p.exists() else {};finished=read(end) if end.exists() else {}
        metrics=[x for x in read(offline) if x['split']=='validation' and x['motion']=='all' and x['phase']=='all'] if offline.exists() else []
        evaluation=read(g/'evaluation_summary.json') if (g/'evaluation_summary.json').exists() else None
        row=dict(group=group,training=finished,progress=progress,validation=metrics,evaluation=evaluation,
            last=str(g/'training/last.pt') if (g/'training/last.pt').exists() else None,
            best=str(g/'training/best.pt') if (g/'training/best.pt').exists() else None,
            failure=read(g/'failure.json') if (g/'failure.json').exists() else None)
        row['verified_videos']=[str(p.parent/'rollout.mp4') for p in sorted((g/'evaluation').glob('*/video_verified.json'))]
        arm=next((x for x in metrics if x['group']=='arm'),None);hand=next((x for x in metrics if x['group']=='hand'),None)
        if not arm or not hand:
            row['residual_readiness']='未测完整离线/闭环；尚不能判断残差基础能力'
        elif evaluation is None:
            row['residual_readiness']='完整离线已测；闭环尚无完整结果，不能判断残差基础能力'
        else:
            better_hold=arm['rmse_rad']<arm['hold_rmse_rad'] and hand['rmse_rad']<hand['hold_rmse_rad']
            degraded=(arm['rmse_rad']>.005645*1.5 or hand['rmse_rad']>.056618*1.5 or
                arm['ee_rmse_mm']>4.9823*1.5 or arm['palm_angle_mean_deg']>.7034*1.5)
            stage_progress=any(r.get('native_lift') or r.get('native_subgoals',0)>0 for r in evaluation['results'])
            row['comparison_flags']=dict(both_joint_groups_better_than_hold=better_hold,
                any_metric_over_1p5_State_reference=degraded,observed_lift_or_subgoal=stage_progress,
                scope='Descriptive development flags only; not a success gate or model-selection rule')
            row['residual_readiness']=('可作为下一阶段开发起点，尚无可靠性/等价性证明；是否完整成功见真实分母' if
                better_hold and not degraded and stage_progress else '当前证据不足以支持残差试验；查看相对保持目标、StateDP误差与原生阶段进展，停止不自动调参')
        evaluation_counts={k:value for k,value in evaluation.items() if k!='results'} if evaluation else None
        rows.append(row);text_rows.append(f"- {group}: updates={progress.get('update',0)}, anchor epochs={progress.get('anchor_equivalent_epochs',0):.3f}; stop={finished.get('stop_reason',row['failure'] or 'not finished')}; validation={json.dumps(metrics,ensure_ascii=False)}; development={json.dumps(evaluation_counts,ensure_ascii=False)}")
        text_rows.append('  '+row['residual_readiness'])
        text_rows.extend(f'  已完整解码视频：[{group} {Path(p).parent.name}]({p})' for p in row['verified_videos'])
    result=dict(status=status,groups=rows,state_reference=dict(arm_rad=.005645,hand_rad=.056618,palm_mm=4.9823,angle_deg=.7034,
        success=0,valid=10,scope='Same development reset specifications; hidden simulator state is not bitwise paired'),
        residual_training_started=False,readiness='Unproven until offline hold comparison and native stage progress are available; 0/10 alone is not acceptance')
    write(root/'result.json',result)
    block='\n'.join(['<!-- VISUAL_PAIR_START -->','## G00/G10 visual pair — '+status['state'],
        f"Run: `{root}`; source `{status.get('source_head','pending')}`; updated UTC epoch `{time.time()}`.",
        f"Training source: `{session.get('training_source_head',session.get('source_head','pending'))}`; recovery of `{session.get('recovery_of','none')}`. Original failed attempts remain separate; training is not repeated.",
        '218 Core (166/30/22); test22 unused. H24/commit12; frozen DINOv2-S/14 RGB + existing Depth; no Actor141/Geometry85 or residual.',
        'The two independent models share initial common parameters and train-only normalizers. G10 also learns its tactile encoder and fusion.',
        *text_rows,'Offline errors use expert causal histories; native rollouts are separate development evidence. Missing stages are unmeasured.',
        'No automatic G01/G11, recollection, retry or parameter search. Future G01 freezes selected G00; G11 freezes selected G10.',
        '<!-- VISUAL_PAIR_END -->'])
    path=CODE/'research_docs/STATE_DP_RESULTS.md';existing=path.read_text()
    import re
    if '<!-- VISUAL_PAIR_START -->' in existing:existing=re.sub(r'<!-- VISUAL_PAIR_START -->.*?<!-- VISUAL_PAIR_END -->',lambda _:block,existing,flags=re.S)
    else:existing+='\n\n'+block+'\n'
    temporary=path.with_suffix('.md.tmp');temporary.write_text(existing);temporary.replace(path)


def vision_evaluate(root):
    import torch
    from tools.tight_state import environment
    from visuotactile.state_benchmark import trial
    from visuotactile.state_resources import lease_for
    from visuotactile.video import verify_video
    from visuotactile.native_evaluation import exit_completed_kit_worker
    s=read(root/'session.json');v=s['vision_pair'];lease=lease_for(root,deadline=s['evaluation_deadline'])
    require(sha(s['dino_checkpoint'])==s['dino_sha256'],'Frozen DINO identity changed before deployment')
    app,env,player,lease=environment(root,True,lease=lease)
    available=[g for g in v['groups'] if (root/g/'offline/result.json').exists()]
    results={g:[] for g in available};plan=read(root/'evaluation_plan.json');saved={}
    for group in available:
        checkpoint=root/group/'training/best.pt';saved[group]=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(saved[group]['configuration']['statistics_sha256']==sha(root/'state_data/statistics.pt'),'Evaluation normalizer differs')
        require(saved[group]['configuration']['dino_sha256']==s['dino_sha256'],'Checkpoint DINO identity differs')
    for index,case in enumerate(plan):
        for group in available:
            checkpoint=root/group/'training/best.pt';out=root/group/'evaluation'/case['identity']
            record=index<v['videos_per_group']
            write(root/'evaluation_progress.json',dict(group=group,case=index,completed=sum(map(len,results.values())),utc=time.time()))
            result=trial(env,lease,out,case['spec'],{},player,str(checkpoint),saved=saved[group],position=True,video=record,
                expert_checkpoint=s['expert_checkpoint'],video_title=f'{group} visual DP | development seed {case["spec"]["seed"]}')
            if result.get('legal') and record:write(out/'video_verified.json',verify_video(out/'rollout.mp4',expected_frames=result['steps']))
            results[group].append(result)
            valid=[r for r in results[group] if r.get('legal') and r.get('software_complete')]
            write(root/group/'evaluation_summary.json',dict(attempts=len(results[group]),valid=len(valid),
                rejected=len(results[group])-len(valid),successes=sum(bool(r.get('success')) for r in valid),
                native_lifts=sum(bool(r.get('native_lift')) for r in valid),
                first_subgoal_reached=sum(r.get('native_subgoals',0)>0 for r in valid),
                alignments=sum(bool(r.get('native_alignment')) for r in valid),
                results=results[group],development_only=True,formal_test22_used=False,
                paired_specs_and_sampler_seed=True,bitwise_hidden_state_pairing_proven=False))
            require(result.get('software_complete') or not result.get('legal'),'Evaluation software failure; no retry')
    write(root/'evaluation_complete.json',dict(complete=True,groups=available,maximum_attempts_each=10,utc=time.time()))
    exit_completed_kit_worker(root)


@supervised_owner
def vision_run(root, evaluation_only=False):
    """Finite cache/pair/offline/single-Kit queue within the existing owner."""
    import fcntl
    lock=(root/'vision.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (root/'launch.json').exists(),'Existing launch must not reset its deadlines')
    if evaluation_only:
        recovery=read(root/'recovery.json')
        require(recovery['passed'] and recovery['original_completed_trials']==0,'Recovery gate failed')
        original=Path(recovery['original_root'])
        original_lock=(original/'vision.lock').open('a');fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,expected in recovery['protected_sha256'].items():
            require(sha(path)==expected,'Recovery artifact changed: '+path)
        require(read(original/'status.json')['owned_workers_exited'],'Original workers have not exited')
    else:
        require(read(root/'vision_acceptance.json')['passed'],'Modality acceptance failed')
        for group in ('G00','G10'):
            require(read(root/('smoke_'+group)/'check/result.json')['complete_restore'],'Trainer restore gate missing')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=CODE,text=True).strip()
    require(not dirty,'Commit actual runtime before launch: '+dirty)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    cfg=read(root/'configuration.json');s=read(root/'session.json');v=cfg['vision_pair'];budget=read(root/'budget.json')
    now=time.time()
    if evaluation_only:
        require(s['evaluation_deadline']==read(original/'session.json')['evaluation_deadline'], 'Recovery cannot reset evaluation deadline')
        require(now<s['evaluation_deadline']<=s['deadline'],'Original evaluation budget exhausted')
        require(sha(root/'evaluation_plan.json')==sha(original/'evaluation_plan.json'),'Recovery plan differs')
        s.update(recovery_started=now,training_source_head=s['source_head'],source_head=head,
            python=sys.executable,cwd=str(CODE),owned_worker_roots=[])
    else:
        s.update(started=now,deadline=now+v['maximum_hours']*3600,cache_deadline=now+v['cache_hours']*3600,
            training_deadline=now+(v['maximum_hours']-2)*3600,offline_deadline=None,evaluation_deadline=None,
            collection_deadline=None,development_deadline=None,
            source_head=head,python=sys.executable,cwd=str(CODE),owned_worker_roots=[str(root/g/'training') for g in v['groups']])
    write(root/'session.json',s);write(root/'launch.json',dict(pid=os.getpid(),source_head=head,cwd=str(CODE),python=sys.executable,
        started=now,configuration_sha256=sha(root/'configuration.json'),data_sha256=sha(root/'state_data/data.json')))
    source_files=subprocess.check_output(['git','ls-files','visuotactile/*.py','tools/*.py','configs/tight_state.json'],cwd=CODE,text=True).splitlines()
    frozen={p:sha(CODE/p) for p in source_files};write(root/'source_freeze.json',frozen)
    write(root/'concurrency.json',dict(mode='parallel',pause_groups=[]))
    (root/'STOP_MONITOR').unlink(missing_ok=True);children=[];monitor=None;failures=[]
    def status(stage,**kw):
        write(root/'status.json',dict(state='RUNNING',stage=stage,pid=os.getpid(),source_head=head,utc=time.time(),**kw))
    def spawn(args,name):
        require(all(sha(CODE/p)==value for p,value in frozen.items()),'Runtime source changed after launch; stop dependent stages')
        log=(root/(name+'.log')).open('a')
        env=dict(os.environ)
        if name=='G10_training':env['RESEARCH_CPUS']='20,21'
        child=subprocess.Popen(RUN+args,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(child);return child
    def wait(child,deadline,stage):
        began=time.time()
        while child.poll() is None:
            if time.time()>=deadline:kill_owned(child);raise TimeoutError(stage+' wall budget exhausted')
            cache=read(root/'cache_progress.json') if (root/'cache_progress.json').exists() else None
            eta=dict(stage=stage,utc=time.time(),stage_hard_stop_utc=deadline,queue_hard_stop_utc=s['deadline'])
            if stage=='shared_dino_cache':
                eta.update(cache_remaining_seconds=cache.get('eta_seconds') if cache else None,
                    training_remaining_seconds=None,reason='Training throughput not measured yet; training <=9h, offline/evaluation <=2h')
            elif stage.endswith('_offline'):
                path=root/stage.split('_')[0]/'offline/progress.json';p=read(path) if path.exists() else {}
                count=p.get('offline_episodes',0);eta['offline_remaining_seconds']=(38-count)*(time.time()-began)/count if count else None
            elif stage=='matched_development':
                path=root/'evaluation_progress.json';p=read(path) if path.exists() else {};count=p.get('completed',0)
                eta['evaluation_remaining_seconds']=(20-count)*(time.time()-began)/count if count else None
            write(root/'eta.json',eta)
            status(stage,worker_pid=child.pid,cache=cache,eta=eta,
                evaluation=read(root/'evaluation_progress.json') if (root/'evaluation_progress.json').exists() else None)
            time.sleep(5)
        if child.returncode:raise RuntimeError(stage+' exit '+str(child.returncode))
    try:
        monitor=spawn(['-m','visuotactile.state_resources','--root',str(root)],'monitor')
        # Bounded telemetry admission; an ordinary process, never a model watcher.
        for _ in range(10):
            if (root/'shared_resources.json').exists() and time.time()-read(root/'shared_resources.json')['utc_epoch']<4:break
            time.sleep(1)
        if evaluation_only:
            wait(spawn(['-m','tools.tight_finish','vision_evaluate','--root',str(root)],'evaluation'),s['evaluation_deadline'],'matched_development')
            require(read(root/'evaluation_complete.json')['complete'],'Evaluation completion receipt missing')
            write(root/'status.json',dict(state='COMPLETE',stage='finished',source_head=head,
                training_source_head=s['training_source_head'],utc=time.time(),exit_code=0,
                reason='Authorized evaluation continuation finished; prior training stopped at fixed wall budget',
                recovery_of=str(original),training_repeated=False))
            return
        wait(spawn(['-m','tools.tight_finish','vision_cache','--root',str(root)],'cache'),s['cache_deadline'],'shared_dino_cache')
        start=time.time();train_deadline=min(start+v['training_hours']*3600,s['deadline']-7200)
        s['training_deadline']=train_deadline;write(root/'session.json',s)
        workers={}
        for group in v['groups']:
            dest=root/group;gs=dict(s,resource_root=str(root),modality=group);write(dest/'session.json',gs)
            runtime=dict(resource_policy='full_gpu_v1',external_monitor_root=str(root),training_deadline=train_deadline,
                modality=group,microbatch=v['microbatch'],random_seed=v['initialization_seed'],
                validation_interval=budget['validation_interval'],early_stop_minimum_update=budget['early_stop_minimum_update'],
                comparison_updates=budget['comparison_updates'],deterministic_validation_loss=True)
            write(dest/'training/session.json',runtime);write(dest/'training/budget.json',budget)
            workers[group]=spawn(['-m','visuotactile.anchor_continue','fresh-train','--job','state','--data',str(root/'state_data'),
                '--root',str(dest/'training'),'--target',str(budget['total_updates']),'--output',str(dest/'training/launch')],group+'_training')
        measured='parallel';baseline=None;serial=None;active='G00';switch_at=None;recorded=set()
        while any(p.poll() is None for p in workers.values()):
            now=time.time();progress={g:read(root/g/'training/progress.json') if (root/g/'training/progress.json').exists() else {} for g in workers}
            live=[g for g,p in workers.items() if p.poll() is None]
            for g,p in workers.items():
                if p.poll() is not None and g not in recorded:
                    recorded.add(g)
                    if p.returncode:
                        failure=dict(stage='training',exit_code=p.returncode,reason='See '+g+'_training.log',utc=now)
                        write(root/g/'failure.json',failure);failures.append(dict(group=g,**failure))
            if now>=train_deadline:
                for g in live:kill_owned(workers[g]);write(root/g/'failure.json',dict(stage='training',reason='fixed_wall_budget',exit_code=workers[g].returncode))
                failures.append(dict(stage='training',reason='fixed_wall_budget'));break
            counts={g:progress[g].get('update',0) for g in workers};anchors={g:progress[g].get('sampled_anchors',0) for g in workers}
            if measured=='parallel' and len(live)==2 and min(counts.values())>=v['parallel_measure_updates']:
                # Measure a second window after initialization, so startup is not counted as throughput.
                if baseline is None:baseline=(now,counts.copy(),anchors.copy())
                elif min(counts[g]-baseline[1][g] for g in live)>=v['parallel_measure_updates']:
                    parallel_rate=sum(anchors[g]-baseline[2][g] for g in live)/(now-baseline[0])
                    serial=(now,counts['G00'],anchors['G00']);measured='serial_probe'
                    write(root/'concurrency.json',dict(mode=measured,pause_groups=['G10'],parallel_anchors_per_second=parallel_rate))
            elif measured=='serial_probe' and len(live)==2 and counts['G00']-serial[1]>=v['serial_compare_updates']:
                serial_rate=(anchors['G00']-serial[2])/(now-serial[0])
                measured='parallel_selected' if parallel_rate>=serial_rate*v['parallel_efficiency_floor'] else 'serial_selected'
                active=min(live,key=lambda g:counts[g]);switch_at=counts[active]+v['serial_switch_updates']
                write(root/'throughput_comparison.json',dict(parallel_anchors_per_second=parallel_rate,serial_anchors_per_second=serial_rate,
                    selected=measured,utc=now,includes_normal_io_and_prefix_work=True))
                write(root/'concurrency.json',dict(mode=measured,pause_groups=[] if measured=='parallel_selected' else [g for g in live if g!=active]))
            elif measured=='serial_selected' and len(live)==2 and counts[active]>=switch_at:
                active=min(live,key=lambda g:counts[g]);switch_at=counts[active]+v['serial_switch_updates']
                write(root/'concurrency.json',dict(mode=measured,pause_groups=[g for g in live if g!=active]))
            if len(live)==1:write(root/'concurrency.json',dict(mode='single_remaining',pause_groups=[]))
            estimates={g:dict(remaining_seconds_unbounded=progress[g].get('remaining_seconds_estimate'),
                budget_remaining_seconds=max(0,train_deadline-now),includes_observed_wait=True) for g in workers}
            predicted=[x['remaining_seconds_unbounded'] for x in estimates.values() if x['remaining_seconds_unbounded'] is not None]
            write(root/'eta.json',dict(stage='training',utc=now,groups=estimates,cache_remaining_seconds=0,
                concurrency=measured,training_completion_utc_estimate=now+min(max(predicted),max(0,train_deadline-now)) if predicted else None,
                post_training_budget_seconds=7200,queue_hard_stop_utc=s['deadline'],
                basis='Per-worker observed wall time includes current/previous concurrency and actual waits; not a single-model speed extrapolation'))
            status('training',groups=progress,worker_pids={g:p.pid for g,p in workers.items()},concurrency=measured,eta=estimates,
                queue_latest_finish_utc=s['deadline'],evaluation_budget_seconds=3600,offline_budget_seconds=3600)
            time.sleep(5)
        for group,p in workers.items():
            if p.poll() is not None and p.returncode and group not in recorded:
                failure=dict(group=group,stage='training',exit_code=p.returncode,reason='See '+group+'_training.log')
                failures.append(failure);write(root/group/'failure.json',failure)
        write(root/'concurrency.json',dict(mode='post_training',pause_groups=[]))
        offline_deadline=min(time.time()+v['offline_hours']*3600,s['deadline']-v['evaluation_hours']*3600)
        s['offline_deadline']=offline_deadline;write(root/'session.json',s)
        for group,p in workers.items():
            dest=root/group
            if not (dest/'training/finished.json').exists():continue
            gs=read(dest/'session.json');gs['offline_deadline']=offline_deadline;write(dest/'session.json',gs)
            try:wait(spawn(['-m','tools.tight_diagnostics','final','--root',str(dest)],group+'_offline'),offline_deadline,group+'_offline')
            except (RuntimeError,TimeoutError) as error:
                failures.append(dict(group=group,stage='offline',reason=repr(error)));write(dest/'failure.json',failures[-1])
        if any((root/g/'offline/result.json').exists() for g in workers):
            s['evaluation_deadline']=min(time.time()+v['evaluation_hours']*3600,s['deadline']);write(root/'session.json',s)
            try:wait(spawn(['-m','tools.tight_finish','vision_evaluate','--root',str(root)],'evaluation'),s['evaluation_deadline'],'matched_development')
            except (RuntimeError,TimeoutError) as error:failures.append(dict(stage='evaluation',reason=repr(error)))
        budget_stop=any((root/g/'training/finished.json').exists() and read(root/g/'training/finished.json')['stop_reason']=='fixed_wall_budget' for g in workers)
        write(root/'status.json',dict(state='FAILED' if failures else 'BUDGET_STOP' if budget_stop else 'COMPLETE',stage='finished',source_head=head,
            utc=time.time(),exit_code=1 if failures else 0,failures=failures,reason='Finite queue ended; no automatic next model'))
    except BaseException as error:
        write(root/'owner_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
        write(root/'status.json',dict(state='BUDGET_STOP' if isinstance(error,TimeoutError) else 'FAILED',stage='stopped',
            source_head=head,utc=time.time(),exit_code=1,reason=repr(error)))
        raise
    finally:
        for p in children:
            if p is not monitor:kill_owned(p)
        (root/'STOP_MONITOR').touch()
        if monitor is not None:
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(monitor)
        result=read(root/'status.json');result.update(owned_workers_exited=True,utc=time.time());write(root/'status.json',result)
        vision_report(root)
    if read(root/'status.json')['state']=='FAILED':raise RuntimeError('Finite queue contains failed stages; see status.json')


def vision_resume(root):
    """User-authorized evaluation-only continuation; original deadlines stay fixed."""
    return vision_run(root,evaluation_only=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','refine','collect','freeze_extension','evaluate','actuator_worker','validate_speed','baseline_evaluate','hand_cap_pilot','vision_prepare','vision_check','vision_cache','vision_run','vision_resume','vision_evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--refinement',action='store_true');p.add_argument('--speed-validation',action='store_true');p.add_argument('--baseline',action='store_true');p.add_argument('--mechanism',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='run':
        try:pipeline(root,a.refinement,a.speed_validation,a.baseline,a.mechanism)
        except BaseException as error:
            if (root/'configuration.json').exists() and read(root/'configuration.json').get('palm_training'):
                status=read(root/'status.json')
                if status.get('state') not in ('FAILED','BUDGET_STOP'):
                    write(root/'status.json',dict(status,state='FAILED',reason=repr(error),exit_code=1,utc=time.time()))
                    write(root/'finish_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
                    palm_report(root)
            raise
    else:
        code=0
        try:globals()[a.mode](root)
        except BaseException:
            write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc()));code=1
        sys.stdout.flush();sys.stderr.flush();os._exit(code)
