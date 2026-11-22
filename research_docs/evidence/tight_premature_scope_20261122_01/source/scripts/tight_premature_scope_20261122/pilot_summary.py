"""Summarize all diagnostic opportunities; expert rescue never becomes autonomy."""
import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import shutil
import time
import zipfile

import h5py
import numpy as np

from scripts.tight_premature_scope_20261122.pilot_prepare import PROJECT, ROOT, EVIDENCE
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def first(mask):
    ids = np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def export_numeric(source, destination):
    """Raw values, one row per transition; no fitted or reconstructed labels."""
    selected = {}
    with h5py.File(source) as f:
        episode=f['episode']
        def select(name, item):
            if not isinstance(item,h5py.Dataset):return
            if name.startswith('transition/torch_'):return
            if name.startswith('obs/') and name.split('/')[-1] in (
                    'prior_plan','prior_plan_valid','camera_pose_w','camera_parent_pose_w',
                    'full_proprio_diagnostic','finger_rotation_base'):
                return
            selected[name]=item[:]
        episode.visititems(select)
    n=len(selected['transition/step_id']);columns=['tick'];schema={}
    arrays=[]
    for name,value in selected.items():
        assert len(value)==n
        names=[name] if value.ndim==1 else [name+'['+','.join(map(str,idx))+']' for idx in np.ndindex(value.shape[1:])]
        schema[name]=dict(dtype=str(value.dtype),shape=list(value.shape),columns=names)
        columns.extend(names); arrays.append(value.reshape(n,-1))
    with gzip.open(destination,'wt',newline='',compresslevel=6) as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        for tick in range(n):
            row=[tick]
            for a in arrays:row.extend(a[tick].tolist())
            writer.writerow(row)
    atomic_json(destination.with_suffix('.schema.json'),dict(source=str(source),source_sha256=sha256(source),
        gzip_sha256=sha256(destination),rows=n,columns=len(columns),datasets=schema,
        scope='Raw numeric HDF5 values: no RGB/depth/features/weights; Python numeric representations preserve stored float32 values',
        omitted='torch RNG byte states, redundant observation prior_plan/camera pose/full_proprio/finger_rotation; inference RNG hashes separately retained'))


def main():
    registration = read(ROOT/'registration.json')
    assert (ROOT/'project_complete.json').exists() and (ROOT/'project_exit.json').exists()
    owner_started=read(ROOT/'project_job_started.json')
    project_complete=read(ROOT/'project_complete.json')
    project_exit=read(ROOT/'project_exit.json')
    assert project_exit['children_reaped']
    lifecycle={'owner':owner_started['pid'],'monitor':read(ROOT/'monitor.json')['pid']}
    lifecycle.update({w.name:read(w/'run_started.json')['pid'] for w in sorted(ROOT.glob('wave*'))})
    process_checks={}
    for name,pid in lifecycle.items():
        proc=Path(f'/proc/{pid}/cmdline')
        command=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc.exists() else None
        matches=command is not None and ('scripts.tight_premature_scope_20261122.pilot_entry' in command or
            ('visuotactile.state_resources' in command and str(ROOT) in command))
        process_checks[name]=dict(pid=pid,current_command=command,matching_owned_process_still_alive=matches)
    assert not any(v['matching_owned_process_still_alive'] for v in process_checks.values())
    timing=dict(registered_utc=registration['registered_utc'],owner_started_utc=owner_started['utc'],
        project_complete_utc=project_complete['utc'],owner_exit_receipt_utc=project_exit['utc'],
        owner_elapsed_seconds=project_exit['utc']-owner_started['utc'],frozen_deadline_utc=registration['deadline'])
    timing['local_times']={k:datetime.fromtimestamp(v,timezone(timedelta(hours=8))).isoformat()
        for k,v in timing.items() if k.endswith('_utc')}
    atomic_json(EVIDENCE/'lifecycle.json',dict(process_checks=process_checks,children_reaped=True,timing=timing,
        scope='Recorded owner, four Kit workers and monitor PIDs; original telemetry ownership labels unchanged'))
    rows, sources, pairs = [], {}, []
    for wave in sorted(ROOT.glob('wave*')):
        plan = read(wave/'plan.json')
        assert sha256(wave/'plan.json') == registration['wave_plan_sha256'][str(wave)]
        assert read(wave/'result.json')['software_complete']
        assert read(wave/'paired_initial_check.json')['passed']
        assert read(wave/'correction_entries.json') == {}
        for case in plan['cases']:
            folder = wave/case['name']; result = read(folder/'result.json')
            assert result['software_complete'] and result['legal']
            assert not result['accepted_for_training'] and result['training_pool_admission'] == 'blocked'
            audit = read(folder/'diagnostic_audit.json')
            assert audit['provenance']['passed']
            if case['diagnostic_arm']=='alignment_rescue':
                assert audit['trigger']['passed']
            else:
                assert audit['trigger'] is None
            assert result['complete_student_policy']==registration['pairs'][case['role']]
            dagger = result['dagger']; switch = dagger['recovery_coverage']['switch_tick']
            receipt = dagger.get('trigger_receipt')
            trigger = receipt.get('at_trigger') if receipt else None
            control_screen=None
            with h5py.File(folder/'numeric.h5') as f:
                t = f['episode/transition']
                key = t['native_keypoints_max_dist'][:]
                before = t['diagnostics_before/successes'][:]
                after = t['diagnostics_after/successes'][:]
                obj = t['diagnostics_after/object_pose_w'][:]
                goal = t['diagnostics_before/goal_pose_w'][:]
                angle = np.degrees(2*np.arccos(np.clip(np.abs((obj[:,3:]*goal[:,3:]).sum(-1)),0,1)))
                force = t['object_contact_force_w'][:]
                fingers = (np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
                if case['diagnostic_arm']=='autonomous':
                    from visuotactile.adaptive_takeover import AdaptiveTrigger,profile
                    replay=AdaptiveTrigger(profile('alignment'),10**9)
                    palm=t['diagnostics_after/palm_body_pose_w'][:]
                    goal_after=t['diagnostics_after/goal_pose_w'][:]
                    for tick in range(len(key)):
                        schedule=replay.observe(tick,obj[tick],palm[tick],force[tick],
                            keypoint_max_distance_m=key[tick],goal_index_before=before[tick],goal_index_after=after[tick],
                            goal_pose_before=goal[tick],goal_pose_after=goal_after[tick])
                        if schedule is not None:break
                    control_screen=replay.receipt()
                if switch is not None:
                    at = int(switch)-1
                    geometry = dict(confirmation_transition_tick=at, expert_first_action_tick=int(switch),
                        keypoint_mm=float(key[at]*1000), orientation_deg=float(angle[at]),
                        object_goal_root_mm=float(np.linalg.norm(obj[at,:3]-goal[at,:3])*1000),
                        minimum_contact_fingers_both_substeps=int(fingers[at]),
                        native_goal_before=int(before[at]), native_goal_after=int(after[at]),
                        gate15mm_previously_entered=bool((key[:at+1]<=.015).any()),
                        native_subgoal_previously_reached=bool((after[:at+1]>=1).any()))
                else:
                    geometry = None
            row = dict(wave=wave.name, role=case['role'], diagnostic_arm=case['diagnostic_arm'],
                seed=case['spec']['seed'], family=case['spec']['family'], case=case['name'],
                legal=result['legal'], software_complete=result['software_complete'], steps=result['steps'],
                native_success=bool(result['success']), native_subgoals=result['native_subgoals'],
                actual_expert_switch_tick=switch, expert_ticks=dagger['expert_ticks'],
                expert_valid_ticks=dagger['expert_valid_ticks'], expert_rescue_success=bool(dagger['expert_takeover_success']),
                autonomous_success=bool(dagger['student_autonomous_success']),
                trigger_last=receipt.get('last') if receipt else None,
                trigger_geometry=geometry, trigger_at=trigger,
                autonomous_replayed_alignment_screen=control_screen,
                autonomous_replayed_alignment_trigger_tick=control_screen['trigger_completed_tick'] if control_screen else None,
                first15mm_tick=first(key<=.015), first_native_subgoal_tick=first(after>=1),
                minimum_keypoint_mm=float(key.min()*1000), accepted_for_training=False,
                result_path=str(folder/'result.json'), numeric_path=str(folder/'numeric.h5'),
                reasons=result.get('reasons'))
            if case['diagnostic_arm'] == 'autonomous':
                assert switch is None and dagger['expert_ticks'] == dagger['expert_valid_ticks'] == 0
            rows.append(row)
            for name in ('result.json','diagnostic_audit.json','sampler_rng_start.json','sampler_rng_end.json','numeric.h5','trace.npz'):
                sources[str(folder/name)] = sha256(folder/name)
            target=EVIDENCE/wave.name/case['name'];target.mkdir(parents=True,exist_ok=True)
            for name in ('result.json','diagnostic_audit.json','sampler_rng_start.json','sampler_rng_end.json',
                         'metadata.json','initial_observable.json','initial_collision.json','started.json','restoration.json'):
                shutil.copyfile(folder/name,target/name)
            export_numeric(folder/'numeric.h5', target/'numeric_ticks.csv.gz')
            noise={}
            for stage in ('start','end'):
                source=folder/f'sampler_rng_{stage}.pt'
                with zipfile.ZipFile(source) as archive:
                    names=[n for n in archive.namelist() if n.endswith('/data/0')]
                    assert len(names)==1
                    values=list(archive.read(names[0]))
                assert len(values)==16
                noise[stage]=dict(generator_state_uint8=values,source_sha256=sha256(source),
                    recorded=read(folder/f'sampler_rng_{stage}.json'))
            atomic_json(target/'inference_noise_states.json',noise)
        for role in 'AB':
            selected={r['diagnostic_arm']:r for r in rows if r['wave']==wave.name and r['role']==role}
            a,b=selected['autonomous'],selected['alignment_rescue']
            n=min(a['steps'],b['steps'],b['actual_expert_switch_tick'] or 10**9)
            with h5py.File(a['numeric_path']) as x,h5py.File(b['numeric_path']) as y:
                ta,tb=x['episode/transition'],y['episode/transition']
                command_error=np.abs(ta['target_after_rad'][:n]-tb['target_after_rad'][:n]).max(-1)
                pa=ta['diagnostics_after/object_pose_w'][:n,:3]-ta['diagnostics_after/base_pose_w'][:n,:3]
                pb=tb['diagnostics_after/object_pose_w'][:n,:3]-tb['diagnostics_after/base_pose_w'][:n,:3]
                object_error=np.linalg.norm(pa-pb,axis=-1)
            pairs.append(dict(wave=wave.name,role=role,seed=a['seed'],autonomous_success=a['native_success'],
                rescue_arm_native_success=b['native_success'],triggered=b['actual_expert_switch_tick'] is not None,
                expert_success=b['expert_rescue_success'],pre_switch_compared_ticks=n,
                first_command_difference_over_1e6rad_tick=first(command_error>1e-6),
                first_object_difference_over1mm_tick=first(object_error>.001),
                maximum_command_difference_before_switch_rad=float(command_error.max()) if n else None,
                maximum_object_difference_before_switch_mm=float(object_error.max()*1000) if n else None,
                exact_counterfactual_claim=False))
        target=EVIDENCE/wave.name;target.mkdir(parents=True,exist_ok=True)
        for name in ('plan.json','result.json','paired_initial_check.json','paired_reset_installed.json','wave_profiles.json','correction_entries.json'):
            shutil.copyfile(wave/name,target/name)
    assert len(rows)==16
    checks={p:dict(expected=d,actual=sha256(p),matched=sha256(p)==d) for p,d in registration['protected_sha256'].items()}
    assert all(v['matched'] for v in checks.values())
    frozen_source=read(ROOT/'wave00/plan.json')['source_sha256']
    source_checks={p:dict(expected=d,actual=sha256(PROJECT/p),matched=sha256(PROJECT/p)==d)
        for p,d in frozen_source.items()}
    assert all(v['matched'] for v in source_checks.values())
    wrapper_checks={p:dict(expected=d,actual=sha256(p),matched=sha256(p)==d)
        for p,d in registration['wrapper_sha256'].items()}
    assert all(v['matched'] for v in wrapper_checks.values())
    assert sha256(registration['job'])==registration['job_sha256']
    atomic_json(EVIDENCE/'protected_hashes_before_after.json',checks)
    atomic_json(EVIDENCE/'source_hashes_before_after.json',source_checks)
    atomic_json(EVIDENCE/'wrapper_hashes_before_after.json',wrapper_checks)
    counts={}
    for role in 'AB':
        controls=[r for r in rows if r['role']==role and r['diagnostic_arm']=='autonomous']
        rescue=[r for r in rows if r['role']==role and r['diagnostic_arm']=='alignment_rescue']
        triggered=[r for r in rescue if r['actual_expert_switch_tick'] is not None]
        counts[role]=dict(autonomous_opportunities=len(controls),autonomous_native_successes=sum(r['native_success'] for r in controls),
            rescue_opportunities=len(rescue),triggered=len(triggered),expert_successes=sum(r['expert_rescue_success'] for r in triggered),
            triggered_previously_entered15mm=sum(r['trigger_geometry']['gate15mm_previously_entered'] for r in triggered),
            triggered_never_entered15mm=sum(not r['trigger_geometry']['gate15mm_previously_entered'] for r in triggered),
            no_trigger=len(rescue)-len(triggered),no_trigger_autonomous_successes=sum(r['autonomous_success'] for r in rescue if r not in triggered),
            autonomous_replay_screen_hits=sum(r['autonomous_replayed_alignment_trigger_tick'] is not None for r in controls),
            autonomous_replay_screen_hits_among_successes=sum(r['autonomous_replayed_alignment_trigger_tick'] is not None and r['native_success'] for r in controls))
    atomic_json(EVIDENCE/'summary.json',dict(rows=rows,paired= pairs,counts=counts,protected_before_after=checks,timing=timing,
        frozen_source_files=len(source_checks),frozen_source_hashes_matched=True,frozen_wrapper_hashes_matched=True,
        opportunities=16,software_incomplete=0,illegal=0,optimizer_updates=0,training_pool_admitted_trajectories=0,
        diagnostic_expert_success_is_autonomous=False,utc=time.time(),registration_sha256=sha256(ROOT/'registration.json')))
    atomic_json(EVIDENCE/'source_hashes.json',sources)
    resource_rows=[json.loads(line) for line in (ROOT/'resources/resources.jsonl').read_text().splitlines() if line.strip()]
    resource_times=[r['utc_epoch'] for r in resource_rows]
    resource_summary=dict(samples=len(resource_rows),
        minimum_available_ram_gib=min(r['ram_available_bytes'] for r in resource_rows)/2**30,
        minimum_gpu_free_mib=min(r['gpu_free_mib'] for r in resource_rows),
        maximum_gpu_used_mib=max(r['gpu_used_mib'] for r in resource_rows),
        maximum_sample_interval_s=max(b-a for a,b in zip(resource_times,resource_times[1:])),
        scope='Sampled resource observations, not a continuous lower-bound guarantee',
        gpu_metric_scope='Whole device, including unrelated external GPU processes; not pilot-only allocation',
        ownership_label_limitation='Monitor known-entry recognition did not cover custom pilot_entry workers; recorded owner/worker/monitor PIDs independently determine lifecycle ownership')
    atomic_json(EVIDENCE/'resource_summary.json',resource_summary)
    keys=['role','seed','diagnostic_arm','native_success','autonomous_success','native_subgoals','steps','actual_expert_switch_tick',
          'expert_ticks','expert_valid_ticks','expert_rescue_success','first15mm_tick','first_native_subgoal_tick','minimum_keypoint_mm',
          'autonomous_replayed_alignment_trigger_tick',
          'legal','software_complete','accepted_for_training','result_path']
    geometry=['confirmation_transition_tick','expert_first_action_tick','keypoint_mm','orientation_deg','object_goal_root_mm',
        'minimum_contact_fingers_both_substeps','native_goal_before','native_goal_after','gate15mm_previously_entered','native_subgoal_previously_reached']
    with (EVIDENCE/'opportunities.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys+geometry);writer.writeheader()
        for row in rows:writer.writerow(dict({k:row[k] for k in keys},**(row['trigger_geometry'] or {})))
    for name in ('project_complete.json','project_exit.json','project_job_started.json'):
        shutil.copyfile(ROOT/name,EVIDENCE/name)
    shutil.copyfile(registration['job'],EVIDENCE/'evaluate_job.json')
    for source in sorted(ROOT.rglob('*')):
        if source.is_file() and (source.suffix in ('.log','.jsonl') or 'failure' in source.name or
                source.name in ('shared_resources.json','monitor.json','monitor_exit.json','resources.json','kit_started.json',
                    'execution_backend_observed.json','execution_backend_requested.json','eval_startup.json','precision.json',
                    'run_started.json','initialization.json','worker_exit.json','project_status.json')):
            destination=EVIDENCE/'logs'/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    text=['# 冻结入口模型：训练 family 对准停滞接管小试','',
        '| 组 | 同场自主完整插入 | 接管机会 | 实际触发 | 触发后专家救回 | 未触发 |',
        '|---|---:|---:|---:|---:|---:|']
    for role,c in counts.items():
        text.append(f"| {role} | {c['autonomous_native_successes']}/{c['autonomous_opportunities']} | {c['rescue_opportunities']} | {c['triggered']} | {c['expert_successes']}/{c['triggered']} | {c['no_trigger']} |")
    text.extend(['',f"Owner实际开始 {timing['local_times']['owner_started_utc']}，关闭回执 {timing['local_times']['owner_exit_receipt_utc']}，历时 {timing['owner_elapsed_seconds']:.2f} 秒。"])
    text.extend(['','自主对照离线复算同一alignment屏幕：'+ '；'.join(
        f"{role}命中{c['autonomous_replay_screen_hits']}/{c['autonomous_opportunities']}，其中最终自主成功{c['autonomous_replay_screen_hits_among_successes']}例"
        for role,c in counts.items())+'。屏幕命中不等于已证明有害错误。'])
    text.extend(['','本表专家救回不是学生自主成功。16机会均来自预先指定的4个原训练family；0训练更新、0入训练池。',
        '触发规则直接复用既有adaptive alignment：连续12tick持物、距目标root≤90mm、keypoint>15mm、36tick关键点改善不足1.5mm、连续2tick确认，stage0后下一动作接管，无定时fallback。',
        'registration 的 trigger_before_native_15mm_gate=True 只表示触发当下 keypoint>15mm 且 goal0，不保证此前从未进入15mm门；请看逐例 gate15mm_previously_entered。',
        '该触发筛选当前 keypoint>15mm 的转正/对准停滞，不将短暂离门普遍视为错误。各次触发朝向、关键点、接触、是否曾入门及后续原生成功见opportunities.csv。',
        '自主对照与接管臂均运行同一完整学生、专家shadow和记录路径；自主臂从未获得专家权限，逐tick命令链审计通过。两臂可观测初态配对通过，但不主张隐藏物理或后续随机流精确配对；pre-switch数值差异见summary.json。',
        '资源遥测的known-entry识别未覆盖自定义pilot_entry，原始日志可将自有worker记为owned=false；生命周期按登记的owner/worker/monitor PID独立核验，资源采样见resource_summary.json。',
        '小试family依据既有训练alignment覆盖选取，未用终测挑选，也不是泛化成功率估计。所有权重、数据manifest、统计、numeric及4条源原轨SHA前后不变。',''])
    (EVIDENCE/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(dict(evidence=str(EVIDENCE),counts=counts,protected=len(checks),software_incomplete=0)))


if __name__ == '__main__':
    main()
