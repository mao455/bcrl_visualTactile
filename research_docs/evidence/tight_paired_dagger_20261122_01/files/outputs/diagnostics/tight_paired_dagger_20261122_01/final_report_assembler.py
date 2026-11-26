"""Read-only, CPU-only final report gate. Writes only a new AUDIT attempt directory.
No production imports until all closed final evidence exists. No model forwards.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='2'
import argparse,csv,datetime,hashlib,json,math,time
from pathlib import Path
PROJECT=Path('/home/CNS2025331827/docs/bc+rl_RSS')
ROOT=PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
AUDIT=PROJECT/'outputs/diagnostics/tight_paired_dagger_20261122_01'
ROLES=('A_new','B_new','A_parent','B_parent')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def truth(x):return str(x).lower() in ('true','1')
def require(ok,message):
    if not ok:raise ValueError(message)
def interval_union_seconds(intervals):
    """Closed observed owner intervals only; overlapping owners count once."""
    merged=[]
    for start,end in sorted(intervals):
        require(math.isfinite(start) and math.isfinite(end) and end>=start,'Invalid closed owner interval')
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    return sum(end-start for start,end in merged),merged

def attempt_costs(root):
    """Small receipts only, no log scans/checkpoint reads; never infer GPU work."""
    rows=[];sources={};warnings=[];closed=[];recovery=[]
    def small(path):
        require(path.stat().st_size<=262144,'Attempt receipt unexpectedly large: '+str(path))
        value=read(path);sources[str(path)]=digest(path);return value
    owners=root/'owners'
    for owner in sorted(owners.iterdir()) if owners.exists() else []:
        if not owner.is_dir():continue
        begin=owner/'project_job_started.json';exit=owner/'project_exit.json'
        if not begin.exists():continue
        try:
            start=small(begin)['utc'];finish=small(exit) if exit.exists() else None
            end=finish['utc'] if finish else None
            status='closed' if finish and finish.get('children_reaped') is True else 'not_closed_or_unreaped'
            seconds=None
            if status=='closed':
                require(end>=start,'Owner exit precedes start')
                closed.append((start,end));seconds=end-start
            row=dict(owner=owner.name,start_utc=start,end_utc=end,observed_closed_wall_seconds=seconds,status=status,
                role='recovery_attempt' if 'recovery' in owner.name else 'original_or_other_attempt',
                start_receipt=str(begin),exit_receipt=str(exit) if exit.exists() else None)
            failure=owner/'project_failure.json'
            if failure.exists():row['failure']=small(failure).get('error')
            rows.append(row)
        except Exception as e:warnings.append(dict(path=str(owner),reason=repr(e)))
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            for sub in ('training',f'backend/{role}/training'):
                d=root/scope/role/sub
                for path in sorted(d.glob('recovery.*.json')):
                    try:
                        receipt=small(path);discarded=receipt.get('known_discarded_update_ids',[])
                        require(len(discarded)==len(set(discarded)),'Duplicate discarded update ID within receipt')
                        if not discarded:continue
                        recovery.append(dict(path=str(path),sha256=sources[str(path)],scope=scope,role=role,component='frontend' if sub=='training' else 'backend',
                            durable_updates=receipt.get('durable_updates'),completed_epochs=receipt.get('completed_epochs'),
                            discarded_update_ids=discarded,discarded_update_count=len(discarded),
                            abandoned_rows=receipt.get('abandoned_rows'),selected_age_or_exposure_contribution=0,
                            interpretation='Actual work consumed in failed attempt; not present in restored model; not added to selected update age or exposure.'))
                    except Exception as e:warnings.append(dict(path=str(path),reason=repr(e)))
    union,merged=interval_union_seconds(closed)
    return dict(owner_attempts=rows,closed_owner_wall_union_seconds=union,closed_owner_merged_intervals=merged,
        closed_owner_sum_before_overlap_removal_seconds=sum(b-a for a,b in closed),
        open_or_unreaped_owner_count=sum(x['status']!='closed' for x in rows),recovery_work=recovery,
        known_discarded_actual_update_count=sum(x['discarded_update_count'] for x in recovery),
        selected_age_or_exposure_contribution_from_discarded=0,warnings=warnings,sources_sha256=sources,
        interpretation='Observable closed owner wall intervals, including setup/cache/validation/waits and failed work; interval union avoids overlap double count. Open or unreaped owners excluded from union. Component finished timers may reset on recovery and are separate diagnostics, never added to owner wall or interpreted as GPU-active/optimizer time. No estimate of GPU-hours, FLOPs, utilization or energy.')

def validate_primary(summary,registration,pairs,rollouts):
    require(summary['cohort']=='test' and summary['n']==registration['n']==64,'Only sealed64 final test qualifies')
    require(len(set(registration['seeds']))==64,'Final seeds must be64 unique')
    require(summary['full_denominator_preserved'] and summary['episodes']==256,'Full256 denominator required')
    require(len(pairs)==64 and {int(x['seed']) for x in pairs}==set(registration['seeds']),'Paired CSV differs from registration')
    require(len({x['family'] for x in pairs})==64,'Paired family denominator differs')
    require(len(rollouts)==256 and len({(int(x['seed']),x['role']) for x in rollouts})==256,'Rollout denominator differs')
    idx={(int(x['seed']),x['role']):x for x in rollouts}
    rates={x['role']:x for x in summary['absolute_success']}
    require(set(rates)==set(ROLES),'Four complete students required')
    for role in ROLES:
        k=sum(truth(x[role+'_success']) for x in pairs);a=rates[role]
        require(a['n']==64 and a['success']==k and abs(a['success_rate']-k/64)<1e-12,'Absolute rate/CSV mismatch '+role)
        for x in pairs:
            require(truth(x[role+'_success'])==truth(idx[int(x['seed']),role]['success']),'Pair/rollout mismatch')
    contrasts={x['name']:x for x in summary['contrasts']}
    for name,left,right in [('A_new_minus_B_new','A_new','B_new'),('A_new_minus_parent','A_new','A_parent'),('B_new_minus_parent','B_new','B_parent')]:
        c=contrasts[name];ls=[truth(x[left+'_success']) for x in pairs];rs=[truth(x[right+'_success']) for x in pairs]
        require(c['n']==64 and c['left_success']==sum(ls) and c['right_success']==sum(rs),'Contrast count mismatch')
        require(abs(c['paired_difference_pp']-(sum(ls)-sum(rs))*100/64)<1e-10,'Contrast difference mismatch')
        for key,value in dict(both_success=sum(a and b for a,b in zip(ls,rs)),gained=sum(a and not b for a,b in zip(ls,rs)),lost=sum(b and not a for a,b in zip(ls,rs)),both_failure=sum(not a and not b for a,b in zip(ls,rs))).items():
            require(c[key]==value,'Contrast transition mismatch '+key)
    require(summary['both_new_reach_50_percent']==all(rates[r]['success_rate']>=.5 for r in ROLES[:2]),'Milestone mismatch')
    return rates,contrasts

def actual_component(directory,component,torch,ledger_class):
    deploy=read(directory/'deployment.json');checkpoint=directory/'best.pt'
    require(deploy['epoch']==60 and deploy['selection']=='preregistered_epoch60_only' and deploy['validation_used_for_selection'] is False,'Fixed epoch60 receipt required')
    require(digest(checkpoint)==deploy['checkpoint_sha256'],'Deployment checkpoint hash mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    require(state['epoch']==60 and state['updates']==deploy['update'],'Deployment age mismatch')
    ledger=state['effective_exposure_state'];visits=state['sampling_exposure' if component=='frontend' else 'episode_counts']
    exposure=ledger_class(ledger['legal'],ledger).summary(visits)
    require(exposure['eligible'] and exposure['minimum_tick_steps']>=50 and not exposure['ancestors_counted'],'Actual selected tick coverage failed')
    require(ledger['updates']==state['updates'],'Ledger age mismatch')
    entry=None
    if component=='frontend':
        e=state['entry_exposure_state'];values=[int(v) for row in e['counts'].values() for v in row.values()]
        require(e['updates']==state['updates'] and values and min(values)>=50,'Actual selected native entry coverage failed')
        entry=dict(entries=len(e['entries']),slots=len(values),minimum_steps=min(values),below50=sum(x<50 for x in values))
    curves=directory/'family_phase_curves.csv';require(curves.exists(),'Family/phase curves missing')
    finished=read(directory/'finished.json')
    result=dict(directory=str(directory),checkpoint=str(checkpoint),checkpoint_sha256=deploy['checkpoint_sha256'],epoch=60,updates=state['updates'],exposure=exposure,entry=entry,
        base_sha256=state.get('configuration',{}).get('base_sha256'),wall_seconds=finished.get('wall_seconds',finished.get('wall_s')),curves=str(curves),visits=visits)
    del state
    return result

def curve_trend(path):
    rows=[x for x in csvrows(path) if x['family']=='__all__' and x['phase']=='__all__'];out={}
    for split in sorted({x['split'] for x in rows}):
        r=sorted((x for x in rows if x['split']==split),key=lambda x:int(x['epoch']));tail=[x for x in r if 51<=int(x['epoch'])<=60];prior=[x for x in r if 41<=int(x['epoch'])<=50]
        if tail and prior:
            a=sum(float(x['normalized_equal_mse']) for x in prior)/len(prior);b=sum(float(x['normalized_equal_mse']) for x in tail)/len(tail)
            out[split]=dict(epochs41_50_mean=a,epochs51_60_mean=b,relative_change=(b-a)/a if a else None,tail_points=len(tail),prior_points=len(prior),scope='Descriptive same-component same-split error; no checkpoint selection or A/B nominal ranking')
    return out

def assemble(root,output,audit):
    inputs={};checks=[];pending=[]
    costs=attempt_costs(root);write(output/'attempt_wall_accounting.json',costs)
    def take(name,path):
        if not path.exists():pending.append(dict(item=name,reason='missing',path=str(path)));return None
        try:x=read(path);inputs[name]=dict(path=str(path),sha256=digest(path));return x
        except Exception as e:pending.append(dict(item=name,reason=repr(e),path=str(path)));return None
    session=take('session',root/'session.json');final=take('final_summary',root/'evaluation/test/summary/primary_summary.json')
    registration=take('final_registration',root/'evaluation/test/registration.json')
    collection=take('collection',root/'collection/collection_summary.json');experts=take('experts',root/'experts_shared/result.json')
    parents=take('actual_parent_ages',root/'actual_parent_ages.json');pairs_identity=take('parent_pairs',root/'parent_pairs.json')
    budget=take('budget',root/'learning/budget_registration.json');encoder=take('encoder_feasibility',audit/'encoder_probe_feasibility.json')
    closure=take('test_summary_exit',root/'cycle_logs/summarize_test_exit.json')
    pairpath=root/'evaluation/test/summary/per_initial_state.csv';rollpath=root/'evaluation/test/summary/rollouts.csv'
    for p in (pairpath,rollpath):
        if not p.exists():pending.append(dict(item=p.name,reason='missing',path=str(p)))
        else:inputs[p.name]=dict(path=str(p),sha256=digest(p))
    for scope in ('common180_parents','learning'):
        preflight=take(scope+'_preflight',root/scope/'complete_exposure_preflight.json')
        if preflight and preflight.get('passed') is not True:pending.append(dict(item=scope+'_preflight',reason='not passed'))
        for role in 'AB':
            for component,sub in [('frontend','training'),('backend',f'backend/{role}/training')]:
                d=root/scope/role/sub
                for name in ('deployment.json','finished.json','best.pt','family_phase_curves.csv'):
                    if not (d/name).exists():pending.append(dict(item=f'{scope}/{role}/{component}/{name}',reason='missing',path=str(d/name)))
    if closure and closure.get('returncode')!=0:pending.append(dict(item='test_summary_exit',reason='nonzero returncode'))
    if pending:return dict(status='pending',checklist=pending,inputs=inputs,actual_checkpoint_reads=0,attempt_wall_accounting=str(output/'attempt_wall_accounting.json'))
    try:
        require(registration['cohort']=='test','Wrong final cohort');require(final['registration_sha256']==digest(root/'evaluation/test/registration.json'),'Final registration hash mismatch')
        rates,contrasts=validate_primary(final,registration,csvrows(pairpath),csvrows(rollpath));checks.append('sealed64/256 denominator, absolute/paired CSV and gains/losses consistent')
        require(collection['opportunities']==96 and all(collection['roles'][r]['opportunities']==48 for r in 'AB'),'Collection denominator differs')
        require(experts['software_complete'] and experts['accepted']==24,'Shared24 experts incomplete')
        require(budget['epochs']==budget['deploy_epoch']==60 and budget['fixed_selection'],'Fixed matched budget not registered')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        from visuotactile.effective_exposure import Ledger
        actual={};trends={};pools={}
        for scope in ('common180_parents','learning'):
            actual[scope]={}
            for role in 'AB':
                actual[scope][role]={}
                for component,sub in [('frontend','training'),('backend',f'backend/{role}/training')]:
                    row=actual_component(root/scope/role/sub,component,torch,Ledger);actual[scope][role][component]=row
                    trends[f'{scope}/{role}/{component}']=curve_trend(Path(row['curves']))
                f=actual[scope][role]['frontend'];b=actual[scope][role]['backend']
                require(b['base_sha256']==f['checkpoint_sha256'],'Rear cache not bound to its new frontend')
                if scope=='common180_parents':
                    for comp in ('frontend','backend'):
                        require(parents['roles'][role][comp+'_epochs']==60 and parents['roles'][role][comp+'_updates']==actual[scope][role][comp]['updates'],'Registered actual parent ages differ')
                    for comp in ('frontend','backend'):require(pairs_identity[role][comp+'_sha256']==actual[scope][role][comp]['checkpoint_sha256'],'Collection parent differs from reconstructed deployment')
                else:
                    data=read(root/scope/role/'state_data/data.json');part=data['pool_partitions'];pool={k:len(v) for k,v in part.items()}
                    require(len(data['paired_pool']['original_ids'])==180 and pool['new_full']==24,'Original180/shared24 pool mismatch')
                    require(pool['own_new_correction']==collection['roles'][role]['valid_trajectories'],'Own correction population differs')
                    for identity in part['own_new_correction']:require(data['entries'][identity]['source_group']==role,'Cross-role correction')
                    multiplicities=budget['frontend']['roles'][role]['episode_multiplicities'];pool['dp_visits_per_epoch_by_partition']={k:sum(multiplicities[i] for i in ids) for k,ids in part.items()};pools[role]=pool
            for comp in ('frontend','backend'):
                require(actual[scope]['A'][comp]['updates']==actual[scope]['B'][comp]['updates'],'A/B actual updates differ '+scope+comp)
        for role in ROLES:
            scope='learning' if role.endswith('_new') else 'common180_parents';r=role[0]
            for comp in ('frontend','backend'):require(registration['pairs'][role][comp+'_sha256']==actual[scope][r][comp]['checkpoint_sha256'],'Final evaluated model differs')
        checks.append('Eight selected components: epoch60 SHA, actual successful-step tick>=50, DP native-entry>=50, paired updates and frontend/rear identity')
        require(not torch.cuda.is_initialized(),'CPU-only assembler unexpectedly initialized CUDA')
        write(output/'actual_exposure_and_age.json',actual);write(output/'curve_trends.json',trends);write(output/'pool_partitions.json',pools)
        text=report(root,final,rates,contrasts,collection,experts,parents,actual,pools,trends,encoder,session,costs)
        (output/'DRAFT_REPORT.md').write_text(text)
        return dict(status='draft_ready_for_root_review',checks=checks,inputs=inputs,actual_checkpoint_reads=8,cuda_initialized=False,report=str(output/'DRAFT_REPORT.md'),limitations='Draft only; final scientific wording and next-action prioritization require root review. No publication or production mutation.')
    except Exception as e:
        return dict(status='pending_validation_failed',checklist=[dict(item='closed_evidence_validation',reason=repr(e))],checks=checks,inputs=inputs)

def report(root,s,rates,contrasts,c,e,p,actual,pools,trends,encoder,session,costs=None):
    pct=lambda x:f'{x*100:.1f}%'
    ci=lambda x:f'[{x[0]:.1f}, {x[1]:.1f}] pp'
    interval=lambda x:f'[{100*x[0]:.1f}%, {100*x[1]:.1f}%]'
    delta=lambda r:contrasts[r+'_new_minus_parent']
    def progress(r):
        c=delta(r);lo,hi=c['paired_bootstrap95_pp']
        return '改善区间排除零' if lo>0 else '下降区间排除零' if hi<0 else '点估计改善、区间跨零' if c['paired_difference_pp']>0 else '点估计下降、区间跨零' if c['paired_difference_pp']<0 else '净差为零、仍有不确定性'
    ab=contrasts['A_new_minus_B_new'];both=s['both_new_reach_50_percent']
    lines=[f'# A/B 配对 DAgger 一轮结果草稿（待主执行者审阅）', '',f"A 新版 {rates['A_new']['success']}/64（{pct(rates['A_new']['success_rate'])}），{progress('A')}；B 新版 {rates['B_new']['success']}/64（{pct(rates['B_new']['success_rate'])}），{progress('B')}。两组均达到预登记50%工程门槛：{'是' if both else '否'}；50%仅是本轮工程里程碑，不能直接宣称部署可用。A−B={ab['paired_difference_pp']:+.1f} pp，配对95%区间{ci(ab['paired_bootstrap95_pp'])}。",'']
    failures={r:64-next(x['native_first_subgoal'] for x in s['funnels'] if x['role']==r+'_new') for r in 'AB'}
    focus='首子目标前' if failures['A']+failures['B']>=sum(x['native_first_subgoal']-x['final_insertion'] for x in s['funnels'] if x['role'].endswith('_new')) else '首子目标确认后'
    lines += [f"下一步优先定位{focus}的最终下发命令与实际反馈：A/B 尚未达首子目标各{failures['A']}/{failures['B']}例；具体纠偏类别应结合本轮开发记录复核后确定。该建议仅据阶段分母定位，未宣称因果瓶颈。",'', '| 完整学生 | 严格完整插入 / 全部 | Wilson 95%区间 | 软件未完成 | 协议未核验 |','| --- | ---: | --- | ---: | ---: |']
    for r in ROLES:
        x=rates[r];lines.append(f"| {r} | {x['success']}/{x['n']}（{pct(x['success_rate'])}） | {interval(x['wilson95'])} | {x['software_incomplete']} | {x['protocol_unverified']} |")
    lines+=['','| 配对比较 | 净差 | 配对95%区间 | 双成功 | 新增/左胜 | 丢失/右胜 |','| --- | ---: | --- | ---: | ---: | ---: |']
    for name in ('A_new_minus_B_new','A_new_minus_parent','B_new_minus_parent'):
        x=contrasts[name];lines.append(f"| {name} | {x['paired_difference_pp']:+.1f} pp | {ci(x['paired_bootstrap95_pp'])} | {x['both_success']} | {x['gained']} | {x['lost']} |")
    lines+=['','绝对率区间为Wilson；差值按64个家族整对重采样200000次、固定seed20261122。宽区间仍允许反向效果。软件未完成、协议缺失或初态不合格均按事前全体分母计失败；逐轨原因保留。专家恢复不计学生自主成功。', '', '| 新版 / 全体64 | 稳定持物代理 | 曾进入15mm门 | 原生首子目标累计10tick确认 | 最终插入 |','| --- | ---: | ---: | ---: | ---: |']
    for x in s['funnels']:
        lines.append(f"| {x['role']} | {x['stable_hold']}/64 | {x['first_15mm_gate']}/64 | {x['native_first_subgoal']}/64 | {x['final_insertion']}/64 |")
    lines+=['','上表各列分母均为全部64，阶段不是严格嵌套漏斗。稳定持物为既有held12物理代理；15mm进门不等于累计10tick首子目标确认。保留曾进门又离开与从未进门标记。', '',f"共享专家示范接受{e['accepted']}/{e['families']}家族，实际尝试{e['attempts']}次；失败、拒绝与重试见experts_shared/results.json。",'','| 组 | 学生机会 | 未触发 | 接管 | 专家恢复 | 有效纠偏 | 合法专家tick / 准入tick |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for role in 'AB':
        x=c['roles'][role];lines.append(f"| {role} | {x['opportunities']} | {x['no_trigger']} | {x['takeovers']} | {x['expert_recovered']} | {x['valid_trajectories']} | {x['legal_expert_ticks']} / {x['admitted_legal_expert_ticks']} |")
    if 'observable_pair_checks' in c:
        pair_checks=c['observable_pair_checks']
        lines+=['',f"采集可观测初态：{pair_checks['passed']}/48对已核验，{pair_checks['unverified']}/48对未核验，失败{pair_checks['failed']}对；不声称48对全部通过。中断前15次机会均已消耗，只继续原登记尚未分配的81次，未重跑已启动机会。",'', '| 组 / 全部48机会 | 软件未完成 | 初态非法 | 初态合法性未知 |','| --- | ---: | ---: | ---: |']
        for role in 'AB':
            x=c['roles'][role];lines.append(f"| {role} | {x['software_incomplete']} | {x['invalid_initialization']} | {x['unknown_initialization']} |")
        lines+=['','上述中断、非法与未核验机会均保留原分母，未当作未触发。只有已核验完整捕获的合法专家后缀可进入训练；保存了RNG端点但执行记录不完整的机会，不据此认证完整噪声消费。']
    lines+=['','| 组 / 触发原因 | 机会 | 接管 | 恢复 | never-entry | 曾进门又离开 | 合法tick |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role in 'AB':
        for stage,x in c['roles'][role]['stages'].items():lines.append(f"| {role} / {stage} | {x['opportunities']} | {x['takeovers']} | {x['recovered']} | {x['never_entered']} | {x['entered_then_left']} | {x['legal_ticks']} |")
    lines+=['','实际数据可不同；薄弱阶段由上述实际分母判断，不因触发少而追加机会。学生真实前缀保留作历史输入，仅合法专家后缀进入动作监督。','', '历史entry本轮前端均epoch50，但祖先年龄未能对齐，因此按登记从共同180、相同初始化规则重建完整父本；历史130-50和entry均保留。父本与本轮各四组件均按登记部署epoch60，后端使用各自选中新DP重建的nominal缓存。','', '| 阶段 / 组 / 组件 | 实际更新 | 最低合法tick成功step | 最低原生入口slot成功step |','| --- | ---: | ---: | ---: |']
    for scope in actual:
        for role in 'AB':
            for comp,x in actual[scope][role].items():lines.append(f"| {scope} / {role} / {comp} | {x['updates']} | {x['exposure']['minimum_tick_steps']} | {x['entry']['minimum_steps'] if x['entry'] else '不适用'} |")
    lines+=['','曝光从实际部署checkpoint的逐成功更新ledger重建，重复窗口、padding、无标签交接和祖先不计。覆盖到监督不等于学会控制。两组分别保存旧完整/旧纠偏/共享新完整/本组新纠偏分区和实际采样，见pool_partitions.json。','', '| 组 / 数据分区 | 轨迹数 | DP每epoch实际登记访问数 |','| --- | ---: | ---: |']
    for role in 'AB':
        for part in ('old_full','old_correction','new_full','own_new_correction'):lines.append(f"| {role} / {part} | {pools[role][part]} | {pools[role]['dp_visits_per_epoch_by_partition'][part]} |")
    lines+=['','训练是否仍改善：curve_trends.json按相同组件、相同split比较epoch41–50与51–60的描述误差，完整家族/阶段曲线在各组件family_phase_curves.csv；该诊断不选模。DP训练与验证预测协议不同，不直接比较两条曲线绝对值；A学q_reached、B学u_expert，不用nominal损失给A/B排名。泛化是否改善只由上方独立64自主结果与不确定性回答。','','编码器便宜定位没有拟合：旧36家族已参加读出；独立96条开发记录没有RGB-D/特征载荷；另两family短前缀有局部与聚合缓存但近孔tick为0，仅33个持物代理tick，无法完成合格的独立阶段比较。未启动四套编码器重训。', '', '本轮检验“扩初态＋自身纠偏＋配对训练”整体；没有仅回放旧数据且等更新的A/B对照，不能单独估计新数据贡献。完整DP＋Transformer、原生120/60/5Hz、三视角逐帧RGB-D、五指力/FK、原可达投影/归一化/严格成功判据保持。初态只声称可观测量配对，不声称隐藏PhysX或未来随机流逐位相同。']
    lines+=['','| 本轮组件 / split | epoch41–50误差均值 | epoch51–60误差均值 | 相对变化 |','| --- | ---: | ---: | ---: |']
    for name,groups in trends.items():
        if not name.startswith('learning/'):continue
        for split,x in groups.items():
            change='不可计算' if x['relative_change'] is None else f"{x['relative_change']*100:+.1f}%"
            lines.append(f"| {name} / {split} | {x['epochs41_50_mean']:.6g} | {x['epochs51_60_mean']:.6g} | {change} |")
    lines+=['','负变化表示该组件该split的后10epoch描述误差下降，正变化表示上升；只是趋势，不等于收敛证明。验证为既有validation6；自主新64的泛化结论仍以上述严格插入率及区间为准。']
    costs=costs if costs is not None else attempt_costs(root)
    stamp=lambda value:datetime.datetime.fromtimestamp(value,datetime.timezone(datetime.timedelta(hours=8))).isoformat() if value is not None else '尚无闭合回执'
    lines+=['',f"登记开始{stamp(session['started_utc'])}；终测汇总{stamp(s['observed_utc'])}；墙钟间隔{(s['observed_utc']-session['started_utc'])/3600:.2f}h（包含等待）。",'', '| 可观测owner尝试 | 开始（+08） | 退出（+08） | 闭合墙钟小时 | 状态 |','| --- | --- | --- | ---: | --- |']
    for row in costs['owner_attempts']:
        hours=f"{row['observed_closed_wall_seconds']/3600:.3f}" if row['observed_closed_wall_seconds'] is not None else '未计入'
        lines.append(f"| {row['owner']} | {stamp(row['start_utc'])} | {stamp(row['end_utc'])} | {hours} | {row['status']} |")
    lines += ['',f"已闭合owner的可观测时间区间并集{costs['closed_owner_wall_union_seconds']/3600:.3f}h；重叠owner壁钟只计一次，未闭合/未回收owner {costs['open_or_unreaped_owner_count']}个不纳入该并集。该时间含准备、等待、缓存、验证和失败尝试，不能解释为GPU有效训练时间或GPU小时。",'', '| 组件结束回执计时（单独诊断） | 日志wall小时 |','| --- | ---: |']
    for scope,roles in actual.items():
        for role,components in roles.items():
            for component,row in components.items():
                seconds=row.get('wall_seconds');value=f'{seconds/3600:.3f}' if seconds is not None else '未记录'
                lines.append(f'| {scope}/{role}/{component} | {value} |')
    lines+=['','DP finished.wall_seconds在恢复后重新计时；上表仅为结束回执提供的计时，各组件/各attempt计时范围可能不同，不相加、不与owner壁钟相加，也不称累计完整训练耗时。原parent_train与parent_train_recovery01已在owner表分列，其他后续recovery同样按独立attempt保留。']
    for row in costs['recovery_work']:
        lines+=['',f"技术中断 {row['scope']}/{row['role']}/{row['component']}：从epoch{row['completed_epochs']}/update{row['durable_updates']}恢复；{row['discarded_update_count']}步未持久化更新（ID {', '.join(map(str,row['discarded_update_ids']))}）确已消耗计算，原始弃用日志保留，但对最终选中模型训练年龄与曝光的贡献为0。其耗时包含在相应原owner尝试区间，未额外再加。"]
    if costs['warnings']:lines+=['','时间回执存在未核验项，详见attempt_wall_accounting.json；未以缺项推算额外计算。']
    lines+=['','未采样的资源、能耗、有效GPU工作量不推算。独立运行、失败尝试、资源回执和历史池完整保留。', '',f'本机根：`{root}`。主结果与逐初态表：`evaluation/test/summary/primary_summary.json`、`per_initial_state.csv`、`rollouts.csv`。本草稿输入SHA与检查见assembly_status.json；实际曝光actual_exposure_and_age.json；attempt时间与弃用计算见attempt_wall_accounting.json；数据/预算见learning/budget_registration.json与各组state_data/data.json；编码器限制见独立AUDIT/encoder_probe_feasibility.json。','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--audit',type=Path,default=AUDIT);p.add_argument('--output',type=Path);a=p.parse_args()
    audit=a.audit.resolve();output=(a.output or audit/('report_assembly_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))).resolve()
    require(output.is_relative_to(audit) and output!=audit,'Output must be a NEW subdirectory inside AUDIT')
    output.mkdir(parents=True,exist_ok=False)
    status=assemble(a.root.resolve(),output,audit);status.update(observed_utc=time.time(),assembler_sha256=digest(Path(__file__)),writes_only=str(output),gpu_or_kit_started=False)
    write(output/'assembly_status.json',status);print(json.dumps(dict(status=status['status'],output=str(output),pending_items=len(status.get('checklist',[]))),ensure_ascii=False))
if __name__=='__main__':main()
