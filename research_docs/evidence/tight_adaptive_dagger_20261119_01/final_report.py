"""Render only completed, audited results; never launches training or physics."""
import collections
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import time

from tools.tight_adaptive_dagger import ROOT,PUBLIC,PROJECT,CONFIG,read,write
from visuotactile.compact import sha256


def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


def videos():
    import av
    destination=PUBLIC/'videos';destination.mkdir(exist_ok=True)
    chosen=[]
    for round_index in (1,2):
        registration=read(ROOT/f'round{round_index:02d}/evaluation/registration.json')
        changes=read(PUBLIC/f'round{round_index:02d}/evaluation/paired_changes.json')
        for role in 'AB':
            for category in ('lost','gained'):
                row=next((r for r in changes if r['role']==role and r[category]),None)
                if row is None:continue
                for wave_name in registration['waves']:
                    wave=Path(wave_name);plan=read(wave/'plan.json')
                    cases=[c for c in plan['cases'] if c['spec']['seed']==row['seed'] and c['role'] in (role+'_parent',role+'_new')]
                    if len(cases)!=2:continue
                    for case in cases:chosen.append((f'r{round_index}_{category}',wave,case))
                    break
        rows=read(PUBLIC/f'round{round_index:02d}/evaluation/per_case.json')
        failed=next((r for r in rows if r['role'].endswith('_new') and not r['success']),None)
        if failed:
            for name in registration['waves']:
                wave=Path(name)
                case=next((c for c in read(wave/'plan.json')['cases'] if c['name']==failed['case']),None)
                if case:chosen.append((f'r{round_index}_failure',wave,case));break
    selected=[];seen=set();total=0
    for category,wave,case in chosen:
        source=wave/case['name']/'autonomous.mp4'
        if str(source) in seen:continue
        seen.add(str(source));result=read(source.parent/'result.json')
        with av.open(str(source)) as movie:frames=sum(1 for _ in movie.decode(video=0))
        assert frames==result['steps']
        target=destination/(category+'_'+case['name']+'.mp4')
        if target.exists():assert sha256(target)==sha256(source)
        elif total+source.stat().st_size<45*2**20:shutil.copyfile(source,target)
        else:target=source
        total+=source.stat().st_size
        selected.append(dict(category=category,role=case['role'],seed=case['spec']['seed'],
            success=result['success'],frames=frames,sha256=sha256(source),source=str(source),published=str(target)))
    spec=importlib.util.spec_from_file_location('adaptive_capture_media',PUBLIC/'capture_media.py')
    capture_media=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture_media)
    for round_index in (1,2):
        records=read(PUBLIC/f'round{round_index:02d}/combined_analysis/records.json')
        for role in 'AB':
            # R1 choices were already inspected during collection. R2 chooses
            # the first accepted own-role record, before any semantic viewing.
            known={'A':'r10_A_layout10_train_28817000_transport',
                   'B':'r10_B_layout10_train_28817004_contact'} if round_index==1 else {}
            row=next(r for r in records if r['role']==role and r['accepted'] and
                     (role not in known or r['case']==known[role]))
            directory=Path(row['root'])/row['case']
            folder=ROOT/'media'/(f'round01_{role}_'+('transport' if role=='A' else 'contact')
                                if round_index==1 else f'round02_{role}_recovery')
            if not (folder/'media.json').exists():capture_media.export(directory,folder)
            receipt=read(folder/'media.json');result=read(directory/'result.json')
            assert receipt['source_sha256']==result['dataset_capture']['sha256']
            source=folder/'capture.mp4'
            assert sha256(source)==receipt['video_sha256']
            with av.open(str(source)) as movie:frames=sum(1 for _ in movie.decode(video=0))
            assert frames==result['steps']
            target=destination/(f'r{round_index}_expert_recovery_'+row['case']+'.mp4')
            if target.exists():assert sha256(target)==sha256(source)
            elif total+source.stat().st_size<45*2**20:shutil.copyfile(source,target)
            else:target=source
            total+=source.stat().st_size
            sheet=destination/(f'r{round_index}_expert_recovery_'+row['case']+'.jpg')
            shutil.copyfile(folder/'contact_sheet.jpg',sheet)
            selected.append(dict(category=f'r{round_index}_expert_recovery',role=role,seed=row['seed'],
                success=result['success'],expert_rescue=True,autonomous_success=False,
                frames=frames,takeover_tick=row['takeover_tick'],sha256=sha256(source),
                source=str(source),published=str(target),contact_sheet=str(sheet),
                scope='Actual complete student prefix and real-state expert suffix; pre-action three-camera viewing derivative'))
    write(PUBLIC/'representative_videos.json',dict(rows=selected,selection='Chosen by recorded lost/gained/failed outcome; actual original videos, no repeated rollouts',
        all_selected_fully_decoded=True,semantic_manual_review_required=True))
    return selected


def generate():
    complete=read(ROOT/'cycle_complete.json');assert complete['rounds']==2 and complete['final_test_complete']
    closure=read(PUBLIC/'closure.json');assert closure['protected_sha_passed']
    freeze=read(ROOT/'final_freeze.json');settings=read(CONFIG/'experiment.json')
    assert freeze['seeds']==settings['final_test_seeds']
    report=['# Tight-Insertion：两轮自适应DAgger、充分曝光与完整闭环结果','',
        '两轮实际采集、四组件训练、独立曝光验收、闭环诊断与最终新初态测试均已完成。所有失败、旧合法数据与必要checkpoint保留；没有自动晋升，也没有改变encoder、网络、控制器、任务初态分布或原生成功判据。','',
        f"本轮从9417461接续，实验分支`feature/tight-adaptive-dagger-20261119`；run为`{ROOT}`。有限session截止保持原登记，最终模型于{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(freeze['utc']))}冻结后才打开{len(freeze['seeds'])}个新终测初态。开发8曾反复使用；每轮新增开发8仍属于诊断人口，不能称为最终未见测试。",'']
    spec=importlib.util.spec_from_file_location('adaptive_quantitative',PUBLIC/'quantitative_report.py')
    quantitative=importlib.util.module_from_spec(spec);spec.loader.exec_module(quantitative)
    collection=[];training=[];counts=[];changes=[];fit=[];offline=[];failures=[];stages=[]
    for round_index in (1,2):
        public=PUBLIC/f'round{round_index:02d}'
        quantitative.export(round_index)
        summary=read(public/'quantitative_summary.json')
        fit.extend([dict(round=round_index,**r) for r in summary['training']])
        failures.extend([dict(round=round_index,**r) for r in summary['failure_stages']])
        stages.extend([dict(round=round_index,**r) for r in summary['collection_stages']])
        offline.extend([dict(round=round_index,comparison='same-record offline',**r) for r in summary['shared_parent_comparisons']])
        offline.extend([dict(round=round_index,comparison='captured parent / different noise',**r) for r in summary['captured_parent_comparisons']])
        for row in read(public/'collection_summary.json')['table']:
            collection.append([round_index,row['role'],row['attempts'],row['expert_takeovers'],row['accepted'],row['expert_success'],row['legal_ticks'],json.dumps(row['stage_accepted'],ensure_ascii=False)])
        for component in read(public/'training_completion.json')['components']:
            assert component['eligible'] and component['minimum_visits']>=50 and component['minimum_tick_steps']>=50 and component['receipt_replay_exact']
            training.append([round_index,component['model'],component['actual_epochs'],component['selected_epoch'],component['actual_updates'],component['selected_updates'],component['minimum_visits'],component['minimum_tick_steps'],'是'])
        for role in 'AB':assert read(public/f'{role}_backend_raw_mask_acceptance.json')['step_receipts_exact']
        rows=read(public/'evaluation/counts.json');counts.extend([dict(round=round_index,**r) for r in rows])
        changes.extend([dict(round=round_index,**r) for r in read(public/'evaluation/paired_changes.json')])
    counts.extend([dict(round='终测',**r) for r in read(PUBLIC/'final_test/counts.json')])
    for row in counts:
        directory=PUBLIC/('final_test' if row['round']=='终测' else f"round{row['round']:02d}/evaluation")
        audit=read(directory/'execution_audit/receipt.json')['rows']
        evaluated=read(directory/'per_case.json')
        seeds={r['seed'] for r in evaluated if r['role']==row['model'] and r['cohort']==row['cohort']}
        subset=[r for r in audit if r['role']==row['model'] and r['seed'] in seeds]
        assert len(subset)==row['n']
        row['grasp']=sum(r['stable_grasp'] for r in subset)
    report += ['## 实际采集','',mdtable(['轮次','学生','完整机会','真实接管','合法纠偏','专家救回','新增合法tick','验收阶段'],collection),'',
        '专家救回不计学生自主成功。未触发、非法或软件未闭合机会分别保留原回执，不混作专家救回失败；完整有效机会的分母包含其所有失败。原资源中断和CPU准入恢复各有独立记录，未把重复日志、未提交优化或失败前缀算作新增训练曝光。阶段之间没有随机化对照。','',
        'Round 1以接触停滞、搬运、对准停滞和支撑不稳分层；Round 2使用第一轮新完整学生，按其真实训练family失败、旧成功丢失和实际可达阶段分别给A/B分配事件屏幕，额外关注原生子目标之后。无固定时刻回退、reset或理想中间状态瞬移；两条R1较晚专家未救回记录支持更早屏幕候选，不证明更早必能救回。','',
        mdtable(['轮次','接管屏幕','完整机会','接管','专家救回','合法纠偏','训练合法tick','排除专家tick','接管tick 最小/中位/最大'],
            [[r['round'],r['stage'],r['opportunities'],r['takeovers'],r['expert_rescues'],r['accepted'],r['accepted_legal_ticks'],r['excluded_expert_ticks'],
              f"{r['takeover_min']}/{r['takeover_median']}/{r['takeover_max']}"] for r in stages if r['role']=='all']),'',
        '## 四组件训练验收','',mdtable(['轮次','组件','完成epoch','选中epoch','实际更新','选中更新','最低访问','最低tick不同步','达标'],training),'',
        '各轮每条轨迹与合法expert tick只计算该轮新提交的optimizer事务，祖先不计，同一步重叠窗口取并集。DP从实际anchor与原mask独立重算；两个后端从原始合法mask、seeded重复deck及32tick/128合法tick窗口独立重建，逐条与提交日志核对。未达50次的早期验证最优无选模资格。曝光是正权loss参与，不保证非零梯度或技能保持。','',
        'A保留q_reached及原几何项，B保留u_expert；各自DP先完成，再按实际选中DP重新生成对应nominal缓存。完整学生前缀继续作为因果历史，只有洁净专家后缀参与监督。原30归一化不变，原示范每deck三访，其他合法轨迹一访。每轮fresh AdamW，DP重建EMA；同轮中断恢复应完整保留optimizer/RNG/曝光，原失败回执不覆盖。','',
        '## 完整自主闭环与保持','',mdtable(['轮次','初态组','完整策略','例数','原生插入','首子目标','后续子目标','抓持诊断','抬升诊断','hold30/60/120'],
        [[r['round'],r['cohort'],r['model'],r['n'],r['success'],r['first_subgoal'],r['later_subgoal'],r['grasp'],r['native_lift'],f"{r['hold30']}/{r['hold60']}/{r['hold120']}"] for r in counts]),'',
        '父/新完整A/B在同波比较并核验可观测初态。72历史完整策略在独立物理波检查，不称同波因果对照；隐藏PhysX状态与后续随机流没有精确重放保证。Hold60仅一秒持物诊断，非原生插入必要条件；抬升与接触代理也不构成严格包含的成功漏斗。','']
    retaining=[]
    for round_index in (1,2):
        for role in 'AB':
            for cohort in ('training','development','new_development'):
                rr=[r for r in changes if r['round']==round_index and r['role']==role and r['cohort']==cohort]
                retaining.append([round_index,role,cohort,sum(r['retained'] for r in rr),sum(r['lost'] for r in rr),sum(r['gained'] for r in rr)])
    report += [mdtable(['轮次','角色','初态组','保留旧成功','丢失旧成功','新增成功'],retaining),'',
        '逐初态完整变化见每轮`evaluation/paired_changes.csv`，命令与物体几何首次分叉见`evaluation/forks.csv`；分叉先后不能单独识别首次有害错误或反事实专家因果。小样本计数不作为显著性证明。','',
        mdtable(['轮次','初态组','完整策略','例数','成功','抬升前失败','抬升后/首子目标前失败','首子目标后失败'],
            [[r['round'],r['cohort'],r['model'],r['opportunities'],r['success'],r['before_lift'],r['after_lift'],r['after_first_subgoal']] for r in failures]),'',
        '## 离线拟合、控制反馈与解释边界','',
        '每轮`offline/per_episode.csv`给出完整记录历史上DP自身目标、DP对专家命令、anchor、correction和最终命令误差，分原30、历史纠偏、本轮纠偏及validation6，并列first24和物理阶段。训练曲线中的train2是固定诊断子集，不是全池均值；A/B的DP目标不同，不直接横比DP数值。在线父策略完整shadow预测与离线采样噪声不完全相同，不能将其差直接当精确匹配训练收益。','',
        mdtable(['轮次','组件','首轮验证','首个达标验证','选中验证','最后验证','达标→末轮验证变化'],
            [[r['round'],r['model'],f"{r['first_validation']:.6g}",f"{r['first_eligible_validation']:.6g}",f"{r['selected_validation']:.6g}",f"{r['last_validation']:.6g}",f"{r['eligible_to_last_validation_change_percent']:+.2f}%"] for r in fit]),'',
        '下面离线比较只连接相同角色、轨迹identity、物理阶段和组件，要求合法tick数相等，按全部tick及关节数汇总平方误差；负百分比表示误差下降。当前轮新增纠偏的父预测来自真实采集记录，父在线7093与新模型离线7091噪声不同，仅作描述性拟合比较；表中与共享旧池离线比较分别标明。完整阶段、first24和缺失父测量人口见`quantitative_summary.json`及`offline/*parent_comparison.csv`。','',
        mdtable(['轮次','角色','人口','组件','父预测来源','轨迹/tick','arm RMSE 父→新(rad)','hand RMSE 父→新(rad)','arm/hand变化'],
            [[r['round'],r['role'],r['cohort'],r['component'],r['comparison'],f"{r['trajectories']}/{r['ticks']}",
              f"{r['parent_arm_rmse_rad']:.6g}→{r['new_arm_rmse_rad']:.6g}",f"{r['parent_hand_rmse_rad']:.6g}→{r['new_hand_rmse_rad']:.6g}",
              '/'.join('n/a' if r[p+'_relative_change_percent'] is None else f"{r[p+'_relative_change_percent']:+.2f}%" for p in ('arm','hand'))]
             for r in offline if r['stage']=='legal' and r['component'] in ('dp_own_target','final')]),'',
        '执行审计分开统计策略提议关节裁剪、可达投影、物理执行器裁剪、关节跟踪及历史衔接；实际命令须满足原1e-6 rad数值容差。相机/触觉时钟检查严格不读未来，命令和反馈连续性检查通过也不能排除所有潜变量或感知限制。成功与失败中的频繁投影不是控制故障的充分证据，跟踪误差亦不是专家命令监督误差。','',
        '至少50次监督已满足，可排除该轮所核验的最低曝光缺口；不能仅由epoch数排除欠拟合。训练/验证曲线、全池误差与自主成功应共同解释。专家在某些晚期状态不能救回、纠偏分布覆盖、组合策略拟合/反馈与感知读出都可能限制恢复；本实验没有隔离encoder单因果作用，任何encoder改造只能作为后续受门槛对照建议，未自动执行。','',
        '## 证据、视频与复现','',
        '配置在`configs/experiments/adaptive_dagger/`；每轮独立manifest、采集结果、选中SHA、训练/验证曲线、逐轨和逐tick压缩曝光、阶段曝光及逐初态结果在本报告同名evidence目录。所有原始RGB-D、触觉、numeric、完整恢复checkpoint和全程视频留本机，必要模型不进入Git。最终新test打开之后，报告生成只读既有结果，0额外optimizer、0额外物理机会。','',
        '大模型路径与SHA以每轮`training_completion.csv`和`final_freeze.json`为准；恢复只能使用新attempt和原截止，不得重启已完成job。完整运行与准入命令均经`python -m tools.tight_project`显式job执行。CPU后处理入口：','',
        '```bash','PYTHONPATH="$PWD" python research_docs/evidence/tight_adaptive_dagger_20261119_01/final_report.py','```','',
        f"关闭保护核验通过{closure['files']}份父权重/数据索引/原统计文件；既有asset audit另保留旧400及原130依赖证明。磁盘余量{closure['free_disk_gib']:.2f}GiB为关闭瞬间观测，不是全程连续保证。每owner监控、资源中断与额外尝试均保留。"]
    (PROJECT/'research_docs/TIGHT_ADAPTIVE_DAGGER_20261119.md').write_text('\n'.join(report)+'\n')
    table_path=PUBLIC/'final_counts.csv'
    with table_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    selected=videos()
    links=[]
    for row in selected:
        video=Path(row['published'])
        target=str(video.relative_to(PROJECT/'research_docs')) if video.is_relative_to(PROJECT/'research_docs') else str(video)
        links.append([row['category'],row['role'],row['seed'],row['frames'],
            '专家纠偏完成' if row.get('expert_rescue') else '自主成功' if row['success'] else '自主失败',
            f"[实际视频]({target})"])
    with (PROJECT/'research_docs/TIGHT_ADAPTIVE_DAGGER_20261119.md').open('a') as stream:
        stream.write('\n'+mdtable(['类别','策略','seed','完整帧数','控制来源与结果','视频'],links)+'\n')
    write(PUBLIC/'final_report_receipt.json',dict(complete=True,rounds=2,training_components=8,
        all_selected_exposure_passed=True,final_test_opened_after_freeze=True,video_rows=len(selected),
        report_sha256=sha256(PROJECT/'research_docs/TIGHT_ADAPTIVE_DAGGER_20261119.md'),
        output='research_docs/TIGHT_ADAPTIVE_DAGGER_20261119.md',utc=time.time(),optimizer_updates=0,physics_updates=0))


if __name__=='__main__':generate()
