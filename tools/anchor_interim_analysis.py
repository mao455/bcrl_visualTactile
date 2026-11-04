"""CPU-only interim comparison, trajectories, and small public artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.state_image_queue import CODE, read, write, sha
from tools.anchor_interim_eval import counts
from tools.h24_diagnosis_report import describe, compare_initial
from tools.h24_diagnosis_common import first_sustained
from tools.grasp_diagnosis_analysis import tick_metrics, window_summary

PUBLIC=CODE/'research_docs/evidence/anchor_interim_20261105_01'
REPORT=CODE/'research_docs/STATE_DP_INTERIM_RESULTS.md'


def analyze(root):
    s=read(root/'session.json');state=read(root/'status.json')
    attempts=read(root/'attempts.json') if (root/'attempts.json').exists() else []
    # The stage commit follows trace/video acceptance; the attempt ledger may
    # still contain its earlier process-exit snapshot until the next stage.
    attempts=[read(root/'stages'/x['name']/'committed.json')
        if (root/'stages'/x['name']/'committed.json').exists() else x for x in attempts]
    accepted=[x for x in attempts if x.get('native_outcome_valid')]
    originals=[x for x in read(Path(s['baseline_root'])/'attempts.json')
        if x.get('valid') and x['model']=='StateDP' and x['ordinal'] in s['closed_loop_ordinals']]
    rows=[];plot_data=None
    for record in accepted:
        trial=record['result']['trials'][0];identity=trial['identity']
        old=next(x for x in originals if x['trial']['identity']==identity)
        ordinal=old['ordinal'];new_path=Path(record['output'])/identity;old_path=Path(old['output'])/identity
        expert=CODE/'outputs/visuotactile/native_dp_v2/position_20261103_01'/f'expert_learning_single_{ordinal:02d}'/identity
        new,ns,nt=describe(new_path,expert);previous,ps,pt=describe(old_path,expert);teacher,es,et=describe(expert)
        assert teacher['result']['success']
        initial=compare_initial(new_path,expert)
        assert initial['observable_equal'] and initial['environment_rng_equal'] and not initial['exported_differences']
        common=min(len(nt['target']),len(pt['target']),len(et['target']))
        prefix={key:window_summary(tick_metrics(trace),common) for key,trace in [('new',nt),('old',pt),('expert',et)]}
        def compact(item,series):
            tick=first_sustained(series['actual_expert_fk_mm'],5.,3)
            return dict(steps=item['steps'],events=item['events'],first_ee_divergence_tick=tick,
                first_ee_divergence_phase='not_observed' if tick is None else
                    'before_expert_contact' if teacher['events']['contact_tick'] is None or tick<teacher['events']['contact_tick'] else 'after_expert_contact',
                continuity=item['continuity'],clipping=item.get('clipping_detail'),
                target_actual_hand_rmse_rad=item['target_actual_hand_rmse_rad'],reference=item['reference'])
        rows.append(dict(ordinal=ordinal,seed=trial['seed'],identity=identity,new_result=trial,old_result=old['trial'],
            expert_result=teacher['result'],new=compact(new,ns),old=compact(previous,ps),
            expert_events=teacher['events'],initial=initial,video=str(new_path/'rollout.mp4'),
            sdk_exit=record['sdk_exit'],trace_sha256=record['trace_sha256'],video_validation=record['video'],
            common_prefix_ticks=common,common_prefix_metrics=prefix))
        if plot_data is None:plot_data=(ordinal,ns,ps,es)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    completed=[x for x in attempts if 'exit_code' in x]
    new_counts=counts([x['new_result'] for x in rows]);old_counts=counts([x['trial'] for x in originals])
    paired_old=counts([x['old_result'] for x in rows])
    payload=dict(state=state,planned=4,starts=len(attempts),maximum_starts=5,valid=len(rows),
        pending=4-len(rows),software_missing=sum(not x.get('valid') for x in completed),
        acceptance_failed=sum(x.get('valid',False) and not x.get('native_outcome_valid',False) for x in completed),
        new=new_counts,old_fixed4=old_counts,old_same_completed_initials=paired_old,rows=rows,
        original_formal='StateDP10000 0/12; ImageDP10000 0/12, unchanged',test40_used=False,
        source_commit=s['source_commit'],analysis_source_sha256=sha(__file__),
        checkpoint=read(root/'checkpoint.json') if (root/'checkpoint.json').exists() else None,
        threshold=dict(ee_mm=5.,consecutive_native_ticks=3),stable_grasp_native='N/A')
    for f in ('monitor_exit.json','parent_pause.json','owner_error.json'):
        if (root/f).exists():payload[f.removesuffix('.json')]=read(root/f)
    restore=Path(s['parent_root'])/'stages/state_train_epoch_150/attempt_1/output/restore.json'
    if restore.exists():payload['training_resume']=read(restore)
    write(PUBLIC/'summary.json',payload)
    keys=['contact','lift','first_subgoal','success'];labels=['Object contact diagnostic','Native lift','First subgoal','Full task success']
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4);width=.36
    for offset,label,c in [(-width/2,'Old 10000: matched completed initials',paired_old),(width/2,'Continued 16584',new_counts)]:
        if c['valid']:
            values=[100*c[k]/c['valid'] for k in keys]
            bars=ax.bar(x+offset,values,width,label=label)
            ax.bar_label(bars,labels=[f"{c[k]}/{c['valid']}" for k in keys],padding=3)
    ax.set_xticks(x,labels,rotation=10);ax.set_ylim(0,115);ax.set_ylabel('Observed proportion (%)')
    ax.set_title(f"Fixed validation4 interim evaluation: {len(rows)}/4 valid; not a monotonic funnel")
    if rows:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(PUBLIC/'stages.png',dpi=140);plt.close(fig)
    if plot_data:
        ordinal,ns,ps,es=plot_data;fig,axes=plt.subplots(2,2,figsize=(10,7))
        for label,series in [('old10000',ps),('continued16584',ns),('expert',es)]:
            tick=series['tick']
            if 'actual_expert_fk_mm' in series:axes[0,0].plot(tick,series['actual_expert_fk_mm'],label=label)
            axes[0,1].plot(tick,series['object_height_m'],label=label)
            axes[1,0].plot(tick,series['hand_jump_rad'],label=label)
            axes[1,1].plot(tick,series['object_contact_N'],label=label)
        for ax,title in zip(axes.flat,['Link7 distance from expert (mm)','Object height (m)','Hand target tick change RMS (rad)','Object normal contact force (N)']):
            ax.set_title(title);ax.set_xlabel('Native control tick (60 Hz)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'Fixed validation ordinal {ordinal}; independent native episodes')
        fig.tight_layout();fig.savefig(PUBLIC/'representative.png',dpi=140);plt.close(fig)
    lines=['# StateDP 续训至100 anchor等效epoch：新增闭环实测','',
        '|模型|累计更新|成功/有效|成功率|软件缺失|完成情况|','|---|---:|---:|---:|---:|---|']
    rate=f"{100*new_counts['success']/len(rows):.1f}%" if rows else '尚无有效结果'
    lines += [f"|StateDP 原模型（固定同4初态）|10000|{old_counts['success']}/{old_counts['valid']}|{100*old_counts['success']/old_counts['valid']:.1f}%|原记录复用|4/4|",
        f"|StateDP 续训|16584|{new_counts['success']}/{len(rows)}|{rate}|{payload['software_missing']}|{len(rows)}/4；{state['state']}|",
        '|ImageDP|10000|原正式0/12|0%|本次未新增尝试|尚未续训，本次未重复评估|','',
        f"本次计划4个初态，最多5次启动；已启动{len(attempts)}次，软件缺失{payload['software_missing']}，结果验收失败{payload['acceptance_failed']}，尚缺有效结果{payload['pending']}个初态。",
        '原正式两组0/12保持不变；新增validation4是阶段诊断，不并入原正式分母，不使用test40。','',
        '|模型|接触诊断|原生抬升|首子目标|完整成功|fall|dropped|timeout|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,c in [('旧10000（本次已完成的同初态）',paired_old),('新16584',new_counts)]:
        lines.append('|'+name+'|'+ '|'.join(f"{c[k]}/{c['valid']}" for k in ['contact','lift','first_subgoal','success','fall','dropped','timeout'])+'|')
    lines+=['','原环境没有原生“稳定抓持成功”离散判据，保持N/A；接触使用已有0.1N诊断，不冒充稳定抓持。接近代理亦不作为成功判据。阶段事件不是强制单调漏斗。','',
        '![阶段对比](evidence/anchor_interim_20261105_01/stages.png)','',
        '|初态seed|新终止|旧/新首次末端偏离tick|专家持续接触tick|新边界手指跳变rad|视频|','|---|---|---|---:|---:|---|']
    for x in rows:
        lines.append(f"|{x['seed']}|{x['new_result']['reasons']}|{x['old']['first_ee_divergence_tick']} / {x['new']['first_ee_divergence_tick']}|{x['expert_events']['contact_tick']}|{x['new']['continuity']['hand_boundary_jump_rms_rad']:.6f}|[本地视频]({x['video']})|")
    if rows:lines+=['','![代表轨迹](evidence/anchor_interim_20261105_01/representative.png)']
    lines+=['','首次偏离沿用5mm连续3个native tick定义；与同初态成功专家按控制tick对齐，不做时间伸缩。末端曲线仅比较公共前缀。',
        '小样本的任务结果与离线误差分开判断；未完成四条前不宣布性能提升或退化。所有视频完整解码，模型自主输出，未加入拼接/平滑/专家接管。',
        f"源码提交：`{s['source_commit']}`。checkpoint/SHA、初态一致性、执行器验收和原始结果见[机器报告](evidence/anchor_interim_20261105_01/summary.json)。",
        f"运行目录：`{root}`。SDK退出码与任务结果分开；fast exit不称正常完整关闭。训练owner在评估释放资源后恢复；原72小时截止不延长。",'']
    if len(rows)==4:
        lines+=['## 本次实测结论','',
            f"4次计划均取得有效结果；实际启动{len(attempts)}次，软件缺失{payload['software_missing']}，验收失败{payload['acceptance_failed']}。"]
        if not new_counts['success'] and not new_counts['lift'] and not new_counts['first_subgoal']:
            lines+=['续训后的模型能够触碰物体，但本次没有完成抬升或首子目标，完整任务0/4；接触覆盖3/4→4/4不代表抓取已经改善，四个初态也不足以宣布统计显著差异。']
        if all(x['new']['first_ee_divergence_phase']=='before_expert_contact' for x in rows):
            lines += [f"四条均在专家持续接触前偏离；新模型首次偏离ticks为`{[x['new']['first_ee_divergence_tick'] for x in rows]}`，旧模型为`{[x['old']['first_ee_divergence_tick'] for x in rows]}`。早期接近轨迹的问题仍在，不能只归因于接触后的失稳。"]
        values={key:float(np.median([x['common_prefix_metrics'][key]['hand_jump_boundary_rms_rad'] for x in rows])) for key in ['old','new','expert']}
        lines += [f"在每组与专家相同控制tick公共前缀上，手指重规划边界跳变RMS的跨初态中位数：旧{values['old']:.6f}、新{values['new']:.6f}、专家{values['expert']:.6f}rad。该项没有证明单一根因；首次轨迹偏差已出现在第一个12tick重规划边界之前。",
            '本次没有发现模型加载、历史初始化、绝对位置执行接口或结果写入故障。训练误差明显低于验证误差，继续增加曝光目前尚未消除闭环失败；这不等于已经证明必须更换网络或损失。',
            '原队列按既定计划继续到150 anchor等效epoch并执行验证/平台期决策；Image仍等待State的实质闭环进展与严格图像输入配对。未重训、未改控制器，原10000和本次16584 checkpoint均保留。']
        if payload.get('training_resume'):
            lines += [f"实际续训恢复：从`{payload['training_resume']['from_update']}`更新恢复，检查`{payload['training_resume']['checks']}`；不是仅发送恢复命令。"]
        if 'monitor_exit' in payload:
            resource=payload['monitor_exit']['summary']
            lines += [f"本次独立仿真监测窗口整卡均值{resource['gpu_time_weighted_mean_percent']:.2f}%，最大完整60秒均值{resource['max_complete_60s_mean_percent']:.2f}%，缺失样本{resource['missing_samples']}。仅描述本次评估窗口，不覆盖此前训练资源超限记录。"]
        lines += ['']
    REPORT.write_text('\n'.join(lines))
    return payload


def publish(root):
    paths=[str(REPORT.relative_to(CODE)),str(PUBLIC.relative_to(CODE))]
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=CODE,text=True).strip()
    assert not staged, 'Preserve unrelated staged changes'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=CODE,text=True).strip()=='feature/native-dp-training-v2'
    subprocess.run(['git','add',*paths],cwd=CODE,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=CODE).returncode==0:return
    subprocess.run(['git','diff','--cached','--check'],cwd=CODE,check=True)
    subprocess.run(['git','commit','-m','Report measured interim State checkpoint evaluation'],cwd=CODE,check=True)
    subprocess.run(['git','push','origin','HEAD:feature/native-dp-training-v2'],cwd=CODE,check=True)
    write(root/'analysis_publication.json',dict(state='published',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args()
    root=a.root.resolve();last=None
    while True:
        path=root/'attempts.json';stamp=path.stat().st_mtime_ns if path.exists() else 0
        finished=(root/'owner_exit.json').exists()
        if stamp!=last or finished:analyze(root);last=stamp
        if not a.watch or finished or time.time()>read(root/'session.json')['deadline']+300:break
        time.sleep(10)
    if a.publish:publish(root)
