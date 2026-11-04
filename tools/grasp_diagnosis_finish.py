"""Bounded CPU-only completion of numerical/closed-loop evidence and integrity audit."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from tools.state_image_queue import CODE,read,write,sha
from tools.grasp_diagnosis_analysis import PUBLIC,FINAL,POSITION
from tools.grasp_diagnosis_report import report
from tools.h24_diagnosis_report import arrays,describe
from tools.h24_diagnosis_common import first_sustained


def finish(root):
    session=read(root/'session.json')
    while not (root/'owner_exit_ablations.json').exists():
        if time.time()>session['deadline']+120:raise RuntimeError('No diagnostic owner completion before deadline')
        time.sleep(5)
    report(root)
    attempts=read(root/'ablation_attempts.json');base=[x for x in read(FINAL/'attempts.json') if x.get('valid')]
    valid=[x for x in attempts if x.get('valid')];comparisons=[]
    for model in ('StateDP','ImageDP'):
        for ordinal in (0,4):
            original=next(x for x in base if x['model']==model and x['ordinal']==ordinal)
            identity=original['trial']['identity'];original_folder=Path(original['output'])/identity
            _,orig_series,orig_trace=describe(original_folder)
            expert=POSITION/f'expert_learning_single_{ordinal:02d}'/identity
            _,ex_series,_=describe(expert)
            for variant in ('original','precision_only','precision_K4','deterministic_precision_only','deterministic_precision_K4'):
                row=original if variant=='original' else next((x for x in valid if x['model']==model and x['ordinal']==ordinal and x['variant']==variant),None)
                if row is None:continue
                folder=Path(row['output'])/identity
                summary,series,trace=describe(folder,expert)
                n=min(len(ex_series['actual_xyz']),len(series['actual_xyz']))
                distance=np.linalg.norm(series['actual_xyz'][:n]-ex_series['actual_xyz'][:n],axis=-1)*1000
                comparisons.append(dict(model=model,ordinal=ordinal,variant=variant,steps=row['trial']['steps'],
                    success=row['trial']['success'],causal_pair_valid=row.get('causal_pair_valid',all(row.get('unchanged_first12_prefix',{'unused':True}).values())),reasons=row['trial']['reasons'],native_lift=row['trial']['native_lift'],
                    subgoals=row['trial']['native_subgoals'],contact=row['trial']['contact'],events=summary['events'],
                    first_ee_divergence_tick=first_sustained(distance,5.,3),
                    first12_target_difference_from_original_rad=float(np.abs(trace['target'][:12]-orig_trace['target'][:12]).max()),
                    hand_boundary_rms_rad=summary['continuity']['hand_boundary_jump_rms_rad'],
                    arm_boundary_rms_rad=summary['continuity']['arm_boundary_jump_rms_rad'],
                    clipped_joint_ticks=summary['clipped_joint_ticks'],clipping_max_rad=summary['clipping_max_rad'],
                    expert_precontact_actual_fk_rms_mm=summary['reference']['actual_expert_precontact_fk_mm'],
                    common_prefix_actual_fk_rms_mm=summary['reference']['actual_expert_fk_mm']))
    original_hashes={}
    for row in read(root/'analysis.json')['rows']:
        for p,digest in row['source_hashes'].items():original_hashes[p]=sha(p)==digest
    source={p:sha(CODE/p)==digest for p,digest in read(root/'source_freeze_ablations.json').items()}
    weights={m:sha(x['path'])==x['sha256'] for m,x in session['checkpoints'].items()}
    exits=[read(p) for p in (root/'stages').glob('*/exit.json')]
    monitor=read(root/'monitor_exit.json')
    integrity=dict(protected_files=len(original_hashes),protected_files_passed=all(original_hashes.values()),
        source_files=len(source),source_passed=all(source.values()),weights=weights,
        owned_stages_reaped=all(x['resources_released'] for x in exits),monitor_exit=monitor,
        changed_protected=[p for p,v in original_hashes.items() if not v],changed_source=[p for p,v in source.items() if not v])
    assert integrity['protected_files_passed'] and integrity['source_passed'] and all(weights.values())
    assert integrity['owned_stages_reaped']
    write(root/'final_integrity.json',integrity);write(PUBLIC/'final_integrity.json',integrity)
    write(root/'ablation_comparisons.json',comparisons);write(PUBLIC/'ablation_comparisons.json',comparisons)
    # Additional evidence from saved plans; no new network inference.
    gap=read(root/'offline_closed_gap.json');actor=read(root/'initial_actor_difference.json')
    write(PUBLIC/'initial_actor_difference.json',actor)
    lines=['','## 完成后的实测对照','',
        '|模型/初态|版本|结果|原生lift|首次末端偏离tick|手指边界RMS rad|专家接触前实际FK RMS mm|',
        '|---|---|---|---:|---:|---:|---:|']
    for c in comparisons:
        result='success' if c['success'] else '/'.join(k for k,v in c['reasons'].items() if v)
        lines.append(f"|{c['model']}/{c['ordinal']}|{c['variant']}|{result}|{int(c['native_lift'])}|{c['first_ee_divergence_tick']}|{c['hand_boundary_rms_rad']:.5f}|{c['expert_precontact_actual_fk_rms_mm']:.2f}|")
    gains=[]
    for model in ('StateDP','ImageDP'):
        r=[c for c in comparisons if c['model']==model]
        for ordinal in (0,4):
            pair={c['variant']:c for c in r if c['ordinal']==ordinal}
            if 'deterministic_precision_only' in pair and 'deterministic_precision_K4' in pair:
                a,b=pair['deterministic_precision_only'],pair['deterministic_precision_K4']
                assert b['causal_pair_valid']
                gains.append(dict(model=model,ordinal=ordinal,boundary_change_rad=b['hand_boundary_rms_rad']-a['hand_boundary_rms_rad'],
                    success_gain=int(b['success'])-int(a['success']),lift_gain=int(b['native_lift'])-int(a['native_lift']),
                    first_divergence_before=a['first_ee_divergence_tick'],first_divergence_after=b['first_ee_divergence_tick']))
    write(PUBLIC/'single_factor_effects.json',gains)
    if len(gains)==2:
        lowered=sum(x['boundary_change_rad']<0 for x in gains);successes=sum(c['success'] for c in comparisons if c['variant']!='original')
        lines += ['',f'确定性K4在2对中有{lowered}对降低全episode手指边界RMS；{len(valid)}条新诊断完整成功共{successes}次。',
            '单因素差值和前12tick逐值验证已保存；小样本只用于机制诊断，不代表总体成功率提升。']
    lines += ['','## 为什么绝对位置离线误差不能代表抓取能力','',
        '下表使用相同原始示范、相同顺序DDIM seed以及共同tick范围；离线列读取专家历史，闭环列读取自己的历史。单位rad。',
        '|模型/初态|首步Arm target误差|专家首步Arm变化|离线Arm公共前缀RMSE|自主Arm公共前缀RMSE|离线Hand RMSE|自主Hand RMSE|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in gap['rows']:
        x=r['initial_and_common']
        lines.append(f"|{r['model']}/{r['ordinal']}|{x['arm_first_target_error_rad']:.5f}|{x['arm_expert_first_target_change_rad']:.5f}|{x['arm_offline_common_error_rad']:.5f}|{x['arm_autonomous_common_error_rad']:.5f}|{x['hand_offline_common_error_rad']:.5f}|{x['hand_autonomous_common_error_rad']:.5f}|")
    lines += ['',
        '首步Arm误差约为专家本步动作变化的4.5–7.7倍；绝对位置RMSE看起来小，仍可相对精细接近运动很大。',
        '专家条件下移动anchor的12tick Arm方向cosine：State train/validation约0.876/0.746，Image约0.830/0.638；幅值比约0.93–1.04。方向和时间结构仍有误差，不是单纯把手指幅度整体缩小就能证明解决。',
        '初始q、proprio、目标和物体尺度一致，但原4lane示范与新独立仿真Actor噪声不同：qd最大差0.269/0.482rad/s，noisy keypoint约0.02m。干净palm/指尖仅浮点差。',
        '这解释了为什么“同物理初态”不等于“与原始示范同网络输入”。State使用这条原生带噪Actor分支，Image不使用Actor；不能把Image失败归因于Actor差异。在线专家与模型的配对初态/观测噪声流仍完全相同。',
        '不从聚合RMSE相减声称误差可因果分解；来源四lane隐藏状态限制仍保留。',
        '证据：[离线/闭环同口径数据](evidence/grasp_diagnosis_20261104_01/offline_closed_gap.json)、[完整性](evidence/grasp_diagnosis_20261104_01/final_integrity.json)、[单因素结果](evidence/grasp_diagnosis_20261104_01/ablation_comparisons.json)。','',
        '下一轮最有价值的是固定前12tick、接触前的小偏差恢复诊断：比较新鲜成功专家观测下的学生动作与自主偏离后同一专家的shadow纠正方向。若确认只在离轨状态无恢复能力，再提出有界roll-in/纠正数据训练；若专家输入下首段已明显错，先定位移动阶段监督与静止/位置偏置。',
        '**本轮没有证据要求立即更换网络或再次从头训练两组10000；也没有证明新增某项loss即可解决。**',
        f"所有视频索引：`{root/'VIDEOS.md'}`。修订计划9条，实际有效{len(valid)}条，启动{len(attempts)}次，软件缺失{sum('exit_code' in x and not x.get('valid') for x in attempts)}次。",
        'SDK fast exit0与正常SDK完整清理仍明确区分；无共享SDK改动。各worker资源回收和旧权重/源码/原始结果SHA复核已完成。','']
    with (CODE/'research_docs/GRASP_FAILURE_DIAGNOSIS.md').open('a') as f:f.write('\n'.join(lines))
    write(root/'completion.json',dict(state='completed' if len(valid)==9 else 'partial',valid=len(valid),planned=9,
        started=len(attempts),training_updates=0,integrity_passed=True,utc=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    with torch.no_grad():finish(a.root)
