"""Final concise conclusions including a validation40 hold baseline; no new trials."""
import argparse
from pathlib import Path
import time

from tools.state_image_queue import CODE,read,write
from tools.grasp_diagnosis_analysis import BASE,PUBLIC


def review(root):
    session=read(root/'session.json')
    while not (root/'completion.json').exists():
        if time.time()>session['deadline']+180:raise RuntimeError('No completed diagnostic evidence')
        time.sleep(5)
    hold=read(root/'validation_hold_baseline.json');rows=read(root/'ablation_comparisons.json')
    attempts=read(root/'ablation_attempts.json');valid=[x for x in attempts if x.get('valid')]
    conclusions={}
    for model,job in [('StateDP','state'),('ImageDP','image')]:
        metrics=read(BASE/'state_image_20261104_02'/(job+'_train')/'metrics.json')
        final=next(x for x in metrics if x['update']==10000)['validation']
        reference={x['identity']:x for x in hold['rows']}
        conclusions[model]=dict(
            arm_worse_than_hold_episodes=sum(x['arm_rmse']>reference[x['identity']]['arm_rmse_rad'] for x in final['episodes']),
            hand_better_than_hold_episodes=sum(x['hand_rmse']<reference[x['identity']]['hand_rmse_rad'] for x in final['episodes']),
            validation_episodes=40,arm_model=final['arm_rmse'],arm_hold=hold['hold']['arm_rmse_rad'],
            hand_model=final['hand_rmse'],hand_hold=hold['hold']['hand_rmse_rad'],
            arm_direction_cosine=final['motion_mean_cosine_amplitude'][0],arm_amplitude_ratio=final['motion_mean_cosine_amplitude'][1],
            precision_only_success=sum(x['success'] for x in rows if x['model']==model and x['variant']=='precision_only'),
            precision_K4_success=sum(x['success'] for x in rows if x['model']==model and x['variant']=='deterministic_precision_K4'),deterministic_baseline_success=sum(x['success'] for x in rows if x['model']==model and x['variant']=='deterministic_precision_only'))
    losses={}
    analysis=read(root/'analysis.json')
    for model in conclusions:
        selected=[x for x in analysis['rows'] if x['model']==model]
        losses[model]={}
        for side in ('student','expert'):
            events=[e for x in selected for e in x[side+'_contact_loss_windows']]
            losses[model][side]=dict(count=len(events),near_replan=sum(e['replan_in_window'] for e in events),
                with_clipping=sum((e['max_clipping_rad'] or 0)>1e-6 for e in events))
    result=dict(conclusions=conclusions,contact_loss_association=losses,
        confidence=dict(absolute_execution_interface_valid='high',early_precontact_motion_error='high',
            arm_offline_fit_insufficient_against_hold='high',autonomous_error_accumulation='medium',
            boundary_jump_unique_cause='unsupported',precision_rounding_main_cause='unsupported'),
        retraining_authorized=False,new_optimizer_updates=0,network_or_loss_changed=False,
        next_experiment='Fixed two-initial first12-tick current-target-anchored motion diagnostic, then matched fresh-expert observation vs off-trajectory expert shadow; only propose targeted continuation if movement fit/recovery deficit is demonstrated.',
        no_test40=True,valid_new_rollouts=len(valid),planned_new_rollouts=9)
    write(root/'review.json',result);write(PUBLIC/'review.json',result)
    lead=['## 审阅结论：Arm的离线误差尚未胜过零运动基线','',
        '**最有力的解释是接触前运动拟合已经不足，随后自主历史中的误差继续累积；尚不能把全部失败归因于某一个机制。**',
        '专家处理后target经过相同绝对接口2/2成功，已排除“该接口根本无法执行专家任务”的解释。原两模型正式结果仍各0/12。','',
        '在完全相同validation40、8670个有效tick和first12口径下：','',
        '|指标|保持当前target|StateDP-10000|ImageDP-10000|','|---|---:|---:|---:|',
        f"|Arm RMSE rad|{hold['hold']['arm_rmse_rad']:.5f}|{conclusions['StateDP']['arm_model']:.5f}|{conclusions['ImageDP']['arm_model']:.5f}|",
        f"|Hand RMSE rad|{hold['hold']['hand_rmse_rad']:.5f}|{conclusions['StateDP']['hand_model']:.5f}|{conclusions['ImageDP']['hand_model']:.5f}|",
        f"|移动Arm方向cosine|不定义|{conclusions['StateDP']['arm_direction_cosine']:.3f}|{conclusions['ImageDP']['arm_direction_cosine']:.3f}|",
        f"|移动Arm幅值比|0|{conclusions['StateDP']['arm_amplitude_ratio']:.3f}|{conclusions['ImageDP']['arm_amplitude_ratio']:.3f}|",'',
        f"Arm误差比保持target更大的验证轨迹：State {conclusions['StateDP']['arm_worse_than_hold_episodes']}/40，Image {conclusions['ImageDP']['arm_worse_than_hold_episodes']}/40。Hand优于保持target：State {conclusions['StateDP']['hand_better_than_hold_episodes']}/40，Image {conclusions['ImageDP']['hand_better_than_hold_episodes']}/40。",
        '这只是同口径离线零运动参照，不声称保持target的机器人能完成任务。复用现有模型验证结果，未新增推理或训练。',
        '不能只看0.02rad绝对数值说“已经拟合得很好”；Arm相对本任务的精细运动尺度仍不够，Hand能降低平均误差也不保证正确空间构型下的稳定抓取。','',
        '### 最小修复与消融决定','']
    for model,c in conclusions.items():
        lead.append(f"- {model}：仅精度修复 {c['precision_only_success']}/2 完整成功；确定性K0 {c['deterministic_baseline_success']}/1、K4 {c['precision_K4_success']}/1。具体有效覆盖率和软件故障见下表，缺失不能算失败。")
    if len(valid)==9 and not any(x['trial']['success'] for x in valid):
        lead += ['','9条新诊断均无完整成功，其中2条K4前缀不匹配，单列为非严格因果结果。舍入修复已通过严格数值合同，但未解决闭环；K4即使降低跳变，也没有恢复稳定任务能力。',
            '因此不把这一小数值修复或K4默认推广为“抓取修复”，旧默认和权重继续保留。']
    effect=next(x for x in read(PUBLIC/'single_factor_effects.json') if x['model']=='StateDP')
    lead += ['', f"**严格消融验收：StateDP PASS，ImageDP FAIL。** State确定性K0/K4前12tick六类字段逐值相同；共同{effect['common_ticks']}tick内手指边界RMS从{effect['hand_boundary_before_rad']:.5f}降至{effect['hand_boundary_after_rad']:.5f}rad，235tick fall变601tick timeout，仍无首子目标或完整成功。",
        'Image的固定输入独立进程前向/DINO检查逐值相同，但真实Kit首计划已差0.0032769rad（早于K4）；初始Actor/物理导出一致。原始live RGB-D/在线DINO/condition没有保存，无法唯一定位到渲染或网络运行期计算。',
        '保留Image原生timeout结果，不把它作为K4收益。停止新增Kit，不用剩余单次启动去凑严格一对。Image下一轮优先记录并逐值对照原始图像、DINO、condition及随机流；不得直接用本次未匹配结果调参。']
    lead += ['', '持续接触丢失靠近重规划（±3tick）的次数：State 5/10，Image 4/7，成功专家8/14；约7/12的控制tick天然处于这种邻域。仅凭时间邻近无法证明因果。',
        '模型接触丢失窗口均有部分joint-limit clipping，但这是常见现象；专家比较的是post-limit target，不能把它的零额外clipping当作未发生上游限位的证据。',
        '**是否改网络/损失：当前均不改，也不重训。** 下一轮最有价值的是固定前12tick、只改变当前target锚定/动作增量连续性的诊断，并比较匹配新鲜专家观测下的学生动作与离轨后的专家shadow纠正。',
        '若证实专家输入下仍无法胜过移动基线，优先提出相对动作/移动阶段监督的有界对照；若主要是离轨后无恢复能力，再提出roll-in纠正数据。两者都不等于授权从头重训两组10000。',
        '[审阅机器证据](evidence/grasp_diagnosis_20261104_01/review.json) · [validation40保持target基线](evidence/grasp_diagnosis_20261104_01/validation_hold_baseline.json)。','',
        '---','']
    path=CODE/'research_docs/GRASP_FAILURE_DIAGNOSIS.md';text=path.read_text();heading,body=text.split('\n',1)
    path.write_text(heading+'\n\n'+'\n'.join(lead)+body.lstrip())
    write(root/'review_complete.json',dict(passed=True,utc=time.time(),valid=len(valid)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();review(a.root)
