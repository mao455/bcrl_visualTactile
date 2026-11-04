"""Small public continuation report; raw logs, tensors, videos and weights stay local."""
import argparse
import json
from pathlib import Path
import subprocess

from tools.state_image_queue import CODE,read,write

PUBLIC=CODE/'research_docs/evidence/anchor_continue_20261104_01'
REPORT=CODE/'research_docs/ANCHOR_BATCH_CONTINUATION.md'


def contact_count(trials):
    # Contact is a structured diagnostic; a non-empty dict is not a contact hit.
    return sum(bool(t['contact']['object_contact']) for t in trials)


def report(root):
    s=read(root/'session.json');status=read(root/'status.json');PUBLIC.mkdir(parents=True,exist_ok=True)
    payload=dict(status=status,source_commit=s['source_commit'],base_commit='a477622',new_training_recipe=dict(
        effective_batch=s['effective_batch'],microbatch=s['microbatch'],warmup=s['warmup'],lr_peak=s['lr_peak'],lr_floor=s['lr_floor'],
        milestones=s['milestones'],targets=s['targets'],anchor_population=s['anchor_population']),models={},
        test40_used=False,old_checkpoints_preserved=True)
    lines=['# H24 StateDP / ImageDP：多轨迹anchor续训','',
        f"当前状态：**{status['state']}**，阶段 `{status.get('stage','preparing')}`。本报告只记录已完成结果；后台任务启动不等于训练或闭环成功。",'',
        '原正式 StateDP-10000 / ImageDP-10000 均0/12。新任务从这两个checkpoint的原始训练权重、EMA、AdamW moments/step及RNG续接，不从随机权重训练。原模型、标签、拆分和结果保留。','',
        '## 已核实的旧预算与新预算','',
        '|项目|旧H24最终训练|本轮续训|','|---|---|---|',
        '|数据|train320 / validation40 / test40；832段|相同；不读test40|',
        '|anchor全集|5547个，原生tick0,12,…|相同，H24标签不偏移|',
        '|每次optimizer更新|2个最多96tick片段，平均13.3354 anchors|64条不同episode各1个anchor，microbatch4累积16次|',
        '|时间相关性|同片段内anchor相邻，未来标签重叠|同batch不重复episode；跨batch仍可能相关，不宣称64个统计独立样本|',
        '|旧10000更新累计|20000片段、133354 anchors；片段epoch24.03846，anchor epoch24.04074|新旧anchor抽样次数相加后除5547|',
        '|GRU|当前权重完整起点前缀，原96tick TBPTT边界|完全保留，选中anchor前不零状态截窗|',
        '|学习率|1e-4、warmup500、10000步cosine终点0|初值3e-6，200步升至3e-5，再按新增23918步总规划cosine至3e-6|',
        '|EMA/optimizer|AdamW(.95,.999),wd1e-6；EMA power.75|moments/计数继续，EMA计数不重置；仅scheduler与采样算法换版本|',
        '|固定核心|H24/执行12；29D绝对位置；Loss B+已校准几何/连续性项|架构、统计、损失权重和执行接口均不改|','',
        '|累计anchor等效epoch|新增optimizer updates|累计optimizer updates|','|---:|---:|---:|']
    for epoch in s['milestones']:
        n=s['targets']['state'][str(epoch)];lines.append(f'|{epoch}|{n}|{10000+n}|')
    lines += ['',
        '旧抽样计数由20261102采样RNG逐次重放20000个片段，并与两旧checkpoint的完整order/cursor/RNG逐值验证。不是用均值估算。',
        '新的episode deck每320次选择覆盖每轨迹一次，每轨迹自己的anchor deck耗尽后才重排；因此是episode均衡采样，长轨迹单个anchor的概率低于短轨迹，不是假装全局anchor均匀。',
        '日志同时保存每个anchor/episode计数、唯一anchor覆盖、原片段被访问次数、前缀重算tick、梯度tick。`episode_selection_passes`只表示选择次数/320，不是完整轨迹训练epoch；单anchor窗口不计完整96tick片段遍历。',
        'microbatch按整个64-anchor batch的有效目标数归一化base/FK/姿态/指尖，按有效邻接对数归一化temporal；尾段padding不能让短microbatch被过度加权。',
        '3e-5峰值为旧峰值的30%，不随batch倍数线性放大；3e-6下限避免继承零LR。此数值是保守续训选择，效果由验证和闭环检验，不能由官方默认证明最优。','',
        '## 与官方Diffusion Policy的相关对照','',
        '|项目|官方公开实现|本项目决定|','|---|---|---|',
        '|窗口batch|image YAML64；lowdim YAML256；DataLoader shuffle|先64跨episode anchor，不照搬256|',
        '|序列|SequenceSampler枚举episode内窗口，边界复制填充|保留原native anchor、H24有效mask和完整因果GRU前缀|',
        '|epoch/step|DataLoader遍历为epoch；scheduler按loader长度、epoch数、累积数计算|明确以5547 anchor抽样曝光计算累计epoch|',
        '|优化|AdamW1e-4、warmup500、cosine、EMA power.75|保留optimizer/EMA，续训调度用更低峰值|',
        '|归一化|具体dataset路径实现不同；不能假定自动train-only|继续原train320统计，不重算或掺validation|',
        '|图像/state区别|官方图像ResNet编码、batch64；低维向量batch256；任务/预算不同|保持已有DINO和State表示，不替换结构|','',
        '官方主来源：[image配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml)、[lowdim配置](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml)、[sampler](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/sampler.py)、[训练循环](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py)。',
        '官方累计epoch较大并不证明我们的150–300足够；数据量、任务、独立窗口定义与GRU计算量不同。','',
        '## 实测训练进度','',
        '|模型|累计updates|累计anchor epoch|最新LR|best验证Arm/Hand rad|阶段|','|---|---:|---:|---:|---|---|']
    curves={}
    for job in ('state','image'):
        directory=root/(job+'_train');progress=read(directory/'progress.json') if (directory/'progress.json').exists() else {}
        metrics=read(directory/'metrics.json') if (directory/'metrics.json').exists() else []
        best=read(directory/'best.json') if (directory/'best.json').exists() else {}
        old=read(Path(s['old_root'])/(job+'_train')/'metrics.json')[-1]
        best_metric=next((m for m in metrics if m['update']==best.get('update')),old)
        logged=[]
        if (directory/'train.jsonl').exists():
            for line in (directory/'train.jsonl').read_text().splitlines():
                try:logged.append(json.loads(line))
                except json.JSONDecodeError:pass
        latest=logged[-1] if logged else {};epoch=latest.get('cumulative_anchor_epochs',s['old_exposure'][job]['anchor_equivalent_epochs'])
        selected=[x for x in logged if x['continuation_update']%50==0]
        if latest and latest not in selected:selected.append(latest)
        thin=[{k:v for k,v in x.items() if k not in ('anchor_pairs','coverage')} for x in selected]
        item=dict(progress=progress,best=best,metrics=metrics,training_curve=thin,base_checkpoint=s['checkpoints'][job],
            old_anchor_exposures=s['old_exposure'][job]['anchors'],latest_coverage=latest.get('coverage'))
        if (root/(job+'_closed_loop.json')).exists():item['closed_loop']=read(root/(job+'_closed_loop.json'))
        if (root/(job+'_stop_decision.json')).exists():item['stop_decision']=read(root/(job+'_stop_decision.json'))
        payload['models'][job]=item;curves[job]=(old,metrics,thin)
        val=best_metric['validation'];lines.append(f"|{job}|{progress.get('update',10000)}|{epoch:.3f}|{latest.get('lr','未启动')}|{val['arm_rmse']:.6f} / {val['hand_rmse']:.6f}|{progress.get('state','pending')}|")
    lines+=['','原模型选择分数保留为best初值；只有完整validation40改善才替换best。每100新增更新保存可恢复last；阶段checkpoint独立保留。每1000新增更新及各里程碑固定seed验证。',
        '全量validation40以及固定train16报告损失、逐关节误差、FK位置/姿态/指尖、前12tick方向幅值和连续性。train16是固定训练子集，不能冒称全train320均值。',
        '达到150后，若最近两个完整验证间隔均不足0.5%改善则停止扩展；两次相对旧模型退化超过20%则选回best并停止，不删除更晚checkpoint。数值非有限时保留最后完整状态并阻塞，禁止无限重试。','',
        '![训练曲线](evidence/anchor_continue_20261104_01/curves.png)','',
        '## 闭环与Image严格配对','',
        'State先用固定validation ordinal4–7各一次，与原同初态结果比较；成功/首子目标出现或native lift计数提高时，再补完既定train4+validation8。小样本阶段进展不能声称显著成功率提升。',
        '只有State出现上述实际进展，才进入Image阶段；否则保留State结果并明确标记Image等待State证据。Image先用旧10000模型进行两个独立启动，记录原始RGB-D、DINO、GRU、condition、DDIM输入/随机流和前12tick命令，严格配对通过后才续训。',
        '原环境没有独立的“稳定抓持成功”离散判据，因此稳定抓持率标记N/A；接触、lift、首子目标、完整成功和失败原因按原定义单列，不能以接触力替代稳定抓取。','',
        '|模型|成功/有效|接触|lift|首子目标|软件缺失|当前证据|','|---|---:|---:|---:|---:|---:|---|']
    attempts=read(root/'attempts.json') if (root/'attempts.json').exists() else []
    videos=['# 续训闭环视频','', '|阶段|模型/初态|结果|视频|','|---|---|---|---|']
    for job in ('state','image'):
        records=[x for x in attempts if x['name'].startswith(job+'_evaluation_')]
        trials=[x['result']['trials'][0] for x in records if x.get('valid')]
        lines.append(f"|{job}|{sum(t['success'] for t in trials)}/{len(trials)}|{contact_count(trials)}|{sum(t['native_lift'] for t in trials)}|{sum(bool(t['native_subgoals']) for t in trials)}|{sum(not x.get('valid',False) and 'exit_code' in x for x in records)}|{'实测' if trials else '尚未执行'}|")
        for record in records:
            if not record.get('valid'):continue
            t=record['result']['trials'][0];video=Path(record['output'])/t['identity']/'rollout.mp4'
            videos.append(f"|{record['name']}|{job}/{t['seed']}|{t['reasons']}|[视频]({video})|")
    if (root/'image_pair_gate.json').exists():payload['image_pair_gate']=read(root/'image_pair_gate.json');lines+=['',f"Image配对：`{payload['image_pair_gate']['groups']}`；通过：{payload['image_pair_gate']['passed']}。"]
    lines += ['','## 可靠性、路径与结论边界','',
        f"运行根目录：`{root.resolve()}`；实际源码提交：`{s['source_commit']}`。新旧checkpoint路径/SHA与采样、指标见[机器报告](evidence/anchor_continue_20261104_01/summary.json)。",
        '独立systemd单owner、连续资源监测、串行GPU worker；仅完整checkpoint边界可恢复，最多两次恢复。Kit最多28次启动（含最多两次恢复），连续同阶段故障停止；无需三次正常退出门控。原始结果、权重和视频只留本地。',
        f"视频索引：`{root.resolve()/'VIDEOS.md'}`。退出状态和任务结果分别记录，SDK fast exit不称正常完整析构。",
        '**是否解决训练不足：尚须实测训练曲线与闭环共同回答。** 增加曝光并不自动证明充分收敛，离线误差下降也不等于抓取成功；不自动修改网络、动作表示或损失。','']
    (root/'VIDEOS.md').write_text('\n'.join(videos)+'\n')
    if (root/'supervision.json').exists():payload['supervision']=read(root/'supervision.json')
    if (root/'monitor_exit.json').exists():payload['monitor_exit']=read(root/'monitor_exit.json')
    write(PUBLIC/'summary.json',payload);REPORT.write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for job,(old,metrics,thin) in curves.items():
        epochs=[s['old_exposure'][job]['anchor_equivalent_epochs']]+[m['cumulative_anchor_epochs'] for m in metrics if m['validation_scope']=='validation40']
        rows=[old]+[m for m in metrics if m['validation_scope']=='validation40']
        for ax,key in zip(axes[0],('arm_rmse','hand_rmse')):
            ax.plot(epochs,[m['validation'][key] for m in rows],'o-',label=job);ax.set_ylabel(key+' rad');ax.set_xlabel('Cumulative anchor-equivalent epochs')
        if thin:
            axes[1,0].plot([x['cumulative_anchor_epochs'] for x in thin],[x['loss'] for x in thin],label=job)
            axes[1,1].plot([x['cumulative_anchor_epochs'] for x in thin],[x['lr'] for x in thin],label=job)
    for ax in axes.flat:
        ax.grid(alpha=.3)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    axes[1,0].set_ylabel('Actual training objective');axes[1,1].set_ylabel('Learning rate')
    for ax in axes[1]:ax.set_xlabel('Cumulative anchor-equivalent epochs')
    fig.tight_layout();fig.savefig(PUBLIC/'curves.png',dpi=130);plt.close(fig)


def publish(root):
    allowed={str(REPORT.relative_to(CODE))}
    paths=subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).splitlines()
    changed=[x[3:] for x in paths]
    if any(p not in allowed and not p.startswith(str(PUBLIC.relative_to(CODE))) for p in changed):
        write(root/'publication.json',dict(state='pending_review',reason='Other worktree changes preserved'));return
    if not changed:return
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CODE,text=True).strip()
    assert branch=='feature/native-dp-training-v2'
    subprocess.run(['git','add',str(REPORT.relative_to(CODE)),str(PUBLIC.relative_to(CODE))],cwd=CODE,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=CODE,check=True)
    subprocess.run(['git','commit','-m','Update measured anchor-continuation results'],cwd=CODE,check=True)
    subprocess.run(['git','push','origin','HEAD:feature/native-dp-training-v2'],cwd=CODE,check=True)
    write(root/'publication.json',dict(state='published',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args();report(a.root)
    if a.publish:publish(a.root)
