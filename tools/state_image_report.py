"""Small public reports; large training/simulator artifacts remain local."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

from tools.state_image_queue import read,write,CODE

PUBLIC=CODE/'research_docs/evidence/state_image_20261104_02'
REPORT=CODE/'research_docs/STATE_IMAGE_DP_IMPROVEMENT.md'
START='<!-- offline-v2-start -->';END='<!-- offline-v2-end -->'


def build(root):
    PUBLIC.mkdir(exist_ok=True,parents=True);status=read(root/'status.json');jobs={};csv_rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for job in ('state','image'):
        directory=root/(job+'_train');log=directory/'train.jsonl';rows=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
        metrics=read(directory/'metrics.json') if (directory/'metrics.json').exists() else []
        audit=read(root/(job+'_audit')/'result.json') if (root/(job+'_audit')/'result.json').exists() else None
        supervised=read(root/(job+'_supervised.json')) if (root/(job+'_supervised.json')).exists() else None
        jobs[job]=dict(updates=rows[-1]['update'] if rows else 0,audit=audit,supervised=supervised,
            last_training=rows[-1] if rows else None,milestones=[dict(update=r['update'],validation={k:v for k,v in r['validation'].items() if k!='episodes'}) for r in metrics])
        if rows:axes[0].plot([r['update'] for r in rows[::10]],[r['loss'] for r in rows[::10]],label=job)
        if metrics:
            axes[1].plot([r['update'] for r in metrics],[r['validation']['arm_rmse'] for r in metrics],'-o',label=job+' arm')
            axes[1].plot([r['update'] for r in metrics],[r['validation']['hand_rmse'] for r in metrics],'--o',label=job+' hand')
            axes[2].plot([r['update'] for r in metrics],[r['validation']['ee_distance_rmse_mm'] for r in metrics],'-o',label=job)
        for r in metrics:
            for joint in range(29):
                v=r['validation'];csv_rows.append(dict(job=job,updates=r['update'],joint_index=joint,
                    validation_episodes=len(v['episodes']),rmse_rad=v['joint_rmse_rad'][joint],mae_rad=v['joint_mae_rad'][joint],p95_rad=v['joint_p95_rad'][joint]))
    for ax,title in zip(axes,['Actual training loss','Validation RMSE (rad)','Validation target FK RMSE (mm)']):
        ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle('Offline v2: 50/100/200 use validation4; formal milestones use validation40',fontsize=10)
    fig.savefig(PUBLIC/'curves.png',dpi=120);plt.close(fig)
    with (PUBLIC/'joint_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['job','updates','joint_index','validation_episodes','rmse_rad','mae_rad','p95_rad'],lineterminator='\n');w.writeheader();w.writerows(csv_rows)
    loops=[];diagnostics=[]
    for path in sorted((root/'stages').glob('*/committed.json')):
        name=path.parent.name
        if not ('_loop_' in name or name.startswith('kit_')):continue
        record=read(path);row={k:record.get(k) for k in ('state','runtime_complete','sdk_graceful_close','shutdown_state','exit_code','timeout','resources_released')}
        row.update(stage=name,trials=(record.get('result') or {}).get('trials',[]))
        (diagnostics if name.startswith('kit_') else loops).append(row)
    summary=dict(run_id=root.name,status=status,jobs=jobs,diagnostics=diagnostics,closed_loop=loops,
        test_used=False,source_commit=read(root/'session.json')['source_commit'],updated=time.time())
    write(PUBLIC/'summary.json',summary);write(root/'public_summary.json',summary)
    section=f'''{START}
## 当前版本：离线训练与 Kit 生命周期解耦

Run `{root.name}`；真实状态 **{status['state']}**，State **{jobs['state']['updates']}** updates，
Image00 **{jobs['image']['updates']}** updates。原 `state_image_20261104_01` 保持 blocked，不改历史结果。
最新用户明确取消三次连续正常退出作为离线训练前提。本版本先完成离线两组全部里程碑，
保存每个 checkpoint，再串行执行各 milestone 的固定 train4+validation8 闭环；无训练与Kit并行。

|项目|新验收与实际实现|
|---|---|
|离线依赖|PyTorch/HDF5/既有DINO缓存/URDF FK；实际worker检查omni/carb/isaacsim/isaaclab未加载|
|模型|原State GRU及I-HistoryDP视觉/双GRU/U-Net；无新网络、无触觉/Correction；原29D post-delay标签|
|损失|Loss B + FK位置/旋转弦距/五指link位置/相邻差分；第一批4个train片段按损失和输出梯度校准|
|权重|每项辅助在所有校准片段不超过base数值及x0梯度范数的5%，合计上界20%；不看validation/test|
|连续性限制|chunk内专家动作差分，尚未实现跨重规划一致性，不能声称边界跳变已解决|
|监督|50、100、200各有验证、真实保存和新进程恢复；State={bool(jobs['state']['supervised'])}，Image={bool(jobs['image']['supervised'])}|
|计划|各从头1000/3000/6000/10000；固定warmup500/cosine10000；Image不依赖State任务成功率|
|Kit|六个有界诊断：最小应用full/fast、基础环境callback/disarmed、完整专家无诊断线程full/fast|
|评估|独立进程、固化checkpoint和初态；运行结果/SDK关闭/进程回收分开，异常退出不记正常关闭|
|恢复|冻结源码/数据/配置SHA；单owner锁；已完成阶段不重复；完整checkpoint边界恢复；原记录保留|

4项新loss CPU测试通过（mask、姿态端点与梯度、手指FK梯度、校准界）；实际模型前缀/在线一致性、
12步独立小样本拟合和视觉首尾缓存对照见机器证据，未通过前不启动对应正式模型。
短拟合更新不进入正式训练累计；正式模型保持初始化种子20261102，batch2段×96ticks及原归一化。
完整旧数据审计与Epsilon回归未重复。模型输入合同沿用原版本，新训练身份和loss_id单独冻结。

![离线训练曲线](evidence/state_image_20261104_02/curves.png)

[逐关节CSV](evidence/state_image_20261104_02/joint_metrics.csv) ·
[监督/离线/闭环机器证据](evidence/state_image_20261104_02/summary.json)。
200之前小验证为validation4；正式里程碑为validation40，不能把两种范围混作连续同口径趋势。
当前Kit诊断已记录{len(diagnostics)}项，闭环已记录{len(loops)}/96项；空缺不是任务失败。

NVIDIA 5.1文档区分完整close和跳过cleanup；fast_shutdown是启动配置，并非env.close成功证据。
本机5.1源码close实际走shutdown_and_release_framework，6.0公开发行说明明确记录该路径的GIL死锁修复。
这提供排查线索，不能证明本机SIGSEGV唯一根因；不升级或修改共享SDK。
来源：[5.1 SimulationApp](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html)、
[6.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html)、
[IsaacLab lifecycle issue4630](https://github.com/isaac-sim/IsaacLab/issues/4630)（公开问题报告，不等于本机复现）。

状态命令：`cat outputs/visuotactile/native_dp_v2/{root.name}/status.json`。
正式训练checkpoint完整保留本地；只发布代码、Markdown、小CSV/PNG/JSON。
{END}

'''
    previous=REPORT.read_text()
    if START in previous:
        a=previous.index(START);b=previous.index(END)+len(END);previous=previous[:a]+section+previous[b:].lstrip('\n')
    else:
        first,tail=previous.split('\n',1);previous=first+'\n\n'+section+'## 以下为旧 blocked 版本的保留证据\n\n'+tail.lstrip('\n')
    REPORT.write_text(previous)


def publish(root):
    allowed={str(REPORT.relative_to(CODE)),*(str(p.relative_to(CODE)) for p in PUBLIC.iterdir() if p.is_file())}
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CODE,text=True).strip()
    assert branch=='feature/native-dp-training-v2'
    changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=CODE,text=True).splitlines())
    untracked=set(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=CODE,text=True).splitlines())
    assert changed|untracked<=allowed,'Unreviewed changes outside generated report'
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=CODE,text=True).strip()
    paths=sorted(changed|untracked)
    assert all((CODE/p).stat().st_size<2_000_000 for p in paths)
    if paths:
        subprocess.run(['git','add',*paths],cwd=CODE,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=CODE,check=True)
        subprocess.run(['git','commit','-m','Update offline State/Image training and isolated evaluation results'],cwd=CODE,check=True)
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=CODE,check=True)
    write(root/'publication.json',dict(state='completed',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),utc=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args();build(a.root)
    if a.publish:publish(a.root)
