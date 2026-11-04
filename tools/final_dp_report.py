"""Durable per-rollout accounting. Missing software is never a policy failure."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import time

from tools.state_image_queue import CODE, read, write, sha

PUBLIC = CODE/'research_docs/evidence/final_dp_20261104_01'
REPORT = CODE/'EVALUATION_SUMMARY.md'


def aggregate(attempts, model):
    rows = [a for a in attempts if a['model'] == model]
    valid = [a for a in rows if a.get('valid')]
    # One frozen initial state may contribute at most one effective result.
    assert len({a['ordinal'] for a in valid}) == len(valid)
    successes = sum(a['trial']['success'] for a in valid)
    return dict(valid=len(valid), success=successes,
        rate=successes/len(valid) if valid else None, attempts=len(rows), planned=12,
        attempts_without_result=sum(a.get('finished') is not None and not a.get('valid') for a in rows),
        software_faults=sum(a.get('finished') is not None and (not a.get('valid') or a.get('exit_code') != 0) for a in rows),
        shutdown_faults=sum(a.get('valid',False) and a.get('exit_code') != 0 for a in rows),
        pending=12-len(valid),
        stages={k:sum(bool(a['trial'].get(k)) for a in valid) for k in ('native_lift','native_subgoals','native_alignment','success')},
        diagnostics=dict(approach=sum(a['trial'].get('approach_proxy',False) for a in valid),
            contact=sum(a['trial'].get('contact',{}).get('object_contact',False) for a in valid),
            stable_grasp=sum(a['trial'].get('contact',{}).get('stable_grasp_diagnostic',False) for a in valid)),
        reasons=dict(Counter(k for a in valid for k,v in a['trial']['reasons'].items() if v)))


def report(root, publish=False):
    PUBLIC.mkdir(parents=True, exist_ok=True)
    session, status = read(root/'session.json'), read(root/'status.json')
    attempts = read(root/'attempts.json') if (root/'attempts.json').exists() else []
    groups = {m:aggregate(attempts,m) for m in ('StateDP','ImageDP')}
    pairs = read(root/'pairs.json') if (root/'pairs.json').exists() else []
    write(PUBLIC/'summary.json',dict(status=status,models=groups,attempts=attempts,pairs=pairs,
        planned_model_episodes=24,maximum_starts=session['maximum_starts'],
        checkpoints=session['checkpoints'],source_commit=session['source_commit'],updated=time.time()))
    lines = ['# EVALUATION_SUMMARY', '',
        '| 模型 | 更新次数 | 成功次数/有效次数 | 成功率 | 软件故障 | 完成情况 |',
        '|---|---:|---:|---:|---:|---|']
    for m,g in groups.items():
        rate=f"{g['rate']:.1%}" if g['rate'] is not None else '未获得有效结果'
        lines.append(f"| {m} | 10000 | {g['success']}/{g['valid']} | {rate} | {g['software_faults']} | {g['valid']}/12；待完成 {g['pending']} |")
    lines += ['',f"运行 `{root.name}`；状态 **{status['state']}**，阶段 `{status.get('stage','pending')}`。",
        '每组预先固定 train4 + validation8，共24个模型episode；前各2次纳入这24条，不额外挑选。',
        f"最多 {session['maximum_starts']} 次启动（每个初态最多一次软件恢复，整个队列最多6次恢复）；不重试完整任务失败。",
        '成功率分母只含校验通过的完整合法episode；软件故障、未执行项和有效结果覆盖率另列。', '',
        '|模型|已启动尝试|未获得有效结果的已结束尝试|有效覆盖率|关闭异常但结果有效|',
        '|---|---:|---:|---:|---:|']
    for m,g in groups.items():
        lines.append(f"|{m}|{g['attempts']}|{g['attempts_without_result']}|{g['valid']}/12|{g['shutdown_faults']}|")
    lines += ['', '## 阶段与失败类型', '',
        '原环境没有接近/稳定抓取的离散成功标志，原生统计记为 N/A；下列前三项沿用此前已冻结的诊断，不能当作任务成功。',
        '接近诊断：最近指尖距离≤0.08m；接触：对象过滤的法向力>0.1N；稳定抓取诊断沿用 `native_evaluation.stable_grasp`，阈值未改。',
        '抬升来自原 `_lifted_object`，首子目标来自 `_successes>0`，alignment来自原 `_is_success`，完整成功来自 `termination_reasons.max_successes`。',
        '各列为各自事件发生次数，不强制构造单调漏斗。', '',
        '|模型|接近诊断|接触诊断|稳定抓取诊断|原生抬升|首子目标|alignment|完整成功|fall|dropped|timeout|其他|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,g in groups.items():
        d,s,r=g['diagnostics'],g['stages'],g['reasons']
        lines.append(f"|{m}|{d['approach']}|{d['contact']}|{d['stable_grasp']}|{s['native_lift']}|{s['native_subgoals']}|{s['native_alignment']}|{s['success']}|{r.get('fall',0)}|{r.get('dropped',0)}|{r.get('timeout',0)}|{r.get('hand_far',0)}|")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    names=['Approach*','Contact*','Stable grasp*','Native lift','First subgoal','Alignment','Task success']
    for i,(m,g) in enumerate(groups.items()):
        values=list(g['diagnostics'].values())+list(g['stages'].values())
        bars=ax.bar(np.arange(7)+(i-.5)*.38,[100*v/g['valid'] if g['valid'] else 0 for v in values],.38,label=f"{m}: n={g['valid']}/12")
        for b in bars[:3]:b.set_hatch('//')
    ax.set_xticks(range(7),names,rotation=15);ax.set_ylim(0,105);ax.set_ylabel('% of valid episodes')
    ax.set_title('Final 10000-update autonomous evaluation (* existing diagnostics)');ax.legend()
    fig.savefig(PUBLIC/'stage_rates.png',dpi=140);plt.close(fig)
    lines += ['', '![阶段对比](research_docs/evidence/final_dp_20261104_01/stage_rates.png)', '',
        '## 配对、控制与故障', '',
        f"已有 {sum(p['passed'] for p in pairs)}/12 对初态、环境合同与初始环境随机状态通过相等校验。",
        '共享同一冻结初态、事件随机流种子、120/60Hz、H24/commit12、DDIM16/eta0/clipFalse与原控制器。',
        '独立Kit每episode重新创建GRU；模型自主输出，专家只作为旧runner中的诊断shadow，未接管。',
        '保存的可观察初态相等不代表完整PhysX内部状态可序列化。test40未使用。',
        '启动SIGABRT、运行异常、关闭SIGSEGV/超时与SDK fast exit分别记录；fast exit0不等于正常SDK清理。',
        '本次仅隔离Kit应用路径并复制原persistent配置，不修改共享SDK、驱动或其他项目。',
        '依据：[NVIDIA Kit命令行](https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/Kernel/CommandLineOptions.html)。', '',
        '## 视频与逐次诊断', '',
        f"本地索引：`{root / 'VIDEOS.md'}`。每个有效模型episode均录制；MP4完整解码并与步数核对。",
        '专家参考复用已成功且有真实视频的固定train初态982403，独立列出，不混入模型成功率。',
        '[逐次结果与故障](research_docs/evidence/final_dp_20261104_01/summary.json)。']
    if (root/'failure_analysis.json').exists():
        detail=read(root/'failure_analysis.json');write(PUBLIC/'failure_analysis.json',detail)
        lines += ['', '[动作、末端和失败前状态](research_docs/evidence/final_dp_20261104_01/failure_analysis.json)。']
    if status.get('reason'):lines += ['', '当前停止原因：'+status['reason']]
    REPORT.write_text('\n'.join(lines)+'\n')
    videos=['# 实际视频索引','', '|模型|初态|结果|步数|视频|','|---|---|---|---:|---|']
    ref=session['expert_video']
    videos.append(f"|Expert reference|train / 982403|success|181|[视频]({ref})|")
    for a in attempts:
        if a.get('valid') and a.get('video'):
            t=a['trial'];outcome='success' if t['success'] else ','.join(k for k,v in t['reasons'].items() if v)
            videos.append(f"|{a['model']}-10000|{t['cohort']} / {t['seed']}|{outcome}|{t['steps']}|[视频]({a['video']['path']})|")
    (root/'VIDEOS.md').write_text('\n'.join(videos)+'\n')
    if publish:
        paths=['EVALUATION_SUMMARY.md','research_docs/evidence/final_dp_20261104_01']
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=CODE,text=True).strip():
            raise RuntimeError('Unrelated staged files: publication stopped')
        assert subprocess.check_output(['git','branch','--show-current'],cwd=CODE,text=True).strip()=='feature/native-dp-training-v2'
        for p in PUBLIC.iterdir():assert p.stat().st_size<2_000_000
        subprocess.run(['git','add','--',*paths],cwd=CODE,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=CODE).returncode:
            subprocess.run(['git','commit','-m','Report final StateDP/ImageDP autonomous evaluation'],cwd=CODE,check=True)
        subprocess.run(['git','push','origin','HEAD:feature/native-dp-training-v2'],cwd=CODE,check=True)
        write(root/'publication.json',dict(state='completed',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),utc=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    report(a.root,a.publish)
