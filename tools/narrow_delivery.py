"""Publish only reviewed, generated lightweight results after workers exit."""
import hashlib
import math
from pathlib import Path
import subprocess

from tools.runtime import CODE, read, write, sha

MARKER='NARROW_FOLLOWUP'
DOCS=['research_docs/STATE_DP_RESULTS.md','research_docs/SLOW_EXPERT_RESULTS.md',
      'research_docs/SESSION_HANDOFF.md','research_docs/CHANGELOG.md','research_docs/CLEANUP_AUDIT.md']
ARTIFACTS=['research_docs/evidence/narrow_followup.json','research_docs/evidence/narrow_followup_curves.png']


def outside(text):
    start=f'<!-- {MARKER}_START -->';end=f'<!-- {MARKER}_END -->'
    if start in text:
        a,rest=text.split(start,1);_,b=rest.split(end,1);text=a+b
    return hashlib.sha256(text.encode()).hexdigest()


def replace(text, block):
    start=f'<!-- {MARKER}_START -->';end=f'<!-- {MARKER}_END -->'
    generated=start+'\n'+block+'\n'+end
    if start in text:
        a,rest=text.split(start,1);_,b=rest.split(end,1);return a+generated+b
    title,sep,rest=text.partition('\n')
    return title+'\n\n'+generated+'\n'+rest


def wilson(success,total):
    if not total:return None
    p=success/total;z=1.959963984540054;den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [mid-half,mid+half]


def git(*args):
    return subprocess.check_output(['git',*args],cwd=CODE,text=True,stderr=subprocess.STDOUT,timeout=120).strip()


def request(root):
    write(root/'publication_request.json',dict(expected_head=git('rev-parse','HEAD'),
        branch='feature/narrow-state-data',outside_sha256={p:outside((CODE/p).read_text()) for p in DOCS},
        source_sha256=sha(Path(__file__)),publish=True))


def finish(root):
    request=read(root/'publication_request.json');result=read(root/'RESULTS.json');s=read(root/'session.json')
    assert read(root/'status.json')['owned_workers_exited']
    assert sha(Path(__file__))==request['source_sha256']
    assert git('rev-parse','HEAD')==request['expected_head'],'Intervening source commit; retain local results for review'
    assert not git('diff','--cached','--name-only'),'Unrelated staged work'
    assert not git('status','--porcelain'),'Working tree changed; retain results without overwriting'
    for p in DOCS:assert outside((CODE/p).read_text())==request['outside_sha256'][p],p
    public=dict(complete=result['complete'],source_commit=s['source_commit'],
        parent_source_commit=read(Path(s['parent'])/'launch_revision.json')['running_commit'],
        decision=result['decision'],original_closed_loop=result['original_closed_loop'],
        original_failure_stages=result['original_failure_stages'],original_curve=result['original_curve'],
        extension=result['extension'],extension_evaluation=result['extension_evaluation'],expansion=result['expansion'],audit=result['audit'],
        final_evaluation=result['final_evaluation'],next_stage=result['next_stage'],
        models=[dict(name=m['name'],sha256=m['sha256']) for m in result['models']],
        resources=read(root/'monitor_exit.json')['summary'],cleanup=dict(deleted_bytes=0,
            reason='All formal data, best/last recovery states and comparison caches still have active scientific references'))
    if public['extension']:public['extension']={k:v for k,v in public['extension'].items() if k!='best_checkpoint'}
    if public['expansion']:public['expansion']={k:v for k,v in public['expansion'].items() if k!='state_data'}
    offline=read(root/'offline_comparison.json')['models']
    public['offline']={name:{cohort:{k:m[cohort][k] for k in ('selection_mse','arm_rmse','hand_rmse','ee_distance_rmse_mm')}
        for cohort in ('train8','validation')} for name,m in offline.items()}
    from visuotactile.narrow_decision import mean
    for name,m in offline.items():
        public['offline'][name]['loss']={key:m[key] for key in ('train8_loss','validation_loss')}
        public['offline'][name]['early']={cohort:{key:mean(r[key] for r in rr) for key in
            ('arm_rmse_rad','hold_arm_rmse_rad','fk_mm','direction_cos','amplitude_ratio')} for cohort,rr in m['early'].items()}
    c=public['original_curve'];old=public['original_closed_loop']
    lines=['## 窄域200条后续：已完成并停止','',
        f"原200条：train {old['train']['successes']}/{old['train']['valid']}，unseen {old['unseen']['successes']}/{old['unseen']['valid']}；原unseen12已用于科研决策，属于开发诊断。",
        f"末段验证改善{100*c['validation_gain']:.2f}%（{c['span_updates']}更新），train8改善{100*c['train_gain']:.2f}%，验证/train8误差比{c['gap']:.2f}。首块未见动作方向cos {c['early_direction']:.3f}，幅值比{c['early_amplitude']:.3f}。",
        f"调度结论：{public['decision']['action']}；{public['decision']['reason']}。阈值仅决定有限研究支出，不是成功验收线或数据不足的因果证明。",
        f"续训：{'执行一次完整状态恢复的有限续训' if public['extension'] else '未追加'}；扩采：{str(public['expansion']['new_successes'])+'条真实轨迹、'+str(public['expansion']['new_families'])+'个新家族，总量'+str(public['expansion']['successes']) if public['expansion'] else '未执行'}。",
        '', '|模型|训练初态完整成功|新独立初态完整成功|95% Wilson区间|','|---|---:|---:|---|']
    final=public['final_evaluation']
    if final:
        for name,cohorts in final['models'].items():
            t,u=cohorts['train'],cohorts['unseen'];ci=wilson(u['successes'],u['valid'])
            interval=f'{ci[0]:.1%}–{ci[1]:.1%}' if ci else 'N/A'
            lines.append(f"|{name}|{t['successes']}/{t['valid']}|{u['successes']}/{u['valid']}|{interval}|")
            public['final_evaluation']['models'][name]['unseen']['wilson95']=ci
        lines+=['','最终初态在本轮开始前固定，仅使用一次并且不触发再训练；动作前非法初态单列，不按任务结果补种子。小样本差异不证明普遍提升。',
                '失败阶段：'+str({name:{k:v['failures'] for k,v in cc.items()} for name,cc in final['models'].items()}),
                '阶段分类来自观测和原生事件，不是因果证明；稳定抓持没有原生离散判据，未触发native_lift不等于物体从未抬起。']
    else:lines+=['|无新增最终测试|—|—|证据不足，安全停止|']
    lines+=['',public['next_stage'],'',
        '原400及当前200条数据、原best/last、新模型和失败证据全部保留。本次未删除仍可比较或恢复的checkpoint；释放0bytes。',
        '[轻量完整统计](evidence/narrow_followup.json) · [训练曲线](evidence/narrow_followup_curves.png)。',
        'GPU为共享整卡读数；阶段、吞吐、等待及资源实测见机器统计，不声称每进程GPU配额。']
    block='\n'.join(lines)
    for p in DOCS:
        body=block if p.endswith('STATE_DP_RESULTS.md') else '\n'.join([lines[0],'',lines[2],lines[4],lines[5],'',
            public['next_stage'],'详见[统一State结果](STATE_DP_RESULTS.md)。原截止与原统计未修改；新阶段固定8h以内，无下一轮扩采。'])
        (CODE/p).write_text(replace((CODE/p).read_text(),body))
    write(CODE/ARTIFACTS[0],public)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    directories=[('200',Path(s['parent'])/'training'),('200 continuation',root/'continuation'),('expanded fresh',root/'expanded_training')]
    for label,d in directories:
        if not (d/'validation.jsonl').exists():continue
        rr=[json.loads(line) for line in (d/'validation.jsonl').read_text().splitlines()]
        for ax,key in zip(axes,['selection_mse','arm_rmse','ee_distance_rmse_mm']):
            ax.plot([r['update'] for r in rr],[r['metrics'][key] for r in rr],label=label+' val')
            ax.plot([r['update'] for r in rr],[r['train8'][key] for r in rr],'--',alpha=.7,label=label+' train8')
            ax.set_title(key);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.3)
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(CODE/ARTIFACTS[1],dpi=140);plt.close(fig)
    for p in DOCS+ARTIFACTS:assert (CODE/p).stat().st_size<2*1024**2
    git('add','--',*(DOCS+ARTIFACTS));git('diff','--cached','--check')
    git('commit','-m','Report bounded narrow StateDP continuation and data comparison')
    if request['publish']:
        assert git('remote','get-url','origin')=='https://github.com/mao455/bcrl_visualTactile.git'
        git('push','origin','HEAD:'+request['branch'])
    write(root/'publication.json',dict(state='published' if request['publish'] else 'committed',commit=git('rev-parse','HEAD')))
