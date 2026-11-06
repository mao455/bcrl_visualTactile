"""Update existing research documents from actual bounded A/B/C receipts."""
from pathlib import Path
import argparse
import json

from tools.runtime import CODE,read,write,sha
from tools import narrow_delivery as publication

publication.MARKER='STATE_MECHANISM'
DOCS=['research_docs/STATE_DP_RESULTS.md','research_docs/SLOW_EXPERT_RESULTS.md','research_docs/SESSION_HANDOFF.md','research_docs/CHANGELOG.md']
ARTIFACT='research_docs/evidence/state_mechanism.json'


def request(root):
    write(root/'publication_request.json',dict(expected_head=publication.git('rev-parse','HEAD'),
        outside_sha256={p:publication.outside((CODE/p).read_text()) for p in DOCS},source_sha256=sha(Path(__file__))))


def synthesis(root):
    if not (root/'closed_loop_result.json').exists():return 'A/B/C尚未全部完成，不作机制结论；以已保存门控和错误为准。'
    c=read(root/'closed_loop_result.json')['groups'];history=read(root/'tiny_training/fit_history.json')
    best=read(root/'tiny_training/finished.json')['best_update'];row=next(x for x in history if x['update']==best)
    fitted=all(x['metrics']['all']['arm_rmse']<.003 and x['metrics']['all']['hand_rmse']<.02 and x['metrics']['all']['fk_mm']<3 for x in row['seeds'])
    if not fitted:
        return '主要待解瓶颈是极小集的监督拟合尚未达到预先记录的精度；不能归因数据量不足。下一轮只优先定位扩散训练目标与实际DDIM预测的拟合差距，先不改网络/控制或扩采。'
    if not c['B_tiny']['successes']:
        return '极小集离线拟合通过但对应初态仍未成功；精准目标回放接口检查已通过，优先研究执行后状态偏离teacher-forced训练轨迹这一机制。下一轮只做有限闭环分布偏移定位；本轮不自动DAgger或改方法。噪声/块跳变的因果归属仍受小样本限制。'
    if c['C4']['successes']>c['C0']['successes']:
        return '极小集闭环成功证明该架构具有记忆并执行少量示范的能力，不是泛化成功。overlap4小样本成功数较多，下一轮唯一优先方向为动作块交接一致性的独立验证；尚不足以改变正式overlap0默认或宣称因果改善。'
    return '极小集闭环成功证明基础执行能力，但275条正式模型及overlap对照未建立可靠闭环改善。下一轮唯一优先方向为同初态多速度示范的时序/条件歧义诊断，而非扩大数据数量；此为待验证机制，不是已证实原因。'


def snapshot(root):
    s=read(root/'session.json');d=read(root/'data_audit.json');status=read(root/'status.json')
    result=dict(status=status,budget={k:s[k] for k in ('started','deadline','maximum_kit_starts','tiny_max_updates','tiny_training_seconds')},
        source_commit=s['source_commit'],data_audit=d,plan=read(root/'plan.json'),interpretation=synthesis(root))
    for name in ('gradient_gate','oracle_result','oracle_results','stochasticity','closed_loop_result','closed_loop_results','monitor_exit','upstream_reference'):
        if (root/(name+'.json')).exists():result[name]=read(root/(name+'.json'))
    if (root/'tiny_training/finished.json').exists():result['tiny_finished']=read(root/'tiny_training/finished.json')
    if (root/'tiny_training/progress.json').exists():result['tiny_progress']=read(root/'tiny_training/progress.json')
    if (root/'tiny_training/fit_history.json').exists():result['tiny_fit_history']=read(root/'tiny_training/fit_history.json')
    if (root/'phases/diagnosis/budget.json').exists():result['runtime_source_commit']=read(root/'phases/diagnosis/budget.json')['source_commit']
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('path','directory','checkpoint','owner','worker','monitor')}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    write(CODE/ARTIFACT,clean(result))
    exp=d['exposure'];duration=d['groups']['all']['duration_s'];v=duration['quantiles']
    lines=['## 2026-11-06：StateDP基础能力、采样与动作块诊断','',
        f"实测 `{status['state']}` / `{status['stage']}`；独立root `state_mechanism_20261106_01`，固定4h含等待，最多15次Kit轨迹启动、1条训练示范的独立随机模型≤4000更新/90min。正式数据/权重保持，A/B/C不改变H24/12/29D/损失/历史/物理。",
        f"只读审查275条/64,663tick：全体均值{duration['mean']:.3f}s、中位{v['median']:.3f}s、p10–p90 {v['p10']:.3f}–{v['p90']:.3f}s，范围{v['min']:.3f}–{v['max']:.3f}s。新增75均值4.676s、中位4.4s；尚不符合多数5–7s的设计预期，没有补等待或改时间戳。State-valid275、Multi-view-valid75、旧固定视角无效200；触觉数值有限及有效标记通过，不等于硬件校准。所有成功标签只在末transition出现，275条原始数值与缓存完全相等，q_after[t]=q_before[t+1]。",
        '92个真实初态家族，68/12/12训练/验证/测试；速度版本不跨split。位置±25mm、相对位置±35mm、姿态20°球、关节初态及速度仍有变化；不是唯一起点。原200/201和扩充75/76是已准入家族上的采集结果，不代表整个连续分布100%成功。',
        '', '|模型|训练episode|实际batch分布|anchor数|抽样总数|anchor等效epoch|独立覆盖|','|---|---:|---|---:|---:|---:|---:|']
    for label,x in exp.items():lines.append(f"|{label}|{x['train_episodes']}|{x['batch_distribution']}|{x['anchors']}|{x['draws']}|{x['anchor_equivalent_epochs']:.3f}|{x['unique_anchors']}|")
    lines+=['','正式一次更新：最多64个不同episode各1个12tick anchor，microbatch4累积，所有microbatch之后才一次clip/AdamW/EMA；207条训练数据每轮64/64/64/15，平均51.75。输入Actor141、robot178、geometry85；从episode开头用当前权重重算历史，96tick边界前detach，非清零；条件[B,512]，目标[B,24,29]，mask[B,24]。不足24的目标零填充且无效；base/FK按有效目标数加权，temporal按有效相邻对数加权。',
        'episode均衡使长轨迹的单anchor曝光更少，家族权重与其速度版本数成正比；275 best各anchor访问17–103次。全部anchor已覆盖，但重复曝光不是独立新样本。阶段曝光及家族逐项权重见机器统计；阶段名称是接触/高度/native子目标代理，不能直接当作稳定抓取标签。',
        'Stanford上游[SequenceSampler](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/sampler.py)枚举窗口起点；均匀遍历这些索引时长episode权重更大，边界按配置重复端点。我们的stride12/episode均衡和显式mask是不同采样合同，并不单凭差异认定错误。ReplayBuffer按episode_ends保留轨迹边界。',
        'B仅选family_066_repair_a055（300tick、5s、alpha.55，train），由接近5s及ID固定选择。诊断多窗口采样在25个anchor上洗牌，batch16/9、micro4，允许同episode多个不同anchor；正式采样器不改。保持275 train-only统计、专家RMS/embedding及Loss B，从随机参数/AdamW/EMA开始；不使用test。']
    if 'gradient_gate' in result:
        g=result['gradient_gate'];lines+=['',f"真实GPU更新审查：从275 last5000完整采样状态取下一批{g['effective_batch']}个episode，记录全部ID/anchor/shape/mask及累积梯度；诊断副本做1次micro累积更新和1次相同噪声的大batch参考更新。梯度相对L2差{g['gradient_relative_l2']:.3g}，参数更新最大差{g['parameter_update_max_difference']:.3g}，passed={g['passed']}。未改正式checkpoint。"]
    if 'oracle_result' in result:lines+=['','A：'+json.dumps(result['oracle_result'],ensure_ascii=False)]
    if 'tiny_finished' in result:lines+=['','B：'+json.dumps(result['tiny_finished'],ensure_ascii=False)]
    if 'closed_loop_result' in result:lines+=['','B/C闭环：'+json.dumps(result['closed_loop_result'],ensure_ascii=False)]
    lines+=['',result['interpretation'],'','[完整轻量统计、真实batch与阶段指标](evidence/state_mechanism.json)。原数据、checkpoint、数值轨迹和视频留本地。']
    for p in DOCS:
        body='\n'.join(lines if p.endswith('STATE_DP_RESULTS.md') else [lines[0],'',lines[2],lines[3],'',result['interpretation'],'','实际A/B/C、采样曝光及最新状态见[StateDP统一结果](STATE_DP_RESULTS.md)和[机器统计](evidence/state_mechanism.json)。'])
        (CODE/p).write_text(publication.replace((CODE/p).read_text(),body))


def finish(root):
    req=read(root/'publication_request.json');assert read(root/'status.json')['owned_workers_exited']
    assert publication.git('rev-parse','HEAD')==req['expected_head'] and not publication.git('status','--porcelain')
    assert sha(Path(__file__))==req['source_sha256']
    for p,h in req['outside_sha256'].items():assert publication.outside((CODE/p).read_text())==h
    snapshot(root);assert (CODE/ARTIFACT).stat().st_size<2*1024**2
    publication.git('add','--',*DOCS,ARTIFACT);publication.git('diff','--cached','--check')
    publication.git('commit','-m','Record bounded StateDP mechanism diagnosis results')
    assert publication.git('remote','get-url','origin')=='https://github.com/mao455/bcrl_visualTactile.git'
    publication.git('push','origin','HEAD:feature/narrow-state-data')
    write(root/'publication.json',dict(state='published',commit=publication.git('rev-parse','HEAD')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('request','snapshot','finish'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.mode](a.root.resolve())
