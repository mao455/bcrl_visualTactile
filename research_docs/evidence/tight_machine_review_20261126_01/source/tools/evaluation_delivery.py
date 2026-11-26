"""Finish a reviewed evaluation report after its sole owner exits; no GPU work.

Publication is restricted to generated light artifacts and an unchanged feature
HEAD. Any intervening human report edit or unrelated staged change leaves the
publication pending review, rather than committing somebody else's work.
"""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import subprocess
import time
import traceback

from tools.runtime import CODE,read,write,sha

RANDOM_START='<!-- RANDOM_EVALUATION_START -->'
RANDOM_END='<!-- RANDOM_EVALUATION_END -->'
START='<!-- EVALUATION_DELIVERY_START -->'
END='<!-- EVALUATION_DELIVERY_END -->'
REPORT=CODE/'research_docs/STATE_DP_RESULTS.md'
ALLOWED=['research_docs/STATE_DP_RESULTS.md','research_docs/evidence/state_dp/random_evaluation.json',
    'research_docs/evidence/state_dp/random_stages.png','research_docs/evidence/state_dp/random_delivery.json']


def outside_generated(text):
    for start,end in ((RANDOM_START,RANDOM_END),(START,END)):
        if start in text:
            before,rest=text.split(start,1);_,after=rest.split(end,1)
            text=before+after
    return '\n'.join(line.rstrip() for line in text.splitlines() if line.strip())


def report_hash(text):
    return hashlib.sha256(outside_generated(text).encode()).hexdigest()


def summarize(summary):
    rows=summary['trials'];assert len(rows)==summary['valid']
    groups={r['group']:r for r in summary['groups']}
    if summary['status']['state']=='completed':
        assert summary['valid']==200 and groups['historical']['valid']==groups['fresh']['valid']==100
    cases={};failures={}
    for cohort in ('train','validation','fresh'):
        selected=[r for r in rows if r['trial']['cohort']==cohort]
        assert len(selected)==groups[cohort]['valid']
        stage=Counter();reason=Counter()
        for row in selected:
            t=row['trial'];diagnostic=row['success_audit'].get('terminal_geometry',{})
            if t['success']:stage['native_success']+=1;continue
            reason['+'.join(k for k,v in t['reasons'].items() if v) or 'other']+=1
            if diagnostic.get('ever_satisfied'):label='terminal_geometry_without_ordered_success'
            elif t['native_subgoals']:label='after_first_subgoal'
            elif not t['contact']['object_contact']:label='no_object_contact'
            elif not t['contact'].get('stable_grasp_diagnostic'):label='contact_without_stable_grasp_diagnostic'
            else:label='stable_grasp_diagnostic_without_first_subgoal'
            stage[label]+=1
        cases[cohort]=dict(stage);failures[cohort]=dict(reason)
    fresh=groups['fresh'];train=groups['train'];validation=groups['validation']
    if fresh['valid']<100:
        decision='全新100尚未完成，证据不足；不自动训练Image或重训State。'
    elif fresh['success']>=10:
        decision='全新初态真实原生成功达到约10%或以上，可建议进入Image baseline；必须先解决严格live Image输入配对，不自动启动训练。'
    elif not fresh['success'] and not validation['success'] and train['success']:
        decision='当前成功仅见训练初态，尚未证明泛化；按失败分布定位，不自动重训。'
    else:
        decision='全新初态原生成功未达到预定约10%参考线；先分析失败分布与终点/顺序门控，不自动修改网络或重训。'
    return dict(state=summary['status']['state'],valid=summary['valid'],planned=200,
        pending=200-summary['valid'],groups=summary['groups'],failure_flags=failures,observed_failure_stages=cases,
        formal_attempts=summary.get('formal_attempts'),actual_starts=summary.get('actual_starts'),
        initialization_rejections=summary.get('initialization_rejections',0),
        decision=decision,stages_are_diagnostics_not_causal_proof=True,
        formal_software_missing=summary['software_missing'],smoke_software_missing=summary['smoke_software_missing'],
        acceptance_failed=summary['acceptance_failed'],shutdown_faults=summary['shutdown_faults'],
        prior_software_starts=summary.get('prior_software_starts',0),checkpoint=summary['checkpoint'],
        evaluation_source_commit=summary['source_commit'])


def finish(root,request):
    summary=read(root/'summary.json');value=summarize(summary)
    resources=root/'resources/resources.json'
    value['resources']=read(resources) if resources.exists() else None
    assert report_hash(REPORT.read_text())==request['outside_report_sha256'],'Report edited outside generated sections; publication requires review'
    lines=[START,'## 随机评估交付状态','',f"状态：**{value['state']}**；有效{value['valid']}/计划200，尚未获得有效结果{value['pending']}。",
        f"正式软件缺失{value['formal_software_missing']}，smoke/此前版本软件缺失{value['smoke_software_missing']}；验收失败{value['acceptance_failed']}，有效结果后的异常关闭{value['shutdown_faults']}。这些均不当作机器人成功或失败。",
        '', '|类别|原生失败类型及次数|观测到的失败阶段|','|---|---|---|']
    labels=dict(terminal_geometry_without_ordered_success='终点条件成立但顺序未完成',after_first_subgoal='首子目标之后',
        no_object_contact='无物体接触',contact_without_stable_grasp_diagnostic='接触但无稳定抓持诊断',
        stable_grasp_diagnostic_without_first_subgoal='稳定抓持诊断之后、首子目标之前',native_success='原生完整成功')
    for cohort in ('train','validation','fresh'):
        flags=', '.join(f'{k}: {n}' for k,n in value['failure_flags'][cohort].items()) or 'N/A'
        stages=', '.join(f'{labels[k]}: {n}' for k,n in value['observed_failure_stages'][cohort].items()) or 'N/A'
        lines.append(f'|{cohort}|{flags}|{stages}|')
    lines += ['',value['decision'],'失败阶段分类只是记录上的定位，不是网络误差/控制延迟的因果证明；终点诊断不替代原生顺序成功。',
        f"本地完整结果：`{root}`；[最终机器摘要](evidence/state_dp/random_delivery.json)。"]
    if value['resources']:
        r=value['resources']
        lines += [f"本版本资源窗口：整卡时间加权GPU均值{r['gpu_time_weighted_mean_percent']:.2f}%，最大完整60秒均值{r['max_complete_60s_mean_percent']:.2f}%；遥测缺失{r['missing_samples']}样本/{r['missing_seconds']:.1f}秒，最低主机可用RAM {r['min_ram_available_bytes']/2**30:.2f}GiB。整卡读数包含其他进程，不代表本项目独占使用量；不覆盖此前失败版本窗口。"]
    expert=root/'stages/smoke_expert/committed.json'
    if expert.exists():
        record=read(expert);video=record.get('retained_video')
        if video:lines += [f"[本轮原生RL专家完整成功视频]({video})；首次成功与终止tick均为{record['result']['trials'][0]['first_success_tick']}。"]
    if (root/'owner_error.json').exists():
        lines += [f"未完成位置：`{summary['status'].get('stage')}`；具体错误保存在`owner_error.json`及对应attempt日志，未执行测试没有计入分母。"]
    lines += [END,'']
    text=REPORT.read_text()
    if START in text:
        before,rest=text.split(START,1);_,after=rest.split(END,1);text=before+'\n'.join(lines)+after
    else:text+='\n'+'\n'.join(lines)
    REPORT.write_text(text);write(CODE/'research_docs/evidence/state_dp/random_delivery.json',value)
    if not request.get('publish'):return dict(state='completed_local',result=value)
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=CODE,text=True,stderr=subprocess.STDOUT,timeout=120).strip()
    assert git('branch','--show-current')=='feature/native-dp-training-v2'
    assert git('rev-parse','HEAD')==request['expected_head'],'Feature HEAD changed; do not publish unreviewed intervening commits'
    assert git('remote','get-url','origin')=='https://github.com/mao455/bcrl_visualTactile.git'
    assert not git('diff','--cached','--name-only'),'Other staged work must not be committed'
    paths=[p for p in ALLOWED if (CODE/p).exists()]
    git('add','--',*paths)
    if git('diff','--cached','--name-only'):git('commit','-m','Report measured State150 random evaluation and failure stages')
    commit=git('rev-parse','HEAD');git('push','origin','HEAD:feature/native-dp-training-v2')
    return dict(state='published',commit=commit,result=value)


def main(root):
    request=read(root/'delivery_request.json');deadline=read(root/'session.json')['deadline']+600
    try:
        assert sha(Path(__file__))==request['source_sha256'],'Delivery source changed'
        while True:
            state=read(root/'status.json')
            if (root/'owner_exit.json').exists() and state.get('pid') is None:break
            if time.time()>deadline:raise TimeoutError('Owner has not delivered a final state within its fixed deadline')
            write(root/'delivery_status.json',dict(state='waiting_owner',pid=os.getpid(),utc=time.time()))
            time.sleep(10)
        assert sha(Path(__file__))==request['source_sha256'],'Delivery source changed'
        write(root/'delivery_status.json',finish(root,request))
    except BaseException:
        write(root/'delivery_status.json',dict(state='blocked',error=traceback.format_exc(),utc=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();main(a.root.resolve())
