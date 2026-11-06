"""Small results and existing-document updates; never upload data or models."""
import argparse
from pathlib import Path
import json

from tools.runtime import CODE, read, write, sha
from tools import narrow_delivery as publication

publication.MARKER='NARROW_REPAIR'
DOCS=['research_docs/STATE_DP_RESULTS.md','research_docs/SLOW_EXPERT_RESULTS.md',
      'research_docs/SESSION_HANDOFF.md','research_docs/CHANGELOG.md']
ARTIFACTS=['research_docs/evidence/narrow_repair.json','research_docs/evidence/narrow_repair_curves.png']


def request(root):
    write(root/'repair_publication_request.json',dict(expected_head=publication.git('rev-parse','HEAD'),
        branch='feature/narrow-state-data',outside_sha256={p:publication.outside((CODE/p).read_text()) for p in DOCS},
        source_sha256=sha(Path(__file__))))


def snapshot(root):
    s=read(root/'session.json');public=dict(status=read(root/'status.json'),budget={k:s[k] for k in
        ('started','deadline','gate_deadline','collection_deadline','maximum_capture_attempts','maximum_evaluation_starts','maximum_training_updates')},
        source_commit=s.get('source_commit'),original400_and200_preserved=True)
    for name in ('gpu_parity','camera_probe','camera_visual_review','reviewed_preaction_failure','recheck_complete','decision','speed_decision',
        'collection_counts','DATA_READY','motion_diagnosis','offline_comparison','FINAL_REPORT','cleanup','final_review'):
        if (root/(name+'.json')).exists():public[name]=read(root/(name+'.json'))
    if (root/'reviewed_collection_resume.json').exists():
        receipt=read(root/'reviewed_collection_resume.json')
        public['reviewed_collection_resume']={k:v for k,v in receipt.items() if k!='input_sha256'}
        public['reviewed_collection_resume']['verified_input_files']=len(receipt['input_sha256'])
    public['phase_source_commits']={p.parent.name:read(p)['source_commit'] for p in (root/'phases').glob('*/budget.json')}
    if (root/'data_diagnosis.json').exists():
        public['data_diagnosis']={k:v for k,v in read(root/'data_diagnosis.json').items() if k not in ('rows','original_student_motion')}
    if (root/'camera_expert.json').exists():public['camera_acceptance']=read(root/'camera_expert.json')['visual']
    if (root/'recheck_results.json').exists():
        public['recheck']=[{k:r.get(k) for k in ('identity','split','precision_name','software_complete','legal','success','steps','native_subgoals','reasons','video')}
            for r in read(root/'recheck_results.json')]
    public['phase_resources']={p.parent.name:read(p)['summary'] for p in (root/'phases').glob('*/monitor_exit.json')}
    if (root/'expanded_training/validation.jsonl').exists():
        public['expanded_validation']=[json.loads(line) for line in (root/'expanded_training/validation.jsonl').read_text().splitlines()]
    # No machine paths, process IDs, large log payloads or checkpoint paths need
    # to be public to verify the scientific counts/thresholds/source identities.
    def sanitize(value):
        if isinstance(value,dict):return {k:sanitize(v) for k,v in value.items() if k not in
            ('owner','worker','monitor','directory','path','state_data','checkpoint','best_checkpoint','old_source_sha256')}
        if isinstance(value,list):return [sanitize(v) for v in value]
        return value
    public=sanitize(public);write(CODE/ARTIFACTS[0],public)
    state=public['status'];d=public.get('data_diagnosis',{});c=d.get('curve',{})
    lines=['## 2026-11-06：视觉与GPU历史修复、单次慢速数据对照','',
        f"实测状态 `{state['state']}` / `{state['stage']}`；独立run `narrow_repair_20261106_01`，10h固定总窗，旧run预算和失败保持。代码和排队不代表训练完成。",
        '白屏已定位：direct fresh入口遗漏固定相机配置，两路停在默认相同位姿，RGB恒定/有效深度0；同初态配置前后物理读数完全相同，配置后三视角有真实内容。首条199帧专家成功，起/中/尾画面目视通过，全帧RGB-D、时钟、固定位姿和mask检查通过；单lane持久Kit，不声明多lane已验证。',
        'GPU差异来自未统一的CuDNN TF32：旧默认condition最大2.52545e-4；关闭TF32后最大1.07288e-6、前12目标1.13249e-6rad、FK0.000229mm。原2e-5门限未放宽；完整序列/逐tick/训练prefix/reset均检查，无优化更新。',
        f"200条平均3.635s，中位3.2s；alpha≤.65仅38条、平均5.190s。best/last均5735且参数一致；末段验证改善{c.get('validation_gain',0)*100:.2f}%，train8改善{c.get('train_gain',0)*100:.2f}%，误差比{c.get('gap',0):.2f}，未见首块幅度比{c.get('early_amplitude',0):.2f}。属于验证平台与闭环偏移证据，不能称可靠收敛。",
        '旧200只标State-valid，Multi-view-valid为0；新增成功必须全部真实RGB-D/触觉重采并通过画面内容验收。不会用新图像拼接旧状态。Image/触觉未来公平比较只用相同有效模态子集。',
        '一次新编排错误把学生送入专家专用speed wrapper，在首次物理步前断言，0物理步/0完成rollout。原失败保留、计入40次学生启动预算；该legacy单元不重跑，正确环境继续其余7个固定复核单元。',
        '后续规则：先3个train家族原速/.55/.60/.65配对试采，按真实成功、接触运动和时长选择连续alpha区间；再32新家族预分24/4/4、每家族2速度版本，最多80采集尝试，至少60新增成功/28新增家族才训练。总量约260–276，未达门控则停止。',
        '新StateDP只随机初始化一次，H24/commit12/29D/5Hz/60Hz/损失/历史不变；按旧4000步平台曝光加500更新余量、实际anchor和吞吐定预算，≤10000更新/3h，validation选best与有界早停。最终旧/新各train4+预先冻结独立unseen12，最终结果不触发新训练。']
    if 'decision' in public:lines += ['','旧模型精度复核：'+str(public['decision'].get('recheck_counts',{})), '后续决定：'+public['decision']['action']+'。']
    if 'speed_decision' in public:lines += ['','真实慢速筛选：'+str(public['speed_decision'])]
    if 'collection_counts' in public:lines += ['','采集计数：'+str(public['collection_counts'])]
    continuation=''
    if 'reviewed_collection_resume' in public:
        continuation=('12:35首次生产停止：第50条原速专家真实掉落，末帧物体离开侧视野；81帧RGB-D和数值合同通过。'
            '成功示范的目标可见性检查错误地先于任务失败分类，导致少记1次尝试/掉落。复核49条成功文件哈希后，'
            '保留原错误和计数快照，补记为50次/49成功/1任务失败，失败轨迹不入训练。'
            '修复仅分离自然任务失败与传感器错误，成功数据门槛不变；8项CPU测试通过。'
            '本次复核后的接续入口只运行原冻结清单剩余26条，不重跑已执行条目或门控，不重置计数、17:12采集/21:12总截止；'
            '独立阶段reviewed_collection_completion仍只允许一次新训练。实际接续进度以上方机器状态和当前计数为准。')
        lines+=['',continuation]
    final=public.get('FINAL_REPORT')
    completed_summary=''
    if final:
        training=final.get('training')
        if training:
            selected=min(public['expanded_validation'],key=lambda row:row['metrics']['selection_mse'])
            outcomes='；'.join(f"{name}: train{cc['train']['successes']}/{cc['train']['valid']}，"
                f"unseen{cc['unseen']['successes']}/{cc['unseen']['valid']}有效、{cc['unseen']['initial_rejections']}次初态拒绝"
                for name,cc in final['final_evaluation']['models'].items())
            completed_summary=(f"已完成一次扩充：{final['expansion']['new_successes']}条新成功、{final['expansion']['new_families']}个新家族，合计{final['expansion']['state_valid']}条State-valid；"
                f"独立随机模型完成{training['updates']}更新，因{training['stop_reason']}结束，validation选第{selected['update']}步。"
                f"最终完整成功：{outcomes}。这轮有限对照已结束，不自动再训练/扩采或启动Image/触觉训练。")
            lines+=['',completed_summary,'','同一原26条validation的比较（不含test）：','',
                '|模型|Arm RMSE(rad)|Hand RMSE(rad)|末端FK RMSE(mm)|','|---|---:|---:|---:|']
            for name,metrics in public['offline_comparison']['models'].items():
                v=metrics['validation'];lines.append(f"|{name}|{v['arm_rmse']:.6f}|{v['hand_rmse']:.6f}|{v['ee_distance_rmse_mm']:.3f}|")
        lines+=['','|模型|train完整成功|独立unseen完整成功|','|---|---:|---:|']
        for name,cohorts in final['final_evaluation']['models'].items():
            t,u=cohorts['train'],cohorts['unseen'];lines.append(f"|{name}|{t['successes']}/{t['valid']}|{u['successes']}/{u['valid']}|")
        lines+=['','失败阶段：'+str({k:{split:v['failures'] for split,v in cc.items()} for k,cc in final['final_evaluation']['models'].items()}),
            '小样本诊断；未放宽成功标准或按结果替换初态。ImageDP仍需严格在线输入配对，触觉效益尚未验证；本轮不自动启动两者。']
    if 'final_review' in public:
        lines+=['',public['final_review']['interpretation_zh'],'',public['final_review']['resource_note_zh']]
    lines+=['','[轻量数值、门控与后台结果](evidence/narrow_repair.json) · [训练曲线](evidence/narrow_repair_curves.png)。原始数据、权重、视频留本地。']
    for p in DOCS:
        body='\n'.join(lines if p.endswith('STATE_DP_RESULTS.md') else [lines[0],'',lines[2],lines[3],lines[4],'',
            continuation,'',completed_summary,'', '实际扩采/训练/评估结果见[StateDP统一记录](STATE_DP_RESULTS.md)和[机器统计](evidence/narrow_repair.json)，不把准备写成成功。'])
        (CODE/p).write_text(publication.replace((CODE/p).read_text(),body))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for label,directory in [('200',Path(s['parent'])/'training'),('expanded fresh',root/'expanded_training')]:
        if not (directory/'validation.jsonl').exists():continue
        rows=[json.loads(x) for x in (directory/'validation.jsonl').read_text().splitlines()]
        for ax,k in zip(axes,('selection_mse','arm_rmse','ee_distance_rmse_mm')):
            ax.plot([r['update'] for r in rows],[r['metrics'][k] for r in rows],label=label+' val')
            ax.plot([r['update'] for r in rows],[r['train8'][k] for r in rows],'--',label=label+' train8')
            ax.set_title(k);ax.set_xlabel('Updates');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(CODE/ARTIFACTS[1],dpi=130);plt.close(fig)


def finish(root):
    req=read(root/'repair_publication_request.json')
    assert read(root/'status.json')['owned_workers_exited'],'Workers not released'
    assert sha(Path(__file__))==req['source_sha256']
    assert publication.git('rev-parse','HEAD')==req['expected_head'],'Intervening commit; keep results local for review'
    assert not publication.git('status','--porcelain'),'Working tree changed; do not overwrite'
    for p,h in req['outside_sha256'].items():assert publication.outside((CODE/p).read_text())==h
    snapshot(root)
    for p in DOCS+ARTIFACTS:assert (CODE/p).stat().st_size<2*1024**2
    publication.git('add','--',*(DOCS+ARTIFACTS));publication.git('diff','--cached','--check')
    publication.git('commit','-m','Record bounded camera/history repair and slow StateDP contrast results')
    assert publication.git('remote','get-url','origin')=='https://github.com/mao455/bcrl_visualTactile.git'
    publication.git('push','origin','HEAD:'+req['branch'])
    write(root/'publication.json',dict(state='published',commit=publication.git('rev-parse','HEAD')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('request','snapshot','finish'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.mode](a.root.resolve())
