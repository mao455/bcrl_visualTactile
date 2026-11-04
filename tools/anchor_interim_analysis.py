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

PUBLIC=CODE/'research_docs/evidence/state_dp'
REPORT=CODE/'research_docs/STATE_DP_RESULTS.md'


def motion_window(pred, truth, before, xyz, exact_xyz, initial_xyz, length):
    """Chronological teacher-forced targets, including the initial target-to-tick0 increment."""
    n=min(int(length),len(pred))
    if n<=0:return dict(ticks=0)
    p,y=pred[:n],truth[:n];error=p-y
    dp=np.diff(np.concatenate((before[None],p)),axis=0)
    dy=np.diff(np.concatenate((before[None],y)),axis=0)
    px,qx=xyz[:n],exact_xyz[:n]
    dx=np.diff(np.concatenate((initial_xyz[None],px)),axis=0)
    ex=np.diff(np.concatenate((initial_xyz[None],qx)),axis=0)
    def direction(a,b,threshold):
        an,bn=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
        moving=bn>threshold
        return dict(moving_ticks=int(moving.sum()),static_ticks=int((~moving).sum()),
            cosine=float(np.mean(np.sum(a[moving]*b[moving],axis=-1)/np.maximum(an[moving]*bn[moving],1e-15))) if moving.any() else None,
            amplitude_ratio=float(an[moving].sum()/bn[moving].sum()) if moving.any() else None,
            amplitude_error=float(np.sqrt(np.mean((an-bn)**2))),
            expert_increment_rms=float(np.sqrt(np.mean(b*b))),predicted_increment_rms=float(np.sqrt(np.mean(a*a))))
    distance=np.linalg.norm(px-qx,axis=-1)*1000
    return dict(ticks=n,arm_rmse_rad=float(np.sqrt(np.mean(error[:,:7]**2))),
        hand_rmse_rad=float(np.sqrt(np.mean(error[:,7:]**2))),joint_rmse_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        arm_delta_joint_rmse_rad=np.sqrt(np.mean((dp[:,:7]-dy[:,:7])**2,axis=0)).tolist(),
        arm_delta_rmse_rad=float(np.sqrt(np.mean((dp[:,:7]-dy[:,:7])**2))),
        arm_hold_rmse_rad=float(np.sqrt(np.mean((before[:7]-y[:,:7])**2))),
        arm_motion=direction(dp[:,:7],dy[:,:7],1e-6),ee_motion=direction(dx,ex,1e-6),
        ee_rmse_mm=float(np.sqrt(np.mean(distance**2))),ee_mean_mm=float(distance.mean()),
        ee_endpoint_mm=float(distance[-1]),ee_integrated_error_mm_s=float(distance.sum()/60),
        arm_net_displacement_error_rad=float(np.linalg.norm(p[-1,:7]-y[-1,:7])))


def aggregate_windows(rows):
    valid=[r for r in rows if r['ticks']]
    if not valid:return dict(episodes=0,ticks=0)
    weight=np.array([r['ticks'] for r in valid],dtype=float)
    result=dict(episodes=len(valid),ticks=int(weight.sum()))
    for key in ('arm_rmse_rad','hand_rmse_rad','joint_rmse_rad','arm_delta_joint_rmse_rad',
                'arm_delta_rmse_rad','arm_hold_rmse_rad','ee_rmse_mm'):
        value=np.sqrt(np.average(np.array([r[key] for r in valid])**2,axis=0,weights=weight))
        result[key]=value.tolist()
    for key in ('ee_mean_mm','ee_endpoint_mm','ee_integrated_error_mm_s','arm_net_displacement_error_rad'):
        result[key]=float(np.mean([r[key] for r in valid]))
    for part in ('arm_motion','ee_motion'):
        moving=[r[part] for r in valid if r[part]['moving_ticks']]
        result[part]=dict(moving_ticks=sum(r[part]['moving_ticks'] for r in valid),
            static_ticks=sum(r[part]['static_ticks'] for r in valid))
        for key in ('cosine','amplitude_ratio','amplitude_error','expert_increment_rms','predicted_increment_rms'):
            result[part][key]=float(np.mean([r[key] for r in moving])) if moving else None
    return result


def offline_assessment(root):
    """Read-only common EMA/DDIM/noise protocol on train320 and validation40; resumable per checkpoint."""
    import torch
    from visuotactile.state_image_offline import setup, no_kit
    from visuotactile.state_history_training import labels
    from visuotactile.position_control import bounded_targets
    from visuotactile.state_motion import metadata_and_contact
    s=read(root/'session.json');parent=Path(s['parent_root'])
    assert (root/'owner_exit.json').exists(), 'Wait for Kit owner to release all resources'
    _,data,ids,vals,records,model,obj,lease,_=setup(parent,'state')
    obj.weights=read(parent/'state_audit/calibration.json')['weights']
    cp=read(root/'checkpoint.json')
    checkpoints=[s['checkpoints']['state']['path'],str(parent/'state_train/continued_6584.pt'),cp['path']]
    out=root/'offline';out.mkdir(exist_ok=True);completed=[]
    for checkpoint in checkpoints:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);update=saved['updates'];cp_hash=sha(checkpoint)
        final=out/f'{update}.json'
        if final.exists():
            previous=read(final);assert previous['checkpoint_sha256']==cp_hash;completed.append(previous);del saved;continue
        assert saved['contract']==model.contract
        model.load_state_dict(saved['ema']);model.eval()
        summary=dict(update=update,checkpoint_sha256=cp_hash,checkpoint=checkpoint,
            ema_step=saved['ema_optimization_step'],lr=saved['optimizer']['param_groups'][0]['lr'],
            cumulative_anchor_epochs=saved.get('cumulative_sampled_anchors',s['old_exposure']['state']['anchors'])/s['anchor_population'],
            source_sha256=sha(__file__),scope='teacher-forced chronological executed12 predictions; EMA; DDIM16 eta0 clipFalse seed7091; loss noise8091',
            contact_definition='Recorded400: first any fingertip-substep force norm >0.1N; source has no object-filtered contact. Not object-contact or grasp success.',
            motion_definition='Every increment including tick0 versus preceding actual target; moving threshold1e-6rad /1e-6m for direction only; static increments retained in RMSE',
            aggregation='Position/delta RMSE pooled over valid ticks; motion direction/amplitude macro-mean per nonstatic episode; integrated error macro-mean mm*s',
            test_used=False,cohorts={},kit_dependency=no_kit())
        del saved
        for cohort,identities in [('train',ids),('validation',vals)]:
            rows=[]
            for i,identity in enumerate(identities):
                rec=records[identity]
                with torch.no_grad(),lease.block():
                    cond,anchors,_=model.encode(rec);anchors=anchors.to(rec['command'].device)
                    raw=model.predict(cond,torch.Generator(device='cuda').manual_seed(7091))
                    bounded=bounded_targets(raw.flatten(0,1),rec['joint_limits']).reshape_as(raw)
                    truth,mask=labels(rec,anchors,24)
                    loss,parts,_=obj.forward(model,cond,truth,mask,torch.Generator(device='cuda').manual_seed(8091))
                    p=bounded[:,:12][mask[:,:12]];y=truth[:,:12][mask[:,:12]]
                    assert len(p)==len(rec['command']) and torch.equal(y,rec['command'])
                    before=rec['robot'][0,58:87]
                    px,_=obj.fk(p);qx,_=obj.fk(y);ix,_=obj.fk(before[None])
                    pn,yn,bn,pxn,qxn,ixn=[v.detach().cpu().numpy() for v in (p,y,before,px[:,0],qx[:,0],ix[0,0])]
                    loss_values=dict(loss=float(loss),**{k:float(v) for k,v in parts.items()})
                _,contact=metadata_and_contact(data['entries'][identity])
                hits=np.flatnonzero(contact);first=int(hits[0]) if len(hits) else len(pn)
                windows={name:motion_window(pn,yn,bn,pxn,qxn,ixn,n) for name,n in
                    [('first12',12),('first24',24),('precontact',first),('full',len(pn))]}
                rows.append(dict(identity=identity,loss=loss_values,contact_tick=first,windows=windows))
                trace=out/f'targets_{update}'/f'{identity}.npz';trace.parent.mkdir(exist_ok=True)
                np.savez_compressed(trace,prediction=pn,expert=yn,before=bn,predicted_ee=pxn,expert_ee=qxn)
                write(out/'progress.json',dict(state='running',update=update,cohort=cohort,completed=i+1,total=len(identities),utc=time.time()))
            summary['cohorts'][cohort]=dict(episodes=len(rows),loss={k:float(np.mean([r['loss'][k] for r in rows])) for k in rows[0]['loss']},
                windows={key:aggregate_windows([r['windows'][key] for r in rows]) for key in ('first12','first24','precontact','full')},rows=rows)
        summary['train_validation_gap']={key:summary['cohorts']['validation']['windows']['full'][key]-summary['cohorts']['train']['windows']['full'][key]
            for key in ('arm_rmse_rad','hand_rmse_rad','ee_rmse_mm')}
        write(final,summary);completed.append(summary)
    write(out/'comparison.json',dict(state='completed',checkpoints=completed))
    write(out/'progress.json',dict(state='completed',utc=time.time(),optimizer_updates=0))
    return completed


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
    cohorts={}
    for cohort in ('train','validation'):
        selected={spec['identity'] for ordinal,spec in enumerate(s['specs']) if ordinal in s['closed_loop_ordinals'] and spec['cohort']==cohort}
        subset=[x for x in rows if x['identity'] in selected]
        cohorts[cohort]=dict(planned=len(selected),new=counts([x['new_result'] for x in subset]),
            old=counts([x['old_result'] for x in subset]),rows=subset)
    planned=len(s['closed_loop_ordinals'])
    payload=dict(state=state,planned=planned,starts=len(attempts),maximum_starts=s['kit_maximum_starts'],valid=len(rows),
        pending=planned-len(rows),software_missing=sum(not x.get('valid') for x in completed),
        acceptance_failed=sum(x.get('valid',False) and not x.get('native_outcome_valid',False) for x in completed),
        cohorts=cohorts,rows=rows,test40_used=False,source_commit=s['source_commit'],analysis_source_sha256=sha(__file__),
        checkpoint=read(root/'checkpoint.json') if (root/'checkpoint.json').exists() else None,
        threshold=dict(ee_mm=5.,consecutive_native_ticks=3),stable_grasp_native='N/A')
    for name in ('monitor_exit.json','parent_pause.json','owner_error.json'):
        if (root/name).exists():payload[name.removesuffix('.json')]=read(root/name)
    offline=read(root/'offline/comparison.json')['checkpoints'] if (root/'offline/comparison.json').exists() else []
    write(PUBLIC/'closed_loop.json',payload)
    if offline:
        compact=[dict(**{k:v for k,v in item.items() if k!='cohorts'},
            cohorts={name:{k:v for k,v in value.items() if k!='rows'} for name,value in item['cohorts'].items()}) for item in offline]
        write(PUBLIC/'offline.json',compact)
    keys=['contact','lift','first_subgoal','success'];labels=['Contact diagnostic','Native lift','First subgoal','Full task success']
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(4);width=.36
    for ax,(cohort,values) in zip(axes,cohorts.items()):
        for offset,label,key in [(-width/2,'Original10000','old'),(width/2,str(s['expected_updates']),'new')]:
            c=values[key]
            if c['valid']:
                bars=ax.bar(x+offset,[100*c[k]/c['valid'] for k in keys],width,label=label)
                ax.bar_label(bars,labels=[f"{c[k]}/{c['valid']}" for k in keys],padding=3)
        ax.set_xticks(x,labels,rotation=18);ax.set_ylim(0,115);ax.set_ylabel('Observed proportion (%)')
        ax.set_title(cohort+(' fitting diagnosis' if cohort=='train' else ' fixed4'))
        if values['new']['valid']:ax.legend()
    fig.tight_layout();fig.savefig(PUBLIC/'stages.png',dpi=140);plt.close(fig)
    if plot_data:
        ordinal,ns,ps,es=plot_data;fig,axes=plt.subplots(2,2,figsize=(10,7))
        for label,series in [('10000',ps),(str(s['expected_updates']),ns),('expert',es)]:
            tick=series['tick']
            if 'actual_expert_fk_mm' in series:axes[0,0].plot(tick,series['actual_expert_fk_mm'],label=label)
            axes[0,1].plot(tick,series['object_height_m'],label=label)
            axes[1,0].plot(tick,series['hand_jump_rad'],label=label)
            axes[1,1].plot(tick,series['object_contact_N'],label=label)
        for ax,title in zip(axes.flat,['EE distance from paired expert (mm)','Object height (m)','Hand target change RMS (rad)','Object normal contact force (N)']):
            ax.set_title(title);ax.set_xlabel('Native tick (60 Hz)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(PUBLIC/'representative.png',dpi=140);plt.close(fig)
    parent=Path(s['parent_root']);history=[r for r in read(parent/'state_train/metrics.json') if r['validation_scope']=='validation40']
    training=[]
    with (parent/'state_train/train.jsonl').open() as stream:
        for line in stream:
            try:training.append(json.loads(line))
            except json.JSONDecodeError:pass
    write(PUBLIC/'training_curve.json',dict(history=history,updates=[{k:v for k,v in r.items() if k in ('update','continuation_update','lr','gradient_norm','grad_norm','cumulative_anchor_epochs','loss')} for r in training[::20]],
        scope='Historical train16 / validation40 EMA curves; common endpoint assessment additionally uses all train320'))
    fig,axes=plt.subplots(2,3,figsize=(13,7));updates=[r['update'] for r in history]
    for cohort,color in [('train','tab:blue'),('validation','tab:orange')]:
        for ax,key in zip(axes.flat,('loss','arm_rmse','hand_rmse','ee_distance_rmse_mm')):
            values=[r[cohort+'_loss']['loss'] if key=='loss' else r[cohort][key] for r in history]
            ax.plot(updates,values,label=cohort+('16' if cohort=='train' else '40'),color=color);ax.set_title(key);ax.legend(fontsize=8)
    for ax,key in zip(axes.flat[4:],('lr','gradient_norm')):
        rows_with=[r for r in training if key in r]
        ax.plot([r['update'] for r in rows_with],[r[key] for r in rows_with]);ax.set_title(key)
    for ax in axes.flat:ax.set_xlabel('Cumulative optimizer updates');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(PUBLIC/'training.png',dpi=140);plt.close(fig)
    lines=['# StateDP 150 anchor等效epoch：训练、闭环与清理','',
        '|模型|初态用途|成功/有效|接触诊断|原生抬升|首子目标|超时|','|---|---|---:|---:|---:|---:|---:|']
    for cohort,value in cohorts.items():
        for label,c in [('10000',value['old']),(str(s['expected_updates']),value['new'])]:
            lines.append(f"|StateDP {label}|{cohort}|{c['success']}/{c['valid']}|{c['contact']}|{c['lift']}|{c['first_subgoal']}|{c['timeout']}|")
    lines += ['',f"计划{planned}初态，实际启动{len(attempts)}，有效{len(rows)}，软件缺失{payload['software_missing']}，验收失败{payload['acceptance_failed']}，待执行/待获得有效结果{payload['pending']}。最多{s['kit_maximum_starts']}次启动。训练初态只作拟合诊断，不并入validation或test。原StateDP/ImageDP正式0/12不变，Image未新训。",
        '原生环境没有离散稳定抓持判据，保持N/A；contact是已有力诊断，不作为稳定抓取。异常关闭与任务失败分开；SDK fast exit不称完整正常关闭。','',
        '![阶段对比](evidence/state_dp/stages.png)','',
        '|seed|cohort|首次5mm连续3tick偏离：旧/新|专家接触tick|终止|视频|','|---|---|---|---:|---|---|']
    for row in rows:
        cohort=next(spec['cohort'] for spec in s['specs'] if spec['identity']==row['identity'])
        lines.append(f"|{row['seed']}|{cohort}|{row['old']['first_ee_divergence_tick']} / {row['new']['first_ee_divergence_tick']}|{row['expert_events']['contact_tick']}|{row['new_result']['reasons']}|[本地视频]({row['video']})|")
    lines += ['','![训练曲线](evidence/state_dp/training.png)','',
        '历史曲线固定train16 / validation40，EMA、H24执行前12。以下端点重算使用完整train320 / validation40，三个checkpoint、归一化、DDIM16/eta0/clipFalse seed7091、loss noise8091和辅助权重相同。Train总loss与Validation总loss可比较；它们不是完整任务成功率。','',
        '|更新|anchor等效epoch|cohort|总loss|Arm rad|Hand rad|FK mm|delta Arm rad/tick|动作方向cos|幅值比|','|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in offline:
        for cohort,value in item['cohorts'].items():
            w=value['windows']['full'];m=w['arm_motion']
            lines.append(f"|{item['update']}|{item['cumulative_anchor_epochs']:.4f}|{cohort}|{value['loss']['loss']:.7g}|{w['arm_rmse_rad']:.6g}|{w['hand_rmse_rad']:.6g}|{w['ee_rmse_mm']:.5g}|{w['arm_delta_rmse_rad']:.6g}|{m['cosine']:.5g}|{m['amplitude_ratio']:.5g}|")
    if not offline:lines+=['','同口径端点离线比较尚未完成，不提前给出150 epoch结论。']
    lines += ['','|更新|cohort|时段|Arm rad|增量rad/tick|hold基线rad|方向cos|幅值比|FK mm|EE积分误差mm·s|','|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in offline:
        for cohort,value in item['cohorts'].items():
            for name in ('first12','first24','precontact'):
                w=value['windows'][name]
                if not w['ticks']:continue
                m=w['arm_motion'];cos=m['cosine'];amp=m['amplitude_ratio']
                lines.append(f"|{item['update']}|{cohort}|{name}|{w['arm_rmse_rad']:.5g}|{w['arm_delta_rmse_rad']:.5g}|{w['arm_hold_rmse_rad']:.5g}|{cos:.5g}|{amp:.5g}|{w['ee_rmse_mm']:.5g}|{w['ee_integrated_error_mm_s']:.5g}|")
    lines += ['','逐关节位置/增量误差、末端方向/幅值、静态tick数量、Train/Validation gap及checkpoint SHA见[端点数值](evidence/state_dp/offline.json)。运动方向只排除专家增量≤1e-6rad或1e-6m的tick；这些tick仍计入RMSE。幅值和cos按episode宏平均，不把微小分母比值池化成任务成功。',
        '原400记录缺少物体过滤的接触力，离线precontact使用现有任一指尖子步力>0.1N的首次诊断事件；闭环专家接触沿用物体过滤力连续3tick的既有定义。二者不混用，不称稳定抓持。',
        '分析使用专家历史观测，明确为teacher-forced离线诊断；闭环始终为模型自主动作，没有专家接管、平滑或控制器修改。',
        f"评估source commit `{s['source_commit']}`；运行目录 `{root}`。训练150 epoch后保持证据审查暂停；本次不会自动增加至200–300或启动Image训练。",'',
        '清理及回归结果见 [CLEANUP_AUDIT.md](CLEANUP_AUDIT.md)。仍在使用的旧目录缓存与checkpoint不按日期删除。','']
    if rows:lines+=['![代表轨迹](evidence/state_dp/representative.png)','']
    REPORT.write_text('\n'.join(lines));return payload


def publish(root):
    paths=[str(REPORT.relative_to(CODE)),str(PUBLIC.relative_to(CODE))]
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=CODE,text=True).strip()
    subprocess.run(['git','add',*paths],cwd=CODE,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=CODE).returncode==0:return
    subprocess.run(['git','diff','--cached','--check'],cwd=CODE,check=True)
    subprocess.run(['git','commit','-m','Report measured State checkpoint assessment'],cwd=CODE,check=True)
    subprocess.run(['git','push','origin','HEAD:feature/native-dp-training-v2'],cwd=CODE,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--offline',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args()
    root=a.root.resolve();last=None
    while True:
        path=root/'attempts.json';stamp=path.stat().st_mtime_ns if path.exists() else 0
        finished=(root/'owner_exit.json').exists()
        if stamp!=last or finished:analyze(root);last=stamp
        if not a.watch or finished or time.time()>read(root/'session.json')['deadline']+300:break
        time.sleep(20)
    if a.offline:
        offline_assessment(root);analyze(root)
    if a.publish:publish(root)
