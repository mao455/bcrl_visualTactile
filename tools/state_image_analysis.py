"""CPU-only postprocessing of committed isolated evaluations; no simulator starts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch

from tools.state_image_queue import CODE,read,write,sha
from tools.h24_diagnosis_report import describe
from tools.h24_diagnosis_common import contact_force
from tools.state_fk_audit import fk_pose, rotation_error
from visuotactile.kinematics import EncoderKinematics,FINGERS


def verify_image_inputs(root):
    target=root/'image_input_integrity.json'
    if target.exists():
        assert read(target)['passed'];return
    source=Path(read(root/'session.json')['source_root']);data=read(source/'data.json');index=read(source/'native_perception.json')
    rows=[]
    for identity in data['splits']['train']+data['splits']['validation']:
        entry=data['entries'][identity];expected=entry['perception_sha256'];assert sha(entry['perception'])==expected
        row=dict(identity=identity,perception_sha256=expected,tail_sha256=None)
        tail=index['entries'][identity].get('tail')
        if tail:
            assert sha(tail['path'])==tail['sha256'];row['tail_sha256']=tail['sha256']
        rows.append(row)
        write(root/'image_integrity_progress.json',dict(checked=len(rows),required=360,utc=time.time()))
    write(target,dict(passed=True,checked=360,test_data_loaded=False,rows=rows,utc=time.time()))


def joint_metrics(error):
    return dict(rmse_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),mae_rad=np.mean(np.abs(error),axis=0).tolist(),
        p95_rad=np.quantile(np.abs(error),.95,axis=0).tolist())


def analyze(root):
    source=read(root/'session.json')['source_root'];data=read(Path(source)/'data.json')
    public=CODE/'research_docs/evidence/state_image_20261104_02';public.mkdir(exist_ok=True,parents=True)
    rows=[];details=[];diagnostics=[]
    for file in sorted((root/'stages').glob('*/committed.json')):
        name=file.parent.name;record=read(file)
        if name.startswith('kit_'):
            job=read(root/'jobs'/(name.removeprefix('kit_')+'.json'))
            kind=('normal_sdk_return' if record.get('sdk_graceful_close') else
                'sdk_fast_exit_zero' if record.get('exit_code')==0 and job['lifecycle']['fast_shutdown'] else
                'shutdown_fault_after_complete_runtime' if record.get('runtime_complete') else 'runtime_or_startup_software_missing')
            item=dict(stage=name,classification=kind,requested=job['lifecycle'],**{k:record.get(k) for k in ('exit_code','timeout','runtime_complete','sdk_graceful_close','resources_released')})
            if job['mode']=='expert' and record.get('runtime_complete'):
                identity=job['specs'][0]['identity']
                reference=CODE/'outputs/visuotactile/native_dp_v2/state_image_20261104_01/kit_gate/stages/03_expert_shutdown'/identity/'trace.npz'
                with np.load(reference) as a,np.load(Path(record['output'])/identity/'trace.npz') as b:
                    item['recorded_trajectory_bytewise']={k:a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a.files if k in b.files and k!='expert_shadow'}
                item['observed_parity_passed']=all(item['recorded_trajectory_bytewise'].values())
            diagnostics.append(item)
            continue
        if '_loop_' not in name:continue
        job,_,update,ordinal=name.split('_');output=Path(record['output'])
        trials=(record.get('result') or {}).get('trials',[])
        if not trials:
            rows.append(dict(job=job,update=int(update),ordinal=int(ordinal),state='software_missing',success=None));continue
        trial=trials[0]
        if not trial.get('legal'):
            rows.append(dict(job=job,update=int(update),ordinal=int(ordinal),state='illegal_initial',success=None));continue
        path=output/trial['identity'];summary,series,trace=describe(path)
        metadata=read(path/'metadata.json');entry=data['entries'][trial['identity']]
        with h5py.File(entry['path'],'r') as f:
            group=f['episodes'][entry['episode']]
            expert_q=group['transition/diagnostics_after/joint_position_rad'][:]
            expert_target=group['transition/target_after_rad'][:]
        count=min(len(trace['joint']),len(expert_q))
        fk=EncoderKinematics(metadata['config']['assets']['robot_urdf'],metadata['joint_order'],links=('iiwa14_link_7',*FINGERS))
        with torch.no_grad():
            actual,rot=fk(torch.as_tensor(trace['joint'][:count]).double())
            expert,exrot=fk(torch.as_tensor(expert_q[:count]).double())
        distance=np.linalg.norm((actual[:,0]-expert[:,0]).numpy(),axis=-1)*1000
        orientation=rotation_error(rot[:,0].numpy(),exrot[:,0].numpy())
        fingers=np.linalg.norm((actual[:,1:]-expert[:,1:]).numpy(),axis=-1)*1000
        pre=np.cumsum(contact_force(trace)[:count]>.1)==0
        reference=dict(alignment='same recorded native tick/common prefix; original expert physical state is not an exact hidden-state replay',
            common_ticks=count,actual_joint_error=joint_metrics(trace['joint'][:count]-expert_q[:count]),
            target_joint_error=joint_metrics(trace['target'][:count]-expert_target[:count]),
            achieved_ee_rmse_mm=float(np.sqrt(np.mean(distance**2))),achieved_rotation_mean_deg=float(orientation.mean()),
            achieved_fingertip_rmse_mm=np.sqrt(np.mean(fingers**2,axis=0)).tolist(),
            student_precontact_ee_rmse_mm=float(np.sqrt(np.mean(distance[pre]**2))) if pre.any() else None,
            student_postcontact_ee_rmse_mm=float(np.sqrt(np.mean(distance[~pre]**2))) if (~pre).any() else None)
        summary.update(recorded_expert_reference=reference,target_actual_joint_error=joint_metrics(trace['joint']-trace['target']))
        event=summary['events']
        phase=('success' if trial['success'] else 'after_alignment' if event['alignment_tick'] is not None else
            'after_first_subgoal' if event['subgoal_tick'] is not None else 'after_lift' if event['lift_tick'] is not None else
            'after_contact' if event['contact_tick'] is not None else 'before_contact')
        item=dict(job=job,update=int(update),ordinal=int(ordinal),identity=trial['identity'],cohort=trial['cohort'],
            state='runtime_completed',success=trial['success'],exit_code=record['exit_code'],sdk_graceful_close=record['sdk_graceful_close'],
            resources_released=record['resources_released'],failure_phase=phase,contact=event['contact_tick'] is not None,
            lift=trial['native_lift'],subgoal=trial['native_subgoals'],alignment=trial['native_alignment'],
            timeout=trial['reasons'].get('timeout'),fall=trial['reasons'].get('fall'),
            ee_reference_rmse_mm=reference['achieved_ee_rmse_mm'],precontact_ee_reference_rmse_mm=reference['student_precontact_ee_rmse_mm'])
        rows.append(item);details.append(dict(**item,metrics=summary))
    write(public/'closed_loop_metrics.json',dict(rows=details,diagnostics=diagnostics,
        limitation='Shutdown faults remain separate from valid completed native outcomes; pre-action/runtime missing is never policy failure'))
    with (public/'closed_loop_summary.csv').open('w') as f:
        names=sorted({k for r in rows for k in r});w=csv.DictWriter(f,fieldnames=names or ['state'],lineterminator='\n');w.writeheader();w.writerows(rows)
    # Offline per-joint comparison, no test-set observations or selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=[];labels=[]
    for job in ('state','image'):
        p=root/(job+'_train')/'metrics.json'
        if not p.exists():continue
        for row in read(p):
            if row['update'] in (1000,3000,6000,10000):values.append(row['validation']['joint_rmse_rad']);labels.append(job+' '+str(row['update']))
    if values:
        fig,ax=plt.subplots(figsize=(9,7),layout='constrained');im=ax.imshow(np.array(values).T,aspect='auto',cmap='magma')
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.set_yticks(range(29),[str(i+1) for i in range(29)])
        ax.set_ylabel('Canonical joint index (Arm1-7, Hand8-29)');ax.set_title('Validation40 per-joint target RMSE, radians')
        fig.colorbar(im,ax=ax,label='rad');fig.savefig(public/'joint_rmse_heatmap.png',dpi=120);plt.close(fig)
    # Append a separate bounded generated section; keep the original blocked report.
    report=CODE/'research_docs/STATE_IMAGE_DP_IMPROVEMENT.md';text=report.read_text();marker='<!-- offline-v2-final-analysis -->'
    if marker in text:text=text.split(marker)[0].rstrip()+'\n'
    table=[]
    for job in ('state','image'):
        for n in (1000,3000,6000,10000):
            found=[r for r in rows if r['job']==job and r['update']==n];complete=[r for r in found if r['state']=='runtime_completed']
            if found:table.append(f"|{job}|{n}|{sum(r['success'] for r in complete)}/{len(complete)}|{12-len(complete)}|{sum(r.get('exit_code')!=0 for r in complete)}|")
    text+=marker+'''\n\n## 闭环与逐关节最终分析

分母仅包含完整合法原生结果，关闭异常另列；不足12条不能称完整配对验收。
与原始专家的实际轨迹误差按共同native tick计算，不声称隐藏物理状态严格回放。

|模型|updates|原生成功/完整结果|缺失或非法初态|有结果但非零退出|
|---|---|---|---|---|
'''+ '\n'.join(table)+'''

[完整阶段/29关节/空间指标](evidence/state_image_20261104_02/closed_loop_metrics.json) ·
[闭环CSV](evidence/state_image_20261104_02/closed_loop_summary.csv)。
''' + ('\n![29关节验证误差](evidence/state_image_20261104_02/joint_rmse_heatmap.png)\n' if values else '')
    report.write_text(text)
    videos=[]
    for p in (root/'stages').glob('*loop_10000*/*/*/*.mp4'):
        count=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames','-of','default=nw=1:nk=1',str(p)],text=True).strip()
        videos.append(dict(path=str(p),decoded_frames=int(count),sha256=sha(p)))
    write(root/'videos.json',videos)
    write(root/'analysis_complete.json',dict(state='completed',closed_loop_rows=len(rows),diagnostic_rows=len(diagnostics),utc=time.time(),videos=videos))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    verify_image_inputs(a.root)
    if a.wait:
        while not (a.root/'owner_exit.json').exists() or subprocess.run(['systemctl','--user','is-active','--quiet','state-image-offline-20261104-02.service']).returncode==0:
            write(a.root/'analysis_status.json',dict(state='pending',waiting_for='training/evaluation owner exit',utc=time.time()));time.sleep(30)
    try:
        analyze(a.root)
        if a.publish:
            from tools.state_image_report import publish
            publish(a.root)
    except BaseException as e:
        import traceback
        write(a.root/'analysis_status.json',dict(state='blocked',reason=repr(e),traceback=traceback.format_exc(),utc=time.time()));raise
