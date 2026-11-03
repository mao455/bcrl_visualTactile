"""Small measured reports for the existing H24 experiment; raw traces stay local."""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tools.state_fk_audit import PHASES, error_summary, local_xyz, read_episode, target_errors
from visuotactile.compact import sha256
from visuotactile.kinematics import EncoderKinematics
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def pooled_errors(rows):
    count=sum(row['valid12_positions'] for row in rows)
    return {key:math.sqrt(sum(row['valid12_positions']*row[key]**2 for row in rows)/count)
        for key in ('arm12_rmse_rad','hand12_rmse_rad')}


def loop_counts(loop):
    result={}
    for cohort in ('train','validation'):
        rows=[row for row in loop['trials'] if row['cohort']==cohort and row.get('legal')]
        result[cohort]=dict(completed=len(rows),registered=4 if cohort=='train' else 8,
            successes=sum(bool(row['success']) for row in rows),
            contact=sum(bool(row.get('contact',{}).get('object_contact')) for row in rows),
            lift=sum(bool(row.get('native_lift')) for row in rows),
            first_subgoal=sum(bool(row.get('native_subgoals')) for row in rows),
            alignment=sum(bool(row.get('native_alignment')) for row in rows),
            fall=sum(bool(row.get('reasons',{}).get('fall')) for row in rows),
            timeout=sum(bool(row.get('reasons',{}).get('timeout')) for row in rows))
    return result


def validated_fk(metadata):
    evidence=read(Path('research_docs/evidence/state_motion_20261103/fk_readonly/fk_validation.json'))
    fk=EncoderKinematics(metadata['config']['assets']['robot_urdf'],metadata['joint_order'],links=('iiwa14_link_7',))
    if not evidence['passed'] or fk.sha256!=evidence['urdf_sha256'] or fk.root!=evidence['root'] or metadata['joint_order']!=evidence['joint_order']:
        raise ValueError('FK asset/order/base validation mismatch')
    return fk


def phases(root, label, data):
    folder=root/(label+'_offline')/'numeric_plans';rows=[]
    if not folder.exists():return rows
    for path in sorted(folder.glob('*.npz')):
        identity=path.stem;episode=read_episode(data['entries'][identity])
        metadata=episode['meta']
        fk=validated_fk(metadata)
        phase=episode['phase']
        with np.load(path) as values:
            truth=values['truth'][:,:12];valid=values['valid'][:,:12]
            indices=values['anchors'][:,None]+np.arange(12)
            phase_window=phase[indices.clip(max=len(phase)-1)]
            before=np.broadcast_to(values['before'][:,None],truth.shape)
            for model,prediction in [('Student',values['A0'][:,:12]),('Hold',before),('Expert',truth)]:
                errors=target_errors(prediction,truth,before,fk)
                for phase_index,name in enumerate(PHASES):
                    mask=valid & (phase_window==phase_index)
                    if mask.any():rows.append(dict(identity=identity,model=model,phase=name,**error_summary(errors,mask)))
    return rows


def closed_fk(root,label):
    previous=Path(read(root/'run_plan.json')['old_root'])
    expert_root=Path(read(previous/'session.json')['previous_root'])/'expert_learning'
    rows=[];curves=[]
    for path in sorted((root/(label+'_loop')).glob('*/trace.npz')):
        identity=path.parent.name;expert_path=expert_root/identity/'trace.npz'
        if not expert_path.exists():continue
        metadata=read(path.parent/'metadata.json');initial=read(path.parent/'initial_observable.json')
        if initial!=read(expert_path.parent/'initial_observable.json'):
            raise ValueError('Observable initial state mismatch: '+identity)
        fk=validated_fk(metadata)
        with np.load(path) as student,np.load(expert_path) as expert:
            count=min(len(student['joint']),len(expert['joint']))
            before=np.cumsum(np.linalg.norm(expert['object_contact'][:count],axis=-1).reshape(count,-1).max(-1)>.1)==0
            hold=np.broadcast_to(np.asarray(initial['joint_position_rad']).reshape(29),expert['joint'][:count].shape)
            for model,prediction,object_pose in [('Student',student['joint'][:count],student['object'][:count]),
                    ('Hold',hold,expert['object'][:count]),('Expert',expert['joint'][:count],expert['object'][:count])]:
                error=target_errors(prediction,expert['joint'][:count],hold,fk)
                summary=dict(identity=identity,model=model,compared_ticks=count,
                    direction_scope='expert actual joint FK displacement from initial actual joint FK >1mm; common tick prefix only',
                    observable_reset_equal=True,exact_simulator_state_proven=False)
                for name,mask in [('approach',before),('other',~before)]:
                    summary[name]=error_summary(error,mask) if mask.any() else None
                    if summary[name]:summary[name]['direction_scope']=summary['direction_scope']
                rows.append(summary)
                base=np.broadcast_to(np.asarray(initial['base_pose_w']).reshape(7),(count,7))
                object_local=local_xyz(object_pose[:,:3],base)
                distance=np.linalg.norm(error['xyz_position']-object_local,axis=-1)*1000
                goal=np.broadcast_to(np.asarray(initial['goal_pose_w']).reshape(7)[:3],(count,3))
                goal_distance=np.linalg.norm(error['xyz_position']-local_xyz(goal,base),axis=-1)*1000
                for tick in range(count):
                    curves.append(dict(identity=identity,model=model,tick=tick,time_s=(tick+1)/60,
                        ee_error_mm=float(error['distance'][tick]),orientation_degree=float(error['rotation'][tick]),
                        ee_to_object_mm=float(distance[tick]),ee_to_initial_goal_mm=float(goal_distance[tick])))
    return dict(episodes=rows,curves=curves,hold_scope='counterfactual fixed initial actual q in expert scene; no Hold rollout or success rate',
        time_scope='post-action actual joint state at (tick+1)/60 seconds; goal fixed to initial target, before subgoal resampling')


def report(root):
    public=Path('research_docs/evidence/h24_parameterization_20261103');public.mkdir(parents=True,exist_ok=True)
    session=read(root/'session.json');data=read(Path(session['source_root'])/'data.json')
    summary=dict(status=read(root/'status.json'),plan=read(root/'run_plan.json'),
        plan_sha256=sha256(root/'run_plan.json'),groups={},training={},updated_epoch=time.time())
    for name in ['batch_audit','kit_acceptance','selection','blocked','setup_complete','fairness','recovery_review']:
        if (root/(name+'.json')).exists():summary[name]=read(root/(name+'.json'))
    if (root/'resources/resources.json').exists():summary['resources']=read(root/'resources/resources.json')
    summary['source_commit']=session['source_commit']
    summary['runtime_source_sha256']=session['implementation_sha256']
    summary['fk_validation_sha256']=sha256(Path('research_docs/evidence/state_motion_20261103/fk_readonly/fk_validation.json'))
    summary['stage_artifacts']={path.stem:read(path) for path in root.glob('*_committed.json')}
    manifest_path=root/'checkpoint_manifest.json'
    checkpoints=read(manifest_path) if manifest_path.exists() else {}
    for name,directory in [('sample',root/'sample_resume/h24_train'),('epsilon',root/'h24_epsilon_train')]:
        for path in directory.glob('milestone_*.pt'):
            key=name+'/'+path.name
            if key not in checkpoints:
                checkpoints[key]=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size,
                    configuration_sha256=sha256(directory/'configuration.json'))
        if (directory/'metrics.json').exists():
            summary['training'][name]=[{key:({metric:value for metric,value in item.items() if metric!='episodes'}
                if key in ('train','validation') else item) for key,item in row.items()} for row in read(directory/'metrics.json')]
    atomic_json(manifest_path,checkpoints);summary['checkpoints']=checkpoints
    for folder in sorted(root.glob('*_offline')):
        if not (folder/'result.json').exists():continue
        label=folder.name.removesuffix('_offline');offline=read(folder/'result.json')
        loop=root/(label+'_loop/result.json')
        cache=root/(label+'_spatial_report.json')
        if not cache.exists() or (loop.exists() and cache.stat().st_mtime<loop.stat().st_mtime):
            atomic_json(cache,dict(phases=phases(root,label,data),closed_actual_fk=closed_fk(root,label)))
        rows=offline.pop('rows')
        offline['pooled_prefix12']=pooled_errors(rows)
        summary['groups'][label]=dict(offline=offline,spatial=read(cache),loop=read(loop) if loop.exists() else None)
        if loop.exists():summary['groups'][label]['closed_loop_counts']=loop_counts(read(loop))
        actual=summary['groups'][label]['spatial']['closed_actual_fk']
        curves=actual.pop('curves',[])
        aggregate=[]
        for model in sorted({row['model'] for row in curves}):
            for tick in sorted({row['tick'] for row in curves if row['model']==model}):
                selected=[row for row in curves if row['model']==model and row['tick']==tick]
                aggregate.append(dict(model=model,time_s=(tick+1)/60,episodes=len(selected),
                    **{key:float(np.mean([row[key] for row in selected])) for key in ('ee_error_mm','orientation_degree','ee_to_object_mm','ee_to_initial_goal_mm')}))
        actual['mean_curves']=aggregate
        with (public/(label+'_metrics.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(row.keys() for row in rows))),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    def sanitize(value):
        if isinstance(value,dict):return {key:sanitize(item) for key,item in value.items()}
        if isinstance(value,list):return [sanitize(item) for item in value]
        if isinstance(value,str):return value.replace(str(root),'outputs/visuotactile/native_dp_v2/h24_parameterization_20261103_01').replace(str(Path.home()),'<home>')
        return value
    atomic_json(public/'summary.json',sanitize(summary))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(14,4))
    for name,rows in summary['training'].items():
        for scope,style in [('train','--o'),('validation','-o')]:
            for axis,key in zip(axes,['arm_rmse','hand_rmse','selection_mse']):
                axis.plot([row['update'] for row in rows],[row[scope][key] for row in rows],style,label=name+' '+scope)
                axis.set(xlabel='Cumulative updates',ylabel=key+(' (rad^2)' if key=='selection_mse' else ' (rad)'))
    for axis in axes:
        if axis.lines:axis.legend(fontsize=6)
    figure.suptitle('First12 actions; train diagnostic16, validation40; raw diffusion losses not compared')
    figure.tight_layout();figure.savefig(public/'training_validation.png',dpi=140);plt.close(figure)
    if summary['groups']:
        figure,axes=plt.subplots(1,2,figsize=(12,4));names=list(summary['groups'])
        for axis,key,scale,title in [(axes[0],'precontact_ee12_target_fk_rmse_m',1000,'Approach target FK RMSE (mm)'),
                                    (axes[1],'arm_boundary_jump_rms_rad',1,'Arm target boundary RMS jump (rad)')]:
            axis.bar(names,[summary['groups'][name]['offline']['summary']['A0'][key]['mean']*scale for name in names])
            axis.set(ylabel=title);axis.tick_params(axis='x',rotation=25)
        figure.suptitle('Full validation40, policy seed7091; aggregate contact proxy')
        figure.tight_layout();figure.savefig(public/'spatial_continuity.png',dpi=140);plt.close(figure)
        for label,group in summary['groups'].items():
            curves=group['spatial']['closed_actual_fk']['mean_curves']
            if not curves:continue
            figure,axes=plt.subplots(1,3,figsize=(14,4))
            for model in ('Student','Hold','Expert'):
                values=[row for row in curves if row['model']==model]
                for axis,key in zip(axes,('ee_error_mm','ee_to_object_mm','episodes')):
                    axis.plot([row['time_s'] for row in values],[row[key] for row in values],label=model)
                    axis.set(xlabel='Time (s)',ylabel=key);axis.legend(fontsize=7)
            figure.suptitle(label+': actual joint FK; same-tick common prefixes; changing episode count')
            figure.tight_layout();figure.savefig(public/(label+'_actual_fk.png'),dpi=140);plt.close(figure)
    document=Path('research_docs/H24_SAMPLE_EPSILON_COMPARISON.md')
    if document.exists():
        text=document.read_text();begin='<!-- h24-results:start -->';end='<!-- h24-results:end -->'
        block=[begin,'当前后台状态：`'+summary['status']['state']+'`。此自动区块待结果复核。','',
            '|组/累计updates|val40 Arm(rad)|val40 Hand(rad)|已完成闭环成功|软件缺失|','|---|---:|---:|---|---:|']
        for name,group in summary['groups'].items():
            metric=group['offline']['pooled_prefix12'];loop=group['loop'];trials=loop['trials'] if loop else []
            block.append(f"|{name}|{metric['arm12_rmse_rad']:.6f}|{metric['hand12_rmse_rad']:.6f}|{sum(bool(row.get('success')) for row in trials)}/{len(trials)}完成（注册12）|{len(loop['missing']) if loop else '未开始'}|")
        if summary['groups']:
            block+=['','以下为同一 val40/seed7091 的 episode 均值；FK 为目标关节的末端误差，实际闭环 FK 另存。','',
                '|组|接触前目标FK(mm)|Motion cosine|Amplitude ratio|Arm边界跳变(rad)|Hand边界跳变(rad)|',
                '|---|---:|---:|---:|---:|---:|']
            for name,group in summary['groups'].items():
                metrics=group['offline']['summary']['A0'];values=[]
                for key,scale in [('precontact_ee12_target_fk_rmse_m',1000),('moving12_direction_cosine',1),
                        ('moving12_amplitude_ratio',1),('arm_boundary_jump_rms_rad',1),('hand_boundary_jump_rms_rad',1)]:
                    value=metrics.get(key,{}).get('mean');values.append('缺失' if value is None else f'{value*scale:.5f}')
                block.append('|'+name+'|'+'|'.join(values)+'|')
            block+=['','|组/初态|完成/注册|完整成功|接触|抬升|首子目标|alignment|fall|timeout|',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for name,group in summary['groups'].items():
                for split,counts in group.get('closed_loop_counts',{}).items():
                    values=[str(counts[key]) for key in ('successes','contact','lift','first_subgoal','alignment','fall','timeout')]
                    block.append(f"|{name}/{split}|{counts['completed']}/{counts['registered']}|"+'|'.join(values)+'|')
        if summary.get('selection'):
            chosen=summary['selection']
            block+=['',f"暂定续训模型：`{chosen['selected']}`；配对 validation 数量 {len(chosen['paired_validation_ids'])}。未建立统计显著优势。",
                '选择依据及各指标原值保存于 summary.selection；test 未参与选择。']
        block += ['', '实际接触/lift/首子目标/alignment/fall/timeout及逐条结果见[summary](evidence/h24_parameterization_20261103/summary.json)。',end]
        if begin in text:text=text[:text.index(begin)]+'\n'.join(block)+text[text.index(end)+len(end):]
        else:text+='\n'+'\n'.join(block)+'\n'
        document.write_text(text)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    torch.set_num_threads(2);report(parser.parse_args().root.resolve())
