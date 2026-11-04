"""Separate teacher-conditioned target fit from autonomous deviation, using saved plans."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from tools.state_image_queue import read,write
from tools.grasp_diagnosis_analysis import BASE,FINAL,PUBLIC
from tools.h24_diagnosis_report import arrays,rms
from tools.h24_comparison_report import validated_fk
from tools.state_fk_audit import fk_pose
from visuotactile.state_motion import motion_metrics,metadata_and_contact


def run(root):
    session=read(root/'session.json');data=read(Path(session['source_root'])/'data.json')
    old=read(BASE/'state_image_20261104_02/session.json')
    stats=torch.load(old['position_statistics'],map_location='cpu',weights_only=False)
    attempts=[x for x in read(FINAL/'attempts.json') if x.get('valid')];rows=[]
    for ordinal in (0,4):
        spec=session['specs'][ordinal];identity=spec['identity'];entry=data['entries'][identity]
        metadata,contact=metadata_and_contact(entry)
        fk=validated_fk(metadata)
        with h5py.File(entry['path']) as f:
            g=f['episodes'][entry['episode']]
            targets=g['transition/target_after_rad'][:];before=g['transition/target_before_rad'][:]
            source_actor=g['transition/expert_actor_input_diagnostic'][:]
            source_q=g['transition/diagnostics_before/joint_position_rad'][:]
            source_proprio=g['obs/proprio'][:]
        limits=np.asarray(metadata['joint_limits_rad'])[0];n=len(targets)
        for model in ('StateDP','ImageDP'):
            p=arrays(root/'parity'/model/(identity+'.npz'));anchors=p['anchors'];mask=p['valid'];truth=p['truth']
            raw=p['offline'];pred=np.clip(raw,limits[:,0],limits[:,1]);indices=anchors[:,None]+np.arange(24)
            contact_by_future=contact[indices.clip(max=n-1)]
            metrics=motion_metrics(pred,truth,mask,before[anchors],stats['action_scale'].numpy(),
                raw*stats['action_scale'].numpy()+stats['action_offset'].numpy(),limits,contact_by_future)
            executed=pred[:,:12].reshape(-1,29)[:n]
            record=next(x for x in attempts if x['ordinal']==ordinal and x['model']==model)
            folder=Path(record['output'])/identity;closed=arrays(folder/'trace.npz');cp=arrays(folder/'plans.npz')
            live_raw=cp.get('raw_plans',cp['plans'])
            with h5py.File(folder/'numeric.h5') as f:
                gg=f['episode'];live_actor=gg['transition/expert_actor_input_diagnostic'][0]
                live_q=gg['transition/diagnostics_before/joint_position_rad'][0]
                live_proprio=gg['obs/proprio'][0]
            common=min(n,len(closed['target']))
            initial={}
            for name,sl in (('arm',slice(0,7)),('hand',slice(7,29))):
                initial[name+'_first_target_error_rad']=rms(closed['target'][0,sl]-targets[0,sl])
                initial[name+'_expert_first_target_change_rad']=rms(targets[0,sl]-before[0,sl])
                initial[name+'_offline_prefix12_error_rad']=rms(executed[:12,sl]-targets[:12,sl])
                initial[name+'_autonomous_prefix12_error_rad']=rms(closed['target'][:12,sl]-targets[:12,sl])
                initial[name+'_offline_common_error_rad']=rms(executed[:common,sl]-targets[:common,sl])
                initial[name+'_autonomous_common_error_rad']=rms(closed['target'][:common,sl]-targets[:common,sl])
            rows.append(dict(model=model,ordinal=ordinal,identity=identity,teacher_conditioned_metrics=metrics,
                initial_and_common=initial,
                first_plan_offline_online_max_rad=float(np.abs(raw[0]-live_raw[0]).max()),
                first_observation=dict(actor_max_abs=float(np.abs(source_actor[0]-live_actor).max()),
                    q_max_rad=float(np.abs(source_q[0]-live_q).max()),proprio_max_abs=float(np.abs(source_proprio[0]-live_proprio).max())),
                teacher_conditioned_target_fk_mm=float(np.linalg.norm(fk_pose(fk,executed[:common])[0]-fk_pose(fk,targets[:common])[0],axis=-1).mean()*1000),
                autonomous_target_fk_same_source_mm=float(np.linalg.norm(fk_pose(fk,closed['target'][:common])[0]-fk_pose(fk,targets[:common])[0],axis=-1).mean()*1000),
                offline_conditions='Original recorded expert observations and strictly past expert targets; single-anchor sequential DDIM RNG7091. No new inference.',
                limitations='Original 4-lane source is not exact hidden-state replay; matched online-expert comparisons are in analysis.json. No subtraction of aggregate RMSEs.'))
    result=dict(rows=rows,training_updates=0,new_inference=0,
        contact_scope='Existing dataset motion metric uses all-finger force, not object-filtered contact; kept separate from online expert contact events.')
    write(root/'offline_closed_gap.json',result);write(PUBLIC/'offline_closed_gap.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    with torch.no_grad():run(a.root)
