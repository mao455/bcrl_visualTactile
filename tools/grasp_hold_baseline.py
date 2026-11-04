"""Evaluate a zero-motion target hold on exactly the existing validation40 ticks."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from tools.state_image_queue import read,write
from tools.grasp_diagnosis_analysis import BASE,PUBLIC


def run(root):
    session=read(root/'session.json');source=read(Path(session['source_root'])/'data.json')
    errors=[];rows=[]
    for identity in source['splits']['validation']:
        entry=source['entries'][identity]
        with h5py.File(entry['path']) as f:
            g=f['episodes'][entry['episode']]
            target=g['transition/target_after_rad'][:];before=g['transition/target_before_rad'][:]
        anchors=np.arange(len(target))//12*12
        error=before[anchors]-target;errors.append(error)
        rows.append(dict(identity=identity,ticks=len(target),arm_rmse_rad=float(np.sqrt(np.mean(error[:,:7]**2))),
            hand_rmse_rad=float(np.sqrt(np.mean(error[:,7:]**2)))))
    err=np.concatenate(errors);metrics={}
    for name in ('state','image'):
        history=read(BASE/'state_image_20261104_02'/(name+'_train')/'metrics.json')
        measured=next(x for x in history if x['update']==10000)['validation']
        assert sorted(x['identity'] for x in measured['episodes'])==sorted(x['identity'] for x in rows)
        metrics[name]=dict(arm_rmse_rad=measured['arm_rmse'],hand_rmse_rad=measured['hand_rmse'],
            moving_arm_direction_cosine=measured['motion_mean_cosine_amplitude'][0],
            moving_arm_amplitude_ratio=measured['motion_mean_cosine_amplitude'][1],
            scope=measured['scope'])
    result=dict(episodes=40,ticks=len(err),test_used=False,new_model_inference=0,training_updates=0,
        definition='At each frozen anchor tick0,12,... hold the current observed target_before for its next12 valid ticks; same unweighted per-joint/tick RMSE as saved final validation40 metrics.',
        hold=dict(arm_rmse_rad=float(np.sqrt(np.mean(err[:,:7]**2))),hand_rmse_rad=float(np.sqrt(np.mean(err[:,7:]**2)))),
        model_metrics=metrics,rows=rows,
        limitation='Teacher-forced zero-motion counterfactual; not a simulated controller or task success result.')
    write(root/'validation_hold_baseline.json',result);write(PUBLIC/'validation_hold_baseline.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
