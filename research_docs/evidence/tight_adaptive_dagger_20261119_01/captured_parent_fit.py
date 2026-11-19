"""Parent predictions actually recorded throughout the new correction histories."""
import argparse
import collections
import json

import h5py
import numpy as np
import torch

from tools.tight_adaptive_dagger import ROOT, PUBLIC, read, write
from tools.tight_frozen_comparison import table
from scripts.tight_balance_20261117.diagnose import phases
from visuotactile.geometry_sampling import STAGES
from visuotactile.position_control import reachable_position_bounds, REACHABLE_POSITION_EXECUTION


def export(round_index):
    torch.set_num_threads(2)
    root=ROOT/f'round{round_index:02d}/collection_combined'
    output=PUBLIC/f'round{round_index:02d}/captured_parent_fit';output.mkdir(parents=True,exist_ok=True)
    entries=read(root/'correction_entries.json');plan=read(root/'plan.json')
    rows=[]
    for case in plan['cases']:
        if case['name'] not in entries:continue
        role=case['role'];identity=case['name'];folder=root/identity
        result=read(folder/'result.json')
        pair=plan['pairs'][role]
        assert result['complete_student_policy']['frontend_sha256']==pair['frontend_sha256']
        assert result['complete_student_policy']['backend_sha256']==pair['backend_sha256']
        with h5py.File(folder/'numeric.h5') as stream:
            t=stream['episode/transition'];d=t['dagger']
            command=t['target_after_rad'][:];reached=t['diagnostics_after/joint_position_rad'][:]
            legal=d['expert_label_valid'][:].reshape(-1).astype(bool)
            control=d['control_source'][:].reshape(-1).astype(bool)
            nominal=d['nominal'][:];proposal=d['student_proposed_target_rad'][:]
            assert np.max(np.abs(command[~control]-proposal[~control]))<=1e-6
            assert np.array_equal(d['u_expert'][:][legal],command[legal])
            assert np.array_equal(d['q_reached'][:][legal],reached[legal])
            assert int(legal.sum())==result['dagger']['expert_valid_ticks']
            metadata=json.loads(stream.attrs['metadata_json'])
            limits=torch.as_tensor(metadata['joint_limits_rad'],dtype=torch.float32).reshape(-1,29,2)[0]
            bounds=reachable_position_bounds(torch.from_numpy(t['target_before_rad'][:]),limits,REACHABLE_POSITION_EXECUTION)
            span=(bounds[...,1]-bounds[...,0]).clamp_min(1e-6).numpy()
            phase=phases(t);onset=np.zeros_like(legal);onset[np.flatnonzero(legal)[:24]]=True
            masks=dict(legal=legal,first24=onset)
            masks.update({name:legal&(phase==k) for k,name in enumerate(STAGES)})
            for stage,mask in masks.items():
                if not mask.any():continue
                for component,pred,target in [('dp_own_target',nominal,reached if role=='A' else command),
                                               ('dp_command',nominal,command),('final',proposal,command)]:
                    error=(pred-target)[mask];scaled=error/span[mask]
                    rows.append(dict(role=role,identity=identity,stage=stage,component=component,ticks=int(mask.sum()),
                        arm_squared_sum=float((error[:,:7]**2).sum()),hand_squared_sum=float((error[:,7:]**2).sum()),
                        arm_span_squared_sum=float((scaled[:,:7]**2).sum()),hand_span_squared_sum=float((scaled[:,7:]**2).sum())))
    totals=collections.defaultdict(collections.Counter)
    for r in rows:
        totals[r['role'],r['stage'],r['component']].update({k:r[k] for k in ('ticks','arm_squared_sum','hand_squared_sum','arm_span_squared_sum','hand_span_squared_sum')})
    summary=[]
    for (role,stage,component),v in totals.items():
        summary.append(dict(role=role,stage=stage,component=component,ticks=v['ticks'],
            arm_rmse_rad=(v['arm_squared_sum']/v['ticks']/7)**.5,hand_rmse_rad=(v['hand_squared_sum']/v['ticks']/22)**.5,
            arm_span_rmse=(v['arm_span_squared_sum']/v['ticks']/7)**.5,hand_span_rmse=(v['hand_span_squared_sum']/v['ticks']/22)**.5))
    table(output/'per_episode.csv',rows);table(output/'summary.csv',summary)
    write(output/'scope.json',dict(accepted=len(entries),rows=len(rows),parents=plan['pairs'],
        scope='Parent source student predictions recorded online with complete actual sensor and executed-action histories, including expert suffix. Only clean executed expert labels used as truth. Each role is evaluated on its own contributed trajectories.',
        limitation='Suffix student proposals were not executed. Capture used live sampler7093; separately exported learned-model offline caches use7091, so do not interpret differences as exact matched-noise changes.',
        training_effect='None; no prefix masks, labels, models or sources changed',optimizer_updates=0,physics_updates=0))
    print(dict(round=round_index,accepted=len(entries),summary=[r for r in summary if r['stage']=='legal']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True)
    export(p.parse_args().round)
