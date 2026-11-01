"""All400 pre-delay commands through the actual unchanged native action function."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import h5py
import torch
from .compact import sha256
from .training_resources import atomic_json


def audit(source,output):
    pipeline=runpy.run_path('isaacsimenvs/tasks/play/utils/action_utils.py')['apply_action_pipeline']
    data=json.loads((source/'data.json').read_text());rows=[]
    with torch.random.fork_rng(devices=[]):
        for identity,entry in data['entries'].items():
            with h5py.File(entry['path'],'r') as f:
                m=json.loads(f.attrs['metadata_json']);g=f['episodes'][entry['episode']]
                def read(key):return torch.from_numpy(g['transition/'+key][:])
                commands=read('command_clipped_pre_delay');before=read('target_before_rad')
                expected=read('target_after_rad');expected_delayed=read('control_trace/delayed_command')
                limits=torch.tensor(m['joint_limits_rad'][0]);perm=torch.tensor([m['joint_order'].index(n) for n in m['joint_order_lab']]);inv=torch.argsort(perm)
                env=SimpleNamespace(num_envs=1,device='cpu',step_dt=m['control_dt'],
                    cfg=SimpleNamespace(action=SimpleNamespace(**m['config']['action']),domain_randomization=SimpleNamespace(use_action_delay=True,action_delay_max=3)),
                    _perm_canon_to_lab=perm,_action_queue=torch.zeros(1,3,29),_prev_targets=before[:1,perm].clone(),_cur_targets=before[:1,perm].clone(),
                    _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),_arm_lower=limits[perm[:7],0][None],_arm_upper=limits[perm[:7],1][None],
                    _hand_lower=limits[perm[7:],0][None],_hand_upper=limits[perm[7:],1][None])
                stage=read('diagnostics_before/native_stage_step').long();goals=read('diagnostics_before/successes').long()
                delay=read('control_trace/delay_index').long();targets=[];delayed=[]
                env._record_action_trace=lambda command,index,start:delayed.append(command[:,inv].clone())
                for tick in range(len(commands)):
                    env.episode_length_buf=stage[tick:tick+1];env._successes=goals[tick:tick+1];env._recorded_delay_index=delay[tick:tick+1]
                    pipeline(env,commands[tick:tick+1]);targets.append(env._cur_targets[:,inv].clone())
                target_error=float((torch.cat(targets)-expected).abs().max());delay_error=float((torch.cat(delayed)-expected_delayed).abs().max())
                rows.append(dict(identity=identity,steps=len(commands),target_max_error=target_error,delayed_max_error=delay_error))
    result=dict(passed=all(r['target_max_error']<=1e-6 and r['delayed_max_error']==0 for r in rows),episodes=len(rows),steps=sum(r['steps'] for r in rows),
        target_max_error=max(r['target_max_error'] for r in rows),delayed_max_error=max(r['delayed_max_error'] for r in rows),rows=rows,
        native_action_sha256=sha256('isaacsimenvs/tasks/play/utils/action_utils.py'),scope='Actual native queue/action math, recorded delay indices and source stage flags; no simulation or controller changes')
    atomic_json(output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if not result['passed']:raise ValueError('Native control mismatch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.source,a.output)
