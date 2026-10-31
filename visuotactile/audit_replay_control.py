"""Real-record pre-delay reference inversion through the native queued pipeline."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .action_reference import command_to_reference, reference_to_command
from .compact import sha256
from .long_data import MANIFEST
from .replay_events import TOLERANCES, error_summary
from .training_resources import atomic_json


def audit(root):
    session=json.loads((root/'session.json').read_text())
    entries={row['id']:row for row in json.loads(MANIFEST.read_text())['episodes']}
    pipeline=runpy.run_path('isaacsimenvs/tasks/play/utils/action_utils.py')['apply_action_pipeline']
    results=[]
    for identity in session['selected']:
        entry=entries[identity]
        with h5py.File(MANIFEST.parent/entry['directory']/'data.h5','r') as stream:
            metadata=json.loads(stream.attrs['metadata_json'])
            group=stream['episodes'][entry['episode']]
            commands=torch.from_numpy(group['transition/command_clipped_pre_delay'][:])
            before=torch.from_numpy(group['transition/target_before_rad'][:])
            limits=torch.tensor(metadata['joint_limits_rad'][0])
            permutation=torch.tensor([metadata['joint_order'].index(name) for name in metadata['joint_order_lab']])
            inverse=torch.argsort(permutation)
            action=metadata['config']['action']
            reference=command_to_reference(commands,before,limits,speed_scale=action['dof_speed_scale'],control_dt=metadata['control_dt'])
            recovered=reference_to_command(reference,before,limits,speed_scale=action['dof_speed_scale'],control_dt=metadata['control_dt'])
            env=SimpleNamespace(num_envs=1,device='cpu',step_dt=metadata['control_dt'],
                cfg=SimpleNamespace(action=SimpleNamespace(**action),domain_randomization=SimpleNamespace(use_action_delay=True,action_delay_max=3)),
                _perm_canon_to_lab=permutation,_action_queue=torch.zeros(1,3,29),
                _prev_targets=before[0:1,permutation].clone(),_cur_targets=before[0:1,permutation].clone(),
                _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),
                _arm_lower=limits[permutation[:7],0][None],_arm_upper=limits[permutation[:7],1][None],
                _hand_lower=limits[permutation[7:],0][None],_hand_upper=limits[permutation[7:],1][None])
            targets=[]
            delayed=[]
            env._record_action_trace=lambda command,index,start:delayed.append(command[:,inverse].clone())
            with torch.random.fork_rng(devices=[]):
                for step in range(len(commands)):
                    env.episode_length_buf=torch.as_tensor(group['transition/diagnostics_before/native_stage_step'][step:step+1]).long()
                    env._successes=torch.as_tensor(group['transition/diagnostics_before/successes'][step:step+1]).long()
                    env._recorded_delay_index=torch.as_tensor(group['transition/control_trace/delay_index'][step:step+1]).long()
                    pipeline(env,recovered[step:step+1])
                    targets.append(env._cur_targets[:,inverse].clone())
            command_error=(commands-recovered).abs()
            metrics=dict(command=error_summary(recovered.numpy(),commands.numpy(),TOLERANCES['command']),
                delayed=error_summary(torch.cat(delayed).numpy(),group['transition/control_trace/delayed_command'][:],TOLERANCES['command']),
                target=error_summary(torch.cat(targets).numpy(),group['transition/target_after_rad'][:],TOLERANCES['target_rad']),
                arm_max=float(command_error[:,:7].max()),hand_max=float(command_error[:,7:].max()),
                per_joint_max=command_error.max(0).values.tolist(),saturated_elements=int((commands.abs()>=1).sum()),
                reference_beyond_limits=int(((reference<limits[:,0])|(reference>limits[:,1])).sum()))
            results.append(dict(identity=identity,steps=len(commands),metrics=metrics))
    passed=all(row['metrics'][name]['first_exceeded'] is None for row in results for name in ('command','delayed','target'))
    report=dict(passed=passed,rows=results,source_sha256={name:sha256(name) for name in
        ('visuotactile/action_reference.py','visuotactile/long_data.py','visuotactile/long_controller.py',
         'isaacsimenvs/tasks/play/utils/action_utils.py')},scope='CPU native math with source task-stage flags, no physics or neural prediction',
        conclusion='Pre-delay equivalent reference; exactly one native delay; no new DP delay/controller layer warranted')
    atomic_json(root/'dp_control_audit.json',report)
    if not passed:
        raise ValueError('Native reference/control equivalence failed; inspect saved evidence')
    print(json.dumps(dict(passed=passed,episodes=len(results),steps=sum(row['steps'] for row in results),
        target_max=max(row['metrics']['target']['maximum'] for row in results))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)
