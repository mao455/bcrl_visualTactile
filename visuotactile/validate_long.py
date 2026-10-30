"""Actual-data causal alignment checks, without new simulator trajectories."""
import argparse
from pathlib import Path
import json

import h5py
import torch

from .long_controller import LongController
from .long_data import inputs_for, load_record, MANIFEST
from .models.long_horizon import LongHorizonConfig, LongHorizonPolicy
from .training_resources import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=25)
    parser.add_argument('--output-name',default='causal_acceptance.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    index = json.loads((args.root/'cache_index.json').read_text())
    identity = index['splits']['train'][0]
    record = load_record(args.root,identity)
    if args.steps<1 or args.steps>len(record['time']):
        raise ValueError('Replay length must fit this complete recorded episode')
    output = args.root/args.output_name
    if output.exists():
        raise ValueError('Preserve previous acceptance evidence; use a fresh output name')
    row = next(row for row in json.loads(MANIFEST.read_text())['episodes'] if row['id'] == identity)
    results = {}
    for group in ('G00-new','G10-new'):
        with torch.random.fork_rng(devices=[]):
            model = LongHorizonPolicy(LongHorizonConfig(group=group)).eval()
        class CachedVision:
            def __init__(self):
                self.index = 0
            def __call__(self, images):
                features = record['patches'][self.index]
                self.index += 1
                return features
        saved = dict(version=2,updates=1,epoch=0,config=vars(model.cfg),model=model.state_dict(),
            metadata=json.loads((args.root/'metadata.json').read_text()),vision={'dino_revision':index['dino_revision']})
        controller = LongController(saved,CachedVision(),device='cpu')
        controller.model.predict_from_condition = lambda condition,generator: record['reference'][:64][None]
        with torch.no_grad():
            full,state = model.encode(inputs_for(record,0,args.steps,model.cfg.tactile))
            max_error = 0.
            with h5py.File(MANIFEST.parent/row['directory']/'data.h5','r') as stream:
                episode = stream['episodes'][row['episode']]
                for step in range(args.steps):
                    names = ('episode_id','step_id','time_s','proprio','actuator_target_rad','camera_time_s',
                             'camera_valid','contact_substeps_w_N','contact_substeps_local_N','contact_time_s','contact_valid')
                    packet = {name:torch.from_numpy(episode['obs/'+name][step:step+1]) for name in names}
                    packet['image'] = torch.zeros(1,3,4,252,448)
                    packet['image'][0,:,3:] = torch.from_numpy(episode['frames/depth_m'][step])
                    packet['depth_valid'] = torch.from_numpy(episode['frames/depth_valid'][step:step+1])
                    controller.act(packet)
                    if step%12 == 0:
                        max_error = max(max_error,float((controller.last_condition[0]-full[step//12]).abs().max()))
                    command = torch.from_numpy(episode['transition/command_clipped_pre_delay'][step:step+1])
                    controller.previous_command = command
                    transition = dict(episode_id=packet['episode_id'],step_id=packet['step_id'],
                        command_clipped_pre_delay=command,
                        time_before_s=torch.from_numpy(episode['transition/time_before_s'][step:step+1]),
                        time_after_s=torch.from_numpy(episode['transition/time_after_s'][step:step+1]),
                        control_trace={name:torch.from_numpy(episode['transition/control_trace/'+name][step:step+1])
                            for name in ('valid','delayed_command','selected_command_time_s')})
                    controller.observe_execution(transition)
            state_error = float((controller.state-state).abs().max())
            if max_error > 5e-5 or state_error > 5e-5:
                raise ValueError('Actual data offline/online alignment failed')
            results[group] = dict(condition_max_error=max_error,memory_max_error=state_error,
                steps=args.steps,episode=identity,source='real recorded expert acknowledgment replay; not autonomous student control')
    atomic_json(output,dict(passed=True,results=results))
    print(results)


if __name__ == '__main__':
    main()
