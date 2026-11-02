"""Real packet to current StateController versus training-window feature parity."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .native_v2_state import StateBC, BC_CONTRACT, StateController, StateWindows
from .training_resources import atomic_json


def audit(source, output):
    output.mkdir(parents=True, exist_ok=False)
    data = json.loads((source/'data.json').read_text())
    identities = data['splits']['train'] + data['splits']['validation']
    stats = torch.load(source/'statistics.pt', map_location='cpu', weights_only=False)
    model = StateBC()
    for key, value in stats.items():
        getattr(model, key).copy_(value)
    saved = dict(contract=BC_CONTRACT, updates=1, ema=model.state_dict())
    rows = []; maxima = torch.zeros(2601)
    for identity in identities:
        windows = StateWindows(source, [identity]); entry = data['entries'][identity]
        with h5py.File(entry['path'], 'r') as stream:
            group = stream['episodes'][entry['episode']]
            metadata = json.loads(stream.attrs['metadata_json'])
            controller = StateController(saved, metadata, device='cpu')
            keys = ('proprio', 'actuator_target_rad', 'episode_id', 'step_id', 'time_s', 'camera_time_s', 'camera_valid')
            packet_arrays = {key: torch.from_numpy(group['obs/'+key][:]) for key in keys}
            diagnostics = {key: torch.from_numpy(value[:]) for key, value in group['transition/diagnostics_before'].items()}
            command = torch.from_numpy(group['transition/command_clipped_pre_delay'][:])
            delayed = torch.from_numpy(group['transition/control_trace/delayed_command'][:])
            selected_time = torch.from_numpy(group['transition/control_trace/selected_command_time_s'][:])
            time_before = torch.from_numpy(group['transition/time_before_s'][:])
            time_after = torch.from_numpy(group['transition/time_after_s'][:])
            conditions = []
            with torch.no_grad():
                for tick in range(len(command)):
                    packet = {key: value[tick:tick+1] for key, value in packet_arrays.items()}
                    diag = {key: value[tick:tick+1] for key, value in diagnostics.items()}
                    controller.act(packet, diag)
                    if tick % 12 == 0:
                        conditions.append(controller.last_condition[0].clone())
                    controller.execution.observe(dict(command_clipped_pre_delay=command[tick:tick+1],
                        episode_id=packet['episode_id'], step_id=packet['step_id'],
                        time_after_s=time_after[tick:tick+1],
                        time_before_s=time_before[tick:tick+1], control_trace=dict(delayed_command=delayed[tick:tick+1],
                        selected_command_time_s=selected_time[tick:tick+1], valid=torch.ones(1, dtype=torch.bool))),
                        command[tick:tick+1], packet['episode_id'], packet['step_id'])
            expected = torch.cat((windows.robot, windows.geometry), -1)
            difference = (torch.stack(conditions)-expected).abs().amax(0)
            maxima = torch.maximum(maxima, difference)
            rows.append(dict(identity=identity, anchors=len(windows), maximum=float(difference.max())))
        if len(rows) % 20 == 0:
            atomic_json(output/'progress.json', dict(episodes=len(rows), maximum=float(maxima.max())))
    np.save(output/'per_feature_max_error.npy', maxima.numpy())
    atomic_json(output/'result.json', dict(software_complete=True, episodes=len(rows), rows=rows,
        maximum=float(maxima.max()), features=2601, anchors=sum(row['anchors'] for row in rows),
        optimizer_updates=0, policy_commands_executed=0, test_used=False,
        scope='Actual online StateController packet assembly, with explicitly teacher-forced source execution history'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args(); torch.set_num_threads(2)
    audit(arguments.source, arguments.output)
