"""History-only shadow intervention; never substitutes commands in the simulator."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .compact import sha256
from .native_v2_state import StateController
from .training_resources import atomic_json


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    cohort = json.loads((root/'bc_state/result.json').read_text())
    if not cohort.get('software_complete') or len(cohort['trials']) != len(session['single_seeds']):
        raise ValueError('Complete StateBC native cohort required')
    checkpoint = root/'bc_compare/state'/f"milestone_{session['bc_updates']}.pt"
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    results = []
    for seed in session['single_seeds']:
        student_path = root/'bc_state'/f'seed_{seed}'/'lane_00.h5'
        expert_path = root/'expert1'/f'seed_{seed}'/'lane_00.h5'
        with h5py.File(student_path, 'r') as student, h5py.File(expert_path, 'r') as expert:
            group = student['episode']; reference = expert['episode']
            controller = StateController(saved, json.loads(student.attrs['metadata_json']), device='cpu')
            keys = ('proprio', 'actuator_target_rad', 'episode_id', 'step_id', 'time_s', 'camera_time_s', 'camera_valid')
            packets = {key: torch.from_numpy(group['obs/'+key][:]) for key in keys}
            diagnostics = {key: torch.from_numpy(value[:]) for key, value in group['transition/diagnostics_before'].items()}
            command = torch.from_numpy(group['transition/command_clipped_pre_delay'][:])
            source = torch.from_numpy(reference['transition/command_clipped_pre_delay'][:])
            delayed = torch.from_numpy(reference['transition/control_trace/delayed_command'][:])
            before = torch.from_numpy(reference['transition/time_before_s'][:])
            selected = torch.from_numpy(reference['transition/control_trace/selected_command_time_s'][:])
            stamps = torch.from_numpy(reference['obs/time_s'][:])
            length = min(len(source), len(command)); rows = []
            with torch.no_grad():
                for tick in range(length):
                    packet = {key: value[tick:tick+1] for key, value in packets.items()}
                    diag = {key: value[tick:tick+1] for key, value in diagnostics.items()}
                    controller.act(packet, diag)
                    if tick % 12 == 0:
                        actual = controller.last_condition.clone(); counterfactual = actual.clone()
                        robot = counterfactual[0, :2340].reshape(13, 180)
                        for row_index, source_tick in enumerate(range(max(0, tick-12), tick+1), start=max(0, 12-tick)):
                            if source_tick == 0:
                                continue
                            past = torch.cat((source[source_tick-1], delayed[source_tick-1],
                                torch.stack((stamps[source_tick]-before[source_tick-1], stamps[source_tick]-selected[source_tick-1])).float()))
                            robot[row_index, 116:176] = (past-controller.model.robot_mean[116:176])/controller.model.robot_std[116:176]
                        observed = controller.model.predict(actual).clamp(-1, 1)[0]
                        shadow = controller.model.predict(counterfactual).clamp(-1, 1)[0]
                        width = min(12, length-tick)
                        target = source[tick:tick+width]
                        record = dict(tick=tick, n=width,
                            reconstructed_student_max_error=float((observed[:width]-command[tick:tick+width]).abs().max()))
                        for name, prediction in [('student_history', observed), ('expert_history_only', shadow)]:
                            for part, selection in [('arm', slice(0, 7)), ('hand', slice(7, 29))]:
                                record[name+'_'+part+'_mse'] = float((prediction[:width, selection]-target[:, selection]).square().mean())
                        record['plan_intervention_rmse'] = float((observed[:width]-shadow[:width]).square().mean().sqrt())
                        rows.append(record)
                    transition = dict(episode_id=packet['episode_id'], step_id=packet['step_id'],
                        command_clipped_pre_delay=command[tick:tick+1],
                        time_before_s=torch.from_numpy(group['transition/time_before_s'][tick:tick+1]),
                        time_after_s=torch.from_numpy(group['transition/time_after_s'][tick:tick+1]),
                        control_trace={key: torch.from_numpy(group['transition/control_trace/'+key][tick:tick+1])
                                       for key in ('valid', 'delayed_command', 'selected_command_time_s')})
                    controller.execution.observe(transition, command[tick:tick+1], packet['episode_id'], packet['step_id'])
            results.append(dict(seed=seed, compared_steps=length, anchors=rows,
                                student_sha256=sha256(student_path), expert_sha256=sha256(expert_path)))
    atomic_json(output/'result.json', dict(software_complete=True, checkpoint_sha256=sha256(checkpoint), rows=results,
        scope='Only strict past action-context116:176 replaced. Same student q/qd/target/geometry/time. '
              'Teacher history was not executed; expert command at equal tick is a descriptive reference, not on-policy oracle.',
        simulator_actions=0, optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2); run(args.root, args.output)
