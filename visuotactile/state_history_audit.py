"""Actual-record Actor normalization, episode history and deployment parity."""
import argparse
import json
from pathlib import Path

import h5py
import torch

from .native_v2_actor_audit import normalized
from .state_history import HistoryBank, HistoryController, PersistentState, STATE_CONTRACT, load_records
from .training_resources import atomic_json


@torch.no_grad()
def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text()); source = Path(session['source_root'])
    data = json.loads((source/'data.json').read_text())
    expert = torch.load(session['expert_checkpoint'], map_location='cpu', weights_only=False)[0]['model']
    stats = torch.load(source/'statistics.pt', map_location='cpu', weights_only=False)
    torch.manual_seed(20261102); model = PersistentState('bc').eval(); model.set_statistics(stats, expert)
    rows = []; raw_rows = 0; norm_error = 0.
    # Exact original Actor rows from all400, including held-out observations for
    # schema validation only. No held-out labels or outcomes select a model.
    for identity, entry in data['entries'].items():
        with h5py.File(entry['path'], 'r') as stream:
            actor = torch.from_numpy(stream['episodes'][entry['episode']]['transition/expert_actor_input_diagnostic'][:])
        expected = normalized(actor, expert)[:, :140]
        actual = model.actor_input(actor)
        norm_error = max(norm_error, float((expected-actual[:, :140]).abs().max()))
        if not torch.equal(actual[:, 140:], expert['a2c_network.extra_params'][0].expand(len(actor), -1)):
            raise ValueError('Frozen group50 embedding mismatch')
        raw_rows += len(actor)
    identities = [s['identity'] for s in session['saved_specs']]
    for identity, record in load_records(source, identities).items():
        full, anchors, final_hidden = model.encode(record)
        bank = HistoryBank(); sequential = []
        for tick in range(len(record['time'])):
            value = bank.observe(0, 7, tick, model, record['actor'][tick], record['robot'][tick],
                record['geometry'][tick], record['time'][tick])
            if value is not None:
                sequential.append(value)
        streaming_error = float((full-torch.cat(sequential)).abs().max())
        segment_error = 0.
        for start in range(0, len(record['time']), 96):
            conditions, selected, _ = model.encode(record, start, min(start+96, len(record['time'])))
            segment_error = max(segment_error, float((conditions-full[selected//12]).abs().max()))
        entry = data['entries'][identity]
        with h5py.File(entry['path'], 'r') as stream:
            metadata = json.loads(stream.attrs['metadata_json']); group = stream['episodes'][entry['episode']]
            saved = dict(contract=STATE_CONTRACT, kind='bc', updates=1, ema=model.state_dict())
            controller = HistoryController(saved, metadata, device='cpu')
            max_input = dict(actor=0., robot=0., geometry=0.); condition_error = 0.
            for tick in range(len(record['time'])):
                packet = {key: torch.from_numpy(value[tick:tick+1]) for key, value in group['obs'].items()
                    if isinstance(value, h5py.Dataset)}
                diagnostics = {key: torch.from_numpy(value[tick:tick+1]) for key, value in group['transition/diagnostics_before'].items()}
                controller.act(record['actor'][tick:tick+1], packet, diagnostics)
                for key in max_input:
                    max_input[key] = max(max_input[key], float((controller.last_inputs[key]-record[key][tick]).abs().max()))
                if tick % 12 == 0:
                    condition_error = max(condition_error, float((controller.last_condition-full[tick//12]).abs().max()))
                transition = dict(command_clipped_pre_delay=record['command'][tick:tick+1],
                    episode_id=packet['episode_id'], step_id=packet['step_id'],
                    time_after_s=torch.from_numpy(group['transition/time_after_s'][tick:tick+1]),
                    control_trace={key: torch.from_numpy(value[tick:tick+1]) for key, value in group['transition/control_trace'].items()},
                    time_before_s=record['time'][tick:tick+1])
                # This audit explicitly executes source feedback (teacher forcing)
                # through the real acknowledgment API, never model predictions.
                controller.command = record['command'][tick:tick+1]
                controller.observe_execution(transition)
        rows.append(dict(identity=identity, ticks=len(record['time']), anchors=len(anchors),
            streaming_condition_max=streaming_error, current_weight_prefix_max=segment_error,
            deployment_input_max=max_input, deployment_condition_max=condition_error))
    passed = norm_error == 0 and all(r['streaming_condition_max'] < 1e-5 and r['current_weight_prefix_max'] < 1e-5
        and r['deployment_condition_max'] < 1e-4 and r['deployment_input_max']['actor'] == 0
        and r['deployment_input_max']['geometry'] == 0 for r in rows)
    atomic_json(output/'result.json', dict(software_complete=True, passed=passed, actor_rows=raw_rows,
        actor_normalization_max=norm_error, group_embedding_exact=True, rows=rows,
        scope='CPU actual source data/controller parity, no task-performance inference'))
    if not passed:
        raise ValueError('State input/history numerical parity failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); torch.set_num_threads(2)
    audit(a.root, a.output)
