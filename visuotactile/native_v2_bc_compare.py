"""Matched small-budget Actor-observation and current-State BC diagnostics."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import h5py
import torch
from torch import nn

from .compact import sha256
from .models.vendor.ema_model import EMAModel
from .native_v2_gates import require_training_foundation
from .native_v2_resources import LeaseV2
from .native_v2_state import BC_CONTRACT, StateBC, StateWindows
from .native_v2_training import next_batch
from .train_long import cpu_snapshot, save
from .training_resources import atomic_json


ACTOR_CONTRACT = dict(BC_CONTRACT, diagnostic_model='ActorBC_chunk_mlp_v1',
                      observation='recorded_expert_actor141_native_history13', diagnostic_only=True)


class ActorBC(StateBC):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(13*143, 256), nn.Mish(), nn.Linear(256, 256),
                                       nn.Mish(), nn.Linear(256, 64*29))
        self.register_buffer('actor_mean', torch.zeros(141))
        self.register_buffer('actor_std', torch.ones(141))


def actor_history(values, times, tick, mean, std):
    history = torch.zeros(13, 143, dtype=values.dtype, device=values.device)
    start = max(0, tick-12)
    count = tick-start+1
    history[-count:, :141] = (values[start:tick+1]-mean)/std
    history[-count:, 141] = (times[tick]-times[start:tick+1]).float()
    history[-count:, 142] = 1.
    return history.flatten()


class ActorController:
    def __init__(self, saved, device='cuda:0'):
        if saved.get('contract') != ACTOR_CONTRACT or saved.get('updates', 0) < 1:
            raise ValueError('Trained diagnostic ActorBC checkpoint required')
        with torch.random.fork_rng(devices=[]):
            self.model = ActorBC().to(device)
        self.model.load_state_dict(saved['ema']); self.model.eval()
        self.device = device; self.values = []; self.times = []; self.plan = None; self.tick = 0

    @torch.no_grad()
    def act(self, actor, packet):
        if int(packet['step_id'][0]) != self.tick:
            raise ValueError('ActorBC native observation clock mismatch')
        self.values.append(actor[0].to(self.device)); self.times.append(packet['time_s'][0].to(self.device))
        self.values = self.values[-13:]; self.times = self.times[-13:]
        condition = actor_history(torch.stack(self.values), torch.stack(self.times), len(self.values)-1,
                                  self.model.actor_mean, self.model.actor_std)[None]
        if self.tick % 12 == 0:
            self.plan = self.model.predict(condition).clamp(-1, 1).cpu()
        command = self.plan[:, self.tick % 12]
        self.tick += 1
        return command

    def observe_execution(self, transition):
        return None


def actor_records(source, identities):
    data = json.loads((source/'data.json').read_text())
    records = {}
    for identity in identities:
        entry = data['entries'][identity]
        with h5py.File(entry['path'], 'r') as stream:
            group = stream['episodes'][entry['episode']]
            values = torch.from_numpy(group['transition/expert_actor_input_diagnostic'][:])
            if values.shape[1] != 141 or not torch.isfinite(values).all():
                raise ValueError('Recorded expert Actor input contract mismatch')
            records[identity] = values
    return records


def features(windows, actor, mean, std):
    return torch.stack([actor_history(actor[identity], windows.records[identity]['time'], tick, mean, std)
                        for identity, tick in windows.samples])


@torch.no_grad()
def evaluate(model, values, windows, lease):
    total = torch.zeros(2); count = 0
    for start in range(0, len(windows), 32):
        with lease.block():
            indices = slice(start, start+32)
            prediction = model.predict(values[indices].cuda()).clamp(-1, 1).cpu()[:, :12]
            target = windows.command[indices, :12]; valid = windows.valid[indices, :12]
            for group_index, selection in enumerate((slice(0, 7), slice(7, 29))):
                errors = (prediction[..., selection]-target[..., selection]).square().mean(-1)
                total[group_index] += torch.where(valid, errors, 0.).sum()
            count += int(valid.sum())
    return dict(arm_rmse=float((total[0]/count).sqrt()), hand_rmse=float((total[1]/count).sqrt()),
                selection_mse=float(total.mean()/count), valid_commands=count,
                scope='Expert teacher-forced inputs; not autonomous success')


def train(root, output):
    require_training_foundation(root)
    session = json.loads((root/'session.json').read_text())
    source = Path(session['source_root']); data = json.loads((source/'data.json').read_text())
    identities = session['bc_train_ids']; validation_ids = session['bc_validation_ids']
    if not set(identities) <= set(data['splits']['train']) or not set(validation_ids) <= set(data['splits']['validation']):
        raise ValueError('Diagnostic split leakage')
    output.mkdir(parents=True, exist_ok=False)
    train_windows = StateWindows(source, identities); validation_windows = StateWindows(source, validation_ids)
    records = actor_records(source, data['splits']['train'] + validation_ids)
    all_train = torch.cat([records[identity] for identity in data['splits']['train']])
    mean = all_train.mean(0); std = all_train.std(0, unbiased=False).clamp_min(1e-3)
    atomic_json(output/'configuration.json', dict(train_ids=identities, validation_ids=validation_ids,
        actor_normalizer_ids=data['splits']['train'], actor_dimension=141, actor_history=13,
        state_dimension=2601, horizon=64, commit=12, hidden=[256, 256], max_updates=session['bc_updates'],
        lr=1e-4, warmup=500, loss='B', batch_anchors=32, data_sha256=sha256(source/'data.json'),
        actor_noise='Original recorded noisy/delayed actor input; online original observation builder',
        input_only_comparison='State13 robot+3 geometry versus Actor13; common hidden/output layers and labels'))
    lease = LeaseV2(root, deadline=session['deadline']); results = {}
    for kind in ('state', 'actor'):
        directory = output/kind; directory.mkdir()
        train_values = (torch.cat((train_windows.robot, train_windows.geometry), -1) if kind == 'state'
                        else features(train_windows, records, mean, std))
        validation_values = (torch.cat((validation_windows.robot, validation_windows.geometry), -1) if kind == 'state'
                             else features(validation_windows, records, mean, std))
        with lease.block():
            torch.manual_seed(20261102)
            model = (StateBC() if kind == 'state' else ActorBC()).cuda()
            for key, value in train_windows.stats.items():
                getattr(model, key).copy_(value.cuda())
            if kind == 'actor':
                model.actor_mean.copy_(mean.cuda()); model.actor_std.copy_(std.cuda())
            averaged = copy.deepcopy(model).eval().requires_grad_(False)
            ema = EMAModel(averaged, power=.75)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
        epoch = cursor = 0; began = time.monotonic(); history = []
        for update in range(1, session['bc_updates']+1):
            indices, epoch, cursor = next_batch(len(train_windows), epoch, cursor, 32)
            with lease.block():
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = model.loss(train_values[indices].cuda(), train_windows.command[indices].cuda(),
                                  train_windows.valid[indices].cuda())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite diagnostic BC loss')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite diagnostic BC gradient')
                scale = min(1., update/500)*.5*(1+math.cos(math.pi*max(0, update-500)/max(1, session['bc_updates']-500)))
                for parameter_group in optimizer.param_groups:
                    parameter_group['lr'] = 1e-4*scale
                optimizer.step(); ema.step(model)
            row = dict(updates=update, loss=float(loss.detach()), gradient_norm=float(norm),
                       lr=optimizer.param_groups[0]['lr'], elapsed_s=time.monotonic()-began,
                       work_s=lease.work_s, wait_s=lease.wait_s, pacing_s=lease.pacing_s)
            with (directory/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if update in (1, 250, session['bc_updates']):
                report = dict(updates=update, train=evaluate(averaged, train_values, train_windows, lease),
                              validation=evaluate(averaged, validation_values, validation_windows, lease))
                history.append(report); atomic_json(directory/'metrics.json', history)
                payload = cpu_snapshot(dict(contract=BC_CONTRACT if kind == 'state' else ACTOR_CONTRACT,
                    group='S-NativeDP', diagnostic_only=True, updates=update, epoch=epoch,
                    model=model.state_dict(), ema=averaged.state_dict(), optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    train_ids=identities, validation_ids=validation_ids))
                save(directory/f'milestone_{update}.pt', payload)
        results[kind] = dict(history=history, checkpoint=str((directory/f'milestone_{update}.pt').resolve()),
                             checkpoint_sha256=sha256(directory/f'milestone_{update}.pt'))
    atomic_json(output/'result.json', dict(software_complete=True, models=results, test_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train(arguments.root, arguments.output)
