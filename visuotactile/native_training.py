"""Full-prefix training, fixed validation, atomic recovery and native DP gates."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import torch

from .compact import sha256
from .native_data import CONTRACT, Episode, GROUPS, labels
from .models.native_dp import NativePolicy, encode_segment, load_policy
from .models.vendor.ema_model import EMAModel
from .native_resources import NativeLease
from .train_long import cpu_snapshot, save
from .training_resources import atomic_json


def fit_passed(metrics, count):
    thresholds = (.20, .10) if count == 1 else (.35, .20)
    return all(metrics[part+'_rmse'] <= threshold and metrics[part+'_rmse'] <= .7*metrics['zero_'+part+'_rmse']
               for part, threshold in zip(('arm', 'hand'), thresholds))


def common_budget(available_seconds, pilots, validation_seconds):
    per_update = sum(pilots.values())
    if set(pilots) != set(GROUPS) or per_update <= 0 or available_seconds <= 0:
        raise ValueError('All three measured pilot timings required')
    budget = min(20000, int(.85*available_seconds/(per_update+sum(validation_seconds.values())/1000))//1000*1000)
    if budget < 5000:
        raise RuntimeError('Insufficient measured budget for 5000 formal updates per model')
    return budget


def initialize_model(root, group, device):
    model = NativePolicy(group).to(device)
    statistics = torch.load(root/'statistics.pt', map_location='cpu', weights_only=False)
    for name, value in statistics.items():
        getattr(model, name).copy_(value.to(device))
    return model


def segments(root, identities, epoch, seed=20261101):
    index = json.loads((root/'data.json').read_text())['entries']
    order = list(identities)
    random.Random(seed+epoch).shuffle(order)
    return [(identity, start, min(start+96, index[identity]['length']))
            for identity in order for start in range(0, index[identity]['length'], 96)]


@torch.no_grad()
def metrics(model, root, identities, lease, seed=7091):
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = dict(arm=0., hand=0., zero_arm=0., zero_hand=0., count=0, anchors=0)
    began = time.monotonic()
    for identity in identities:
        episode = Episode(root, identity, model.group)
        for start in range(0, len(episode), 96):
            with lease.block():
                condition, anchors, _ = encode_segment(model, episode, start, min(start+96, len(episode)))
                prediction = model.predict(condition, generator).clamp(-1, 1)[:, :12]
                command, valid = labels(episode.record, anchors)
                command, valid = command[:, :12].to(device), valid[:, :12].to(device)
                for part, selection in (('arm', slice(0, 7)), ('hand', slice(7, 29))):
                    totals[part] += float(((prediction[..., selection]-command[..., selection]).square().mean(-1)*valid).sum())
                    totals['zero_'+part] += float((command[..., selection].square().mean(-1)*valid).sum())
                totals['count'] += int(valid.sum())
                totals['anchors'] += len(anchors)
    if not totals['count']:
        raise ValueError('Empty evaluation')
    return dict(**{name+'_rmse': math.sqrt(totals[name]/totals['count']) for name in
                  ('arm', 'hand', 'zero_arm', 'zero_hand')},
        selection_mse=.5*(totals['arm']+totals['hand'])/totals['count'],
        episodes=len(identities), valid_targets=totals['count'], anchors=totals['anchors'],
        elapsed_s=time.monotonic()-began, seed=seed,
        scope='EMA sampled first12 native commands, clipped exactly as deployment; teacher-forced history')


def network_gate(root, group, output, lease):
    data = json.loads((root/'data.json').read_text())
    episode = Episode(root, data['splits']['train'][0], group)
    dino_evidence = None
    dino = None
    if group == 'I-HistoryDP':
        from .models.native_vision import NativeDinoSpatial
        if sha256(Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth') != data['dino_weights_sha256']:
            raise ValueError('Frozen DINO weights changed since cache creation')
        with lease.block():
            dino = NativeDinoSpatial().to('cuda:0').eval()
            if any(parameter.requires_grad for parameter in dino.parameters()):
                raise ValueError('DINO must remain frozen')
            rgb = Episode(root, data['splits']['train'][0], 'I-OfficialDP').frame(0, 'cuda:0')['rgb']
            actual = dino(rgb).cpu()
            cached = episode.perception['patches'][0]
            difference = float((actual.float()-cached.float()).abs().max())
            if difference > .005:
                atomic_json(output/'cache_failure.json', dict(max_abs=difference, threshold=.005))
                raise ValueError('Pinned DINO perception cache reproduction failed')
            dino_evidence = dict(frozen=True, parameters=sum(parameter.numel() for parameter in dino.parameters()),
                cache_max_abs_difference=difference, cache_precision='FP16',
                backbone_compute='FP32 transformer; TF32-enabled convolution; fixed six-image batch')
            del rgb, actual, cached
    with lease.block():
        torch.manual_seed(20261101)
        model = initialize_model(root, group, 'cuda:0')
        model.train()
        condition, anchors, _ = encode_segment(model, episode, 0, min(24, len(episode)))
        command, valid = labels(episode.record, anchors)
        loss = model.loss(condition, command.cuda(), valid.cuda())
        loss.backward()
        gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        if not torch.isfinite(loss) or any(not math.isfinite(value) for value in gradients.values()):
            raise FloatingPointError('Network gate nonfinite loss/gradient')
        required = ['robot_encoder', 'denoiser', 'condition_projection']
        required += ['native_gru', 'visual_gru', 'vision.rgb', 'vision.depth'] if group == 'I-HistoryDP' else []
        if any(not any(value > 0 for name, value in gradients.items() if name.startswith(prefix)) for prefix in required):
            raise ValueError('Required trainable branch has no effective gradient')
        model.zero_grad(set_to_none=True)
        robot = episode.record['robot'][0].cuda().requires_grad_()
        stamp = episode.record['time'][0].cuda()
        frame = episode.frame(0, 'cuda:0')
        current = model.condition(model.observe(robot, stamp, 0, frame), stamp)
        target_gradient = torch.autograd.grad(current.square().sum(), robot)[0][58:87]
        altered = robot.detach().clone()
        altered[58:87] += .05
        altered[87:116] += .05
        intervention = model.condition(model.observe(altered, stamp, 0, frame), stamp)
        difference = float((current-intervention).abs().max())
        if difference <= 1e-7 or not torch.isfinite(target_gradient).all() or target_gradient.norm() <= 0:
            raise ValueError('Controller target is not connected to condition')
        del current, intervention, loss, condition
        model.eval()
    with torch.no_grad():
        full_conditions = []
        raw_episode = Episode(root, episode.record['identity'], 'I-OfficialDP') if dino is not None else None
        with lease.block():
            full, _, full_memory = encode_segment(model, episode, 0, len(episode))
        memory = None
        for tick in range(len(episode)):
            with lease.block():
                stamp = episode.record['time'][tick].cuda()
                frame = episode.frame(tick, 'cuda:0') if tick % 12 == 0 else None
                if dino is not None and frame is not None:
                    raw = raw_episode.frame(tick, 'cuda:0')
                    frame['patches'] = dino(raw['rgb'].float()/255.)
                memory = model.observe(episode.record['robot'][tick].cuda(), stamp, tick,
                    frame, memory)
                if tick % 12 == 0:
                    full_conditions.append(model.condition(memory, stamp))
        online = torch.cat(full_conditions)
        condition_error = float((full-online).abs().max())
        memory_error = max([float((getattr(memory, name)-getattr(full_memory, name)).abs().max())
                            for name in ('native', 'visual') if getattr(memory, name) is not None] or [0.])
        action_error = 0.
        for start in range(0, len(full), 8):
            with lease.block():
                first = model.predict(full[start:start+8], torch.Generator(device='cuda:0').manual_seed(7091+start))
                second = model.predict(online[start:start+8], torch.Generator(device='cuda:0').manual_seed(7091+start))
                action_error = max(action_error, float((first-second).abs().max()))
        if max(condition_error, memory_error) > 1e-5 or action_error > 1e-4:
            raise ValueError('Complete-episode FP32 online/offline parity gate failed')
    atomic_json(output/'result.json', dict(passed=True, group=group, real_episode=episode.record['identity'],
        parameters=sum(parameter.numel() for parameter in model.parameters()), gradient_norms=gradients,
        controller_target_condition_delta=difference, controller_target_gradient=float(target_gradient.norm()),
        condition_max_error=condition_error, memory_max_error=memory_error, action_max_error=action_error,
        production_precision='FP32; no unvalidated mixed-precision promotion', native_ticks=len(episode)))
    if dino_evidence is not None:
        atomic_json(output/'dino_frozen.json', dino_evidence)


def train(root, group, stage, output, lease, resume=False):
    data = json.loads((root/'data.json').read_text())
    session = json.loads((root/'session.json').read_text())
    identities = data['splits']['train'][:1 if stage == 'fit1' else 4] if stage.startswith('fit') else data['splits']['train']
    total = {'fit1': 1000, 'fit4': 2000, 'pilot': 20}.get(stage)
    if stage == 'formal':
        total = json.loads((root/'frozen.json').read_text())['common_updates']
    configuration = dict(contract=CONTRACT, group=group, stage=stage, total_updates=total,
        identities=identities, split=data['splits'], source_hashes=session['source_hashes'],
        data_sha256=sha256(root/'data.json'), statistics_sha256=sha256(root/'statistics.pt'),
        optimizer=dict(lr=1e-4, betas=[.95, .999], weight_decay=1e-6), warmup=500,
        precision='FP32', gradient_accumulation_segments=4, segment_ticks=96,
        schedule='cosine', selection='EMA sampled first12 arm/hand equal native command MSE')
    if group == 'I-HistoryDP':
        configuration['native_perception_sha256'] = sha256(root/'native_perception.json')
    atomic_json(output/'configuration.json', configuration)
    with lease.block():
        torch.manual_seed(20261101)
        model = initialize_model(root, group, 'cuda:0')
        averaged = copy.deepcopy(model).eval().requires_grad_(False)
        ema = EMAModel(averaged, power=.75)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
    updates = epoch = cursor = anchors_seen = segments_seen = 0
    best, stagnation, history = math.inf, 0, []
    if resume:
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        if saved['configuration'] != configuration:
            raise ValueError('Resume identity mismatch; no runtime source migration')
        with lease.block():
            model.load_state_dict(saved['model'])
            averaged.load_state_dict(saved['ema'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['cpu_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'])
        updates, epoch, cursor = saved['updates'], saved['epoch'], saved['cursor']
        anchors_seen, segments_seen = saved['anchors_seen'], saved['segments_seen']
        best, stagnation, history = saved['best'], saved['stagnation'], saved['history']
        ema.optimization_step = saved['ema_step']
        del saved
    began = time.monotonic()
    initial_updates = updates
    def checkpoint(best_path=False):
        payload = cpu_snapshot(dict(contract=CONTRACT, group=group, configuration=configuration,
            model=model.state_dict(), ema=averaged.state_dict(), optimizer=optimizer.state_dict(),
            updates=updates, epoch=epoch, cursor=cursor, anchors_seen=anchors_seen, segments_seen=segments_seen,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), ema_step=ema.optimization_step,
            scheduler=dict(last_update=updates, total=total, warmup=500),
            best=best, stagnation=stagnation, history=history))
        save(output/('best.pt' if best_path else 'last.pt'), payload)
        if updates == 1:
            readback = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
            if readback['updates'] != 1 or readback['contract'] != CONTRACT:
                raise ValueError('Initial real-update checkpoint readback failed')
            atomic_json(output/'startup_verified.json', dict(updates=1, checkpoint_sha256=sha256(output/'last.pt'),
                readback=True, group=group, stage=stage, formal_training=stage == 'formal'))
    try:
        while updates < total:
            schedule = segments(root, identities, epoch)
            batch = schedule[cursor:cursor+4]
            if not batch:
                epoch, cursor = epoch+1, 0
                continue
            counts = [int(labels(Episode(root, identity, 'S-NativeDP').record, torch.arange(start, stop, 12))[1].sum())
                      for identity, start, stop in batch]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for (identity, start, stop), count in zip(batch, counts):
                episode = Episode(root, identity, group)
                with lease.block():
                    condition, anchors, _ = encode_segment(model, episode, start, stop)
                    command, valid = labels(episode.record, anchors)
                    loss = model.loss(condition, command.cuda(), valid.cuda())*count/sum(counts)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    loss_value += float(loss.detach())
                anchors_seen += len(anchors)
            with lease.block():
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite training gradient')
                scale = min(1., (updates+1)/500)*.5*(1+math.cos(math.pi*max(0, updates-500)/max(1, total-500)))
                for parameter_group in optimizer.param_groups:
                    parameter_group['lr'] = 1e-4*scale
                optimizer.step()
                ema.step(model)
            updates += 1
            cursor += len(batch)
            segments_seen += len(batch)
            if cursor == len(schedule):
                epoch, cursor = epoch+1, 0
            status = dict(state='training', group=group, stage=stage, updates=updates, total_updates=total,
                completed_traversals=epoch, next_segment=cursor, anchors_seen=anchors_seen,
                segments_seen=segments_seen, loss=loss_value, lr=optimizer.param_groups[0]['lr'],
                elapsed_s=time.monotonic()-began, seconds_per_update=(time.monotonic()-began)/max(1, updates-initial_updates))
            atomic_json(output/'status.json', status)
            with (output/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(status)+'\n')
            if updates == 1 or updates % 100 == 0 or updates == total:
                checkpoint()
            validate = (stage.startswith('fit') and (updates % 100 == 0 or updates == total)) or (stage == 'formal' and updates % 1000 == 0)
            if validate:
                score = metrics(averaged, root, identities if stage.startswith('fit') else data['splits']['validation'], lease)
                previous_best = best
                if score['selection_mse'] < best:
                    best = score['selection_mse']
                    checkpoint(True)
                stagnation = 0 if best < previous_best*.995 else stagnation+1
                history.append(dict(updates=updates, traversals=epoch, metrics=score))
                atomic_json(output/'validation.json', history)
                checkpoint()
                if stage.startswith('fit') and fit_passed(score, len(identities)):
                    break
                if stage == 'formal' and updates >= 5000 and epoch >= 20 and stagnation >= 6:
                    break
        if stage == 'pilot':
            score = metrics(averaged, root, data['splits']['validation'], lease)
            checkpoint(True)
        elif stage.startswith('fit'):
            score = history[-1]['metrics']
            if not fit_passed(score, len(identities)):
                raise RuntimeError('Fitting gate thresholds not reached within fixed update limit')
        else:
            score = history[-1]['metrics']
        checkpoint()
        atomic_json(output/'result.json', dict(passed=True, updates=updates, completed_traversals=epoch,
            best=best if math.isfinite(best) else None, metrics=score,
            elapsed_s=time.monotonic()-began, seconds_per_update=status['seconds_per_update'],
            anchors_seen=anchors_seen, segments_seen=segments_seen, best_sha256=sha256(output/'best.pt')))
    except BaseException as error:
        atomic_json(output/'interruption.json', dict(error=repr(error), executed_updates=updates,
            recovery='only the atomic last checkpoint cursor is committed; prefix is reconstructed'))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--stage', choices=('network', 'fit1', 'fit4', 'pilot', 'formal', 'test'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(exist_ok=args.resume, parents=True)
    session = json.loads((args.root/'session.json').read_text())
    gate_recovery = session.get('late_gate_recovery', {}).get('authorized')
    deadline = (session['deadline'] if args.stage == 'test' else
                session['training_deadline'] if args.stage == 'formal' else
                session['deadline'] if gate_recovery else session['gate_deadline'])
    lease = NativeLease(args.root, deadline=deadline)
    if args.stage == 'network':
        network_gate(args.root, args.group, args.output, lease)
    elif args.stage == 'test':
        marker = args.root/(args.group+'_test_started.json')
        if marker.exists():
            raise RuntimeError('Held-out test was already started; never evaluate twice')
        atomic_json(marker, dict(started=time.time()))
        checkpoint_path = Path(json.loads((args.root/'selected.json').read_text())[args.group]['path'])
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        with lease.block():
            model = load_policy(saved, 'cuda:0')
        ids = json.loads((args.root/'data.json').read_text())['splits']['test']
        result = metrics(model, args.root, ids, lease, seed=7093)
        atomic_json(args.output/'result.json', dict(passed=True, metrics=result, checkpoint_sha256=sha256(checkpoint_path)))
    else:
        train(args.root, args.group, args.stage, args.output, lease, args.resume)


if __name__ == '__main__':
    main()
