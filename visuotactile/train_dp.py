"""Train the command imitation baselines B0 and B1 on the production manifest.

The loader is intentionally episode ordered.  A chunk is only a TBPTT boundary;
state is carried to the next chunk and is cleared only at an episode boundary.
This entry point never starts Isaac Sim and never exposes transition-after data
to either policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import subprocess
import signal
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch

from .training_resources import ResourceGuard, atomic_json
from .compact import EpisodeFile
from .dataset import EpisodeSequenceDataset
from .history import detach_state
from .manifest import ManifestReader, ManifestSequenceDataset, assert_group_disjoint
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .models.tactile_dp import TactileCommandDiffusionPolicy


def _move(value, device):
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(reader: ManifestReader, seed: int):
    ids = list(reader.entries)
    groups = {}
    for key in ids:
        groups.setdefault(reader.entries[key]["pair_id"], []).append(key)
    rng = np.random.default_rng(seed)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    ordered = [key for group in group_names for key in groups[group]]
    n = len(ordered)
    n_val = max(1, round(n * .10))
    n_test = max(1, round(n * .10))
    train = ordered[:n - n_val - n_test]
    validation = ordered[n - n_val - n_test:n - n_test]
    test = ordered[n - n_test:]
    assert_group_disjoint(reader, train, validation)
    assert_group_disjoint(reader, train, test)
    assert_group_disjoint(reader, validation, test)
    return train, validation, test


def _assert_no_own_plan(manifest_ds: ManifestSequenceDataset):
    """Verify the B0/B1 no-own-plan contract against stored arrays."""
    for key in manifest_ds.episodes:
        ds = manifest_ds.datasets[key]
        row = manifest_ds.reader.entries[key]
        path = manifest_ds.reader.path.parent / row["directory"] / "data.h5"
        with EpisodeFile(path) as f:
            g = f["episodes"][row["episode"]]
            length = int(g.attrs["length"])
            if np.asarray(g["obs/previous_nominal_valid"][:length]).any():
                raise ValueError(f"{key}: previous_nominal_valid is present in no-plan baseline")
            if np.asarray(g["obs/prior_plan_valid"][:length]).any():
                raise ValueError(f"{key}: prior_plan_valid is present in no-plan baseline")
            if np.asarray(g["obs/previous_nominal_action"][:length]).any():
                raise ValueError(f"{key}: previous_nominal_action is nonzero in no-plan baseline")
            if np.asarray(g["obs/prior_plan"][:length]).any():
                raise ValueError(f"{key}: prior_plan is nonzero in no-plan baseline")


def _fit_proprio_stats(manifest_ds: ManifestSequenceDataset):
    total = 0
    s = None
    ss = None
    for ds in manifest_ds.datasets.values():
        with EpisodeFile(ds.path) as f:
            for episode in ds.episodes:
                x = np.asarray(f["episodes"][episode]["obs/proprio"][:], dtype=np.float64)
                total += len(x)
                s = x.sum(0) if s is None else s + x.sum(0)
                ss = (x * x).sum(0) if ss is None else ss + (x * x).sum(0)
    if total == 0:
        raise ValueError("No training proprioception rows")
    mean = torch.from_numpy((s / total).astype(np.float32))
    var = torch.from_numpy(np.maximum(ss / total - mean.numpy() ** 2, 1e-6).astype(np.float32))
    return mean, var.sqrt().clamp_min(1e-3), total


def _make_model(variant, cfg):
    if variant == "b0":
        return NominalDiffusionPolicy(cfg)
    if variant == "b1":
        return TactileCommandDiffusionPolicy(cfg)
    raise ValueError(variant)


def _condition(model, variant, chunk, state):
    obs = chunk["obs"]
    if variant == "b0":
        return model.encoder(obs, state, return_state=True, return_sequence=True)
    tactile = chunk["tactile"]
    tactile_valid = chunk["tactile_valid"]
    return model.encode(obs, tactile, tactile_valid, state,
                        return_state=True, return_sequence=True)


def _pad_time(value, length):
    """Pad only the time axis with zeros; masks identify the padded rows."""
    if value.shape[0] == length:
        return value
    if value.shape[0] > length:
        return value[:length]
    return torch.cat((value, torch.zeros((length-value.shape[0], *value.shape[1:]),
                                         dtype=value.dtype)), 0)


def _batch_episode_chunks(dataset, ids, chunk_size, batch_size):
    """Pack independent episode streams without concatenating their histories.

    A group owns fixed slots until every slot reaches its end.  Short tails are
    zero padded and masked; a slot is never replaced by another episode inside
    the group, so its carried visual/tactile/fusion state cannot cross an
    episode boundary.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for group_start in range(0, len(ids), batch_size):
        group = ids[group_start:group_start+batch_size]
        iterators = [iter(dataset.chunks(key, chunk_size)) for key in group]
        ordinal = 0
        length = max(len(dataset.datasets[key]) for key in group)
        while True:
            rows = []
            active = False
            for iterator in iterators:
                try:
                    rows.append(next(iterator)); active = True
                except StopIteration:
                    rows.append(None)
            if not active:
                break
            reference = next(row for row in rows if row is not None)
            obs = {}
            for name in reference["obs"]:
                obs[name] = torch.stack([_pad_time(
                    row["obs"][name].squeeze(0) if row is not None else torch.zeros_like(reference["obs"][name].squeeze(0)),
                    chunk_size) for row in rows], 0)
            target = torch.stack([_pad_time(row["action_target"] if row is not None else
                                            torch.zeros_like(reference["action_target"]), chunk_size)
                                  for row in rows], 0)
            action_valid = torch.stack([_pad_time(row["action_valid"] if row is not None else
                                                   torch.zeros_like(reference["action_valid"], dtype=torch.bool), chunk_size)
                                        for row in rows], 0).bool()
            result = dict(episode=group, start=ordinal*chunk_size,
                          stop=(ordinal+1)*chunk_size, episode_start=ordinal == 0,
                          group_index=group_start//batch_size,
                          group_end=(ordinal+1)*chunk_size >= length,
                          obs=obs, action_target=target, action_valid=action_valid)
            if "tactile" in reference:
                tactile = {}
                for name in reference["tactile"]:
                    tactile[name] = torch.stack([_pad_time(
                        row["tactile"][name] if row is not None else torch.zeros_like(reference["tactile"][name]),
                        chunk_size) for row in rows], 0)
                tactile_valid = torch.stack([_pad_time(
                    row["tactile_valid"] if row is not None else torch.zeros_like(reference["tactile_valid"], dtype=torch.bool),
                    chunk_size) for row in rows], 0).bool()
                result.update(tactile=tactile, tactile_valid=tactile_valid)
            yield result
            ordinal += 1


def _amp_context(device, mode):
    if mode == "off" or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def _prefetch(stream, enabled):
    """One CPU producer thread, one pending chunk, no CUDA or shared HDF5 handles."""
    if not enabled:
        yield from stream
        return
    sentinel = object()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hdf5-reader") as pool:
        future = pool.submit(next, stream, sentinel)
        while True:
            value = future.result()
            if value is sentinel:
                break
            future = pool.submit(next, stream, sentinel)
            yield value


def _run_epoch(model, dataset, ids, variant, device, chunk_size, optimizer=None,
               throttle_ms=0, log=None, *, batch_size=1, amp="off", scaler=None,
               resource_guard=None, prefetch=0, start_group=0, prior=None,
               on_group=None, on_progress=None):
    training = optimizer is not None
    model.train(training)
    totals = dict(loss_sum=0., chunks=0, steps=0, valid_targets=0, updates=0,
                  skipped_updates=0, grad_norm_sum=0., padded_steps=0,
                  io_wait_s=0., compute_s=0., wall_seconds=0.)
    if prior:
        totals.update({k:v for k,v in prior.items() if k in totals})
    started = time.perf_counter()
    previous_wall = totals["wall_seconds"]
    state = None
    stream = _prefetch(_batch_episode_chunks(dataset, ids[start_group*batch_size:],
                                             chunk_size, batch_size), prefetch)
    tick = time.perf_counter()
    for chunk in stream:
        totals["io_wait_s"] += time.perf_counter()-tick
        if chunk["episode_start"]:
            state = None
        if resource_guard:
            resource_guard.before_chunk(log)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_start = time.perf_counter()
        data = _move(chunk, device)
        with _amp_context(device, amp):
            cond, state = _condition(model, variant, data, state)
            b, t, hidden = cond.shape
            target_data = data["action_target"]
            current_valid = data["obs"]["valid"].bool()
            valid = (data["action_valid"].bool() & current_valid[..., None]).flatten(0,1)
            target = target_data.flatten(0,1)
            loss = model.loss_from_condition(cond.reshape(b*t, hidden), target, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {variant} loss at {chunk['episode']}:{chunk['start']}")
        skipped = False
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None and scaler.is_enabled():
                old = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                skipped = scaler.get_scale() < old
            else:
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
            totals["updates"] += int(not skipped)
            totals["skipped_updates"] += int(skipped)
            if torch.isfinite(norm):
                totals["grad_norm_sum"] += float(norm)
        state = detach_state(state)
        count = int(valid.sum())
        totals["loss_sum"] += float(loss.detach())*count
        totals["chunks"] += 1
        totals["steps"] += int(current_valid.sum())
        totals["padded_steps"] += b*t-int(current_valid.sum())
        totals["valid_targets"] += count
        # Do not keep old image/condition graphs alive during the next forward.
        del data, cond, loss, target_data, target, valid, current_valid
        if device.type == "cuda":
            torch.cuda.synchronize()
        totals["compute_s"] += time.perf_counter()-compute_start
        if throttle_ms:
            time.sleep(throttle_ms/1000.)
        totals["wall_seconds"] = previous_wall+time.perf_counter()-started
        group = start_group+chunk["group_index"]
        if on_progress and totals["chunks"] % 25 == 0:
            on_progress(totals, group)
        if log and totals["chunks"] % 100 == 0:
            log(f"chunks={totals['chunks']} groups={group+1}/{(len(ids)+batch_size-1)//batch_size} "
                f"steps={totals['steps']} loss={totals['loss_sum']/max(totals['valid_targets'],1):.6f} "
                f"wall_s={totals['wall_seconds']:.1f}")
        if chunk["group_end"] and on_group:
            on_group(group+1, dict(totals))
        tick = time.perf_counter()
    totals.update(loss=totals['loss_sum']/max(totals['valid_targets'],1),
                  mean_grad_norm=totals['grad_norm_sum']/max(totals['updates'],1))
    return totals


def _save(path, model, optimizer, cfg, args, manifest_hash, splits, epoch, metrics, mean, std, scaler=None, *, metadata=None,
          progress=None, optimizer_updates=None):
    payload = dict(
        checkpoint_version=2, command_checkpoint_version=2, variant=args.variant, epoch=epoch,
        metadata=metadata, progress=progress,
        training_source_sha256=getattr(args, 'training_source_sha256', {}),
        optimizer_updates=(optimizer_updates if optimizer_updates is not None else
                           sum(x['train']['updates'] for x in metrics)),
        training_options=dict(batch_size=args.batch_size, chunk_size=args.chunk_size, amp=args.amp),
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), config=asdict(cfg),
        manifest_sha256=manifest_hash, train_episodes=splits[0],
        validation_episodes=splits[1], test_episodes=splits[2], seed=args.seed,
        history_contract=dict(memory_version=2, chunk_size=args.chunk_size,
                             horizon=cfg.horizon, replan_steps=cfg.replan_steps,
                             tactile_physical_hz=120, command_hz=60,
                             own_plan_mode="none", action_label="command_clipped_pre_delay",
                             camera_order=["fixed_high", "fixed_side", "wrist_rgbd"]),
        perception_version=cfg.perception_version,
        proprio_mean=mean, proprio_std=std,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        amp_mode=args.amp, amp_state=scaler.state_dict() if scaler is not None else None,
        metrics=metrics)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--variant", choices=("b0", "b1"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="cuda")
    p.add_argument("--throttle-ms", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, choices=(1,2,4), default=1)
    p.add_argument("--amp", choices=("off","bf16","fp16"), default="bf16")
    p.add_argument("--vision-checkpoint-images", type=int, default=12)
    p.add_argument("--prefetch", type=int, choices=(0,1), default=1)
    p.add_argument("--cuda-memory-fraction", type=float, default=.36)
    p.add_argument("--save-every-groups", type=int, default=8)
    p.add_argument("--min-gpu-free-mib", type=float, default=4096.0)
    p.add_argument("--min-ram-available-gib", type=float, default=10.0)
    p.add_argument("--resource-poll-s", type=float, default=2.0)
    p.add_argument("--resource-wait-timeout-s", type=float, default=3600.0)
    p.add_argument("--resume", type=Path)
    args = p.parse_args(argv)
    args.training_source_sha256 = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__),Path(__file__).with_name('rollout.py'),Path(__file__).with_name('dataset.py'),
        Path(__file__).with_name('compact.py'),*Path(__file__).with_name('models').glob('*.py'))}
    if args.epochs < 1 or args.chunk_size < 1:
        p.error("epochs and chunk-size must be positive")
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    torch.set_num_threads(min(4, max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))))
    torch.manual_seed(args.seed)
    # Float32 fixed-weight replay equivalence should not depend on TF32's
    # batch-shape-dependent convolution rounding. BF16 autocast is explicit.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device('cuda:0' if args.device == 'cuda' else args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'last.pt').exists() and not args.resume:
        raise ValueError("Output already has a checkpoint; use --resume or a fresh output")
    stop = [False]
    def handle_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    stop_requested = lambda: stop[0] or (args.output/'STOP').exists()
    log_path = args.output / "train.log"
    with log_path.open("a", buffering=1) as log_file:
        def log(message):
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            line = f"{stamp} {message}"
            print(line, flush=True)
            log_file.write(line + "\n")

        reader = ManifestReader(args.manifest)
        if reader.manifest.get("target_successes") != 400 or len(reader.entries) != 400:
            raise ValueError("B0/B1 formal training requires the complete 400 episode production manifest")
        splits = _split(reader, args.seed)
        log(f"manifest={args.manifest} episodes={len(reader.entries)} split={list(map(len,splits))}")
        train = ManifestSequenceDataset(args.manifest, ids=splits[0], stage="production",
                                        include_tactile=args.variant == "b1", require_command=True)
        validation = ManifestSequenceDataset(args.manifest, ids=splits[1], stage="production",
                                             include_tactile=args.variant == "b1", require_command=True)
        test = ManifestSequenceDataset(args.manifest, ids=splits[2], stage="production",
                                       include_tactile=args.variant == "b1", require_command=True)
        _assert_no_own_plan(train)
        _assert_no_own_plan(validation)
        _assert_no_own_plan(test)
        mean, std, train_rows = _fit_proprio_stats(train)
        cfg = PolicyConfig(views=3, channels=4, perception_version=2,
                           memory_version=2, horizon=12, replan_steps=6,
                           own_plan_mode="none", vision_checkpoint_images=args.vision_checkpoint_images)
        model = _make_model(args.variant, cfg).to(device)
        if args.variant == "b0":
            model.encoder.proprio_mean.copy_(mean.to(device)); model.encoder.proprio_std.copy_(std.to(device))
        else:
            model.visual.proprio_mean.copy_(mean.to(device)); model.visual.proprio_std.copy_(std.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        if args.amp == "fp16" and device.type != "cuda":
            p.error("fp16 AMP requires CUDA")
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.type == "cuda")
        resource_guard = None
        if device.type == "cuda":
            resource_guard = ResourceGuard(
                args.output, min_gpu_free_mib=args.min_gpu_free_mib,
                min_ram_available_gib=args.min_ram_available_gib,
                poll_s=args.resource_poll_s, wait_timeout_s=args.resource_wait_timeout_s,
                stop_requested=stop_requested)
            resource_guard.before_chunk(log)
        metadata = train.datasets[splits[0][0]].metadata
        start_epoch = 0
        resume_progress = None
        history = []
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            if payload.get("variant") != args.variant or payload.get("manifest_sha256") != _manifest_hash(args.manifest):
                raise ValueError("Resume checkpoint does not match variant or manifest")
            if int(payload.get("seed", args.seed)) != args.seed:
                raise ValueError("Resume checkpoint split seed differs")
            if (payload.get("train_episodes") != splits[0] or
                    payload.get("validation_episodes") != splits[1] or
                    payload.get("test_episodes") != splits[2]):
                raise ValueError("Resume checkpoint episode split differs")
            from .contracts import assert_policy_compatible
            assert_policy_compatible(payload['metadata'], metadata)
            if (payload.get('command_checkpoint_version') != 2 or payload['config'] != asdict(cfg)
                    or payload['training_options'] != dict(batch_size=args.batch_size,
                                                          chunk_size=args.chunk_size, amp=args.amp)):
                raise ValueError("Resume model/training protocol differs")
            model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
            resume_progress = payload.get('progress')
            if payload.get("amp_mode", args.amp) != args.amp:
                raise ValueError("Resume checkpoint AMP mode differs")
            if payload.get("amp_state") is not None:
                scaler.load_state_dict(payload["amp_state"])
            start_epoch = int(payload["epoch"]); history = payload.get("metrics", [])
            torch.set_rng_state(payload["torch_rng"])
            if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
            log(f"resuming epoch={start_epoch}")
        best_val = min((x["validation"]["loss"] for x in history), default=float("inf"))
        manifest_hash = _manifest_hash(args.manifest)
        (args.output / "config.json").write_text(json.dumps(dict(variant=args.variant, config=asdict(cfg),
            manifest_sha256=manifest_hash, split_sizes=list(map(len,splits)), train_rows=train_rows,
            seed=args.seed, history_contract={"memory_version":2,"chunk_size":args.chunk_size,
                "horizon":12,"replan_steps":6,"own_plan_mode":"none",
                "action_label":"command_clipped_pre_delay"}), indent=2))
        atomic_json(args.output/'launch.json', dict(pid=os.getpid(), started_utc=time.time(),
                    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        log(f"device={device} params={sum(p.numel() for p in model.parameters())} train_rows={train_rows}")
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.perf_counter()
            previous_updates = sum(x['train']['updates'] for x in history)
            def publish_progress(totals, group):
                rate = totals['steps']/max(totals['wall_seconds'],1e-9)
                atomic_json(args.output/'status.json', dict(state='training', pid=os.getpid(),
                    updated_utc=time.time(), variant=args.variant, epoch=epoch+1,
                    completed_epochs=epoch, group=group, train=totals,
                    steps_per_second=rate,
                    eta_seconds=((args.epochs-epoch)*train_rows-totals['steps'])/max(rate,1e-9)*1.15,
                    eta_basis='measured training rate + estimated 15% validation/save/test; closed-loop evaluation separate'))
            def group_checkpoint(next_group, totals):
                if next_group % args.save_every_groups == 0 or stop_requested():
                    _save(args.output/'last.pt', model, optimizer, cfg, args, manifest_hash, splits,
                          epoch, history, mean, std, scaler, metadata=metadata,
                          progress=dict(next_group=next_group, train=totals),
                          optimizer_updates=previous_updates+totals['updates'])
                if stop_requested():
                    atomic_json(args.output/'status.json', dict(state='stopped',pid=os.getpid(),
                        epoch=epoch, next_group=next_group, checkpoint=str(args.output/'last.pt')))
                    raise SystemExit(0)
            train_result = _run_epoch(model, train, splits[0], args.variant, device, args.chunk_size,
                optimizer, args.throttle_ms, lambda s: log(f"epoch={epoch+1} train {s}"),
                batch_size=args.batch_size, amp=args.amp, scaler=scaler, resource_guard=resource_guard,
                prefetch=args.prefetch, start_group=(resume_progress or {}).get('next_group',0),
                prior=(resume_progress or {}).get('train'), on_group=group_checkpoint,
                on_progress=publish_progress)
            resume_progress = None
            # Save completed training before validation, so interruption there
            # resumes validation without repeating optimizer updates.
            _save(args.output/'last.pt', model, optimizer, cfg, args, manifest_hash, splits,
                epoch, history, mean, std, scaler, metadata=metadata,
                progress=dict(next_group=(len(splits[0])+args.batch_size-1)//args.batch_size, train=train_result),
                optimizer_updates=previous_updates+train_result['updates'])
            with torch.no_grad():
                val_result = _run_epoch(model, validation, splits[1], args.variant, device, args.chunk_size,
                    batch_size=args.batch_size, amp=args.amp, resource_guard=resource_guard, prefetch=args.prefetch)
            record = dict(epoch=epoch+1, train=train_result, validation=val_result,
                          eta_seconds=(args.epochs-epoch-1)*(train_result['wall_seconds']+val_result['wall_seconds']))
            history.append(record)
            save_start = time.perf_counter()
            if val_result['loss'] <= best_val:
                _save(args.output/'best.pt',model,optimizer,cfg,args,manifest_hash,splits,
                      epoch+1,history,mean,std,scaler,metadata=metadata)
                best_val=val_result['loss']
            _save(args.output/'last.pt',model,optimizer,cfg,args,manifest_hash,splits,
                  epoch+1,history,mean,std,scaler,metadata=metadata)
            record['save_seconds'] = time.perf_counter()-save_start
            with (args.output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            atomic_json(args.output/'status.json', dict(state='epoch_complete',pid=os.getpid(),
                updated_utc=time.time(),completed_epochs=epoch+1,latest=record))
            log(f"epoch={epoch+1}/{args.epochs} train_loss={train_result['loss']:.6f} "
                f"val_loss={val_result['loss']:.6f} epoch_s={time.perf_counter()-epoch_start:.1f} "
                f"eta_h={record['eta_seconds']/3600:.2f}")
            if stop_requested():
                return
        best = torch.load(args.output/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(best['model'])
        model.eval()
        with torch.no_grad():
            test_result = _run_epoch(model, test, splits[2], args.variant, device,
                                     args.chunk_size, None, args.throttle_ms,
                                     batch_size=args.batch_size, amp=args.amp,
                                     resource_guard=resource_guard)
        from .command_metrics import sampled_command_metrics
        sampled = sampled_command_metrics(model, test, splits[2], args.variant, device,
                                          args.chunk_size, args.batch_size, resource_guard=resource_guard)
        report = dict(variant=args.variant, epochs=args.epochs, test=test_result, sampled_commands=sampled,
                      split_sizes=list(map(len, splits)), manifest_sha256=manifest_hash,
                      checkpoint=str(args.output / "best.pt"), selected_epoch=best['epoch'],
                      action_label="transition/command_clipped_pre_delay",
                      tactile=(args.variant == "b1"), training_complete=True,
                      resources=(resource_guard.summary() if resource_guard else None))
        atomic_json(args.output / "report.json", report)
        atomic_json(args.output/'status.json', dict(state='complete', pid=os.getpid(),
                    updated_utc=time.time(), report=report))
        log(json.dumps(report))


if __name__ == "__main__":
    main()
