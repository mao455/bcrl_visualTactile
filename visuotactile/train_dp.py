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
import statistics
import subprocess
import time

import numpy as np
import torch

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


class ResourceGuard:
    """Pause this trainer before a chunk when host/card headroom is low.

    The guard only sleeps this process.  It never signals or changes another
    process, and it does not alter the simulator/control timing (training has
    no simulator clock).  GPU telemetry is whole-card, so an unrelated job's
    memory spike also causes a safe wait.
    """
    def __init__(self, output, *, min_gpu_free_mib=6000.0,
                 min_ram_available_gib=8.0, poll_s=2.0,
                 wait_timeout_s=900.0):
        self.output = Path(output)
        self.min_gpu_free_mib = float(min_gpu_free_mib)
        self.min_ram_available_bytes = int(float(min_ram_available_gib) * (1024 ** 3))
        self.poll_s = max(0.25, float(poll_s))
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.samples = []
        self.wait_seconds = 0.0
        self.started = time.monotonic()
        self._last_wait_log = -1e9
        self._last_sample = -1e9
        self._last_row = None

    @staticmethod
    def _ram_available():
        try:
            fields = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                name, value = line.split(':', 1)
                fields[name] = int(value.strip().split()[0]) * 1024
            return fields.get('MemAvailable')
        except (OSError, ValueError):
            return None

    @staticmethod
    def _gpu():
        try:
            line = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                 '--format=csv,noheader,nounits'], text=True, timeout=3
            ).splitlines()[0]
            util, used, free = (float(x.strip()) for x in line.split(','))
            return dict(util_percent=util, used_mib=used, free_mib=free)
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            return None

    def sample(self):
        now = time.monotonic()
        gpu = self._gpu()
        ram = self._ram_available()
        row = dict(utc_epoch=time.time(), elapsed_s=now-self.started,
                   gpu=gpu, ram_available_bytes=ram,
                   gpu_telemetry_missing=gpu is None,
                   ram_telemetry_missing=ram is None)
        self.samples.append(row)
        self._last_sample = now
        self._last_row = row
        self.output.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        (self.output / 'resources.json').write_text(json.dumps(summary, indent=2))
        with (self.output / 'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        return row

    def _safe(self, row):
        gpu = row['gpu']
        ram = row['ram_available_bytes']
        return (gpu is not None and gpu['free_mib'] >= self.min_gpu_free_mib and
                ram is not None and ram >= self.min_ram_available_bytes)

    def before_chunk(self, log=None):
        # Avoid an nvidia-smi subprocess for every small TBPTT chunk.  A fresh
        # sample is still taken at least once per poll interval and on every
        # wait iteration after a threshold breach.
        if self._last_row is not None and time.monotonic()-self._last_sample < self.poll_s:
            if self._safe(self._last_row):
                return self._last_row
        waited = 0.0
        while True:
            row = self.sample()
            if self._safe(row):
                self.wait_seconds += waited
                return row
            if waited >= self.wait_timeout_s:
                raise RuntimeError(
                    'resource guard timeout: require '
                    f'gpu_free>={self.min_gpu_free_mib:.0f}MiB and '
                    f'ram_available>={self.min_ram_available_bytes/(1024**3):.1f}GiB; '
                    f'last={row}')
            now = time.monotonic()
            if log is not None and now-self._last_wait_log >= 30:
                log('resource_wait ' + json.dumps(row, sort_keys=True))
                self._last_wait_log = now
            start = time.monotonic()
            time.sleep(self.poll_s)
            waited += time.monotonic()-start

    def summary(self):
        gpu_rows = [x['gpu'] for x in self.samples if x['gpu'] is not None]
        ram_rows = [x['ram_available_bytes'] for x in self.samples
                    if x['ram_available_bytes'] is not None]
        util = [x['util_percent'] for x in gpu_rows]
        free = [x['free_mib'] for x in gpu_rows]
        return dict(samples=len(self.samples), missing_gpu=sum(x['gpu'] is None for x in self.samples),
                    missing_ram=sum(x['ram_available_bytes'] is None for x in self.samples),
                    gpu_util_mean_percent=(statistics.fmean(util) if util else None),
                    gpu_util_max_percent=(max(util) if util else None),
                    gpu_memory_used_max_mib=(max(x['used_mib'] for x in gpu_rows) if gpu_rows else None),
                    gpu_free_min_mib=(min(free) if free else None),
                    ram_available_min_bytes=(min(ram_rows) if ram_rows else None),
                    wait_seconds=self.wait_seconds, thresholds=dict(
                        gpu_free_mib=self.min_gpu_free_mib,
                        ram_available_bytes=self.min_ram_available_bytes),
                    updated_utc=time.time())


def _run_epoch(model, dataset, ids, variant, device, chunk_size, optimizer=None,
               throttle_ms=0, log=None, *, batch_size=1, amp="off", scaler=None,
               resource_guard=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    chunks = 0
    steps = 0
    valid_targets = 0
    updates = 0
    skipped = 0
    grad_norm_sum = 0.0
    started = time.perf_counter()
    state = None
    for ep_index, chunk in enumerate(_batch_episode_chunks(dataset, ids, chunk_size, batch_size)):
        if chunk["episode_start"]:
            state = None
        if resource_guard is not None:
            resource_guard.before_chunk(log)
        tick = time.perf_counter()
        data = _move(chunk, device)
        with _amp_context(device, amp):
            cond, state = _condition(model, variant, data, state)
            b, t, hidden = cond.shape
            target_data = data["action_target"]
            valid_data = data["action_valid"].bool()
            current_valid = data["obs"]["valid"].bool()
            valid_data = valid_data & current_valid[..., None]
            target = target_data.reshape(b * t, target_data.shape[2], 29)
            valid = valid_data.reshape(b * t, valid_data.shape[2])
            loss = model.loss_from_condition(cond.reshape(b * t, hidden), target, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {variant} loss at chunk {ep_index} step {chunk['start']}")
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite {variant} gradient at chunk {ep_index}:{chunk['start']}")
            old_scale = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < old_scale:
                    skipped += 1
            else:
                optimizer.step()
            updates += 1
            grad_norm_sum += float(grad_norm.detach().cpu())
            state = detach_state(state)
        else:
            state = detach_state(state)
        total_loss += float(loss.detach().cpu())
        chunks += 1
        steps += int(current_valid.sum().item())
        valid_targets += int(valid.sum().item())
        if throttle_ms:
            time.sleep(throttle_ms / 1000.0)
        if log and chunks % 100 == 0:
            elapsed = time.perf_counter() - started
            log(f"chunks={chunks} groups={ep_index+1}/{(len(ids)+batch_size-1)//batch_size} steps={steps} "
                f"loss={total_loss/chunks:.6f} wall_s={elapsed:.1f}")
    return dict(loss=total_loss / max(chunks, 1), chunks=chunks, steps=steps,
                valid_targets=valid_targets, updates=updates, skipped_updates=skipped,
                mean_grad_norm=grad_norm_sum/max(updates,1),
                wall_seconds=time.perf_counter() - started)


def _save(path, model, optimizer, cfg, args, manifest_hash, splits, epoch, metrics, mean, std, scaler=None):
    payload = dict(
        checkpoint_version=1, variant=args.variant, epoch=epoch,
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
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="cuda")
    p.add_argument("--throttle-ms", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, choices=(1,2,4), default=1)
    p.add_argument("--amp", choices=("off","bf16","fp16"), default="bf16")
    p.add_argument("--min-gpu-free-mib", type=float, default=6000.0)
    p.add_argument("--min-ram-available-gib", type=float, default=8.0)
    p.add_argument("--resource-poll-s", type=float, default=2.0)
    p.add_argument("--resource-wait-timeout-s", type=float, default=900.0)
    p.add_argument("--resume", type=Path)
    args = p.parse_args(argv)
    if args.epochs < 1 or args.chunk_size < 1:
        p.error("epochs and chunk-size must be positive")
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    torch.set_num_threads(min(4, max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))))
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
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
                           memory_version=2, horizon=12, replan_steps=6)
        model = _make_model(args.variant, cfg).to(device)
        if args.variant == "b0":
            model.encoder.proprio_mean.copy_(mean.to(device)); model.encoder.proprio_std.copy_(std.to(device))
        else:
            model.visual.proprio_mean.copy_(mean.to(device)); model.visual.proprio_std.copy_(std.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        if args.amp == "fp16" and device.type != "cuda":
            p.error("fp16 AMP requires CUDA")
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16" and device.type == "cuda")
        resource_guard = None
        if device.type == "cuda":
            resource_guard = ResourceGuard(
                args.output, min_gpu_free_mib=args.min_gpu_free_mib,
                min_ram_available_gib=args.min_ram_available_gib,
                poll_s=args.resource_poll_s, wait_timeout_s=args.resource_wait_timeout_s)
            resource_guard.before_chunk(log)
        start_epoch = 0
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
            model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
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
        log(f"device={device} params={sum(p.numel() for p in model.parameters())} train_rows={train_rows}")
        for epoch in range(start_epoch, args.epochs):
            before = {name: value.detach().cpu().clone() for name,value in model.state_dict().items()
                      if torch.is_floating_point(value)}
            train_result = _run_epoch(model, train, splits[0], args.variant, device, args.chunk_size,
                                      optimizer, args.throttle_ms, lambda s: log(f"epoch={epoch+1} train {s}"),
                                      batch_size=args.batch_size, amp=args.amp, scaler=scaler,
                                      resource_guard=resource_guard)
            with torch.no_grad():
                val_result = _run_epoch(model, validation, splits[1], args.variant, device,
                                        args.chunk_size, None, args.throttle_ms,
                                        lambda s: log(f"epoch={epoch+1} val {s}"),
                                        batch_size=args.batch_size, amp=args.amp,
                                        resource_guard=resource_guard)
            train_result["parameter_delta_l1"] = sum(float((value.detach().cpu()-before[name]).abs().sum())
                                                       for name,value in model.state_dict().items()
                                                       if name in before)
            record = dict(epoch=epoch + 1, train=train_result, validation=val_result,
                          eta_seconds=(args.epochs - epoch - 1) * train_result["wall_seconds"])
            history.append(record)
            (args.output / "metrics.jsonl").open("a").write(json.dumps(record) + "\n")
            _save(args.output / "last.pt", model, optimizer, cfg, args, manifest_hash, splits,
                  epoch + 1, history, mean, std, scaler)
            if val_result["loss"] <= best_val:
                _save(args.output / "best.pt", model, optimizer, cfg, args, manifest_hash, splits,
                      epoch + 1, history, mean, std, scaler)
                best_val = val_result["loss"]
            log(f"epoch={epoch+1}/{args.epochs} train_loss={train_result['loss']:.6f} "
                f"val_loss={val_result['loss']:.6f} epoch_s={train_result['wall_seconds']+val_result['wall_seconds']:.1f} "
                f"eta_h={record['eta_seconds']/3600:.2f} "
                f"resource_wait_s={resource_guard.wait_seconds if resource_guard else 0.0:.1f}")
        model.eval()
        with torch.no_grad():
            test_result = _run_epoch(model, test, splits[2], args.variant, device,
                                     args.chunk_size, None, args.throttle_ms,
                                     batch_size=args.batch_size, amp=args.amp,
                                     resource_guard=resource_guard)
        report = dict(variant=args.variant, epochs=args.epochs, test=test_result,
                      split_sizes=list(map(len, splits)), manifest_sha256=manifest_hash,
                      checkpoint=str(args.output / "best.pt"),
                      action_label="transition/command_clipped_pre_delay",
                      tactile=(args.variant == "b1"), training_complete=True,
                      resources=(resource_guard.summary() if resource_guard else None))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        log(json.dumps(report))


if __name__ == "__main__":
    main()
