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
from dataclasses import asdict
from pathlib import Path
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
    tactile = {k: (v.unsqueeze(0) if v.ndim >= 1 else v) for k, v in chunk["tactile"].items()}
    tactile_valid = chunk["tactile_valid"].unsqueeze(0)
    return model.encode(obs, tactile, tactile_valid, state,
                        return_state=True, return_sequence=True)


def _run_epoch(model, dataset, ids, variant, device, chunk_size, optimizer=None,
               throttle_ms=0, log=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    chunks = 0
    steps = 0
    started = time.perf_counter()
    for ep_index, key in enumerate(ids):
        state = None
        for chunk in dataset.chunks(key, chunk_size):
            tick = time.perf_counter()
            data = _move(chunk, device)
            cond, state = _condition(model, variant, data, state)
            b, t, hidden = cond.shape
            target_data = data["action_target"]
            valid_data = data["action_valid"]
            if target_data.ndim == 3:
                target_data = target_data.unsqueeze(0)
                valid_data = valid_data.unsqueeze(0)
            target = target_data.reshape(b * t, target_data.shape[2], 29)
            valid = valid_data.reshape(b * t, valid_data.shape[2])
            loss = model.loss_from_condition(cond.reshape(b * t, hidden), target, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {variant} loss at episode {key} step {chunk['start']}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite {variant} gradient at {key}:{chunk['start']}")
                optimizer.step()
                state = detach_state(state)
            else:
                state = detach_state(state)
            total_loss += float(loss.detach().cpu())
            chunks += 1
            steps += t
            if throttle_ms:
                time.sleep(throttle_ms / 1000.0)
            if log and chunks % 100 == 0:
                elapsed = time.perf_counter() - started
                log(f"chunks={chunks} episodes={ep_index+1}/{len(ids)} steps={steps} "
                    f"loss={total_loss/chunks:.6f} wall_s={elapsed:.1f}")
    return dict(loss=total_loss / max(chunks, 1), chunks=chunks, steps=steps,
                wall_seconds=time.perf_counter() - started)


def _save(path, model, optimizer, cfg, args, manifest_hash, splits, epoch, metrics, mean, std):
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
        metrics=metrics)
    torch.save(payload, path)


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
    p.add_argument("--resume", type=Path)
    args = p.parse_args(argv)
    if args.epochs < 1 or args.chunk_size < 1:
        p.error("epochs and chunk-size must be positive")
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
            start_epoch = int(payload["epoch"]); history = payload.get("metrics", [])
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
            train_result = _run_epoch(model, train, splits[0], args.variant, device, args.chunk_size,
                                      optimizer, args.throttle_ms, lambda s: log(f"epoch={epoch+1} train {s}"))
            with torch.no_grad():
                val_result = _run_epoch(model, validation, splits[1], args.variant, device,
                                        args.chunk_size, None, args.throttle_ms,
                                        lambda s: log(f"epoch={epoch+1} val {s}"))
            record = dict(epoch=epoch + 1, train=train_result, validation=val_result,
                          eta_seconds=(args.epochs - epoch - 1) * train_result["wall_seconds"])
            history.append(record)
            (args.output / "metrics.jsonl").open("a").write(json.dumps(record) + "\n")
            _save(args.output / "last.pt", model, optimizer, cfg, args, manifest_hash, splits,
                  epoch + 1, history, mean, std)
            if val_result["loss"] <= best_val:
                _save(args.output / "best.pt", model, optimizer, cfg, args, manifest_hash, splits,
                      epoch + 1, history, mean, std)
                best_val = val_result["loss"]
            log(f"epoch={epoch+1}/{args.epochs} train_loss={train_result['loss']:.6f} "
                f"val_loss={val_result['loss']:.6f} epoch_s={train_result['wall_seconds']+val_result['wall_seconds']:.1f} "
                f"eta_h={record['eta_seconds']/3600:.2f}")
        model.eval()
        with torch.no_grad():
            test_result = _run_epoch(model, test, splits[2], args.variant, device,
                                     args.chunk_size, None, args.throttle_ms)
        report = dict(variant=args.variant, epochs=args.epochs, test=test_result,
                      split_sizes=list(map(len, splits)), manifest_sha256=manifest_hash,
                      checkpoint=str(args.output / "best.pt"),
                      action_label="transition/command_clipped_pre_delay",
                      tactile=(args.variant == "b1"), training_complete=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        log(json.dumps(report))


if __name__ == "__main__":
    main()
