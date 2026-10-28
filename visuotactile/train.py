"""CPU smoke or research training on recorded actions; never synthetic labels."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.utils.data import default_collate
from .dataset import EpisodeDataset, EpisodeSequenceDataset, encode_episode_chunks
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .compact import EpisodeFile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--smoke-only", action="store_true", help="Explicitly allow unsuccessful/incomplete real trajectories")
    p.add_argument("--resume", type=Path)
    p.add_argument("--memory-version",type=int,choices=(1,2),default=2)
    p.add_argument("--chunk-size",type=int,default=8,help="TBPTT chunk size, never an episode reset")
    args = p.parse_args()
    if args.memory_version == 2 and (args.batch_size != 1 or args.chunk_size < 1):
        p.error("Episode recurrence currently supports one ordered episode per batch and positive chunk size")
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps/batch-size must be positive")
    if args.data.suffix == ".json":
        p.error("ManifestSequenceDataset is available for ordered replay; this legacy trainer has no grouped manifest split or memory2 validation yet. Do not merge paired variants into random train/validation rows.")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    with EpisodeFile(args.data) as f:
        metadata = json.loads(f.attrs["metadata_json"])
        if metadata.get("nominal_plan_source") == "untrained_perception_probe" and not args.smoke_only:
            raise ValueError("Untrained perception probe plans are engineering fixtures, not formal training data")
        names = list(f["episodes"])
        shape = f["episodes"][names[0]]["obs/image"].shape
        perception_version = int(f.attrs["schema_version"])
    if perception_version == 2 and not args.smoke_only and (
        metadata.get("nominal_plan_source") != "trained_checkpoint" or not metadata.get("nominal_model_sha256")
    ):
        p.error("Formal own-plan-conditioned BC requires identified causal model replay. Pure expert actions are labels, not own predictions. An explicit no-plan baseline needs a matching deployment contract first.")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = PolicyConfig(views=shape[1], channels=shape[2], perception_version=perception_version,memory_version=args.memory_version)
    eligible = EpisodeDataset(args.data, allow_incomplete=args.smoke_only, allow_unsuccessful=args.smoke_only)
    names = eligible.episodes
    # Whole episode split. One episode cannot honestly supply validation.
    val_names = names[-1:] if len(names)>1 else []
    train_names = names[:-1] if val_names else names
    model = NominalDiffusionPolicy(cfg)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
    first_step = 0
    if args.resume:
        state = torch.load(args.resume,map_location="cpu",weights_only=False)
        if (state["data_sha256"] != data_hash or asdict(PolicyConfig.from_dict(state["config"])) != asdict(cfg)
                or state["smoke_only"] != args.smoke_only):
            raise ValueError("Resume requires identical dataset and model configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_step = state["step"]
        train_names, val_names = state["train_episodes"], state["validation_episodes"]
        torch.set_rng_state(state["torch_rng"])
    options = dict(history=cfg.history,stride=cfg.stride,horizon=cfg.horizon,
                   perception_version=cfg.perception_version,
                   allow_incomplete=args.smoke_only,allow_unsuccessful=args.smoke_only)
    dataset_class = EpisodeSequenceDataset if cfg.memory_version == 2 else EpisodeDataset
    train = dataset_class(args.data,episodes=train_names,**options)
    validation = EpisodeDataset(args.data,episodes=val_names,**options) if val_names else None
    if not args.resume:
        mean,std = train.proprio_statistics()
        model.encoder.proprio_mean.copy_(mean)
        model.encoder.proprio_std.copy_(std)
    losses = []
    started = time.perf_counter()
    model.train()
    def ordered_batches():
        while True:
            # Whole episodes may be reordered; every chosen episode starts at 0.
            for name in train.episodes:
                yield from encode_episode_chunks(model.encoder,train.chunks(name,args.chunk_size),detach_between_chunks=True)
    ordered = ordered_batches() if cfg.memory_version == 2 else None
    for step in range(first_step,first_step+args.steps):
        if ordered is not None:
            conditions,chunk,_ = next(ordered)
            loss = model.loss_from_condition(conditions[0],chunk["action_target"],chunk["action_valid"])
        else:
            ids = torch.randint(len(train),(args.batch_size,)).tolist()
            batch = default_collate([train[i] for i in ids])
            loss = model.loss(batch["obs"],batch["action_target"],batch["action_valid"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite denoising objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 20 == 0:
            print(f"TRAIN step={step+1} denoising_loss={losses[-1]:.6f}",flush=True)
    state = dict(checkpoint_version=cfg.perception_version,model=model.state_dict(),optimizer=optimizer.state_dict(),config=asdict(cfg),
                 step=first_step+args.steps,torch_rng=torch.get_rng_state(),data_sha256=data_hash,
                 train_episodes=train_names,validation_episodes=val_names,metadata=metadata,
                 memory_training=("ordered full-episode chunks, explicit TBPTT; resume restarts at episode boundary"
                                  if cfg.memory_version == 2 else "legacy random windows"),
                 smoke_only=args.smoke_only,nominalization="none; original actions may contain preload")
    torch.save(state,args.output/"checkpoint.pt")
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        validation_loss = None
        if validation and cfg.memory_version == 1:
            batch = default_collate([validation[i] for i in torch.linspace(0,len(validation)-1,steps=min(16,len(validation))).long().tolist()])
            validation_loss = float(model.loss(batch["obs"],batch["action_target"],batch["action_valid"]))
        check_batch = default_collate([train[0]])["obs"]
        predicted = model.predict(check_batch,generator=torch.Generator().manual_seed(9))
        restored = NominalDiffusionPolicy(cfg)
        restored.load_state_dict(torch.load(args.output/"checkpoint.pt",map_location="cpu",weights_only=False)["model"])
        restored.eval()
        reloaded = restored.predict(check_batch,generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(predicted,reloaded,rtol=0,atol=0)
    report = dict(steps=args.steps,total_steps=state["step"],train_frames=len(train),
                  train_episodes=train_names,validation_episodes=val_names,smoke_only=args.smoke_only,
                  first_loss=losses[0],last_loss=losses[-1],validation_denoising_loss=validation_loss,
                  checkpoint_roundtrip_exact=True,wall_seconds=time.perf_counter()-started,
                  loss_kind="masked epsilon prediction on real recorded native actions; no success claim")
    (args.output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
