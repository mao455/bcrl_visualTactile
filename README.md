> Tight-Insertion 当前研究：在隔离树实现最多3轮、总18小时的共享纠偏池DAgger，QG01优先、G11随后；旧两组21条成功纠偏已纳入共享历史池。每轮继续更新现有U-Net前端和Transformer后端，真实update0参与选模并保留完整配对回退。当前新GPU启动以owner实际凭据为准。[研究入口](research_docs/README.md) · [合同](research_docs/CONTRACTS.md) · [结果](research_docs/STATE_DP_RESULTS.md) · [交接](research_docs/SESSION_HANDOFF.md)。

# Play2Perfect

### What Matters in Dexterous Play Pretraining for Precise Assembly?

Official code release for **Play2Perfect**, a 2-stage reinforcement-learning pipeline for
contact-rich robotic assembly.

📄 **Paper:** https://arxiv.org/abs/2606.26428

https://github.com/user-attachments/assets/6bfafdbc-41f4-472a-aeaa-444f7e50076b

## Overview

Play2Perfect learns precise, contact-rich assembly in two stages:

1. **Play** — pretrain a policy by playing with random objects in free space.
2. **Finetune** — finetune the play policy on a contact-rich assembly task.

This repository provides:

- **Training** code for both stages in Isaac Sim (Isaac Lab).
- **Interactive (viser) evaluation** for four assembly tasks using our released checkpoints.
- A **minimal sim-to-real deployment** reference.

> This release targets **Isaac Sim only**.

## Project structure

```
play2perfect/
├── isaacsimenvs/          # Training: Stage-1 play env + Stage-2 precise-assembly env
│   ├── train.py           # Single training entry point (both stages)
│   ├── cfg/               # Hydra task + train configs
│   └── tasks/{play,precise_assembly}/
├── evaluation/         # The 4 assembly tasks: problem registry + viser evaluation
│   ├── problems/          # peg-in-hole, fabrica beam parts 0 & 2, furniture-bench
│   └── eval_isaacsim.py   # Interactive viser evaluation
├── rl_games/              # Vendored RL library (PPO + SAPG)
├── deployment/            # Minimal sim-to-real reference
├── assets/urdf/           # Robot + task assets
└── docs/                  # Installation + deployment guides
```

## The four assembly tasks

| Task | `problem` key |
|------|----------------|
| Tight insertion (L-peg, 0.5 mm tolerance) | `tight_insertion` |
| Beam assembly — step 1 | `beam_assembly_step1` |
| Beam assembly — step 2 | `beam_assembly_step2` |
| Screwing (furniture leg) | `screwing` |

## Local visual–tactile research extension

The `visuotactile/` extension implements three-view RGB-D policies, causal robot and tactile histories, trajectory recording, and a Transformer residual controller. The frontend remains the existing conditional U-Net; the DINO backbone stays frozen. Current work implements up to three bounded rounds of success-filtered expert takeover aggregation using a shared correction pool and the same data, training and evaluation modules.

The completed one-round DAgger checkpoints supply each initial frontend/backend pair. Each new round freezes a shared data snapshot, fine-tunes the frontend, refreshes its nominal cache, trains its paired backend and evaluates the autonomous student. Core, historical corrections and current corrections receive explicit sampling weights; the original validation and held-out correction development episodes have separate metrics. Update zero is measured and may remain selected. Whole paired models can be retained or rolled back.

Student prefixes retain actual observations and execution history with zero supervision. Only valid, actually executed expert suffixes from successful takeovers enter the correction pool. Command targets and reached configurations remain distinct, and test22 remains unused. Resume uses the original deadline and durable checkpoints; implemented scripts and CPU tests do not establish a new GPU launch or improved assembly performance. Read [the contracts](research_docs/CONTRACTS.md), [perception inputs](research_docs/PERCEPTION.md), and [results](research_docs/STATE_DP_RESULTS.md).

This repository, [mao455/bcrl_visualTactile](https://github.com/mao455/bcrl_visualTactile), is a research fork of the original
[Play2Perfect](https://github.com/kushal2000/play2perfect) release. The original
assets, environments, control semantics and RL code remain under the upstream MIT
license. New visual–tactile research code is kept in `visuotactile/`, with its
engineering record in `research_docs/`. Diffusion Policy modules copied under
`visuotactile/models/vendor/` retain their MIT license and source commit in
`LICENSE` and `PROVENANCE.md`. Checkpoints, trajectories, logs and local runtime
artifacts are intentionally excluded from the public repository.

## Installation

See [docs/isaacsim_installation.md](docs/isaacsim_installation.md) for the full Isaac Sim /
Isaac Lab setup (Python 3.11, `.venv_isaacsim`).

## Quick start

### 1. Download checkpoints

```bash
# Stage-1 play policy + the 4 finetuned assembly policies
python download_checkpoints.py
```

### 2. Interactive evaluation (viser)

```bash
# One problem from a single checkpoint
python evaluation/eval_isaacsim.py \
  --checkpoint-path pretrained_assembly/tight_insertion/model.pth \
  --problem tight_insertion \
  --port 8044

# ...or load all four downloaded policies and switch between them in the GUI
python evaluation/eval_isaacsim.py --policies-dir pretrained_assembly --port 8044
```

Open `http://localhost:8044` in a browser to watch the policy roll out the insertion.

### Offline success-rate evaluation

Headless, batched eval that scores each problem over many parallel envs (one
i.i.d. first-episode trial per env) and prints a success-rate table:

```bash
python evaluation/eval_offline.py --policies-dir pretrained_assembly
```

### 3. Train Stage 1 (play)

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-Play-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --headless
```

### 4. Finetune Stage 2 (precise assembly)

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-PreciseAssembly-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --checkpoint pretrained_policy/model.pth \
  env.precise_assembly.problem=tight_insertion \
  --headless
```

## Deployment

See [docs/deployment.md](docs/deployment.md) for the minimal sim-to-real reference.

## Formatting

```bash
./format.sh   # ruff over the Python packages + xmllint over URDFs
```

## Acknowledgements

This codebase builds on excellent prior work:

- [**rl_games**](https://github.com/Denys88/rl_games) — the high-throughput RL library our
  training loop is built on (vendored here).
- [**SAPG**](https://sapg-rl.github.io/) — Split and Aggregate Policy Gradients, the
  large-scale PPO variant we use for training.
- [**SimToolReal**](https://github.com/tylerlum/simtoolreal) — our prior work on
  object-centric tool manipulation, which the Stage-1 "play" pretraining is based on.

## Citation

```bibtex
@misc{lum2026play2perfectmattersdexterousplay,
      title={Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?},
      author={Tyler Ga Wei Lum and Kushal Kedia and C. Karen Liu and Jeannette Bohg},
      year={2026},
      eprint={2606.26428},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.26428},
}
```

## Contact

- Kushal Kedia — kk837@cornell.edu
- Tyler Lum — tylerlum@stanford.edu
