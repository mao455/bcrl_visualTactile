# Architecture and staged acceptance gates

## Frozen boundaries
Original environment/control/physics/RL remain authoritative. New package `visuotactile` is independent of Isaac Lab for model, data and history code; `visuotactile.sim` loads only after AppLauncher. Opt-in subclasses reuse task `_setup_scene`, `_pre_physics_step`, rewards and reset, attaching sensors/capturing records without replacing task math. No change to existing Gym IDs or teacher RL algorithms.

Policy-visible nominal inputs are explicitly selected camera history, encoder/command history, externally supplied task ID/goal command and their validity/time metadata. Correction sees its OWN visual/history encoder plus tactile history and frozen nominal predicted action chunk. It does not write features into the nominal network. Previous corrected commands must not become nominal action-history inputs; nominal action history is its own plan history. Encoder measurements can naturally reflect contact, which is unavoidable proprioception, not an extra tactile channel.

## Current implementation: perception v2

The current component and tensor contracts are in `PERCEPTION.md` and schema v2 in `DATA_SCHEMA.md`. `observations.py` owns the nominal allowlist, task vocabulary and operator command validation; `kinematics.py` supplies encoder-only FK; `models/vision.py` and `models/tactile.py` provide replaceable encoders. The adapter attaches two cameras/contact sensors and a read-only post-scene-update FK observer. Original task/RL/control/assets remain unchanged.

`NominalController.act` constructs private previous-nominal/remaining-plan fields **before** adding the current observation to history and predicting a new chunk. Collector records these exact inputs. `CorrectionContext` separately constructs prior-issued-command and target/error feedback. `CorrectionEpisodeDataset` uses the same causal windows and returns expert command supervision separately. `probe_perception` compares offline/online tensors on real episodes, checks gradients without weight updates, and generates visibility evidence. Goal commands are optional external inputs, not extracted from task success state.

v1 encoder/checkpoint paths remain explicitly available. The following earlier choices and milestone text are the **historical v1 baseline**, superseded by v2 where noted. No formal model training was performed in this perception milestone.

## Provisional initial implementation choices
- RGB-D per-view CNN + GRU over finite history; tensor view axis allows adding calibrated cameras without changing data schema. Start with one verified view, inspect occlusion before deciding on two.
- Conditional temporal denoiser and DDPM training / DDIM inference for action sequences, following Diffusion Policy. Preserve 29-D native normalized action space. Bounded action chunk execution at 60 Hz; 6-step replanning gives nominal 10 Hz. Independent correction period configured in control steps; no assumed high-frequency requirement.
- HDF5 episode files, complete step streams, metadata/provenance; dataset windows use same boundary padding/masks as online history. Full episodes separated from truncated captures. Train/validation split by episode, never random frame split.
- Initial legal task context: externally configured task identity (which assembly to perform), with fixture visible in image. No simulator goal pose or automatic success-driven phase enters policy. Fixed task ID may be insufficient; explicit commanded goal pose or vision-derived fixture estimator can be added with provenance, not privileged fallback.
- Contact encoder: per-finger vector MLP + temporal GRU; correction policy separately fuses visual/proprio history and nominal future chunk, predicts a complete candidate native action. Hard no-contact nominal bypass; configured blend on contact, clipping and original actuator path. This is a replaceable baseline, not a frozen research conclusion or guaranteed safety controller.
- Correction initially disabled unless a trained checkpoint exists. Candidate training: causal frozen nominal predictions on expert/DAgger states, imitate expert action in contact and regularize deviations outside contact. Expert action is target only, no future state inputs. Later contact-outcome prediction/risk learning and on-policy data required to evaluate improvements. No claim that fitting demonstration action difference proves tactile utility.

## Phases and gates
1. Audit + persistent contracts: trace action/observation/reset and primary code; document uncertainty.
2. Sensors + collection: real image inspection, actual distal-body force readout, timestamp/boundary/action records; original paths still runnable.
3. Dataset + nominal: causal window/reset/leak tests, real recorded data denoising training, checkpoint save/load and independent policy rollout. Small data smoke test has no performance claim.
4. Correction: separate encoder and policy, real input/output/combination integration; disabled controller exact equivalence. Only train if real paired data and meaningful objective available.
5. Evaluation: same seeds/tasks/success rule, log success + force proxies + latency + provenance; camera and tactile ablations; significant data collection/GPU runs deferred while protected background workload occupies GPU.

## Open research questions
Nominalization of preload; tactile force calibration/slip observability; legal target pose estimator and phase logic; camera visibility/3D ambiguity; history length; denoiser capacity/step count; correction objective and output representation; contact threshold/blending; frequency vs inference latency; CPU/GPU physics differences. Track evidence rather than silently freezing choices.

## Implemented call graph (2026-10-26)

`collect.main` boots AppLauncher → loads the original task config → `InstrumentedAssemblyEnv` → original `PreciseAssemblyEnv` → `PlayEnv` → installed `DirectRLEnv`. Native `env.step` remains the only execution entry. The subclass adds camera/contact reporting, captures pre-reset diagnostics and asserts kinematic table/hole positions remain at their reset values. It never implements reward/action math.

| Files | Responsibility and boundary |
| --- | --- |
| `sim/environment.py` | Optional scene instrumentation and cached sensor packet; no new collisions or gains |
| `collect.py`, `recording.py` | Released teacher or independent nominal rollout; per-environment HDF5 episodes, real action provenance |
| `history.py`, `dataset.py` | Shared causal indexing; successful teacher episodes; expert actions are supervision only |
| `models/nominal.py`, `models/vendor/` | RGB-D/proprio history encoder and attributed MIT diffusion denoiser; exact input allowlist |
| `train.py`, `rollout.py` | Real-data loss, checkpoint/optimizer/RNG restore; native 60-Hz chunk cursor and 10-Hz replanning |
| `models/correction.py` | Independent tactile encoder, visual encoder, motion/issued-action history GRU and predicted-chunk GRU → full 29-D candidate; bounded composition helper |
| `correction.py` | Causal feedback window and shadow integration; no trained checkpoint loader and no live candidate execution |
| `probe_correction.py` | CPU-only replay of real contact episodes with frozen nominal predictions; validates past-command/reset alignment, never trains or executes |
| `contracts.py`, `evaluate.py` | Input/calibration compatibility, experiment fingerprints, full-trajectory alignment checks and comparable metrics |
| `regression.py`, `tests/test_visuotactile.py` | Separate-process rendered reference and local information/reset/control tests |
| `run_cpu.sh` | Limits only the newly launched process; leaves other workloads untouched |

### Histories actually implemented

Nominal sees four RGB-D/encoder observations at stride 3, spanning 0.15 seconds. Its internal GRU starts from zero for each finite window, with padded frames masked. Own predicted action chunks/cursors are stored for analysis and execution; they are **not yet neural inputs**. This choice avoids feeding demonstrated or corrected commands into nominal. Causal self-plan conditioning is a later ablation requiring a consistent training rollout construction, not an excuse to substitute expert history.

Correction has its own matching visual history plus twelve consecutive encoder/issued-command samples at 60 Hz and 24 chronological force samples at 120 Hz. A command stored at observation t is the command actually issued at t−1. At a full reset its validity is false. Current/future nominal predictions start at the active chunk cursor; padded unpredicted tail is masked. No phase/reset diagnostic is fed as task information. Episode/step IDs only manage lifecycle. Subgoal changes do not reset any policy history.

The shadow path computes the candidate on real measurements, then records it under `transition/correction_shadow/untrained_candidate_diagnostic`. The environment always receives nominal. This establishes tensor/lifecycle integration, not a learned controller. A trained correction loader, justified objective, data coverage and matched rollout ablations remain acceptance gates before enabling correction.

### Development order from this milestone

1. Improve/validate camera coverage and lawful goal command information under occlusion; retain fixed external task identity meanwhile.
2. Gather more successful episodes with scene health and provenance checks; train a competent nominal baseline and evaluate held-out reset seeds.
3. Generate causal frozen-nominal plans on expert and on-policy states; optionally query the expert for labels. Separate these from model inputs.
4. Compare proposed full-command supervised correction and contact-outcome objectives; train only from real transitions. Require no-contact equality, tactile-zero/shuffle and same-budget nominal-only ablations.
5. Only then enable trained correction and consider constrained joint fine-tuning. Add self-plan history, additional views or more expressive temporal models based on measured failure modes.
