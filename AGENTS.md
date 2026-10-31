# Play2Perfect visual–tactile imitation research

## Completed — 2026-10-31 DP loss study and Transformer residual

Read DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031.md and the latest SESSION_HANDOFF.
The single bounded run dp_loss_trt_20261031_01 finished at05:01:00 UTC;
all owned trainers, cache workers, simulators, owner and monitor exited.
Four DP lines each completed12 additional train320 epochs/9984 updates;
validation selected G00-A epoch8 and the other three epoch12, test40 once each.
Force/masked Transformer each completed5 epochs/3730 updates, selected epoch5,
test40 once each. Loss B modestly improves prefix command errors; residual
contact arm error improves about25% offline, but masked is slightly better.
All18 autonomous diagnostics are complete: each of six groups0/3, all601-step
timeouts, zero subgoals/native lift. G10-B briefly rises7.69cm then drops.
Saved physical initial states match, but precontact commands differ across
base/force/masked; these are not strict trajectory pairs or causal benefit evidence.
New G01 uses user-authorized past_actual_mixed_commands_v1, allowing indirect
residual/tactile influence on nominal. Same-input wrapper and trained residual
parity passed; existing DP BF16 batch/native recurrence is not bitwise equivalent.
Whole-card mean33.1988%, maximum complete60s39.4259%, missing/violations0;
minimum free VRAM10.6592GiB/RAM28.8657GiB. Earlier session violations remain.
Original400/bests/formal0/50/native control/sim wrapper remain unchanged.
No automatic new epochs, trials, G11,20Hz, roll-in or retries; only reviewed
code/docs/small statistics/curve PNGs go to origin feature, never raw artifacts.

## Current override — 2026-10-31 DP loss comparison and Transformer residual

User explicitly authorizes new paired DP A/B continuation, independent trained
Transformer correction, bounded diagnostics, curves and origin feature push.
Read DP_LOSS_AND_TRANSFORMER_RESIDUAL_20261031.md before changing this stage.
Use one new run/session with a six-hour total window; no deadline reset.
Freeze all native control/physics/DR/reset/reward/timeout and sim wrapper.
Original400/split/bests/formal results protected. Both losses use sample/x0,
same starting epoch40 EMA, fresh optimizers/schedules, equal paired budgets.
Whole-card batch/complete60s means<=50%, target35–40%; reserve4GiB/10GiB.
Historical no-new-training clauses below do not override this explicit task.
User subsequently explicitly authorizes new G01/G11 to read past actually
executed mixed commands, with nominal's indirect residual/tactile influence
clearly disclosed. Use past_actual_mixed_commands_v1 for the new residual;
expert replay is teacher forcing, deployment uses actual submitted/delayed
feedback. Original DP A/B and old checkpoint contracts remain unchanged.

## Completed — 2026-10-31 long-horizon candidates

Pipeline finished at03:41:43 Asia/Singapore; all owned workers/monitor exited.
G00-new/G10-new each completed40 train320 traversals/33280 updates, validation
selected epoch40 EMA, held-out test40 once, three autonomous diagnostics each.
Both0/3: five601-step timeouts; G10 seed981004 native lift(+0.107093m), then
fall at600 steps. All zero subgoals; fingertip force does not establish stable
object grasp. Test native arm errors1.14538/1.15438 exceed zero baseline0.95577.
Read LONG_HORIZON_20261030 and evidence/long_horizon_final_20261031.json.
Six finite traces, H64/replan12/reference/native60Hz alignment, three exact
saved initial-state pairs, deployed/test/best hashes and two video hashes pass.
Adaptive training whole-card mean37.8641%; entire replacement monitor16.7201%,
maximum complete60s41.3339%, missing0. Earlier RAM9.825729GiB and preliminary
telemetry gap remain: do not claim full-session resource compliance.
No automatic new epochs, test, diagnostics or retries; original400/bests/formal
0/50 remain protected. Later sections contain historical running snapshots.

## Current override — 2026-10-30 long-horizon pretrained DP

User authorizes new G00-new/G10-new implementation, real bounded training,
three preregistered autonomous starts each, classified artifact deletion,
and reviewed code/docs/small-report commits/pushes to origin feature only.
Deadline 2026-10-31 12:00 Asia/Singapore; reserve final two hours for diagnostics.
Read LONG_HORIZON_20261030.md and current handoff. Fixed native120/60Hz;
planner5Hz/H64/commit12, explicit>=2s histories plus native60Hz persistent memory.
Pinned frozen DINOv2-S/14; absolute pre-delay references converted through
native commands/env.step. Original400/split/native control/old results protected.
All owned GPU workers share long_resources lease and continuous monitor.
Preliminary long_horizon_20261030 outputs INVALID: current-command leakage,
wrong split. Never deploy or count their training. Replacement root:
outputs/visuotactile/run/long_horizon_20261030_v2. Completed-stage no-training
and no-cleanup restrictions below do not override this authorization.
Epsilon pilots completed3300 updates each but did not pass arm-command fitting.
Independent sample-target pilots use the same architecture/cache and3300-update budget;
never flip an epsilon checkpoint's prediction_type. Explicit fitting gate and
background continuation are in run_long; failures do not authorize reruns.
Read actual pipeline/status/report and the current handoff, not historical PIDs.
Latest2026-10-31: user targets whole-card GPU mean around40%, retaining both50% caps.
Shared adaptive pacing replaces fixed2.5x sleep; 45% admission and6GiB/14GiB headroom.
Sample pilots passed; full training continues from2845/2852-update episode checkpoints
after a reviewed runtime-only scheduler restart. Do not reset budgets or repeat test/trials.

## Completed — 2026-10-30 videos and exact parameter census

Six A/B/C viewing runs (seed981002, autonomous and expert roll-in) are complete;
all601-step native timeouts, no subgoal/lift. Autonomous gates0; local switch88,
513 student steps, observed gate counts6/10/7. Expert prefixes and saved switch
states match exactly; images/plans are not claimed bitwise equal. All six plus
four previous B0/B1 videos fully decoded; index is ignored demos/residual_20261030_01/README.md.
Read EXPERIMENT_GROUPS_AND_PARAMETERS_20261030 and the newest SESSION_HANDOFF.
Whole-card mean27.9960%, maximum complete60s60.3794%: the50% window limit WAS
exceeded in four recorded intervals despite admission waits; never label this
session fully resource-compliant. Telemetry missing0; VRAM/RAM reserves held.
Recorder/monitor/all owned workers exited; no training started, no auto-reruns.
Original400 manifest/best/100 formal traces/0/50 and11 prior residual traces intact.
Pinned full neural examples were counted on CPU/meta,57 imported source hashes
verified; no author checkpoints/robot-task reproduction or new RL claims.

## Current — 2026-10-30 experiment videos and parameter census

The user requests representative experiment videos, per-group data explanations,
a full pinned-source DP parameter comparison, and reviewed commit/push to origin.
This authorizes a separate viewing cohort, not new training or changes to old counts.
Record at most six new runs: seed981002, A/B/C, autonomous and real expert roll-in.
Reuse the four existing B0/B1 videos in the index. Use fresh outputs and a two-hour
viewing window including resource waits; all original control/resource/data rules hold.
Unmatched switch states must be labeled, never called paired performance evidence.
Report precise parameter-count scope, source pins and unknown published settings.

## Completed — 2026-10-30 independent residual v1

This authorized bounded session is finished; all owned workers and monitor exited.
Read CORRECTION_TRAINING and the newest SESSION_HANDOFF before historical progress.
B/C each completed5 full epochs,7875 updates,320 unique training episodes; best epoch5,
one held-out test40. Offline command errors improved, but no task success was observed.
All12 preallocated starts are resolved:11 native601-step timeouts, one switch-validation
interruption before student control. The failed eighth trial was not rerun; only9–12
continued in a fresh directory under the same session/deadline. One local seed's C is
explicitly unmatched; seed981002 has a complete exact-switch A/B/C triad, all failures.
Autonomous A/B/C each0/2 with no contact gate, no first subgoal or lift. No tactile
causal benefit or RL is established. Original400/best/100 formal traces and0/50 are intact.
Whole-card mean22.0602%, maximum complete60s37.1971%, telemetry missing0; early worker
wait-attribution gaps remain documented.82 tests plus real artifact checks passed.
Do not automatically restart, add trials/epochs, or launch DAgger/SAC from suggestions.
This session's reviewed code/docs/small sanitized statistics are authorized for the
origin feature branch only; native controller and research sim wrapper remain read-only.

## Latest override — 2026-10-30 trained independent residual Correction

The current user authorizes policy/data/history changes, first independent
Correction training, bounded autonomous and expert-roll-in diagnostics, and
reviewed code/docs/small sanitized report commits/pushes to an origin feature branch.
Freeze formal B0 v2/no-own-plan; never feed mixed commands or tactile into nominal.
New explicit residual version uses observed-contact exact-base bypass; train on
causally generated B0 plans over existing expert observations, labeled teacher forcing.
First GPU stage <=90 minutes; entire training/diagnostic window <=6 hours including
waits. Start fixed train16/val8 complete episodes, then at most5 full-split epochs;
at most5 additional epochs only with validation improvement and remaining budget.
At most12 preallocated diagnostic rollouts; local expert-roll-in outcomes stay
separate from autonomous performance and original100 formal trials. No full50 epochs.
Native controller/assets/physics/reset/DR/reward/timeout and research sim wrapper
remain read-only. Whole-card batch and complete60s GPU means <=50%,45% admission;
4GiB free VRAM/10GiB RAM and disk reserves, continuous startup-through-exit telemetry.
Preserve original400 data, best weights and0/50 reports; no unrelated process changes.
This overrides historical no-Correction-training/no-next-stage instructions below.

## Latest override — 2026-10-30 independent audit and action history

The current user authorizes independent source/result/upstream review, evidence-based
adapter/model/data fixes, versioned action history, lightweight tests, a few diagnostic
rollouts and bounded small-demonstration overfit checks. Preserve old checkpoints,
400 records and formal 0/50 results. No automatic full retraining, recollection or
100-trial evaluation. Native controller/physics/reset/timeout remain read-only.
Whole-card time-weighted and complete 60-second GPU means must remain <=50%; retain
4GiB VRAM/10GiB RAM and disk reserves, record work/wait/missing telemetry.
Past executed expert commands may be explicitly teacher-forced; never call them
student predictions. New history changes checkpoint contracts; never toggle old weights.
The user explicitly authorizes focused commits and pushes of reviewed code/docs/small
sanitized reports to origin on a feature branch; never push upstream or raw artifacts.
Current audit progress and evidence supersede the viewing-only scope below.
Audit completed: see research_docs/INDEPENDENT_AUDIT_20261030.md and latest handoff.
Version3 issued history passed real closed-loop alignment; short-prefix fitting did
not improve approach/lift or task success. Seven diagnostic rollouts are separate
from the original100; expert succeeded once, six students timed out. One pre-action
native SIGABRT attempt is preserved; its sole new-directory recovery completed.
All owned diagnostic jobs exited. Do not restart these or the proposed next phase.

## Latest override — 2026-10-30: completed baselines, four videos, GPU mean <=50%

B0/B1 each completed 50 epochs; validation selected epoch49 for both. Formal
50 paired legal starts /100 trials completed at07:39:36 Asia/Singapore, all
601-step native timeouts with zero first-subgoal hits, 0/50 successes each.
No training/evaluation process is currently active. Do not restart training,
collection or the full evaluation automatically. Read the latest
BASELINE_TRAINING and SESSION_HANDOFF before using historical progress below.

The user NOW restores whole-card time-weighted batch and complete60-second
GPU mean <=50%, superseding yesterday's unrestricted GPU permission. Keep
4GiB free VRAM/10GiB available RAM, bounded queues, and protect unrelated jobs.
ResourceGuard defaults to a50% cap with45% admission margin; wait only our
process between chunks/control calls, never alter120/60Hz dt or action scale.
A sampled gate cannot prevent unpredictable external bursts; report actual
means, gaps and waits, never claim per-process attribution or a hard GPU quota.
User requests two B0 and two B1 videos. Use the first two legal formal seeds
981001/981002 with actual best checkpoints, three native views and full natural
termination. These are separate viewing reruns, not added formal test samples.
Preserve original100 results and all data/checkpoints. No new parameter training.
Video delivery completed at10:35:17 local: four601-frame/60fps MP4s in
outputs/visuotactile/demos/baselines_20261030 (README is the viewing index).
Whole-card mean29.50%, maximum complete60-second mean38.15%, missing0; service
exited. Video reruns are not bitwise replays; B1 seed981002 diverged late from
the original numeric trace, documented without changing the formal results.

## Historical authorization and snapshots (overridden above where conflicting)

## Latest override — B0/B1 command baselines, 2026-10-29

User now authorizes B0/B1 parameter training and paired evaluation: 400 production
records, fixed 320/40/40 split, 50 complete epochs each, three 448×252 RGB-D views,
H=12/replan=6, memory2 and own_plan_mode=none at training AND deployment.
Supervision is command_clipped_pre_delay; B1 separately adds observed five-finger
force/FK history and a fusion GRU. These are command imitation baselines, not
preload nominalization. B2/B3/Correction/RL training remain out of scope.
The user's latest message permits full GPU utilization; the historical 50% cap
is superseded for training. Keep RAM/VRAM reserves and bounded data queues.
Common local engineering fixes are authorized. Preserve unrelated productive
jobs. Read the current B0/B1 section of SESSION_HANDOFF and BASELINE_TRAINING.md;
old PIDs and no-training clauses below are historical. No automatic recollection.

Current background root is `outputs/visuotactile/run/dp_command_20261029`;
`python -m visuotactile.run_baselines --root <root> --status` reads status only.
Both trainers use B=2/T=32/BF16, bounded prefetch and independent parameters.
User service has MemoryHigh12G/MemoryMax16G, no swap; wait when VRAM free<4GiB
or RAM available<10GiB. Full GPU permission does not remove these memory gates.
STOP asks owned children once to save at a complete episode-group boundary.
Do not infer a process is live from historical PIDs or report early diagnostic
weights as final results. Best-validation weights precede test and 50-pair evaluation.

Latest review (2026-10-29 evening): user explicitly authorizes publishing 76bb58e
and its review follow-up. It is public on origin/main. Do not restart active
trainers to apply evaluation-only changes. Runtime launch hashes match 76bb58e;
future evaluation imports the new protocol2 implementation: persistent CPU
physics/RTX scene, CUDA FP32 policy/images/memory/DDIM, full three-view quality,
50 matched legal pairs, best epoch/hash admission, no incomplete report marker.
Keep memory guards; the old 50% utilization cap remains removed. Exact current
evidence and ETA: BASELINE_TRAINING.md and evidence/baseline_review_20261029.json.


## Authoritative overnight authorization — 2026-10-29

The latest user request overrides conflicting historical restrictions below. Target: **400 audited natural successes total by 2026-10-29 08:00 Asia/Singapore (00:00 UTC)**. Start checked at 01:19 local, about 6h40m remaining. Existing compatible v4 s930014 and s930016 count. Use persistent simulation, batched expert inference and independently recorded environments where validated; CPU/GPU and TiledCamera may be evaluated without upgrading shared installations. No training or changes to original hand/assets/control/physics/success criteria. Preserve all unrelated processes.

**Final production status (measured 2026-10-29 03:26:58 Asia/Singapore):** `outputs/visuotactile/data/close_v4/manifest.json` contains 400/400 audited `stage=production` episodes. `progress.json` and `status.json` are `complete`; the collector and supervisor have exited. Do not restart collection. Full SHA256/size verification and production Dataset (`stage="production"`) spot checks passed. Final evidence and statistics are in `research_docs/NIGHT_COLLECTION.md` and `research_docs/SESSION_HANDOFF.md`.

60GB/40GB limits and old absolute deadlines are superseded: manage actual free disk with reserve and bounded temporary queues. Time-weighted batch and complete 60-second GPU means must stay <=50%; do not throttle solely on instantaneous peaks. Common engineering failures must be diagnosed, fixed and retried within bounds with unique attempt IDs. No unconditional stop-on-first-error policy. Do not globally reseed other environments on individual reset. Check first 1–2 complete new successes, then continue production autonomously. Camera check is limited to current wrist and one lower candidate, keeping both v4 fixed views. Prefer retaining v4 unless the alternative is clearly better. Simplify outputs under data/demos/run/tmp; never delete unclassified partial files or valid data. Read `research_docs/NIGHT_COLLECTION.md`, current handoff and the latest user-direction entry; historical snapshots below are evidence, not current restrictions.


**Historical snapshot (superseded by the final status above, 2026-10-29 00:05 CST):** v4 production was then 2/400 and PID2070793 had exited after a native FileThread SIGABRT. One bounded 1-step strace diagnosed inotify_add_watch ENOSPC. The later bounded recovery completed the authorized batch; do not use this paragraph as a current count.

**Historical implementation status (superseded by the final status above):** The following production-boundary details remain useful implementation evidence, but their old counts and deadlines are historical. Read the latest `SESSION_HANDOFF.md` and `NIGHT_COLLECTION.md` for the completed batch. `collection_resources.Supervisor` still monitors startup and conversion/audit; original physics/control remain 120/60Hz; lossless compact retains every original raw field.


Read `research_docs/USER_RESEARCH_DIRECTION.md` first, then `research_docs/README.md`, `CONTRACTS.md`, `HISTORY_AND_TRACE.md`, `ARCHITECTURE.md`, `PERCEPTION.md`, and the latest entries of `CHANGELOG.md` before changing this research system. This is a long-term research project, not a replacement of the original RL environment.

For released-expert collection/comparisons, also read `research_docs/RL_EXPERT_TRIALS.md`. Original includes native DR; force_1p5 changes only force_scale. Record requested world wrench separately from actual per-physics-step composer buffers/world conversion (wrench_trace3): this installed Lab caches body-frame wrench transforms. Never silently fix the shared simulator or count zero-exposure pairs as robustness evidence. Preserve failed attempts; only a pre-HDF5 startup SIGABRT may be retried once under the same seed in a new directory. Software errors are separate from task failures. The user authorized bounded expert evaluation for this milestone, with zero parameter training.

**Previous bounded dataset milestone (historical; latest authorization below supersedes its expansion limit):** read `research_docs/SESSION_HANDOFF.md` and `INSERTION_DATASET.md` before further collection. Its stricter rule is **no retries or replacement seeds**; it stopped after 20 attempts (17 successful records, two pre-action invalid initializations, one Kit SIGSEGV). Do not automatically resume, enlarge to 400/1200 or train. The optional local `world_hold` wrapper was verified for real force exposure; default legacy/shared Lab remain unchanged, never pool configurations. Old pilot includes initial box penetrations: old75% is not a legal-start production estimate. Storage1 has formal lossless RGB/depth HDF5 plus two viewing-only MP4s. Manifest Dataset defaults to successful scan episodes; split initial-state groups. Pure expert own-plan conditioning and memory2 validation are unresolved; formal trainer guards reject misuse. Raw outputs stay local.

## Current authorization: triple cameras and 400 successful episodes

The latest user explicitly supersedes the previous bounded milestone's no-400 rule. Read `research_docs/CAMERA_VALIDATION.md` and the latest `SESSION_HANDOFF.md`. After camera visibility, paired expert regression, dynamic calibration, storage and resource gates pass, run a NEW frozen triple-view batch toward 400 audited natural successes; do not ask again whether to start. Do not restart or mix the old batch. Current production profile: `wrist_bracket`, two fixed + link7-mounted wrist RGB-D, 448×252, quality / AA Off, original released expert, world_hold, 120Hz physics / 60Hz control. Original dual-camera defaults still work. Camera layout3 / perception2 / compact storage2: old dual checkpoints must fail compatibility. Use render-product CameraParams for moving calibration; CPU USD camera.data pose is stale in this installed fork. Diagnostic world poses never become nominal inputs. Preserve unknown/partial files. `collect_success_batch` stops on software/audit errors, source drift, wall/disk bounds; 400 means valid successes, not attempts. No training. The bracket's real hardware fit remains unverified.

## Non-negotiable constraints
- Preserve original assets, collisions, limits, friction, gains, dynamics, coordinate transforms, reward/success criteria, reset, noise and delay mechanisms. Extend opt-in; do not edit vendored RL to implement BC.
- Nominal diffusion policy must never read tactile features, correction outputs/mixed commands or privileged state. Its default action context is private own predictions. Past expert commands may be explicitly labeled teacher-forcing history in a separately specified future variant; current/future expert commands remain supervision, never model predictions. Explicit typed/allowlisted inputs, separate training supervision. Ordinary BC is NOT demonstration nominalization; pressure intent can remain in its targets.
- Tactile encoder and correction policy are separate modules. No fake losses/random labels; an untrained correction must not control the robot.
- Keep 120 Hz physics / 60 Hz original action processing. A 10 Hz policy can produce/replan chunks of 60 Hz actions; do not change decimation to change network frequency. Arm and hand action semantics differ.
- Trace each observation to a real sensor or declared external command. Simulation object/goal poses, success stage, reward, force disturbances and future state are diagnostics/labels only unless a separately validated estimator supplies them.
- Persistent causal memory updates at every native observation, separately from replanning; per-environment episode IDs and reset masks are required. Chunk boundaries and gradient detach do not clear memory. Do not use random zero-state windows for memory2 training. Never use episode_length_buf==0 alone: it also resets at task subgoals.
- Keep full raw trajectories and action provenance. Trace2 distinguishes submitted/clipped/delayed commands, target radians and actual motion; terminal images are unavailable, never substitute reset frames. Only a read-only optional hook was added to original action_utils; do not change its math or RNG. Never silently call unsuccessful expert rollouts successful demonstrations.
- User's background GPU workload is protected: no signals, renice, driver or global changes. Use low priority, two CPU threads, few environments. RTX cameras still use GPU; monitor and bound probes; do not claim zero contention.
- Every substantive change updates research_docs/CHANGELOG.md (paths, before/after, I/O, tests and actual results), contracts when changed, and current status. Clearly distinguish implemented, tested, and research hypotheses.
- Do not automatically freeze open architecture/hyperparameter choices. Do not claim performance from smoke tests.

## Environment
Current perception defaults to v2: two calibrated fixed RGB-D cameras, 224×126, metric depth + explicit validity, encoder-only URDF FK and five normal-force sensors. Read the versioned schema before interpreting images/checkpoints. v1 is explicit legacy, single-view normalized depth; never upgrade its weights/data by changing dimensions. Nominal's own-action/remaining-plan history is privately generated. Separately returned expert_history is not automatically an encoder input. memory_version2 (persistent recurrence) and trace_version2 (execution provenance) are independent of perception/schema2; missing memory_version means legacy1. Old weights must never silently acquire new memory semantics. Optional goal poses must be operator commands in robot_base; absent poses stay masked. Correction gets target/tracking-error history but remains shadow-only. No correction training objective has been established. Perception probes use forward/backward checks with zero parameter updates; do not promote their untrained predictions to formal training data. Task ID is categorical, a provisional interface; shared task structure and final goal representation remain research questions. Preserve user direction changes append-only in USER_RESEARCH_DIRECTION.md, separate from engineering decisions/test evidence. This project excludes the separate cuRobo/small-object/batch-PKL project. No formal training is authorized in this milestone. Camera/force timestamps are synchronous simulation acquisition times, not a validated asynchronous real sensor stack.

Use `.venv_isaacsim/bin/python` (project-local overlay of conda isaaclab), not base Python 3.13. Original Isaac Lab source is a local 2.3.2 fork. Use repo rl_games. Boot AppLauncher before importing Isaac Lab simulation/task modules. See prior `outputs/compat_check/REPORT.txt`.

**Critical CPU regression finding:** the original CPU no-render path resets kinematic table/hole/goal transforms to their spawn origin on the first physics step in this installed stack. Prior no-render success rates are not valid quality evidence. A rendered-original reference and the instrumented path are bitwise equal over the documented 32-step probe, including objects/goals/RNG/rewards. See `research_docs/AUDIT.md`. Do not silently disable rendering for CPU evaluation or claim global Isaac Lab compatibility from that bounded test.

Use `bash visuotactile/run_cpu.sh ...` for new tests (nice 19, two threads, affinity 22,23). Never signal unrelated jobs. Fresh output directories are mandatory. Student camera augmentation/delay/pose randomization are currently rejected, not silently disabled; original teacher/action DR is unchanged. New tests must preserve this distinction.

## Public repository workflow

Read `research_docs/GIT_WORKFLOW.md` before repository operations. `origin` is the user's public fork (`mao455/bcrl_visualTactile`); `upstream` is `kushal2000/play2perfect`. Never push to `upstream`. Before a public push, inspect the staged file list, ignored artifacts, credentials scan and license/provenance files. Do not upload `outputs/`, `pretrained_*`, checkpoints, trajectories, logs, virtual environments or private configuration. Keep the original commit history and MIT license. After each substantive code change, update the changelog and relevant contracts/status, run the focused tests, make a focused commit, and report its ID and GitHub URL. Pushing requires explicit authorization for that update.

Production startup follow-up: read the appended CAMERA_VALIDATION/SESSION_HANDOFF. New400 uses process-local `--disable-extension-watch` after two complete paired probes; no system/IDE changes. First root `insertion_triple400_20261028` is blocked before any episode (seed930000). Current successor is `insertion_triple400_watchoff_20261028`, preserving that attempt and continuing at930001; never retry or erase the predecessor. Stop again on unexplained software errors. Do not mistake old startup/validation counts for successful production records.

Latest production status (2026-10-28 21:18 CST): **blocked at 3/400, no driver running**. Successes s930001/2/3 are committed and rehashed; s930004 rejected before action; s930005 failed before scene/HDF5 with LLVM out of memory, distinct from prior watch exhaustion. Preserve all three successes and error evidence. The existing zero-episode predecessor option must reject this nonempty batch; do not delete data to bypass it. Diagnose native startup allocation before a deliberate resume; user400 authorization persists.

## Current milestone override (2026-10-28 late evening)

The user has now set the production hard limits to **60,000,000,000 decimal
bytes across all retained and temporary project outputs**, a **six-hour wall
clock budget**, and a conservative **whole-card GPU utilization average of at
most 50%**.  The collector records five-second whole-card samples, a rolling
60-second mean and a batch mean; process attribution is not trusted on this
host, so the metric includes unrelated GPU work.  Throttling may only sleep
between episodes and must never change the 120 Hz physics or 60 Hz native
control contract.  If telemetry is unavailable, the GPU gate is unproven.

The newly selected calibration is `wrist_bracket_close` (camera layout version
4).  It changes only `fixed_high` from `(0.32,-0.45,1.55)` to
`(0.288,-0.405,1.463)` with the same target, focal length, side view, wrist
mount, resolution and rendering settings.  Version-3 episodes must stay in a
separate dataset; metadata must be checked for all three camera IDs before a
trajectory can count.  MP4s are optional viewing derivatives; lossless HDF5 is
the source of truth and a manifest must explicitly list any video files.

On 2026-10-28, version-4 seeds 930006--930010 were probed.  930006, 930008
and 930009 were rejected before any expert action for the native object/table
/hole box-overlap gate; 930007 completed with only two cameras because a name
mapping bug was found and fixed, so it is diagnostic and not a three-view
success; 930010 stopped at startup with `LLVM ERROR: out of memory`.  Do not
blindly retry this blocker or count these probes as the two required final
configuration successes.  Preserve the evidence, classify/delete only useless
payloads, and keep the old three version-3 successes untouched.

Read [INITIALIZATION_AND_PERTURBATIONS.md](research_docs/INITIALIZATION_AND_PERTURBATIONS.md)
for the concrete reset/DR trace and three real initial-state rows.  It is a
report of measured data, not a replacement for the original task configuration.
