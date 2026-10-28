# Play2Perfect visual–tactile imitation research

## Authoritative overnight authorization — 2026-10-29

The latest user request overrides conflicting historical restrictions below. Target: **400 audited natural successes total by 2026-10-29 08:00 Asia/Singapore (00:00 UTC)**. Start checked at 01:19 local, about 6h40m remaining. Existing compatible v4 s930014 and s930016 count. Use persistent simulation, batched expert inference and independently recorded environments where validated; CPU/GPU and TiledCamera may be evaluated without upgrading shared installations. No training or changes to original hand/assets/control/physics/success criteria. Preserve all unrelated processes.

60GB/40GB limits and old absolute deadlines are superseded: manage actual free disk with reserve and bounded temporary queues. Time-weighted batch and complete 60-second GPU means must stay <=50%; do not throttle solely on instantaneous peaks. Common engineering failures must be diagnosed, fixed and retried within bounds with unique attempt IDs. No unconditional stop-on-first-error policy. Do not globally reseed other environments on individual reset. Check first 1–2 complete new successes, then continue production autonomously. Camera check is limited to current wrist and one lower candidate, keeping both v4 fixed views. Prefer retaining v4 unless the alternative is clearly better. Simplify outputs under data/demos/run/tmp; never delete unclassified partial files or valid data. Read `research_docs/NIGHT_COLLECTION.md`, current handoff and the latest user-direction entry; historical snapshots below are evidence, not current restrictions.


**STOPPED final snapshot, 2026-10-29 00:05 CST:** v4 production2/400, PID2070793 exited after s930017 native FileThread SIGABRT. One bounded1-step strace diagnosed real inotify_add_watch ENOSPC despite extension watcher off, user watch524205/524288. No more blind restarts, no global/other-process changes. Preserve s930014/016 plus all attempts; startup diagnostic is censored and not production. Read latest SESSION_HANDOFF before any resume. User400 authorization persists, six-hour deadline must not reset.

**Latest authoritative status (2026-10-28 continuation):** Read `research_docs/PROJECT_GUIDE_ZH.md` and the latest appended SESSION_HANDOFF. v4 close-high calibration passed two full expert pairs (220/214steps), 143-field lossless conversion and 3-view model connectivity; no training. Prior sections below are historical. Production is bounded by absolute deadline **2026-10-28T20:51:19Z** (starts from first probe14:51:19Z, never reset by new directory), all-project60,000,000,000bytes. `collection_resources.Supervisor` monitors startup AND conversion/audit, time-weights samples, marks60s window unavailable until populated; only its own new process group may be wall-paused. Original physics/control remain120/60Hz. `--global-output-guard` reserves uncompressed next row plus compact copy before every write. Renderer SM utilization is unavailable in pmon C+G samples; use conservative whole-card values. MALLOC_ARENA_MAX=2 is a finite process-local trial, not a proved LLVM fix; unknown native/audit errors still stop. Two acceptance records stay separate from new400; oldv3 three successes are rehashed and preserved, never relabelled. Ordinary production MP4s were explicitly made optional in manifest before cleanup. Lossless compact retains every original raw field.


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
