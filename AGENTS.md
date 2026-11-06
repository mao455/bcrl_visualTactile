## 2026-11-06 当前授权：数据可信度与StateDP变更审计

先重读研究目标与冻结历史，逐条审核275条原始数据全部模态、分布、减速、成功/失败及采样偏差；基点84cbb98至0c209b9做真实Git/AST/运行SHA核查。静态通过后独立分层专家/逐tick target/H24回放，root dataset_trust_20261106_01，固定3h含准备等待、最多60轨迹启动、8GiB新增/60GiB磁盘保留、共享目标35%及6GiB VRAM/12GiB RAM。仅CPU审核和冻结RL专家/记录命令，正式数据新增0、学生rollout0、optimizer更新0。所有原400/275、manifest、正式checkpoint和失败证据只读。审计报告和准入建议为终点，用户决定下一阶段；不自动扩采/重训/学生误差调参/Image/nominal/触觉。H24/12、5/60/120Hz、target_after_rad而非q_after、29D、网络/损失/归一化/历史/原生物理保持。新脚本独立，原始验收标记不改写。

# Completed — StateDP mechanism diagnosis (2026-11-06)

state_mechanism_20261106_01 finished19:39:53 UTC+8; all15 starts resolved,
all owned workers/monitor/owner exited. No automatic restarts or new GPU work.
A native2/2, processed-target replay1/2, recorded-slow0/1(censored); requested,
submitted and actuator targets match exactly, hidden physical state is not exact.
B single5s train trajectory stopped800 updates after two fitting gates; Arm~.00199,
Hand~.00735rad/FK~1.74mm but closed loop0/2, no lift/subgoal. C0/C4 each0/4,
2/4 native lifts and zero subgoals; overlap4 reduces average arm boundary jump44%,
without established task benefit. Tiny first observable/actor inputs match source;
early error grows in closed loop. This supports error-amplification diagnosis,
not a unique causal conclusion or permission for DAgger/retraining/recollection.
Protected hashes and best800 optimizer/EMA readback pass. Legacy per-rollout
checkpoint_sha256 is frozen expert/shadow provenance; actual student source SHA
is separately documented in final_review.json; do not confuse these identities.
Whole-card mean27.47%, max complete60s52.11%, missing0; do not claim <=50%
throughout. Keep formal data/models, tiny best/last, traces and all8 videos.
Read STATE_DP_RESULTS final review and SESSION_HANDOFF. The following authorizes
this completed run and is historical, not another run.

# Authorization — bounded StateDP mechanism diagnosis (2026-11-06)

Latest user explicitly authorizes CPU contract/data/sampler audit, real full-model
microbatch gradient equivalence, five expert/target Oracle starts, ONE fresh
single-training-trajectory overfit with diagnostic multi-window sampling, and
fixed State275 best4500 overlap0/4 comparison. Read current CONTRACTS and
SESSION_HANDOFF, then actual state_mechanism_20261106_01 status. No formal data
collection, formal-model continuation, network/loss changes, Image/tactile or DAgger.
New root has fixed4h INCLUDING waits from prepare,15 Kit starts (A5/B2/C8), tiny
training<=4000 updates/90min,12GiB output/60GiB disk reserve. Same shared_recent_v1
40% target/6GiB VRAM/12GiB RAM. One monitor/worker at a time, no automatic retry.
Prior completed-run prohibitions below do not override this explicit new task.
Preserve400+200+75, formal best/last and all failed evidence. B uses one5s TRAIN
trajectory, full-prefix history/mask/loss and unchanged train-only normalizers;
its16/9 shuffled anchor batches differ from formal64/64/64/15 episode sampling.
A position replay is open loop; exact hidden physical replay is NOT a gate.
C4 is diagnostic only; formal overlap remains0. Short real startup verification,
then yield without LLM polling. Existing MD/one small result auto-publish only
origin feature/narrow-state-data when reviewed HEAD/worktree guards pass.

# Completed — camera/history repair and the single StateDP contrast (2026-11-06)

Run narrow_repair_20261106_01 finished15:19:38 UTC+8, all owned processes exited.
Read STATE_DP_RESULTS/SESSION_HANDOFF and final_review.json. Collection76 attempts,
75 successes/1 native fall,32 new families; total275 State-valid,75 Multi-view-valid.
Fresh StateDP stopped5000 updates at validation plateau, best4500; old200 was not
extended. Final old/new each train0/4 and unseen0/10 legal, plus2 initial rejects
per model. All32 planned starts resolved; mostly unresolved grasp/lift. Offline
improvements did not establish task benefit. Earlier recheck train1/2 is not stable
performance. No automatic new training, trials, data expansion, Image or tactile.
Keep all400+200+75 formal data, failures, best/last, caches and videos. Only two
completed disposable smoke checkpoints removed at final review (2,306,195,456B).
Measured whole-card mean30.81%, max complete60s52.19%; three monitor handoff gaps
total2518.41s, query failures0. Do not claim <=50% throughout or gap-free telemetry.
Completed status counter corrected from stale0 to actual5000; original snapshot
preserved. The authorization and startup notes below are historical, not new work.

# Authorization — camera/history repair and one slower StateDP contrast (2026-11-06)

Latest user explicitly authorizes camera content repair, measured GPU parity,
existing-model diagnosis/recheck, true slow RL expert pilots, one bounded expansion
and ONE independently random-initialized StateDP. Historical no-new-run clauses
below apply to their completed roots, not this new authorization.
Read STATE_DP_RESULTS/CONTRACTS/SESSION_HANDOFF and actual machine status.
Production stopped12:35 after50 captures:49 accepted successes, one native fall
incorrectly classified as fixed-view failure after the falling object left frame.
All81 failed-capture RGB-D frames and numeric contracts pass; failure stays out of
training. Reviewed classification fix preserves successful-data thresholds and old
errors/counter snapshots. continue-reviewed permits ONLY the frozen26 unexecuted
jobs in reviewed_collection_completion; no failed-attempt replay, no gate rerun or
budget reset. Check actual status/service narrow-repair-20261106-01-completion;
this review receipt is not evidence that the continuation has started/completed.
Root narrow_repair_20261106_01 has a fixed10h window including all waits;
3h correctness gate,6h collection boundary, training<=3h and final1h reserved.
80capture attempts,32 new families,40 student starts,<=10000 updates,50GiB output,
60GiB disk reserve; shared_recent_v1 target40%,6GiB VRAM/12GiB RAM. No budget reset,
automatic retry, second expansion, model redesign, Image or tactile training.
Same H24/commit12/29D/5Hz/60Hz/120Hz/loss/history. Preserve400+200, best/last,
expert/assets, old blocked follow-up and failure evidence. New model is fresh,
not old200 fine-tuning. Old200 State-valid only, Multi-view-valid0. New captures
require actual all-frame RGB-D content acceptance and exact numeric contracts.
New32 families pre-split24/4/4; original test8 never training. Final12 independent
seeds frozen at prepare, neither test outcomes nor new student success select data.

GPU audit reproduced CuDNN TF32 default discrepancy; unified FP32 passes original
2e-5 condition gate (max1.073e-6), prefix12 target1.132e-6rad/FK0.000229mm.
Fixed camera missing configure_camera_views proved in same-initial-state A/B;
first199-frame real expert success passes all-view content and visual review.
One orchestration software failure used expert-only env for student, zero physics
steps; retained/counts toward40. That legacy cell is NOT rerun. Corrected worker
uses ordinary position-capable env for remaining7 diagnostic cells.
Persistent ordinary tools.narrow_repair_pipeline handles remaining gates,
collection, random training and final evaluation; per-stage source frozen.
Do not edit active Python, launch second owner, or infer success from authorization.
CPU publication updates existing MD/generated small evidence and pushes only
origin feature/narrow-state-data when HEAD/docs/worktree checks pass. No raw uploads.
Short startup supervision, then yield without long LLM polling. Later invocation
reads actual results. Completed prior sections below are historical evidence.

# Completed — four State200 failure viewing reruns (2026-11-06)

Completed10:06:05 UTC+8: four legal viewing reruns, all native timeouts,
601/824/601/601 steps; only train_family_004 reached first subgoal. All four
original and four wrist-cropped videos fully decoded; owner/worker/monitor exited.
Original observable starts match exactly, first plans differ up to3.67e-4rad and
physical trajectories diverge: NOT exact historical replay. Formal counts unchanged.
Read evidence/narrow_failure_videos.json. Whole-card mean17.177%, max complete60s
27.013%, missing0; protected manifests/model/result/budget hashes verified.
Camera issue found: both fixed RGB views constant238 in all200 demonstration first
frames (wrist nonconstant). Preserve data; ImageDP requires camera validity repair
and acceptance. StateDP does not read RGB. No new starts/retries/training authorized
by this completed viewing task. Existing stopped follow-up remains blocked.


User authorizes reviewed code/records push to origin feature/narrow-state-data,
including the existing new-data generator, and several failure videos. Use
`tools.narrow_videos`: four fixed original cases (train_family_000/004,
unseen_00/04), best update5735, H24/commit12 unchanged, no expert takeover.
Independent root narrow_failure_videos_20261106_01: one hour including waits,
one Kit worker, four starts, no automatic retry, 4GiB output / 60GiB disk reserve.
Keep shared_recent_v1 target40%, 6GiB VRAM/12GiB RAM admission and full telemetry.
These are viewing reruns, not exact old playback or new formal tests. Fully decode
all delivered videos and compare original observable starts/plans/outcomes.
Do not restart the blocked follow-up, train, collect, or reset either old budget.
Raw videos/data/weights stay local; publish code, concise records and lightweight
video hashes/results only. Read latest SESSION_HANDOFF/STATE_DP_RESULTS.

# Current task — bounded conditional StateDP follow-up (2026-11-06)

## Actual status — 2026-11-06 review

Predecessor completed5735 updates: train0/4, unseen0/10 legal plus2 pre-action rejections; all14 legal trials timeout. Follow-up started02:13:16 UTC+8 and stopped at its first GPU history-condition parity assertion (<2e-5); actual GPU maximum was not logged. All owned processes exited, new updates/captures/student starts0. CPU-only recheck passes at1.91e-6 FP32 /4e-15 FP64, which does not replace GPU acceptance or establish root cause. Preserve failure, threshold and fixed follow-up deadline10:13:16; no automatic retry or deadline reset. Read latest SESSION_HANDOFF and evidence/narrow_followup.json. Earlier startup snapshots are historical.


Latest user authorizes ONE bounded continuation when supported by validation and
closed-loop evidence, then at most ONE expansion from200 to260–296 total real
successes and ONE independently initialized StateDP comparison. No architecture,
H24/commit12,29D position, loss/weights/history/control/physics change. Original400
and new200, original checkpoints, source and original13:57:39 deadline are immutable.

The predecessor narrow_state_20261105_02 must complete normally and release all
workers/monitor/owner lock before new GPU work. Isolated source worktree is
bc+rl_RSS_narrow_followup (detached reviewed commits published only to origin's
feature/narrow-state-data); never edit the still-active narrow_state worktree.
New root narrow_state_followup_20261106_01 has its own fixed8h execution window,
including all stage/resource waits; dependency wait ends at original deadline+15min.
One worker start per stage, no automatic retry or second owner/recovery. At most
96 new capture attempts,52 student starts,45GiB added files,60GiB free disk,
6GiB VRAM/12GiB RAM admission and existing shared_recent_v1 target40%.

Read SESSION_HANDOFF, CONTRACTS and STATE_DP_RESULTS. tools.narrow_followup audits
data/cache/statistics and recorded deployed control/history before decisions.
If extending, restore original last model/AdamW moments/EMA/RNG/anchor decks and
continue the ORIGINAL schedule at its existing floor, no rewarm or reset. New
expanded training has a separate seed and fresh states; only data changes.
New32 families selected by geometry-only coverage relative to original TRAIN
families, within same domain; freeze24/4/4 split before expert outcomes. Old test8
families remain test. New16 final unseen seeds are frozen at prepare, tested once,
never fed back to scheduling; original unseen12 is explicitly development evidence.

Ordinary scripts handle waiting, stage transitions, safety stops and final lightweight
publication to the same origin feature branch. Publication refuses intervening HEAD,
unreviewed worktree/staged changes or edits outside generated report blocks. No raw
data/checkpoints/logs pushed. Short startup verification then yield; no LLM polling.
Authorization/queued stages are NOT completion or success. Machine status is primary.

## Predecessor authorization and startup snapshot (historical)

The latest user explicitly authorizes fixing initialization admission, a NEW bounded
run, direct narrow initial sampling, frozen-expert continuous-speed capture, then
AUTOMATIC random-initialized StateDP training and bounded unseen closed-loop evaluation.
Work only in feature/narrow-state-data. Original400/expert/assets/native control,
physics/success/120Hz/60Hz remain protected. Old quarantined narrow_state_20261105_01
is immutable and never resumed; old State evaluation remains paused.

Read SESSION_HANDOFF, SLOW_EXPERT_RESULTS, STATE_DP_RESULTS and CONTRACTS. The installed
CPU scene-query overlap gives contradictory articulated-shape hits after rollouts despite
identical live link poses. New admission uses actual collision-mesh conservative OBBs
at live link poses, with1mm envelope, plus original static box checks; no physics step,
SDK patch or per-step state forcing. Native reset parameters directly define new families.

New pipeline:60 target fresh families (minimum50), split before speed outcomes; about220
bounded planned speed attempts, target200/minimum150 successful full sensor trajectories.
Only new train families fit statistics. H24/commit12/29D target, model, Loss B and fixed
auxiliary weights,60Hz continuous history and native position execution remain unchanged.
Fresh random policy/optimizer/EMA/scheduler; no old DP weights. Final train4 diagnostic
and12 preregistered unseen starts are separate; no outcome-based seed replacement.

An ordinary owner transitions family/data validation -> disposable GPU update -> formal
training -> final evaluation. New run fixed16h including waits: collection6h, training up
to9h while reserving1h for evaluation. Each stage at most2 reviewed starts, no automatic
retry; no old deadline reset.6GiB VRAM/12GiB RAM admission, shared_recent_v1 target40%,
60GiB disk reserve and75GiB run budget. Protect unrelated processes.

Verify multiple actual episode/save/reset transitions once, then yield without LLM polling.
Startup is not DATA_READY, DATA_READY is not trained, trained is not task success. Read
machine JSON for actual state. Preserve partial/quarantined records and error denominators.
Commit/push reviewed code/docs/small reports to origin feature only; no raw artifacts,
weights/videos, unrelated worktree changes, upstream or force push.

## Verified startup

Production code b119470 is running in native single-lane persistent Kit under
narrow-state-data-20261105-02-production.service, root narrow_state_20261105_02.
60/68 direct candidates,44/8/8;4/4 pilots plus2/2 first production episodes passed.
One disposable CUDA update/save/restore passed; DATA_READY=false and formal updates0
at startup. Read actual machine status later; do not infer final success from this.
Within the unchanged16h absolute window, stage allocation was reviewed at the completed
pilot boundary: collection6h, training at most9h, final1h evaluation. Absolute deadline
2026-11-06 13:57:39 UTC+8. Old v1 quarantine remains. Do not restart this live service
or spend model calls polling it; ordinary owner handles transitions and bounded stops.
