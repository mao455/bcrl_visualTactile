# Current task — bounded conditional StateDP follow-up (2026-11-06)

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
