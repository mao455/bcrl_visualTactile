# Current task — direct narrow StateDP pipeline (2026-11-05)

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
