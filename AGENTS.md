# Current task — narrow StateDP data collection (2026-11-05)

The user authorizes initial-state analysis, a coherent narrowed distribution, frozen-expert
closed-loop speed augmentation, minimal real acceptance, bounded independent collection,
focused cleanup, and reviewed commits/pushes to origin feature/narrow-state-data.
Read research_docs/CONTRACTS.md, SLOW_EXPERT_RESULTS.md, SESSION_HANDOFF.md first.

This worktree is bc+rl_RSS_narrow_state, based on expert-speed commit84cbb98.
Preserve other worktrees and their changes. Old State random evaluation remains paused
at143 valid; old expert capture is blocked by a missing post-observation field. Do not
resume either. Original400, assets, expert, SDK and original control/reset/physics remain
read-only. The new initial distribution is conditional native sampling, not a physics edit.

H24 predicts24 native60Hz target ticks; execute12/replan5Hz. Labels remain29D canonical
transition/target_after_rad, distinct from actual q. Keep model, objective/weights, persistent
GRU/history and position execution mapping. Only new train families fit numeric statistics.
All velocity variants of a family share one split. No Image, DINO cache, tactile training,
formal State training or automatic large evaluation. A disposable single optimizer update
is allowed for data/gradient/checkpoint acceptance; future formal State starts from random
parameters and fresh optimizer/EMA/scheduler/counters, never old DP checkpoints.

Use one persistent Kit, frozen source inventory, bounded attempts/deadline/disk and ordinary
process monitoring. Latest shared_recent_v1 resource policy:6GiB free VRAM/12GiB available
RAM, recent-load pacing and40% target, no obsolete hard50% veto. Protect unrelated jobs.
Check actual run JSON; startup is not dataset readiness. After multiple real episodes and
State smoke acceptance, release the independent service and end model polling. No model
sleep loop. Failure stops with evidence; do not widen distribution or repeat failed tasks
to fill counts. Preserve incomplete payloads; do not delete unknown or referenced files.

Update current docs in place and commit only reviewed code/tests/light statistics. Never
push raw data, weights, credentials, videos, upstream, unrelated changes, or force push.
Historical authorizations/results remain in prior branches and RESEARCH_HISTORY.md.
