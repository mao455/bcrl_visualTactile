# 2026-11-16 supervised grasp diagnosis and night continuation

This bundle implements the latest user request. It is **not** evidence that two
real cycles or unattended readiness have passed. Actual service receipts in the
output roots are authoritative. All raw observations, expert labels and current
rollback pairs stay on the research machine; only scripts and small reports go
on the feature branch.

1. `request_dp_boundary.py` waits for session13 A/B to finish exactly12 epochs /
   2460 optimizer updates, then requests its ordinary cycle STOP before backend
   preparation. It never changes old02:05:20 deadline or interrupts a later stage.
   Started as `tight-fresh-dp-boundary-20261116.service` under supervision.
   `wait_finish26.py` persistently waits for this receipt AND both old owner/monitor exits, then launches step2 once; no new GPU work overlaps the old session.
2. After both old owner and monitor have exited, from a clean frozen worktree:
   `python scripts/tight_fresh_night_20261116/finish26.py launch --expected-commit <HEAD>`.
   New root `tight_fresh_20261116_00` ends09:00 +08. It binds the completed DP
   weights/data/cache read-only, performs **zero repeated completed DP updates**, refreshes
   plans/current trainable vision, trains rear12/latest4, and runs fixed grasp8
   per role with the existing full-task subset. Original train2 full-policy
   grasp probes, current-pair offline errors and hard/new-development2 each
   diagnose why grasp remains unreliable. Original session13 evidence is intact.
   If the old temporary deadline expires first, `finish26.py` can instead resume
   an incomplete DP from its last complete epoch in the **original frozen trainer**.
   Model, AdamW, EMA, RNG, decks and total2460-update budget stay exact. Completed
   B is reused; old uncommitted partial-epoch updates are disclosed and excluded.
   Append-only logs are copied into the new identity, never linked back to old
   evidence. This fallback is prepared, not evidence it has run successfully.
3. The agent reviews these results and selects each complete source pair by
   actual grasp behavior. `launch.py --expected-commit <HEAD> --role-source A=<root>
   --role-source B=<root>` then creates `tight_fresh_20261116_01`, bounded by09:00.
   Before collection, the owner measures retained A11/B09 under the current GPU queue on the same fixed development seeds and compares the finished15 candidate. Historical CPU rates alone cannot select a GPU candidate. Each small real loop has4 genuine student-prefix opportunities on two hard
   registered training families, contact-formation triggers,4 DP epochs and4 rear
   epochs with latest4 for both. Eight fixed autonomous executions include the
   normal full-task subset; four short rechecks of the just-corrected training
   cases share that same queue. Development and train-family recheck metrics stay
   separate. No stage-success claim follows from loss.
4. At least two consecutive real loops must be personally observed. The owner
   cannot write its own `night/supervised_release.json`; the agent may create it
   only after inspecting complete capture/admission/training/checkpoint/autonomy
   and transition evidence. Required keys: `configuration_sha256`,
   `reviewed_rounds` (first two actual indices),
   `two_real_complete_cycles_observed: true`. Two rounds without a clear hold
   gain safely stop for a new diagnosis instead of repeating blindly.

The durable systemd owner has finite runtime/memory bounds, one simultaneous
simulation stage,10s stage heartbeat,900s progress timeout (collector additionally
has its existing180s camera-progress watchdog), atomic receipts and one transient
training-resource retry. `tight_fresh_cycle --resume` requires identical frozen
source/budget; skips only bound completed stages and resumes saved optimizer state.
Interrupted physical cases are never overwritten or silently restarted. STOP at
the session root requests owned cleanup. Failure state gives the exact stage log;
there is no unlimited retry. These mechanisms still require real-cycle validation.

`retire_last.py` removed9 named closed final optimizer checkpoints,9.753GiB,
retaining every selected best at that point. `archive_optimizer.py` then archived
17 obsolete/intermediate selected checkpoints without their Adam states, saving
9.220GiB: **both model and EMA tensors were read back and checked exactly**; all
other checkpoint fields remain. Their new `best.weights_only.pt` files are
inference/diagnostic archives, not optimizer-resume checkpoints; sidecars retain
original SHA and archive SHA. A11/B09,05/06, B13,14 and current15 are excluded.
Source captures, frozen visual payloads and all success/failure traces are retained.

The first supervised cycle has one declared post-DP coordinator recovery exercise. Both actual DP completions precede it; ordinary `--resume` must skip their bound receipts and continue rear preparation. The attempt is explicitly distinguished from software/resource/data failure. It consumes the one allowed retry for that cycle. Actual cycle acceptance is required before claiming recovery reliability.
