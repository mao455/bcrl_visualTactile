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

CPU diagnostic after00: `python -m tools.tight_fresh_expert_counterfactual --root <00>/round_15 --validation-numeric <09>/new_family_15/r15_A_expanded_session20261115_09_early15_s29000120_grasp_early/numeric.h5 --output <00>/round_15/expert_counterfactual.json`. This first validates150 recorded real shadow ticks, then queries four saved autonomous histories without simulation. The suggestions are never admitted into the training pool. Original report paths are immutable; use a new explicit output for a deliberate new diagnostic.

`retire_closed_matches.py` retires only four closed Nov12 validation-best containers after complete-payload equivalence and original hashes bind them to retained bests. All model, Adam and RNG state remains in those bests; exact original container hashes and equivalence receipts remain under01/closed_checkpoint_dedup. Actual saving4.3397GiB.

The optional `--decompose-student` CPU expert diagnosis additionally reconstructs recorded final projection, then compares projected nominal and full policy at the same stored histories. A remains a different label domain; these suggestions are not new training labels or autonomous bypass trials.

`recover_prepared16.py` is the explicit supervised recovery for the prepared-only evaluation guard defect. It runs only the bound unstarted evaluation with original workers, then reuses complete receipts; it does not repeat training. Future coordinators now permit exactly plan.json/session.json and refuse any startup/case artifact. `tools.tight_fresh_cycle_readout` reuses completed receipts and one numeric read to summarize actual selected DP/rear phase exposure and autonomous outcomes without scanning images.

After round16's prepared-only guard and camera-refill failures, the separate repaired evaluation completed, but neither candidate gained development holding. `launch_after_camera_repair.py --expected-commit <HEAD>` creates02 from its cumulative data and retained A11/B15, applies the explicit early grasp profiles, and requires new clean rounds17/18. Both old failures remain. Same09:00 deadline; no unattended release from the repaired evaluation alone. `training_sampling_grasp.json` changes eligible planning-anchor distribution only; `backend_grasp_weighting.json` increases early legal rear loss weight, including original expert approach, rather than repeating the failed first12-onset control.

One operational status command: `python -m tools.tight_fresh_night_status --root <02>`. It reads existing small receipts, does no hashing/training/simulation and never releases the barrier. Closed Nov13 adaptive exact-container deduplication is reproduced by `retire_closed_adaptive_matches.py`; initial/round1v3 Adam-only archives use `archive_optimizer.py --closed-initial-fresh`. Original hashes and retained payload equality are local under02. These storage operations preserve all raw observations and current rollback states.

`tools.tight_fresh_fixed_rear_fit` is a prepared bounded train2 diagnosis if supervised cycles show no holding gain: frozen actual DP, real updated Transformer, original full trajectories, no held-out fitting. Its output is diagnostic and does not replace a collection source without actual autonomous evidence. No run is implied by this script's presence. `archive_optimizer.py --closed-pre-fresh` applies only to six named closed legacy checkpoints; every nonoptimizer field and both parameter sets are verified exactly.


The bounded original-train2 fit completed in02/fixed_fit (A/B80passes,560updates). It is diagnostic only. `launch_fixed_fit_comparison.py --expected-commit <frozen SHA>` may be run after02 has safely ended with17/18 complete: it creates03, reuses fits, runs4source+4fitted full-policy300-control-tick evaluations serially, and keeps09:00 absolute deadline. No additional fitting is performed. Do not treat the prepared launcher as an executed comparison.


`retire_closed_night_duplicates.py` is eligible only after02 safe shutdown and completed17/18. It compares six specific DP final/selected snapshots, preserves every distinct state, and may remove only complete-payload-equal last serializations with retained selected hashes and no open FD/mapping. The local inventory/finished receipts, not this prepared script, establish actual reclaimed bytes. Future night configurations can set latest_repeat and backend_latest_repeat independently; both default4, no active configuration is mutated.


### 2026-11-16 06:30 +08 — two real loops passed, bounded fitting diagnosis, new pre-contact cycle prepared

17 and18 finished cleanly without manual stage recovery;18 began automatically after17. All owned02/03 simulation/training workers and monitors have now exited. TWO consecutive actual capture/admission/DP/new-DP-input/rear/autonomy loops were supervised.02 stopped safely for two rounds without clear development hold gain, not a software failure. No unattended release has been written. Monitor5716 samples, no missing, minimum free VRAM4286MiB and RAM18.99GiB; fullGPU permitted. This establishes these actual loops, not immunity to future faults.

18 admitted4/4 rescues on existing81/94,1118 legal expert controls. Cumulative344train/170registeredtrainfamilies plus6validation; trajectories are not independent initial families. DP both4epochs/712updates, selected A3/534,B4/712. Rear both4/2641updates, selected A3/1988,B4/2641; new4 each12/16 visits in selected A/B networks. Selected latest early hand command RMSE .010166/.009983rad. Early rear loss mass44.73%, raw~21.24%. Development hold60 A0/3,B1/4 versus retained-source1/3,1/4: reject both. Training rechecks A1/2,B0/2; A81 held and achieved one native subgoal. Four full-task insertion0/4, seven other legal trials grasp-censored, one pre-action illegal. Historical A81/A94 initial q differs .02942/.02040rad, so old-to-new family observations are NOT matched-state learning proof. Stage union seconds: capture169.16, frozen cache133.05, DP624.20, rear inputs828.17, rear training593.98, evaluation170.06; entire42.56min. Collection85.13valid/h remains far below600/h target.

03 actual fixed-original-train2 comparison reused the finished80pass/560update rear fits with unchanged A11/B15 DP. Four source then four fitted300tick full-policy trials completed serially (60.73/82.40worker seconds). Every pair's initial observable q/object position matches exactly, not hidden PhysX state. Source A0/2,B0/2 hold60; fitted A1/2,B0/2. Original formal029 A now holds73ticks after contact71/lift110 versus source contact88/no sustained lift; formal050 A remains unstable. B lost prior brief lifts despite lower saved-history error. All8 short trials are censored for insertion, no subgoal/alignment. This is one real fixed-training-case A improvement, not general grasp stability or adoption of the diagnostic weights.

Counterfactual actual-state expert queries passed the existing150tick mapping validation, without simulation/updates/admission. Fitted B's100:150 hand deviation is .03181/.02315rad versus source .01337/.00966; better offline fit did not preserve favorable closed-loop states. A029 holds despite an earlier command-deviation threshold crossing: do not make command mismatch alone a failure trigger. Evidence supports genuine pre-contact coverage plus better fit; no single exclusive root cause is established. The previous B17 early-validation check favors the same epoch3 as global validation, so selection-metric replacement is unsupported here.

Storage: attempted closed-last dedup stopped before deletion because best/last were already hardlinks. Preserved failure; no duplicated-byte saving claimed. Explicit archive_closed_grasp_artifacts.py then finished:8 obsolete Adam payloads retired, every model/EMA and other field exact; six unused learned-plan caches losslessly zipped, every original-byte SHA verified, exact restore manifests retained. Current A11/B09/B15/05/06 and selected18 SHA unchanged, raw histories/frozen DINO untouched. Actual available disk22.123→27.695GiB (+5.573GiB); old16/17 optimizer resume intentionally unavailable, their weights retained with explicit archive identities. No active file removed.

Prepared launch_precontact19.py creates NEW04 identity with retained A11/B15, cumulative18 data, real v3 approach opportunities on hard76/86,8epochs and latest12 for BOTH DP and rear. The6-earlier-of8 retrospective result remains zero new rescues until live collection. Combined timing/exposure change is an evidence-driven recovery experiment, not component causality. One finite cycle only, same09:00 deadline and16GiB runtime reserve;27.5GiB admission accounts for measured recent~9GiB output plus~2GiB checkpoint temporary. Need actual startup/collection evidence before claiming it runs.
