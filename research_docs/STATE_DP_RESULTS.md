# StateDP 150 anchor-equivalent epoch assessment and cleanup

Status: the existing continuation is still training toward update 20918 (150 anchor-equivalent epochs). No 150-epoch result is claimed yet. The original 10000 and 16584 checkpoints remain protected until the common-protocol comparison is complete.

The existing `tools.anchor_interim_eval` entry now accepts the milestone and frozen ordinals as configuration. It pauses only the verified queue owner while its current trainer finishes normally, evaluates the resulting full checkpoint, and holds further training for evidence review. Model, loss, sampling, H24/commit12, absolute 29D targets, controller and data split are unchanged. No new trainer or reset is introduced.

The cohort is the previous validation4 (ordinals 4–7, seeds 984101–984104), followed by train4 (ordinals 0–3, seeds 982401–982404). Train results are fitting diagnostics, never pooled into validation or test success rates. Eight valid episodes are planned, with at most nine starts and one replacement; startup/shutdown failures remain separate. Stable grasp remains N/A because the environment has no native discrete criterion. No test40 is used.

Current evidence: at 16584, validation4 had 0/4 full success, 4/4 contact diagnostics, no native lift or subgoal, all 601-tick timeouts. First sustained 5 mm end-effector deviation from paired experts occurred at ticks 3, 5, 7, 5, before expert contact and before the first replan. This does not establish boundary discontinuity as the first cause. See the [interim results](STATE_DP_INTERIM_RESULTS.md) until consolidation is complete.

The requested cleanup starts with import, entrypoint and artifact dependency inventories. The 400 original successes, 320/40/40 split, current numeric cache, old-directory image caches still used by current training, train-only statistics, active process files and complete recovery checkpoints are protected. Obsolete weights and duplicated runtime caches will be removed only from a recorded, revalidated deletion plan. Active frozen source changes wait for a complete training boundary. Current and unknown dependencies are not deleted on age alone.

Validation so far: four owner-isolation tests pass, including PID reuse protection, idempotent release, native contact counting, and the new explicit evidence-review hold. Full post-cleanup regression and 150-epoch offline/closed-loop outcomes are pending.
