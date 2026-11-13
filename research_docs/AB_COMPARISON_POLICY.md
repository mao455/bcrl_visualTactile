# Tight-Insertion A/B comparison and progress publication policy

User direction recorded 2026-11-13: publish verified new information to GitHub automatically and give both groups equal comparison opportunities. This applies to the current session through its existing absolute deadline, 2026-11-14 00:05:15 Asia/Shanghai; it does not extend training or evaluation budgets.

## What must match

A scientific A/B comparison requires the same immutable data snapshot and split identities; legal supervision masks; sampling order, weights and actual per-stratum exposure; incremental DP and Transformer update budgets; common loss, output and history fixes; optimizer treatment, schedules and model-selection opportunities; and evaluation identities, seeds, reset specifications, natural termination and native success criteria. Register these before observing outcomes. Check actual completion and coverage receipts afterward: equal configured epochs alone are insufficient.

Use complete frozen DP + Transformer pairs. A changed DP needs its corresponding refreshed backend cache, generated with the deployment planning clock, remaining old plan and causal history. A cache from another DP or a per-tick newly generated nominal plan cannot substitute for the deployed commit12 policy. Preserve all original data and historical evidence, test22 isolation, legal correction semantics, physics and execution constraints.

Do not add trials to only the promising group, replace difficult seeds, grant extra selection attempts, or silently count software failures as completed task failures. If a recoverable failure leaves unmatched opportunities, restore the registered counterpart within the existing deadline or report the affected comparison as incomplete. Additional diagnosis may be useful, but must remain separately labeled.

## Differences the research intentionally retains

A learns the expert's actually reached target (`q_reached`) with the established geometry. B learns the expert command (`u_expert`). Both Transformers learn the final expert command. Never exchange these labels to create superficial equality. Target-specific frontend statistics can differ; shared rear command statistics and common processing must agree. Different frontend statistics are not evidence of unequal rear command normalization.

Historical model initializations and earlier exposure differ. A matched continuation establishes equality of the registered incremental protocol, not equality of all prior training or a strictly controlled architecture comparison. Disclose this limitation beside conclusions. Every new nonessential group-specific intervention must either be applied comparably or explicitly qualify the resulting comparison.

## Current evidence and its limits

The completed `adaptive_contact_recovery_01/round_01` records equal actual incremental exposure: each group has 514 DP updates and 765,303 supervised DP target slots; each Transformer has 1,362 updates and 193,581 supervised control ticks. Each received the same 10 autonomous development evaluation opportunities. A succeeded 0/10 and B 1/10. The 21 previously omitted onset windows, containing 210 legal correction ticks, were included in both groups. These are observed records, not proof that every historical factor was controlled or that B has a stable advantage.

The running `success_replication_01` compares the current and previous **B** policy pairs on new development seeds. It is a B-version replication, not an A/B cohort. Do not combine its extra B trials with A's older denominator. If those seeds are later used to compare A and B, the corresponding frozen A pair must receive the same registered opportunities at a safe boundary and within the unchanged budget. Otherwise report B-only findings separately.

## Review gates and publication

`tools/tight_ab_fairness.py` inspects the existing round layout. Prepared-mode checks compare declared contracts, data and evaluation plans. Completed-mode checks additionally compare actual finished updates, supervision totals, coverage ledgers and completed evaluation opportunities. Missing or unequal required evidence fails the check. A passing receipt concerns its documented evidence scope; it does not establish bitwise simulator parity, matching historical exposure or causal superiority.

The next scientific Agent invocation must review this policy and the gate receipt before making an A/B claim. Existing frozen workers are not modified or restarted to introduce this check. The background publisher reports evidence; it does not decide new experiments or perform scientific reasoning.

Automatic publication is restricted to `research_docs/TIGHT_LIVE_STATUS.md` and `research_docs/evidence/tight_execution_learning_20261113/live_status.json` on `origin/feature/tight-state-foundation`. It reports completed and running stages separately, preserves the B-only distinction, and publishes no raw trajectories, checkpoints, videos or private configuration. Git operations serialize on the shared `tight-publication.lock`, use ordinary non-force pushes and retain error/pending receipts on failures. Reviewed source and research decisions still require the scientific Agent's normal review. Publication ends at the original absolute deadline.
