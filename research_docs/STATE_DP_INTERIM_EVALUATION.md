# StateDP 100-anchor-epoch interim closed-loop evaluation

2026-11-05 user requests a new evaluation before the continuation finishes.
This is an additional bounded diagnostic cohort; original StateDP/ImageDP10000
formal results remain 0/12 each. No new Image checkpoint is available.

Run: `outputs/visuotactile/native_dp_v2/anchor_interim_20261105_01`.
Read its `status.json`, `summary.json`, and `EVALUATION_SUMMARY.md` for actual progress.
Preparation or service launch is not a completed rollout.

The current State worker finishes its already-running 100-anchor-epoch stage
(cumulative16584 optimizer updates, original10000 plus6584). Only its verified
owner PID is temporarily suspended; the trainer and resource monitor continue.
After the child exits and releases GPU resources, its complete milestone checkpoint
is hard-linked and SHA-verified in the evaluation root. No optimizer update is
interrupted, repeated or reset. The original 72-hour deadline is unchanged.

Four fixed validation initials, original ordinals4–7 / seeds984101–984104,
run independently with the same physical/control settings as the original formal
evaluation. Reuse the original10000 results on these exact saved observable and
event-RNG exports. Do not claim identical unobservable simulator internals.
H24/commit12, absolute29D position, original controller/delay, EMA/DDIM16/eta0,
clip=False and sampler seed7091 are unchanged. No smoothing, teacher forcing,
expert takeover, test40, architecture or loss change.

Four planned episodes, maximum5 Kit starts including at most one startup recovery;
two consecutive same-stage failures stop. Deadline4h including at most1h waiting
for the training milestone. Each valid result requires finite traces, submitted
and actuator target parity, exact recorded initial exports, full video decode,
and worker/GPU resource release. SDK fast exit and graceful teardown are distinct.
Contact uses the existing `contact.object_contact` force diagnostic. Native lift,
subgoal, fall, dropped and timeout are reported separately. Native stable-grasp
success has no discrete definition and remains N/A.

Independent persistent service owns evaluation. Both queues share a GPU lease;
the training owner cannot launch its next worker during evaluation. Normal/failure
cleanup and systemd ExecStopPost resume only the recorded PID/start-time identity.
Other project processes are never signalled. Parent control state is recorded in
`evaluation_interlude.json`; original status/results are not rewritten as success.

Implementation: `tools/anchor_interim_eval.py`; three focused CPU tests passed
for contact/timeout counting, PID-reuse protection and idempotent recovery.
Existing training/runtime sources are unchanged. Raw weights, traces and videos
remain local. Small reviewed results will be published after actual evaluation.
