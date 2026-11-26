# Capacity18 independent regression review

Scope: inactive source and synthetic CPU tests only. No helper activation, actual process inspection or mutation, scientific outcome reads, model/cache payload reads, signals, or evaluation retries.

Reviewed source SHA-256: `1ffcc8ec77fdee626291612d8d579fb9780066c862a679e80b591d71b80dd541` (`host_init_capacity18.py`). Independent test SHA-256: `1a0ae247cb94df9bb253dca8158bc3abffd2dcfe7dcb4ea03fa1a13b3864f9c9` (`test_independent_host_init_capacity18_review.py`).

Validation: `/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python -m pytest -q outputs/diagnostics/tight_paired_dagger_20261122_01/test_independent_host_init_capacity18_review.py` — **25 passed in 0.08 s**.

The independent fixtures cover the complete synthetic owner/worker/launch/dispatch/plan identity graph, worker and owner PID reuse, dispatch SHA/PID/attempt/rerun changes, Tgid and task starttick mismatch, readiness and capacity races with zero affinity writes, partial expansion failure, restoration while holding the lifecycle lock, a thread born after expansion, source/self pin drift, invalid birth timestamps, per-thread cap checks and truthful elapsed time, and external process birth reuse.

The first draft had a failclosed defect: an evidence write or an identity-inspection error during failed restoration could release the lock before the worker exited. The reviewed revision retries the cooperative global STOP and evidence while retaining the lock, and treats identity uncertainty as a live target. Independent injected evidence I/O and AccessDenied regressions pass. Original `owner/startup_stops` receipts remain outside the aid's write paths.

No remaining blocker was found in this reviewed source snapshot. Activation and actual run/closure evidence require the root agent's separate exact registration/review. Synthetic tests do not prove a constructor speedup, absence of contention, or exact physical equivalence. Linux affinity calls target integer TIDs; token checks narrow but cannot atomically eliminate TID-reuse races. SIGKILL or host loss can bypass Python finally/flock restoration. The 40-second placement cap is checked between calls; actual elapsed time includes restoration and can exceed the nominal cap, which is explicitly recorded.
