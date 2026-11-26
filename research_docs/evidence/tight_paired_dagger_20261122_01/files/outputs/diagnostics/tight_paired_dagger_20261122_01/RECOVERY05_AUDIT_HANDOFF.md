# Recovery05 integrity and noise audit handoff

`audit_collection_recovery05.py` replaces the old all48-pairs audit entrypoint only for this interrupted96 ledger. It does not patch or loosen production validation. The full frozen scientific plan, production source SHA, collection parent SHA, published aggregate SHA and physical `evidence_directory` are checked first. Exactly15 original consumed assignments and81 originally unassigned suffix assignments must retain original global order; native batch queue ordinals use each attempt's local ordinal.

The wrapper validates all47 actual passing paired checks, both initialization legality flags, pair names and attempts, and every finite nonnegative numeric difference against1e-5. The actual cross-attempt missing pair stays explicitly unverified. It never creates a substitute passing pair or reports all48 verified.

Integrity audits every complete legal trajectory with the existing causal trigger replay, complete student parent binding, full numeric/compact history, actual issued/applied/target/q-reached chains and legal expert masks. Every complete compact capture gets a whole-file SHA check. Admitted suffixes additionally bind `capture_complete.json` source_verified status/steps/path/positive source-field count and the result's verified capture receipt, as well as all-frame camera acceptance. Incomplete/illegal/unknown trajectories cannot enter action supervision and stay in the96 denominator. Unknown initialization remains separate from invalid initialization and no-trigger.

Noise audits complete legal trajectories only; a complete trajectory with a missing/mismatched timing or endpoint receipt fails rather than silently skipping. The full native endpoint consumption includes post-takeover student shadow planning. Saved endpoints for incomplete/corrupt executions are only inventoried, never treated as complete execution certificates. Every one of48 pair rows remains in output; pairs lacking a certified endpoint get `paired_start_equal:null`. There is no observed takeover-boundary RNG state claim. The default is CPU-only; real CUDA reconstruction requires the explicit flag and root's quiet GPU resource window.

After the continuation finishes and the aggregate indexes are published, root can execute:

```
PY AUDIT/audit_collection_recovery05.py integrity --collection ROOT/collection --output AUDIT/collection_integrity_actual01_recovery05
PY AUDIT/audit_collection_recovery05.py noise --collection ROOT/collection --output AUDIT/collection_noise_cpu_recovery05
```

For final native CUDA reconstruction, use a separate new output in the explicit external-quiet GPU window:

```
PY AUDIT/audit_collection_recovery05.py noise --collection ROOT/collection --output AUDIT/collection_noise_cuda_recovery05 --reconstruct-cuda-noise
```

Here PY is the existing isaaclab Python, ROOT is the current paired run, and AUDIT is this directory. These commands were not run on actual collection payloads by this implementation task. Whole-file capture verification reads potentially many GiB: root must schedule it with the existing low-priority resource/coexistence policy and should not describe CPU-only as zero I/O impact.

Final report assembler wording now includes the actual47/1 paired-initial check counts, per-role incomplete/invalid/unknown counts, and no-retry15+81 accounting. It does not relabel technical failures no-trigger or report incomplete endpoint consumption as certified. No previous scientific gates were removed.

Validation:51 CPU fixtures pass across the recovery audit, independent review and assembler; original noise's16 pure fixtures pass in a separate process. Running that older fixture after the integrity fixture in one process fails its `torch not in sys.modules` assertion because integrity intentionally imports CPU torch; this is a test isolation requirement, not CUDA use. The independent review's tolerance expected failures are now fixed and promoted to normal passing tests. Temporary fixture paths only; no Kit, GPU, training, real capture audit, frozen production or configuration changes.

Current source SHA and exact test results are in `recovery05_audit_ready.json`. There was a concurrent audit/test API replacement during development; the current test file has been reconciled to the single current implementation and both suites now pass. Root owns all activation.
