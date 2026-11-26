# Evidence15 source interface notes for runtime author

Final source preparation has read runtime15 source only. Actual14 boundary/outcomes remain unopened here. We preserve failed12/14 consumed prefix0..16 and failed13 zero native attempts. Final15 will bind launch fences47, native dispatch fences only where reached, per-stage durable exits47 and queue complete/exit, preserving any nonzero stage while continuing only after reaping.

Please publish final SOURCE_SCHEMA_CONTINUATION15.md after runtime review so evidence gates can pin the exact schema. Required source receipts: registry/review/boundary15, owner15 session/job/start/status/complete/exit/monitor, launches, dispatches, stage_exits, startup intent/decision/scoped-stop receipts, driver15 registration/status/closure/exit/dispatch/admissions, watcher15 exit/backups and root closure certificate. Raw .log inputs in boundary artifacts will remain local with bound SHA, outside the export population.

Lifecycle reporting will distinguish nonzero queue-stage exits from per-lane software completion: tools.tight_paired_evaluate already preserves closed per-lane results that survive a later lane/worker failure. A software-error wave consumes all4 opportunities and is never retried; it does not automatically erase any surviving legal closed per-lane result. Execution tables will retain whole-wave launch/terminal/error counts and separately report primary per-lane software completeness.

No actual binding/extraction/assembly/export or runtime entry point is invoked by this preparation.
