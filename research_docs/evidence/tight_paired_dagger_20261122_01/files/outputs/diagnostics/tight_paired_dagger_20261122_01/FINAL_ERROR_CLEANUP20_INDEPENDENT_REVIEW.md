# Final error cleanup20 independent source review

Status: source review passed, inactive only. Reviewed source SHA `a1d7d87325160028e9e3a1176ed9bea96ce7e627e7dc811f59554f4b1dc3a658` and fixture SHA `aeb51f5360a583aea0f2834277b72ecc86dbe5a72be3b9cd513806b76e4993ff`. Independent execution passed all 21 CPU fixtures; every signal syscall was mocked. No runtime activation, real signal, process mutation, or main evaluation outcome was read.

The original `bc2c7a...ad35e` version could send TERM despite a helper-local STOP already present. A mocked counterexample confirmed one signal call. The revision checks STOP, unchanged root registration and source binding before each signal, again after durable intent I/O; the same real-guard fixture now makes zero calls. Per-signal intents precede syscalls and the registration permits only the exclusive fixed actual01 receipt directory.

The finite original59..63 scope, 120-second monotonic unchanged actual log/progress check, complete terminal RAM-reserve traceback requirement, frozen scientific/source identities, exact owner/driver/worker ancestry and group census, schedule lock, TERM-first order and fresh-verified 15-second KILL grace pass source review. Exceptions, progress, identity changes and STOPs fail conservatively; the helper does not retry a consumed wave, change the queue, fabricate owner exits, or interpret outcomes.

A durable intent proves only intent. Signal delivery and filesystem evidence cannot be atomic; missing post-signal evidence cannot prove no signal. The repeated process census and killpg also have a residual kernel-level race. A TERM-induced log change can refuse further KILL, and an exited/zombie leader does not certify descendant closure. Native-owner reaping plus a final exact helper/worker/descendant census remain necessary. Evidence I/O failure must be retained and reconciled from actual owner/process records, not converted to success.

This review does not activate the aid. Separate concrete root registration and actual execution/closure review remain required if it is used.
