# Host initialization capacity19 — inactive clock-interface correction

This is a new source, registry, review and actual-audit identity. Capacity18 source, registration, attempted process and all failure evidence remain unchanged. Root's actual18 failure review found that raw psutil create_time derives from integer `/proc/stat` btime while immutable launch/status/dispatch receipts use realtime; the fractional difference incorrectly rejected a genuine owned worker before mutation. Capacity19 changes only that clock interface and its evidence. It retains capacity18's finite wave43–63 range, capacity samples, at most2 spare whitelist CPUs1/2/4/5, unchanged nice10, per-thread identity, lock/restoration/failure handling,40s trigger budget, source pins, once/wave and no-external-mutation policy.

Raw PID/create_time remains the exact identity used by same(), owner registration, ancestry and thread ownership. Before any chronology comparison, mapped_birth validates `/proc/PID/stat` starttick, positive integer SC_CLK_TCK and `/proc/stat` integer btime against raw psutil create_time within10microseconds, and rechecks exact live raw identity before and after mapping.

The independent contemporaneous sample records `wall_before`, CLOCK_BOOTTIME and `wall_after`. It rejects nonfinite, reversed or wider-than50ms brackets. Realtime boot offset lies in `[wall_before−BOOTTIME, wall_after−BOOTTIME]`; with starttick quantization, process birth lies in `[offset_lower+starttick/HZ, offset_upper+(starttick+1)/HZ]`. Full sample, HZ/tick/raw source and mapped bounds are included for worker and owner in each identity binding. There is no arbitrary one-second slack or replacing the raw identity with a shifted birth.

Every feasible birth in the interval must fit the original18 chronology domain: a status/start/dispatch receipt requires the entire interval inside `[receipt−120s, receipt+50ms]`; a launch requires the entire interval inside `[launch−50ms, launch+120s]`. Whole-interval containment rejects intervals that merely overlap a valid window. The unchanged50ms allowance is applied after explicit mapping; it does not absorb the observed roughly0.89s raw-btime discrepancy. A mapped future birth or old/reused identity beyond these domains is rejected before affinity writes.

The contemporaneous bracket bounds the current mapping. It cannot prove that the historical realtime−BOOTTIME offset remained stable across the receipts; earlier clock steps cannot be excluded. Worker launch/status and owner receipt checks constrain inconsistencies; they do not establish complete historical clock stability. Existing SIGKILL/host-failure and integer-TID syscall limitations remain; source-only CPU tests do not certify actual activation/restoration.

Root activation artifacts are `HOST_INIT_CAPACITY19_REGISTRATION.json` and `ROOT_HOST_INIT_CAPACITY19_REVIEW.json`, with new version `host_init_capacity19_v1` and exact audit `host_init_capacity19_actual01`. The source pin set is exactly capacity19 self plus the same frozen15/fragment13 dependencies; all other source_guard fields follow18's schema. The exact already-live continuation15 project owner raw PID/create_time remains `owner_identity`. Additional root evidence may record parent18 source/failure closure and this clock policy, without altering frozen production or changing the helper's eligible original waves.

After independent source review and root's actual identity/binding review, activation CLI is:

```sh
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python -m outputs.diagnostics.tight_paired_dagger_20261122_01.host_init_capacity19 --apply
```

No helper was activated or actual process queried/mutated during this source implementation. CPU fixtures mock all process/affinity/kernel reads and clocks. Actual19 activation, every intent and restoration/exit, and independent process/service closure must be reviewed separately. Preserve18 and19 attempts even if either fails, and retain all original64×4 opportunities without retries or replacements.
