# Fragment13 independent source review — inactive preparation only

Reviewed on 2026-11-24 by the independent `source_review13` agent. The final reviewed snapshot has no remaining blocking defect in the requested paths. This is source and synthetic CPU verification; it does not certify an actual recovery12 boundary or authorize fragment13 activation.

No actual evaluation directory, result, weight, active process census, telemetry, service state or lifecycle census was read. No `prepare`, `driver`, `project`, `worker` or watcher entry point was invoked. No running12 source, process, service, schedule or artifact was changed. Only this new review document was written by this reviewer; source corrections were coordinated with the preparation owner.

## Findings and corrections verified

| Finding on the initial snapshot | Final source disposition |
| --- | --- |
| Census ignored empty directories and special entries, allowing an unexplained suffix directory to be described as never launched. | `inventory()` records directories and rejects special entries. Synthetic empty-directory and FIFO cases refuse. |
| Boundary certificate could omit the last native worker and resource monitor. | Required closure artifacts now include `monitor.json`, `project_status.json` and the native start receipt; the mandatory closed PID set includes owner, monitor, last worker, driver and watcher. |
| Exclusive dispatch/timeout files were fsynced without fsyncing their parent directories. | `write_new()` fsyncs the parent directory. STOP and newly created owner/dispatch directory names are also synced. Native stage/readiness adapters fsync the committed file and parent directory while holding the shared lifecycle lock. |
| Resolving a requested wave before checking for a symlink allowed `wave03 -> wave04` to create the `wave04` fence and dispatch the wrong original wave. | Dispatch now requires the literal canonical path before resolving/using it; the project precheck also rejects aliased wave paths. The dedicated synthetic symlink-alias fixture passes. |
| Direct collection CUDA-audit sources/dependencies were absent from the fragment source identity, and recorded skipped helper exit hashes were not rechecked. | The relevant audit sources/dependencies are now bound; `validate_binding()` rechecks the exact skipped rc0 receipt hash and return code. |

The missing-only census retains the original64 population and consumes every stage in the original durable launch prefix, including an empty launch log. Original12 wrapper commands, job identity, frozen plan hashes, status launch boundary and pristine whole-wave suffix are checked. Outcome fields are not parsed to choose the suffix. Actual census execution remains a later root action after truthful closure.

## Native interface review

The fragment job keeps each original stage's `worker --output <original E/test/waveNN>` arguments and maps only the entry module. `tools.tight_project.commands()` produces exactly one `worker` token. The native paired worker still installs paired execution, enters the native no-expert scope and calls the complete autonomous evaluator. Plan/session bytes, policy pairs, inference noise and success/control behavior are not rewritten by the adapter.

The native `tight_fresh_batch_run.run()` imports `state_resources.lease_for` at invocation, after the scoped adapter is installed. Its exact E/test root and original deadline map to the new C owner before native `lease_for()` chooses `FullLease`. That lease reads C's real native monitor telemetry and resources. E's wave plan and session remain the native worker inputs. The watcher retains raw monitor rows; it recognizes the exact fragment owner/job/worker/plan/dispatch fence without manufacturing owned telemetry.

The timeout intent is written and fsynced before the final recheck. The owner `project_status.json` publication, worker `initialization.json` publication and timeout STOP linearization all acquire `lifecycle13.lock`. Both monkeypatch sites match the native imports/call sites: `tight_project.execute()` imports the patched training-resource writer after installation, and the batch worker uses its patched module-level writer. Stage or readiness changes during intent fsync therefore cause an abandoned decision with no STOP. This review found no remaining stale timeout decision race in those requested paths.

## CPU verification

Registered interpreter:

`/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python`

Command:

`/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python -m pytest -q -p no:cacheprovider outputs/diagnostics/tight_paired_dagger_20261122_01/test_test_fragment13.py`

Result: **22 passed in 0.40s**. Fixtures use temporary directories and synthetic identities only.

An additional temporary-directory watcher probe imported `shared_capacity13` under that registered interpreter. It accepted the exact synthetic worker and rejected a wrong wave argv, wrong owner ancestry, receipt-incompatible process birth time and missing dispatch fence. A mocked watcher process probe accepted the exact clean idle gap and rejected stale heartbeat, errors and PID mismatch. No real process lookup, capacity query or service call was used. Bare system `python` lacked `psutil`; the registered interpreter imported and executed these CPU probes successfully.

## Reviewed snapshot hashes

| File | SHA256 |
| --- | --- |
| `test_fragment13.py` | `79e43e6f2496b0ccfbd40e200f1dcb53ca1d9c2579083ef27632c69997e03647` |
| `shared_capacity13.py` | `10d2e095f686bafaed8ea7156f83ba89fcefaa46cf38f01ef21cfa29e6a94103` |
| `test_test_fragment13.py` | `2e1e1dce0cef88268cee2413ee9d9afc27518a9979d04ea9ab41ff2d83815f90` |
| `TEST_FRAGMENT13_PREPARATION_RECIPE.md` | `295609b78a3d6a20fc2ad7178e226bdbd7594fadb08b4384bc30c13507a6aee2` |

The actual actor identities, closure evidence, original64 launch boundary, source/weight/session protection hashes, actual capacity, watcher schedule restoration and any eventual fragment outcome still require the later root checks described in the recipe. This source review supplies none of those runtime facts.
