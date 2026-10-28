# Project output inventory and cleanup ledger (2026-10-28)

The inventory is restricted to this repository's `outputs/` tree and follows
no symlinks. Sizes below are actual file-byte sums after the cleanup; decimal
GB is used for the user limit.

| root | bytes | decimal GB | treatment |
| --- | ---: | ---: | --- |
| `outputs/visuotactile/camera_v3_20261028` | 2,532,569,088 | 2.533 | selected bracket evidence, small audit JSON, three delivered demos and compact/raw qualified records retained |
| `outputs/visuotactile` (all) | 7,117,987,840 | 7.118 | prior causal/perception/insertion evidence and qualified data retained |
| `outputs/compat_check` | 406,133,811 | 0.406 | compatibility evidence retained |
| `outputs/research_audit` | 463,230 | 0.00046 | source audit retained |
| `outputs/source_review_177a080_200ea34_6fc04a8` | 13,636,062 | 0.014 | local source review retained |
| **all `outputs/`** | **10,074,898,432** | **10.075** | below 60,000,000,000-byte hard limit |

The three old version-3 production episodes remain checksummed in
`insertion_triple400_watchoff_20261028`; they are not mixed with the new
version-4 candidate. The version-4 probe root is only about 0.27 MB after
removing unusable payloads.

## Deletion

`outputs/visuotactile/camera_v3_20261028/candidate_cleanup_ledger.json` records
the per-file hashes and paths for **3,037,673,433 bytes** deleted. These were
candidate camera raw HDF5, repeated MP4/PNG/NPZ and conversion payloads after
their review summaries and metadata had been retained. The selected
`bracket_17`, `bracket_920101`, and `bracket_448` raw/compact records, their
original synchronized videos, calibration metadata, source hashes and the
small review artifacts were not deleted. No checkpoint, asset, dependency,
credential, unrelated project or protected process output was touched.

The version-4 probe ledger separately records seeds 930006--930010. Invalid
starts and the two-camera accidental success have only metadata/results kept;
their raw HDF5/images were not eligible training data. The LLVM startup failure
never created a trajectory.

This is a point-in-time audit. Any new collector must re-run the byte sum before
starting an attempt and include temporary conversion files in the same limit.
