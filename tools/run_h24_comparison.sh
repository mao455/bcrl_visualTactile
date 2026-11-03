#!/usr/bin/env bash
# The service owns its monitor and children; no nested shell variable expansion.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec bash visuotactile/run_cpu.sh -m tools.h24_comparison_run run --root "${1:?absolute run root required}"
