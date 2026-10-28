#!/usr/bin/env bash
# Apply limits to this new process only. Cameras still use the GPU renderer.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${RESEARCH_THREADS:-2}" MKL_NUM_THREADS="${RESEARCH_THREADS:-2}" OPENBLAS_NUM_THREADS="${RESEARCH_THREADS:-2}" PXR_WORK_THREAD_LIMIT="${RESEARCH_THREADS:-2}"
exec nice -n 19 taskset -c "${RESEARCH_CPUS:-22,23}" .venv_isaacsim/bin/python "$@"
