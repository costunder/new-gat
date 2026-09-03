#!/usr/bin/env bash
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}" || exit 1
# Dataset-aware reference sizes: ZINC 128/64/10, Peptides 256/64/6.
exec python -B "${project_root}/scripts/run_cycle_scaling.py" "$@" --versions v2 --profiles reference --model-seeds 0
