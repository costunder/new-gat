#!/usr/bin/env bash
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}" || exit 1
exec python -B -m research.cycle_pe.v2.benchmark "$@" --prepare-only --allow-download --device cpu
