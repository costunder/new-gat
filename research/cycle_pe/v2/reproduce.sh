#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Keep this entrypoint on v2 even if generic selection flags are supplied.
exec bash "${project_root}/scripts/paper.sh" "$@" --suite benchmark --tracks cycle_pe --cycle-pe-version v2
