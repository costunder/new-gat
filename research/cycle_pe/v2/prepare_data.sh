#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${project_root}/scripts/paper.sh" "$@" --suite benchmark --tracks cycle_pe --cycle-pe-version v2 --prepare-only --allow-download
