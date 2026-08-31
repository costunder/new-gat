#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${project_root}/scripts/conda_env.sh"
export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"
exec "${environment_python}" -B -u -m research.conductance_gat.c_learning.intervene "$@"
