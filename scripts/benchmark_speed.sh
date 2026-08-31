#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/conda_env.sh"

inspection_only=0
for argument in "$@"; do
    case "${argument}" in
        --help|-h) inspection_only=1 ;;
    esac
done
if [[ "${inspection_only}" == "0" ]]; then
    "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet
fi

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"
exec "${environment_python}" scripts/benchmark_speed.py "$@"
