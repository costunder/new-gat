#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/conda_env.sh"

prepare_only=0
inspection_only=0
for argument in "$@"; do
    case "${argument}" in
        --prepare-only) prepare_only=1 ;;
        --help|-h|--dry-run) inspection_only=1 ;;
    esac
done

if [[ "${prepare_only}" == "1" && "${inspection_only}" == "0" ]]; then
    if ! "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet; then
        echo "Research dependencies are missing or incompatible in ${CONDA_PREFIX}."
        echo "Installing the complete locked GPU environment before preparing data."
        bash "${project_root}/scripts/setup_gpu.sh"
        "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet
    fi
fi

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"
"${environment_python}" scripts/run_paper.py "$@"
