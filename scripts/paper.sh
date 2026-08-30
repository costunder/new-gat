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
    dependency_status=0
    "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet || dependency_status=$?
    case "${dependency_status}" in
        0) ;;
        2)
            # Repair the installed stack, never reselect it from the current
            # driver's capability. Unknown Torch builds stop before installation.
            bootstrap_profile="$("${environment_python}" "${project_root}/scripts/gpu_profiles.py" --installed-profile)"
            echo "Research dependencies are missing or incompatible in ${CONDA_PREFIX}."
            echo "Installing the complete locked GPU environment before preparing data."
            bash "${project_root}/scripts/setup_gpu.sh" --profile "${bootstrap_profile}"
            "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet
            ;;
        *)
            # Host ABI failures (3) and unexpected checker errors are not
            # repaired by reinstalling wheels. Preserve the checker's status.
            exit "${dependency_status}"
            ;;
    esac
fi

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"
"${environment_python}" scripts/run_paper.py "$@"
