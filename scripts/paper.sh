#!/usr/bin/env bash

main() {
    local project_root prepare_only inspection_only argument dependency_status
    local bootstrap_profile setup_status status
    if ! project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; then
        printf '%s\n' 'Could not resolve the project root; no paper task was started.' >&2
        return 2
    fi
    if ! source "${project_root}/scripts/conda_env.sh"; then
        printf '%s\n' 'Conda environment validation failed; no paper task was started.' >&2
        return 2
    fi

    prepare_only=0
    inspection_only=0
    for argument in "$@"; do
        case "${argument}" in
            --prepare-only) prepare_only=1 ;;
            --help|-h|--dry-run) inspection_only=1 ;;
        esac
    done

    if [[ "${prepare_only}" == "1" && "${inspection_only}" == "0" ]]; then
        "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet
        dependency_status=$?
        case "${dependency_status}" in
            0) ;;
            2)
                # Repair the installed stack, never reselect it from the current
                # driver's capability. Unknown Torch builds stop before installation.
                bootstrap_profile="$("${environment_python}" "${project_root}/scripts/gpu_profiles.py" --installed-profile)"
                status=$?
                if (( status != 0 )); then
                    printf 'Could not identify the installed GPU profile (status %s); no environment change was attempted.\n' "${status}" >&2
                    return "${status}"
                fi
                printf 'Research dependencies are missing or incompatible in %s.\n' "${CONDA_PREFIX}"
                printf '%s\n' 'Installing the complete locked GPU environment before preparing data.'
                bash "${project_root}/scripts/setup_gpu.sh" --profile "${bootstrap_profile}"
                setup_status=$?
                if (( setup_status != 0 )); then
                    printf 'GPU environment setup failed with status %s; data preparation was not started.\n' "${setup_status}" >&2
                    return "${setup_status}"
                fi
                "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet
                status=$?
                if (( status != 0 )); then
                    printf 'Dependencies remain invalid after setup (status %s); data preparation was not started.\n' "${status}" >&2
                    return "${status}"
                fi
                ;;
            *)
                # Host ABI failures and unexpected checker errors are not repaired.
                printf 'Dependency validation failed with status %s; no environment change or paper task was started.\n' "${dependency_status}" >&2
                return "${dependency_status}"
                ;;
        esac
    fi

    export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
    if ! cd "${project_root}"; then
        printf 'Could not enter project root: %s\n' "${project_root}" >&2
        return 2
    fi
    "${environment_python}" scripts/run_paper.py "$@"
    status=$?
    if (( status != 0 )); then
        printf 'Paper runner failed with status %s; inspect the Python manifest and logs.\n' "${status}" >&2
    fi
    return "${status}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'scripts/paper.sh must be executed, not sourced; no action was taken.' >&2
    return 2
fi
main "$@"
