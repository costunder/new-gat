#!/usr/bin/env bash

main() {
    local project_root inspection_only argument status
    if ! project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; then
        printf '%s\n' 'Could not resolve the project root; no benchmark was started.' >&2
        return 2
    fi
    if ! source "${project_root}/scripts/conda_env.sh"; then
        printf '%s\n' 'Conda environment validation failed; no benchmark was started.' >&2
        return 2
    fi

    inspection_only=0
    for argument in "$@"; do
        case "${argument}" in
            --help|-h) inspection_only=1 ;;
        esac
    done
    if [[ "${inspection_only}" == "0" ]]; then
        if ! "${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet; then
            printf '%s\n' 'Dependency validation failed; no benchmark was started.' >&2
            return 2
        fi
    fi

    export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
    if ! cd "${project_root}"; then
        printf 'Could not enter project root: %s\n' "${project_root}" >&2
        return 2
    fi
    "${environment_python}" scripts/benchmark_speed.py "$@"
    status=$?
    if (( status != 0 )); then
        printf 'Speed benchmark runner failed with status %s; inspect its output and result files.\n' "${status}" >&2
    fi
    return "${status}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'scripts/benchmark_speed.sh must be executed, not sourced; no action was taken.' >&2
    return 2
fi
main "$@"
