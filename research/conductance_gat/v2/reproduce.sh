#!/usr/bin/env bash

main() {
    local project_root status
    if ! project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; then
        printf '%s\n' 'Could not resolve the project root; no experiment was started.' >&2
        return 2
    fi
    if ! source "${project_root}/scripts/conda_env.sh"; then
        printf '%s\n' 'Conda environment validation failed; no experiment was started.' >&2
        return 2
    fi
    export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
    if ! cd "${project_root}"; then
        printf 'Could not enter project root: %s\n' "${project_root}" >&2
        return 2
    fi
    "${environment_python}" -B scripts/run_conductance_v2.py "$@"
    status=$?
    if (( status != 0 )); then
        printf 'Conductance V2 runner failed with status %s; inspect its manifest and logs.\n' "${status}" >&2
    fi
    return "${status}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'research/conductance_gat/v2/reproduce.sh must be executed, not sourced; no action was taken.' >&2
    return 2
fi
main "$@"
