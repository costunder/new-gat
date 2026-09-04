#!/usr/bin/env bash

main() {
    local project_root status
    if ! project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; then
        printf '%s\n' 'Could not resolve the project root; no dataset preparation was started.' >&2
        return 2
    fi
    if ! cd "${project_root}"; then
        printf 'Could not enter project root: %s\n' "${project_root}" >&2
        return 2
    fi
    python -B -m research.cycle_pe.v2.benchmark "$@" --prepare-only --allow-download --device cpu
    status=$?
    if (( status != 0 )); then
        printf 'Cycle PE V2 data preparation failed with status %s; inspect the Python output and cache state.\n' "${status}" >&2
    fi
    return "${status}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'research/cycle_pe/v2/prepare_data.sh must be executed, not sourced; no action was taken.' >&2
    return 2
fi
main "$@"
