#!/usr/bin/env bash

main() {
    local project_root status
    if ! project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; then
        printf '%s\n' 'Could not resolve the project root; no experiment was started.' >&2
        return 2
    fi
    if ! cd "${project_root}"; then
        printf 'Could not enter project root: %s\n' "${project_root}" >&2
        return 2
    fi
    # Dataset-aware reference sizes: ZINC 128/64/10, Peptides 256/64/6.
    python -B "${project_root}/scripts/run_cycle_scaling.py" "$@" --versions v2 --profiles reference --model-seeds 0
    status=$?
    if (( status != 0 )); then
        printf 'Cycle PE V2 runner failed with status %s; inspect its manifest and logs.\n' "${status}" >&2
    fi
    return "${status}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'research/cycle_pe/v2/reproduce.sh must be executed, not sourced; no action was taken.' >&2
    return 2
fi
main "$@"
