#!/usr/bin/env bash

main() {
    local project_root status
    if ! project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; then
        printf '%s\n' 'Could not resolve the project root; no dataset preparation was started.' >&2
        return 2
    fi
    bash "${project_root}/scripts/paper.sh" --suite benchmark --prepare-only --allow-download "$@"
    status=$?
    if (( status != 0 )); then
        printf 'Dataset preparation failed with status %s; inspect the Python run manifest and logs.\n' "${status}" >&2
    fi
    return "${status}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'scripts/prepare_data.sh must be executed, not sourced; no action was taken.' >&2
    return 2
fi
main "$@"
