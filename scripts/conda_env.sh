#!/usr/bin/env bash
# Sourced by the Bash entrypoints after they set project_root.
# Never create an environment or fall back to a PATH/system Python here.

conda_env_main() {
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        printf '%s\n' 'scripts/conda_env.sh is a source-only validation library; no environment was changed.' >&2
        return 2
    fi
    if [[ -z "${project_root:-}" ]]; then
        printf '%s\n' 'project_root is not set; Conda validation was not attempted.' >&2
        return 2
    fi
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        printf '%s\n' 'No active Conda environment. Create and activate a dedicated environment first:' >&2
        printf '%s\n' '  conda create -n new-gat python=3.11 pip -y' >&2
        printf '%s\n' '  conda activate new-gat' >&2
        return 2
    fi

    environment_python="${CONDA_PREFIX%/}/bin/python"
    if [[ ! -x "${environment_python}" ]]; then
        printf 'The active Conda environment has no executable Python: %s\n' "${environment_python}" >&2
        printf '%s\n' 'Activate the dedicated new-gat environment before running this script.' >&2
        return 2
    fi

    if ! "${environment_python}" "${project_root}/scripts/verify_conda_env.py"; then
        printf '%s\n' 'Conda environment verification failed; no downstream command was started.' >&2
        return 2
    fi
    return 0
}

conda_env_main "$@"
