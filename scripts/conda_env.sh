#!/usr/bin/env bash
# Sourced by the Bash entrypoints after they set project_root.
# Never create an environment or fall back to a PATH/system Python here.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active Conda environment. Create and activate a dedicated environment first:" >&2
    echo "  conda create -n new-gat python=3.11 pip -y" >&2
    echo "  conda activate new-gat" >&2
    exit 2
fi

environment_python="${CONDA_PREFIX%/}/bin/python"
if [[ ! -x "${environment_python}" ]]; then
    echo "The active Conda environment has no executable Python: ${environment_python}" >&2
    echo "Activate the dedicated new-gat environment before running this script." >&2
    exit 2
fi

if ! "${environment_python}" "${project_root}/scripts/verify_conda_env.py"; then
    exit 2
fi
