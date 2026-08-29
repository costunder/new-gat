#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${VENV_DIR:-${project_root}/.venv}"

if [[ -n "${PYTHON:-}" ]]; then
    python_command="${PYTHON}"
elif [[ "${USE_ACTIVE_ENV:-0}" == "1" ]]; then
    python_command="python"
else
    python_command="${venv_dir}/bin/python"
fi

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"

"${python_command}" scripts/run_all.py --device "${DEVICE:-auto}" "$@"
