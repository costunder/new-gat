#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${VENV_DIR:-${project_root}/.venv-gpu}"

if [[ -n "${PYTHON:-}" ]]; then
    environment_python="${PYTHON}"
elif [[ "${USE_ACTIVE_ENV:-0}" == "1" ]]; then
    environment_python="python"
else
    environment_python="${venv_dir}/bin/python"
fi

if [[ ! -x "${environment_python}" ]] && ! command -v "${environment_python}" >/dev/null 2>&1; then
    echo "GPU Python was not found. Run bash scripts/setup_gpu.sh first." >&2
    exit 2
fi

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"
"${environment_python}" scripts/run_paper.py "$@"
