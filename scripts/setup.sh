#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON:-python3}"
venv_dir="${VENV_DIR:-${project_root}/.venv}"

if [[ "${USE_ACTIVE_ENV:-0}" == "1" ]]; then
    environment_python="${PYTHON:-python}"
else
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
        "${python_command}" -m venv "${venv_dir}"
    fi
    environment_python="${venv_dir}/bin/python"
fi

if ! "${environment_python}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "Python 3.11 or newer is required: ${environment_python}" >&2
    exit 2
fi

if [[ "${SKIP_DEPS:-0}" == "1" ]]; then
    # Use an already-provisioned cluster environment without changing its packages.
    "${environment_python}" -m pip install --no-deps --no-build-isolation -e "${project_root}"
elif [[ "${USE_LOCK:-0}" == "1" ]]; then
    "${environment_python}" -m pip install --upgrade pip
    "${environment_python}" -m pip install "setuptools>=75" wheel
    "${environment_python}" -m pip install -r "${project_root}/requirements-lock.txt"
    "${environment_python}" -m pip install --no-deps --no-build-isolation -e "${project_root}"
else
    "${environment_python}" -m pip install --upgrade pip
    "${environment_python}" -m pip install "setuptools>=75" wheel
    # Flexible minimum versions from pyproject.toml are the portable default.
    # A preinstalled CUDA-specific PyTorch satisfying torch>=2.2 is preserved.
    "${environment_python}" -m pip install --no-build-isolation -e "${project_root}[dev]"
fi

cd "${project_root}"
"${environment_python}" -m pytest -q
