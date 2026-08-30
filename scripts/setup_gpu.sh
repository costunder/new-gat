#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "scripts/setup_gpu.sh requires Linux with an NVIDIA GPU (workstation or server)." >&2
    exit 2
fi
source "${project_root}/scripts/conda_env.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; verify the NVIDIA driver, or request a GPU allocation on a managed cluster." >&2
    exit 2
fi
nvidia-smi -L

cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
if [[ -z "${cuda_version}" ]]; then
    echo "Could not read the driver CUDA compatibility from nvidia-smi." >&2
    exit 2
fi
# Select a complete, versioned profile before changing the active environment.
# Explicit CUDA_WHEEL_TAG requests are validated, never silently downgraded.
requested_tag="${CUDA_WHEEL_TAG:-auto}"
profile_selection="$("${environment_python}" "${project_root}/scripts/gpu_profiles.py" \
    --driver-cuda "${cuda_version}" --cuda-tag "${requested_tag}" --check-host)"
read -r wheel_tag lock_name <<< "${profile_selection}"

constraints_file="${project_root}/constraints-${wheel_tag}.txt"
lock_file="${project_root}/${lock_name}"
torch_index_url="https://download.pytorch.org/whl/${wheel_tag}"
if [[ ! -f "${constraints_file}" || ! -f "${lock_file}" ]]; then
    echo "GPU lock files are missing: ${constraints_file} or ${lock_file}" >&2
    exit 2
fi

torch_version="$(sed -n 's/^torch==//p' "${constraints_file}")"
if [[ -z "${torch_version}" || "${torch_version}" == *$'\n'* ]]; then
    echo "${constraints_file} must contain exactly one torch==version pin." >&2
    exit 2
fi
echo "GPU profile: ${wheel_tag} (requested: ${requested_tag}; nvidia-smi CUDA compatibility: ${cuda_version})"
echo "Locked dependencies: ${lock_name}; torch==${torch_version}"

"${environment_python}" -m pip install --upgrade pip
"${environment_python}" -m pip install "setuptools>=75" wheel
echo "Installing torch==${torch_version}+${wheel_tag} from ${torch_index_url}"
"${environment_python}" -m pip install --upgrade \
    --constraint "${constraints_file}" \
    "torch==${torch_version}+${wheel_tag}" \
    --index-url "${torch_index_url}"
"${environment_python}" -m pip install \
    --constraint "${constraints_file}" \
    --requirement "${lock_file}"
"${environment_python}" -m pip install \
    --no-deps --no-build-isolation -e "${project_root}"

cd "${project_root}"
"${environment_python}" -m pip check
snapshot_dir="${ENVIRONMENT_SNAPSHOT_DIR:-${project_root}}"
mkdir -p "${snapshot_dir}"
lock_report="${snapshot_dir}/.gpu-environment.json"
freeze_report="${snapshot_dir}/.gpu-environment.freeze.txt"
"${environment_python}" scripts/verify_gpu_lock.py \
    --lock "${lock_file}" \
    --constraints "${constraints_file}" \
    --cuda-tag "${wheel_tag}" \
    --json-out "${lock_report}"
freeze_temporary="$(mktemp "${freeze_report}.tmp.XXXXXX")"
"${environment_python}" -m pip freeze --all > "${freeze_temporary}"
mv -f "${freeze_temporary}" "${freeze_report}"
"${environment_python}" scripts/gpu_preflight.py \
    --device "${DEVICE:-cuda}" \
    --require-paper-deps \
    --min-free-gb "${MIN_FREE_GB:-2}"

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
    "${environment_python}" -m pytest -q
fi

echo "Exact environment report: ${lock_report}"
echo "Resolved transitive snapshot: ${freeze_report}"
echo "GPU environment ready. Follow README.md for dataset preparation and experiments."
