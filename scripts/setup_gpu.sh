#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'scripts/setup_gpu.sh must be executed, not sourced; no environment change was made.' >&2
    return 2
fi

usage() {
    printf '%s\n' \
        'Usage: bash scripts/setup_gpu.sh [--profile PROFILE]' \
        '' \
        'Profiles: auto (default), cu118, cu126, cu130, cu132, legacy-cu118.' \
        'legacy-cu118 is opt-in and requires a separate, dedicated Conda environment.' \
        'CUDA_WHEEL_TAG remains optional; conflicts with --profile are rejected.' \
        'The installer does not create environments or change NVIDIA drivers.'
}

run_checked() {
    local description="$1"
    shift
    "$@"
    checked_status=$?
    if (( checked_status != 0 )); then
        printf '%s failed with status %s; setup stopped without running later steps.\n' "${description}" "${checked_status}" >&2
    fi
    return "${checked_status}"
}

main() {
local requested_profile profile_seen project_root cuda_output cuda_version
local requested_tag profile_selection profile_id wheel_tag lock_name constraints_name
local constraints_file lock_file torch_index_url torch_version default_snapshot_dir
local snapshot_dir lock_report freeze_report freeze_temporary status checked_status

requested_profile="auto"
profile_seen=0
while (( $# > 0 )); do
    case "$1" in
        --help|-h)
            usage
            return 0
            ;;
        --profile|--profile=*)
            if (( profile_seen )); then
                echo "--profile may only be specified once." >&2
                return 2
            fi
            profile_seen=1
            if [[ "$1" == "--profile" ]]; then
                if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
                    echo "--profile requires a profile name." >&2
                    return 2
                fi
                requested_profile="$2"
                shift 2
            else
                requested_profile="${1#--profile=}"
                shift
            fi
            case "${requested_profile}" in
                auto|cu118|cu126|cu130|cu132|legacy-cu118) ;;
                *)
                    echo "Unsupported profile: ${requested_profile}" >&2
                    usage >&2
                    return 2
                    ;;
            esac
            ;;
        *)
            echo "Unknown setup argument: $1" >&2
            usage >&2
            return 2
            ;;
    esac
done

if ! project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; then
    printf '%s\n' 'Could not resolve the project root; no environment change was made.' >&2
    return 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "scripts/setup_gpu.sh requires Linux with an NVIDIA GPU (workstation or server)." >&2
    return 2
fi
if ! source "${project_root}/scripts/conda_env.sh"; then
    printf '%s\n' 'Conda environment validation failed; no package installation was attempted.' >&2
    return 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; verify the NVIDIA driver, or request a GPU allocation on a managed cluster." >&2
    return 2
fi
if ! run_checked 'GPU inventory query' nvidia-smi -L; then
    return "${checked_status}"
fi

if ! cuda_output="$(nvidia-smi)"; then
    printf '%s\n' 'nvidia-smi failed while reading driver CUDA compatibility; no package installation was attempted.' >&2
    return 2
fi
cuda_version="$(printf '%s\n' "${cuda_output}" | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
if [[ -z "${cuda_version}" ]]; then
    echo "Could not read the driver CUDA compatibility from nvidia-smi." >&2
    return 2
fi
# Select a complete, versioned profile before changing the active environment.
# Explicit CUDA_WHEEL_TAG requests are validated, never silently downgraded.
requested_tag="${CUDA_WHEEL_TAG:-auto}"
if ! profile_selection="$("${environment_python}" "${project_root}/scripts/gpu_profiles.py" \
    --driver-cuda "${cuda_version}" --cuda-tag "${requested_tag}" \
    --profile "${requested_profile}" --check-host)"; then
    printf '%s\n' 'GPU profile selection failed; no package installation was attempted.' >&2
    return 2
fi
read -r profile_id wheel_tag lock_name constraints_name <<< "${profile_selection}"

constraints_file="${project_root}/${constraints_name}"
lock_file="${project_root}/${lock_name}"
torch_index_url="https://download.pytorch.org/whl/${wheel_tag}"
if [[ ! -f "${constraints_file}" || ! -f "${lock_file}" ]]; then
    echo "GPU lock files are missing: ${constraints_file} or ${lock_file}" >&2
    return 2
fi

torch_version="$(sed -n 's/^torch==//p' "${constraints_file}")"
if [[ -z "${torch_version}" || "${torch_version}" == *$'\n'* ]]; then
    echo "${constraints_file} must contain exactly one torch==version pin." >&2
    return 2
fi
echo "GPU profile: ${profile_id}; CUDA wheel: ${wheel_tag}"
echo "Requested profile: ${requested_profile}; CUDA_WHEEL_TAG: ${requested_tag}; nvidia-smi CUDA compatibility: ${cuda_version}"
echo "Locked dependencies: ${lock_name}; torch==${torch_version}"

if ! run_checked 'pip upgrade' "${environment_python}" -m pip install --upgrade pip; then
    return "${checked_status}"
fi
if ! run_checked 'setuptools and wheel installation' "${environment_python}" -m pip install "setuptools>=75" wheel; then
    return "${checked_status}"
fi
echo "Installing torch==${torch_version}+${wheel_tag} from ${torch_index_url}"
if ! run_checked 'PyTorch installation' "${environment_python}" -m pip install --upgrade \
    --constraint "${constraints_file}" \
    "torch==${torch_version}+${wheel_tag}" \
    --index-url "${torch_index_url}"; then
    return "${checked_status}"
fi
if ! run_checked 'locked dependency installation' "${environment_python}" -m pip install \
    --constraint "${constraints_file}" \
    --requirement "${lock_file}"; then
    return "${checked_status}"
fi
if ! run_checked 'editable project installation' "${environment_python}" -m pip install \
    --no-deps --no-build-isolation -e "${project_root}"; then
    return "${checked_status}"
fi

if ! cd "${project_root}"; then
    printf 'Could not enter project root: %s\n' "${project_root}" >&2
    return 2
fi
if ! run_checked 'installed dependency check' "${environment_python}" -m pip check; then
    return "${checked_status}"
fi
if [[ "${profile_id}" == "legacy-cu118" ]]; then
    default_snapshot_dir="${CONDA_PREFIX%/}/.new-gat-environment"
else
    default_snapshot_dir="${project_root}"
fi
snapshot_dir="${ENVIRONMENT_SNAPSHOT_DIR:-${default_snapshot_dir}}"
if ! mkdir -p "${snapshot_dir}"; then
    printf 'Could not create environment snapshot directory: %s\n' "${snapshot_dir}" >&2
    return 2
fi
lock_report="${snapshot_dir}/.gpu-environment.json"
freeze_report="${snapshot_dir}/.gpu-environment.freeze.txt"
if ! run_checked 'GPU lock verification' "${environment_python}" scripts/verify_gpu_lock.py \
    --lock "${lock_file}" \
    --constraints "${constraints_file}" \
    --cuda-tag "${wheel_tag}" \
    --profile "${profile_id}" \
    --json-out "${lock_report}"; then
    return "${checked_status}"
fi
if ! freeze_temporary="$(mktemp "${freeze_report}.tmp.XXXXXX")"; then
    printf 'Could not allocate a temporary environment snapshot beside %s.\n' "${freeze_report}" >&2
    return 2
fi
"${environment_python}" -m pip freeze --all > "${freeze_temporary}"
status=$?
if (( status != 0 )); then
    printf 'Environment freeze failed with status %s; partial temporary file remains at %s.\n' "${status}" "${freeze_temporary}" >&2
    return "${status}"
fi
if ! mv -f "${freeze_temporary}" "${freeze_report}"; then
    printf 'Could not publish the environment snapshot; temporary file remains at %s.\n' "${freeze_temporary}" >&2
    return 2
fi
if ! run_checked 'GPU preflight' "${environment_python}" scripts/gpu_preflight.py \
    --device "${DEVICE:-cuda}" \
    --require-paper-deps \
    --min-free-gb "${MIN_FREE_GB:-2}"; then
    return "${checked_status}"
fi

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
    if ! run_checked 'unit test suite' "${environment_python}" -m pytest -q; then
        return "${checked_status}"
    fi
fi

echo "Exact environment report: ${lock_report}"
echo "Resolved transitive snapshot: ${freeze_report}"
echo "GPU environment ready. Follow docs/GETTING_STARTED.md for data and experiments."
return 0
}

main "$@"
