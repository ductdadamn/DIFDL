#!/usr/bin/env bash
# ==============================================================================
# DIFDL Lab Server Setup (/storage + Anaconda + Shared Account Safe)
#
# Designed specifically for shared lab servers (e.g., ict14 & sibling nodes):
#   1. Places ALL Conda envs, package caches, temp build files, model caches
#      (PyTorch, HuggingFace, ModelScope, PaddleOCR), and repo inside /storage
#      so $HOME is NEVER bloated.
#   2. Safe for shared Unix accounts: NEVER modifies ~/.bashrc, ~/.gitconfig,
#      or any directory outside ${TEAM_ROOT}.
#   3. Installs libjpeg-turbo headers via conda-forge inside the env prefix so
#      jpeg2dct==0.2.4 compiles without needing sudo/root!
#   4. Shared across all cluster nodes mounting /storage: install once, use on
#      any node via `source /storage/difdl_ds/activate_difdl.sh`.
#
# Usage (run from inside /storage/difdl_ds/DIFDL):
#   bash scripts/setup_server_storage.sh                     # Main env
#   bash scripts/setup_server_storage.sh --with-anytext      # Main + AnyText env
#   bash scripts/setup_server_storage.sh --cuda cu118        # Older driver
#   TEAM_ROOT=/storage/nhom_difdl bash scripts/setup_server_storage.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Auto-detect writable user directory under /storage (since /storage root is root-owned 755)
CURRENT_USER="$(whoami)"
if [ -z "${TEAM_ROOT:-}" ]; then
    if [ -d "/storage/${CURRENT_USER}" ] && [ -w "/storage/${CURRENT_USER}" ]; then
        TEAM_ROOT="/storage/${CURRENT_USER}/difdl_ds"
    elif [ -d "/storage/shared" ] && [ -w "/storage/shared" ]; then
        TEAM_ROOT="/storage/shared/difdl_ds_${CURRENT_USER}"
    else
        echo "[ERROR] Neither /storage/${CURRENT_USER} nor /storage/shared is writable!"
        echo "        Check 'whoami' and pass --team-root /storage/<your_writable_dir>/difdl_ds"
        exit 1
    fi
fi
CUDA_TAG="cu121"
WITH_ANYTEXT=0
PADDLE_MODE="gpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --team-root) TEAM_ROOT="$2"; shift 2 ;;
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        --with-anytext) WITH_ANYTEXT=1; shift ;;
        --cpu-paddle) PADDLE_MODE="cpu"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Safety check: ensure TEAM_ROOT is inside /storage (unless explicitly overridden)
if [[ "${TEAM_ROOT}" != /storage/* ]]; then
    echo "[WARN] TEAM_ROOT (${TEAM_ROOT}) is not under /storage!"
    echo "       Make sure you have enough disk quota here."
fi

ENV_PREFIX="${TEAM_ROOT}/envs/difdl"
ANYTEXT_PREFIX="${TEAM_ROOT}/envs/anytext"
CACHE_ROOT="${TEAM_ROOT}/.cache"

mkdir -p "${TEAM_ROOT}"
chmod 700 "${TEAM_ROOT}" 2>/dev/null || true
mkdir -p \
    "${TEAM_ROOT}/envs" \
    "${TEAM_ROOT}/datasets" \
    "${TEAM_ROOT}/checkpoints_backup" \
    "${CACHE_ROOT}/conda_pkgs" \
    "${CACHE_ROOT}/pip" \
    "${CACHE_ROOT}/tmp" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/modelscope" \
    "${CACHE_ROOT}/paddlex"

# Redirect EVERY cache & temp folder to /storage (zero $HOME usage!)
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export PADDLE_PDX_CACHE_HOME="${CACHE_ROOT}/paddlex"

echo "======================================================================"
echo " DIFDL Lab Server Setup (/storage Shared-Account Safe Mode)"
echo " Repo root    : ${REPO_ROOT}"
echo " Team storage : ${TEAM_ROOT}"
echo " Conda prefix : ${ENV_PREFIX}"
echo " Cache dir    : ${CACHE_ROOT}"
echo " CUDA tag     : ${CUDA_TAG} | Paddle: ${PADDLE_MODE}"
echo "======================================================================"

# 1. Locate Anaconda / Miniconda on the server
if ! command -v conda >/dev/null 2>&1; then
    for cand in \
        "/opt/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/usr/local/anaconda3/etc/profile.d/conda.sh" \
        "/storage/anaconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "${cand}" ]; then
            # shellcheck disable=SC1090
            source "${cand}"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 'conda' command not found in PATH or standard locations."
    echo "        Run 'which conda' or check where Anaconda is installed on ict14."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "[OK] Found Conda at: ${CONDA_BASE}"

# 2. Create Conda env in /storage with Python 3.10 + libjpeg-turbo + C/C++ compilers
if [ ! -d "${ENV_PREFIX}/bin" ]; then
    echo "[INFO] Creating Conda env at ${ENV_PREFIX} (Python 3.10 + libjpeg-turbo)..."
    conda create --prefix "${ENV_PREFIX}" -y -c conda-forge \
        python=3.10 pip "setuptools<75" wheel libjpeg-turbo pkg-config c-compiler cxx-compiler
else
    echo "[INFO] Existing Conda env found at ${ENV_PREFIX}; reusing..."
fi

conda activate "${ENV_PREFIX}"

# Expose conda-forge headers/libs so jpeg2dct==0.2.4 builds cleanly without sudo
export CPATH="${ENV_PREFIX}/include:${CPATH:-}"
export C_INCLUDE_PATH="${ENV_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${ENV_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# 3. Install numpy<2 first so jpeg2dct compiles against numpy C-API cleanly
echo "[INFO] Installing numpy<2 and build prerequisites..."
python -m pip install --upgrade "pip<26" "setuptools<75" wheel "numpy>=1.26.4,<2.0.0"

# 4. Install PyTorch + DIFDL requirements into /storage env
echo "[INFO] Installing DIFDL requirements (CUDA=${CUDA_TAG})..."
if [ "${CUDA_TAG}" = "cu121" ]; then
    python -m pip install --no-build-isolation -r "${REPO_ROOT}/requirements.txt"
else
    python -m pip install \
        "torch==2.3.0+${CUDA_TAG}" "torchvision==0.18.0+${CUDA_TAG}" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    grep -vE "^(torch|torchvision|--extra-index-url)" "${REPO_ROOT}/requirements.txt" > "${TMPDIR}/difdl_reqs_notorch.txt"
    python -m pip install --no-build-isolation -r "${TMPDIR}/difdl_reqs_notorch.txt"
fi

# 5. Install PaddlePaddle runtime for src/seg_char.py
if [ "${PADDLE_MODE}" = "gpu" ]; then
    echo "[INFO] Installing paddlepaddle-gpu..."
    python -m pip install "paddlepaddle-gpu>=3.0.0" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ || {
        echo "[WARN] paddlepaddle-gpu wheel unavailable; installing CPU paddlepaddle..."
        python -m pip install "paddlepaddle>=3.0.0"
    }
else
    python -m pip install "paddlepaddle>=3.0.0"
fi

python -m pip install "numpy>=1.26.4,<2.0.0"

# 6. Write convenient activation script in /storage (does NOT touch ~/.bashrc!)
ACTIVATE_FILE="${TEAM_ROOT}/activate_difdl.sh"
cat > "${ACTIVATE_FILE}" <<EOF
# Source this file after SSHing into any lab server sharing /storage:
#   source ${ACTIVATE_FILE}
export TEAM_ROOT="${TEAM_ROOT}"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export PADDLE_PDX_CACHE_HOME="${CACHE_ROOT}/paddlex"

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
conda activate "${ENV_PREFIX}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
echo "[DIFDL] Activated env: ${ENV_PREFIX} (caches -> ${CACHE_ROOT})"
EOF
chmod +x "${ACTIVATE_FILE}"

# 7. Optional: Isolated AnyText Conda env in /storage
if [ "${WITH_ANYTEXT}" -eq 1 ]; then
    echo "[INFO] Setting up isolated AnyText Conda env at ${ANYTEXT_PREFIX}..."
    if [ ! -d "${ANYTEXT_PREFIX}/bin" ]; then
        conda create --prefix "${ANYTEXT_PREFIX}" -y -c conda-forge python=3.10 pip "setuptools<75" wheel
    fi
    conda activate "${ANYTEXT_PREFIX}"
    python -m pip install \
        "torch==2.1.2+${CUDA_TAG}" "torchvision==0.16.2+${CUDA_TAG}" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    python -m pip install \
        "numpy>=1.23.5,<1.26.0" "albumentations==0.4.3" "opencv-python-headless==4.8.1.78" \
        "pytorch-lightning==1.5.0" "omegaconf==2.1.1" "einops==0.3.0" "transformers==4.19.2" \
        "timm==0.6.12" "open-clip-torch==2.0.2" "easydict==1.10" "addict==2.4.0" \
        "modelscope==1.10.0" "Pillow>=9.5.0,<11.0.0" "scipy" "tqdm" "freetype-py"

    mkdir -p "${TEAM_ROOT}/third_party"
    if [ ! -d "${TEAM_ROOT}/third_party/AnyText" ]; then
        git clone https://github.com/tyxsspa/AnyText.git "${TEAM_ROOT}/third_party/AnyText"
    fi
    conda activate "${ENV_PREFIX}"
fi

# 8. Verify main environment
echo "[INFO] Verifying DIFDL environment..."
"${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/verify_env.py"

echo "======================================================================"
echo " [DONE] Lab Server /storage environment is ready!"
echo " Next time you SSH into ANY server sharing /storage, just run:"
echo "   source ${ACTIVATE_FILE}"
echo "======================================================================"
