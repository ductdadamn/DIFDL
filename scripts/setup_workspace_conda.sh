#!/usr/bin/env bash
# ==============================================================================
# DIFDL Workspace Anaconda Setup (RTX 5060 Ti 16GB / Blackwell sm_120 & General GPUs)
#
# Designed for single-node GPU servers with a /workspace directory:
#   1. Stores ALL Conda envs, caches, datasets, checkpoints, and repo inside
#      WORKSPACE_ROOT (default: /workspace/difdl_ds) — zero $HOME bloat.
#   2. Uses Anaconda (`conda`) exclusively with conda-forge libjpeg-turbo +
#      C/C++ compilers so jpeg2dct==0.2.4 builds cleanly without sudo.
#   3. Auto-detects RTX 50-series (Blackwell sm_120, e.g. RTX 5060 Ti 16GB)
#      via `nvidia-smi` and installs PyTorch +cu128 (required for sm_120).
#      Older GPUs default to PyTorch 2.3.0+cu121 (or pass `--cuda cu128`).
#
# Usage:
#   bash scripts/setup_workspace_conda.sh
#   bash scripts/setup_workspace_conda.sh --workspace /workspace/difdl_ds
#   bash scripts/setup_workspace_conda.sh --with-anytext
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default WORKSPACE_ROOT: parent of DIFDL if already inside /workspace, else /workspace/difdl_ds
if [ -z "${WORKSPACE_ROOT:-}" ]; then
    if [[ "${REPO_ROOT}" == /workspace/* ]]; then
        WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
    else
        WORKSPACE_ROOT="/workspace/difdl_ds"
    fi
fi

CUDA_TAG="auto"
WITH_ANYTEXT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace) WORKSPACE_ROOT="$2"; shift 2 ;;
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        --with-anytext) WITH_ANYTEXT=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Auto-detect Blackwell RTX 50-series (requires cu128 for sm_120 kernels)
if [ "${CUDA_TAG}" = "auto" ]; then
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if echo "${GPU_NAMES}" | grep -qE "RTX 50|RTX PRO|B100|B200|GB200"; then
        CUDA_TAG="cu128"
    else
        CUDA_TAG="cu121"
    fi
fi

ENV_PREFIX="${WORKSPACE_ROOT}/envs/difdl"
ANYTEXT_PREFIX="${WORKSPACE_ROOT}/envs/anytext"
CACHE_ROOT="${WORKSPACE_ROOT}/.cache"

mkdir -p \
    "${WORKSPACE_ROOT}/envs" \
    "${WORKSPACE_ROOT}/datasets" \
    "${WORKSPACE_ROOT}/checkpoints_backup" \
    "${CACHE_ROOT}/conda_pkgs" \
    "${CACHE_ROOT}/pip" \
    "${CACHE_ROOT}/tmp" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/modelscope" \
    "${CACHE_ROOT}/paddlex" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/gdown"

export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export PADDLE_PDX_CACHE_HOME="${CACHE_ROOT}/paddlex"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export NO_ALBUMENTATIONS_UPDATE=1

echo "======================================================================"
echo " DIFDL Workspace Anaconda Setup"
echo " Repo root      : ${REPO_ROOT}"
echo " Workspace root : ${WORKSPACE_ROOT}"
echo " Conda prefix   : ${ENV_PREFIX}"
echo " Cache dir      : ${CACHE_ROOT}"
echo " CUDA wheel tag : ${CUDA_TAG}"
echo "======================================================================"

# 1. Locate Anaconda / Miniconda
if ! command -v conda >/dev/null 2>&1; then
    for cand in \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/opt/anaconda3/etc/profile.d/conda.sh" \
        "/opt/miniconda3/etc/profile.d/conda.sh" \
        "/usr/local/anaconda3/etc/profile.d/conda.sh" \
        "/workspace/anaconda3/etc/profile.d/conda.sh" \
        "/workspace/miniconda3/etc/profile.d/conda.sh" \
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
    echo "[ERROR] 'conda' not found. Load Anaconda or check its install path on this server."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "[OK] Found Conda at: ${CONDA_BASE}"

# 2. Create Conda env in workspace with Python 3.10 + libjpeg-turbo + compilers + gdown
if [ ! -d "${ENV_PREFIX}/bin" ]; then
    echo "[INFO] Creating Conda env at ${ENV_PREFIX}..."
    conda create --prefix "${ENV_PREFIX}" -y -c conda-forge \
        python=3.10 pip "setuptools<75" wheel libjpeg-turbo pkg-config c-compiler cxx-compiler gdown
else
    echo "[INFO] Reusing existing Conda env at ${ENV_PREFIX}..."
fi

conda activate "${ENV_PREFIX}"

export CPATH="${ENV_PREFIX}/include:${CPATH:-}"
export C_INCLUDE_PATH="${ENV_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${ENV_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# 3. Install numpy<2 + build tools before jpeg2dct
python -m pip install --upgrade "pip<26" "setuptools<75" wheel "numpy>=1.26.4,<2.0.0"

# 4. Install PyTorch matching GPU architecture (cu128 for RTX 5060 Ti sm_120, cu121 for older GPUs)
if [ "${CUDA_TAG}" = "cu128" ]; then
    echo "[INFO] Installing PyTorch >=2.7.0+cu128 (Blackwell sm_120 support)..."
    python -m pip install "torch>=2.7.0,<2.9.0" torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    grep -vE "^(torch|torchvision|--extra-index-url)" "${REPO_ROOT}/requirements.txt" > "${TMPDIR}/difdl_reqs_notorch.txt"
    python -m pip install --no-build-isolation -r "${TMPDIR}/difdl_reqs_notorch.txt"
elif [ "${CUDA_TAG}" = "cu121" ]; then
    echo "[INFO] Installing requirements.txt (PyTorch 2.3.0+cu121)..."
    python -m pip install --no-build-isolation -r "${REPO_ROOT}/requirements.txt"
else
    echo "[INFO] Installing PyTorch 2.3.0+${CUDA_TAG} + requirements..."
    python -m pip install "torch==2.3.0+${CUDA_TAG}" "torchvision==0.18.0+${CUDA_TAG}" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    grep -vE "^(torch|torchvision|--extra-index-url)" "${REPO_ROOT}/requirements.txt" > "${TMPDIR}/difdl_reqs_notorch.txt"
    python -m pip install --no-build-isolation -r "${TMPDIR}/difdl_reqs_notorch.txt"
fi

# 5. Install CPU PaddlePaddle in main env (keeps PyTorch CUDA/cuDNN untouched)
echo "[INFO] Installing CPU paddlepaddle==3.0.0 in main env..."
python -m pip install "paddlepaddle==3.0.0"

# 6. Keep numpy<2 and headless OpenCV (prevents libGL.so.1 crash on headless servers)
python -m pip install "numpy>=1.26.4,<2.0.0"
python -m pip install --force-reinstall --no-deps "opencv-python-headless==4.10.0.84"

# 7. Create workspace activation helper (does NOT touch ~/.bashrc)
ACTIVATE_FILE="${WORKSPACE_ROOT}/activate_difdl.sh"
cat > "${ACTIVATE_FILE}" <<EOF
# Activate DIFDL Conda environment inside workspace:
#   source ${ACTIVATE_FILE}
export WORKSPACE_ROOT="${WORKSPACE_ROOT}"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export PADDLE_PDX_CACHE_HOME="${CACHE_ROOT}/paddlex"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export NO_ALBUMENTATIONS_UPDATE=1

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
conda activate "${ENV_PREFIX}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
echo "[DIFDL] Activated env: ${ENV_PREFIX} (workspace: ${WORKSPACE_ROOT})"
EOF
chmod +x "${ACTIVATE_FILE}"

# 8. Optional isolated AnyText Conda env inside workspace
if [ "${WITH_ANYTEXT}" -eq 1 ]; then
    echo "[INFO] Setting up isolated AnyText Conda env at ${ANYTEXT_PREFIX}..."
    if [ ! -d "${ANYTEXT_PREFIX}/bin" ]; then
        conda create --prefix "${ANYTEXT_PREFIX}" -y -c conda-forge python=3.10 pip "setuptools<70" wheel
    fi
    conda activate "${ANYTEXT_PREFIX}"
    python -m pip install --upgrade "pip<26" "setuptools<70" wheel "Cython<3.0"
    if [ "${CUDA_TAG}" = "cu128" ]; then
        python -m pip install "torch>=2.7.0,<2.9.0" torchvision \
            --index-url https://download.pytorch.org/whl/cu128
    else
        python -m pip install --no-build-isolation \
            --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
            "torch==2.1.2+${CUDA_TAG}" "torchvision==0.16.2+${CUDA_TAG}"
    fi
    python -m pip install --no-build-isolation \
        "setuptools<70" "numpy>=1.23.5,<1.26.0" "scipy==1.10.1" "Pillow==9.5.0" \
        "albumentations==1.3.1" "opencv-python-headless==4.8.1.78" \
        "pytorch-lightning==1.5.0" "torchmetrics==0.6.0" "pyDeprecate==0.3.1" \
        "omegaconf==2.1.1" "einops==0.3.0" "kornia==0.6.0" \
        "huggingface-hub==0.23.5" "transformers==4.30.2" "diffusers==0.10.2" "safetensors" \
        "timm==0.6.12" "open-clip-torch==2.0.2" \
        "easydict==1.10" "addict==2.4.0" "datasets==2.18.0" "modelscope==1.10.0" \
        "simplejson" "sortedcontainers" "tqdm" "freetype-py"
    mkdir -p "${WORKSPACE_ROOT}/third_party"
    if [ ! -d "${WORKSPACE_ROOT}/third_party/AnyText" ]; then
        git clone https://github.com/tyxsspa/AnyText.git "${WORKSPACE_ROOT}/third_party/AnyText"
    fi
    conda activate "${ENV_PREFIX}"
fi

# 9. Verify environment & CUDA kernel execution on actual GPU
echo "[INFO] Verifying DIFDL environment..."
"${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/verify_env.py"

echo "======================================================================"
echo " [DONE] Workspace environment ready!"
echo " Activate on future logins with:"
echo "   source ${ACTIVATE_FILE}"
echo "======================================================================"
