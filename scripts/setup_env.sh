#!/usr/bin/env bash
# ==============================================================================
# DIFDL Synchronized Environment Setup (Python 3.10 + PyTorch 2.3.0)
# Works identically across local GPUs (RTX 3050 / 4060) and any Linux Lab Server
#
# Usage:
#   bash scripts/setup_env.sh                       # Default: CUDA 12.1 + Paddle GPU
#   bash scripts/setup_env.sh --cuda cu118          # For older Lab Server drivers (CUDA 11.8)
#   bash scripts/setup_env.sh --cpu-paddle          # Use CPU PaddlePaddle
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_VER="3.10"
PADDLE_MODE="gpu"
CUDA_TAG="cu121"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-paddle) PADDLE_MODE="cpu"; shift ;;
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo " DIFDL Environment Setup"
echo " Repo root : ${REPO_ROOT}"
echo " Target env: ${VENV_DIR} (Python ${PYTHON_VER})"
echo " CUDA tag  : ${CUDA_TAG} | Paddle: ${PADDLE_MODE}"
echo "======================================================================"

# 1. Check system libjpeg-dev (required to build jpeg2dct==0.2.4)
if [ ! -f /usr/include/jpeglib.h ] && [ ! -f /usr/local/include/jpeglib.h ]; then
    echo "[ERROR] libjpeg header (jpeglib.h) not found!"
    echo "        jpeg2dct==0.2.4 requires libjpeg-dev before building."
    echo "        Run: sudo apt-get update && sudo apt-get install -y build-essential libjpeg-dev"
    exit 1
fi
echo "[OK] Found libjpeg header (jpeglib.h)."

# 2. Ensure uv is available (no sudo required on Lab Server)
export PATH="${HOME}/.local/bin:${PATH}"
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=5
if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] Installing uv (standalone Python/venv manager, no sudo needed)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=5
fi
echo "[OK] uv version: $(uv --version)"

# 3. Install portable CPython 3.10 and create .venv
echo "[INFO] Ensuring Python ${PYTHON_VER} is installed via uv..."
uv python install "${PYTHON_VER}"

if [ -d "${VENV_DIR}" ]; then
    echo "[INFO] Existing ${VENV_DIR} found; reusing/updating..."
else
    echo "[INFO] Creating ${VENV_DIR} with Python ${PYTHON_VER}..."
    uv venv "${VENV_DIR}" --python "${PYTHON_VER}" --seed
fi

# 4. Install build helpers first so jpeg2dct compiles cleanly against numpy
echo "[INFO] Installing build helpers (setuptools, wheel, numpy<2)..."
uv pip install --python "${VENV_DIR}/bin/python" "setuptools<75" wheel "numpy>=1.26.4,<2.0.0"

# 5. Install main requirements (PyTorch 2.3.0 + ADCD-Net + PaddleOCR + LaMa)
echo "[INFO] Installing requirements.txt (CUDA=${CUDA_TAG})..."
if [ "${CUDA_TAG}" = "cu121" ]; then
    uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation -r "${REPO_ROOT}/requirements.txt"
else
    # Override torch/torchvision index for cu118 or cpu
    uv pip install --python "${VENV_DIR}/bin/python" \
        "torch==2.3.0+${CUDA_TAG}" "torchvision==0.18.0+${CUDA_TAG}" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    grep -vE "^(torch|torchvision|--extra-index-url)" "${REPO_ROOT}/requirements.txt" > /tmp/difdl_reqs_notorch.txt
    uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation -r /tmp/difdl_reqs_notorch.txt
    rm -f /tmp/difdl_reqs_notorch.txt
fi

# 6. Install PaddlePaddle runtime for src/seg_char.py (PP-OCRv5_server_det)
if [ "${PADDLE_MODE}" = "gpu" ]; then
    echo "[INFO] Installing paddlepaddle-gpu (CUDA 12.x compatible)..."
    uv pip install --python "${VENV_DIR}/bin/python" "paddlepaddle-gpu>=3.0.0" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ || {
        echo "[WARN] paddlepaddle-gpu wheel unavailable; falling back to CPU paddlepaddle..."
        uv pip install --python "${VENV_DIR}/bin/python" "paddlepaddle>=3.0.0"
    }
else
    echo "[INFO] Installing paddlepaddle (CPU)..."
    uv pip install --python "${VENV_DIR}/bin/python" "paddlepaddle>=3.0.0"
fi

# 7. Ensure numpy<2.0.0 and opencv-python-headless remain clean after all installs
uv pip install --python "${VENV_DIR}/bin/python" "numpy>=1.26.4,<2.0.0"

# 8. Verify installation
echo "[INFO] Running environment verification..."
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/verify_env.py"

echo "======================================================================"
echo " [DONE] Environment ready!"
echo " Activate with: source ${VENV_DIR}/bin/activate"
echo " Or run directly: .venv/bin/python scripts/verify_env.py"
echo "======================================================================"
