#!/usr/bin/env bash
# ==============================================================================
# AnyText Isolated Environment Setup (Server Lab / High-VRAM GPU)
#
# Why isolated (.venv_anytext)?
#   AnyText (github.com/tyxsspa/AnyText) pins legacy albumentations==0.4.3,
#   pytorch-lightning==1.5.0, transformers==4.19.2, and timm==0.6.12 which
#   conflict with ADCD-Net's albumentations==2.0.8 & PyTorch 2.3.0.
#   Keeping AnyText in .venv_anytext prevents breaking the main .venv.
#
# Usage:
#   bash scripts/setup_anytext.sh              # Default: cu121
#   bash scripts/setup_anytext.sh --cuda cu118 # For older Lab Server drivers
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANYTEXT_DIR="${REPO_ROOT}/third_party/AnyText"
VENV_DIR="${REPO_ROOT}/.venv_anytext"
CUDA_TAG="cu121"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

export PATH="${HOME}/.local/bin:${PATH}"
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=5

if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "[INFO] Cloning AnyText into ${ANYTEXT_DIR} (if not already present)..."
mkdir -p "${REPO_ROOT}/third_party"
if [ ! -d "${ANYTEXT_DIR}" ]; then
    git clone https://github.com/tyxsspa/AnyText.git "${ANYTEXT_DIR}"
fi

echo "[INFO] Creating isolated Python 3.10 venv at ${VENV_DIR}..."
uv python install 3.10
uv venv --clear "${VENV_DIR}" --python 3.10 --seed

echo "[INFO] Installing AnyText PyTorch (${CUDA_TAG}) & dependencies..."
uv pip install --python "${VENV_DIR}/bin/python" "setuptools<70" wheel "Cython<3.0"

uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "setuptools<70" \
    "torch==2.1.2+${CUDA_TAG}" "torchvision==0.16.2+${CUDA_TAG}" \
    "numpy==1.23.5" "scipy==1.10.1" "Pillow==9.5.0" \
    "albumentations==1.3.1" "opencv-python-headless==4.8.1.78" \
    "pytorch-lightning==1.5.0" "torchmetrics==0.6.0" "pyDeprecate==0.3.1" \
    "omegaconf==2.1.1" "einops==0.3.0" "kornia==0.6.0" \
    "huggingface-hub==0.23.5" "transformers==4.30.2" "diffusers==0.10.2" "safetensors" \
    "timm==0.6.12" "open-clip-torch==2.0.2" \
    "easydict==1.10" "addict==2.4.0" "datasets==2.18.0" "modelscope==1.10.0" \
    "simplejson" "sortedcontainers" "tqdm" "freetype-py"

echo "[INFO] Verifying AnyText venv..."
"${VENV_DIR}/bin/python" -c "
import torch, albumentations, pytorch_lightning, transformers, modelscope
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())
print('albumentations:', albumentations.__version__, '| lightning:', pytorch_lightning.__version__)
"
echo "[DONE] AnyText isolated env ready at ${VENV_DIR}"
