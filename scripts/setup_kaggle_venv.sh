#!/usr/bin/env bash
# ==============================================================================
# Pinned, isolated Python 3.10 venvs for the Kaggle synthesis notebooks.
#
# Why: the Kaggle image changes under us (Python 3.13, numpy 2, Pillow 13 as of 11/2026)
# and unpinned installs pull incompatible versions (e.g. paddleocr 3.7 + paddle 3.0
# -> "Type of attribute: strides is not right"). Each tool gets its own venv with
# versions that are known to work together; the Kaggle Python is only used for display.
#
# Usage:  bash scripts/setup_kaggle_venv.sh ocr     -> /tmp/venv_ocr   (PaddleOCR CPU, notebook 01)
#         bash scripts/setup_kaggle_venv.sh lama    -> /tmp/venv_lama  (LaMa GPU,      notebook 02)
# AnyText (notebook 03) keeps using scripts/setup_anytext.sh.
# Idempotent: re-running skips the install if the venv already passed its check.
# ==============================================================================
set -euo pipefail

NAME="${1:?usage: setup_kaggle_venv.sh ocr|lama}"
VENV="/tmp/venv_${NAME}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR=/tmp/uv_cache UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5

if [ -f "${VENV}/.ok" ]; then
    echo "[OK] ${VENV} already set up"; cat "${VENV}/.ok"; exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
uv python install 3.10 >/dev/null
uv venv --clear "${VENV}" --python 3.10 --seed >/dev/null
PY="${VENV}/bin/python"

case "${NAME}" in
  ocr)
    # paddle 3.0.0 + paddleocr 3.0.x are released together (PP-OCRv5); do NOT let paddleocr float.
    uv pip install --python "${PY}" "paddlepaddle==3.0.0" "paddleocr>=3.0.0,<3.1" \
        "numpy>=1.26.4,<2.0.0" "opencv-python-headless<4.12" "Pillow>=9.5,<12"
    CHECK='import numpy, paddle, paddleocr, PIL; print("numpy", numpy.__version__, "| paddle", paddle.__version__, "| paddleocr", paddleocr.__version__, "| Pillow", PIL.__version__)'
    ;;
  lama)
    # Same torch as the project env (requirements.txt); simple-lama resolves its own Pillow/numpy pins.
    uv pip install --python "${PY}" --extra-index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.3.0+cu121" "torchvision==0.18.0+cu121" "simple-lama-inpainting==0.1.2" "numpy>=1.26.4,<2.0.0"
    CHECK='import torch, numpy, PIL, simple_lama_inpainting; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| numpy", numpy.__version__, "| Pillow", PIL.__version__)'
    ;;
  *) echo "unknown venv '${NAME}' (expected ocr|lama)"; exit 1 ;;
esac

"${PY}" -c "${CHECK}" | tee "${VENV}/.ok"
echo "[DONE] ${VENV} ready -> use ${PY}"
