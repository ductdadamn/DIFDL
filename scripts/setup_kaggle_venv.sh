#!/usr/bin/env bash
# ==============================================================================
# Pinned, isolated Python 3.10 venvs for the Kaggle synthesis notebooks.
#
# Why: the Kaggle image changes under us (Python 3.13, numpy 2, Pillow 13 as of 11/2026), and
# pinning only the top-level packages is NOT enough: unpinned sub-dependencies float to today's
# versions (paddleocr 3.0.3 pulled paddlex 3.7.2 -> "PaddlePredictorOption.__init__() takes 1
# positional argument"). So every venv is resolved with `--exclude-newer <date>` = "as if it were
# that day", when the whole dependency set was released together. See docs/KAGGLE_RUN_LOG.md.
#
# Usage:  bash scripts/setup_kaggle_venv.sh ocr     -> /tmp/venv_ocr   (PaddleOCR CPU, notebook 01)
#         bash scripts/setup_kaggle_venv.sh lama    -> /tmp/venv_lama  (LaMa GPU,      notebook 02)
# AnyText (notebook 03) uses scripts/setup_anytext.sh (same technique).
# Idempotent: skipped if the venv already passed its check with the same snapshot date.
# ==============================================================================
set -euo pipefail

NAME="${1:?usage: setup_kaggle_venv.sh ocr|lama}"
VENV="/tmp/venv_${NAME}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR=/tmp/uv_cache UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
unset UV_SYSTEM_PYTHON  # Kaggle sets it; it only produces warnings for `uv venv`

case "${NAME}" in
  ocr)  SNAPSHOT="${OCR_EXCLUDE_NEWER:-2025-07-15T00:00:00Z}" ;;   # paddle 3.0 / paddleocr 3.0.x / paddlex 3.0.x
  lama) SNAPSHOT="${LAMA_EXCLUDE_NEWER:-2024-07-01T00:00:00Z}" ;;  # torch 2.3.0 era (project env)
  *) echo "unknown venv '${NAME}' (expected ocr|lama)"; exit 1 ;;
esac

if [ -f "${VENV}/.ok" ] && grep -q "snapshot=${SNAPSHOT}" "${VENV}/.ok"; then
    echo "[OK] ${VENV} already set up"; cat "${VENV}/.ok"; exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
uv python install 3.10 >/dev/null
uv venv --clear "${VENV}" --python 3.10 --seed >/dev/null
PY="${VENV}/bin/python"
UVI=(uv pip install --python "${PY}" -q)

case "${NAME}" in
  ocr)
    "${UVI[@]}" --exclude-newer "${SNAPSHOT}" \
        "paddlepaddle==3.0.0" "paddleocr>=3.0.0,<3.1" "paddlex>=3.0.0,<3.1" \
        "numpy>=1.26.4,<2.0.0" "opencv-python-headless<4.12" "Pillow>=9.5,<12"
    CHECK='import numpy, paddle, paddleocr, paddlex, PIL; print("numpy", numpy.__version__, "| paddle", paddle.__version__, "| paddleocr", paddleocr.__version__, "| paddlex", paddlex.__version__, "| Pillow", PIL.__version__)'
    ;;
  lama)
    # torch from the PyTorch index (no --exclude-newer: that index may not publish upload times) ...
    "${UVI[@]}" --extra-index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.3.0+cu121" "torchvision==0.18.0+cu121"
    # ... everything else from PyPI frozen at the snapshot; the installed torch satisfies the pins
    "${UVI[@]}" --exclude-newer "${SNAPSHOT}" \
        "torch==2.3.0+cu121" "torchvision==0.18.0+cu121" "simple-lama-inpainting==0.1.2" "numpy>=1.26.4,<2.0.0"
    CHECK='import torch, numpy, PIL, simple_lama_inpainting; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| numpy", numpy.__version__, "| Pillow", PIL.__version__)'
    ;;
esac

OKTXT="$("${PY}" -c "${CHECK}")"   # set -e: a failed import aborts here, so no .ok marker is written
printf '%s\nsnapshot=%s\n' "${OKTXT}" "${SNAPSHOT}" > "${VENV}/.ok"
cat "${VENV}/.ok"
"${PY}" -m pip freeze > "${VENV}/freeze.txt" 2>/dev/null || uv pip freeze --python "${PY}" > "${VENV}/freeze.txt"
echo "[DONE] ${VENV} ready -> use ${PY} (full lock: ${VENV}/freeze.txt)"
