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
# Usage:  bash scripts/setup_kaggle_venv.sh ocr     -> /tmp/venv_ocr   (PaddleOCR, GPU if visible else CPU; notebook 01)
#         bash scripts/setup_kaggle_venv.sh lama    -> /tmp/venv_lama  (LaMa GPU,      notebook 02)
#         bash scripts/setup_kaggle_venv.sh jpeg    -> /tmp/venv_jpeg  (Pillow only, CPU; notebook 05 JPEG stress)
# AnyText (notebook 03) uses scripts/setup_anytext.sh (same technique).
# Idempotent: skipped if the venv already passed its check with the same snapshot date.
# ==============================================================================
set -euo pipefail

NAME="${1:?usage: setup_kaggle_venv.sh ocr|lama|jpeg}"
VENV="/tmp/venv_${NAME}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR=/tmp/uv_cache UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
unset UV_SYSTEM_PYTHON  # Kaggle sets it; it only produces warnings for `uv venv`

case "${NAME}" in
  ocr)  SNAPSHOT="${OCR_EXCLUDE_NEWER:-2025-07-15T00:00:00Z}" ;;   # paddle 3.0 / paddleocr 3.0.x / paddlex 3.0.x
  lama) SNAPSHOT="${LAMA_EXCLUDE_NEWER:-2024-07-01T00:00:00Z}" ;;  # torch 2.3.0 era (project env)
  jpeg) SNAPSHOT="${JPEG_EXCLUDE_NEWER:-2024-07-01T00:00:00Z}" ;;  # Pillow 10.3 = fixed libjpeg-turbo encoder
  *) echo "unknown venv '${NAME}' (expected ocr|lama|jpeg)"; exit 1 ;;
esac

# ocr: GPU build if a GPU is visible (override: OCR_DEVICE=cpu|gpu); recorded in .ok as want=...
WANT_DEVICE=n/a  # only meaningful for ocr
if [ "${NAME}" = ocr ]; then
    WANT_DEVICE=cpu
    if [ -n "${OCR_DEVICE:-}" ]; then WANT_DEVICE="${OCR_DEVICE}"
    elif nvidia-smi -L >/dev/null 2>&1; then WANT_DEVICE=gpu; fi
    echo "[ocr] wanted device: ${WANT_DEVICE}"
fi

if [ -f "${VENV}/.ok" ] && grep -q "snapshot=${SNAPSHOT}" "${VENV}/.ok" && grep -q "want=${WANT_DEVICE}" "${VENV}/.ok"; then
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
    # The PyPI package `paddlepaddle` is CPU-only; GPU needs `paddlepaddle-gpu` from Paddle's own index
    # (PyPI's paddlepaddle-gpu stops at 2.6). Installed first WITHOUT the snapshot (like torch below);
    # its CUDA/cuDNN come as pinned nvidia-* wheels, so the Kaggle system CUDA does not matter.
    # If no GPU, or the GPU build fails its matmul test, fall back to the CPU build (never a dead venv).
    GPU_SMOKE='import paddle; assert paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0, "no CUDA device"; paddle.device.set_device("gpu:0"); x = paddle.ones([64, 64]); print("gpu matmul ok:", float((x @ x).sum()))'
    DEVICE=cpu
    if [ "${WANT_DEVICE}" = gpu ]; then
        for tag in ${PADDLE_CUDA_TAGS:-cu126 cu118}; do
            echo "[ocr] installing paddlepaddle-gpu==3.0.0 (${tag}) ..."
            if "${UVI[@]}" --extra-index-url "https://www.paddlepaddle.org.cn/packages/stable/${tag}/" \
                   --index-strategy unsafe-best-match "paddlepaddle-gpu==3.0.0" "numpy>=1.26.4,<2.0.0" \
               && "${PY}" -c "${GPU_SMOKE}"; then
                DEVICE=gpu; break
            fi
            echo "[WARN] paddlepaddle-gpu ${tag} not usable, trying next"
            uv pip uninstall --python "${PY}" -q paddlepaddle-gpu >/dev/null 2>&1 || true
        done
        [ "${DEVICE}" = gpu ] || echo "[WARN] GPU paddle failed -> falling back to CPU build (slower OCR)"
    fi
    PADDLE_PKG=$([ "${DEVICE}" = gpu ] && echo "paddlepaddle-gpu==3.0.0" || echo "paddlepaddle==3.0.0")
    if [ "${DEVICE}" = gpu ]; then
        # paddle-gpu already installed: resolve the rest at the snapshot (paddleocr/paddlex don't depend on paddle)
        "${UVI[@]}" --exclude-newer "${SNAPSHOT}" \
            "paddleocr>=3.0.0,<3.1" "paddlex>=3.0.0,<3.1" \
            "numpy>=1.26.4,<2.0.0" "opencv-python-headless<4.12" "Pillow>=9.5,<12"
        "${PY}" -c "${GPU_SMOKE}"   # numpy may have been downgraded: re-test
    else
        "${UVI[@]}" --exclude-newer "${SNAPSHOT}" "${PADDLE_PKG}" \
            "paddleocr>=3.0.0,<3.1" "paddlex>=3.0.0,<3.1" \
            "numpy>=1.26.4,<2.0.0" "opencv-python-headless<4.12" "Pillow>=9.5,<12"
    fi
    CHECK='import numpy, paddle, paddleocr, paddlex, PIL; print("numpy", numpy.__version__, "| paddle", paddle.__version__, "| cuda", paddle.device.is_compiled_with_cuda(), "| paddleocr", paddleocr.__version__, "| paddlex", paddlex.__version__, "| Pillow", PIL.__version__)'
    CHECK="${CHECK}; print('device=${DEVICE}')"
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
  jpeg)
    # The JPEG encoder is part of the result (quant tables, chroma subsampling, DCT rounding), so it is pinned
    "${UVI[@]}" --exclude-newer "${SNAPSHOT}" "Pillow==10.3.0" "numpy>=1.26.4,<2.0.0"
    CHECK='import PIL, numpy; from PIL import features; print("Pillow", PIL.__version__, "| libjpeg_turbo", features.version("libjpeg_turbo"), "| numpy", numpy.__version__)'
    ;;
esac

OKTXT="$("${PY}" -c "${CHECK}")"   # set -e: a failed import aborts here, so no .ok marker is written
printf '%s\nsnapshot=%s\nwant=%s\n' "${OKTXT}" "${SNAPSHOT}" "${WANT_DEVICE}" > "${VENV}/.ok"
cat "${VENV}/.ok"
"${PY}" -m pip freeze > "${VENV}/freeze.txt" 2>/dev/null || uv pip freeze --python "${PY}" > "${VENV}/freeze.txt"
echo "[DONE] ${VENV} ready -> use ${PY} (full lock: ${VENV}/freeze.txt)"
