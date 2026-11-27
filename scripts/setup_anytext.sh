#!/usr/bin/env bash
# ==============================================================================
# AnyText isolated environment (Kaggle T4 / Lab Server), inference only.
#
# Versions follow AnyText's own environment.yaml (torch 2.0.1, transformers 4.30.2,
# open_clip 2.7.0, torchmetrics 0.11.4, modelscope 1.10.0, ...) minus demo/translator/training-only
# packages (gradio, streamlit, tensorflow, basicsr, ...). Packages that environment.yaml only pulled in
# INDIRECTLY but the inference code imports (scikit-image via basicsr -> cldm/recognizer.py, six) are
# listed explicitly; scripts/anytext_import_scan.py verifies the whole import tree (KAGGLE_RUN_LOG #10). Every transitive dependency is resolved as of
# ANYTEXT_EXCLUDE_NEWER, the AnyText v1.1 release era (see docs/KAGGLE_RUN_LOG.md #3, #4).
# The AnyText repo is pinned to a commit. Model weights are downloaded separately by
# scripts/download_anytext_model.sh (modelscope 1.10 cannot download from today's hub).
#
# Usage:
#   bash scripts/setup_anytext.sh               # default cu118 (torch 2.0.1 has no cu121 wheels)
#   bash scripts/setup_anytext.sh --cuda cu117
# Idempotent: skipped if ${VENV_DIR}/.ok matches the current snapshot + commit.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANYTEXT_DIR="${REPO_ROOT}/third_party/AnyText"
VENV_DIR="${REPO_ROOT}/.venv_anytext"
CUDA_TAG="cu118"
ANYTEXT_COMMIT="${ANYTEXT_COMMIT:-3d4a260}"                       # tyxsspa/AnyText HEAD on 2025-03-07
ANYTEXT_EXCLUDE_NEWER="${ANYTEXT_EXCLUDE_NEWER:-2024-01-15T00:00:00Z}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
STAMP="snapshot=${ANYTEXT_EXCLUDE_NEWER} commit=${ANYTEXT_COMMIT} cuda=${CUDA_TAG} deps=2"  # bump deps= when the package list changes

export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}" UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
unset UV_SYSTEM_PYTHON  # Kaggle sets it; it only produces warnings for `uv venv`

if [ -f "${VENV_DIR}/.ok" ] && grep -qF "${STAMP}" "${VENV_DIR}/.ok"; then
    echo "[OK] AnyText env already set up"; cat "${VENV_DIR}/.ok"; exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

echo "[INFO] AnyText repo @ ${ANYTEXT_COMMIT} -> ${ANYTEXT_DIR}"
mkdir -p "${REPO_ROOT}/third_party"
[ -d "${ANYTEXT_DIR}/.git" ] || git clone -q https://github.com/tyxsspa/AnyText.git "${ANYTEXT_DIR}"
git -C "${ANYTEXT_DIR}" checkout -q "${ANYTEXT_COMMIT}"

echo "[INFO] Python 3.10 venv -> ${VENV_DIR}"
uv python install 3.10 >/dev/null
uv venv --clear "${VENV_DIR}" --python 3.10 --seed >/dev/null
PY="${VENV_DIR}/bin/python"
UVI=(uv pip install --python "${PY}" -q)

echo "[INFO] torch 2.0.1+${CUDA_TAG} (PyTorch index, no snapshot: it may not publish upload times)"
"${UVI[@]}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "torch==2.0.1+${CUDA_TAG}" "torchvision==0.15.2+${CUDA_TAG}"

echo "[INFO] AnyText deps (environment.yaml pins, snapshot ${ANYTEXT_EXCLUDE_NEWER})"
"${UVI[@]}" --exclude-newer "${ANYTEXT_EXCLUDE_NEWER}" \
    "torch==2.0.1+${CUDA_TAG}" "torchvision==0.15.2+${CUDA_TAG}" \
    "setuptools<70" "numpy==1.23.5" "Pillow==9.5.0" "opencv-python-headless==4.7.0.72" \
    "pytorch-lightning==1.5.0" "torchmetrics==0.11.4" "omegaconf==2.2.3" "einops==0.4.1" \
    "transformers==4.30.2" "kornia==0.6.7" "open_clip_torch==2.7.0" "timm==0.6.7" \
    "addict==2.4.0" "safetensors==0.4.0" "modelscope==1.10.0" "easydict==1.10" \
    "diffusers==0.10.2" "imageio==2.9.0" "jieba==0.42.1" "sentencepiece==0.1.99" \
    "ujson" "fsspec" "tqdm" "scipy" "simplejson" "sortedcontainers" "datasets" "oss2"     "scikit-image" "six" "matplotlib"

# xformers is optional (speed only); never fail the setup because of it
"${UVI[@]}" --exclude-newer "${ANYTEXT_EXCLUDE_NEWER}" "xformers==0.0.20" \
    || echo "[WARN] xformers not installed (optional)"

echo "[INFO] Verifying AnyText venv..."
OKTXT="$("${PY}" -c "
import torch, pytorch_lightning, transformers, modelscope, open_clip, diffusers, cv2, numpy
from importlib.metadata import version as v  # open_clip 2.7.0 has no __version__ attribute (KAGGLE_RUN_LOG #11)
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| lightning', v('pytorch-lightning'),
      '| transformers', v('transformers'), '| modelscope', v('modelscope'),
      '| open_clip', v('open_clip_torch'), '| numpy', v('numpy'))
")"
# every module the AnyText inference code can import must resolve (static scan, nothing executed)
"${PY}" "${SCRIPT_DIR}/anytext_import_scan.py" --anytext-dir "${ANYTEXT_DIR}"
printf '%s\n%s\n' "${OKTXT}" "${STAMP}" > "${VENV_DIR}/.ok"
"${PY}" -m pip freeze > "${VENV_DIR}/freeze.txt" 2>/dev/null || true
cat "${VENV_DIR}/.ok"
echo "[DONE] AnyText env ready at ${VENV_DIR} (weights: bash scripts/download_anytext_model.sh)"
