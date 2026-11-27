#!/usr/bin/env bash
# ==============================================================================
# Download the AnyText ModelScope model repo (weights + ms_wrapper.py + configuration.json)
# to a local dir, using a CURRENT modelscope client in its own tiny venv.
#
# Why: modelscope 1.10 (required to RUN AnyText) can no longer download from today's hub
# ("File README.md download incomplete, content_length: None", docs/KAGGLE_RUN_LOG.md #4).
# The old client loads a local dir fine, so: new client downloads, old client runs.
#
# Usage: bash scripts/download_anytext_model.sh [LOCAL_DIR]     (default /tmp/anytext_model, ~5-6GB)
# Idempotent: skipped when LOCAL_DIR already holds configuration.json + ms_wrapper.py + .done
# ==============================================================================
set -euo pipefail

DEST="${1:-/tmp/anytext_model}"
REVISION="${ANYTEXT_MODEL_REVISION:-v1.1.3}"
VENV=/tmp/venv_msdl
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}" UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
unset UV_SYSTEM_PYTHON

if [ -f "${DEST}/.done" ] && [ -f "${DEST}/ms_wrapper.py" ]; then
    echo "[OK] AnyText model already at ${DEST}"; cat "${DEST}/.done"; exit 0
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
if [ ! -x "${VENV}/bin/python" ]; then
    uv python install 3.10 >/dev/null
    uv venv --clear "${VENV}" --python 3.10 --seed >/dev/null
    uv pip install --python "${VENV}/bin/python" -q modelscope
fi

"${VENV}/bin/python" - "${DEST}" "${REVISION}" <<'PY'
import sys, json, time
from pathlib import Path
import modelscope
from modelscope import snapshot_download

dest, rev = Path(sys.argv[1]), sys.argv[2]
errors = []
for model_id in ("damo/cv_anytext_text_generation_editing", "iic/cv_anytext_text_generation_editing"):
    for revision in (rev, None):
        try:
            t0 = time.time()
            kw = {"revision": revision} if revision else {}
            path = snapshot_download(model_id, local_dir=str(dest), **kw)
            files = sorted(p.name for p in Path(path).iterdir())
            assert "configuration.json" in files and "ms_wrapper.py" in files, f"unexpected repo content: {files}"
            info = {"model_id": model_id, "revision": revision or "default", "path": str(path),
                    "modelscope_client": modelscope.__version__, "minutes": round((time.time() - t0) / 60, 1),
                    "files": files}
            (dest / ".done").write_text(json.dumps(info, indent=1))
            print(json.dumps(info, indent=1))
            sys.exit(0)
        except Exception as exc:
            errors.append(f"{model_id}@{revision}: {type(exc).__name__}: {exc}")
            print("[WARN]", errors[-1][:300])
print("[FAIL] could not download AnyText model:\n  " + "\n  ".join(errors))
sys.exit(1)
PY
