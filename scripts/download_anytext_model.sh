#!/usr/bin/env bash
# ==============================================================================
# Get the AnyText model repo (anytext_v1.1.ckpt + clip-vit-large-patch14/ + ms_wrapper.py +
# configuration.json, ~7.5 GB) into a local dir, using CURRENT download clients in a tiny venv.
#
# Why (docs/KAGGLE_RUN_LOG.md #4, #9):
#  - modelscope 1.10 (required to RUN AnyText) can no longer download from today's hub;
#    the old client loads a local dir fine -> new client downloads, old client runs.
#  - the full ModelScope repo is ~15 GB: 7.9 GB of it is the zh->en translator (unused,
#    use_translator=False) -> skipped.
#  - ModelScope rate-limits anonymous downloads from shared IPs ("操作过于频繁，请稍后再试")
#    -> retry with back-off, then fall back to the HuggingFace mirror camenduru/anytext
#    (same anytext_v1.1.ckpt + clip; its ms_wrapper.py is an older revision -> preferred only
#    as a fallback; if ModelScope answers for the 2 small files, those are used instead).
#
# Order: 1) complete copy already in DEST or under /kaggle/input -> done (no download)
#        2) ModelScope (damo/ then iic/, revision v1.1.3), 3 attempts with back-off
#        3) HuggingFace camenduru/anytext @ pinned commit
# Usage: bash scripts/download_anytext_model.sh [DEST]   (default /tmp/anytext_model)
#        ANYTEXT_SOURCE=hf|modelscope to force one source.
# ==============================================================================
set -euo pipefail

DEST="${1:-/tmp/anytext_model}"
VENV=/tmp/venv_msdl
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}" UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
export HF_HUB_ENABLE_HF_TRANSFER=0
unset UV_SYSTEM_PYTHON

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
if [ ! -f "${VENV}/.ok" ]; then
    uv python install 3.10 >/dev/null
    uv venv --clear "${VENV}" --python 3.10 --seed >/dev/null
    uv pip install --python "${VENV}/bin/python" -q modelscope huggingface_hub
    touch "${VENV}/.ok"
fi

"${VENV}/bin/python" - "${DEST}" "${ANYTEXT_SOURCE:-auto}" <<'PY'
import hashlib, json, os, shutil, sys, time
from pathlib import Path

dest, source = Path(sys.argv[1]), sys.argv[2]
REQUIRED = ("configuration.json", "ms_wrapper.py", "anytext_v1.1.ckpt", "clip-vit-large-patch14/pytorch_model.bin")
SKIP_RE = [r"nlp_csanmt_translation_zh2en/.*", r"description/.*"]           # modelscope (regex)
HF_REPO, HF_REV = "camenduru/anytext", "9712932572600a5947976405ad84b84184a6a5ce"
CKPT_SHA256 = "09c88d145aa08f067641dc1d10c8890855939fa756d4cdc991e781a5f3918f59"  # anytext_v1.1.ckpt (HF LFS)

def complete(d): return all((Path(d) / f).exists() for f in REQUIRED)

def finish(path, info):
    path = Path(path)
    missing = [f for f in REQUIRED if not (path / f).exists()]
    assert not missing, f"downloaded dir lacks {missing}"
    info.update(path=str(path), size_gb=round(sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e9, 2))
    (path / ".done").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1)); sys.exit(0)

# 1) already there?
if complete(dest):
    print(f"[OK] AnyText model already complete at {dest}"); sys.exit(0)
if Path("/kaggle/input").exists():
    for p in sorted(Path("/kaggle/input").rglob("ms_wrapper.py")):
        if complete(p.parent):
            print(f"[OK] complete AnyText model found in Kaggle input: {p.parent} (anytext_loader 'auto' uses it; no download)")
            sys.exit(0)

errors = []

def from_modelscope():
    from modelscope import snapshot_download
    for attempt, wait in enumerate((0, 60, 180), 1):
        if wait:
            print(f"[modelscope] waiting {wait}s before attempt {attempt} (rate limit back-off)...", flush=True); time.sleep(wait)
        for model_id in ("damo/cv_anytext_text_generation_editing", "iic/cv_anytext_text_generation_editing"):
            try:
                t0 = time.time()
                path = snapshot_download(model_id, revision="v1.1.3", local_dir=str(dest), ignore_file_pattern=SKIP_RE)
                finish(path, {"source": "modelscope", "model_id": model_id, "revision": "v1.1.3",
                              "minutes": round((time.time() - t0) / 60, 1)})
            except SystemExit:
                raise
            except Exception as exc:
                msg = f"modelscope {model_id} (attempt {attempt}): {type(exc).__name__}: {str(exc)[:250]}"
                errors.append(msg); print("[WARN]", msg, flush=True)

def from_hf():
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(HF_REPO, revision=HF_REV, local_dir=str(dest),
                             allow_patterns=["anytext_v1.1.ckpt", "configuration.json", "ms_wrapper.py",
                                             "clip-vit-large-patch14/*"])
    info = {"source": "huggingface", "model_id": HF_REPO, "revision": HF_REV, "minutes": round((time.time() - t0) / 60, 1)}
    # Prefer the official v1.1.3 wrapper if ModelScope answers for 2 tiny files
    try:
        from modelscope.hub.file_download import model_file_download
        tmp = dest / "_ms_small"; tmp.mkdir(exist_ok=True)
        for f in ("ms_wrapper.py", "configuration.json"):
            got = model_file_download("damo/cv_anytext_text_generation_editing", f, revision="v1.1.3", local_dir=str(tmp))
            shutil.copy(got, dest / f)
        info["ms_wrapper"] = "modelscope v1.1.3"
    except Exception as exc:
        info["ms_wrapper"] = f"huggingface mirror (older revision; modelscope said {type(exc).__name__})"
    h = hashlib.sha256()
    with open(Path(path) / "anytext_v1.1.ckpt", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    info["ckpt_sha256"] = h.hexdigest()
    info["ckpt_sha256_matches_known_v1.1"] = h.hexdigest() == CKPT_SHA256
    if not info["ckpt_sha256_matches_known_v1.1"]:
        print(f"[WARN] anytext_v1.1.ckpt sha256 {h.hexdigest()} != known {CKPT_SHA256} (still usable if the doctor edit test passes)")
    finish(path, info)

dest.mkdir(parents=True, exist_ok=True)
if source in ("auto", "modelscope"):
    from_modelscope()
if source in ("auto", "hf"):
    try:
        from_hf()
    except SystemExit:
        raise
    except Exception as exc:
        errors.append(f"huggingface {HF_REPO}: {type(exc).__name__}: {str(exc)[:250]}")
print("[FAIL] could not get the AnyText model:\n  " + "\n  ".join(errors))
print("Way out: download it once in any session that works, then upload the dir as a private Kaggle Dataset "
      "and Add Input -> anytext_loader 'auto' finds it under /kaggle/input.")
sys.exit(1)
PY
