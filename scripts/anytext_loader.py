"""Single place that builds the AnyText ModelScope pipeline (used by gen_anytext_findit2.py and kaggle_doctor.py).

Facts this relies on (checked against tyxsspa/AnyText @ 3d4a260):
  - `ms_wrapper.py` / `configuration.json` live in the ModelScope MODEL repo, not in the GitHub repo;
    they come with the weights (scripts/download_anytext_model.sh -> /tmp/anytext_model,
    or a complete copy under /kaggle/input). 'auto' never falls back to the hub.
  - The wrapper imports the GitHub repo's code (cldm/, ldm/, util.py, ...), so cwd + sys.path must be the repo.
  - demo.py passes model, model_revision, use_fp16, use_translator, font_path (+ optional model_path).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HUB_ID = "damo/cv_anytext_text_generation_editing"
DEFAULT_LOCAL = Path("/tmp/anytext_model")
REQUIRED = ("configuration.json", "ms_wrapper.py", "anytext_v1.1.ckpt", "clip-vit-large-patch14/pytorch_model.bin")


def is_complete(d: Path) -> bool:
    return all((d / f).exists() for f in REQUIRED)


def find_kaggle_input_copy(root: Path = Path("/kaggle/input"), max_depth: int = 6):
    """A model dir uploaded as a Kaggle Dataset / notebook output (no download needed)."""
    if not root.exists():
        return None
    for p in sorted(root.rglob("ms_wrapper.py")):
        if len(p.relative_to(root).parts) <= max_depth and is_complete(p.parent):
            return p.parent
    return None


def resolve_model(model: str) -> tuple[str, bool]:
    """'auto' -> /tmp/anytext_model if complete, else a complete copy under /kaggle/input, else ERROR.
    'input' -> only a copy under /kaggle/input.
    (No silent fallback to the hub: modelscope 1.10 cannot download from today's hub and the hub
    rate-limits Kaggle IPs -> confusing secondary errors, KAGGLE_RUN_LOG #9.) Returns (model, is_local)."""
    if model == "input":  # only a copy under /kaggle/input (read-only) -> doctor's read-only test
        found = find_kaggle_input_copy()
        if found is None:
            raise FileNotFoundError("no complete AnyText model dir under /kaggle/input "
                                    "(Add Input -> the notebook/dataset that holds anytext_model/)")
        model = str(found)
    if model == "auto":
        found = DEFAULT_LOCAL if is_complete(DEFAULT_LOCAL) else find_kaggle_input_copy()
        if found is None:
            raise FileNotFoundError(
                f"no complete AnyText model dir ({', '.join(REQUIRED)}) in {DEFAULT_LOCAL} or /kaggle/input; "
                f"run scripts/download_anytext_model.sh (or pass --model <dir> / the hub id explicitly)")
        model = str(found)
    p = Path(model)
    if p.exists():
        missing = [f for f in ("configuration.json", "ms_wrapper.py") if not (p / f).exists()]
        if missing:
            raise FileNotFoundError(f"{p} is not a complete AnyText model dir (missing {missing}); "
                                    f"run scripts/download_anytext_model.sh")
        return str(p.resolve()), True
    return model, False


def load_pipeline(anytext_dir, model="auto", revision="v1.1.3", fp16=True, font=None):
    anytext_dir = Path(anytext_dir).resolve()
    font = Path(font) if font else anytext_dir / "font" / "Arial_Unicode.ttf"
    if not font.exists():
        raise FileNotFoundError(f"missing font {font} (any TTF works; the notebook copies DejaVuSans there)")
    model, is_local = resolve_model(model)
    os.chdir(anytext_dir)
    if str(anytext_dir) not in sys.path:
        sys.path.insert(0, str(anytext_dir))
    # the CLIP dir holds the full CLIP (text+vision); AnyText only builds CLIPTextModel -> transformers prints
    # ~400 "weights not used" names. Expected and harmless; keep the log readable.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    from modelscope.pipelines import pipeline

    kw = {"model": model, "use_fp16": fp16, "use_translator": False, "font_path": str(font)}
    if not is_local:
        kw["model_revision"] = revision
    print(f"[AnyText] pipeline(model={model}, local={is_local}, fp16={fp16}, font={font.name})", flush=True)
    return pipeline("my-anytext-task", **kw)
