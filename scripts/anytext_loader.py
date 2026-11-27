"""Single place that builds the AnyText ModelScope pipeline (used by gen_anytext_findit2.py and kaggle_doctor.py).

Facts this relies on (checked against tyxsspa/AnyText @ 3d4a260):
  - `ms_wrapper.py` / `configuration.json` live in the ModelScope MODEL repo, not in the GitHub repo;
    they come with the weights (scripts/download_anytext_model.sh -> /tmp/anytext_model).
  - The wrapper imports the GitHub repo's code (cldm/, ldm/, util.py, ...), so cwd + sys.path must be the repo.
  - demo.py passes model, model_revision, use_fp16, use_translator, font_path (+ optional model_path).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HUB_ID = "damo/cv_anytext_text_generation_editing"
DEFAULT_LOCAL = Path("/tmp/anytext_model")


def resolve_model(model: str) -> tuple[str, bool]:
    """'auto' -> local download dir if complete, else the hub id. Returns (model, is_local)."""
    if model == "auto":
        model = str(DEFAULT_LOCAL) if (DEFAULT_LOCAL / "ms_wrapper.py").exists() else HUB_ID
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
    from modelscope.pipelines import pipeline

    kw = {"model": model, "use_fp16": fp16, "use_translator": False, "font_path": str(font)}
    if not is_local:
        kw["model_revision"] = revision
    print(f"[AnyText] pipeline(model={model}, local={is_local}, fp16={fp16}, font={font.name})", flush=True)
    return pipeline("my-anytext-task", **kw)
