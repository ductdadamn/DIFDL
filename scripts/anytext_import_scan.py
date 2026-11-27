#!/usr/bin/env python3
"""Find every third-party module AnyText inference can import, and which of them are missing.

Static (ast) walk starting from the model's ms_wrapper.py (+ the repo modules it pulls in), following
repo-local imports transitively; top-level AND function-level imports count (AnyText imports some lazily,
e.g. cldm/recognizer.py -> skimage), `if __name__ == "__main__":` blocks are ignored.
Nothing is executed. Run it with the AnyText venv's python so find_spec answers for that env:

  /tmp/DIFDL/.venv_anytext/bin/python scripts/anytext_import_scan.py \
      --anytext-dir third_party/AnyText [--model-dir /tmp/anytext_model]
Exit 1 if any module is missing (prints them + which file imports them). See KAGGLE_RUN_LOG #10.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

# ms_wrapper.py's own repo-level imports (used when the model dir is not available yet)
DEFAULT_ROOTS = ["cldm.model", "cldm.ddim_hacked", "t3_dataset", "util", "lora_util", "bert_tokenizer"]
PACKAGES = ["cldm", "ldm", "ocr_recog"]
# imported only on code paths inference never takes: onnx recognizer (cldm/recognizer.py, lazy),
# translator, super-res degradation for training (ldm/modules/image_degradation), xformers (try/except)
OPTIONAL = {"onnxruntime", "tensorflow", "albumentations", "xformers"}


def imports_of(path: Path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # repo files contain invalid escape sequences
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    skip = set()
    for node in ast.walk(tree):  # drop `if __name__ == "__main__":` bodies
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and \
                isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
            skip.update(id(n) for n in ast.walk(node))
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or ""), node.level


def local_file(repo: Path, mod: str, base: Path | None = None, level: int = 0):
    root = repo if level == 0 else base.parents[level - 1]
    parts = [p for p in mod.split(".") if p]
    for cand in (root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__init__.py")):
        if parts and cand.exists():
            return cand
    return None


def scan(repo: Path, start_files, start_mods):
    stdlib = set(sys.stdlib_module_names)
    todo = list(start_files) + [f for m in start_mods if (f := local_file(repo, m))]
    # Model classes are instantiated from YAML `target:` strings (instantiate_from_config), which a static
    # import walk cannot follow -> also scan every module of the packages those targets live in.
    todo += [f for pkg in PACKAGES for f in sorted((repo / pkg).rglob("*.py"))]
    seen, external = set(), {}
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for mod, level in imports_of(f):
            lf = local_file(repo, mod, f, level) if (mod or level) else None
            if lf is None and level == 0 and mod:
                # `from pkg import sub` where pkg is local: follow pkg
                lf = local_file(repo, mod.split(".")[0])
            if lf is not None or level > 0:
                if lf is not None:
                    todo.append(lf)
                continue
            top = mod.split(".")[0]
            if top and top not in stdlib:
                external.setdefault(top, set()).add(str(f.relative_to(repo) if f.is_relative_to(repo) else f.name))
    return external, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anytext-dir", type=Path, default=Path("/tmp/DIFDL/third_party/AnyText"))
    ap.add_argument("--model-dir", type=Path, default=Path("/tmp/anytext_model"))
    args = ap.parse_args()
    repo = args.anytext_dir.resolve()
    wrapper = args.model_dir / "ms_wrapper.py"
    starts = [wrapper] if wrapper.exists() else []
    external, files = scan(repo, starts, DEFAULT_ROOTS)
    missing = {m: w for m, w in external.items() if importlib.util.find_spec(m) is None}
    print(f"[scan] {len(files)} files from {'ms_wrapper.py' if starts else 'default roots'}; "
          f"{len(external)} third-party modules: {sorted(external)}")
    hard = {m: w for m, w in missing.items() if m not in OPTIONAL}
    for m, w in sorted(missing.items()):
        print(f"  [{'MISSING' if m in hard else 'optional'}] {m}  <- {sorted(w)[:4]}")
    if hard:
        print(f"[FAIL] {len(hard)} missing: {sorted(hard)}")
        sys.exit(1)
    print("[OK] every required import resolves in this env")


if __name__ == "__main__":
    main()
