"""Shared helpers for findit2 AI-synthesis scripts (LaMa / AnyText).

Only depends on numpy + Pillow so it runs on a CPU laptop, on Kaggle, and inside
the isolated AnyText venv (numpy 1.23 / Pillow 9.5).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SPLITS = ("train", "val", "test")
# Known malformed row in findit2/train.txt (see AGENTS.md section 3).
BAD_ROWS = {"X51008114260.png"}
CROP = 512
ALIGN = 8


# ----------------------------------------------------------------------------- data roots
KAGGLE_INPUT = Path("/kaggle/input")


def _find_unique(base: Path, pred, what: str, max_depth: int = 5) -> Path:
    hits, stack = [], [(Path(base), 0)]
    while stack:
        d, depth = stack.pop()
        try:
            if pred(d):
                hits.append(d)
                continue
            if depth < max_depth:
                stack += [(c, depth + 1) for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")]
        except (PermissionError, OSError):
            continue
    if not hits:
        raise SystemExit(f"[ERROR] Could not find {what} under {base}. Pass the path explicitly.")
    if len(hits) > 1:
        raise SystemExit(f"[ERROR] Found several {what} under {base}: {sorted(map(str, hits))}. Pass one explicitly.")
    print(f"[auto] {what}: {hits[0]}")
    return hits[0]


def _is_findit2(d: Path) -> bool:
    return all((d / f"{s}.txt").is_file() for s in SPLITS) and (d / "train").is_dir()


def _is_tsroie(d: Path) -> bool:
    return (d / "images").is_dir() and (d / "masks").is_dir() and "sroie" in d.name.lower() \
        and any((d / "images").glob("*.jpg"))


def resolve_findit2_root(p, base: Path = KAGGLE_INPUT) -> Path:
    """`auto` -> search `base` (default /kaggle/input) for the findit2 root, else return the path as given."""
    return _find_unique(base, _is_findit2, "findit2 root (train/val/test.txt + train/)") if str(p) == "auto" else Path(p)


def resolve_tsroie_root(p, base: Path = KAGGLE_INPUT):
    """`auto` -> search for a *SROIE* dir with images/ and masks/; returns None if absent (T-SROIE is optional)."""
    if str(p) != "auto":
        return Path(p)
    try:
        return _find_unique(base, _is_tsroie, "T-SROIE root (images/*.jpg + masks/)")
    except SystemExit as exc:
        if "Could not find" in str(exc):
            print("[auto] T-SROIE not found - T-SROIE split will be skipped")
            return None
        raise


# ----------------------------------------------------------------------------- splits
def read_findit2_clean(findit2_root: Path, split: str) -> tuple[list[str], dict]:
    """Return sorted pristine image names (forged == 0) of one findit2 split + stats."""
    txt = Path(findit2_root) / f"{split}.txt"
    clean, stats = [], {"rows": 0, "forged": 0, "bad_rows": 0, "missing_files": 0}
    with open(txt, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats["rows"] += 1
            name = (row.get("image") or "").strip()
            forged = (row.get("forged") or "").strip()
            if not name or name in BAD_ROWS or forged not in ("0", "1"):
                stats["bad_rows"] += 1
                continue
            if forged == "1":
                stats["forged"] += 1
                continue
            if not (Path(findit2_root) / split / name).exists():
                stats["missing_files"] += 1
                continue
            clean.append(name)
    return sorted(set(clean)), stats


def receipt_id(name: str) -> str:
    """Receipt ID shared by findit2 (X51005....png) and T-SROIE (X51005..._k.jpg)."""
    return Path(name).stem.split("_")[0]


def load_split_lists(splits_dir: Path) -> dict[str, list[str]]:
    out = {}
    for s in SPLITS:
        p = Path(splits_dir) / f"findit2_clean_{s}.txt"
        out[s] = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


# ----------------------------------------------------------------------------- masks
def boxes_to_mask(h: int, w: int, boxes, dilate: int = 0) -> np.ndarray:
    """Binary uint8 {0,1} mask of axis-aligned boxes [x1,y1,x2,y2], optionally grown by `dilate` px."""
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1) - dilate), max(0, int(y1) - dilate)
        x2, y2 = min(w, int(x2) + dilate), min(h, int(y2) + dilate)
        m[y1:y2, x1:x2] = 1
    return m


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Square (Chebyshev) dilation of a {0,1} mask by `px` pixels."""
    if px <= 0:
        return mask.copy()
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.array(img.filter(ImageFilter.MaxFilter(2 * px + 1))) > 0).astype(np.uint8)


def soft_blend_mask(gt: np.ndarray, grow: int = 3, sigma: float = 3.0) -> np.ndarray:
    """Float32 alpha in [0,1]: 1 inside GT, Gaussian ~3px falloff outside, exact 0 far away."""
    grown = dilate(gt, grow).astype(np.float32) * 255.0
    blurred = np.array(Image.fromarray(grown.astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma)),
                       dtype=np.float32) / 255.0
    alpha = np.maximum(blurred, gt.astype(np.float32))
    alpha[alpha < 1.0 / 255.0] = 0.0
    return alpha


def blend(ai: np.ndarray, orig: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """I_out = alpha * I_AI + (1 - alpha) * I_orig (uint8 RGB)."""
    a = alpha[..., None]
    out = a * ai.astype(np.float32) + (1.0 - a) * orig.astype(np.float32)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    out[alpha == 0] = orig[alpha == 0]  # guarantee bit-exact background
    return out


# ----------------------------------------------------------------------------- geometry
def union_box(boxes):
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _aligned_range(lo: int, hi: int, align: int = ALIGN):
    lo_a = -(-lo // align) * align  # ceil to multiple
    hi_a = (hi // align) * align
    return list(range(lo_a, hi_a + 1, align)) if lo_a <= hi_a else []


def choose_crop(box, img_w: int, img_h: int, rng: np.random.Generator,
                size: int = CROP, margin: int = 16):
    """Random size x size window (top-left multiple of 8) containing `box` with `margin` px slack."""
    if img_w < size or img_h < size:
        return None
    x1, y1, x2, y2 = box
    xs = _aligned_range(max(0, x2 + margin - size), min(x1 - margin, img_w - size))
    ys = _aligned_range(max(0, y2 + margin - size), min(y1 - margin, img_h - size))
    if not xs or not ys:
        return None
    x0, y0 = int(rng.choice(xs)), int(rng.choice(ys))
    return [x0, y0, x0 + size, y0 + size]


def to_crop_coords(boxes, crop):
    x0, y0 = crop[0], crop[1]
    return [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0] for b in boxes]


# ----------------------------------------------------------------------------- io
def stable_seed(*parts) -> int:
    h = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def save_mask(mask: np.ndarray, path: Path):
    save_png((mask > 0).astype(np.uint8) * 255, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_paths(out_root: Path, entry: dict) -> dict[str, Path]:
    """Output layout compatible with src/build_path_pkl.py: <gen>/<split>/{images,masks,pristine}/<id>.png"""
    base = Path(out_root) / entry["gen"] / entry["split"]
    name = entry["id"] + ".png"
    return {k: base / k / name for k in ("images", "masks", "pristine")}
