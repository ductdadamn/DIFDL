#!/usr/bin/env python3
"""Pick edit regions on pristine findit2 receipts and write generation manifests.

Stage 1 (OCR cache): one JSON per receipt in --ocr-cache with [{"box":[x1,y1,x2,y2],"text":..,"score":..}]
  --ocr-source paddle : PaddleOCR 3.x det+rec (CPU is fine, ~1-3 s/receipt)
  --ocr-source sroie  : SROIE task-1 box files (<stem>.txt, lines "x1,y1,...,x4,y4,TEXT")
  --ocr-source cache  : only reuse existing cache files
Stage 2 (selection, deterministic for a given --seed):
  manifest_lama.jsonl    : Removal on train/val/test (1-2 text lines per 512 crop)
  manifest_anytext.jsonl : Replacement (digits changed, same format) + Insertion (blank paper) on test only
  selection_stats.json   : counts and rejection reasons (small-text rate -> Limitations section)

Usage:
  python scripts/select_edit_regions.py --findit2-root findit2 --ocr-source paddle
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from synth_common import (CROP, SPLITS, boxes_to_mask, choose_crop, dilate, load_split_lists,
                          resolve_findit2_root, stable_seed, union_box, write_jsonl)

ROOT = Path(__file__).resolve().parents[1]
DIGIT = re.compile(r"\d")


# ----------------------------------------------------------------------------- OCR
def ocr_paddle_factory(det_model="PP-OCRv5_server_det", rec_model="PP-OCRv5_mobile_rec"):
    # Pin model names explicitly: letting paddleocr pick defaults pulled PP-OCRv6 models that the
    # pinned paddle runtime cannot load. det = same model as src/seg_char.py.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(text_detection_model_name=det_model, text_recognition_model_name=rec_model,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)

    def _get(res, key):
        try:
            return res[key]
        except Exception:
            return res.json["res"][key]

    def run(img_path: Path):
        items = []
        for res in ocr.predict(str(img_path)):
            texts, scores = _get(res, "rec_texts"), _get(res, "rec_scores")
            try:
                boxes = [list(map(int, b)) for b in _get(res, "rec_boxes")]
            except Exception:
                polys = _get(res, "rec_polys")
                boxes = [[int(min(p[:, 0])), int(min(p[:, 1])), int(max(p[:, 0])), int(max(p[:, 1]))]
                         for p in map(np.asarray, polys)]
            items += [{"box": b, "text": str(t), "score": float(s)} for b, t, s in zip(boxes, texts, scores)]
        return items

    return run


def ocr_sroie_factory(box_dir: Path):
    def run(img_path: Path):
        p = box_dir / (img_path.stem + ".txt")
        items = []
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = ln.strip().split(",", 8)
            if len(parts) < 9:
                continue
            xy = list(map(int, parts[:8]))
            xs, ys = xy[0::2], xy[1::2]
            items.append({"box": [min(xs), min(ys), max(xs), max(ys)], "text": parts[8], "score": 1.0})
        return items

    return run


def build_cache(args, names_by_split):
    args.ocr_cache.mkdir(parents=True, exist_ok=True)
    runner = None
    todo = [(s, n) for s in SPLITS for n in names_by_split[s]
            if not (args.ocr_cache / (Path(n).stem + ".json")).exists()]
    if todo and args.ocr_source == "cache":
        raise SystemExit(f"[ERROR] {len(todo)} receipts have no OCR cache and --ocr-source=cache")
    if todo:
        runner = ocr_paddle_factory(args.det_model, args.rec_model) if args.ocr_source == "paddle" \
            else ocr_sroie_factory(args.sroie_box_dir)
        print(f"[OCR] Running {args.ocr_source} on {len(todo)} receipts...")
    for i, (s, n) in enumerate(todo, 1):
        items = runner(args.findit2_root / s / n)
        (args.ocr_cache / (Path(n).stem + ".json")).write_text(json.dumps(items), encoding="utf-8")
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")


# ----------------------------------------------------------------------------- text helpers
def mutate_digits(text: str, rng: np.random.Generator) -> str:
    """Change digits but keep format (12.50 -> 17.90); guarantees a different string."""
    chars = list(text)
    pos = [i for i, c in enumerate(chars) if c.isdigit()]
    for _ in range(10):
        new = chars[:]
        k = max(1, int(round(len(pos) * 0.5)))
        for i in rng.choice(pos, size=min(k, len(pos)), replace=False):
            lo = 1 if (i == pos[0] and chars[i] != "0" and len(pos) > 1) else 0
            new[i] = str(int(rng.integers(lo, 10)))
        if new != chars:
            return "".join(new)
    i = pos[-1]
    chars[i] = str((int(chars[i]) + 1) % 10)
    return "".join(chars)


def random_receipt_token(rng: np.random.Generator) -> str:
    a, b = int(rng.integers(1, 200)), int(rng.integers(0, 100))
    d, m, y = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.integers(2016, 2019))
    choices = [f"{a}.{b:02d}", f"RM {a}.{b:02d}", f"TOTAL {a}.{b:02d}", f"DISC -{b // 10}.{b:02d}",
               f"{d:02d}/{m:02d}/{y}", f"QTY {int(rng.integers(1, 9))}", f"TIP {b // 10}.00"]
    return choices[int(rng.integers(0, len(choices)))]


# ----------------------------------------------------------------------------- selection
class Receipt:
    def __init__(self, split, name, items, size, args):
        self.split, self.name, self.stem = split, name, Path(name).stem
        self.w, self.h = size
        self.items = items
        self.used: set[int] = set()
        self.k = 0
        self.args = args
        self._gray = None
        hs = [it["box"][3] - it["box"][1] for it in items]
        self.text_mask = dilate(boxes_to_mask(self.h, self.w, [it["box"] for it in items]), 6)
        self.median_h = int(np.median(hs)) if hs else 20

    def gray(self):
        if self._gray is None:
            with Image.open(self.args.findit2_root / self.split / self.name) as im:
                self._gray = np.array(im.convert("L"), dtype=np.float32)
        return self._gray

    def next_id(self, gen, op):
        self.k += 1
        return f"{gen}_{op[:3]}_{self.split}_{self.stem}_{self.k:02d}"


def line_ok(it, min_h=0) -> bool:
    x1, y1, x2, y2 = it["box"]
    w, h = x2 - x1, y2 - y1
    return it["score"] >= 0.7 and 10 <= w <= CROP - 48 and max(8, min_h) <= h <= 80


def candidates(rc: Receipt, rule: str, rng):
    out = []
    for i, it in enumerate(rc.items):
        if i in rc.used:
            continue
        if rule == "replace":
            if DIGIT.search(it["text"]) and len(it["text"]) <= 20 and line_ok(it, rc.args.min_text_h):
                out.append(i)
        elif line_ok(it):
            out.append(i)
    rng.shuffle(out)
    if rule == "remove":  # digit lines first (amounts/dates), random order within each group
        out.sort(key=lambda i: 0 if DIGIT.search(rc.items[i]["text"]) else 1)
    return out


def make_removal(rc: Receipt, rng, stats):
    cands = candidates(rc, "remove", rng)
    if not cands:
        return None
    first = cands[0] if rng.random() < 0.7 else int(rng.choice(cands))
    boxes, idx = [rc.items[first]["box"]], [first]
    if rng.random() < 0.3:  # optionally a second nearby line in the same crop
        for j in cands:
            if j != first:
                u = union_box(boxes + [rc.items[j]["box"]])
                if u[2] - u[0] <= CROP - 48 and u[3] - u[1] <= CROP - 48:
                    boxes.append(rc.items[j]["box"]); idx.append(j)
                    break
    crop = choose_crop(union_box(boxes), rc.w, rc.h, rng)
    if crop is None:
        rc.used.update(idx); stats["remove:reject_no_crop"] += 1
        return None
    rc.used.update(idx)
    return {"op": "removal", "boxes": boxes, "orig_text": [rc.items[i]["text"] for i in idx],
            "new_text": "", "crop": crop}


def make_replacement(rc: Receipt, rng, stats):
    cands = candidates(rc, "replace", rng)
    if not cands:
        return None
    i = int(rng.choice(cands))
    rc.used.add(i)
    box = rc.items[i]["box"]
    crop = choose_crop(box, rc.w, rc.h, rng)
    if crop is None:
        stats["replace:reject_no_crop"] += 1
        return None
    return {"op": "replacement", "boxes": [box], "orig_text": [rc.items[i]["text"]],
            "new_text": mutate_digits(rc.items[i]["text"], rng), "crop": crop}


def make_insertion(rc: Receipt, rng, stats):
    text = random_receipt_token(rng)
    th = int(np.clip(rc.median_h, rc.args.min_text_h, 40))
    tw = int(min(CROP - 64, 0.6 * th * len(text) + 8))
    g = rc.gray()
    for _ in range(60):
        x1 = int(rng.integers(8, max(9, rc.w - tw - 8)))
        y1 = int(rng.integers(8, max(9, rc.h - th - 8)))
        box = [x1, y1, x1 + tw, y1 + th]
        if box[2] > rc.w - 8 or box[3] > rc.h - 8:
            continue
        if rc.text_mask[y1:box[3], x1:box[2]].any():
            continue
        patch = g[y1:box[3], x1:box[2]]
        if patch.mean() < 200 or patch.std() > 12:  # must be blank, bright paper
            continue
        crop = choose_crop(box, rc.w, rc.h, rng)
        if crop is None:
            continue
        rc.text_mask[max(0, y1 - 6):box[3] + 6, max(0, x1 - 6):box[2] + 6] = 1  # don't reuse spot
        return {"op": "insertion", "boxes": [box], "orig_text": [], "new_text": text, "crop": crop}
    stats["insert:reject_no_blank_area"] += 1
    return None


def fill_quota(receipts, gen, maker, quota, rng, stats, args):
    rows, order = [], list(range(len(receipts)))
    rng.shuffle(order)
    while len(rows) < quota:
        progress = False
        for r in order:
            if len(rows) >= quota:
                break
            rc = receipts[r]
            res = maker(rc, rng, stats)
            if res is None:
                continue
            progress = True
            sid = rc.next_id(gen, res["op"])
            rows.append({"id": sid, "gen": gen, "split": rc.split, "src": f"{rc.split}/{rc.name}",
                         "img_size": [rc.w, rc.h], "mask_dilate": args.mask_dilate,
                         "seed": stable_seed(args.seed, sid), **res})
        if not progress:
            print(f"[WARN] {gen}/{maker.__name__}: candidates exhausted at {len(rows)}/{quota}")
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findit2-root", type=Path, default=ROOT / "findit2")
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data_synth" / "splits")
    ap.add_argument("--ocr-cache", type=Path, default=ROOT / "data_synth" / "ocr_cache")
    ap.add_argument("--ocr-source", choices=["paddle", "sroie", "cache"], default="paddle")
    ap.add_argument("--sroie-box-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--lama", type=int, nargs=3, default=[1000, 200, 200], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--anytext", type=int, default=250, help="AnyText samples on findit2/test")
    ap.add_argument("--insert-ratio", type=float, default=0.3, help="share of AnyText samples that are Insertion")
    ap.add_argument("--min-text-h", type=int, default=14, help="min text-line height (px) for AnyText")
    ap.add_argument("--mask-dilate", type=int, default=3, help="GT mask growth around OCR boxes (px)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--det-model", default="PP-OCRv5_server_det")
    ap.add_argument("--rec-model", default="PP-OCRv5_mobile_rec")
    ap.add_argument("--ocr-test", type=int, default=0, metavar="N",
                    help="smoke test: OCR the first N test receipts, print lines + timing, write nothing, exit")
    args = ap.parse_args()
    args.findit2_root = resolve_findit2_root(args.findit2_root)
    if args.ocr_source == "sroie" and args.sroie_box_dir is None:
        ap.error("--sroie-box-dir is required with --ocr-source sroie")

    names = load_split_lists(args.splits_dir)
    if args.ocr_test:
        import time
        run = ocr_paddle_factory(args.det_model, args.rec_model) if args.ocr_source == "paddle" \
            else ocr_sroie_factory(args.sroie_box_dir)
        for n in names["test"][:args.ocr_test]:
            t0 = time.time()
            items = run(args.findit2_root / "test" / n)
            hs = sorted(it["box"][3] - it["box"][1] for it in items)
            print(f"{n}: {len(items)} lines in {time.time() - t0:.1f}s | median line height "
                  f"{hs[len(hs) // 2] if hs else 0}px")
            for it in items[:6]:
                print("   ", it)
        return
    build_cache(args, names)

    receipts = {s: [] for s in SPLITS}
    stats = Counter()
    for s in SPLITS:
        for n in names[s]:
            items = json.loads((args.ocr_cache / (Path(n).stem + ".json")).read_text(encoding="utf-8"))
            with Image.open(args.findit2_root / s / n) as im:
                size = im.size
            if min(size) < CROP:
                stats[f"{s}:skip_receipt_smaller_than_{CROP}"] += 1
                continue
            receipts[s].append(Receipt(s, n, items, size, args))

    rng = np.random.default_rng(args.seed)
    lama_rows = []
    for s, q in zip(SPLITS, args.lama):
        lama_rows += fill_quota(receipts[s], "lama", make_removal, q, rng, stats, args)

    n_ins = int(round(args.anytext * args.insert_ratio))
    any_rows = fill_quota(receipts["test"], "anytext", make_replacement, args.anytext - n_ins, rng, stats, args)
    any_rows += fill_quota(receipts["test"], "anytext", make_insertion, n_ins, rng, stats, args)

    write_jsonl(lama_rows, args.out / "manifest_lama.jsonl")
    write_jsonl(any_rows, args.out / "manifest_anytext.jsonl")

    counts = defaultdict(int)
    for r in lama_rows + any_rows:
        counts[f"{r['gen']}/{r['op']}/{r['split']}"] += 1
    n_ocr_lines = sum(len(rc.items) for s in SPLITS for rc in receipts[s])
    # Limitations metric: share of digit lines on test too small for AnyText (< --min-text-h px)
    digit_lines = [it for rc in receipts["test"] for it in rc.items
                   if DIGIT.search(it["text"]) and line_ok(it)]
    small = sum(1 for it in digit_lines if it["box"][3] - it["box"][1] < args.min_text_h)
    summary = {"counts": dict(counts), "receipts": {s: len(receipts[s]) for s in SPLITS},
               "ocr_lines_total": n_ocr_lines,
               "test_digit_lines": len(digit_lines),
               "test_digit_lines_too_small_for_anytext": small,
               "test_small_text_rate": round(small / max(1, len(digit_lines)), 4),
               "rejections": dict(stats), "seed": args.seed}
    (args.out / "selection_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
