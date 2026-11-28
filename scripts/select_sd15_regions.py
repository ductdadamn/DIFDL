#!/usr/bin/env python3
"""Pick SD-1.5 Inpainting Removal regions (Seen AI, 500/100/200) WITHOUT re-running notebook 01.

Reads the released notebook-01 output (ocr_cache/, splits/, manifest_lama.jsonl, manifest_anytext.jsonl):
  train : NEW regions (other text lines than LaMa/AnyText on the same receipts, separate --seed), so SD and
          LaMa train samples do not repeat the same crop/mask; <= --max-per-receipt crops per receipt.
  val   : --sd15[1] LaMa val entries (random subset, fixed seed)  -> paired with LaMa
  test  : ALL LaMa test entries (same receipt, crop and mask)     -> paired controlled benchmark
          (only the eraser differs: LaMa = CNN vs SD-1.5 = latent diffusion)
Paired rows keep `pair_id` = the LaMa id. GT mask policy is identical to LaMa (text-line boxes + mask_dilate px).
Writes <out>/manifest_sd15.jsonl + <out>/sd15_selection_stats.json and asserts the invariants above.

Kaggle (CPU):  python scripts/select_sd15_regions.py --findit2-root auto --prep auto --out /kaggle/working/data_synth
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from select_edit_regions import Receipt, fill_quota, make_removal
from synth_common import (CROP, KAGGLE_INPUT, SPLITS, load_split_lists, read_jsonl, receipt_id,
                          resolve_findit2_root, stable_seed, write_jsonl)

ROOT = Path(__file__).resolve().parents[1]


def resolve_prep(p) -> Path:
    """Directory holding manifest_lama.jsonl + ocr_cache/ + splits/ (notebook-01 output)."""
    if str(p) != "auto":
        return Path(p)
    hits = sorted({m.parent for m in KAGGLE_INPUT.rglob("manifest_lama.jsonl")
                   if (m.parent / "ocr_cache").is_dir() and (m.parent / "splits").is_dir()})
    if len(hits) != 1:
        raise SystemExit(f"[ERROR] expected one notebook-01 output (manifest_lama.jsonl + ocr_cache/ + splits/) "
                         f"under {KAGGLE_INPUT}, got {hits}. Pass --prep explicitly.")
    print(f"[auto] notebook-01 output: {hits[0]}")
    return hits[0]


def paired(lama_rows, split, n, rng, seed):
    rows = sorted((r for r in lama_rows if r["split"] == split), key=lambda r: r["id"])
    if n < len(rows):
        keep = sorted(rng.choice(len(rows), size=n, replace=False))
        rows = [rows[i] for i in keep]
    out = []
    for r in rows:
        sid = "sd15_" + r["id"].split("_", 1)[1]
        out.append({**r, "id": sid, "gen": "sd15", "pair_id": r["id"], "seed": stable_seed(seed, sid)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findit2-root", type=Path, default=ROOT / "findit2")
    ap.add_argument("--prep", default=str(ROOT / "data_synth"), help="notebook-01 output dir, or 'auto'")
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--sd15", type=int, nargs=3, default=[500, 100, 200], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--max-per-receipt", type=int, default=2, help="train crops per receipt (483 receipts < 500)")
    ap.add_argument("--seed", type=int, default=123, help="different from notebook 01 (42) -> other regions")
    args = ap.parse_args()
    args.findit2_root = resolve_findit2_root(args.findit2_root)
    prep = resolve_prep(args.prep)

    lama_rows = read_jsonl(prep / "manifest_lama.jsonl")
    any_path = prep / "manifest_anytext.jsonl"
    any_rows = read_jsonl(any_path) if any_path.exists() else []
    dil = {r["mask_dilate"] for r in lama_rows}
    assert len(dil) == 1, f"LaMa manifest mixes mask_dilate values {dil}"
    args.mask_dilate = dil.pop()
    args.min_text_h = 14  # only used by Replacement candidates (not here); Receipt expects it

    rng = np.random.default_rng(args.seed)
    stats = Counter()

    # ---------------- train: new regions on the same pristine receipts, avoiding every line already edited
    used_boxes = defaultdict(set)  # "train/<name>" -> {tuple(box)} used by LaMa or AnyText
    for r in lama_rows + any_rows:
        for b in r["boxes"]:
            used_boxes[r["src"]].add(tuple(b))
    names = load_split_lists(prep / "splits")
    receipts = []
    for n in names["train"]:
        items = json.loads((prep / "ocr_cache" / (Path(n).stem + ".json")).read_text(encoding="utf-8"))
        with Image.open(args.findit2_root / "train" / n) as im:
            size = im.size
        if min(size) < CROP:
            stats["train:skip_receipt_smaller_than_512"] += 1
            continue
        rc = Receipt("train", n, items, size, args)
        taken = used_boxes.get(f"train/{n}", set())
        rc.used = {i for i, it in enumerate(items) if tuple(it["box"]) in taken}
        stats["train:lines_excluded_used_by_lama_anytext"] += len(rc.used)
        receipts.append(rc)

    def capped_removal(rc, rng_, st):
        if rc.k >= args.max_per_receipt:
            return None
        return make_removal(rc, rng_, st)
    capped_removal.__name__ = "make_removal"

    train_rows = fill_quota(receipts, "sd15", capped_removal, args.sd15[0], rng, stats, args)
    val_rows = paired(lama_rows, "val", args.sd15[1], rng, args.seed)
    test_rows = paired(lama_rows, "test", args.sd15[2], rng, args.seed)
    rows = train_rows + val_rows + test_rows

    # ---------------- invariants
    for r in train_rows:
        clash = used_boxes.get(r["src"], set()) & {tuple(b) for b in r["boxes"]}
        assert not clash, f"{r['id']}: train region reuses a LaMa/AnyText line {clash}"
    ids_by_split = {s: {receipt_id(r["src"].split("/", 1)[1]) for r in rows if r["split"] == s} for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not ids_by_split[a] & ids_by_split[b], f"receipt IDs in both {a} and {b}"
    lama_test = {r["id"]: r for r in lama_rows if r["split"] == "test"}
    assert len(test_rows) == len(lama_test) or args.sd15[2] < len(lama_test), "test not fully paired"
    for r in val_rows + test_rows:
        l = {k: v for k, v in next(x for x in lama_rows if x["id"] == r["pair_id"]).items()}
        for k in ("src", "crop", "boxes", "mask_dilate", "split", "op"):
            assert r[k] == l[k], f"{r['id']}: '{k}' differs from its LaMa pair"
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"

    write_jsonl(rows, args.out / "manifest_sd15.jsonl")
    per_receipt = Counter(r["src"] for r in train_rows)
    summary = {"counts": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
               "train_receipts_used": len(per_receipt),
               "train_crops_per_receipt": dict(sorted(Counter(per_receipt.values()).items())),
               "paired_with_lama": {"val": len(val_rows), "test": len(test_rows)},
               "mask_dilate": args.mask_dilate, "seed": args.seed, "max_per_receipt": args.max_per_receipt,
               "rejections": dict(stats)}
    if summary["counts"]["train"] < args.sd15[0]:
        print(f"[WARN] train quota short: {summary['counts']['train']}/{args.sd15[0]} "
              f"(raise --max-per-receipt to allow more crops per receipt)")
    (args.out / "sd15_selection_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] {len(rows)} rows -> {args.out / 'manifest_sd15.jsonl'} (invariants checked)")


if __name__ == "__main__":
    main()
