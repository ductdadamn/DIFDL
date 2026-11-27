#!/usr/bin/env python3
"""Build leakage-free split lists by findit2 receipt ID (CSV tasks #6 + #7 prerequisite).

Outputs (default data_synth/splits/):
  findit2_clean_{train,val,test}.txt   pristine findit2 images (forged == 0), source for LaMa/AnyText
  tsroie_{train,val,test}.txt          T-SROIE_test images assigned by receipt-ID prefix
  split_summary.json                   counts + stats
Fails (exit 1) if any receipt ID appears in two splits or a T-SROIE image has no findit2 ID.

Usage:
  python scripts/split_findit2_ids.py --findit2-root findit2 --tsroie-root T-SROIE_test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from synth_common import SPLITS, read_findit2_clean, receipt_id, resolve_findit2_root, resolve_tsroie_root

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLEAN = {"train": 483, "val": 159, "test": 183}
EXPECTED_TSROIE = {"train": 998, "val": 255, "test": 326}


def all_ids(findit2_root: Path, split: str) -> set[str]:
    with open(findit2_root / f"{split}.txt", newline="", encoding="utf-8") as f:
        return {receipt_id(r["image"]) for r in csv.DictReader(f) if (r.get("image") or "").strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findit2-root", type=Path, default=ROOT / "findit2")
    ap.add_argument("--tsroie-root", type=Path, default=ROOT / "T-SROIE_test")
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth" / "splits")
    args = ap.parse_args()
    args.findit2_root = resolve_findit2_root(args.findit2_root)
    args.tsroie_root = resolve_tsroie_root(args.tsroie_root)
    args.out.mkdir(parents=True, exist_ok=True)

    errors, summary = [], {"findit2_clean": {}, "findit2_stats": {}, "tsroie": {}}

    # 1. findit2 split IDs must be disjoint
    ids = {s: all_ids(args.findit2_root, s) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            inter = ids[a] & ids[b]
            if inter:
                errors.append(f"findit2 {a}/{b} share {len(inter)} IDs, e.g. {sorted(inter)[:3]}")

    # 2. pristine lists
    for s in SPLITS:
        clean, stats = read_findit2_clean(args.findit2_root, s)
        (args.out / f"findit2_clean_{s}.txt").write_text("\n".join(clean) + "\n", encoding="utf-8")
        summary["findit2_clean"][s] = len(clean)
        summary["findit2_stats"][s] = stats

    # 3. T-SROIE by receipt-ID prefix
    ts_imgs = sorted((args.tsroie_root / "images").glob("*.jpg")) \
        if args.tsroie_root is not None and args.tsroie_root.exists() else []
    if ts_imgs:
        id2split = {i: s for s in SPLITS for i in ids[s]}
        buckets, orphans = {s: [] for s in SPLITS}, []
        for p in ts_imgs:
            s = id2split.get(receipt_id(p.name))
            (buckets[s] if s else orphans).append(p.name)
        for s in SPLITS:
            (args.out / f"tsroie_{s}.txt").write_text("\n".join(buckets[s]) + "\n", encoding="utf-8")
            summary["tsroie"][s] = len(buckets[s])
        summary["tsroie"]["orphans"] = len(orphans)
        if orphans:
            errors.append(f"{len(orphans)} T-SROIE images have no findit2 ID, e.g. {orphans[:3]}")
    else:
        print(f"[WARN] No T-SROIE images under {args.tsroie_root}/images — skipped T-SROIE split.")

    (args.out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    for s in SPLITS:
        if summary["findit2_clean"][s] != EXPECTED_CLEAN[s]:
            print(f"[WARN] findit2 clean {s}: {summary['findit2_clean'][s]} (expected {EXPECTED_CLEAN[s]})")
        if ts_imgs and summary["tsroie"][s] != EXPECTED_TSROIE[s]:
            print(f"[WARN] T-SROIE {s}: {summary['tsroie'][s]} (expected {EXPECTED_TSROIE[s]})")
    if errors:
        print("[FAIL] " + "\n[FAIL] ".join(errors))
        sys.exit(1)
    print(f"[OK] Splits are ID-disjoint. Lists written to {args.out}")


if __name__ == "__main__":
    main()
