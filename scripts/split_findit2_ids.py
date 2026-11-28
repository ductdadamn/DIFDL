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


def read_rows(findit2_root: Path, split: str) -> list[dict]:
    with open(findit2_root / f"{split}.txt", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("image") or "").strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findit2-root", type=Path, default=ROOT / "findit2")
    ap.add_argument("--tsroie-root", type=Path, default=ROOT / "T-SROIE_test")
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth" / "splits")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if a receipt ID appears in two findit2 splits (default: keep it in the earliest split)")
    args = ap.parse_args()
    args.findit2_root = resolve_findit2_root(args.findit2_root)
    args.tsroie_root = resolve_tsroie_root(args.tsroie_root)
    args.out.mkdir(parents=True, exist_ok=True)

    errors, summary = [], {"findit2_clean": {}, "findit2_stats": {}, "tsroie": {}, "id_conflicts": []}

    # 1. Receipt IDs shared by several findit2 splits (present in the official findit2 split files).
    #    Policy: each ID belongs to its EARLIEST split (train < val < test) and is dropped from later
    #    splits, so every derived dataset (LaMa / AnyText / T-SROIE) stays ID-disjoint.
    rows = {s: read_rows(args.findit2_root, s) for s in SPLITS}
    id2split = {}
    for s in SPLITS:
        for r in rows[s]:
            id2split.setdefault(receipt_id(r["image"]), s)
    for s in SPLITS:
        for r in rows[s]:
            rid = receipt_id(r["image"])
            if id2split[rid] != s:
                summary["id_conflicts"].append({"id": rid, "kept_in": id2split[rid], "dropped_from": s,
                                                "image": r["image"], "forged": r.get("forged")})
    for c in summary["id_conflicts"]:
        print(f"[CONFLICT] {c['id']}: kept in {c['kept_in']}, dropped from {c['dropped_from']} "
              f"({c['image']}, forged={c['forged']})")
    if summary["id_conflicts"] and args.strict:
        errors.append(f"{len(summary['id_conflicts'])} receipt IDs appear in several splits (--strict)")

    # 2. pristine lists (conflicting IDs removed from the later split)
    for s in SPLITS:
        clean, stats = read_findit2_clean(args.findit2_root, s)
        stats["dropped_id_conflict"] = sum(1 for n in clean if id2split.get(receipt_id(n)) != s)
        clean = [n for n in clean if id2split.get(receipt_id(n)) == s]
        (args.out / f"findit2_clean_{s}.txt").write_text("\n".join(clean) + "\n", encoding="utf-8")
        summary["findit2_clean"][s] = len(clean)
        summary["findit2_stats"][s] = stats

    # 3. T-SROIE by receipt-ID prefix
    ts_imgs = sorted((args.tsroie_root / "images").glob("*.jpg")) \
        if args.tsroie_root is not None and args.tsroie_root.exists() else []
    if ts_imgs:
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
        print("[WARN] No T-SROIE images found - T-SROIE split lists not written (needed later for "
              "ds_hybrid / Test Set 1; re-run with --tsroie-root once T-SROIE is uploaded).")

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
