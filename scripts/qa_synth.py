#!/usr/bin/env python3
"""Automatic QA + contact sheet for generated findit2 samples (LaMa / AnyText).

Checks per manifest entry:
  - images/, masks/, pristine/ files exist, are 512x512, mask non-empty
  - tampered vs pristine identical outside the (grown) GT mask   -> GT is trustworthy
  - something actually changed inside the mask                    -> generator did work
Global checks: no receipt ID in two splits, AnyText only on test, quota per gen/split.
Writes <out>/qa/<gen>_report.json and <gen>_contact_sheet.png (pristine | tampered | mask overlay | |diff|x4).

Usage: python scripts/qa_synth.py --gen lama        (exit 1 on any hard failure)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from synth_common import CROP, dilate, read_jsonl, receipt_id, sample_paths

ROOT = Path(__file__).resolve().parents[1]
# outside-band tolerance: LaMa is copied back bit-exact; AnyText has a Gaussian band (grow 3 + ~3 sigma)
OUTSIDE_GROW = {"lama": 0, "anytext": 14}


def load(p: Path, mode="RGB"):
    with Image.open(p) as im:
        return np.array(im.convert(mode))


def contact_sheet(rows, path: Path, cell=256):
    tiles = []
    for pr, tp, m in rows:
        ov = tp.copy()
        ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        diff = np.clip(np.abs(tp.astype(int) - pr.astype(int)) * 4, 0, 255).astype(np.uint8)
        tiles.append(np.hstack([np.array(Image.fromarray(x).resize((cell, cell))) for x in (pr, tp, ov, diff)]))
    if tiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.vstack(tiles)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", choices=["lama", "anytext"], required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--sheet-n", type=int, default=24, help="rows in the contact sheet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    manifest = args.manifest or args.out / f"manifest_{args.gen}.jsonl"
    entries = [e for e in read_jsonl(manifest) if e["gen"] == args.gen]

    hard, soft = defaultdict(list), defaultdict(list)
    split_ids = defaultdict(set)
    done = Counter()
    ok_entries = []

    for e in entries:
        split_ids[e["split"]].add(receipt_id(e["src"].split("/")[-1]))
        p = sample_paths(args.out, e)
        if not all(x.exists() for x in p.values()):
            soft["missing_outputs"].append(e["id"])
            continue
        tp, pr, m = load(p["images"]), load(p["pristine"]), load(p["masks"], "L") > 127
        if tp.shape[:2] != (CROP, CROP) or pr.shape != tp.shape or m.shape != tp.shape[:2]:
            hard["bad_size"].append(e["id"]); continue
        if not m.any():
            hard["empty_mask"].append(e["id"]); continue
        outside = dilate(m.astype(np.uint8), OUTSIDE_GROW[args.gen]) == 0
        max_out = int(np.abs(tp.astype(int) - pr.astype(int))[outside].max(initial=0))
        if max_out > 0:
            hard["changed_outside_mask"].append((e["id"], max_out))
        mean_in = float(np.abs(tp.astype(int) - pr.astype(int))[m].mean())
        if mean_in < 2.0:
            soft["almost_no_change_inside_mask"].append((e["id"], round(mean_in, 2)))
        done[e["split"]] += 1
        ok_entries.append((e, p))

    splits = list(split_ids)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            if split_ids[a] & split_ids[b]:
                hard["receipt_id_leakage"].append(f"{a}/{b}: {sorted(split_ids[a] & split_ids[b])[:5]}")
    if args.gen == "anytext" and any(e["split"] != "test" for e in entries):
        hard["anytext_outside_test"].append("manifest contains non-test AnyText rows")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(ok_entries), size=min(args.sheet_n, len(ok_entries)), replace=False) if ok_entries else []
    rows = [(load(ok_entries[i][1]["pristine"]), load(ok_entries[i][1]["images"]),
             load(ok_entries[i][1]["masks"], "L")) for i in sorted(pick)]
    sheet = args.out / "qa" / f"{args.gen}_contact_sheet.png"
    contact_sheet(rows, sheet)

    report = {"gen": args.gen, "manifest_entries": len(entries),
              "manifest_per_split": dict(Counter(e["split"] for e in entries)),
              "generated_ok_per_split": dict(done),
              "hard_failures": {k: v[:50] for k, v in hard.items()}, "hard_failure_counts": {k: len(v) for k, v in hard.items()},
              "warnings": {k: v[:50] for k, v in soft.items()}, "warning_counts": {k: len(v) for k, v in soft.items()},
              "contact_sheet": str(sheet)}
    rp = args.out / "qa" / f"{args.gen}_report.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("manifest_per_split", "generated_ok_per_split",
                                             "hard_failure_counts", "warning_counts")}, indent=2))
    print(f"[QA] report: {rp}\n[QA] sheet : {sheet}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
