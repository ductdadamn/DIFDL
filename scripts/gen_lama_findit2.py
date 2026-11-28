#!/usr/bin/env python3
"""Generate LaMa Removal samples (Seen AI) from data_synth/manifest_lama.jsonl.

Per entry: LaMa inpaints a CONTEXT WINDOW at native resolution = the 512x512 crop grown by --context px
per side (<= 1024x1024 by default; never resampled), with the dilated text-line mask. Pixels outside the
mask are copied back from the original (bit-exact GT), then the 512x512 crop is saved.
(Inpainting the whole receipt OOM'd on 7016x4961 scans: 25 GiB on a 16 GB T4, see KAGGLE_RUN_LOG #8.)
  data_synth/lama/<split>/images/<id>.png    tampered crop (PNG, lossless)
  data_synth/lama/<split>/masks/<id>.png     GT mask 0/255
  data_synth/lama/<split>/pristine/<id>.png  same crop from the untouched receipt (image-level control)
Resumable: entries whose 3 outputs exist are skipped.

Kaggle (T4):   pip install simple-lama-inpainting   (do NOT install the full requirements.txt)
               python scripts/gen_lama_findit2.py --findit2-root /kaggle/input/<ds>/findit2 \
                   --manifest /kaggle/input/<ds>/manifest_lama.jsonl --out /kaggle/working/data_synth
CPU dry run:   python scripts/gen_lama_findit2.py --backend dummy --limit 5   (pipeline test only!)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from synth_common import (boxes_to_mask, load_rgb, read_jsonl, resolve_findit2_root, sample_paths,
                          save_mask, save_png)

ROOT = Path(__file__).resolve().parents[1]


def make_backend(name: str):
    if name == "dummy":
        def run(rgb, mask):  # flat paper colour: only for testing the I/O pipeline
            out = rgb.copy()
            out[mask > 0] = np.median(rgb[mask == 0], axis=0).astype(np.uint8)
            return out
        return run

    import torch
    from simple_lama_inpainting import SimpleLama

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LaMa] device={dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    try:
        lama = SimpleLama(device=dev)
    except TypeError:  # older simple-lama without the device kwarg (picks CUDA automatically)
        lama = SimpleLama()

    def run(rgb, mask):
        h, w = mask.shape
        with torch.no_grad():
            res = lama(Image.fromarray(rgb), Image.fromarray(mask * 255))
        res = np.array(res.convert("RGB"))
        if res.shape[0] < h or res.shape[1] < w:
            raise RuntimeError(f"LaMa output {res.shape} smaller than input {(h, w)}")
        return res[:h, :w]  # simple-lama pads to a multiple of 8

    def free():
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    run.free = free
    return run


def context_window(crop, w, h, ctx):
    x0, y0, x1, y1 = crop
    return max(0, x0 - ctx), max(0, y0 - ctx), min(w, x1 + ctx), min(h, y1 + ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findit2-root", type=Path, default=ROOT / "findit2")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data_synth" / "manifest_lama.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--backend", choices=["lama", "dummy"], default="lama")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None)
    ap.add_argument("--limit", type=int, default=0, help="process at most N new entries (smoke test)")
    ap.add_argument("--context", type=int, default=256,
                    help="px of paper context around the 512 crop given to LaMa (retries with 128, 0 on OOM)")
    args = ap.parse_args()
    args.findit2_root = resolve_findit2_root(args.findit2_root)

    entries = [e for e in read_jsonl(args.manifest) if e["gen"] == "lama"]
    if args.split:
        entries = [e for e in entries if e["split"] == args.split]
    todo = [e for e in entries if not all(p.exists() for p in sample_paths(args.out, e).values())]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[LaMa] {len(entries)} entries in manifest, {len(todo)} to generate")
    if not todo:
        return

    run = make_backend(args.backend)
    cache_src, cache_rgb = None, None
    failed, t0 = [], time.time()
    for i, e in enumerate(sorted(todo, key=lambda e: e["src"]), 1):
        if e["src"] != cache_src:
            cache_src, cache_rgb = e["src"], load_rgb(args.findit2_root / e["src"])
        rgb = cache_rgb
        h, w = rgb.shape[:2]
        mask = boxes_to_mask(h, w, e["boxes"], dilate=e["mask_dilate"])
        x0, y0, x1, y1 = e["crop"]
        if mask[y0:y1, x0:x1].sum() != mask.sum():
            failed.append((e["id"], "mask not fully inside the crop")); continue

        out, err = None, None
        for ctx in dict.fromkeys((args.context, 128, 0)):  # shrink the window if the GPU runs out of memory
            wx0, wy0, wx1, wy1 = context_window(e["crop"], w, h, ctx)
            win_rgb, win_mask = rgb[wy0:wy1, wx0:wx1], mask[wy0:wy1, wx0:wx1]
            try:
                pred = run(win_rgb, win_mask)
            except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError
                err = str(exc).strip().splitlines()[-1][:160]
                getattr(run, "free", lambda: None)()
                if "out of memory" not in str(exc).lower():
                    break
                print(f"  [OOM] {e['id']} window {wx1 - wx0}x{wy1 - wy0} -> retry smaller", flush=True)
                continue
            out = rgb[y0:y1, x0:x1].copy()  # the mask lies inside the crop (checked above)
            pc, mc = pred[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0], mask[y0:y1, x0:x1]
            out[mc > 0] = pc[mc > 0]  # bit-exact outside the GT region
            break
        if out is None:
            failed.append((e["id"], err)); continue

        p = sample_paths(args.out, e)
        save_png(out, p["images"])
        save_mask(mask[y0:y1, x0:x1], p["masks"])
        save_png(rgb[y0:y1, x0:x1], p["pristine"])

        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el / i:.2f}s/img  ETA {(len(todo) - i) * el / i / 60:.1f} min", flush=True)

    if failed:
        print(f"[WARN] {len(failed)} failed (re-run to retry; outputs of others are kept):")
        for sid, msg in failed[:20]:
            print(f"   {sid}: {msg}")


if __name__ == "__main__":
    main()
