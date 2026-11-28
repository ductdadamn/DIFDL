#!/usr/bin/env python3
"""Generate SD-1.5 Inpainting Removal samples (Seen AI) from data_synth/manifest_sd15.jsonl.

Per entry (crop is already 512x512 at native resolution -> no resize):
  1. GT = text-line boxes + mask_dilate px (identical to LaMa; test/val rows are the SAME crop+mask as LaMa).
  2. SD gets a mask covering every 8x8 latent cell of the GT (+4px), the "blank paper / no text" prompt and a
     "text, digits, ink..." negative prompt (guidance 7.5, 25 steps, strength 1.0, seed from the manifest).
  3. SD VAE-decodes the WHOLE image, so the result is pasted back with an alpha feathered INSIDE the GT
     (0.3 at the GT border -> 1 at 3px inside, exactly 0 outside): outside the GT stays bit-exact (exact label),
     no hard seam from the VAE colour shift.
  4. Ink QA: share of dark (<120) pixels inside the GT vs a 6px paper ring around it in the original.
     Fails -> retry seed+1, seed+2; still failing -> keep the attempt with the least ink and flag it
     (the region is still edited, so the label stays correct; the flag rate goes to Limitations).
Outputs: data_synth/sd15/<split>/{images,masks,pristine,raw}/<id>.png  (raw = un-pasted SD output)
         data_synth/qa/sd15_ink.jsonl  ({id, attempts, dark, dark_ring, flag})
--textdemo N: N AnyText-replacement crops with a prompt that WRITES the new text (no negative), saved to
         data_synth/sd15_textdemo/ (NOT part of the dataset: figure "SD-1.5 cannot write text" vs AnyText).
Resumable: entries whose outputs exist are skipped.

CPU dry run:  python scripts/gen_sd15_findit2.py --backend dummy --limit 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth_common import (blend, boxes_to_mask, dilate, inner_feather, latent_cover_mask, load_rgb, read_jsonl,
                          resolve_findit2_root, sample_paths, save_mask, save_png, to_crop_coords, write_jsonl)

ROOT = Path(__file__).resolve().parents[1]
DARK = 120


def make_backend(name: str, model: str):
    if name == "dummy":
        def run(rgb, sd_mask, seed, prompt=None, negative=None):
            out = rgb.copy()
            out[:] = np.clip(out.astype(int) + 3, 0, 255).astype(np.uint8)  # mimic a global VAE shift
            out[sd_mask > 0] = np.median(rgb[sd_mask == 0], axis=0).astype(np.uint8)
            if seed % 3 == 0:  # every 3rd seed draws an ink stroke -> exercises retry / flag
                ys, xs = np.nonzero(sd_mask)
                out[ys.min() + 4:ys.min() + 7, xs.min():xs.max()] = 10
            return out
        return run, "dummy"

    import torch
    from sd15_loader import inpaint, load_pipeline, resolve_model
    mdir, origin = resolve_model(model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SD15] model={mdir} ({origin}) device={dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)
    pipe = load_pipeline(mdir, dev)

    def run(rgb, sd_mask, seed, prompt=None, negative=None):
        kw = {} if prompt is None else {"prompt": prompt, "negative": negative}
        return inpaint(pipe, rgb, sd_mask, seed, **kw)
    return run, origin


def ink_stats(out, orig, gt):
    g_out = np.asarray(Image.fromarray(out).convert("L"))
    g_org = np.asarray(Image.fromarray(orig).convert("L"))
    ring = (dilate(gt, 6) > 0) & (gt == 0)
    dark = float((g_out[gt > 0] < DARK).mean())
    dark_ring = float((g_org[ring] < DARK).mean()) if ring.any() else 0.0
    return dark, dark_ring, dark <= max(0.02, dark_ring + 0.02)


def textdemo(args, run):
    rows = [e for e in read_jsonl(args.anytext_manifest) if e["op"] == "replacement"]
    rows = sorted(rows, key=lambda e: e["id"])[:args.textdemo]
    out_dir = args.out / "sd15_textdemo"
    sheet_rows = []
    for e in rows:
        rgb = load_rgb(args.findit2_root / e["src"])
        x0, y0, x1, y1 = e["crop"]
        crop = rgb[y0:y1, x0:x1].copy()
        gt = boxes_to_mask(crop.shape[0], crop.shape[1], to_crop_coords(e["boxes"], e["crop"]), dilate=e["mask_dilate"])
        prompt = f'a scanned thermal paper receipt with black printed text "{e["new_text"]}"'
        raw = run(crop, latent_cover_mask(gt), e["seed"], prompt=prompt, negative="")
        out = blend(raw, crop, inner_feather(gt))
        save_png(crop, out_dir / "orig" / f"{e['id']}.png")
        save_png(out, out_dir / "sd15" / f"{e['id']}.png")
        save_png(raw, out_dir / "raw" / f"{e['id']}.png")
        cells = [crop, out]
        if args.anytext_out:
            hits = sorted(Path(args.anytext_out).rglob(f"anytext/test/images/{e['id']}.png"))
            if hits:
                cells.append(load_rgb(hits[0]))
        sheet_rows.append((e, cells))
        print(f"  [textdemo] {e['id']}: {e['orig_text']} -> {e['new_text']!r}", flush=True)
    cell = 256
    ncol = max(len(c) for _, c in sheet_rows) if sheet_rows else 2
    sheet = Image.new("RGB", (cell * ncol, (cell + 18) * len(sheet_rows) + 18), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), " | ".join(["original", "SD-1.5 (asked to write)", "AnyText"][:ncol]), fill="black")
    for r, (e, cells) in enumerate(sheet_rows):
        y = 18 + r * (cell + 18)
        d.text((4, y), f"{e['id']}  '{' '.join(e['orig_text'])}' -> '{e['new_text']}'", fill="black")
        for c, a in enumerate(cells):
            sheet.paste(Image.fromarray(a).resize((cell, cell)), (c * cell, y + 16))
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(out_dir / "textdemo_sheet.png")
    print(f"[textdemo] {len(sheet_rows)} samples -> {out_dir}/textdemo_sheet.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findit2-root", type=Path, default=ROOT / "findit2")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data_synth" / "manifest_sd15.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--backend", choices=["sd15", "dummy"], default="sd15")
    ap.add_argument("--model", default="auto", help="auto | input | download | <local model dir>")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None)
    ap.add_argument("--limit", type=int, default=0, help="process at most N new entries (smoke test)")
    ap.add_argument("--retries", type=int, default=2, help="extra seeds tried when ink QA fails")
    ap.add_argument("--textdemo", type=int, default=0, metavar="N", help="only make N 'SD writes text' demos")
    ap.add_argument("--anytext-manifest", type=Path, default=None)
    ap.add_argument("--anytext-out", default=None, help="dir containing anytext/test/images (for the demo sheet)")
    args = ap.parse_args()
    args.findit2_root = resolve_findit2_root(args.findit2_root)

    if args.textdemo:
        if not args.anytext_manifest:
            ap.error("--textdemo needs --anytext-manifest")
        run, _ = make_backend(args.backend, args.model)
        textdemo(args, run)
        return

    entries = [e for e in read_jsonl(args.manifest) if e["gen"] == "sd15"]
    if args.split:
        entries = [e for e in entries if e["split"] == args.split]
    todo = [e for e in entries if not all(p.exists() for p in sample_paths(args.out, e).values())]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[SD15] {len(entries)} entries in manifest, {len(todo)} to generate")
    if not todo:
        return

    run, origin = make_backend(args.backend, args.model)
    qa_path = args.out / "qa" / "sd15_ink.jsonl"
    qa = {r["id"]: r for r in read_jsonl(qa_path)} if qa_path.exists() else {}
    cache_src, cache_rgb = None, None
    failed, n_flag, t0 = [], 0, time.time()
    try:
        for i, e in enumerate(sorted(todo, key=lambda e: e["src"]), 1):
            if e["src"] != cache_src:
                cache_src, cache_rgb = e["src"], load_rgb(args.findit2_root / e["src"])
            x0, y0, x1, y1 = e["crop"]
            crop = cache_rgb[y0:y1, x0:x1].copy()
            if crop.shape[:2] != (512, 512):
                failed.append((e["id"], f"crop shape {crop.shape}")); continue
            gt = boxes_to_mask(512, 512, to_crop_coords(e["boxes"], e["crop"]), dilate=e["mask_dilate"])
            full = boxes_to_mask(*cache_rgb.shape[:2], e["boxes"], dilate=e["mask_dilate"])
            if int(gt.sum()) != int(full.sum()):
                failed.append((e["id"], "mask not fully inside the crop")); continue
            sd_mask, alpha = latent_cover_mask(gt), inner_feather(gt)

            best, err = None, None
            for a in range(args.retries + 1):
                try:
                    raw = run(crop, sd_mask, e["seed"] + a)
                except RuntimeError as exc:
                    err = str(exc).strip().splitlines()[-1][:160]
                    break
                out = blend(raw, crop, alpha)
                dark, dark_ring, ok = ink_stats(out, crop, gt)
                if best is None or dark < best[2]:
                    best = (out, raw, dark, dark_ring, a + 1)
                if ok:
                    break
            if best is None:
                failed.append((e["id"], err)); continue
            out, raw, dark, dark_ring, attempts = best
            flag = not ink_stats(out, crop, gt)[2]
            n_flag += flag
            assert (out[gt == 0] == crop[gt == 0]).all(), "outside-GT pixels changed"  # never ship a wrong label

            p = sample_paths(args.out, e)
            save_png(out, p["images"])
            save_mask(gt, p["masks"])
            save_png(crop, p["pristine"])
            save_png(raw, p["images"].parent.parent / "raw" / p["images"].name)
            qa[e["id"]] = {"id": e["id"], "split": e["split"], "attempts": attempts, "dark": round(dark, 4),
                           "dark_ring": round(dark_ring, 4), "flag": bool(flag)}
            if attempts > 1 or flag:
                print(f"  [ink] {e['id']}: attempts={attempts} dark={dark:.3f} ring={dark_ring:.3f}"
                      + ("  FLAG (kept least-ink attempt)" if flag else ""), flush=True)
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el / i:.2f}s/img  ETA {(len(todo) - i) * el / i / 60:.1f} min", flush=True)
                write_jsonl(sorted(qa.values(), key=lambda r: r["id"]), qa_path)
    finally:
        write_jsonl(sorted(qa.values(), key=lambda r: r["id"]), qa_path)

    rows = list(qa.values())
    print(f"[SD15] model {origin} | {len(rows)} QA records | retried {sum(r['attempts'] > 1 for r in rows)} | "
          f"flagged {sum(r['flag'] for r in rows)} ({sum(r['flag'] for r in rows) / max(1, len(rows)):.1%})")
    if failed:
        print(f"[WARN] {len(failed)} failed (re-run to retry; outputs of others are kept):")
        for sid, msg in failed[:20]:
            print(f"   {sid}: {msg}")


if __name__ == "__main__":
    main()
