#!/usr/bin/env python3
"""Generate AnyText Replacement + Insertion samples (Unseen Test Set 2) from manifest_anytext.jsonl.

Per entry (findit2/test only): crop the 512x512 window, run AnyText `text-editing` with the text box as
draw_pos, then soft-blend the result back with a ~3px Gaussian border:
    I_out = M_soft * I_AI + (1 - M_soft) * I_orig        (bit-exact outside the blend band)
GT mask = binary box mask (dilated by `mask_dilate`) BEFORE blurring.
Outputs: data_synth/anytext/test/{images,masks,pristine}/<id>.png (+ raw/<id>.png un-blended AI output).
Resumable; per-entry seed comes from the manifest.

Run inside the isolated AnyText env (scripts/setup_anytext.sh) from any cwd:
  python scripts/gen_anytext_findit2.py --anytext-dir third_party/AnyText \
      --findit2-root /kaggle/input/<ds>/findit2 --manifest /kaggle/input/<ds>/manifest_anytext.jsonl \
      --out /kaggle/working/data_synth --limit 5
  Needs: scripts/setup_anytext.sh, scripts/download_anytext_model.sh (weights -> /tmp/anytext_model)
  and any TTF at third_party/AnyText/font/Arial_Unicode.ttf (the notebook copies DejaVuSans).
CPU dry run (no AnyText; draws text with PIL to test blending/I/O):  --backend dummy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synth_common import (blend, boxes_to_mask, load_rgb, read_jsonl, resolve_findit2_root, sample_paths,
                          save_mask, save_png, soft_blend_mask, to_crop_coords)

ROOT = Path(__file__).resolve().parents[1]


def make_backend(args):
    if args.backend == "dummy":
        def run(crop_rgb, box, text, seed):
            out = crop_rgb.copy()
            x1, y1, x2, y2 = box
            out[y1:y2, x1:x2] = np.median(crop_rgb.reshape(-1, 3), axis=0).astype(np.uint8)
            im = Image.fromarray(out)
            ImageDraw.Draw(im).text((x1 + 2, y1), text, fill=(30, 30, 30), font=ImageFont.load_default())
            return np.array(im)
        return run

    from anytext_loader import load_pipeline  # chdirs into the AnyText repo (its code expects that)

    pipe = load_pipeline(args.anytext_dir, model=args.model, revision=args.model_revision, fp16=not args.fp32)
    params = {"show_debug": False, "image_count": 1, "ddim_steps": args.ddim_steps,
              "cfg_scale": args.cfg_scale, "strength": 1.0}

    def run(crop_rgb, box, text, seed):
        h, w = crop_rgb.shape[:2]
        pos = np.zeros((h, w, 3), dtype=np.uint8)
        x1, y1, x2, y2 = box
        if args.draw_pos_format == "numpy":  # ms_wrapper: numpy draw_pos = white positions on black
            pos[y1:y2, x1:x2] = 255
            draw_pos = pos
        else:  # file path: ms_wrapper inverts it (black positions on white)
            pos[:] = 255
            pos[y1:y2, x1:x2] = 0
            draw_pos = str(args.tmp_dir / "draw_pos.png")
            Image.fromarray(pos).save(draw_pos)
        inp = {"prompt": args.prompt.format(text=text), "seed": int(seed) % (2 ** 31),
               "draw_pos": draw_pos, "ori_image": crop_rgb}
        results, code, warn, _ = pipe(inp, mode="text-editing", **params)
        if code < 0 or not results:
            raise RuntimeError(f"AnyText failed (code={code}): {warn}")
        ai = np.asarray(results[0])[..., :3].astype(np.uint8)
        if ai.shape[:2] != (h, w):  # AnyText works on multiples of 64; 512 should round-trip
            ai = np.array(Image.fromarray(ai).resize((w, h), Image.Resampling.BICUBIC))
        return ai

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findit2-root", type=Path, default=ROOT / "findit2")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data_synth" / "manifest_anytext.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--backend", choices=["anytext", "dummy"], default="anytext")
    ap.add_argument("--anytext-dir", type=Path, default=ROOT / "third_party" / "AnyText")
    ap.add_argument("--model", default="auto",
                    help="'auto' = /tmp/anytext_model if downloaded (scripts/download_anytext_model.sh), "
                         "else the ModelScope hub id; or any local model dir")
    ap.add_argument("--model-revision", default="v1.1.3")
    ap.add_argument("--prompt", default='a scanned thermal paper receipt with black printed text "{text}"')
    ap.add_argument("--ddim-steps", type=int, default=20)
    ap.add_argument("--cfg-scale", type=float, default=9.0)
    ap.add_argument("--draw-pos-format", choices=["numpy", "file"], default="numpy")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--blend-grow", type=int, default=3)
    ap.add_argument("--blend-sigma", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # resolve every path before the backend may chdir into the AnyText repo
    args.findit2_root = resolve_findit2_root(args.findit2_root)
    args.findit2_root, args.manifest, args.out = (p.resolve() for p in (args.findit2_root, args.manifest, args.out))
    args.tmp_dir = args.out / "anytext" / "_tmp"
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in read_jsonl(args.manifest) if e["gen"] == "anytext"]
    assert all(e["split"] == "test" for e in entries), "AnyText is Unseen Test only — manifest has non-test rows"
    todo = [e for e in entries if not all(p.exists() for p in sample_paths(args.out, e).values())]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[AnyText] {len(entries)} entries in manifest, {len(todo)} to generate")
    if not todo:
        return

    run = make_backend(args)
    failed, t0 = [], time.time()
    for i, e in enumerate(todo, 1):
        x0, y0, x1, y1 = e["crop"]
        crop = load_rgb(args.findit2_root / e["src"])[y0:y1, x0:x1]
        box = to_crop_coords(e["boxes"], e["crop"])[0]
        gt = boxes_to_mask(crop.shape[0], crop.shape[1], [box], dilate=e["mask_dilate"])
        try:
            ai = run(crop, box, e["new_text"], e["seed"])
        except Exception as exc:  # keep going; failed ids are listed at the end
            failed.append((e["id"], str(exc)[:200]))
            continue
        alpha = soft_blend_mask(gt, grow=args.blend_grow, sigma=args.blend_sigma)
        p = sample_paths(args.out, e)
        save_png(ai, p["images"].parent.parent / "raw" / p["images"].name)
        save_png(blend(ai, crop, alpha), p["images"])
        save_mask(gt, p["masks"])
        save_png(crop, p["pristine"])
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el / i:.1f}s/img  ETA {(len(todo) - i) * el / i / 60:.1f} min")

    if failed:
        print(f"[WARN] {len(failed)} failed:")
        for sid, msg in failed[:20]:
            print(f"   {sid}: {msg}")


if __name__ == "__main__":
    main()
