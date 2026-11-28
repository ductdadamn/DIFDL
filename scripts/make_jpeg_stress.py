#!/usr/bin/env python3
"""JPEG stress copies (Test Set 3, CSV #10): re-encode the PNG test sets ONCE as real .jpg files at QF 70 and 50.

For every generator test set <src>/<gen>/test/{images,masks,pristine}/<id>.png it writes
  <out>/<gen>/test_qf<Q>/images/<id>.jpg     tampered crop, single JPEG pass from the lossless PNG
  <out>/<gen>/test_qf<Q>/pristine/<id>.jpg   pristine crop, SAME compression (otherwise AUC/FPR under stress
                                             would be confounded by "fake = jpg, real = png")
  <out>/<gen>/test_qf<Q>/masks/<id>.png      GT mask copied byte-for-byte (never compressed)
  <out>/stress_manifest.json                 QF, subsampling, Pillow/libjpeg version, file counts per gen/QF
Encoder: Pillow/libjpeg, baseline, 4:2:0 chroma subsampling, optimize=False, standard IJG tables scaled by QF
(= what a phone / social network upload produces). Real files on disk: the DCT coefficients and quantization
tables are exactly what a detector (ADCD-Net's DCT branch) will read. Idempotent: existing outputs are skipped.

Kaggle (CPU):  python scripts/make_jpeg_stress.py --src auto --out /kaggle/working/data_synth_stress
Local:         python scripts/make_jpeg_stress.py --src data_synth --out data_synth_stress --gens lama anytext
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import PIL
from PIL import Image, features

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
GENS = ("lama", "sd15", "anytext", "textdiffuser", "tsroie")
SUBSAMPLING = 2  # Pillow: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0

# IJG standard luminance table (natural order). libjpeg scales it with quality -> used to verify the output.
STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def expected_luma_table(q: int) -> list[int]:
    scale = 5000 // q if q < 50 else 200 - 2 * q
    return [min(255, max(1, (t * scale + 50) // 100)) for t in STD_LUMA]


def find_test_dirs(src: Path, gens) -> dict[str, Path]:
    """{gen: <...>/<gen>/test} for each gen found under src (searched recursively, e.g. /kaggle/input)."""
    found = {}
    for gen in gens:
        hits = sorted({p.parent.parent for p in src.rglob(f"{gen}/test/images") if p.is_dir()})
        hits = [h for h in hits if (h / "test" / "masks").is_dir()]
        if len(hits) > 1:
            raise SystemExit(f"[ERROR] several '{gen}/test' dirs under {src}: {hits}. Pass --src explicitly.")
        if hits:
            found[gen] = hits[0] / "test"
            print(f"[auto] {gen}: {found[gen]}")
        else:
            print(f"[skip] {gen}: no {gen}/test/{{images,masks}} under {src}")
    return found


def encode(src_png: Path, dst_jpg: Path, q: int):
    with Image.open(src_png) as im:
        rgb = im.convert("RGB")
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_jpg.with_suffix(".tmp")
    rgb.save(tmp, format="JPEG", quality=q, subsampling=SUBSAMPLING, optimize=False, progressive=False)
    tmp.replace(dst_jpg)  # atomic: an interrupted run never leaves a half-written .jpg that would be skipped


def verify(jpg: Path, q: int) -> str | None:
    with Image.open(jpg) as im:
        if im.format != "JPEG":
            return f"format {im.format}"
        luma = im.quantization.get(0)
        if luma is None or sorted(luma) != sorted(expected_luma_table(q)):  # order-free: zigzag vs natural
            return f"luma quant table does not match QF {q}"
        if im.layer[0][1:3] != (2, 2):  # Y sampled 2x2 relative to chroma = 4:2:0
            return f"subsampling {im.layer}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data_synth"),
                    help="root holding <gen>/test/... ; 'auto' = search /kaggle/input")
    ap.add_argument("--out", type=Path, default=ROOT / "data_synth_stress")
    ap.add_argument("--gens", nargs="+", default=list(GENS), choices=GENS)
    ap.add_argument("--qf", type=int, nargs="+", default=[70, 50])
    ap.add_argument("--limit", type=int, default=0, help="per gen, smoke test")
    args = ap.parse_args()
    src = KAGGLE_INPUT if args.src == "auto" else Path(args.src)

    tests = find_test_dirs(src, args.gens)
    if not tests:
        raise SystemExit(f"[ERROR] no generator test set found under {src}")

    summary = {"encoder": {"pillow": PIL.__version__, "libjpeg_turbo": features.version("libjpeg_turbo"),
                           "subsampling": "4:2:0", "optimize": False, "progressive": False,
                           "passes": 1, "source": "lossless PNG"},
               "qf": args.qf, "sets": {}}
    bad = []
    for gen, tdir in tests.items():
        ids = sorted(p.stem for p in (tdir / "images").glob("*.png"))
        if args.limit:
            ids = ids[:args.limit]
        has_pristine = (tdir / "pristine").is_dir()
        for q in args.qf:
            odir = args.out / gen / f"test_qf{q}"
            n = {"images": 0, "pristine": 0, "masks": 0, "skipped_existing": 0}
            for sid in ids:
                mask_src = tdir / "masks" / f"{sid}.png"
                if not mask_src.exists():
                    bad.append(f"{gen}/{sid}: mask missing"); continue
                jobs = [("images", tdir / "images" / f"{sid}.png")]
                if has_pristine and (tdir / "pristine" / f"{sid}.png").exists():
                    jobs.append(("pristine", tdir / "pristine" / f"{sid}.png"))
                for kind, p in jobs:
                    dst = odir / kind / f"{sid}.jpg"
                    if dst.exists():
                        n["skipped_existing"] += 1
                    else:
                        encode(p, dst, q)
                    n[kind] += 1
                    err = verify(dst, q)
                    if err:
                        bad.append(f"{gen}/qf{q}/{kind}/{sid}: {err}")
                mdst = odir / "masks" / f"{sid}.png"
                if not mdst.exists():
                    mdst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(mask_src, mdst)
                n["masks"] += 1
            summary["sets"][f"{gen}/test_qf{q}"] = n
            print(f"[JPEG] {gen} QF{q}: {n}", flush=True)

    summary["errors"] = bad[:50]
    summary["n_errors"] = len(bad)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stress_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if bad:
        raise SystemExit(f"[FAIL] {len(bad)} problems (see stress_manifest.json)")
    print(f"[OK] all JPEG files verified (format, QF table, 4:2:0) -> {args.out}")


if __name__ == "__main__":
    main()
