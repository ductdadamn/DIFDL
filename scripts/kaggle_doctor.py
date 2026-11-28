#!/usr/bin/env python3
"""Probe what a Kaggle session can do BEFORE running the real notebooks.

Each stage runs a list of independent checks (one failure never stops the others) and prints
`[PASS]/[FAIL] <check>: <detail>` plus a summary; results are also saved to
/kaggle/working/doctor/<stage>.json. Paste the summary back when something fails.

  python scripts/kaggle_doctor.py --stage env                          # Kaggle python: hw, disk, net, inputs
  /tmp/venv_ocr/bin/python  scripts/kaggle_doctor.py --stage ocr       # PaddleOCR on 1 real receipt
  /tmp/venv_lama/bin/python scripts/kaggle_doctor.py --stage lama      # LaMa on GPU, 1 synthetic inpaint
  /tmp/venv_sd15/bin/python scripts/kaggle_doctor.py --stage sd15      # SD-1.5 inpainting: download, load, 1 edit
  /tmp/DIFDL/.venv_anytext/bin/python scripts/kaggle_doctor.py --stage anytext [--full]
        (--full loads the weights from scripts/download_anytext_model.sh and runs 1 edit)
Stdlib-only at import time; runs on Python 3.10-3.13.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
RESULTS: list[dict] = []


def check(name, fn):
    t0 = time.time()
    try:
        detail, ok = fn(), True
    except (Exception, SystemExit) as exc:  # report and continue (resolve_* helpers raise SystemExit)
        tb = traceback.format_exc().strip().splitlines()
        detail, ok = f"{type(exc).__name__}: {exc} | " + " / ".join(tb[-4:-1]), False
    dt = time.time() - t0
    RESULTS.append({"check": name, "ok": ok, "detail": str(detail), "sec": round(dt, 1)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s): {detail}", flush=True)


def versions(*mods):
    out = {}
    for m in mods:
        try:
            mod = importlib.import_module(m)
            ver = getattr(mod, "__version__", None)
            if ver is None:  # e.g. open_clip 2.7.0: no __version__ -> ask the installed distribution
                from importlib.metadata import PackageNotFoundError, packages_distributions, version
                try:
                    ver = version(packages_distributions().get(m, [m])[0])
                except PackageNotFoundError:
                    ver = "?"
            out[m] = ver
        except Exception as exc:
            out[m] = f"MISSING ({type(exc).__name__})"
    return out


def sh(cmd, timeout=30):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or r.stderr).strip()


def first_receipt():
    from synth_common import resolve_findit2_root
    root = resolve_findit2_root("auto")
    return sorted((root / "test").glob("*.png"))[0]


# ----------------------------------------------------------------------------- stages
def stage_env(args):
    check("python", lambda: f"{sys.version.split()[0]} @ {sys.executable} | {platform.platform()}")
    check("cpu_ram", lambda: f"{os.cpu_count()} cpus | " + next(
        l for l in open("/proc/meminfo") if l.startswith("MemTotal")).strip())
    def gpu():
        out = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
        if "MiB" not in out:
            raise RuntimeError(f"no GPU visible (Accelerator = None?): {out[:120]}")
        return out
    check("gpu", gpu)
    for d in ("/tmp", "/kaggle/working"):
        check(f"disk {d}", lambda d=d: "free {:.1f} GB / total {:.1f} GB".format(
            *(x / 1e9 for x in (shutil.disk_usage(d).free, shutil.disk_usage(d).total))))
    check("tools", lambda: {t: bool(shutil.which(t)) for t in ("git", "curl", "bash", "gcc", "uv", "nvcc")})
    check("env_vars", lambda: {k: v for k, v in os.environ.items()
                               if k.startswith(("UV_", "PIP_", "CUDA", "PYTHON", "KAGGLE_KERNEL"))})
    check("kernel_pkgs", lambda: versions("numpy", "PIL", "torch", "cv2", "matplotlib", "IPython"))

    import urllib.request
    urls = {"github": "https://github.com", "pypi": "https://pypi.org/simple/paddleocr/",
            "pytorch_index": "https://download.pytorch.org/whl/cu121/torch/",
            "uv_installer": "https://astral.sh/uv/install.sh",
            "python_builds": "https://github.com/astral-sh/python-build-standalone/releases",
            "paddle_models": "https://paddle-model-ecology.bj.bcebos.com/",
            "paddle_gpu_index": "https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/",
            "huggingface": "https://huggingface.co", "modelscope": "https://www.modelscope.cn",
            "lama_weights": "https://github.com/enesmsahin/simple-lama-inpainting/releases"}
    for name, url in urls.items():
        def probe(url=url):
            req = urllib.request.Request(url, headers={"User-Agent": "difdl-doctor"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return f"HTTP {r.status}"
            except urllib.error.HTTPError as e:  # reachable, just not a 200 page
                return f"reachable (HTTP {e.code})"
        check(f"net {name}", probe)

    def inputs():
        rows = []
        for d in sorted(Path("/kaggle/input").rglob("*")):
            if d.is_dir() and len(d.relative_to("/kaggle/input").parts) <= 4:
                n = sum(1 for f in d.iterdir() if f.is_file())
                rows.append(f"{d.relative_to('/kaggle/input')} ({n} files)")
        return rows[:40]
    check("kaggle_inputs", inputs)
    check("findit2_auto", lambda: str(first_receipt().parent.parent))
    check("tsroie_auto", lambda: str(__import__("synth_common").resolve_tsroie_root("auto")))


def stage_ocr(args):
    from importlib import metadata
    check("versions", lambda: versions("paddle", "paddleocr", "paddlex", "numpy", "PIL", "cv2"))
    check("declared_deps", lambda: {p: [r for r in (metadata.requires(p) or []) if "extra" not in r][:12]
                                    for p in ("paddleocr", "paddlex")})
    def paddle_dev():
        import paddle
        from select_edit_regions import resolve_paddle_device
        return (f"compiled_with_cuda={paddle.device.is_compiled_with_cuda()} | default={paddle.device.get_device()}"
                f" | OCR will use {resolve_paddle_device(args.device)}")
    check("paddle_device", paddle_dev)
    state = {}

    def build():
        from select_edit_regions import ocr_paddle_factory
        state["run"] = ocr_paddle_factory(args.det_model, args.rec_model, args.device)
        return f"det={args.det_model} rec={args.rec_model} device={args.device}"
    check("PaddleOCR init", build)

    def run():
        img = first_receipt()
        state["run"](img)  # warm-up (first GPU call compiles kernels)
        t0 = time.time()
        items = state["run"](img)
        hs = sorted(it["box"][3] - it["box"][1] for it in items) or [0]
        return (f"{img.name}: {len(items)} lines, {time.time() - t0:.1f}s, median h={hs[len(hs) // 2]}px, "
                f"sample={items[:3]}")
    check("PaddleOCR predict (1 receipt)", run)


def stage_lama(args):
    import numpy as np
    check("versions", lambda: versions("torch", "torchvision", "numpy", "PIL", "cv2", "simple_lama_inpainting"))
    import torch
    check("cuda", lambda: f"{torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}"
          f" | torch cuda {torch.version.cuda}")
    state = {}

    def load():
        from simple_lama_inpainting import SimpleLama
        try:
            state["lama"] = SimpleLama(device=torch.device("cuda"))
        except TypeError:
            state["lama"] = SimpleLama()
        return f"TORCH_HOME={os.environ.get('TORCH_HOME')}"
    check("SimpleLama load (downloads big-lama.pt)", load)

    def run():
        from PIL import Image
        img = first_receipt()
        with Image.open(img) as im:
            rgb = np.array(im.convert("RGB"))
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask[h // 3:h // 3 + 30, w // 4:w // 2] = 255
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = np.array(state["lama"](Image.fromarray(rgb), Image.fromarray(mask)))
        return (f"in {rgb.shape} -> out {out.shape} in {time.time() - t0:.1f}s, "
                f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    check("LaMa inpaint (1 full receipt)", run)


def stage_anytext(args):
    check("versions", lambda: versions("torch", "torchvision", "numpy", "PIL", "cv2", "modelscope", "transformers",
                                       "diffusers", "pytorch_lightning", "open_clip", "timm", "datasets"))
    import torch
    check("cuda", lambda: f"{torch.cuda.is_available()} | torch cuda {torch.version.cuda}")
    ad = Path(args.anytext_dir).resolve()
    check("anytext_repo", lambda: f"{ad} @ " + sh(f"git -C '{ad}' log -1 --format='%h %cs %s'"))
    check("font", lambda: f"{(ad / 'font' / 'Arial_Unicode.ttf').stat().st_size} bytes")

    check("import modelscope.pipelines", lambda: (__import__("modelscope.pipelines"), "OK")[1])

    def import_scan():
        import importlib.util
        from anytext_import_scan import DEFAULT_ROOTS, OPTIONAL, scan
        from anytext_loader import resolve_model
        try:  # scan the wrapper of the model dir actually used (auto/input/dir), else the default roots
            wrapper = Path(resolve_model(args.model)[0]) / "ms_wrapper.py"
        except FileNotFoundError:
            wrapper = Path("/nonexistent")
        external, files = scan(ad, [wrapper] if wrapper.exists() else [], DEFAULT_ROOTS)
        missing = sorted(m for m in external if m not in OPTIONAL and importlib.util.find_spec(m) is None)
        if missing:
            raise RuntimeError(f"missing modules {missing} <- " + "; ".join(f"{m}: {sorted(external[m])[:2]}" for m in missing))
        return f"{len(files)} files, {len(external)} third-party modules all importable"
    check("import scan (ms_wrapper + repo, transitive)", import_scan)

    def model_dir():
        from anytext_loader import resolve_model
        model, is_local = resolve_model(args.model)
        if not is_local:
            raise RuntimeError(f"no local model dir (got hub id {model}); run scripts/download_anytext_model.sh")
        done = Path(model) / ".done"
        return f"{model} | " + (json.loads(done.read_text())["model_id"] if done.exists() else "no .done")
    check("model dir (ms_wrapper.py + configuration.json)", model_dir)
    if not args.full:
        print("(skip pipeline load: pass --full to load the downloaded weights and run 1 edit)")
        return

    state = {}

    # Read-only check (KAGGLE_RUN_LOG #12): a model dir under /kaggle/input is read-only. Two pieces of evidence:
    #  - is the dir writable at all? (probe file). Not writable + load/edit PASS => proven on read-only.
    #  - audit: list (size, mtime) of every file before load and after the edit; any new/changed file means
    #    modelscope/AnyText writes into the model dir => would crash on /kaggle/input.
    def snapshot(d):
        return {str(f.relative_to(d)): (f.stat().st_size, f.stat().st_mtime_ns) for f in d.rglob("*") if f.is_file()}

    def ro_probe():
        from anytext_loader import resolve_model
        d = Path(resolve_model(args.model)[0])
        state["mdir"], state["snap"] = d, snapshot(d)
        probe = d / ".doctor_write_probe"
        try:
            probe.write_text("x"); probe.unlink(); state["writable"] = True
        except OSError as exc:
            state["writable"] = False
            return f"{d} is READ-ONLY ({type(exc).__name__}: {exc.strerror}) -> load/edit below test the read-only case for real"
        return f"{d} is writable -> read-only safety is judged by the write audit after the edit"
    check("model dir writable?", ro_probe)

    def load():
        from anytext_loader import load_pipeline
        t0 = time.time()
        state["pipe"] = load_pipeline(ad, model=args.model)
        return f"loaded in {time.time() - t0:.0f}s"
    check("AnyText pipeline load", load)

    def edit():
        import numpy as np
        if "pipe" not in state:
            raise RuntimeError("skipped: pipeline did not load (see previous FAIL)")
        img = np.full((512, 512, 3), 235, np.uint8)
        pos = np.zeros((512, 512, 3), np.uint8)
        pos[200:240, 150:330] = 255
        t0 = time.time()
        res, code, warn, _ = state["pipe"]({"prompt": 'a receipt with black printed text "12.50"', "seed": 42,
                                            "draw_pos": pos, "ori_image": img},
                                           mode="text-editing", image_count=1, ddim_steps=20, show_debug=False)
        out = np.asarray(res[0]) if res else None
        changed = None if out is None else float(np.abs(out.astype(int) - img.astype(int)).mean(axis=2)[200:240, 150:330].mean())
        outside = None if out is None else float(np.abs(out.astype(int) - img.astype(int))[:150].mean())
        return (f"code={code} warn={warn!r} out={None if out is None else out.shape} {time.time() - t0:.1f}s "
                f"| mean|diff| inside box={changed} vs top area={outside} (inside >> outside => draw_pos polarity OK)")
    check("AnyText text-editing (1 synthetic)", edit)

    def write_audit():
        if "snap" not in state or "pipe" not in state:
            raise RuntimeError("skipped: model dir or pipeline load failed (see FAIL above)")
        after = snapshot(state["mdir"])
        changed = sorted(k for k in after if state["snap"].get(k) != after[k])
        gone = sorted(set(state["snap"]) - set(after))
        if changed or gone:
            raise RuntimeError(f"load/edit WROTE into the model dir -> breaks on read-only /kaggle/input: "
                               f"new/changed={changed[:10]} removed={gone[:10]}")
        how = "proven: dir is read-only and load+edit passed" if not state["writable"] else               "inferred: dir writable but load+edit wrote nothing into it"
        return f"0 files written in {state['mdir']} ({len(after)} files) -> read-only safe ({how})"
    check("model dir untouched by load+edit (read-only safe)", write_audit)


def stage_sd15(args):
    import numpy as np
    check("versions", lambda: versions("torch", "numpy", "PIL", "diffusers", "transformers", "accelerate",
                                       "huggingface_hub", "safetensors"))
    import torch
    check("cuda", lambda: f"{torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}"
          f" | torch cuda {torch.version.cuda}")
    from sd15_loader import REPO, REVISION, inpaint, load_pipeline, resolve_model
    state = {}

    def model():
        t0 = time.time()
        state["mdir"], origin = resolve_model("auto" if args.model == "auto" else args.model)
        size = sum(f.stat().st_size for f in Path(state["mdir"]).rglob("*") if f.is_file()) / 2**30
        return f"{state['mdir']} ({origin}, {size:.2f} GB, {time.time() - t0:.0f}s) | pinned {REPO}@{REVISION[:7]}"
    check("model (fp16 safetensors, pinned revision)", model)

    def load():
        t0 = time.time()
        state["pipe"] = load_pipeline(state["mdir"])
        p = state["pipe"]
        return (f"{type(p).__name__} in {time.time() - t0:.0f}s | unet in_channels={p.unet.config.in_channels} "
                f"(9 = inpainting UNet) | scheduler={type(p.scheduler).__name__} | safety_checker={p.safety_checker}")
    check("pipeline load (fp16, cuda)", load)

    def edit():
        from PIL import Image
        from synth_common import blend, boxes_to_mask, inner_feather, latent_cover_mask
        from gen_sd15_findit2 import ink_stats
        with Image.open(first_receipt()) as im:
            rgb = np.array(im.convert("RGB"))
        crop = rgb[:512, :512].copy() if min(rgb.shape[:2]) >= 512 else np.full((512, 512, 3), 230, np.uint8)
        gt = boxes_to_mask(512, 512, [[96, 200, 400, 228]], dilate=3)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for seed in (1, 2):  # 2nd call = steady-state speed for the ETA
            t0 = time.time()
            raw = inpaint(state["pipe"], crop, latent_cover_mask(gt), seed)
            times.append(time.time() - t0)
        out = blend(raw, crop, inner_feather(gt))
        outside = int(np.abs(out.astype(int) - crop.astype(int))[gt == 0].max(initial=0))
        vae_shift = float(np.abs(raw.astype(int) - crop.astype(int))[gt == 0].mean())
        inside = float(np.abs(out.astype(int) - crop.astype(int))[gt > 0].mean())
        dark, ring, ok = ink_stats(out, crop, gt)
        if raw.shape != (512, 512, 3) or outside != 0 or inside < 1.0:
            raise RuntimeError(f"raw {raw.shape}, max|diff| outside GT after paste={outside} (must be 0), "
                               f"mean|diff| inside={inside:.2f} (must be > 1)")
        return (f"{times[-1]:.1f}s/img (1st {times[0]:.1f}s), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB | "
                f"outside GT after paste: max|diff|=0 (bit-exact) | raw SD outside GT mean|diff|={vae_shift:.2f} "
                f"(VAE shift the paste removes) | inside mean|diff|={inside:.1f} | ink dark={dark:.3f} ring={ring:.3f} ok={ok}")
    check("SD-1.5 inpaint 1 crop + inner-feather paste", edit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["env", "ocr", "lama", "sd15", "anytext"], required=True)
    ap.add_argument("--det-model", default="PP-OCRv5_server_det")
    ap.add_argument("--rec-model", default="PP-OCRv5_mobile_rec")
    ap.add_argument("--device", default="auto", help="ocr stage: auto | gpu:0 | cpu")
    ap.add_argument("--anytext-dir", default="/tmp/DIFDL/third_party/AnyText")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default="auto", help="AnyText / SD-1.5 model: auto (/tmp/anytext_model, else /kaggle/input copy) | input (/kaggle/input copy only) | dir | hub id")
    ap.add_argument("--label", default="", help="suffix for the result json (e.g. _input -> anytext_input.json)")
    ap.add_argument("--out", default="/kaggle/working/doctor")
    args = ap.parse_args()

    print(f"===== DOCTOR stage={args.stage} python={sys.version.split()[0]} =====", flush=True)
    {"env": stage_env, "ocr": stage_ocr, "lama": stage_lama, "sd15": stage_sd15, "anytext": stage_anytext}[args.stage](args)

    n_fail = sum(not r["ok"] for r in RESULTS)
    print(f"===== SUMMARY stage={args.stage}{args.label}: {len(RESULTS) - n_fail} PASS / {n_fail} FAIL =====")
    for r in RESULTS:
        if not r["ok"]:
            print(f"  FAIL {r['check']}: {r['detail'][:400]}")
    try:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"{args.stage}{args.label}.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    except OSError:
        pass


if __name__ == "__main__":
    main()
