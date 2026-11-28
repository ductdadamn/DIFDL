"""SD-1.5 Inpainting: pinned model download/locate + pipeline load (shared by gen_sd15_findit2.py and the doctor).

Model: stable-diffusion-v1-5/stable-diffusion-inpainting (the official mirror after runwayml/ was taken down),
pinned to one commit; only the fp16 safetensors are fetched (~2 GB, not the 4 GB .ckpt).
`resolve_model("auto")`: a complete copy under /kaggle/input (e.g. a saved notebook output) else download.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"  # HF commit, lastModified 2024-09-06
ALLOW = ["model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*", "*/config.json",
         "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
         "text_encoder/model.fp16.safetensors"]
NEEDED = ["model_index.json", "unet/diffusion_pytorch_model.fp16.safetensors",
          "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors",
          "tokenizer/vocab.json", "scheduler/scheduler_config.json"]
KAGGLE_INPUT = Path("/kaggle/input")

PROMPT = "blank plain thermal receipt paper background, clean empty paper texture, no text"
NEGATIVE = "text, letters, numbers, digits, words, ink, black dots, lines, symbols, watermark"


def is_complete(d: Path) -> bool:
    return all((Path(d) / f).is_file() for f in NEEDED)


def find_input_copy() -> Path | None:
    if not KAGGLE_INPUT.is_dir():
        return None
    for mi in sorted(KAGGLE_INPUT.rglob("model_index.json")):
        if is_complete(mi.parent) and "StableDiffusionInpaintPipeline" in mi.read_text():
            return mi.parent
    return None


def resolve_model(model: str = "auto", cache: str = "/tmp/sd15_model") -> tuple[Path, str]:
    """-> (local dir, where it came from). model: auto | input | download | <local dir>."""
    if model not in ("auto", "input", "download"):
        if not is_complete(Path(model)):
            raise FileNotFoundError(f"{model} is not a complete SD-1.5 inpainting dir (needs {NEEDED})")
        return Path(model), "dir"
    if model in ("auto", "input"):
        hit = find_input_copy()
        if hit:
            return hit, "kaggle-input"
        if model == "input":
            raise FileNotFoundError("no complete SD-1.5 inpainting copy under /kaggle/input")
    if is_complete(Path(cache)):
        return Path(cache), "cache"
    from huggingface_hub import snapshot_download
    p = snapshot_download(REPO, revision=REVISION, allow_patterns=ALLOW, local_dir=cache)
    if not is_complete(Path(p)):
        raise RuntimeError(f"download incomplete in {p}")
    return Path(p), f"hf:{REPO}@{REVISION[:7]}"


def load_pipeline(model_dir: Path, device: str = "cuda"):
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.float16 if device == "cuda" else torch.float32, variant="fp16",
        safety_checker=None, requires_safety_checker=False, local_files_only=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def inpaint(pipe, rgb, sd_mask, seed: int, prompt: str = PROMPT, negative: str = NEGATIVE,
            steps: int = 25, guidance: float = 7.5):
    """rgb uint8 HxWx3 (512x512), sd_mask {0,1} -> raw SD output uint8 HxWx3 (whole image VAE-decoded)."""
    import numpy as np
    import torch
    from PIL import Image
    h, w = sd_mask.shape
    g = torch.Generator(device=pipe.device).manual_seed(int(seed) % (2 ** 63))
    out = pipe(prompt=prompt, negative_prompt=negative or None, image=Image.fromarray(rgb),
               mask_image=Image.fromarray((sd_mask > 0).astype(np.uint8) * 255), height=h, width=w,
               num_inference_steps=steps, guidance_scale=guidance, strength=1.0, generator=g).images[0]
    return np.array(out.convert("RGB"))
