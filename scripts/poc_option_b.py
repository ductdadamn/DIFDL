"""Live Option B PoC with real ADCDNet.pth & docres.pkl weights on RTX 4060 (8GB).

Evaluates zero-shot ADCD-Net on:
  1) T-SROIE_test (GAN-edited receipts)
  2) OSTF_test Diffusion (AnyText, TextDiffuser, UDiffText, DiffSTE)
  3) OSTF_test GAN (SRNet, MOSTEL, STEFANN)
  4) findit2/test manual CPI receipts
  5) poc_out/samples simulated AI-inpaint receipts
Then fine-tunes ADCDNetWithSRM (+960 SRM residual params) for 30 steps on a disjoint
mixed split using gradient checkpointing (fp16, bs=4) and re-evaluates on held-out sets.
Saves visual heatmaps/predictions to poc_out/vis_comparison.png.
"""

import ast
import csv
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from jpeg2dct.numpy import load as jpeg2dct_load
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import cfg  # noqa: E402
from loss.focal_loss import supcon_parallel  # noqa: E402
from loss.lovasz_loss import LovaszLoss  # noqa: E402
from loss.soft_ce_loss import SoftCrossEntropyLoss  # noqa: E402
from model.model import ADCDNet  # noqa: E402
from model.restormer import TransformerBlock  # noqa: E402

IMG_NORM = T.Compose(
    [T.ToTensor(), T.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))]
)


def extract_text_mask_and_boxes(rgb: np.ndarray, max_boxes: int = 8):
    """Approximate OCR text mask + candidate boxes via morphological connected components."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)

    ocr_mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if 8 <= bh <= 80 and 8 <= bw <= int(0.8 * w) and area >= 40:
            ocr_mask[y : y + bh, x : x + bw] = 1
        if 12 <= bh <= 48 and 24 <= bw <= 180 and area >= 120:
            if x > 10 and y > 10 and (x + bw) < (w - 10) and (y + bh) < (h - 10):
                boxes.append((int(x), int(y), int(bw), int(bh)))
    boxes.sort(key=lambda b: b[1])
    return ocr_mask, boxes[:max_boxes]


def simulate_ai_text_edit(rgb: np.ndarray, boxes, rng: np.random.Generator):
    """Simulate AI removal + text inpainting artifacts and produce binary GT mask."""
    h, w = rgb.shape[:2]
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if not boxes:
        return rgb.copy(), gt_mask

    pick_n = min(2, len(boxes))
    chosen_idx = rng.choice(len(boxes), size=pick_n, replace=False)

    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in chosen_idx:
        x, y, bw, bh = boxes[int(idx)]
        pad = 2
        inpaint_mask[
            max(0, y - pad) : min(h, y + bh + pad), max(0, x - pad) : min(w, x + bw + pad)
        ] = 255

    edited = cv2.inpaint(rgb, inpaint_mask, 3, cv2.INPAINT_TELEA)
    for step_i, idx in enumerate(chosen_idx):
        x, y, bw, bh = boxes[int(idx)]
        if step_i == 1:
            fake_val = f"{rng.integers(10, 99)}.{rng.integers(10, 99)}"
            scale = max(0.35, min(0.85, bh / 32.0))
            cv2.putText(
                edited,
                fake_val,
                (x + 2, y + bh - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            patch = edited[y : y + bh, x : x + bw]
            edited[y : y + bh, x : x + bw] = cv2.GaussianBlur(patch, (3, 3), 0.6)
        gt_mask[y : y + bh, x : x + bw] = 1
    return edited, gt_mask


def multi_jpeg_extract(pil_rgb: Image.Image, qf_list, qt_dict):
    """Re-compress through JPEG stream matching ADCD-Net/ds.py and read DCT via jpeg2dct."""
    img = pil_rgb.convert("L")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
        for qf in qf_list:
            img.save(tmp.name, "JPEG", quality=int(qf))
            img.close()
            img = Image.open(tmp.name)
        img_rgb = Image.open(tmp.name).convert("RGB")
        dct_y, _, _ = jpeg2dct_load(tmp.name, normalized=False)

    rows, cols, _ = dct_y.shape
    dct = np.empty((8 * rows, 8 * cols), dtype=np.int32)
    for j in range(rows):
        for i in range(cols):
            dct[8 * j : 8 * (j + 1), 8 * i : 8 * (i + 1)] = dct_y[j, i].reshape(8, 8)
    qt = qt_dict[qf_list[-1]].long()
    return np.clip(np.abs(dct), 0, 20), img_rgb, qt.clamp(0, 63)


class SRMHighFreqBranch(nn.Module):
    """Fixed 3-kernel SRM high-pass residual extractor + zero-init residual modulation."""

    def __init__(self, out_channels: int = 96):
        super().__init__()
        f1 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ) / 4.0
        f2 = np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        ) / 12.0
        f3 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ) / 2.0

        kernels = np.stack([f1, f2, f3], axis=0)[:, None, :, :]
        weight = torch.from_numpy(np.repeat(kernels, 3, axis=1) / 3.0)
        self.register_buffer("srm_weight", weight)

        self.res_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
        )
        self.srm_scale = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)
        self.srm_bias = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.srm_scale.weight)
        nn.init.zeros_(self.srm_scale.bias)
        nn.init.zeros_(self.srm_bias.weight)
        nn.init.zeros_(self.srm_bias.bias)

    def forward(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        res = F.conv2d(img, self.srm_weight.to(dtype=img.dtype), padding=2)
        res = torch.clamp(res, -3.0, 3.0)
        srm_map = self.res_net(res)
        return feat * (1.0 + self.srm_scale(srm_map)) + self.srm_bias(srm_map)


class ADCDNetWithSRM(ADCDNet):
    """ADCD-Net augmented with zero-init SRM residual modulation (preserves ADCDNet.pth)."""

    def __init__(self):
        super().__init__()
        self.srm_branch = SRMHighFreqBranch(out_channels=96)

    def forward(self, img, dct, qt, mask, ocr_mask, is_train=True):
        img_size = img.size(2)
        as_feat, ms_dct_feats = self.dct_encoder(dct, qt)
        align_logits = self.dct_align_head(as_feat)
        align_score = F.softmax(align_logits, dim=1)[:, -1]

        feat, cnt_feats, frg_feats, pp_feats = self.restormer_loc(
            img=torch.cat([self.add_coord(img), torch.zeros_like(ocr_mask)], 1),
            ms_dct_feats=ms_dct_feats,
            dct_align_score=align_score,
        )

        pp_maps = self.get_pp_map(pp_feats, ocr_mask, img_size)
        feat = feat * self.pp_scale_proj(pp_maps) + self.pp_bias_proj(pp_maps)
        feat = self.srm_branch(img, feat)
        logits = self.out_head(feat)

        rec_items, focal_losses = None, None
        if is_train:
            shuffle_rec_img = self.restormer_rec(cnt_feats, frg_feats, is_shuffle=True)
            norm_dct = dct.float() / 20.0
            rec_items = (shuffle_rec_img, norm_dct)
            focal_losses = tuple(
                [supcon_parallel(self.focal_proj[i](pp_feats[i]), mask) for i in range(len(pp_feats))]
            )
        return logits, pp_feats, align_logits, rec_items, focal_losses


def enable_grad_checkpointing(model: nn.Module):
    """Wrap Restormer TransformerBlocks with activation checkpointing to fit 8GB VRAM."""
    for mod in model.modules():
        if isinstance(mod, TransformerBlock) and not getattr(mod, "_ckpt_wrapped", False):
            orig_forward = mod.forward

            def _ckpt_forward(x, _orig=orig_forward):
                if torch.is_grad_enabled() and x.requires_grad:
                    return checkpoint(_orig, x, use_reentrant=False)
                return _orig(x)

            mod.forward = _ckpt_forward
            mod._ckpt_wrapped = True


def load_qt_dict():
    with open(ROOT / "doctamper_root" / "exp_data" / "qt_table.pk", "rb") as f:
        raw = pickle.load(f)
    return {k: torch.LongTensor(v) for k, v in raw.items()}


def collect_datasets(qt_dict):
    """Gather disjoint evaluation and quick-adapt splits across T-SROIE, OSTF, and findit2."""
    # 1. T-SROIE: split strictly by receipt base ID so same receipt never appears in train & eval
    ts_dir = ROOT / "T-SROIE_test"
    ts_imgs = sorted((ts_dir / "images").glob("*.jpg"))
    ts_eval = ts_imgs[:50]
    ts_eval_ids = {p.stem.split("_")[0] for p in ts_eval}
    ts_train = [p for p in ts_imgs[50:] if p.stem.split("_")[0] not in ts_eval_ids][:60]

    # 2. OSTF: Diffusion (anytext, textdiff, udifftext, diffste) vs GAN (srnet, mostel, stefann)
    ostf_dir = ROOT / "OSTF_test"
    diff_prefixes = ("anytext", "textdiff", "udifftext", "diffste")
    gan_prefixes = ("srnet", "mostel", "stefann")
    ostf_diff_eval, ostf_gan_eval, ostf_train = [], [], []
    for pref in diff_prefixes:
        files = sorted((ostf_dir / "images").glob(f"{pref}_*.jpg"))
        ostf_diff_eval.extend(files[:10])
        ostf_train.extend(files[10:20])
    for pref in gan_prefixes:
        files = sorted((ostf_dir / "images").glob(f"{pref}_*.jpg"))
        ostf_gan_eval.extend(files[:13])
        ostf_train.extend(files[13:20])

    # 3. findit2/test manual CPI receipts
    findit_cpi = []
    with open(ROOT / "findit2" / "test.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["forged"] != "1" or row["forgery annotations"] == "0":
                continue
            ann = ast.literal_eval(row["forgery annotations"])
            boxes = []
            is_cpi = False
            for reg in ann.get("regions", []):
                ma = reg["region_attributes"].get("Modified area", {})
                if isinstance(ma, dict) and ma.get("CPI"):
                    is_cpi = True
                if reg["region_attributes"].get("Original area") == "no":
                    sa = reg["shape_attributes"]
                    boxes.append((sa["x"], sa["y"], sa["width"], sa["height"]))
            if is_cpi and boxes:
                findit_cpi.append((ROOT / "findit2" / "test" / row["image"], boxes))
                if len(findit_cpi) >= 15:
                    break

    # 4. Simulated AI-inpaint on findit2/train pristine receipts (eval=8, train=16 disjoint)
    pristine_names = []
    with open(ROOT / "findit2" / "train.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["forged"] == "0":
                pristine_names.append(row["image"])
                if len(pristine_names) >= 24:
                    break

    rng = np.random.default_rng(42)
    sim_dir = ROOT / "poc_out" / "sim_eval"
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_eval, sim_train = [], []
    for idx, name in enumerate(pristine_names):
        rgb = np.array(Image.open(ROOT / "findit2" / "train" / name).convert("RGB"))
        _, boxes = extract_text_mask_and_boxes(rgb)
        edited, gt_mask = simulate_ai_text_edit(rgb, boxes, rng)
        jpg_path = sim_dir / name.replace(".png", ".jpg")
        msk_path = sim_dir / name.replace(".png", "_mask.png")
        Image.fromarray(edited).save(jpg_path, "JPEG", quality=85)
        Image.fromarray(gt_mask * 255).save(msk_path)
        if idx < 8:
            sim_eval.append((jpg_path, msk_path))
        else:
            sim_train.append((jpg_path, msk_path))

    return {
        "tsroie_eval": [(p, ts_dir / "masks" / p.name.replace(".jpg", ".png")) for p in ts_eval],
        "ostf_diff_eval": [(p, ostf_dir / "masks" / p.name.replace(".jpg", ".png")) for p in ostf_diff_eval],
        "ostf_gan_eval": [(p, ostf_dir / "masks" / p.name.replace(".jpg", ".png")) for p in ostf_gan_eval],
        "findit_cpi_eval": findit_cpi,
        "sim_ai_eval": sim_eval,
        "train_pairs": (
            [(p, ts_dir / "masks" / p.name.replace(".jpg", ".png")) for p in ts_train]
            + [(p, ostf_dir / "masks" / p.name.replace(".jpg", ".png")) for p in ostf_train]
            + sim_train
        ),
    }


def prepare_eval_sample(img_path, mask_or_boxes, qt_dict, max_size: int = 768):
    """Load RGB image & binary GT mask, resize if > max_size, pad square //16, extract DCT."""
    rgb = np.array(Image.open(img_path).convert("RGB"))
    h, w = rgb.shape[:2]

    if isinstance(mask_or_boxes, list):
        gt = np.zeros((h, w), dtype=np.uint8)
        for x, y, bw, bh in mask_or_boxes:
            gt[y : y + bh, x : x + bw] = 1
    else:
        gt = (cv2.imread(str(mask_or_boxes), 0) != 0).astype(np.uint8)

    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        nh, nw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)
        h, w = nh, nw

    ocr_mask, _ = extract_text_mask_and_boxes(rgb)
    dct, img_pil, qt = multi_jpeg_extract(Image.fromarray(rgb), [100], qt_dict)
    dct = dct[:h, :w]
    t_img = IMG_NORM(img_pil)
    t_dct = torch.from_numpy(dct).long()
    t_ocr = torch.from_numpy(ocr_mask).unsqueeze(0).long()

    s_max = ((max(h, w) + 15) // 16) * 16
    t_img_p = F.pad(t_img, (0, s_max - t_img.shape[-1], 0, s_max - t_img.shape[-2]), value=0.0).unsqueeze(0)
    t_dct_p = F.pad(t_dct, (0, s_max - t_dct.shape[-1], 0, s_max - t_dct.shape[-2]), value=0).unsqueeze(0)
    t_ocr_p = F.pad(t_ocr, (0, s_max - t_ocr.shape[-1], 0, s_max - t_ocr.shape[-2]), value=0).unsqueeze(0)
    t_qt = qt.unsqueeze(0)
    t_dummy_m = torch.zeros_like(t_ocr_p)

    return t_img_p, t_dct_p, t_qt, t_dummy_m, t_ocr_p, gt, rgb


@torch.no_grad()
def evaluate_subset(model, items, qt_dict, max_size: int = 768, keep_vis: int = 0):
    """Evaluate Pixel-F1 and IoU matching ADCD-Net/main.py protocol (forged images)."""
    model.eval()
    f1s, ious = [], []
    vis_items = []
    eps = 1e-8

    for idx, (img_path, gt_ref) in enumerate(items):
        t_img, t_dct, t_qt, t_m, t_ocr, gt_np, rgb_np = prepare_eval_sample(
            img_path, gt_ref, qt_dict, max_size=max_size
        )
        h, w = gt_np.shape
        if gt_np.sum() == 0:
            continue

        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                t_img.cuda(), t_dct.cuda(), t_qt.cuda(), t_m.cuda(), t_ocr.cuda(), is_train=False
            )[0]

        crop_logit = logits[0, :, :h, :w]
        prob = F.softmax(crop_logit.float(), dim=0)[1].cpu().numpy()
        pred = (prob >= 0.5).astype(np.uint8)

        inter = float((pred * gt_np).sum())
        pred_s = float(pred.sum())
        gt_s = float(gt_np.sum())
        union = pred_s + gt_s - inter
        p = inter / (pred_s + eps)
        r = inter / (gt_s + eps)
        f1 = 2 * p * r / (p + r + eps)
        iou = inter / (union + eps)
        f1s.append(f1)
        ious.append(iou)

        if idx < keep_vis:
            vis_items.append((img_path.name, rgb_np, gt_np, prob, f1))

    return float(np.mean(f1s)), float(np.mean(ious)), vis_items


def make_train_batch(train_pairs, indices, qt_dict, rng, crop_size: int = 256):
    """Sample 256x256 JPEG-augmented training crops centered on tampered pixels."""
    imgs, dcts, qts, masks, ocr_masks = [], [], [], [], []
    for idx in indices:
        img_path, msk_path = train_pairs[idx % len(train_pairs)]
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gt = (cv2.imread(str(msk_path), 0) != 0).astype(np.uint8)
        h, w = gt.shape
        if h < crop_size or w < crop_size:
            ph, pw = max(0, crop_size - h), max(0, crop_size - w)
            rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            gt = cv2.copyMakeBorder(gt, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            h, w = gt.shape

        ys, xs = np.where(gt > 0)
        if len(ys):
            pick = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[pick]), int(xs[pick])
        else:
            cy, cx = h // 2, w // 2
        y0 = max(0, min(h - crop_size, cy - crop_size // 2))
        x0 = max(0, min(w - crop_size, cx - crop_size // 2))
        y0, x0 = (y0 // 8) * 8, (x0 // 8) * 8

        p_rgb = rgb[y0 : y0 + crop_size, x0 : x0 + crop_size]
        p_gt = gt[y0 : y0 + crop_size, x0 : x0 + crop_size]
        p_ocr, _ = extract_text_mask_and_boxes(p_rgb)

        qf = int(rng.integers(75, 100))
        dct, img_pil, qt = multi_jpeg_extract(Image.fromarray(p_rgb), [qf], qt_dict)
        imgs.append(IMG_NORM(img_pil))
        dcts.append(torch.from_numpy(dct).long())
        qts.append(qt)
        masks.append(torch.from_numpy(p_gt).unsqueeze(0).long())
        ocr_masks.append(torch.from_numpy(p_ocr).unsqueeze(0).long())

    return (
        torch.stack(imgs).cuda(),
        torch.stack(dcts).cuda(),
        torch.stack(qts).cuda(),
        torch.stack(masks).cuda(),
        torch.stack(ocr_masks).cuda(),
        torch.ones(len(indices), dtype=torch.long).cuda(),
    )


def quick_finetune(model, train_pairs, qt_dict, steps: int = 30, batch_size: int = 4):
    """Quick 30-step adaptation of ADCDNetWithSRM on RTX 4060 using gradient checkpointing."""
    enable_grad_checkpointing(model)
    model.train()

    backbone_params, new_params = [], []
    for name, p in model.named_parameters():
        if name.startswith(("srm_branch.", "out_head.")):
            new_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 1e-5}, {"params": new_params, "lr": 1e-4}],
        weight_decay=1e-4,
    )
    scaler = GradScaler()
    loc_ce = SoftCrossEntropyLoss(smooth_factor=0.05, reduction="mean", ignore_index=None)
    loc_lovasz = LovaszLoss(mode="multiclass", per_image=True)
    rec_l1 = nn.L1Loss()
    align_ce = nn.CrossEntropyLoss()

    rng = np.random.default_rng(123)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()

    for step in range(steps):
        idx_list = rng.choice(len(train_pairs), size=batch_size, replace=False).tolist()
        img, dct, qt, mask, ocr_mask, is_align = make_train_batch(train_pairs, idx_list, qt_dict, rng)

        with autocast(device_type="cuda", dtype=torch.float16):
            logits, norm_feats, align_logits, rec_items, focal_losses = model(
                img, dct, qt, mask, ocr_mask, is_train=True
            )
        rec, norm_dct = rec_items
        rec_loss = rec_l1(rec[:, :3], img) + rec_l1(rec[:, -1], norm_dct)
        norm_loss = 0.1 * torch.stack([f.norm(dim=1).mean() for f in norm_feats]).mean()
        align_loss = align_ce(align_logits, is_align)
        valid_f = [
            0.2 * (loss.sum() / (loss != 0).sum().clamp_min(1)) for loss in focal_losses
        ]
        focal_loss = torch.stack(valid_f).sum()
        ce_loss = 3.0 * loc_ce(logits.float(), mask)
        iou_loss = loc_lovasz(logits.float(), mask)
        total_loss = ce_loss + iou_loss + rec_loss + align_loss + focal_loss + norm_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(total_loss.item()))

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    print(
        f"\n[Fine-tune 30 steps | bs={batch_size}, fp16 + grad-ckpt] Done in {dt:.1f}s | "
        f"Peak VRAM: {peak_mib:.1f} MiB | Loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def save_vis_grid(before_vis, after_vis, out_path: Path):
    """Save 5-column visual comparison grid (Input, GT, ADCD-Net Zero-shot, Adapted+SRM, Overlay)."""
    cell = 256
    rows = []
    for (name, rgb, gt, p_before, f1_b), (_, _, _, p_after, f1_a) in zip(before_vis, after_vis):
        in_p = cv2.resize(rgb, (cell, cell))
        gt_p = cv2.cvtColor((cv2.resize(gt, (cell, cell), interpolation=cv2.INTER_NEAREST) * 255), cv2.COLOR_GRAY2BGR)

        def heat_panel(prob, label):
            u8 = np.uint8(np.clip(prob, 0, 1) * 255)
            hm = cv2.applyColorMap(cv2.resize(u8, (cell, cell)), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            cv2.putText(hm, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            return hm

        p1 = heat_panel(p_before, f"Zero F1:{f1_b:.2f}")
        p2 = heat_panel(p_after, f"Adapt F1:{f1_a:.2f}")
        ov = cv2.addWeighted(in_p, 0.65, p2, 0.35, 0)
        cv2.putText(in_p, name[:18], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 30, 30), 2)
        cv2.putText(gt_p, "Ground Truth", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        rows.append(np.hstack([in_p, gt_p, p1, p2, ov]))

    grid = np.vstack(rows)
    Image.fromarray(grid).save(out_path)
    print(f"[Visual] Saved comparison grid to {out_path.relative_to(ROOT)}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")
    qt_dict = load_qt_dict()
    datasets = collect_datasets(qt_dict)
    print(
        f"[Data] Eval sizes -> T-SROIE:{len(datasets['tsroie_eval'])}, "
        f"OSTF-Diff:{len(datasets['ostf_diff_eval'])}, OSTF-GAN:{len(datasets['ostf_gan_eval'])}, "
        f"FindIt2-CPI:{len(datasets['findit_cpi_eval'])}, Sim-AI:{len(datasets['sim_ai_eval'])} | "
        f"Quick-Adapt Train pool:{len(datasets['train_pairs'])}"
    )

    model = ADCDNetWithSRM().cuda()
    ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[Weights] Loaded ADCDNet.pth | Missing (new SRM only): {len(miss)} | Unexpected: {len(unexp)}")

    eval_suites = [
        ("T-SROIE (GAN)", datasets["tsroie_eval"], 512, 2),
        ("OSTF (Diffusion: AnyText/TextDiff/etc.)", datasets["ostf_diff_eval"], 512, 2),
        ("OSTF (GAN: SRNet/MOSTEL/STEFANN)", datasets["ostf_gan_eval"], 512, 0),
        ("FindIt2 (Manual CPI, max=768)", datasets["findit_cpi_eval"], 768, 1),
        ("Sim-AI on FindIt2 (max=768)", datasets["sim_ai_eval"], 768, 1),
    ]

    print("\n=== ZERO-SHOT ADCDNet.pth BENCHMARK ===")
    zero_scores, vis_before = {}, []
    for label, items, msz, n_vis in eval_suites:
        f1, iou, vis = evaluate_subset(model, items, qt_dict, max_size=msz, keep_vis=n_vis)
        zero_scores[label] = (f1, iou)
        vis_before.extend(vis)
        print(f"  {label:<42} | Pixel-F1: {f1:.4f} | IoU: {iou:.4f}")

    quick_finetune(model, datasets["train_pairs"], qt_dict, steps=30, batch_size=4)

    print("\n=== AFTER 30-STEP ADAPT (ADCDNet + SRM on 4060) ===")
    vis_after = []
    for label, items, msz, n_vis in eval_suites:
        f1, iou, vis = evaluate_subset(model, items, qt_dict, max_size=msz, keep_vis=n_vis)
        f1_0, iou_0 = zero_scores[label]
        vis_after.extend(vis)
        print(
            f"  {label:<42} | Pixel-F1: {f1:.4f} ({f1 - f1_0:+.4f}) | "
            f"IoU: {iou:.4f} ({iou - iou_0:+.4f})"
        )

    save_vis_grid(vis_before, vis_after, ROOT / "poc_out" / "vis_comparison.png")


if __name__ == "__main__":
    main()
