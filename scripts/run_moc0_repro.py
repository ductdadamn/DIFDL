#!/usr/bin/env python3
"""Milestone 0 (Moc 0): ADCD-Net Sanity-Check Reproduction on 20% DocTamperV1 (20 Epochs).

Implements the advisor reproduction requirement in MASTER_PLAN.md (Section IV.1, Task #17):
  - Deterministic 20% subset of DocTamperV1-TrainingSet (24,000 / 120,000 samples, seed=42),
    saved to subset_20pct_ids.json for full reproducibility.
  - Scratch training initialized ONLY from backbone `weights/docres.pkl` (does NOT load
    pre-trained `ADCDNet.pth`).
  - Exact 6-loss combination from `src/main.py` (SoftCE + Lovasz + Rec L1 + Align CE +
    Focal SupCon + Feature Norm) with safe NaN-free focal denominator.
  - Designed for shared 8GB VRAM GPUs (e.g., 1x or 2x Quadro RTX 4000 on ict14):
      * Activation checkpointing (`--grad-ckpt`, enabled by default) + FP16 AMP
      * Single-GPU (`--gpus 0`) or Multi-GPU DDP (`--gpus 0,1`) with auto free MASTER_PORT
      * Auto-resume from `ckpt/latest.pth` and disk-quota-safe checkpointing (`latest` + `best`)
      * Per-epoch validation on DocTamperV1 (`TestingSet`, `FCD`, `SCD` samples), TensorBoard
        logging (`tb_log/`), `metrics_epoch.csv`, and auto-exported 300 DPI `moc0_curves.png`.

Usage examples on ict14 (/storage):
  # 1. Pre-flight check (verifies symlinks/paths & prints dataset counts, 0 GPU memory used):
  python scripts/run_moc0_repro.py --check-only

  # 2. Full 20-epoch reproduction on 2 GPUs (when both GPUs 0,1 have >=4.5GB free):
  python scripts/run_moc0_repro.py --gpus 0,1 --epochs 20 --subset-ratio 0.20 --train-bs 4 --accum-step 2

  # 3. Single-GPU mode (if 1 GPU on ict14 is busy with another team's job):
  python scripts/run_moc0_repro.py --gpus 0 --epochs 20 --subset-ratio 0.20 --train-bs 4 --accum-step 4
"""

import argparse
import copy
import csv
import json
import logging
import math
import os
import os.path as op
import random
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Moc 0: ADCD-Net 20% DocTamperV1 Reproduction")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs, e.g. '0,1' or '0'")
    parser.add_argument("--doctamper-root", type=str, default=str(repo_root / "doctamper_root"),
                        help="Path to doctamper_root containing DocTamperV1/ and exp_data/")
    parser.add_argument("--docres-ckpt", type=str, default=str(repo_root / "weights" / "docres.pkl"),
                        help="Path to DocRes backbone checkpoint (docres.pkl)")
    parser.add_argument("--out-dir", type=str, default=str(repo_root / "exp_out" / "moc0_repro_20pct_20ep"),
                        help="Output directory for checkpoints, TensorBoard logs, CSV, and PNG curves")
    parser.add_argument("--subset-ratio", type=float, default=0.20, help="Fraction of DocTamperV1-TrainingSet (default: 0.20)")
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Optional cap on samples per epoch (0 = use full subset, e.g. 24,000)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--train-bs", type=int, default=4,
                        help="Batch size per GPU (must be >=2 due to BatchNorm1d in dct_align_head)")
    parser.add_argument("--accum-step", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--val-bs", type=int, default=2, help="Validation batch size per GPU (512x512 full-res)")
    parser.add_argument("--val-samples", type=int, default=100, help="Samples per DocTamper val split per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial AdamW learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="CosineAnnealing minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes per GPU")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--print-freq", type=int, default=100, help="Log every N training steps")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable Restormer activation checkpointing")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh even if ckpt/latest.pth exists")
    parser.add_argument("--master-port", type=int, default=0, help="DDP master port (0 = auto-find free port)")
    parser.add_argument("--check-only", action="store_true", help="Verify paths/LMDBs and exit without using GPU")
    return parser.parse_args()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def preflight_check(args) -> int:
    """Validate DocTamperV1 LMDBs, quantization tables, JPEG records, and docres.pkl."""
    import lmdb

    dt_root = Path(args.doctamper_root)
    v1_root = dt_root / "DocTamperV1"
    qt_path = dt_root / "exp_data" / "qt_table.pk"
    pks_dir = dt_root / "exp_data" / "pks"
    docres_path = Path(args.docres_ckpt)

    missing = []
    for label, p in [
        ("DocTamperV1-TrainingSet", v1_root / "DocTamperV1-TrainingSet"),
        ("DocTamperV1-TestingSet", v1_root / "DocTamperV1-TestingSet"),
        ("DocTamperV1-FCD", v1_root / "DocTamperV1-FCD"),
        ("DocTamperV1-SCD", v1_root / "DocTamperV1-SCD"),
        ("qt_table.pk", qt_path),
        ("exp_data/pks", pks_dir),
        ("docres.pkl", docres_path),
    ]:
        if not p.exists():
            missing.append(f"  - {label}: {p}")

    if missing:
        print("=" * 72)
        print("[ERROR] Missing required files/directories for Moc 0 Reproduction:")
        print("\n".join(missing))
        print("-" * 72)
        print("Tip: Put datasets in /storage/<user>/difdl_ds/datasets/ and symlink them:")
        print("  ln -s /path/to/doctamper_root /storage/<user>/difdl_ds/DIFDL/doctamper_root")
        print("  ln -s /path/to/weights        /storage/<user>/difdl_ds/DIFDL/weights")
        print("=" * 72)
        sys.exit(1)

    if args.train_bs < 2:
        raise ValueError(f"--train-bs must be >= 2 (BatchNorm1d in dct_align_head), got {args.train_bs}")

    env = lmdb.open(str(v1_root / "DocTamperV1-TrainingSet"), max_readers=8, readonly=True,
                    lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        total_train = int(txn.get(b"num-samples"))
    env.close()

    subset_n = max(args.train_bs * len(args.gpus.split(",")), int(round(total_train * args.subset_ratio)))
    if args.max_train_samples > 0:
        subset_n = min(subset_n, args.max_train_samples)

    print("=" * 72)
    print("MOC 0 REPRODUCTION PRE-FLIGHT CHECK PASSED")
    print(f"  DocTamperV1-TrainingSet total : {total_train:,} samples")
    print(f"  Selected subset ({args.subset_ratio*100:.1f}%, seed={args.seed}) : {subset_n:,} samples/epoch")
    print(f"  Epochs                        : {args.epochs}")
    print(f"  GPUs                          : {args.gpus} (train_bs={args.train_bs}/GPU, accum={args.accum_step})")
    print(f"  DocRes init checkpoint        : {docres_path}")
    print(f"  Output directory              : {args.out_dir}")
    print("=" * 72)
    return total_train


def build_or_load_subset_indices(out_dir: Path, total_train: int, subset_ratio: float, max_samples: int, seed: int):
    subset_file = out_dir / "subset_20pct_ids.json"
    target_n = max(2, int(round(total_train * subset_ratio)))
    if max_samples > 0:
        target_n = min(target_n, max_samples)

    if subset_file.exists():
        payload = json.loads(subset_file.read_text(encoding="utf-8"))
        indices = payload["indices"]
        if len(indices) == target_n and payload.get("seed") == seed and payload.get("total_train") == total_train:
            return indices

    rng = np.random.default_rng(seed)
    indices = sorted(int(x) for x in rng.choice(total_train, size=target_n, replace=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_file.write_text(
        json.dumps(
            {
                "total_train": total_train,
                "subset_ratio": subset_ratio,
                "subset_count": len(indices),
                "seed": seed,
                "indices": indices,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return indices


def plot_moc0_curves(history_rows, out_png: Path):
    if not history_rows:
        return
    epochs = [int(r["epoch"]) for r in history_rows]
    total_loss = [float(r["loss_total"]) for r in history_rows]
    ce_loss = [float(r["loss_ce"]) for r in history_rows]
    iou_loss = [float(r["loss_iou"]) for r in history_rows]
    rec_loss = [float(r["loss_rec"]) for r in history_rows]

    f1_mean = [float(r["val_f1_mean"]) for r in history_rows]
    f1_test = [float(r["val_f1_TestingSet"]) for r in history_rows]
    f1_fcd = [float(r["val_f1_FCD"]) for r in history_rows]
    f1_scd = [float(r["val_f1_SCD"]) for r in history_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax1.plot(epochs, total_loss, "o-", lw=2.2, label="Total Loss")
    ax1.plot(epochs, ce_loss, "s--", lw=1.4, label="Loc SoftCE (x3)")
    ax1.plot(epochs, iou_loss, "^--", lw=1.4, label="Loc Lovasz")
    ax1.plot(epochs, rec_loss, "d--", lw=1.4, label="Rec L1 (RGB+DCT)")
    ax1.set_title("Moc 0 Reproduction: Training Loss (20% DocTamperV1)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fontsize=9)

    ax2.plot(epochs, f1_mean, "o-", lw=2.4, color="#d62728", label="Mean Val Pixel-F1")
    ax2.plot(epochs, f1_fcd, "s--", lw=1.4, color="#1f77b4", label="FCD (sample)")
    ax2.plot(epochs, f1_scd, "^--", lw=1.4, color="#2ca02c", label="SCD (sample)")
    ax2.plot(epochs, f1_test, "d--", lw=1.4, color="#9467bd", label="TestingSet (sample)")
    ax2.set_title("Moc 0 Reproduction: Validation Pixel-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pixel-F1")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_main(rank: int, world_size: int, args, subset_indices, master_port: int):
    # Set visible GPUs BEFORE importing src/cfg or initializing CUDA context
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.amp import GradScaler, autocast
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.checkpoint import checkpoint
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_distributed = world_size > 1

    if is_distributed:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    seed_everything(args.seed + rank)

    # Configure `cfg` dynamically BEFORE importing `ds` and `model`
    import cfg

    cfg.gpus = args.gpus
    cfg.device_n = world_size
    cfg.mode = "train"
    cfg.train_bs = args.train_bs
    cfg.val_bs = args.val_bs
    cfg.accum_step = args.accum_step
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.min_lr = args.min_lr
    cfg.weight_decay = args.weight_decay
    cfg.dl_workers = args.workers
    cfg.val_sample_n = args.val_samples
    cfg.root = args.doctamper_root
    cfg.data_root = op.join(cfg.root, "DocTamperV1")
    cfg.ocr_root = op.join(cfg.root, "DocTamperData")
    cfg.qt_path = op.join(cfg.root, "exp_data", "qt_table.pk")
    cfg.jpeg_record_dir = op.join(cfg.root, "exp_data", "pks")
    cfg.docres_ckpt_path = args.docres_ckpt
    cfg.ckpt = None  # Scratch reproduction: ONLY load docres.pkl, NOT ADCDNet.pth!
    cfg.val_name_list = ["TestingSet_sample", "FCD_sample", "SCD_sample"]

    from ds import DtdValDs, bbox_2_mask, get_align_aug, get_non_align_aug, img_totsr, load_data, load_or_fallback_bboxes, load_qt, mask_totsr, multi_jpeg, pad_collate
    import lmdb
    from PIL import Image
    from loss.lovasz_loss import LovaszLoss
    from loss.soft_ce_loss import SoftCrossEntropyLoss
    from model.model import ADCDNet
    from model.restormer import TransformerBlock
    from utils import AverageMeter

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(levelname)s %(asctime)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    class DocTamperSubsetTrainDs(Dataset):
        """Deterministic 20% subset of DocTamperV1-TrainingSet with clean per-index sampling."""

        def __init__(self, indices):
            lmdb_path = op.join(cfg.data_root, "DocTamperV1-TrainingSet")
            self.lmdb_path = lmdb_path
            self.env = None
            self.indices = list(indices)
            self.ocr_dir = op.join(cfg.ocr_root, "TrainingSet/char_seg")
            self.qts = load_qt(cfg.qt_path)
            self.cur_epoch = 1
            self.align_aug = get_align_aug()
            self.non_align_aug = get_non_align_aug()

        def _get_env(self):
            if self.env is None:
                self.env = lmdb.open(
                    self.lmdb_path,
                    max_readers=64,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
            return self.env

        def set_epoch(self, epoch: int):
            self.cur_epoch = epoch

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_index = self.indices[idx]
            img_name = "%06d" % real_index
            env = self._get_env()
            with env.begin(write=False) as txn:
                img, mask = load_data(real_index, txn)

            img_np = np.array(img.convert("RGB"))
            c_bbox = load_or_fallback_bboxes(img_np, self.ocr_dir, img_name)

            if random.random() > 0.5:
                aug_out = self.align_aug(image=img_np, mask=mask, bboxes=c_bbox)
                is_align = True
            else:
                aug_out = self.non_align_aug(image=img_np, mask=mask, bboxes=c_bbox)
                is_align = False

            img_aug, mask_aug, c_bbox_aug = aug_out["image"], aug_out["mask"], aug_out["bboxes"]
            h, w = mask_aug.shape
            ocr_mask = bbox_2_mask(c_bbox_aug, h, w)
            img_pil = Image.fromarray(img_aug)

            # Match original ADCD-Net progressive JPEG curriculum: 100 -> 75 across epochs
            min_qf = max(100 - (self.cur_epoch - 1), 75)
            num_jpeg = random.randint(1, 3)
            dct, img_jpeg, qfs = multi_jpeg(
                copy.deepcopy(img_pil),
                num_jpeg=num_jpeg,
                min_qf=min_qf,
                upper_bound=100,
            )
            qt = self.qts[qfs[-1]]
            img_tensor = img_totsr(img_jpeg)
            mask_tensor = mask_totsr(image=mask_aug.copy())["image"]
            ocr_tensor = mask_totsr(image=ocr_mask.copy())["image"]

            return {
                "img": img_tensor,
                "dct": np.clip(np.abs(dct), 0, 20),
                "qt": qt,
                "mask": mask_tensor.long(),
                "ocr_mask": ocr_tensor.long(),
                "img_name": img_name,
                "min_qf": min_qf,
                "is_align": is_align,
            }

    def enable_grad_checkpointing(model: nn.Module):
        for mod in model.modules():
            if isinstance(mod, TransformerBlock) and not getattr(mod, "_ckpt_wrapped", False):
                orig_forward = mod.forward

                def _ckpt_forward(x, _orig=orig_forward):
                    if torch.is_grad_enabled() and x.requires_grad:
                        return checkpoint(_orig, x, use_reentrant=False)
                    return _orig(x)

                mod.forward = _ckpt_forward
                mod._ckpt_wrapped = True

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "ckpt"
    tb_dir = out_dir / "tb_log"
    csv_path = out_dir / "metrics_epoch.csv"
    png_path = out_dir / "moc0_curves.png"

    tb_writer = None
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))
        (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Build train & val dataloaders
    train_ds = DocTamperSubsetTrainDs(subset_indices)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if is_distributed else None
    train_dl = DataLoader(
        train_ds,
        batch_size=args.train_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dls = {}
    for val_raw in ["TestingSet", "FCD", "SCD"]:
        v_ds = DtdValDs(val_raw, is_sample=True)
        v_sampler = DistributedSampler(v_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        val_dls[val_raw] = DataLoader(
            v_ds,
            batch_size=args.val_bs,
            shuffle=False,
            sampler=v_sampler,
            num_workers=max(1, args.workers // 2) if args.workers > 0 else 0,
            collate_fn=pad_collate,
        )

    # Model initialization (from docres.pkl ONLY)
    raw_model = ADCDNet()
    if not args.no_grad_ckpt:
        enable_grad_checkpointing(raw_model)
        if rank == 0:
            logging.info("Enabled Restormer activation checkpointing for 8GB VRAM safety.")
    raw_model = raw_model.to(device)
    model = DDP(raw_model, device_ids=[rank], find_unused_parameters=True) if is_distributed else raw_model

    opt_steps_per_epoch = math.ceil(len(train_dl) / args.accum_step)
    total_opt_steps = max(1, opt_steps_per_epoch * args.epochs)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=args.min_lr)
    scaler = GradScaler("cuda")

    loc_ce = SoftCrossEntropyLoss(smooth_factor=0.1, reduction="mean", ignore_index=None)
    loc_lovasz = LovaszLoss(mode="multiclass", per_image=True)
    rec_l1 = nn.L1Loss()
    align_ce = nn.CrossEntropyLoss()
    eps = 1e-8

    start_epoch = 1
    global_step = 0
    best_f1 = -1.0
    history_rows = []

    latest_ckpt = ckpt_dir / "latest.pth"
    if not args.no_resume and latest_ckpt.exists():
        cp = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        target_mod = model.module if is_distributed else model
        target_mod.load_state_dict(cp["model"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        scheduler.T_max = total_opt_steps
        scaler.load_state_dict(cp["scaler"])
        start_epoch = int(cp["epoch"]) + 1
        global_step = int(cp["global_step"])
        best_f1 = float(cp.get("best_f1", -1.0))
        if rank == 0 and csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                history_rows = list(csv.DictReader(f))
        if rank == 0:
            logging.info(f"Resumed from {latest_ckpt} at epoch {start_epoch} (best_f1={best_f1:.4f})")

    @torch.no_grad()
    def compute_f1(logit, y):
        pred = logit.argmax(1)
        y_ = y.squeeze(1)
        matched = (pred * y_).sum((1, 2))
        pred_sum = pred.sum((1, 2))
        y_sum = y_.sum((1, 2))
        p = (matched / (pred_sum + eps)).mean().item()
        r = (matched / (y_sum + eps)).mean().item()
        f1 = 2 * p * r / (p + r + eps)
        return f1

    @torch.no_grad()
    def evaluate():
        torch.cuda.empty_cache()
        model.eval()
        scores = {}
        for val_name, dl in val_dls.items():
            sum_f1 = torch.zeros(1, device=device)
            num_images = torch.zeros(1, device=device)
            for items in dl:
                img = items[0].to(device, non_blocking=True)
                dct = items[1].to(device, non_blocking=True)
                qt = items[2].to(device, non_blocking=True)
                mask = items[3].to(device, non_blocking=True)
                ocr_mask = items[4].to(device, non_blocking=True)
                img_names = items[5]
                sizes = items[6]

                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(img, dct, qt, mask, ocr_mask, is_train=False)[0]

                for logit, each_y, (h, w), name in zip(logits, mask, sizes, img_names):
                    if name != "padding":
                        crop_logit = logit[..., :h, :w].unsqueeze(0)
                        crop_y = each_y[..., :h, :w].unsqueeze(0)
                        sum_f1 += compute_f1(crop_logit, crop_y)
                        num_images += 1.0

            if is_distributed:
                dist.all_reduce(sum_f1, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_images, op=dist.ReduceOp.SUM)
            scores[val_name] = (sum_f1.item() / max(1.0, num_images.item()))
        model.train()
        scores["mean"] = float(np.mean([scores["TestingSet"], scores["FCD"], scores["SCD"]]))
        return scores

    if rank == 0:
        logging.info(
            f"Starting Moc 0 Reproduction: epochs {start_epoch}..{args.epochs}, "
            f"steps/epoch={len(train_dl)}, world_size={world_size}"
        )

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.time()
        train_ds.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        meters = defaultdict(AverageMeter)
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}/{args.epochs:02d}", disable=(rank != 0))

        for step_in_ep, items in enumerate(pbar, start=1):
            global_step += 1
            img = items["img"].to(device, non_blocking=True)
            dct = items["dct"].to(device, non_blocking=True)
            qt = items["qt"].to(device, non_blocking=True)
            mask = items["mask"].to(device, non_blocking=True)
            ocr_mask = items["ocr_mask"].to(device, non_blocking=True)
            is_align = items["is_align"].to(device, non_blocking=True)
            min_qf_val = float(items["min_qf"][0])

            with autocast(device_type="cuda", dtype=torch.float16):
                logits, norm_feats, align_logits, rec_items, focal_losses = model(
                    img, dct, qt, mask, ocr_mask, is_train=True
                )
                rec, norm_dct = rec_items
                rec_loss = cfg.rec_w * (rec_l1(rec[:, :3], img) + rec_l1(rec[:, -1], norm_dct))
                norm_loss = cfg.norm_w * torch.stack([ft.norm(dim=1).mean() for ft in norm_feats]).mean()
                align_loss = align_ce(align_logits, is_align.long())
                # Safe clamp_min(1) avoids 0/0 NaN when a random crop has <3 tampered pixels at 1/16 scale
                focal_terms = [cfg.focal_w * (fl.sum() / (fl != 0).sum().clamp_min(1)) for fl in focal_losses]
                focal_loss = torch.stack(focal_terms).sum()
                ce_loss = cfg.ce_w * loc_ce(logits.float(), mask)
                iou_loss = loc_lovasz(logits.float(), mask)
                total_loss = ce_loss + iou_loss + rec_loss + align_loss + focal_loss + norm_loss

            scaler.scale(total_loss / args.accum_step).backward()

            if step_in_ep % args.accum_step == 0 or step_in_ep == len(train_dl):
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    scheduler.step()

            with torch.no_grad():
                tr_f1 = compute_f1(logits.detach(), mask)
                align_acc = (align_logits.detach().argmax(1) == is_align).float().mean().item()

            stat_dict = {
                "total": total_loss.item(),
                "ce": ce_loss.item(),
                "iou": iou_loss.item(),
                "rec": rec_loss.item(),
                "align_ce": align_loss.item(),
                "focal": focal_loss.item(),
                "norm": norm_loss.item(),
                "train_f1": tr_f1,
                "align_acc": align_acc,
                "min_qf": min_qf_val,
            }

            for k, v in stat_dict.items():
                t_v = torch.tensor(v, device=device)
                if is_distributed:
                    dist.reduce(t_v, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    meters[k].update(t_v.item() / world_size)

            if rank == 0 and (global_step % args.print_freq == 0 or step_in_ep == len(train_dl)):
                cur_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{meters['total'].avg:.3f}", f1=f"{meters['train_f1'].avg:.3f}", lr=f"{cur_lr:.1e}")
                for k, m in meters.items():
                    tb_writer.add_scalar(f"step_losses/{k}", m.val, global_step)
                tb_writer.add_scalar("step_losses/lr", cur_lr, global_step)

        # Free training batch tensors before 512x512 full-res validation
        del img, dct, qt, mask, ocr_mask, logits, norm_feats, align_logits, rec_items, focal_losses
        val_scores = evaluate()
        torch.cuda.empty_cache()
        ep_sec = time.time() - ep_t0

        if rank == 0:
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            cur_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"[Epoch {epoch:02d}/{args.epochs:02d}] {ep_sec/60:.1f} min | PeakVRAM={peak_gb:.2f}GB | "
                f"Loss={meters['total'].avg:.4f} (CE={meters['ce'].avg:.4f}, IoU={meters['iou'].avg:.4f}) | "
                f"Val F1: Mean={val_scores['mean']:.4f} "
                f"(Test={val_scores['TestingSet']:.4f}, FCD={val_scores['FCD']:.4f}, SCD={val_scores['SCD']:.4f})"
            )

            for k, m in meters.items():
                tb_writer.add_scalar(f"epoch_losses/{k}", m.avg, epoch)
            tb_writer.add_scalar("val/F1_mean", val_scores["mean"], epoch)
            tb_writer.add_scalar("val/F1_TestingSet", val_scores["TestingSet"], epoch)
            tb_writer.add_scalar("val/F1_FCD", val_scores["FCD"], epoch)
            tb_writer.add_scalar("val/F1_SCD", val_scores["SCD"], epoch)
            tb_writer.flush()

            row = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": f"{cur_lr:.6e}",
                "min_qf": int(meters["min_qf"].avg),
                "loss_total": f"{meters['total'].avg:.6f}",
                "loss_ce": f"{meters['ce'].avg:.6f}",
                "loss_iou": f"{meters['iou'].avg:.6f}",
                "loss_rec": f"{meters['rec'].avg:.6f}",
                "loss_align": f"{meters['align_ce'].avg:.6f}",
                "loss_focal": f"{meters['focal'].avg:.6f}",
                "loss_norm": f"{meters['norm'].avg:.6f}",
                "train_f1": f"{meters['train_f1'].avg:.6f}",
                "val_f1_TestingSet": f"{val_scores['TestingSet']:.6f}",
                "val_f1_FCD": f"{val_scores['FCD']:.6f}",
                "val_f1_SCD": f"{val_scores['SCD']:.6f}",
                "val_f1_mean": f"{val_scores['mean']:.6f}",
                "epoch_sec": f"{ep_sec:.1f}",
                "peak_vram_gb": f"{peak_gb:.2f}",
            }
            history_rows = [r for r in history_rows if int(r["epoch"]) != epoch]
            history_rows.append(row)
            history_rows.sort(key=lambda r: int(r["epoch"]))

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(history_rows)

            plot_moc0_curves(history_rows, png_path)

            target_mod = model.module if is_distributed else model
            is_best = val_scores["mean"] > best_f1
            if is_best:
                best_f1 = val_scores["mean"]

            ckpt_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "best_f1": best_f1,
                "val_scores": val_scores,
                "model": target_mod.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(ckpt_payload, latest_ckpt)
            if is_best:
                torch.save(ckpt_payload, ckpt_dir / "best.pth")
                logging.info(f"  -> Saved new best checkpoint (Mean Val F1 = {best_f1:.4f})")

    if rank == 0 and tb_writer is not None:
        tb_writer.close()
        logging.info(f"Moc 0 Reproduction finished! Best Mean Val F1 = {best_f1:.4f}")
        logging.info(f"Curves PNG: {png_path} | CSV: {csv_path} | TensorBoard: {tb_dir}")

    if is_distributed:
        dist.destroy_process_group()


def main():
    args = parse_args()
    total_train = preflight_check(args)
    if args.check_only:
        return

    out_dir = Path(args.out_dir)
    subset_indices = build_or_load_subset_indices(
        out_dir=out_dir,
        total_train=total_train,
        subset_ratio=args.subset_ratio,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    world_size = len(gpu_list)
    master_port = args.master_port if args.master_port > 0 else find_free_port()

    if world_size == 1:
        worker_main(0, 1, args, subset_indices, master_port)
    else:
        import torch.multiprocessing as mp

        mp.spawn(
            worker_main,
            args=(world_size, args, subset_indices, master_port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
