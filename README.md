# Document Image Forgery Detection and Localization: Benchmarking and Adapting SOTA Methods

Đồ án nhóm 7 thành viên ngành Khoa học Dữ liệu (Data Science) — Khung thời gian: **16/11/2026 – 12/01/2027** (Code Freeze: **20/12/2026**).  
**Định hướng chính thức (Hướng B — Phương án 3A+):** *Benchmarking & Adapting SOTA `ADCD-Net` (ICCV 2025) trên **100% miền Tài liệu & Hóa đơn** với tập dữ liệu **Đa Generator** (Seen: `T-SROIE` GAN, `LaMa` CNN Removal, `SD-1.5 Inpainting` Diffusion Removal; Unseen Test: `AnyText`, `TextDiffuser` và Case Study Commercial AI `ChatGPT/Gemini`) cùng nhánh tần số cao đối chiếu nền giấy **`BG-SRM`** (tách biệt rõ ràng với Tần số thấp `PPE` và không phải Edge Detection).*

Tài liệu quản lý dự án:
- [MASTER_PLAN.md](./MASTER_PLAN.md) — Kế hoạch tổng thể 8 tuần (Generator Zoo, lý thuyết Tần số Cao/Thấp `BG-SRM`, thiết kế 3A+, lộ trình 3 Squads).
- [TIEN_DO_CONG_VIEC.csv](./TIEN_DO_CONG_VIEC.csv) — Bảng theo dõi tiến độ 32 đầu việc của 7 thành viên (nhập trực tiếp vào Google Sheets / Excel).
- [AGENTS.md](./AGENTS.md) — Tóm tắt ngữ cảnh & quy tắc kỹ thuật tự động nạp cho các phiên làm việc AI.
- [docs/DATA_SYNTH.md](./docs/DATA_SYNTH.md) — Quy trình sinh dữ liệu `LaMa` / `AnyText` trên `findit2` (split theo ID, manifest, Kaggle, QA).

---

## 1. Cấu trúc Thư mục Dự án

```text
group_def/
├── .gitignore                 # Chặn dữ liệu lớn, weights, .venv/, poc_out/, thư mục tham chiếu gốc
├── AGENTS.md                  # Ngữ cảnh & quy tắc làm việc tự động nạp cho session mới
├── README.md
├── MASTER_PLAN.md             # Kế hoạch tổng thể (Hướng B - Phương án 3A+)
├── TIEN_DO_CONG_VIEC.csv      # Sheet theo dõi tiến độ 7 thành viên
├── docs/                      # Tài liệu đọc hiểu, công thức metrics, biểu đồ EDA & Explainability
│   └── eda_figs/
├── scripts/                   # Scripts PoC, split ID findit2, sinh LaMa/SD-1.5/AnyText/TextDiffuser, eval_metrics.py
│   └── poc_option_b.py        # Script PoC đã chạy kiểm chứng trên RTX 4060 8GB
├── src/                       # Mã nguồn làm việc (bản copy sạch từ ADCD-Net để phát triển)
│   ├── cfg.py                 # Đã trỏ sẵn về doctamper_root/ và weights/
│   ├── ds.py                  # Đã vá BGR->RGB và fallback ocr_mask khi thiếu DocTamperOCR/
│   ├── main.py
│   ├── seg_char.py
│   ├── model/
│   └── loss/
│
├── weights/                   # [Ignored] ADCDNet.pth và docres.pkl
├── T-SROIE_test/              # [Ignored] 1.579 ảnh hóa đơn GAN (998 train / 255 val / 326 test theo ID findit2)
├── findit2/                   # [Ignored] Bộ Find It Again! (483 train / 159 val / 183 test ảnh sạch)
├── doctamper_root/            # [Ignored] DocTamperV1 -> archive (120k train / 30k test / 2k FCD / 18k SCD)
└── ADCD-Net/                  # [Ignored] Repo gốc tham khảo — KHÔNG chỉnh sửa
```

---

## 2. Thiết lập Môi trường, Hạ tầng & Chạy Thử Nghiệm Nhanh (PoC)

### 2.1. Phân bổ Hạ tầng Thực tế
- **Train chính thức (Mốc 0, Mốc 1, Mốc 2):** Chạy DDP trên **Server Lab `2 × Quadro RTX 4000 (8GB)`** *(Turing CC `7.5`, chỉ dùng `fp16` AMP + `enable_grad_checkpointing()`, **không dùng `bf16`**; `bs = 4/GPU × 2 GPU × accum_step = 4` $\Rightarrow$ `effective batch size = 32`, `dl_workers = 4`, `TMPDIR=/dev/shm`)*.
- **Sinh dữ liệu Removal nhẹ (`LaMa`, `SD-1.5 Inpainting`):** Chạy ngay trên **RTX 4060 8GB** của nhóm (`~2–4 GB VRAM` ở `FP16`).
- **Sinh dữ liệu Diffusion sửa chữ (`AnyText`, `TextDiffuser`):** Chạy 1 lần duy nhất trên **Kaggle (`T4 16GB`) hoặc Google Colab (`T4 16GB`)** miễn phí, rồi tải về lưu cố định trên server.

### 2.2. Yêu cầu hệ thống & thư viện
- Python `3.10` + gói hệ thống `libjpeg-dev` (bắt buộc trước khi build `jpeg2dct`: `sudo apt-get install -y libjpeg-dev`)
- PyTorch `2.3.0+cu121`
- Thư viện chính: `albumentations==2.0.8`, `jpeg2dct==0.2.4`, `efficientnet_pytorch==0.7.1`, `einops==0.8.2`, `opencv-python-headless`, `scipy`, `tqdm`, `tensorboard`, `paddleocr`, `simple-lama-inpainting`, `diffusers` / `transformers` / `accelerate` (cho `SD-1.5 Inpainting`), và môi trường riêng trên Kaggle/Colab cho `AnyText` & `TextDiffuser`.

### 2.3. Chạy lại PoC kiểm chứng trên RTX 4060 (8GB VRAM)
```bash
.venv/bin/python scripts/poc_option_b.py
```
*(Lưu ý: khi train trên GPU `8GB` bắt buộc bật `enable_grad_checkpointing` và `batch_size >= 2` để giữ VRAM ở mức `~3.49 GB` thay vì bị OOM ở `>7.5 GB`.)*

---

## 3. Phân công 3 Squads & Quy ước Git

| Squad | Thành viên | Nhánh Git phụ trách | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **Squad 1 (Data & AI Synthesis)** | Thiết, Mỹ | `feat/data`, `docs/eda` | Khớp split `T-SROIE` theo ID `findit2` (`998/255/326`), chạy `LaMa` + `SD-1.5 Inpainting` & chuẩn bị mảnh `512×512` cho `AnyText`/`TextDiffuser`, Stress Test `QF 70/50`, tạo ~25–30 mẫu **Commercial AI (`GPT/Gemini/Canva`)**, EDA, Visual QA, hình 300 DPI (`PPE` vs `BG-SRM`) & Slide Deck. |
| **Squad 2 (Model & Infra)** | Viết *(PM & Report Lead)*, Quag, Zang | `feat/model`, `feat/infra` | Viết `ds_hybrid.py` (60% `T-SROIE + LaMa + SD-1.5` / 40% `DocTamperV1`, `min_qf=65`), nâng cấp **`BG-SRM`** đối chiếu nền giấy `ocr_mask == 0` + `grad-ckpt`, chạy `AnyText`/`TextDiffuser` trên Kaggle `T4`, chạy `seg_char.py`, train DDP Mốc 0/1/2 trên **Server `2×RTX 4000 (8GB)`**, chủ biên Báo cáo chính. |
| **Squad 3 (Benchmark & Academic)** | Zam, Kiều *(Pháp – Async)* | `feat/eval`, `docs/academic` | Lý thuyết Tần số cao/thấp & Generator Training Domain, chốt Metrics Protocol, viết `eval_metrics.py`, chạy Zero-shot **TruFor / DTD / ADCD-Net**, chấm điểm đủ **4 tập Test** cho Mốc 1/2/Ablation/Stress & Error Analysis. |

### Quy ước Làm việc trên Git
1. **Không commit trực tiếp lên `main`:** Tạo nhánh mới `feat/*` hoặc `docs/*` từ `main`, sau đó tạo Pull Request để **Viết (PM)** review và merge.
2. **Không chỉnh sửa `ADCD-Net/` gốc:** Mọi chỉnh sửa code thực hiện trong `src/` và `scripts/`.
3. **Không commit dữ liệu/weights:** Tuyệt đối không dùng `git add -f` cho ảnh dataset, `.pth`, `.pkl`, `.npy`.
4. **Lịch đồng bộ VN–Pháp:** Gửi file cần **Kiều** review trước **15:00 VN (9:00 CET)**; họp Sync nhóm cố định vào **20:30 VN (14:30 CET) Thứ 4 & Chủ nhật**.
