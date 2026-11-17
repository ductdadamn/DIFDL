# Document Image Forgery Detection and Localization: Benchmarking and Adapting SOTA Methods

Đồ án nhóm 7 thành viên ngành Khoa học Dữ liệu (Data Science) — Khung thời gian: **16/11/2026 – 12/01/2027** (Code Freeze: **20/12/2026**).  
**Định hướng chính thức (Hướng B):** *Benchmarking & Adapting `ADCD-Net` cho bài toán Phát hiện và Khoanh vùng Chỉnh sửa Tài liệu/Văn bản bằng AI (AI-Assisted Manipulation: GAN, Diffusion & LaMa Inpainting) với nhánh lọc nhiễu tần số cao `SRMHighFreqBranch`.*

Tài liệu quản lý dự án:
- [MASTER_PLAN.md](./MASTER_PLAN.md) — Kế hoạch tổng thể 8 tuần (Thiết kế dữ liệu Seen/Unseen AI, kiến trúc `ADCDNetWithSRM`, lộ trình 3 Squads).
- [TIEN_DO_CONG_VIEC.csv](./TIEN_DO_CONG_VIEC.csv) — Bảng theo dõi tiến độ 32 đầu việc của 7 thành viên (nhập trực tiếp vào Google Sheets / Excel).

---

## 1. Cấu trúc Thư mục Dự án

```text
group_def/
├── .gitignore                 # Chặn dữ liệu lớn, weights, .venv/, poc_out/, thư mục tham chiếu gốc
├── README.md
├── MASTER_PLAN.md             # Kế hoạch tổng thể (Hướng B: AI-Assisted Manipulation)
├── TIEN_DO_CONG_VIEC.csv      # Sheet theo dõi tiến độ 7 thành viên
├── docs/                      # Tài liệu đọc hiểu, công thức metrics, biểu đồ EDA & Explainability
│   └── eda_figs/
├── scripts/                   # Scripts PoC, chia split Seen/Unseen AI, sinh LaMa, eval_metrics.py
│   └── poc_option_b.py        # Script PoC đã chạy kiểm chứng trên RTX 4060 8GB
├── src/                       # Mã nguồn làm việc (bản copy sạch từ ADCD-Net để phát triển)
│   ├── cfg.py                 # Đã trỏ sẵn về doctamper_root/ và weights/
│   ├── ds.py
│   ├── main.py
│   ├── seg_char.py
│   ├── model/
│   └── loss/
│
├── weights/                   # [Ignored] ADCDNet.pth và docres.pkl
├── T-SROIE_test/              # [Ignored] 1.579 ảnh hóa đơn SROIE sửa bằng GAN/SRNet
├── OSTF_test/                 # [Ignored] 3.046 ảnh từ 9 bộ sinh AI (AnyText, TextDiffuser, DiffSTE...)
├── findit2/                   # [Ignored] Bộ Find It Again! (483 train / 159 val / 183 test ảnh sạch)
├── doctamper_root/            # [Ignored] Dữ liệu DocTamperV1 & exp_data (qt_table.pk, pks/)
└── ADCD-Net/                  # [Ignored] Repo gốc tham khảo — KHÔNG chỉnh sửa
```

---

## 2. Thiết lập Môi trường & Chạy Thử Nghiệm Nhanh (PoC)

### 2.1. Yêu cầu hệ thống & thư viện
- Python `3.10` + gói hệ thống `libjpeg-dev` (bắt buộc trước khi build `jpeg2dct`: `sudo apt-get install -y libjpeg-dev`)
- PyTorch `2.3.0+cu121`
- Thư viện chính: `albumentations==2.0.8`, `jpeg2dct==0.2.4`, `efficientnet_pytorch==0.7.1`, `einops==0.8.2`, `opencv-python-headless`, `scipy`, `tqdm`, `tensorboard`, `paddleocr`, `simple-lama-inpainting`.

### 2.2. Chạy lại PoC kiểm chứng trên RTX 4060 (8GB VRAM)
```bash
.venv/bin/python scripts/poc_option_b.py
```
*(Lưu ý quan trọng từ kết quả đo thực tế trên 8GB VRAM: khi train trên RTX 4060 bắt buộc bật `enable_grad_checkpointing` và `batch_size >= 2` để giữ VRAM ở mức `~3.49 GB` thay vì bị OOM ở `>7.5 GB`.)*

---

## 3. Phân công 3 Squads & Quy ước Git

| Squad | Thành viên | Nhánh Git phụ trách | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **Squad 1 (Data & AI Synthesis)** | Thiết, Mỹ | `feat/data`, `docs/eda` | Tách Seen vs. Unseen AI (`T-SROIE` theo ID hóa đơn, `OSTF`), chạy `LaMa` trên `findit2` sạch, tạo Stress Test `QF 70/50`, EDA, Visual QA, Explainability Figures & Slide Deck. |
| **Squad 2 (Model & Infra)** | Viết *(PM & Report Lead)*, Quag, Zang | `feat/model`, `feat/infra` | Vá `BGR -> RGB`, viết `ds_hybrid.py` (60/40), tích hợp `SRMHighFreqBranch` + `grad-ckpt`, chạy `seg_char.py`, train Mốc 0/1/2 DDP trên `4×3090`, chủ biên Báo cáo chính. |
| **Squad 3 (Benchmark & Academic)** | Zam, Kiều *(Pháp – Async)* | `feat/eval`, `docs/academic` | Tóm tắt SOTA & AI artifacts, chốt Metrics Protocol, code `eval_metrics.py`, chạy Zero-shot **TruFor / DTD / ADCD-Net**, chấm điểm Mốc 1/2/Ablation/Stress Test & Error Analysis. |

### Quy ước Làm việc trên Git
1. **Không commit trực tiếp lên `main`:** Tạo nhánh mới `feat/*` hoặc `docs/*` từ `main`, sau đó tạo Pull Request để **Viết (PM)** review và merge.
2. **Không chỉnh sửa `ADCD-Net/` gốc:** Mọi chỉnh sửa code thực hiện trong `src/` và `scripts/`.
3. **Không commit dữ liệu/weights:** Tuyệt đối không dùng `git add -f` cho ảnh dataset, `.pth`, `.pkl`, `.npy`.
4. **Lịch đồng bộ VN–Pháp:** Gửi file cần **Kiều** review trước **15:00 VN (9:00 CET)**; họp Sync nhóm cố định vào **20:30 VN (14:30 CET) Thứ 4 & Chủ nhật**.
