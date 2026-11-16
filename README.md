# Document Image Forgery Detection and Localization: Benchmarking and Adapting SOTA Methods

Đồ án nhóm 7 thành viên ngành Khoa học Dữ liệu (Data Science) — Khung thời gian: **16/11/2026 – 12/01/2027** (Code Freeze: **20/12/2026**).

Tài liệu chi tiết kế hoạch và phân công theo dõi tại:
- [MASTER_PLAN.md](./MASTER_PLAN.md) — Kế hoạch tổng thể 8 tuần (Mạch 3 bước: Bắt bệnh SOTA $\rightarrow$ Thích nghi Dữ liệu & Độ phân giải $\rightarrow$ Nhánh nhẹ `get_cm_map`).
- [TIEN_DO_CONG_VIEC.csv](./TIEN_DO_CONG_VIEC.csv) — Bảng theo dõi tiến độ 32 đầu việc (nhập trực tiếp vào Google Sheets / Excel).

---

## 1. Cấu trúc Thư mục Dự án

```text
group_def/
├── .gitignore                 # Chặn dữ liệu nặng, weights, logs và thư mục tham chiếu gốc
├── README.md
├── MASTER_PLAN.md             # Tài liệu thiết kế kỹ thuật & lộ trình 8 tuần
├── TIEN_DO_CONG_VIEC.csv      # Sheet theo dõi tiến độ 7 thành viên
├── docs/                      # Tài liệu đọc hiểu, công thức metrics, biểu đồ EDA & Explainability
│   └── eda_figs/
├── scripts/                   # Scripts EDA, parse_findit2.py, synth_copymove.py, eval_metrics.py
├── src/                       # Mã nguồn làm việc (bản copy sạch từ ADCD-Net để phát triển)
│   ├── cfg.py
│   ├── ds.py
│   ├── main.py
│   ├── seg_char.py
│   ├── model/
│   └── loss/
│
├── ADCD-Net/                  # [Ignored] Repo gốc tham khảo — KHÔNG chỉnh sửa
├── findit2/                   # [Ignored] Bộ Find It Again! (577 train / 193 val / 218 test)
└── doctamper_root/            # [Ignored] Dữ liệu DocTamperV1 & exp_data (qt_table.pk, pks/)
```

---

## 2. Thiết lập Môi trường & Dữ liệu (Dành cho các máy Local / Server 4×3090)

### 2.1. Yêu cầu môi trường
- Python `3.10` (`3.10.13`)
- PyTorch `2.3.0+cu121`
- Thư viện chính: `albumentations==2.0.8`, `jpeg2dct`, `efficientnet_pytorch`, `einops`, `opencv-python`, `scikit-learn`, `tqdm`, `tensorboard`, `paddleocr` (dùng khi chạy `seg_char.py`).

### 2.2. Chuẩn bị dữ liệu & trọng số (Không đẩy lên GitHub)
Sau khi `git clone` repository về máy/server, đặt các thư mục sau đúng tại gốc repo (đều nằm trong `.gitignore`):
1. `findit2/` — Chứa `train/`, `val/`, `test/` và `train.txt`, `val.txt`, `test.txt`.
2. `doctamper_root/` — Chứa `DocTamperV1` và `exp_data/` (`qt_table.pk`, `pks/`).
3. `weights/` — Tải `docres.pkl` và `ADCDNet.pth` từ link chính thức trong [ADCD-Net](./ADCD-Net/README.md) (Google Drive của tác giả).

---

## 3. Phân công 3 Squads & Quy ước Git

| Squad | Thành viên | Nhánh Git phụ trách | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **Squad 1 (Data & EDA)** | Thiết, Mỹ | `feat/data`, `docs/eda` | Parse `findit2` (chỉ tô `Original area == 'no'`, loại `X51008114260.png`), EDA, sinh ~2.500 ảnh Synthetic Copy-Move (`< 200 px`), Visual QA, Explainability Figures & Slide Deck. |
| **Squad 2 (Model & Infra)** | Viết *(PM & Report Lead)*, Quag, Zang | `feat/model`, `feat/infra` | Vá `BGR -> RGB` trong `GeneralValDs`, viết `ReceiptHybridTrainDs` (60/40), thiết kế `get_cm_map` (`1/4`), chạy `seg_char.py`, train Mốc 0/1/2 DDP trên `4×3090`, tổng hợp Báo cáo chính. |
| **Squad 3 (Benchmark & Academic)** | Zam, Kiều *(Pháp – Async)* | `feat/eval`, `docs/academic` | Tóm tắt SOTA, chốt Metrics Protocol, code `eval_metrics.py`, chạy Zero-shot **TruFor / DTD / ADCD-Net**, khảo sát `val_max_size`, Stress Test JPEG & Error Analysis. |

### Quy ước Làm việc trên Git
1. **Không commit trực tiếp lên `main`:** Tạo nhánh mới từ `main` cho từng đầu việc (ví dụ: `feat/data-parser`, `feat/eval-metrics`, `feat/cm-branch`), sau đó tạo Pull Request để **Viết (Leader)** review và merge.
2. **Không chỉnh sửa `ADCD-Net/` gốc:** Mọi chỉnh sửa code mô hình/dataloader thực hiện trong `src/`.
3. **Không commit file nặng:** Tuyệt đối không dùng `git add -f` cho ảnh dataset, `.pth`, `.pkl`, `.npy`.
4. **Lịch đồng bộ VN–Pháp:** Gửi file cần **Kiều** review trước **15:00 VN (9:00 CET)**; họp Sync nhóm cố định vào **20:30 VN (14:30 CET) Thứ 4 & Chủ nhật**.
