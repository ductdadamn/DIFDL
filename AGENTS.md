# AGENTS.md — Ngữ cảnh Dự án & Quy tắc Làm việc (DIFDL — Hướng B / Phương án 3A+)

## 1. Tổng quan Dự án & Mốc thời gian
- **Đề tài:** *Document Image Forgery Detection and Localization: Benchmarking and Adapting SOTA Methods* (Đồ án nhóm năm cuối ngành Khoa học Dữ liệu — 7 sinh viên, 8 tuần: `16/11/2026` – `12/01/2027`).
- **Ngôn ngữ giao tiếp:** Trả lời bằng **tiếng Việt** ngắn gọn, rõ ràng, giữ thuật ngữ chuyên ngành tiếng Anh phổ biến.
- **Triết lý:** **100% miền Tài liệu & Hóa đơn (Document/Receipt)**, lý giải xử lý tín hiệu chặt chẽ – thực nghiệm công bằng – dễ bảo vệ trong 15 phút; mọi ý tưởng mở rộng ngoài phạm vi đẩy vào Chương 5 (*Future Work*).
- **Các mốc cứng (Hard Deadlines):**
  - `29/11/2026`: Báo cáo tiến độ 1 (EDA, Mốc 0 Reproduction trên `2×RTX 4000 8GB`, xong dữ liệu đa Generator & bảng Zero-shot).
  - `06/12/2026`: Mốc 1 (Data Adaptation Baseline — Đa Generator 60/40).
  - `13/12/2026`: Mốc 2 (`ADCDNetWithBGSRM` + đánh giá đủ 4 tập Test).
  - `20/12/2026`: **Code & Experiment Freeze** (`v1.0-freeze`).
  - `12/01/2027`: Nộp báo cáo chính thức.

---

## 2. Quyết định Kỹ thuật Đã Khóa (Cập nhật `26/11/2026` — KHÔNG đề xuất lại hướng đã loại)
- **Vị thế SOTA chuẩn xác của `ADCD-Net` (ICCV 2025, Bảng 1 `ADCD-Net/fig/docpro.png`):**
  - Top 1 toàn cục 7 bộ Doc Protocol (`Avg. All = 0.588`), Top 1 liên miền (`Avg. C = 0.450`), và **Top 1 trên hóa đơn `T-SROIE` (`0.623` > `CAT-Net` `0.609` > `FFDN` `0.533` > `DTD` `0.525` > `TruFor` `0.213`)** *(chỉ kém `FFDN` ở In-Domain `DocTamper`: `0.771` vs. `0.813`)*.
  - Nhánh `FindItAgain!` (ICDAR'23) thiên về kiểm tra logic OCR/toán học; chưa có mô hình Visual Forensics pretrained riêng cho hóa đơn mạnh hơn `ADCD-Net`.
- **Cải tiến đề xuất (`BG-SRM` — Background-Guided Truncated SRM):**
  - Phân tách rõ 3 dải tần số: **Tần số thấp (`PPE` sẵn có)** đo màu nền/độ sáng vĩ mô; **Tần số khối `8×8` (`FPH` DCT)** bắt lệch lưới JPEG; **Tần số cao (`BG-SRM`)** bắt *over-smoothing* hoặc *VAE/upsampling artifacts* của hạt nhiễu giấy in nhiệt.
  - **Chứng minh KHÔNG PHẢI Edge Detection bằng 2 cơ chế:** (1) **Truncation/Clamping `[-3, +3]`** cắt bão hòa mép chữ đen–trắng (`0 → 255`), và (2) **Đối chiếu phương sai nhiễu nền giấy thật (`ocr_mask == 0`)** đo độ lệch nhiễu cục bộ so với nền giấy chuẩn của chính tờ hóa đơn đó.
  - Điều biến dư khởi tạo `0` vào `feat` trước `out_head`, bảo toàn `weights/ADCDNet.pth` và **toàn bộ 6 hàm loss gốc** trong `src/main.py`.
- **Các hướng ĐÃ LOẠI BỎ (không đưa lại vào phạm vi chính):**
  - Tự build & train từ đầu (*from scratch*) một model riêng cho hóa đơn trên 483 hóa đơn sạch (đã cân nhắc sau đề xuất của Quag và loại vì chắc chắn overfit, yếu hơn hẳn SOTA pretrained 120k ảnh và gây vỡ tiến độ `20/12`).
  - Copy-Move (`get_cm_map`) trên `findit2` manual CPI (PoC Zero-shot Pixel-F1 = `0.0278` $\rightarrow$ Future Work).
  - `OSTF_test` (3.046 ảnh cảnh tự nhiên/poster làm nhiễu `PPE` $\rightarrow$ Future Work).
  - Web demo Streamlit (thay bằng hình trực quan hóa 300 DPI).
  - Fine-tune `TruFor` và `DTD` (chỉ đánh giá **Zero-shot**).

---

## 3. Dữ liệu Đa Generator & Quy tắc Chống Rò rỉ (Zero Data Leakage)
Toàn bộ dữ liệu hóa đơn chia tách tuyệt đối theo ID gốc của `findit2/{train,val,test}.txt`:
- **Dữ liệu đã xác minh trên đĩa:**
  - `T-SROIE_test/` (`1.579` ảnh `512×512` JPG): khớp 100% ID `findit2` (`998` train / `255` val / `326` test / `0` ảnh ngoài).
  - `findit2/` (`988` PNG, median `932×1697`): `train.txt` (`483` sạch — **bỏ qua dòng lỗi `X51008114260.png`**), `val.txt` (`159` sạch), `test.txt` (`183` sạch).
  - `doctamper_root/DocTamperV1 -> archive`: `TrainingSet` (`120.000`), `TestingSet` (`30.000`), `SCD` (`18.000`), `FCD` (`2.000`) + `exp_data/qt_table.pk` + `exp_data/pks/`.
- **Đa dạng Generator, Hành vi gian lận (Removal / Replacement / Insertion) & Miền huấn luyện gốc:**
  - **Train (~4.300 mẫu/epoch — 60% Hóa đơn AI / 40% `DocTamperV1`):** `998` `T-SROIE` (*SRNet GAN — chữ tổng hợp, Replacement*) + `~1.000` `LaMa` (*FFC CNN — `Places2` ảnh tự nhiên, Removal*) + `~500` `SD-1.5 Inpainting` (*Latent Diffusion — `LAION` ảnh tự nhiên, Removal* nhằm khử biến nhiễu CNN vs. Diffusion; kèm 5–10 mẫu thử viết chữ minh họa hạn chế miền gốc) + `~1.800` `DocTamperV1-TrainingSet` replay buffer. Nén JPEG `min_qf = 65` ngay từ Epoch 1.
  - **Val (~555 mẫu):** `255` `T-SROIE` val + `~200` `LaMa` val + `~100` `SD-1.5` val (từ 159 hóa đơn sạch `findit2/val`).
  - **Test Set 1 (Seen AI — ~726 ảnh):** `326` `T-SROIE` test + `~200` `LaMa` test + `~200` `SD-1.5 Removal` test (từ 183 hóa đơn sạch `findit2/test`).
  - **Test Set 2 (Unseen Open-Source AI — ~400–450 ảnh):** `~250` **`AnyText`** (*Diffusion+ControlNet — `AnyWord-3M`, Replacement & Insertion*) + `~150–200` **`TextDiffuser` / `TextDiffuser-2`** *(hoặc `MOSTEL` dự phòng; `MARIO-10M` OCR, Replacement)* trên mảnh `512×512` của `findit2/test`, **ghép biên mềm Gaussian 3px** ($I_{\text{out}} = M \odot I_{\text{AI}} + (1 - M) \odot I_{\text{orig}}$); **tuyệt đối không dùng lúc Train**.
  - **Test Set 3 (Pristine & Stress):** `183` ảnh sạch `findit2/test` (đo Image FPR $\downarrow$) + bản nén JPEG `QF=70, 50` của Test Set 1 & 2.
  - **Test Set 4 (Commercial AI Wild Case Study — ~25–30 ảnh):** Hóa đơn `findit2/test` sửa bằng `ChatGPT (gpt-image-1)` / `Gemini Image Editing` / `Canva` (gồm ca ghép cục bộ đo Pixel-F1 và ca xuất nguyên tấm đo Image Detection + phân tích *Limitations* khi mất nền giấy gốc).

---

## 4. Ba Mốc Thực nghiệm & Thước đo
1. **Mốc 0 — Sanity-check Reproduction:** Train `src/main.py` từ `weights/docres.pkl` (đủ 6 loss gốc) trên **15–20% `DocTamperV1-TrainingSet`** trong **10–15 epochs** (`2×RTX 4000 8GB`).
2. **Mốc 1 — Data Adaptation Baseline:** Fine-tune `ADCD-Net` nguyên bản từ `weights/ADCDNet.pth` trên tập 60/40 đa Generator (`min_qf=65`).
3. **Mốc 2 — Proposed (`ADCDNetWithBGSRM`):** Fine-tune thêm nhánh `BG-SRM` trên **cùng dữ liệu & siêu tham số** với Mốc 1 (tùy chọn chạy thêm ablation `Vanilla SRM` vs. `BG-SRM` nếu kịp giờ GPU).
- **Baselines đối chiếu:** `TruFor` (Zero-shot), `DTD` (Zero-shot), `ADCD-Net` gốc (Zero-shot).
- **Metrics:** Pixel-F1 & IoU (ảnh giả), Image-level ROC-AUC (toàn tập), Image-level FPR $\downarrow$ (183 ảnh sạch), Stress Test `QF=70, 50`.

---

## 5. Hạ tầng Phần cứng & Môi trường
- **Server Lab được GVHD cấp:** **1 server gồm `2 × Quadro RTX 4000 (8GB)`** *(kiến trúc Turing, Compute Capability `7.5` — hỗ trợ tốt `fp16` AMP / Tensor Cores, **không dùng `bf16`**; truy cập qua SSH public key)*.
  - **Chiến lược train DDP trên `2 × RTX 4000 (8GB)`:** **DDP 2 GPU + `fp16` AMP + `enable_grad_checkpointing()`** trên `TransformerBlock`: `bs = 4/GPU` (đo thực tế `3.49 GB VRAM/GPU`) × `2 GPU` × `accum_step = 4` → `effective batch size = 32`.
- **Sinh `AnyText` & `TextDiffuser` (~10–12 GB VRAM ở `fp16`):** Chạy 1 lần (~15–25 phút/model) miễn phí trên **Kaggle (`T4 16GB`) hoặc Google Colab (`T4 16GB`)**, rồi tải về server.
- **Máy cá nhân của nhóm:** 3 × RTX 4060 8GB (Viết [máy hiện tại], Quag, desktop Zam — chạy mượt `LaMa` ~2GB và `SD-1.5` ~4GB), 1 × RTX 3050 (Zang), laptop văn phòng (Kiều, Mỹ, Thiết).
- **Môi trường Python cục bộ:** `/home/florian_viet/project/group_def/.venv` (Python `3.10.22`, `torch==2.3.0+cu121`, `albumentations==2.0.8`, `jpeg2dct==0.2.4` [cần `libjpeg-dev`], `lmdb`, `opencv-python-headless`, `efficientnet-pytorch`, `einops`).

---

## 6. Phân công 7 Thành viên
- **Squad 1 (Data/QA/Slide):** **Thiết** (split `T-SROIE`, script `LaMa` + `SD-1.5` + `AnyText`/`TextDiffuser`, JPEG stress), **Mỹ** (EDA, tạo ~25–30 mẫu **Commercial AI `GPT/Gemini/Canva`**, Visual QA, hình 300 DPI `PPE` vs `BG-SRM`, Slide Deck).
- **Squad 2 (Model/Infra):** **Viết** *(người dùng — Leader/PM & chủ biên Báo cáo)*, **Quag** (`src/ds_hybrid.py`, nâng cấp **`BG-SRM`** đối chiếu `ocr_mask == 0` + grad checkpointing), **Zang** (server lab `2×RTX 4000 8GB`, chạy `AnyText`/`TextDiffuser` trên Kaggle `T4`, `seg_char.py`, train DDP Mốc 0/1/2).
- **Squad 3 (Eval/Học thuật):** **Zam** (Zero-shot `TruFor`/`DTD`/`ADCD-Net`, `eval_metrics.py` trên đủ 4 tập Test), **Kiều** (Cố vấn học thuật, **tại Pháp CET = UTC+1, chậm VN 6h** $\rightarrow$ gửi review trước `15:00 VN`; họp sync Thứ 4 & CN `20:30 VN = 14:30 CET`; lý thuyết Tần số cao/thấp, Generator Domain & Error Analysis Commercial AI).

---

## 7. Cấu trúc Thư mục, Quy tắc Bất khả xâm phạm & Các Lỗi Kỹ thuật Đã Biết
- **TUYỆT ĐỐI KHÔNG chỉnh sửa các thư mục tham chiếu/dữ liệu gốc:** `ADCD-Net/`, `findit2/`, `doctamper_root/archive/`, `T-SROIE_test/`, `OSTF_test/`. Code phát triển chỉ đặt trong `src/` và `scripts/`.
- **KHÔNG commit/stage:** `docs/GIAI_THICH_CHUYEN_HUONG_B.md` (giữ untracked có chủ đích) và dữ liệu/weights nặng (đã có trong `.gitignore`).
- **Trạng thái code `src/`:** `src/cfg.py` đã trỏ đường dẫn đúng; `src/ds.py` đã vá `BGR -> RGB` (dòng 417) và thêm `load_or_fallback_bboxes()`.
- **4 bẫy kỹ thuật cần nhớ:**
  1. **Cấm `bs = 1` khi train:** `BatchNorm1d` trong `dct_align_head` crash (`ValueError`) $\rightarrow$ luôn dùng `bs >= 2`, `drop_last=True`.
  2. **VRAM 8GB bắt buộc bật `enable_grad_checkpointing()`:** Tắt sẽ OOM ở `bs = 2`; bật giảm xuống `1.97 GB` (`bs=2`) và `3.49 GB` (`bs=4`).
  3. **Tối ưu DataLoader server:** Đổi `dl_workers = 0` sang `dl_workers = 4` + `pin_memory=True, persistent_workers=True` kèm `OMP_NUM_THREADS=1`, `cv2.setNumThreads(0)`.
  4. **I/O `/tmp` trong `multi_jpeg()` (`src/ds.py`):** Chạy server đặt `TMPDIR=/dev/shm` (hoặc tối ưu sang `BytesIO` + NumPy `reshape/transpose`).

---

## 8. File Tham chiếu Chính
- `MASTER_PLAN.md`: Kế hoạch 8 tuần chi tiết, Generator Zoo, lý thuyết tần số & cấu trúc báo cáo 5 chương.
- `TIEN_DO_CONG_VIEC.csv`: Bảng phân công 32 đầu việc của 7 thành viên.
- `scripts/poc_option_b.py`: Code PoC đã kiểm chứng `SRMHighFreqBranch` và `enable_grad_checkpointing()`.
- `src/cfg.py`, `src/ds.py`, `src/main.py`, `src/model/model.py`: Pipeline huấn luyện `ADCD-Net`.
