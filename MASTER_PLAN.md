# KẾ HOẠCH TỔNG THỂ — ĐỒ ÁN NHÓM DATA SCIENCE (HƯỚNG B: AI-ASSISTED DOCUMENT FORENSICS)

**Đề tài:** *Document Image Forgery Detection and Localization: Benchmarking and Adapting SOTA Methods*  
**Đối tượng thực hiện:** Nhóm 7 sinh viên năm cuối ngành Khoa học Dữ liệu (Data Science)  
**Thời gian thực hiện:** 8 tuần (`16/11/2026` – `12/01/2027`)  
**Trạng thái hiện tại (`18/11/2026`):** Đã kiểm chứng toàn bộ phần cứng & dữ liệu gốc (`DocTamperV1` 120k LMDB, `T-SROIE_test` 1.579 ảnh khớp ID `findit2`, `ADCDNet.pth`, `docres.pkl`, `jpeg2dct`) và sửa xong lỗi nền tảng trong [src/ds.py](./src/ds.py). Chính thức khóa thiết kế **Hướng B (100% Miền Tài liệu & Hóa đơn — Phương án 3A)**.  
**Triết lý thực hiện:** **Đúng miền Tài liệu (100% Document/Receipt) – Dễ hiểu cho cả 7 thành viên – Chống rò rỉ tuyệt đối.** Bỏ `OSTF_test` (vốn là ảnh cảnh tự nhiên/poster phim) khỏi Train & Test chính để tránh làm hỏng nhánh nền giấy `PPE`; tập trung 100% vào hóa đơn & tài liệu với thiết kế **Seen AI (`T-SROIE` GAN + `LaMa` Removal) vs. Unseen AI (`AnyText` Diffusion trên chính hóa đơn `findit2/test`)**.

---

## I. Câu chuyện Bảo vệ 15 Phút (3 Đóng góp Cốt lõi Dễ Trình Bày)

Toàn bộ đồ án xoay quanh một mạch kể chuyện nhất quán, trực quan trên **đúng 1 miền Tài liệu & Hóa đơn**:

1. **Bước 1 — Bắt bệnh SOTA trên Hóa đơn chỉnh sửa bằng AI (Empirical Diagnosis):**
   - Benchmark Zero-shot 3 mô hình (**TruFor**, **DTD**, **ADCD-Net**) trên hóa đơn bị AI sửa (`T-SROIE` GAN, `LaMa-FindIt2` xóa chữ CNN, và `AnyText-FindIt2` viết đè số tiền bằng Diffusion).
   - **Điểm yếu thực tế:** Các mô hình cắt-dán truyền thống (`TruFor`, `DTD`) sụp đổ; còn `ADCD-Net` gốc dù có nền tảng tốt trên tài liệu nhưng vẫn sụt giảm trước hóa đơn in nhiệt bị AI xóa/sửa bằng công cụ mới (**Unseen Diffusion `AnyText`**) và khi ảnh qua nén mạng xã hội (`QF 70, 50`).
2. **Bước 2 — Thích nghi hóa Dữ liệu & Sửa Lỗi Pipeline (Mốc An Toàn 1 — Data Adaptation):**
   - **Zero Data Leakage tuyệt đối:** Khớp `T-SROIE` (1.579 ảnh) và `LaMa-FindIt2` theo **cùng bộ ID hóa đơn chuẩn của `findit2/{train,val,test}.txt`**, đảm bảo 1 tờ hóa đơn không bao giờ xuất hiện ở cả Train và Test.
   - **Phương án 3A (Seen vs. Unseen AI ngay trên Hóa đơn):**
     - *Seen AI lúc Train (60%):* `T-SROIE` (998 ảnh GAN) + `LaMa-FindIt2` (~1.500 ảnh xóa chữ trên 483 hóa đơn sạch `findit2/train`) + 40% `DocTamperV1-TrainingSet` replay buffer.
     - *Unseen AI lúc Test:* **`AnyText-FindIt2` (~300 mảnh `512×512` hóa đơn `findit2/test` được AI Diffusion `AnyText` viết đè số tiền mới trên Server `4×RTX 3090`)** — công cụ mà mô hình **chưa từng thấy khi train**!
   - **Vá 3 lỗi ngầm của repo gốc:** (1) `BGR → RGB` trong `GeneralValDs`, (2) fallback trích mask chữ khi thiếu file `.pkl` của `DocTamperOCR` (đã sửa trong `src/ds.py`), (3) bỏ `init_S = 0` để nén JPEG dải rộng `QF ∈ [65, 100]` ngay từ Epoch 1 khi Fine-tune, và bật `enable_grad_checkpointing` cho RTX 4060 8GB.
3. **Bước 3 — Thích nghi hóa Kiến trúc Nhẹ `SRMHighFreqBranch` (Mốc Nâng Cao 2 — Điểm nhấn):**
   - Tích hợp 3 bộ lọc pháp y **SRM 5×5 cố định** (*Spatial Rich Models*) + khối `res_net` nhẹ (**`+960` tham số = `0.004%`**) điều biến dư lên `feat` trước `out_head`, giữ nguyên 100% đầu vào `Restormer`, giữ nguyên 6 hàm Loss gốc của `ADCD-Net/main.py` và tương thích trọn vẹn với `weights/ADCDNet.pth`.
   - **Ablation bắt buộc:** Đối chứng công bằng giữa **Mốc 1 (Fine-tune không SRM)** và **Mốc 2 (Fine-tune + SRM)** trên cùng dữ liệu Seen AI, Unseen AI (`AnyText`), 183 ảnh sạch (`FPR`) và Stress Test `QF 70/50`.

### Các mốc thời gian cứng (Hard Deadlines)
| Mốc thời gian | Sự kiện | Tiêu chí hoàn thành |
| :--- | :--- | :--- |
| **18/11/2026** | **Khóa thiết kế Hướng B (3A)** | Đủ dữ liệu (`DocTamperV1`, `T-SROIE`, `findit2`), vá xong `src/ds.py`, chốt phân công 7 người. |
| **29/11/2026** | **Báo cáo tiến độ 1 với GVHD** | Xong EDA, Sanity-check Reproduction (`Mốc 0`), xong dữ liệu `LaMa` + `AnyText` (`4×3090`) & Bảng Zero-shot. |
| **06/12/2026** | **Mốc An Toàn 1** | Xong Fine-tune `ADCD-Net` gốc trên Tập Train 60/40 (`T-SROIE + LaMa + DocTamperV1`) trên Server `4×3090`. |
| **13/12/2026** | **Mốc Nâng Cao 2** | Xong huấn luyện `ADCDNetWithSRM` + đánh giá đầy đủ Seen AI, Unseen AI (`AnyText`), Pristine & Stress Test. |
| **20/12/2026** | **Code & Experiment Freeze** | Xong Ablation Study (Mốc 1 vs. Mốc 2), khóa toàn bộ code, checkpoint, bảng số liệu và hình Heatmap. |
| **21/12/2026 – 11/01/2027** | **Báo cáo, Slide & Mock Defense** | Hoàn tất Báo cáo 5 chương, duyệt GVHD, hoàn tất Slide Deck và 2 buổi bảo vệ thử cả 7 người. |
| **12/01/2027** | **Nộp báo cáo chính thức** | Nộp bản PDF/in hoàn chỉnh và repository GitHub tag `v1.0-freeze`. |

---

## II. Phân bổ Nguồn lực (7 Thành viên – 3 Squads) & Hạ tầng

Hạ tầng hiện có: **3 máy RTX 4060 8GB** (Viết, Quag, desktop Zam), **1 máy RTX 3050** (Zang), **laptop văn phòng** (Kiều, Mỹ, Thiết, laptop Zam), và **Server Lab (4 × RTX 3090 24GB)** để train chính thức.  
*(Theo dõi tiến độ chi tiết từng đầu việc tại [TIEN_DO_CONG_VIEC.csv](./TIEN_DO_CONG_VIEC.csv)).*

| Tiểu nhóm | Thành viên & Vai trò | Hạ tầng sử dụng | Trách nhiệm chính |
| :--- | :--- | :--- | :--- |
| **Squad 1: Data & AI Synthesis** | - **Thiết** (Data Eng)<br>- **Mỹ** (EDA, Visual QA & Slide Lead) | Laptop VP (+ RTX 4060 chạy `LaMa`; phối hợp Zang chạy `AnyText` trên `4×3090`) | - **Thiết:** Split `T-SROIE_test` khớp tuyệt đối 100% theo ID của `findit2/{train,val,test}.txt` (`998/255/326`); viết script `gen_ai_findit2.py` cắt mảnh `512×512` chạy **`LaMa`** (Train/Val/Test) & chuẩn bị mảnh cho **`AnyText`** (Unseen Test với ghép biên mềm 3px); tạo Stress Test JPEG (`QF 70, 50`).<br>- **Mỹ:** EDA trực quan hóa `T-SROIE` & `findit2`, Visual QA 50 mẫu `LaMa` & `AnyText`, xuất hình Heatmap 300 DPI & chủ trì **Slide Deck** bảo vệ. |
| **Squad 2: Model & Infra** | - **Viết** (Leader / PM & **Report Lead**)<br>- **Quag** (Architecture & DataLoader Lead)<br>- **Zang** (Server & Training Lead) | 2 × RTX 4060 (Viết, Quag) + RTX 3050 (Zang) + **Server 4×RTX 3090** | - **Viết:** Điều phối chung (PM), review/merge PR, giữ mốc Code Freeze, **chủ biên & viết Báo cáo chính** (Ch.1, Ch.4, Ch.5 & biên tập toàn cuốn), tổ chức Mock Defense.<br>- **Quag:** Kiểm thử `src/ds.py` đã vá, viết `src/ds_hybrid.py` (nạp 60% `T-SROIE + LaMa` & 40% `DocTamperV1` với `min_qf=65` từ Epoch 1), tích hợp `ADCDNetWithSRM` + `enable_grad_checkpointing` vào `src/model/model.py`, soạn nháp kỹ thuật Ch.3.<br>- **Zang:** Dựng env đồng bộ (`jpeg2dct`, `libjpeg-dev`, `PaddleOCR`, `AnyText` trên `4×3090`), chạy `AnyText` (~5 phút cho 300 ảnh Unseen Test) + `seg_char.py` (`ocr_mask`), chạy **Mốc 0** (Reproduction), **Mốc 1** (Fine-tune) & **Mốc 2** (`+ SRM`) DDP trên `4×3090` + backup checkpoint. |
| **Squad 3: Benchmark, Eval & Academic** | - **Zam** (Benchmark & Eval Lead)<br>- **Kiều** (Academic Advisor — *đang trao đổi tại Pháp, chậm VN 6h*) | Desktop RTX 4060 + Laptop VP (Zam) + Laptop VP (Kiều) | - **Zam:** Cài & chạy Zero-shot **TruFor, DTD, ADCD-Net gốc** trên RTX 4060 (không cần fine-tune TruFor/DTD), code `eval_metrics.py`, chấm điểm Mốc 1, Mốc 2 (Seen AI, Unseen `AnyText`, Pristine FPR), Stress Test JPEG & bàn giao số liệu Ch.4 cho Viết.<br>- **Kiều (Async VN–Pháp):** Soạn note tóm tắt SOTA (`SRM`, `VAE/Diffusion artifacts`), chốt Metrics Protocol, làm **Error Analysis**, soạn nháp Ch.2 & review học thuật chéo cho báo cáo của Viết. |

> **Quy ước phối hợp lệch múi giờ với Kiều (Pháp CET = UTC+1, chậm hơn VN 6 tiếng):** Gửi tài liệu cần Kiều review trước **15:00 VN (9:00 sáng Pháp)**; họp Sync ngắn Thứ 4 & Chủ nhật cố định vào **20:30 VN (14:30 chiều Pháp)**.

---

## III. Thiết kế Dữ liệu Chuẩn (100% Tài liệu & Hóa đơn — Zero Data Leakage)

### 1. Nguồn dữ liệu đã xác minh trong Workspace
- **`T-SROIE_test/` (1.579 ảnh `512×512`):** Toàn bộ `1.579` ảnh đều khớp 100% với ID hóa đơn của `findit2`: **`998` ảnh thuộc `findit2/train.txt`**, **`255` ảnh thuộc `findit2/val.txt`**, và **`326` ảnh thuộc `findit2/test.txt`** (`0` ảnh ngoài).
- **`findit2/` (988 ảnh hóa đơn thật):** `train.txt` (483 ảnh sạch), `val.txt` (159 ảnh sạch), `test.txt` (183 ảnh sạch).
- **`doctamper_root/` (`DocTamperV1`):** Đã nối symlink chuẩn `DocTamperV1 -> archive`: `TrainingSet` (`120.000` mẫu), `TestingSet` (`30.000`), `FCD` (`2.000`), `SCD` (`18.000`) + `qt_table.pk` + `pks/`.
- *(Lưu ý: `OSTF_test/` gồm ảnh cảnh tự nhiên/poster phim được loại khỏi thí nghiệm chính để giữ độ tinh khiết 100% miền Tài liệu/Hóa đơn và không làm lệch nhánh nền giấy `PPE`.)*

### 2. Phân chia Train / Val / Test (Phương án 3A — Seen vs. Unseen AI trên Hóa đơn)
1. **Tập Train (~4.300 mẫu — Tỷ lệ 60% Hóa đơn AI / 40% DocTamperV1):**
   - **60% Receipt AI (~2.500 mẫu — Seen AI):**
     - **`998` ảnh `T-SROIE` (Train IDs):** Tất cả ảnh có ID nằm trong `findit2/train.txt` (AI GAN/SRNet).
     - **`~1.500` mảnh `512×512` `LaMa-FindIt2` (Train IDs):** Chạy `simple-lama-inpainting` (~2GB VRAM) xóa các cụm số/chữ trên 483 hóa đơn sạch `findit2/train`, áp dụng Resize ngẫu nhiên + nén **JPEG `QF ∈ [65, 100]` ngay từ Epoch 1** (bỏ `init_S = 0` của code gốc).
   - **40% `DocTamperV1-TrainingSet` (~1.800 mẫu/epoch):** Replay buffer giữ vững năng lực bắt chỉnh sửa tài liệu tổng quát và lưới DCT.
2. **Tập Validation (~505 mẫu — Theo đúng ID `findit2/val.txt`):**
   - **`255` ảnh `T-SROIE` (Val IDs)** + **`~250` mảnh `LaMa-FindIt2` (Val IDs)** sinh từ 159 hóa đơn sạch `findit2/val`. Dùng để chọn checkpoint tốt nhất.
3. **Bộ 3 Tập Test Chuẩn (100% từ ID của `findit2/test.txt` — Tuyệt đối tách biệt khỏi Train/Val):**
   - **Test Set 1 — Seen AI trên Hóa đơn (`~626` ảnh):**
     - **`326` ảnh `T-SROIE` (Test IDs)** (GAN/SRNet).
     - **`~300` mảnh `512×512` `LaMa-FindIt2-Test`** sinh bằng `LaMa` trên 183 hóa đơn sạch `findit2/test`.
   - **Test Set 2 — Unseen AI trên Hóa đơn (`~300` ảnh `AnyText-FindIt2` — Điểm nhấn Phương án 3A):**
     - Chạy **Latent Diffusion `AnyText`** trên Server `4×RTX 3090` (~5 phút cho 300 ảnh) để viết đè số tiền/ngày tháng mới trên mảnh `512×512` của 183 hóa đơn sạch `findit2/test`.
     - **Bắt buộc ghép theo mặt nạ làm mượt biên 3px:** $I_{\text{out}} = M \odot I_{\text{AnyText}} + (1 - M) \odot I_{\text{orig\_512}}$ để các dòng chữ không sửa xung quanh giữ nguyên độ nét gốc.
     - Toàn bộ `AnyText-FindIt2` **hoàn toàn không xuất hiện lúc Train**, giúp kiểm chứng khả năng tổng quát hóa từ *GAN + CNN Inpainting* sang *Diffusion Text Editing* trên chính hóa đơn!
   - **Test Set 3 — Pristine Control & JPEG Stress Test:**
     - **183 ảnh sạch nguyên bản (`findit2/test`)** để đo **Image-level FPR ($\downarrow$)**.
     - Bản sao nén **JPEG QF = 70 và QF = 50** của Test Set 1 & 2 để đo độ bền vững.

### 3. Những điểm kỹ thuật đã vá sẵn trong `src/`
- **Đã vá trong [src/ds.py](./src/ds.py):** (1) Chuyển `BGR → RGB` ở dòng 417 (`GeneralValDs`), (2) Thêm `load_or_fallback_bboxes` để tự đọc `.pkl`, `.png`, hoặc trích box hình thái học nếu thiếu `DocTamperOCR/` (tránh lỗi `ocr_mask` toàn số `0` khi train).
- **Cần làm trong `src/ds_hybrid.py`:** Với dữ liệu hóa đơn AI, đọc trực tiếp `ocr_mask` dạng `.png` do `src/seg_char.py` (`PP-OCRv5_server_det`) tạo ra qua `additional_targets={'ocr_mask': 'mask'}`, và đặt `min_qf = 65` ngay từ Epoch 1.

---

## IV. Thiết kế Mô hình & Thực nghiệm (Giữ Nguyên Loss Gốc)

1. **Mốc 0 — Sanity-check Reproduction (Yêu cầu của GVHD):** Chạy `src/main.py` (khởi tạo `weights/docres.pkl`) trên **15–20% `DocTamperV1-TrainingSet`** trong **10–15 epochs** trên Server `4×RTX 3090`, xuất biểu đồ TensorBoard báo cáo GVHD `29/11`.
2. **Mốc 1 (An toàn) — Data Adaptation (`ADCD-Net` Fine-tune 60/40):** Nạp `weights/ADCDNet.pth`, **giữ nguyên kiến trúc và 100% 6 hàm Loss của `src/main.py`**, Fine-tune trên Tập Train `T-SROIE + LaMa + DocTamperV1` (`bs=32`, `lr=1e-5` – `5e-5`, `min_qf=65`).
3. **Mốc 2 (Nâng cao) — `ADCDNetWithSRM` (`+960` tham số):**
   ```mermaid
   flowchart LR
       IMG["Ảnh (RGB + DCT + QT)"] --> ENC["ADCD-Net (DocRes + FPH + HCD)"]
       OCR["ocr_mask"] --> PPE["PPE (get_pp_map → pp_maps [B,4,H,W])"]
       ENC --> PPE
       ENC -->|"feat [B,96,H,W]"| MOD["PP Scale & Bias"]
       PPE --> MOD
       IMG -->|"RGB"| SRM["3 Bộ lọc SRM 5×5 cố định → res_net → srm_map [B,1,H,W]"]
       MOD --> FUSE["feat * (1 + srm_scale) + srm_bias (Zero-init)"]
       SRM --> FUSE
       FUSE --> OUT["out_head → Mask Dự Đoán"]
   ```
   - 3 bộ lọc **SRM 5×5 cố định** trích xuất phần dư tần số cao, điều biến dư khởi tạo `0` vào `feat` trước `out_head` để giữ nguyên `ADCDNet.pth`.
   - **Ablation Study:** So sánh trực tiếp **Mốc 1 (không SRM)** vs. **Mốc 2 (+ SRM)** trên Seen AI, Unseen `AnyText`, `183` ảnh sạch và `QF=70/50`.

---

## V. Thước đo, Bảng Kết quả & Trực quan hóa

### 1. Thước đo Đánh giá
1. **Localization (trên ảnh giả `forged == 1`):** **Pixel-level F1** và **IoU**.
2. **Detection (trên toàn tập gồm cả ảnh giả + 183 ảnh sạch):** **Image-level ROC-AUC** (Max hoặc Top-$k\%$ pixel xác suất).
3. **Độ sạch & Độ bền (Test Set 3):** **Image-level FPR ($\downarrow$)** trên 183 ảnh sạch `findit2/test` và **Pixel-F1** ở **JPEG QF = 70, 50**.

### 2. Bảng Kết quả Trung tâm cho Báo cáo & Slide
| Mô hình / Phiên bản | Seen AI: `T-SROIE` (326 ảnh: F1 / IoU) | Seen AI: `LaMa-FindIt2` (300 ảnh: F1 / IoU) | **Unseen AI**: `AnyText-FindIt2` (300 ảnh: F1 / IoU) | Toàn tập (Image AUC) | 183 Ảnh sạch (Image FPR $\downarrow$) | Stress QF=50 (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. TruFor** (Zero-shot, ảnh tự nhiên) | ... | ... | ... | ... | ... | ... |
| **2. DTD** (Zero-shot, tài liệu) | ... | ... | ... | ... | ... | ... |
| **3. ADCD-Net** (Zero-shot gốc `ADCDNet.pth`) | ... | ... | ... | ... | ... | ... |
| **4. ADCD-Net + Data Adapt 60/40** *(Mốc 1)* | ... | ... | ... | ... | ... | ... |
| **5. ADCD-Net + Data + `SRM`** *(Mốc 2 — Ours)* | **...** | **...** | **...** | **...** | **...** | **...** |

### 3. Trực quan hóa (Thay cho Web Demo)
Xuất 5–7 bộ hình (300 DPI) theo mẫu [poc_out/vis_comparison.png](./poc_out/vis_comparison.png): Grid so sánh 6 cột (`Input`, `GT`, `TruFor`, `DTD`, `ADCD-Net gốc`, `Mốc 1`, `Mốc 2 + SRM`), hình `srm_map` vs `pp_maps`, và `Error Analysis`.

---

## VI. Lộ trình 8 Tuần (`16/11/2026` – `12/01/2027`)

- **Tuần 1 (`16/11 – 22/11/2026`) — Chốt Phương án 3A, Env & Script Dữ liệu:**
  - *Đã xong (`18/11`):* PoC RTX 4060, sửa symlink `DocTamperV1` (120k LMDB), vá `src/ds.py` (`BGR→RGB` & fallback OCR).
  - *Squad 1:* Thiết viết `split_tsroie_by_findit2.py` (`998/255/326`) & `gen_ai_findit2.py` (`512×512` `LaMa` + script `AnyText` ghép biên 3px); Mỹ vẽ EDA `T-SROIE` & `findit2`.
  - *Squad 2:* Zang dựng env đồng bộ trên Server `4×3090` (bao gồm `AnyText`) & RTX 3050; Quag viết `src/ds_hybrid.py` (`min_qf=65`, đọc `ocr_mask.png`).
  - *Squad 3:* Kiều (Pháp) soạn note lý thuyết SOTA/SRM & chốt Metrics Protocol; Zam cài `TruFor`, `DTD` và viết `scripts/eval_metrics.py`.
- **Tuần 2 (`23/11 – 29/11/2026`) — Reproduction (`Mốc 0`), Sinh xong `LaMa` + `AnyText` & Zero-shot:**
  - *Squad 1 & Zang:* Sinh xong `LaMa-FindIt2` (Train/Val/Test), `AnyText-FindIt2` (~300 ảnh Unseen Test trên `4×3090`), Stress Test `QF 70/50`; Mỹ kiểm tra Visual QA 50 mẫu.
  - *Squad 2:* Zang chạy `seg_char.py` xuất `ocr_mask` + chạy **Mốc 0** trên 15–20% `DocTamperV1` (`4×3090`), xuất biểu đồ TensorBoard báo cáo GVHD `29/11`.
  - *Squad 3:* Zam chạy Zero-shot (**TruFor, DTD, ADCD-Net gốc**) trên cả 3 tập Test.
- **Tuần 3 (`30/11 – 06/12/2026`) — Mốc An Toàn 1 (`Fine-tune 60/40`):**
  - *Squad 2:* Zang chạy DDP **Mốc 1** trên `4×3090`; Quag tích hợp `SRMHighFreqBranch` + `enable_grad_checkpointing` vào `src/model/model.py` và debug trên RTX 4060.
  - *Squad 3 & PM:* Zam chấm điểm Mốc 1; Viết dựng khung Báo cáo Chương 1.
- **Tuần 4 (`07/12 – 13/12/2026`) — Mốc Nâng Cao 2 (`ADCDNetWithSRM`):**
  - *Squad 2:* Zang train **Mốc 2** trên Server `4×3090` (`lr=1e-5` backbone, `1e-4` nhánh `srm_branch`).
  - *Squad 3:* Zam chấm điểm Mốc 2, so sánh Ablation Mốc 1 vs. Mốc 2 (đặc biệt trên **Unseen `AnyText`**) và Stress Test `QF 70, 50`.
- **Tuần 5 (`14/12 – 20/12/2026`) — Hình ảnh, Error Analysis & `Code Freeze`:**
  - *Mỹ & Kiều:* Xuất hình 300 DPI (`srm_map`, Heatmap) & viết Error Analysis.
  - *Viết:* Rà khớp 100% số liệu, cập nhật `README.md` và **KHÓA CODE (`v1.0-freeze`) tối 20/12/2026**.
- **Tuần 6 – Tuần 8 (`21/12/2026 – 12/01/2027`) — Viết Báo cáo Chính, Slide & Mock Defense:**
  - *Tuần 6 (`21/12 – 27/12`):* Kiều giao nháp Ch.2, Quag/Thiết giao nháp Ch.3; **Viết** viết chính Ch.1, Ch.4, Ch.5 và biên tập toàn cuốn gửi GVHD duyệt ngày `27/12`.
  - *Tuần 7 (`28/12 – 03/01`):* Mỹ hoàn tất Slide Deck (`03/01`); Viết chỉnh sửa theo góp ý GVHD.
  - *Tuần 8 (`04/01 – 11/01`):* Tổ chức **2 buổi Mock Defense** (`20:00 VN = 14:00 CET`), chốt bản in/PDF và **nộp ngày 12/01/2027**.

---

## VII. Cấu trúc Báo cáo 5 Chương & Future Work

- **Chương 1 — Giới thiệu:** Gian lận hóa đơn bằng AI, khoảng trống trên Diffusion/Inpainting & đóng góp 3 bước.
- **Chương 2 — Cơ sở Lý thuyết:** RGB/DCT forensics, dấu vết AI (GAN, FFC Inpainting, VAE/Diffusion upsampling, SRM filters), tổng quan **TruFor, DTD, ADCD-Net**.
- **Chương 3 — Phương pháp Đề xuất:** Chiến lược dữ liệu 100% Document/Receipt chống rò rỉ theo ID `findit2` (Seen `T-SROIE + LaMa` vs. Unseen `AnyText` ghép biên 3px), vá lỗi pipeline `src/ds.py`, kiến trúc `SRMHighFreqBranch`.
- **Chương 4 — Thực nghiệm & Phân tích:** Reproduction (`Mốc 0`), Bảng Benchmark tổng hợp, Stress Test JPEG, Ablation (Mốc 1 vs. Mốc 2), trực quan hóa `srm_map`/Heatmap & Error Analysis.
- **Chương 5 — Kết luận & Future Work:**
  1. **Mở rộng Cross-domain sang Ảnh Cảnh tự nhiên:** Thử nghiệm và thích nghi trên tập dữ liệu chữ cảnh tự nhiên như `OSTF` (9 bộ sinh AI ngoài trời/poster).
  2. **Mở rộng kiểm thử các mô hình sinh mới & đa phương thức:** Thử thêm `TextDiffuser-2`, `Flux Fill`, `SDXL Inpainting`, và bổ sung nhánh **Copy-Move (`get_cm_map`)** để phát hiện cả giả mạo truyền thống lẫn AI.
  3. **Fine-tune toàn bộ đối trọng & tối ưu đa tỷ lệ:** Fine-tune cả `TruFor`, `DTD`, `FFDN` trên cùng tập hóa đơn AI để đối chứng toàn diện ở cấp độ bài báo quốc tế.

---

## VIII. Quy tắc Quản trị Rủi ro

1. **Giữ nguyên thư mục tham chiếu gốc:** Không sửa trực tiếp `ADCD-Net/`, `findit2/`, `doctamper_root/archive/`, `T-SROIE_test/`. Code phát triển đặt trong `src/` và `scripts/`.
2. **Chống tràn VRAM 8GB khi dev:** Luôn bật `enable_grad_checkpointing` và `bs >= 2` (`BatchNorm1d` trong `dct_align_head`) trên RTX 4060.
3. **Backup Checkpoint Server 3090 & Cố định Seed:** Lưu checkpoint mỗi epoch ra ngoài server và lưu seed/ID split vào `.json`.
4. **Sync 2 buổi/tuần (`20:30 VN = 14:30 CET` Thứ 4 & Chủ nhật):** Đảm bảo Kiều tại Pháp và cả nhóm đồng bộ liên tục.
