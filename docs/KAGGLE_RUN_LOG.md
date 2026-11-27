# Nhật ký chạy Kaggle — các lần lỗi & bài học

Mỗi lần chạy hỏng được ghi theo khuôn: **Triệu chứng → Nguyên nhân gốc → Cách sửa (commit) → Bài học**. Nhãn cell `NBx-y` khớp với nhãn trong notebook (sinh bởi `notebooks/kaggle/make_notebooks.py`). Log thô của từng lần chạy để ở `notebooks/kaggle/nb*_output.md` (không commit).

## Checklist trước khi viết/sửa notebook (rút ra từ các lỗi bên dưới)
1. **Chạy `00_doctor` trước** trên một session mới. Chỉ chạy 01/02/03 khi doctor không còn `FAIL` (trừ các check được ghi là tùy chọn).
2. **Không cài gì vào Python của Kaggle.** Mọi script chạy trong venv Python 3.10 ở `/tmp`. Python của Kaggle chỉ để clone repo và hiển thị ảnh. Image Kaggle thay đổi không báo trước (11/2026: Python 3.13, numpy 2.1, Pillow 13).
3. **Khóa cả cây dependency, không chỉ package chính:** dùng `uv pip install --exclude-newer <ngày>` (snapshot), cộng với pin rõ các package "anh em" phát hành cùng nhau (paddle / paddleocr / paddlex).
4. **Ghim tên model**, không để thư viện tự chọn model mặc định, vì mặc định đổi theo phiên bản.
5. Mỗi bước nặng có **cell smoke test nhỏ** (1–10 mẫu) và **chạy tiếp được** sau khi bị ngắt.
6. Dữ liệu "chính thức" chưa chắc sạch: kiểm tra giả định (ví dụ split disjoint) bằng code và **in ra** chỗ vi phạm, thay vì chỉ assert.

---

## #1 — NB1-4, run 1 (27/11/2026): `ValueError: (InvalidArgument) Type of attribute: strides is not right`
- **Triệu chứng:** `PaddleOCR(...)` crash khi tạo predictor cho `PP-OCRv6_medium_det`. NB1-2 có cảnh báo pip conflict (pyyaml, pillow).
- **Nguyên nhân gốc:**
  - Cài thẳng vào Python 3.13 của Kaggle với `paddlepaddle==3.0.0` nhưng để `paddleocr` tự lấy bản mới nhất (3.7.0).
  - paddleocr 3.7 mặc định dùng model **PP-OCRv6** (định dạng mới), mà runtime paddle 3.0 không đọc được.
- **Cách sửa (`bdccb66`):**
  - Thêm venv cô lập `scripts/setup_kaggle_venv.sh ocr`.
  - Ghim `paddleocr>=3.0,<3.1`.
  - Ghim tên model `PP-OCRv5_server_det` / `PP-OCRv5_mobile_rec`.
- **Bài học:** checklist mục 2 và 4.

## #2 — NB1-3, run 1 (27/11/2026): `[FAIL] findit2 train/val share 1 IDs ['X51006619709']`
- **Triệu chứng:** script chia split thoát với exit 1.
- **Nguyên nhân gốc:**
  - Chính file split của findit2 có `X51006619709.png` ở **cả** train và val. Bản ở val là `forged=1`, nên không mất ảnh sạch nào.
  - Phát hiện kèm một lỗi tiềm ẩn: `id2split` ghi đè theo thứ tự, nên ID trùng bị gán vào split **sau cùng**. Ảnh T-SROIE của ID đó lẽ ra có thể rơi vào test.
- **Cách sửa (`bdccb66`):**
  - Mỗi ID thuộc về **split sớm nhất** và bị bỏ khỏi các split sau.
  - Script in `[CONFLICT]` và ghi vào `split_summary.json → id_conflicts`.
  - Thêm `--strict` để dừng hẳn nếu muốn.
- **Bài học:** checklist mục 6. Khi viết báo cáo, ghi rõ chính sách này trong phần chia dữ liệu (Ch.3).

## #3 — NB1-4, run 2 (27/11/2026): `TypeError: PaddlePredictorOption.__init__() takes 1 positional argument but 2 were given`
- **Triệu chứng:** venv OCR dựng thành công (`paddle 3.0.0 | paddleocr 3.0.3`), nhưng `PaddleOCR(...)` crash trong `prepare_common_init_args`.
- **Nguyên nhân gốc:**
  - `paddleocr 3.0.3` khai báo `paddlex` không có cận trên, nên uv lấy **paddlex 3.7.2**. API của paddlex 3.7 lệch với paddleocr 3.0.
  - Lỗi lặp lại cùng kiểu với #1, chỉ ở tầng dependency sâu hơn. Ghim package chính thôi là chưa đủ.
- **Cách sửa (commit này):**
  - Mọi venv được resolve bằng **snapshot ngày** `--exclude-newer`: OCR `2025-07-15`, LaMa `2024-07-01`, AnyText `2024-08-01`.
  - Ghim thêm `paddlex>=3.0,<3.1`.
  - Mỗi venv lưu `freeze.txt` (khóa đầy đủ) và file `.ok` chỉ được ghi khi import thành công.
  - Torch được cài riêng, không qua snapshot (index PyTorch có thể không có thời điểm upload).
- **Phát hiện kèm (vô hại):** `uv venv` cảnh báo `--system flag has no effect` vì Kaggle đặt sẵn `UV_SYSTEM_PYTHON`. Các script giờ `unset` biến này.
- **Bài học:** checklist mục 3. Thêm `00_doctor` + `scripts/kaggle_doctor.py` để phát hiện loại lỗi này **trước** khi chạy notebook thật, đúng như đề xuất của Thiết: tách cell chẩn đoán ra để không phải viết lại notebook nhiều lần.

---

## Rủi ro còn mở (chưa kiểm chứng trên Kaggle)
| Rủi ro | Doctor check phát hiện | Phương án nếu FAIL |
|---|---|---|
| Ngày snapshot OCR chưa đúng (paddleocr 3.0.x / paddlex 3.0.x chưa có hoặc đã quá) | `ocr: versions`, `declared_deps`, `PaddleOCR init` | Đổi `OCR_EXCLUDE_NEWER=<ngày>` rồi chạy lại cell, không cần sửa code |
| `uv` không resolve được torch khi có snapshot (lỗi "no version of torch") | build venv lama / anytext | Bỏ torch khỏi lệnh cài thứ hai trong script setup |
| Repo AnyText (HEAD) thay đổi | `anytext: anytext_repo` (in commit + ngày) | Ghim commit trong `setup_anytext.sh` |
| Chiều của `draw_pos` (trắng/đen) | `anytext --full: text-editing` (so diff trong box với ngoài box) | `--draw-pos-format file` |
| OCR CPU quá chậm (>8 s/hóa đơn) | `ocr: PaddleOCR predict` (thời gian) | Đổi det sang `PP-OCRv5_mobile_det` |
