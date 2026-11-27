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

## #4 — NB0-8 (doctor `anytext --full`), run 1 (27/11/2026): `FileDownloadError: File README.md download incomplete, content_length: None`
- **Triệu chứng:** load pipeline AnyText thất bại ngay khi tải file đầu tiên của model. Check sau đó báo `KeyError: 'pipe'`, nhưng đó chỉ là hệ quả.
- **Nguyên nhân gốc:**
  - `modelscope==1.10.0` (11/2023) là bản AnyText bắt buộc để **chạy** model, nhưng client này không còn tương thích với **server** ModelScope hiện nay: server không gửi header `content-length` nữa.
  - Đây là loại lỗi mà khóa theo ngày không chặn được, vì bên kia là một dịch vụ đang chạy, không phải package.
- **Cách sửa (commit này):**
  - `scripts/download_anytext_model.sh` tải trọng số bằng **client modelscope mới** trong venv riêng `/tmp/venv_msdl` (thử `damo/…` rồi `iic/…`, revision `v1.1.3` rồi mặc định) vào `/tmp/anytext_model`, và kiểm tra có `ms_wrapper.py` + `configuration.json`.
  - modelscope 1.10 chỉ còn **load từ thư mục local**, thông qua `scripts/anytext_loader.py`, module dùng chung cho script sinh dữ liệu và doctor.
- **Phát hiện kèm:** `setup_anytext.sh` cũ ghim các phiên bản **khác** với `environment.yaml` chính thức của AnyText (torch 2.1.2 thay vì 2.0.1, open_clip 2.0.2 thay vì 2.7.0, torchmetrics 0.6.0 thay vì 0.11.4…). Đã căn lại theo bản chính thức, khóa ngày 2024-01-15, ghim repo ở commit `3d4a260`, và bỏ các package chỉ dùng cho demo/translator.
- **Bài học:**
  - Khi tích hợp một repo bên ngoài, **đọc file environment/inference của chính repo đó** trước khi tự ghim phiên bản.
  - Tách bước *tải* (dùng client mới) khỏi bước *chạy* (dùng runtime cũ đã ghim).

## #5 — NB0-7 (doctor `anytext`), run 1 (27/11/2026): `ModuleNotFoundError: No module named 'ms_wrapper'` — lỗi của chính doctor
- **Nguyên nhân gốc:** check doctor (và `gen_anytext_findit2.py`) giả định `ms_wrapper.py` nằm trong repo GitHub của AnyText. Thực tế file này nằm trong **repo model** trên ModelScope và chỉ có sau khi tải trọng số. Môi trường không có lỗi gì.
- **Cách sửa (commit này):** thay bằng check "thư mục model có đủ `ms_wrapper.py` + `configuration.json`", và bỏ đoạn import sai trong script sinh dữ liệu.
- **Bài học:** check chẩn đoán cũng là code và cũng cần kiểm chứng. Một FAIL có thể là do giả định của check sai, không nhất thiết là môi trường lỗi.

## #6 — NB1-5, run 1 (27/11/2026): OCR toàn bộ trên CPU chạy quá lâu
- **Triệu chứng:** không có lỗi, nhưng OCR ~825 hóa đơn (`PP-OCRv5_server_det`, ảnh ~932×1697) trên 4 nhân CPU của Kaggle chậm tới mức không chấp nhận được.
- **Nguyên nhân gốc:** gói PyPI `paddlepaddle` **chỉ có CPU**. Bản GPU là gói khác, `paddlepaddle-gpu`, và bản 3.0.0 chỉ có trên index riêng của Paddle (PyPI dừng ở 2.6). Notebook 01 lại được đặt `Accelerator = None`.
- **Cách sửa (commit này):**
  - `setup_kaggle_venv.sh ocr` tự phát hiện GPU (`nvidia-smi`). Có GPU thì cài `paddlepaddle-gpu==3.0.0` từ index Paddle (thử `cu126` rồi `cu118`), **không** qua snapshot giống torch, rồi chạy phép nhân ma trận trên GPU để kiểm tra. Nếu thất bại thì tự lùi về bản CPU, không bao giờ để venv hỏng. Ép thiết bị bằng `OCR_DEVICE=cpu|gpu`. Kết quả ghi vào `.ok` (`device=`, `want=`).
  - `select_edit_regions.py --device auto|gpu:0|cpu` (mặc định auto) và in `s/receipt` + ETA mỗi 25 hóa đơn.
  - Notebook 01 chuyển sang GPU T4. Doctor kiểm tra mạng tới index Paddle và in thiết bị OCR sẽ dùng.
- **Bài học:** "ghim phiên bản" phải bao gồm cả **biến thể phần cứng** của gói (CPU/GPU là hai tên gói khác nhau). In tốc độ/ETA ngay từ đầu để phát hiện chậm sớm, thay vì đợi hết.

## #7 — NB1-5, run 2 (27/11/2026): OCR GPU xong `825/825 1.2s/receipt` nhưng cell vẫn chạy lâu trên CPU
- **Triệu chứng:** OCR trên GPU xong khoảng 17 phút, sau đó cell vẫn chạy và Kaggle chỉ thấy CPU hoạt động, không có dòng log nào.
- **Nguyên nhân gốc:**
  - Bước chọn vùng sửa (chạy CPU, đúng như thiết kế) tạo `Receipt.text_mask` cho **cả 825 hóa đơn** bằng `MaxFilter(13)` của PIL trên ảnh độ phân giải gốc.
  - Đo được: 0.7 s cho ảnh 932×1697, và **13.5 s** cho ảnh scan 7016×4961 (có trong findit2, xem log `max_side_limit`).
  - Trong khi đó mask này chỉ phục vụ Insertion, tức chỉ hóa đơn test.
- **Cách sửa (commit này):**
  - Mask được tạo lazy, và nới mỗi box thêm 6px thay vì dilation. Kết quả **giống hệt từng pixel** (đã kiểm chứng), nhưng chỉ mất ~0.007 s/ảnh.
  - In thêm `[SELECT] ...` để biết đang ở bước nào.
- **Bài học:** đo luôn cả bước "nhẹ" sau bước nặng, vì dữ liệu thật có ảnh lớn hơn median rất nhiều. Mỗi pha dài phải in log tiến độ.

## #8 — NB2-4 (smoke 10 mẫu LaMa), run 1 (27/11/2026): `CUDA out of memory. Tried to allocate 25.45 GiB`
- **Triệu chứng:** 4 mẫu đầu sinh tốt. Tới mẫu thứ 5 thì TorchScript của LaMa OOM. Cảnh báo `cudnnException ... NOT_SUPPORTED` đi kèm vô hại: cuDNN chỉ thử một plan khác.
- **Nguyên nhân gốc:**
  - Thiết kế cũ cho LaMa inpaint **cả tấm hóa đơn** ở độ phân giải gốc để có ngữ cảnh giấy.
  - findit2 có các bản scan cỡ 7016×4961 (đã thấy ở #7). Bộ nhớ của FFC/FFT tăng theo diện tích ảnh, nên cần khoảng 25 GiB, trong khi T4 chỉ có 16 GB.
  - Smoke test và doctor chỉ thử ảnh cỡ median nên không bắt được.
- **Cách sửa (commit này):**
  - LaMa chỉ inpaint một **cửa sổ ngữ cảnh**: mảnh 512 nới thêm `--context 256` px mỗi phía (tối đa 1024×1024), vẫn ở độ phân giải gốc, không resize.
  - Khi OOM thì thử lại với 128, rồi 0. Mẫu lỗi được liệt kê ở cuối, các mẫu khác chạy tiếp (trước đây một mẫu lỗi làm chết cả cell).
  - Kiểm tra thêm: mask phải nằm trọn trong crop.
  - Phát hiện kèm: venv LaMa in `want=cpu` (nhãn chỉ dành cho OCR), đã đổi thành `want=n/a`.
- **Bài học:**
  - Smoke test phải gồm **ca biên của dữ liệu** (ảnh lớn nhất), không chỉ N mẫu đầu.
  - Vòng lặp sinh dữ liệu dài phải chịu lỗi từng mẫu, không được để một mẫu làm chết cả lần chạy.

## #9 — NB0-8/NB0-9 (doctor AnyText), run 2 (27/11/2026): không có model dir + `HTTPError 操作过于频繁，请稍后再试!`
- **Triệu chứng:**
  - `model dir` FAIL ("got hub id"): script tải không tạo được `/tmp/anytext_model`.
  - `pipeline load` FAIL với mã 10013201001, nghĩa là ModelScope báo "thao tác quá thường xuyên, thử lại sau".
- **Nguyên nhân gốc:**
  1. ModelScope **giới hạn tần suất tải** với khách ẩn danh, mà IP của Kaggle dùng chung nên càng dễ bị chặn.
  2. Repo ModelScope nặng **~15 GB**, chứ không phải 5–6 GB như mình ước: 7.9 GB trong đó là model dịch zh→en, không dùng tới vì `use_translator=False`. Tải toàn bộ vừa lâu vừa sinh ra rất nhiều request.
  3. Ở chế độ `auto`, loader **lặng lẽ chuyển sang hub** khi không có thư mục local. Vì vậy lỗi gốc (tải thất bại) bị che bởi một lỗi thứ cấp (bị chặn tần suất). Đây là lỗi thiết kế của mình.
- **Cách sửa (commit này):**
  - `download_anytext_model.sh` thử theo thứ tự:
    1. Dùng bản đã có trong `/tmp/anytext_model` hoặc **dưới `/kaggle/input`**.
    2. ModelScope `v1.1.3`, **bỏ qua** `nlp_csanmt_translation_zh2en/` và `description/`, thử 3 lần với thời gian chờ tăng dần 0 / 60 / 180 s.
    3. Mirror HuggingFace `camenduru/anytext`, ghim theo commit, chỉ tải ckpt + clip + wrapper, rồi kiểm tra sha256 của ckpt. `ms_wrapper.py` lấy từ ModelScope v1.1.3 nếu tải được hai file nhỏ này, nếu không thì dùng bản cũ hơn có sẵn trên mirror.
  - `anytext_loader.resolve_model('auto')` không bao giờ chuyển sang hub nữa: không có model thì báo lỗi ngay, nói rõ cần chạy gì. Model dir được coi là đủ khi có cả ckpt và clip.
- **Bài học:**
  - Ngoài phiên bản client, dịch vụ bên ngoài còn có thể **chặn tần suất**, nên luôn cần nguồn dự phòng và cách lưu lại (Kaggle Dataset) để không phải tải lại.
  - **Không lặng lẽ chuyển sang phương án khác** khi bước trước đã lỗi, vì như vậy lỗi gốc sẽ bị che mất.
  - Kiểm tra kích thước repo trước khi ước lượng thời gian và dung lượng ổ đĩa.

## #10 — NB0-9 (doctor `anytext --full`), run 3 (27/11/2026): `ModuleNotFoundError: AnyTextPipeline: No module named 'skimage'`
- **Tiến triển:** bản sửa #9 hoạt động. Model tải về được từ ModelScope (`damo/...`), và NB0-8 cho 6 PASS / 0 FAIL.
- **Triệu chứng:** khi load pipeline thì thiếu `skimage`.
- **Nguyên nhân gốc:**
  - `cldm/recognizer.py` của AnyText import `skimage` (dùng `_umeyama`), nhưng `environment.yaml` không ghi `scikit-image`. Gói này chỉ được kéo vào **gián tiếp** qua `basicsr==1.4.2`.
  - Lúc dựng venv, mình bỏ các gói "chỉ dùng cho demo/train" (basicsr, gradio, tensorflow…) mà không kiểm tra **cây import thực tế**, nên làm rơi luôn dependency gián tiếp này.
  - Check `import modelscope.pipelines` của doctor không bắt được, vì code của AnyText chỉ được import khi load pipeline.
- **Cách sửa (commit này):**
  - Thêm `scikit-image`, `six`, `matplotlib` vào `setup_anytext.sh` (vẫn khóa theo snapshot). Tăng `deps=2` trong stamp để venv cũ tự được dựng lại.
  - Thêm `scripts/anytext_import_scan.py`: quét tĩnh bằng `ast`, không chạy code, bắt đầu từ `ms_wrapper.py` và đi theo **mọi** import cục bộ trong repo (kể cả import trong hàm). Script liệt kê mọi module bên thứ ba, rồi kiểm tra bằng `find_spec` trong chính venv.
  - Setup chạy bước quét này trước khi ghi `.ok`, và doctor có thêm check `import scan`. Thiếu module nào thì báo **tất cả cùng lúc**, kèm file import nó, thay vì mỗi lần chạy chỉ lộ ra một lỗi.
- **Bài học:** khi rút gọn danh sách package của một repo ngoài, phải **kiểm tra cây import thực tế** bằng máy, không đoán theo tên package. Gói trung gian (basicsr) có thể là thứ duy nhất kéo vào một dependency thật.

---

## #11 — NB0-5 (setup AnyText), run 4 (27/11/2026): `AttributeError: module 'open_clip' has no attribute '__version__'`
- **Triệu chứng:** mọi package đã cài xong, nhưng dòng kiểm tra cuối của `setup_anytext.sh` crash khi in `open_clip.__version__`. Do `set -e`, script dừng luôn: không ghi `.ok`, chưa chạy import scan (#10).
- **Nguyên nhân gốc:** `open_clip_torch==2.7.0` (bản ghim theo `environment.yaml`) không export `__version__` ở cấp package; chỉ các bản mới hơn mới có. Câu lệnh in version được viết theo thói quen với bản mới, chưa từng chạy với bản 2.7.0.
- **Cách sửa (commit này):**
  - Setup lấy version qua `importlib.metadata.version('<tên distribution>')` thay vì `module.__version__`.
  - `versions()` của doctor: nếu module không có `__version__` thì hỏi `importlib.metadata` (qua `packages_distributions`), không trả `?` nữa.
  - Danh sách package không đổi nên không cần tăng `deps=`. Vì `.ok` chưa được ghi, chạy lại NB0-5 sẽ cài lại (nhanh, nhờ cache của uv) rồi chạy tiếp import scan.
- **Bài học:** câu lệnh kiểm tra cũng là code. Đọc version phải qua metadata của distribution, không qua thuộc tính của module.

---

## Rủi ro còn mở (chưa kiểm chứng trên Kaggle)
| Rủi ro | Doctor check phát hiện | Phương án nếu FAIL |
|---|---|---|
| Ngày snapshot OCR chưa đúng (paddleocr 3.0.x / paddlex 3.0.x chưa có hoặc đã quá) | `ocr: versions`, `declared_deps`, `PaddleOCR init` | Đổi `OCR_EXCLUDE_NEWER=<ngày>` rồi chạy lại cell, không cần sửa code |
| Cả ModelScope lẫn mirror HF đều hỏng/chặn | `download_anytext_model.sh` in `[FAIL]` | Tải 1 lần ở session nào được, upload thư mục model thành Kaggle Dataset private → Add Input; `auto` tự tìm |
| `ms_wrapper.py` bản mirror HF (cũ hơn) không nhận `font_path`/`text-editing` | doctor `anytext --full` | Chạy lại lúc ModelScope hết chặn để lấy wrapper v1.1.3 (`ANYTEXT_SOURCE=modelscope`) |
| `uv` không resolve được torch khi có snapshot (lỗi "no version of torch") | build venv lama / anytext | Bỏ torch khỏi lệnh cài thứ hai trong script setup |
| Chiều của `draw_pos` (trắng/đen) | `anytext --full: text-editing` (so diff trong box với ngoài box) | `--draw-pos-format file` |
| `paddlepaddle-gpu` không cài/chạy được (index Paddle chậm/không vào được, lệch cuDNN) | NB1-2 in `[WARN] ... falling back to CPU`; doctor `net paddle_gpu_index`, `paddle_device` | Tự lùi về CPU; nếu CPU vẫn chậm, thêm `--det-model PP-OCRv5_mobile_det` vào NB1-5 |
