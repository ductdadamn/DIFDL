# Sinh dữ liệu AI trên findit2 (LaMa + SD-1.5 + AnyText) — Hướng dẫn chạy

Phụ trách: Squad 1 (CSV #6, #7, #8 – phần AnyText). Toàn bộ dữ liệu sinh ra nằm trong `data_synth/` (đã có trong `.gitignore`).

## Quy ước dữ liệu (đã chốt)
| Mục | Quy ước |
|---|---|
| Nguồn | Chỉ hóa đơn sạch (`forged == 0`) của `findit2/{train,val,test}.txt`, bỏ `X51008114260.png`. AnyText **chỉ** dùng `test`. |
| Kích thước | Mảnh `512×512`, góc trên-trái là bội số của 8. |
| Định dạng | **PNG không nén** cho ảnh và mask. Bản JPEG `QF 70/50` (Test Set 3) do notebook 05 tạo riêng, xem mục *Stress Test JPEG*. |
| LaMa (Removal) | Inpaint một **cửa sổ ngữ cảnh** (mảnh 512 + 256px mỗi phía, tự giảm 128 → 0 nếu thiếu VRAM; KAGGLE_RUN_LOG #8) với mask là các dòng chữ OCR nới rộng 3px, sau đó chép lại phần ngoài mask từ ảnh gốc, nên GT = mask chính xác tới từng bit. |
| SD-1.5 Inpainting (Removal) | **Train 500:** vùng mới, khác mọi dòng LaMa/AnyText đã dùng (seed 123, ≤ 2 mảnh/hóa đơn). **Val 100 / Test 200:** **cùng crop + mask** với LaMa (`pair_id`), so sánh CNN và Diffusion có cặp đối chứng. GT giống hệt LaMa (dòng OCR + 3px). SD nhận mask phủ trọn các ô latent 8×8 của GT, prompt "blank receipt paper, no text" và negative cấm chữ. Kết quả được ghép bằng alpha **feather vào trong GT** (0.3 ở mép → 1 ở 3px bên trong, đúng 0 bên ngoài), nên ngoài GT bit-exact như LaMa, không lộ viền do VAE đổi màu toàn ảnh. |
| Chính sách ghép biên | Dữ liệu **train** (LaMa, SD-1.5): ngoài GT bit-exact, nhãn chính xác tới từng pixel. Dữ liệu **chỉ dùng test** (AnyText, TextDiffuser): Gaussian ~3px **ra ngoài** GT theo MASTER_PLAN. Với hai bộ này, dải chuyển tiếp không tính là giả. |
| QA mực SD-1.5 | Tỷ lệ pixel tối (<120) trong GT phải ≤ max(2%, tỷ lệ ở vòng giấy 6px quanh GT + 2%). Không đạt → thử lại seed+1, seed+2; vẫn không đạt → giữ bản ít mực nhất, `flag=true` trong `qa/sd15_ink.jsonl`. Vùng đó vẫn bị sửa nên nhãn vẫn đúng. Báo tỷ lệ flag ở Limitations. |
| AnyText (Replacement / Insertion) | `text-editing` trên mảnh 512, rồi ghép biên Gaussian ~3px: `I_out = M_soft·I_AI + (1−M_soft)·I_orig`. GT = mask nhị phân **trước** khi blur. File `raw/` lưu ảnh AI chưa ghép. |
| Đối chứng sạch | Mỗi mẫu có thêm `pristine/<id>.png`: cùng crop, lấy từ hóa đơn gốc. Dùng file này khi tính Image-level AUC/FPR để so ảnh cùng kích thước và cùng định dạng. |
| Tái lập | Seed toàn cục 42, mỗi mẫu có `seed` riêng trong manifest. |

Cấu trúc thư mục đầu ra:
```
data_synth/
  splits/        findit2_clean_*.txt, tsroie_*.txt, split_summary.json
  ocr_cache/     <stem>.json  (box + text OCR của từng hóa đơn)
  manifest_lama.jsonl, manifest_anytext.jsonl, selection_stats.json
  manifest_sd15.jsonl, sd15_selection_stats.json          (notebook 04, từ output nb01)
  lama/{train,val,test}/{images,masks,pristine}/<id>.png
  sd15/{train,val,test}/{images,masks,pristine,raw}/<id>.png
  sd15_textdemo/{orig,sd15,raw}/, textdemo_sheet.png      (hình minh họa, KHÔNG thuộc dataset)
  anytext/test/{images,masks,pristine,raw}/<id>.png
  qa/            <gen>_report.json, <gen>_contact_sheet.png
```

## Bộ dữ liệu đã phát hành — v1 (28/11/2026)
Dùng **đúng các version dưới đây** cho train và đánh giá, và ghi rõ trong báo cáo (Chương 3). Nếu ai chạy lại notebook, Kaggle tạo Version 2 với output **khác** → không trộn.

| Notebook (Kaggle, private) | Version | Code (`feat/data`) | Nội dung output | Kết quả QA |
|---|---|---|---|---|
| [nb01-prep](https://www.kaggle.com/code/mitepiccagole/nb01-prep) | 1 | `6a032a5` (code chọn vùng không đổi từ đó) | `splits/`, `ocr_cache/`, `manifest_lama.jsonl` (1400), `manifest_anytext.jsonl` (250), `selection_stats.json` | sạch 483/159/183; 21 hóa đơn < 512px bị bỏ; `test_small_text_rate` = 0.0007 |
| [nb02-lama](https://www.kaggle.com/code/mitepiccagole/nb02-lama) | 1 | `4d16255` | `lama/{train,val,test}/{images,masks,pristine}/` | `generated_ok` 1000/200/200, 0 lỗi, 0 cảnh báo |
| [nb03-anytext](https://www.kaggle.com/code/mitepiccagole/nb03-anytext) | 1 | `944cbe9` | `anytext/test/{images,masks,pristine,raw}/` (175 replacement + 75 insertion) | `generated_ok` 250, 0 lỗi; 4 text chuẩn hóa NFKC (KAGGLE_RUN_LOG #13) |
| [anytext-save](https://www.kaggle.com/code/mitepiccagole/anytext-save) | 1 | `739a295` (script tải) | `anytext_model/` — trọng số AnyText v1.1.3 từ ModelScope `damo/` (~7.5GB) | doctor đọc từ `/kaggle/input` (chỉ đọc): 11 PASS (#12) |

**Lấy dữ liệu** (cần được chủ notebook thêm làm *Collaborator — Can view*):
- Trên Kaggle: *Add Input → Notebooks* → tìm tên notebook → output nằm ở `/kaggle/input/notebooks/mitepiccagole/<notebook>/data_synth/`.
- Về server/máy: `kaggle kernels output mitepiccagole/nb02-lama -p <thư-mục>` (tương tự `nb03-anytext`), dùng API token của chính mình (`~/.kaggle/kaggle.json`). Lệnh này lấy **version mới nhất** → kiểm tra trang notebook vẫn là Version 1.

**Lưu ý khi dùng:**
- Train: chỉ `lama/{train,val}` và `sd15/{train,val}` (`images + masks`). **Không** train trên `pristine/`, `raw/`, `sd15_textdemo/` hay bất cứ gì của `anytext/` (Test Set 2 = AI chưa thấy).
- Test Set 1: `lama/test` và `sd15/test` là **cặp** (cùng crop + mask, nối theo `pair_id`). Báo F1 riêng cho từng generator; hiệu F1 giữa hai bên chỉ phản ánh khác biệt CNN và Diffusion. Hai bộ dùng chung `pristine/`, nên khi tính AUC/FPR gộp chỉ lấy mỗi pristine một lần.
- Đánh giá AnyText: báo kết quả **tách theo `op`** (replacement / insertion, cột `op` trong manifest) bên cạnh số tổng — insertion thường đậm/khác font nên dễ phát hiện hơn.
- **Không chạy lại nb01**: OCR trên GPU có thể lệch nhẹ → manifest khác → không khớp dữ liệu LaMa/AnyText đã sinh. T-SROIE: chạy riêng `scripts/split_findit2_ids.py` (danh sách findit2 sạch không phụ thuộc T-SROIE).

## Chạy toàn bộ trên Kaggle (không cần GPU hay package trên laptop)
Ba notebook trong [`notebooks/kaggle/`](../notebooks/kaggle/). Lên Kaggle chọn **Create → Notebook → File → Import Notebook**, sau đó chỉnh *Settings* như bảng dưới. Mỗi notebook tự `git clone -b feat/data` repo vào `/tmp/DIFDL`, nên không cần upload code. Nếu repo để private, thêm secret `GITHUB_TOKEN` (*Add-ons → Secrets*); token không được ghi vào output.

| Notebook | Accelerator | Input cần thêm | Thời gian (ước tính) | Output `/kaggle/working/data_synth/` |
|---|---|---|---|---|
| `01_prep_findit2` | GPU T4 (tự lùi về CPU) | Dataset findit2 của nhóm | ~10–20 phút GPU (CPU: vài giờ) | `splits/`, `ocr_cache/`, `manifest_*.jsonl`, `selection_stats.json` |
| `02_lama_findit2` | GPU T4 | Dataset nhóm + **output notebook 01** | 15–30 phút | `lama/**`, `qa/lama_*` |
| `03_anytext_findit2` | GPU T4 | Dataset nhóm + **output notebook 01** | 15 phút dựng env + 40–60 phút | `anytext/test/**`, `qa/anytext_*` |

**Môi trường cô lập (quan trọng):** image Kaggle thay đổi không báo trước (11/2026 đang là Python 3.13, numpy 2, Pillow 13), và cài package không ghim phiên bản từng gây lỗi (`paddleocr` 3.7 kéo model PP-OCRv6 mà `paddle` 3.0 không đọc được). Vì vậy **mọi script đều chạy trong venv Python 3.10 ghim phiên bản** đặt ở `/tmp`, dựng bằng `uv`: `scripts/setup_kaggle_venv.sh ocr|lama` và `scripts/setup_anytext.sh`. Python của Kaggle chỉ dùng để clone repo và hiển thị ảnh. Notebook được sinh từ [`notebooks/kaggle/make_notebooks.py`](../notebooks/kaggle/make_notebooks.py): sửa file này rồi chạy lại để tạo notebook, không sửa `.ipynb` bằng tay. Các cell code có nhãn `# NBx-y` để báo lỗi theo nhãn.

**Chạy `00_doctor` trước (GPU T4, ~20–30 phút)** trên một session mới. Notebook này kiểm tra GPU, ổ đĩa, mạng tới từng nguồn tải và dataset, rồi dựng thử 3 venv và chạy mỗi tool trên 1 ảnh. Chỉ chạy 01/02/03 khi không còn `FAIL`. Các lần lỗi và bài học được ghi trong [KAGGLE_RUN_LOG.md](KAGGLE_RUN_LOG.md).

Cả ba notebook đều bật **Internet = On**. Notebook 01 phải **Save Version (Save & Run All)** xong trước, rồi notebook 02/03 mới thêm output của nó qua *Add Input → Your Work*. Hai notebook 02 và 03 chạy song song được.

Các script tự dò dữ liệu trong `/kaggle/input` (`--findit2-root auto`, `--tsroie-root auto`): tìm thư mục có đủ `train.txt`, `val.txt`, `test.txt` + `train/`, và thư mục `*SROIE*` có `images/` + `masks/`. Nếu tìm thấy 0 hoặc từ 2 kết quả trở lên, script dừng và yêu cầu truyền đường dẫn cụ thể.

### Notebook 01 — chia split & chọn vùng
- Bản thân file split của findit2 có ID hóa đơn trùng giữa các split (ví dụ `X51006619709` có ở cả train và val). Quy tắc xử lý: mỗi ID **thuộc về split sớm nhất** (train → val → test) và bị bỏ khỏi các split sau. Script in dòng `[CONFLICT]` và ghi vào `split_summary.json → id_conflicts`. Thêm `--strict` nếu muốn script dừng khi gặp trùng. Vì vậy số lượng có thể lệch 1–2 so với 483/159/183 (in WARN).
- T-SROIE không có trong dataset findit2 trên Kaggle, nên khi không tìm thấy, script chỉ bỏ qua phần chia T-SROIE. Phần này **không cần cho LaMa/AnyText**, nhưng cần cho `ds_hybrid.py` và Test Set 1: upload T-SROIE lên Kaggle rồi chạy lại cell split.
- Cell smoke test OCR (`--ocr-test 2`) in số dòng, thời gian và chiều cao chữ trung vị trước khi OCR toàn bộ. Model được ghim là `PP-OCRv5_server_det` (giống `src/seg_char.py`) và `PP-OCRv5_mobile_rec`. OCR cache được ghi dần, nên khi session bị ngắt chỉ cần *Run All* lại.
- `selection_stats.json` → `test_small_text_rate` là tỉ lệ dòng số cao dưới 14px, quá nhỏ để AnyText viết. **Đưa con số này vào phần Limitations.**
- Nếu có annotation SROIE task-1, có thể thay OCR bằng `--ocr-source sroie --sroie-box-dir <dir>`.

### Notebook 02 — LaMa
- Venv `/tmp/venv_lama`: torch 2.3.0+cu121 (giống env chính của dự án) và `simple-lama-inpainting==0.1.2`. File `big-lama.pt` được tải vào `/tmp/torch`.
- Smoke test 10 mẫu → QA → xem contact sheet → chạy đủ. Mẫu đã có sẽ được bỏ qua.

### Notebook 04 — SD-1.5 Inpainting Removal
- Add Input: findit2 + **nb01-prep** (bắt buộc) + **nb02-lama** (kiểm tra cặp) + **nb03-anytext** (hình "SD viết chữ").
- `select_sd15_regions.py` đọc `ocr_cache/` + manifest của nb01 (không OCR lại), viết `manifest_sd15.jsonl` và kiểm tra bất biến: train không trùng dòng LaMa/AnyText, không rò ID hóa đơn giữa các split, val/test khớp LaMa từng mục.
- Venv `/tmp/venv_sd15`: torch 2.3.0+cu121, diffusers 0.29. Model `stable-diffusion-v1-5/stable-diffusion-inpainting` ghim commit `8a4288a` (bản `runwayml/` đã bị gỡ khỏi HF), chỉ tải fp16 safetensors (~2 GB) vào `/tmp/sd15_model`. Safety checker tắt (ảnh hóa đơn, tránh bị bôi đen nhầm).
- Smoke test 10 mẫu → contact sheet → chạy đủ 800 (~45–70 phút T4). `qa_synth.py --gen sd15 --pair-root <lama>` kiểm tra thêm: ngoài GT bit-exact, mask và pristine trùng với cặp LaMa, và tỷ lệ flag.
- `--textdemo 10`: SD được yêu cầu **viết** số mới trên cùng crop/hộp với AnyText replacement → `sd15_textdemo/textdemo_sheet.png` (gốc | SD-1.5 | AnyText) cho Mỹ (CSV #21) và Kiều (Error Analysis).

### Notebook 05 — Stress Test JPEG QF 70 / 50 (Test Set 3, CSV #10)
- CPU, ~2 phút. Add Input các notebook sinh dữ liệu (`nb02-lama`, `nb03-anytext`, sau này SD-1.5 / TextDiffuser / T-SROIE). `scripts/make_jpeg_stress.py --src auto` tự tìm mọi `<gen>/test/`.
- Mỗi ảnh được nén **1 lần** từ PNG gốc ra file `.jpg` thật trên đĩa (không mô phỏng trong RAM): Pillow 10.3 ghim (venv `jpeg`), bảng lượng tử IJG chuẩn theo QF, 4:2:0, baseline. Script đọc lại từng file và kiểm tra định dạng, bảng lượng tử và subsampling.
- **`pristine/` cũng được nén cùng QF.** Nếu chỉ nén ảnh giả, mô hình có thể phân biệt giả/thật nhờ định dạng file → AUC/FPR dưới nén vô nghĩa. Mask giữ PNG, chép nguyên từng byte.
- Output: `data_synth_stress/<gen>/test_qf{70,50}/{images,pristine}/<id>.jpg` + `masks/<id>.png` + `stress_manifest.json`. Khi đánh giá, đưa thẳng file `.jpg` vào từng model, **không** chuyển lại sang PNG. Lưu ý cho Zam: `GeneralValDs` của ADCD-Net (`src/ds.py`, `jpeg_record=[100]`) giải mã ảnh rồi nén lại ở QF100 trước khi lấy DCT, nên nó thấy dấu vết nén trong điểm ảnh chứ **không** đọc bảng lượng tử QF70/50 của file. Muốn ADCD-Net dùng DCT gốc của file thì phải sửa loader. Đây là quyết định của Metrics Protocol, phải ghi rõ và áp dụng giống nhau cho mọi mô hình.

### Notebook 03 — AnyText
- `scripts/setup_anytext.sh` dựng venv theo `environment.yaml` của AnyText (torch 2.0.1, modelscope 1.10, repo ghim ở commit `3d4a260`) tại `/tmp/DIFDL/.venv_anytext`.
- Trọng số: Add Input notebook **`anytext-save`** (bản đã lưu, không cần tải). Nếu không có, `scripts/download_anytext_model.sh` tải (~7.5GB) vào `/tmp/anytext_model` bằng client modelscope mới (1.10 không tải được từ server hiện tại), có retry và dự phòng mirror HF.
- Font: AnyText cần một file TTF tại `font/Arial_Unicode.ttf`. Notebook copy `DejaVuSans.ttf` của matplotlib sang tên đó, đủ cho chữ Latin và số.
- **Kiểm tra bằng mắt sau smoke test 5 mẫu** (contact sheet + thư mục `raw/`):
  - Chữ mới phải nằm **đúng trong box**. Nếu AnyText vẽ chữ chỗ khác hoặc vẽ lại cả mảnh, nghĩa là ảnh vị trí bị đảo màu: xóa `data_synth/anytext` rồi chạy lại với `--draw-pos-format file`.
  - Chữ phải đọc được và khớp `new_text`. Nếu chữ nhỏ bị vỡ, thử `--ddim-steps 30`, hoặc chạy lại notebook 01 với `--min-text-h` lớn hơn.

## QA & bàn giao
- `qa_synth.py` trả exit 1 nếu có: ảnh sai kích thước, mask rỗng, ảnh bị thay đổi ngoài vùng mask, hoặc leakage ID giữa các split.
- Contact sheet gồm các cột: pristine | tampered | mask (đỏ) | |diff|×4. Gửi cho Mỹ làm Visual QA (CSV #9).
- Bàn giao: Zang lấy output về server (`kaggle kernels output mitepiccagole/nb02-lama -p ...`, tương tự `nb03-anytext`), chạy `src/seg_char.py` để tạo `ocr/`, rồi `src/build_path_pkl.py`. Tên mask trùng tên ảnh (`.png`), tương thích với `GeneralValDs`.

## Chạy thử pipeline không cần GPU
Hai script sinh dữ liệu có `--backend dummy` (tô màu nền hoặc vẽ chữ bằng PIL). Chế độ này chỉ dùng để kiểm tra I/O, crop, blend và QA (chỉ cần numpy + Pillow), **không** dùng làm dữ liệu thật.
