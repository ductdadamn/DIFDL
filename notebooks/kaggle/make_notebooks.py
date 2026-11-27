"""Generate the 3 Kaggle notebooks of the findit2 synthesis pipeline (edit here, then re-run).

    python notebooks/kaggle/make_notebooks.py

Design rule: every pipeline script runs inside a pinned Python 3.10 venv under /tmp
(scripts/setup_kaggle_venv.sh, scripts/setup_anytext.sh). The Kaggle kernel's own Python
(3.13 / numpy 2 / Pillow 13 as of 11/2026, changes without notice) only clones the repo and
displays images, so Kaggle image updates cannot break the pipeline.
Code cells are labelled "# NB<n>-<k>" so errors can be reported by label.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def build(n, cells, gpu):
    out, k = [], 0
    for kind, src in cells:
        if kind == "md":
            out.append(md(src))
            continue
        k += 1
        body = f"# NB{n}-{k}\n" + src.strip("\n")
        out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                    "source": body.splitlines(keepends=True)})
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True}}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": out}


CLONE = r'''
# Clone the repo into /tmp (NOT /kaggle/working: notebook output must not contain the repo or a token)
import os, subprocess
REPO, BRANCH, DEST = "ductdadamn/DIFDL", "feat/data", "/tmp/DIFDL"
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")   # only needed if the repo is private
except Exception:
    pass
public_url = f"https://github.com/{REPO}.git"
url = f"https://x-access-token:{token}@github.com/{REPO}.git" if token else public_url
if not os.path.exists(DEST):
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, url, DEST], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git clone failed: " + r.stderr.replace(token or "<no-token>", "***"))
    subprocess.run(["git", "-C", DEST, "remote", "set-url", "origin", public_url], check=True)
else:
    subprocess.run(["git", "-C", DEST, "pull", "-q"], check=False)
print(subprocess.run(["git", "-C", DEST, "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
%cd /tmp/DIFDL
'''

FIND_PREP = r'''
# Locate the output of notebook 01 (Add Input -> Your Work -> 01_prep_findit2)
from pathlib import Path
hits = sorted(Path("/kaggle/input").rglob("manifest_{gen}.jsonl"))
assert len(hits) == 1, f"expected exactly one manifest_{gen}.jsonl under /kaggle/input, got {{hits}}"
MAN = str(hits[0]); OUT = "/kaggle/working/data_synth"
print("manifest:", MAN)
'''

SHOW = r'''
from IPython.display import Image, display
import json
display(Image(filename=f"{{OUT}}/qa/{gen}_contact_sheet.png", width=900))
print(json.dumps(json.load(open(f"{{OUT}}/qa/{gen}_report.json"))["hard_failure_counts"], indent=2))
'''

FONT = r'''
# AnyText only needs *a* TTF at font/Arial_Unicode.ttf; DejaVuSans (bundled with matplotlib) covers Latin + digits
import matplotlib, shutil
from pathlib import Path
src = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
dst = Path("/tmp/DIFDL/third_party/AnyText/font/Arial_Unicode.ttf"); dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst); print("font ->", dst)
os.environ["MODELSCOPE_CACHE"] = "/tmp/ms"; os.environ["HF_HOME"] = "/tmp/hf"
PY = "/tmp/DIFDL/.venv_anytext/bin/python"
'''

# ------------------------------------------------------------------ 00 doctor (GPU)
NB0 = build(0, [
    ("md", '''
# 00 — Doctor: kiểm tra Kaggle làm được gì TRƯỚC khi chạy 01/02/03
**Settings:** Accelerator = **GPU T4**, Internet = On, Add Input = dataset findit2.
Mỗi cell in `[PASS]/[FAIL] ...` và một dòng `SUMMARY`; một check lỗi không chặn các check khác.
**Gửi lại:** chỉ cần các dòng `SUMMARY` + `FAIL` (cell cuối gom sẵn). Kết quả cũng lưu ở `/kaggle/working/doctor/*.json`.
Thời gian: ~20–30 phút (dựng 3 venv). Venv nằm trong `/tmp`, nên chỉ dùng lại được trong cùng session.
'''),
    ("code", CLONE),
    ("md", "## 1. Phần cứng, ổ đĩa, mạng tới từng nguồn tải, dataset (Python của Kaggle)"),
    ("code", '''
!python scripts/kaggle_doctor.py --stage env
'''),
    ("md", "## 2. Venv OCR (paddle) + chạy OCR trên 1 hóa đơn thật"),
    ("code", '''
os.environ["PADDLE_PDX_CACHE_HOME"] = "/tmp/paddlex"
!bash scripts/setup_kaggle_venv.sh ocr
!/tmp/venv_ocr/bin/python scripts/kaggle_doctor.py --stage ocr
'''),
    ("md", "## 3. Venv LaMa + inpaint 1 hóa đơn trên GPU"),
    ("code", '''
os.environ["TORCH_HOME"] = "/tmp/torch"
!bash scripts/setup_kaggle_venv.sh lama
!/tmp/venv_lama/bin/python scripts/kaggle_doctor.py --stage lama
'''),
    ("md", "## 4. Env AnyText (~10–15 phút) + tải trọng số (~5–6GB) + kiểm tra"),
    ("code", '''
!bash scripts/setup_anytext.sh 2>&1 | tail -n 5
'''),
    ("code", FONT),
    ("code", """
# Weights + ms_wrapper.py (~5-6GB) with a CURRENT modelscope client (1.10 can't download from today's hub)
!bash scripts/download_anytext_model.sh /tmp/anytext_model
"""),
    ("code", '''
!{PY} scripts/kaggle_doctor.py --stage anytext
'''),
    ("md", "## 5. Load AnyText và thử sửa 1 ảnh — kiểm tra luôn chiều `draw_pos` (inside >> outside)"),
    ("code", '''
!{PY} scripts/kaggle_doctor.py --stage anytext --full
'''),
    ("md", "## 6. Tóm tắt để gửi lại"),
    ("code", '''
import json, glob
for f in sorted(glob.glob("/kaggle/working/doctor/*.json")):
    res = json.load(open(f)); fails = [r for r in res if not r["ok"]]
    print(f"{f.split('/')[-1]}: {len(res) - len(fails)} PASS / {len(fails)} FAIL")
    for r in fails:
        print("   FAIL", r["check"], "->", r["detail"][:500])
'''),
], gpu=True)

# ------------------------------------------------------------------ 01 prep (GPU, CPU fallback)
NB1 = build(1, [
    ("md", '''
# 01 — Chia split & chọn vùng sửa trên findit2 (GPU T4 cho OCR)
**Settings:** Accelerator = **GPU T4**, Internet = On, Add Input = dataset findit2.
OCR chạy GPU (`paddlepaddle-gpu 3.0.0`); nếu không có GPU hoặc bản GPU lỗi thì NB1-2 tự cài bản CPU (chậm hơn nhiều, vẫn đúng kết quả).
Mọi script chạy trong venv Python 3.10 ghim phiên bản (`/tmp/venv_ocr`) → không phụ thuộc Python/numpy của Kaggle.
Output (`/kaggle/working/data_synth`): `splits/`, `ocr_cache/`, `manifest_*.jsonl`, `selection_stats.json`.
Bị ngắt → *Run All* lại (OCR cache được dùng lại). Xong → **Save Version (Save & Run All)**.
'''),
    ("code", CLONE),
    ("code", '''
# Pinned OCR venv: paddlepaddle-gpu 3.0.0 if a GPU is visible (else CPU build) + paddleocr 3.0.x
# (~5-10 min first time: the GPU wheel + CUDA libs are ~3 GB). Last lines must show device=gpu.
!bash scripts/setup_kaggle_venv.sh ocr
PY = "/tmp/venv_ocr/bin/python"
os.environ["PADDLE_PDX_CACHE_HOME"] = "/tmp/paddlex"
'''),
    ("md", "## 1. Chia split theo ID hóa đơn (ID trùng giữa các split → giữ ở split sớm nhất, in `[CONFLICT]`)"),
    ("code", '''
!{PY} scripts/split_findit2_ids.py --findit2-root auto --tsroie-root auto --out /kaggle/working/data_synth/splits
'''),
    ("md", "## 2. Smoke test OCR trên 2 hóa đơn (in số dòng, thời gian, chiều cao chữ trung vị)"),
    ("code", '''
!{PY} scripts/select_edit_regions.py --findit2-root auto --splits-dir /kaggle/working/data_synth/splits --ocr-test 2
'''),
    ("md", "## 3. OCR toàn bộ (~825 hóa đơn) & chọn vùng sửa — in `s/receipt` + ETA mỗi 25 hóa đơn; hóa đơn đã có trong `ocr_cache/` được bỏ qua"),
    ("code", '''
!{PY} scripts/select_edit_regions.py --findit2-root auto \\
    --splits-dir /kaggle/working/data_synth/splits --ocr-cache /kaggle/working/data_synth/ocr_cache \\
    --ocr-source paddle --out /kaggle/working/data_synth
'''),
    ("code", '''
!head -c 600 /kaggle/working/data_synth/manifest_anytext.jsonl; echo; wc -l /kaggle/working/data_synth/manifest_*.jsonl
'''),
], gpu=True)

# ------------------------------------------------------------------ 02 LaMa (GPU)
NB2 = build(2, [
    ("md", '''
# 02 — Sinh LaMa Removal (Seen AI) trên findit2 (GPU T4)
**Settings:** Accelerator = GPU T4, Internet = On. Add Input: dataset findit2 + output notebook **01_prep_findit2**.
Venv ghim `/tmp/venv_lama` (torch 2.3.0+cu121, simple-lama 0.1.2) — giống env chính của dự án.
Output: `/kaggle/working/data_synth/lama/{train,val,test}/{images,masks,pristine}/` + `qa/`. Chạy lại được (bỏ qua mẫu đã có).
'''),
    ("code", CLONE),
    ("code", '''
# Pinned LaMa venv (~3-5 min first time: downloads torch cu121)
!bash scripts/setup_kaggle_venv.sh lama
PY = "/tmp/venv_lama/bin/python"
os.environ["TORCH_HOME"] = "/tmp/torch"   # big-lama.pt (~200MB) is downloaded here on first use
'''),
    ("code", FIND_PREP.format(gen="lama")),
    ("md", "## Smoke test 10 mẫu → QA → xem contact sheet"),
    ("code", '''
!{PY} scripts/gen_lama_findit2.py --findit2-root auto --manifest {MAN} --out {OUT} --limit 10
!{PY} scripts/qa_synth.py --gen lama --out {OUT} --manifest {MAN} --sheet-n 10
'''),
    ("code", SHOW.format(gen="lama")),
    ("md", "## Chạy đủ (~1.400 mẫu) → QA cuối"),
    ("code", '''
!{PY} scripts/gen_lama_findit2.py --findit2-root auto --manifest {MAN} --out {OUT}
!cp {MAN} {OUT}/
!{PY} scripts/qa_synth.py --gen lama --out {OUT} --manifest {MAN}
'''),
    ("code", SHOW.format(gen="lama")),
], gpu=True)

# ------------------------------------------------------------------ 03 AnyText (GPU)
NB3 = build(3, [
    ("md", '''
# 03 — Sinh AnyText Replacement + Insertion (Unseen Test Set 2) (GPU T4)
**Settings:** Accelerator = GPU T4, Internet = On. Add Input: dataset findit2 + output notebook **01_prep_findit2**.
Venv ghim `/tmp/DIFDL/.venv_anytext` (theo `environment.yaml` của AnyText: torch 2.0.1, modelscope 1.10); trọng số (~5–6GB) tải vào `/tmp/anytext_model`.
Output: `/kaggle/working/data_synth/anytext/test/{images,masks,pristine,raw}/` + `qa/`.
'''),
    ("code", CLONE),
    ("md", "## 1. Dựng env AnyText (~10–15 phút)"),
    ("code", '''
!bash scripts/setup_anytext.sh 2>&1 | tail -n 5
'''),
    ("code", FONT),
    ("code", """
# Weights + ms_wrapper.py (~5-6GB) with a CURRENT modelscope client (1.10 can't download from today's hub)
!bash scripts/download_anytext_model.sh /tmp/anytext_model
"""),
    ("code", FIND_PREP.format(gen="anytext")),
    ("md", '''
## 2. Smoke test 5 mẫu
Kiểm tra bằng mắt: chữ mới phải nằm **đúng trong box đỏ** và đọc được giống `new_text`.
Nếu chữ ở sai chỗ / cả mảnh bị vẽ lại → xóa `{OUT}/anytext` và chạy lại với `--draw-pos-format file`.
'''),
    ("code", '''
!{PY} scripts/gen_anytext_findit2.py --anytext-dir third_party/AnyText \\
    --findit2-root auto --manifest {MAN} --out {OUT} --limit 5
!{PY} scripts/qa_synth.py --gen anytext --out {OUT} --manifest {MAN} --sheet-n 5
'''),
    ("code", SHOW.format(gen="anytext")),
    ("code", '''
import json
for e in [json.loads(l) for l in open(MAN)][:5]:
    print(e["id"], e["op"], repr(e["orig_text"]), "->", repr(e["new_text"]))
'''),
    ("md", "## 3. Chạy đủ (~250 mẫu, 40–60 phút) → QA cuối"),
    ("code", '''
!{PY} scripts/gen_anytext_findit2.py --anytext-dir third_party/AnyText \\
    --findit2-root auto --manifest {MAN} --out {OUT}
!cp {MAN} {OUT}/
!rm -rf {OUT}/anytext/_tmp
!{PY} scripts/qa_synth.py --gen anytext --out {OUT} --manifest {MAN}
'''),
    ("code", SHOW.format(gen="anytext")),
], gpu=True)

if __name__ == "__main__":
    for name, book in [("00_doctor.ipynb", NB0), ("01_prep_findit2.ipynb", NB1), ("02_lama_findit2.ipynb", NB2),
                       ("03_anytext_findit2.ipynb", NB3)]:
        (HERE / name).write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", HERE / name)
