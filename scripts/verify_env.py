"""Verify DIFDL synchronized Python 3.10 environment across local GPUs and Lab Servers."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def check_version(mod_name, attr="__version__"):
    mod = __import__(mod_name)
    ver = getattr(mod, attr, "installed")
    print(f"  [OK] {mod_name:24s} {ver}")
    return mod


def main():
    print("=" * 70)
    print("DIFDL ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    assert sys.version_info[:2] == (3, 10), f"Expected Python 3.10, got {sys.version_info[:2]}"

    # 1. Core libraries
    torch = check_version("torch")
    check_version("torchvision")
    np = check_version("numpy")
    assert int(np.__version__.split(".")[0]) < 2, f"Expected numpy<2, got {np.__version__}"
    check_version("albumentations")
    check_version("efficientnet_pytorch")
    check_version("einops")
    check_version("cv2")
    check_version("scipy")
    PIL = check_version("PIL")
    check_version("lmdb")
    check_version("tqdm")
    check_version("tensorboard")
    check_version("sklearn")
    check_version("matplotlib")

    # 2. CUDA / cuDNN / GPU check
    cuda_ok = torch.cuda.is_available()
    cudnn_ver = torch.backends.cudnn.version() if cuda_ok else None
    print(f"  [OK] CUDA available           {cuda_ok} (torch CUDA: {torch.version.cuda}, cuDNN: {cudnn_ver})")
    if cuda_ok:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"       GPU {idx}: {props.name} ({vram_gb:.2f} GB VRAM)")

    # 3. jpeg2dct round-trip functional test
    from jpeg2dct.numpy import load as jpeg2dct_load
    from PIL import Image

    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(arr).save(tmp_path, format="JPEG", quality=85)
        dct_y, dct_cb, dct_cr = jpeg2dct_load(tmp_path, normalized=False)
        assert dct_y.shape == (8, 8, 64), f"Unexpected DCT shape: {dct_y.shape}"
        print(f"  [OK] jpeg2dct functional test dct_y={dct_y.shape}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 4. PaddleOCR (TextDetection for PP-OCRv5_server_det)
    try:
        import paddle
        from paddleocr import TextDetection  # noqa: F401
        print(
            f"  [OK] paddleocr.TextDetection  paddle={paddle.__version__} "
            f"(compiled_with_cuda={paddle.is_compiled_with_cuda()})"
        )
    except Exception as exc:
        print(f"  [WARN] paddleocr import issue: {exc}")

    # 5. simple-lama-inpainting
    try:
        from simple_lama_inpainting import SimpleLama  # noqa: F401
        print("  [OK] simple_lama_inpainting   SimpleLama importable")
    except Exception as exc:
        print(f"  [WARN] simple_lama_inpainting import issue: {exc}")

    # 6. ADCD-Net model import smoke test
    try:
        from model.model import ADCDNet  # noqa: F401
        print("  [OK] src/model/model.py       ADCDNet importable")
    except Exception as exc:
        print(f"  [FAIL] src/model/model.py import failed: {exc}")
        sys.exit(1)

    print("=" * 70)
    print("ALL CORE CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
