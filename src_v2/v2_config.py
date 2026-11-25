# -*- coding: utf-8 -*-
"""
Cấu hình trung tâm cho Phiên bản V2 (Chạy xuyên đêm - Cách 1):
  - Cô lập hoàn toàn đầu ra sang outputs_v2/ (tuyệt đối không ghi đè outputs/ hay docs/ của V1).
  - Ghi nhận & kiểm tra mã băm SHA-256 của toàn bộ file .docx và kết quả V1.
  - Cung cấp context manager chống Windows tự Sleep trong lúc đang chạy GPU.
"""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"

# Ép toàn bộ cache về thư mục dự án trước khi import bất kỳ thư viện ML nào
os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hf" / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["TEMP"] = str(CACHE_DIR / "tmp")
os.environ["TMP"] = str(CACHE_DIR / "tmp")

DATA_DIR = Path(os.environ.get("SKIN_DATA_DIR", str(ROOT_DIR.parent / "Data" / "SkinDisease")))
EXT_DATA_DIR = ROOT_DIR / "data_external"
PAD_UFES_DIR = EXT_DATA_DIR / "pad_ufes_20"

# Thư mục V1 (CHỈ ĐỌC - KHÔNG GHI ĐÈ)
V1_SRC_DIR = ROOT_DIR / "src"
V1_OUT_DIR = ROOT_DIR / "outputs"
V1_FEAT_DIR = V1_OUT_DIR / "features"

# Thư mục V2 (ĐẦU RA MỚI TÁCH BIỆT HOÀN TOÀN)
SRC_V2_DIR = ROOT_DIR / "src_v2"
OUT_V2_DIR = ROOT_DIR / "outputs_v2"
TABLES_V2_DIR = OUT_V2_DIR / "tables"
FIG_V2_DIR = OUT_V2_DIR / "figures"
CKPT_V2_DIR = OUT_V2_DIR / "checkpoints"
FEAT_V2_DIR = OUT_V2_DIR / "features"
WEB_DEMO_DIR = ROOT_DIR / "web_demo"

for _d in (
    CACHE_DIR / "matplotlib",
    CACHE_DIR / "tmp",
    EXT_DATA_DIR,
    PAD_UFES_DIR,
    SRC_V2_DIR,
    OUT_V2_DIR,
    TABLES_V2_DIR,
    FIG_V2_DIR,
    CKPT_V2_DIR,
    FEAT_V2_DIR,
    WEB_DEMO_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Danh sách các tệp V1 & .docx bắt buộc giữ nguyên 100% (kiểm tra bằng SHA-256)
PROTECTED_FILES = [
    ROOT_DIR / "Tom_Tat_1_Trang.docx",
    ROOT_DIR / "docs" / "Tom_Tat_1_Trang.docx",
    Path(r"D:\NCKH\BieuMau\BM01_Nhap_ThuyetMinh_NCKH_2027.docx"),
    Path(r"D:\NCKH\BieuMau\BM01. Thuyết minh đề tài NCKH SV năm 2027.docx"),
    V1_OUT_DIR / "phase1_full_results.json",
    V1_OUT_DIR / "benchmark_novel6way.csv",
    V1_OUT_DIR / "ablation_novel6way.csv",
    V1_OUT_DIR / "benchmark_generalized_22way.csv",
]

SHA256_LOCK_FILE = OUT_V2_DIR / "protected_files_sha256_lock.json"

# Phân chia lớp giống hệt V1 để đối chiếu nhất quán 100%
BASE_CLASSES = [
    "Unknown_Normal",
    "Benign_tumors",
    "Eczema",
    "Tinea",
    "Psoriasis",
    "Actinic_Keratosis",
    "Vitiligo",
    "SkinCancer",
    "Acne",
    "Warts",
    "Lichen",
    "DrugEruption",
    "Vascular_Tumors",
    "Infestations_Bites",
    "Bullous",
    "Vasculitis",
]

META_VAL_CLASSES = [
    "Warts",
    "DrugEruption",
    "Infestations_Bites",
    "Vasculitis",
]
META_TRAIN_CLASSES = [c for c in BASE_CLASSES if c not in META_VAL_CLASSES]

NOVEL_CLASSES = [
    "Seborrh_Keratoses",
    "Moles",
    "Sun_Sunlight_Damage",
    "Lupus",
    "Rosacea",
    "Candidiasis",
]

ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES

# Nâng cấp lên 5 random seeds độc lập (5 x 200 = 1.000 episodes/cấu hình)
SEEDS_V2 = [42, 123, 2027, 7, 99]
SHOT_LIST = [1, 5, 10]
N_EPISODES_PER_SEED = 200
BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def record_or_verify_protected_files(mode: str = "record") -> dict:
    """
    mode='record': Ghi mã băm SHA-256 ban đầu trước khi chạy V2 (nếu chưa có file lock).
    mode='verify': Đối chiếu lại toàn bộ các file được bảo vệ, báo lỗi ngay nếu bất kỳ byte nào bị thay đổi.
    """
    current_hashes = {}
    for p in PROTECTED_FILES:
        if p.exists():
            current_hashes[str(p)] = compute_sha256(p)

    if mode == "record" and not SHA256_LOCK_FILE.exists():
        SHA256_LOCK_FILE.write_text(json.dumps(current_hashes, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "RECORDED", "files": current_hashes}

    saved_hashes = json.loads(SHA256_LOCK_FILE.read_text(encoding="utf-8"))
    mismatches = []
    for k, old_h in saved_hashes.items():
        new_h = current_hashes.get(k)
        if new_h != old_h:
            mismatches.append({"file": k, "expected": old_h, "actual": new_h})

    if mismatches:
        raise RuntimeError(f"PHÁT HIỆN FILE ĐƯỢC BẢO VỆ BỊ THAY ĐỔI: {mismatches}")
    return {"status": "VERIFIED_100_PERCENT_INTACT", "checked_count": len(saved_hashes), "files": saved_hashes}


class PreventWindowsSleep:
    """Ngăn Windows tự Sleep khi đang chạy tiến trình huấn luyện đêm, tự nhả lại khi thoát."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __enter__(self):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        except Exception:
            pass


def summarize_5seed_episodes(seed_to_ep_values: dict) -> dict:
    """
    Nhận dict {seed: [ep1, ep2, ...]} (5 seeds x 200 episodes = 1000 giá trị):
      - Tính trung bình của từng seed -> lấy Mean ± SD trên 5 trung bình seed (chuẩn báo cáo đa seed).
      - Đồng thời tính 95% CI trên toàn bộ 1.000 episodes để đối chiếu trực tiếp với V1.
    """
    seed_means = [float(np.nanmean(vals)) for vals in seed_to_ep_values.values()]
    all_vals = np.concatenate([np.asarray(v, dtype=np.float64) for v in seed_to_ep_values.values()])
    valid_vals = all_vals[~np.isnan(all_vals)]

    mean_5s = float(np.mean(seed_means))
    std_5s = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    ci95_ep = float(1.96 * np.std(valid_vals, ddof=1) / math.sqrt(len(valid_vals))) if len(valid_vals) > 1 else 0.0
    return {
        "mean": mean_5s,
        "seed_std": std_5s,
        "ep_ci95": ci95_ep,
        "per_seed_means": {str(k): round(float(np.nanmean(v)), 3) for k, v in seed_to_ep_values.items()},
    }
