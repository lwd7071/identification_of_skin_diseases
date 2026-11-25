# -*- coding: utf-8 -*-
"""Cấu hình trung tâm cho Giai đoạn 1 (đặt toàn bộ cache & đầu ra vào New folder)."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"

# Ép toàn bộ cache về thư mục dự án trước khi import bất kỳ thư viện ML nào
os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hf" / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")

DATA_DIR = Path(os.environ.get("SKIN_DATA_DIR", str(ROOT_DIR.parent / "Data" / "SkinDisease")))
SRC_DIR = ROOT_DIR / "src"
OUT_DIR = ROOT_DIR / "outputs"
FEAT_DIR = OUT_DIR / "features"
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
DOCS_DIR = ROOT_DIR / "docs"

for _d in (OUT_DIR, FEAT_DIR, CKPT_DIR, FIG_DIR, DOCS_DIR, CACHE_DIR / "matplotlib"):
    _d.mkdir(parents=True, exist_ok=True)

# 16 lớp phổ biến (Base) sắp xếp giảm dần theo số ảnh gốc
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

# Tách 4 lớp Base làm tập Meta-Validation để chọn siêu tham số (không đụng tập Novel)
META_VAL_CLASSES = [
    "Warts",
    "DrugEruption",
    "Infestations_Bites",
    "Vasculitis",
]
META_TRAIN_CLASSES = [c for c in BASE_CLASSES if c not in META_VAL_CLASSES]

# 6 lớp ít mẫu nhất làm tập Novel (bệnh hiếm cần học Few-Shot)
NOVEL_CLASSES = [
    "Seborrh_Keratoses",
    "Moles",
    "Sun_Sunlight_Damage",
    "Lupus",
    "Rosacea",
    "Candidiasis",
]

ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES

SEEDS = [42, 123, 2027]
SHOT_LIST = [1, 5, 10]
N_EPISODES_EVAL = 600
N_QUERY = 15

BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
