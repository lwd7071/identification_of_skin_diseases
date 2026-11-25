# -*- coding: utf-8 -*-
"""
Bước 2: Trích xuất đặc trưng (Feature Extraction) bằng fp16 trên GPU RTX 3050 (4GB VRAM):
  1. ResNet-50 (ImageNet-1K V2) -> đặc trưng 2048 chiều cho nhánh CNN-ProtoNet.
  2. BiomedCLIP (ViT-B/16 + PubMedBERT) -> đặc trưng ảnh 512 chiều + đặc trưng văn bản
     (tên lớp đơn giản, mô tả triệu chứng lâm sàng, và 18 khái niệm hình thái SKINCON).
  3. Kiểm tra thêm rò rỉ ngữ nghĩa (Cosine > 0.985 giữa train và test cùng lớp) để đánh dấu mask sạch tuyệt đối.
"""
import csv
import gc
import json
import time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from tqdm import tqdm
import open_clip

from config import (
    ALL_CLASSES,
    BIOMEDCLIP_MODEL_ID,
    DATA_DIR,
    FEAT_DIR,
    OUT_DIR,
    SRC_DIR,
)


def iter_batches_threaded(rows, transform, batch_size: int = 64, max_workers: int = 8):
    """Giải mã JPEG song song bằng luồng nhẹ (tránh lỗi tràn pagefile WinError 1455 của multiprocessing trên Windows)."""
    def _load_one(idx):
        img_path = DATA_DIR / rows[idx]["rel_path"]
        with Image.open(img_path) as img:
            return transform(img.convert("RGB")), idx

    n = len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in range(0, n, batch_size):
            batch_indices = range(start, min(start + batch_size, n))
            loaded = list(pool.map(_load_one, batch_indices))
            imgs = torch.stack([ item[0] for item in loaded ], dim=0)
            idxs = [ item[1] for item in loaded ]
            yield imgs, idxs


def load_manifest():
    manifest_path = OUT_DIR / "clean_manifest.csv"
    rows = []
    cls_to_idx = {c: i for i, c in enumerate(ALL_CLASSES)}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = cls_to_idx[r["class"]]
            rows.append(r)
    return rows


@torch.no_grad()
def extract_resnet50(rows, device):
    print("\n[1/2] Đang trích xuất đặc trưng ResNet-50 (ImageNet1K_V2)...")
    t0 = time.time()
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=weights)
    resnet.fc = torch.nn.Identity()  # Xuất vector 2048 chiều trước lớp FC
    resnet = resnet.to(device).eval()

    tfm = weights.transforms()
    n_batches = (len(rows) + 63) // 64

    feats = torch.zeros((len(rows), 2048), dtype=torch.float32)
    use_amp = device.type == "cuda"
    for imgs, idxs in tqdm(iter_batches_threaded(rows, tfm, batch_size=64), total=n_batches, desc="ResNet-50", ncols=90):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            out = resnet(imgs)
        feats[idxs] = out.float().cpu()

    out_file = FEAT_DIR / "resnet50_features.pt"
    torch.save({"features": feats}, out_file)
    print(f"-> Hoàn tất ResNet-50 ({feats.shape}) trong {time.time() - t0:.1f}s | Lưu tại: {out_file}")

    del resnet
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def extract_biomedclip(rows, device):
    print(f"\n[2/2] Đang tải và trích xuất đặc trưng BiomedCLIP ({BIOMEDCLIP_MODEL_ID})...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID)
    model = model.to(device).eval()

    n_batches = (len(rows) + 63) // 64

    img_feats = torch.zeros((len(rows), 512), dtype=torch.float32)
    use_amp = device.type == "cuda"
    for imgs, idxs in tqdm(iter_batches_threaded(rows, preprocess, batch_size=64), total=n_batches, desc="BiomedCLIP Images", ncols=90):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            emb = model.encode_image(imgs)
            emb = F.normalize(emb.float(), dim=-1)
        img_feats[idxs] = emb.cpu()

    # Mã hóa văn bản: (a) Tên bệnh đơn giản, (b) Mô tả triệu chứng lâm sàng, (c) 18 khái niệm hình thái SKINCON
    with open(SRC_DIR / "clinical_concepts.json", "r", encoding="utf-8") as f:
        concept_db = json.load(f)

    simple_text_embs = []
    clinical_text_embs = []
    for cls_name in ALL_CLASSES:
        clean_name = cls_name.replace("_", " ").lower()
        # Prompt đơn giản chỉ có tên bệnh (dùng cho Ablation)
        tok_simple = tokenizer([f"this is a dermatological photograph of {clean_name}"], context_length=256).to(device)
        e_simple = F.normalize(model.encode_text(tok_simple).float(), dim=-1)
        simple_text_embs.append(e_simple.squeeze(0).cpu())

        # Ensemble các câu mô tả triệu chứng lâm sàng của lớp bệnh
        prompts = concept_db["classes"][cls_name]["prompts"]
        tok_clin = tokenizer(prompts, context_length=256).to(device)
        e_clin = F.normalize(model.encode_text(tok_clin).float(), dim=-1)
        e_clin_mean = F.normalize(e_clin.mean(dim=0), dim=-1)
        clinical_text_embs.append(e_clin_mean.cpu())

    simple_text_embs = torch.stack(simple_text_embs, dim=0)      # [22, 512]
    clinical_text_embs = torch.stack(clinical_text_embs, dim=0)  # [22, 512]

    concept_prompts = [c["en"] for c in concept_db["concepts_list"]]
    tok_concepts = tokenizer(concept_prompts, context_length=256).to(device)
    concept_embs = F.normalize(model.encode_text(tok_concepts).float(), dim=-1).cpu()  # [18, 512]

    logit_scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 100.0

    # Kiểm tra thêm rò rỉ ngữ nghĩa gần trùng (cropped/augmented duplicates: Cosine > 0.985 giữa train & test cùng lớp)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    is_train = torch.tensor([r["split"] == "train" for r in rows], dtype=torch.bool)
    keep_mask = torch.ones(len(rows), dtype=torch.bool)
    semantic_near_dups = 0
    for c_idx in range(len(ALL_CLASSES)):
        tr_idx = torch.where(is_train & (labels == c_idx))[0]
        te_idx = torch.where((~is_train) & (labels == c_idx))[0]
        if len(tr_idx) > 0 and len(te_idx) > 0:
            sim_mat = img_feats[tr_idx] @ img_feats[te_idx].T  # [N_tr, N_te]
            max_sim, _ = sim_mat.max(dim=1)
            leak_local = max_sim > 0.985
            n_leak = int(leak_local.sum().item())
            if n_leak > 0:
                keep_mask[tr_idx[leak_local]] = False
                semantic_near_dups += n_leak

    out_file = FEAT_DIR / "biomedclip_features.pt"
    torch.save(
        {
            "image_features": img_feats,
            "simple_text_features": simple_text_embs,
            "clinical_text_features": clinical_text_embs,
            "concept_features": concept_embs,
            "concept_meta": concept_db["concepts_list"],
            "logit_scale": logit_scale,
            "labels": labels,
            "splits": [r["split"] for r in rows],
            "roles": [r["role"] for r in rows],
            "rel_paths": [r["rel_path"] for r in rows],
            "clean_semantic_mask": keep_mask,
            "semantic_near_dups_flagged": semantic_near_dups,
        },
        out_file,
    )
    print(f"-> Phát hiện & loại thêm {semantic_near_dups} ảnh train quá giống test ở mức ngữ nghĩa (Cosine > 0.985).")
    print(f"-> Hoàn tất BiomedCLIP trong {time.time() - t0:.1f}s | Lưu tại: {out_file}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG TRÊN THIẾT BỊ: {device} ===")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    rows = load_manifest()
    print(f"Số mẫu từ clean_manifest.csv: {len(rows)}")

    extract_resnet50(rows, device)
    extract_biomedclip(rows, device)


if __name__ == "__main__":
    main()
