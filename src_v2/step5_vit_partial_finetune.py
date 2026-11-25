# -*- coding: utf-8 -*-
"""
BƯỚC 5 V2 (Để cuối cùng - Tách bảng riêng biệt):
  - Tinh chỉnh nhẹ trực tiếp trên ảnh gốc (Partial Unfreezing 2 khối Transformer cuối + Projection Head)
    của nhánh thị giác BiomedCLIP (ViT-B/16) với độ chính xác hỗn hợp fp16 trên GPU 4GB VRAM.
  - Bọc cơ chế an toàn bộ nhớ: batch_size=16, grad_accum=4; nếu thiếu VRAM tự động hạ xuống mở khóa 1 khối cuối & batch_size=8.
  - Chống học vẹt vào lớp Base bằng điều chuẩn L2-SP (phạt độ lệch khỏi trọng số gốc BiomedCLIP: ||theta - theta_0||^2)
    + dừng sớm trên 4 lớp Meta-Val.
  - Xuất kết quả ra bảng ĐỘC LẬP: outputs_v2/tables/step5_vit_finetune_separate.csv (5 random seeds).
"""
import csv
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from step1_baselines_5seeds import evaluate_visual_few_shot_5seeds, evaluate_zero_shot_5seeds, flatten_metric_row
from step2_multistage_and_multicrop import evaluate_samp_variant_5seeds, load_manifest_rows, tune_samp_hyperparams_on_meta_val
from v2_config import (
    ALL_CLASSES,
    BASE_CLASSES,
    BIOMEDCLIP_MODEL_ID,
    CKPT_V2_DIR,
    DATA_DIR,
    FEAT_V2_DIR,
    META_TRAIN_CLASSES,
    META_VAL_CLASSES,
    N_EPISODES_PER_SEED,
    NOVEL_CLASSES,
    PreventWindowsSleep,
    SEEDS_V2,
    SHOT_LIST,
    TABLES_V2_DIR,
    V1_FEAT_DIR,
    record_or_verify_protected_files,
)


def configure_partial_unfreeze(model, n_last_blocks: int = 2):
    """Đóng băng toàn bộ BiomedCLIP, chỉ mở khóa n_last_blocks cuối của ViT + chuẩn hóa/chiếu đầu ra."""
    for p in model.parameters():
        p.requires_grad = False

    vis = model.visual
    blocks = None
    if hasattr(vis, "trunk") and hasattr(vis.trunk, "blocks"):
        blocks = vis.trunk.blocks
        if hasattr(vis.trunk, "norm"):
            for p in vis.trunk.norm.parameters():
                p.requires_grad = True
    elif hasattr(vis, "transformer") and hasattr(vis.transformer, "resblocks"):
        blocks = vis.transformer.resblocks

    assert blocks is not None, "Không tìm thấy danh sách Transformer blocks trong model.visual"
    total_b = len(blocks)
    for b_idx in range(max(0, total_b - n_last_blocks), total_b):
        for p in blocks[b_idx].parameters():
            p.requires_grad = True

    if hasattr(vis, "head"):
        for p in vis.head.parameters():
            p.requires_grad = True
    if hasattr(vis, "proj") and isinstance(vis.proj, torch.nn.Parameter):
        vis.proj.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable_params)
    n_all = sum(p.numel() for p in model.parameters())
    return trainable_params, n_train, n_all


def load_batch_images(rows, indices, tfm, pool):
    def _ld(idx):
        with Image.open(DATA_DIR / rows[idx]["rel_path"]) as im:
            return tfm(im.convert("RGB"))
    imgs = list(pool.map(_ld, indices))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def extract_all_images_with_model(model, rows, eval_tfm, device, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    n = len(rows)
    out_feats = torch.zeros((n, 512), dtype=torch.float32)
    use_amp = device.type == "cuda"
    with ThreadPoolExecutor(max_workers=8) as pool:
        for st in range(0, n, batch_size):
            b_idx = list(range(st, min(st + batch_size, n)))
            imgs = load_batch_images(rows, b_idx, eval_tfm, pool).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                emb = F.normalize(model.encode_image(imgs).float(), dim=-1)
            out_feats[b_idx] = emb.cpu()
    return out_feats


def run_partial_finetune(rows, train_mask, test_mask, labels, stage_embs_dev, cls_to_id, device, n_blocks=2, batch_size=16, grad_accum=4, total_steps=240):
    """Huấn luyện tinh chỉnh nhẹ với điều chuẩn L2-SP trên 12 lớp Meta-Train, chọn checkpoint trên 4 lớp Meta-Val."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    model, _, eval_preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
    model = model.to(device)
    norm_op = next(t for t in eval_preprocess.transforms if isinstance(t, transforms.Normalize))

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_op.mean, std=norm_op.std),
    ])

    trainable_params, n_train_p, n_total_p = configure_partial_unfreeze(model, n_last_blocks=n_blocks)
    print(f"  -> Mở khóa {n_blocks} block cuối ViT: {n_train_p:,} / {n_total_p:,} tham số ({n_train_p / n_total_p * 100:.2f}%)", flush=True)

    # Lưu bản sao trọng số gốc để tính điều chuẩn L2-SP (giữ không gian chung với Text Encoder PubMedBERT)
    theta_0 = [p.detach().clone() for p in trainable_params]

    optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-5, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    meta_tr_ids = [cls_to_id[c] for c in META_TRAIN_CLASSES]
    meta_val_ids = [cls_to_id[c] for c in META_VAL_CLASSES]
    local_map = {cid: i for i, cid in enumerate(meta_tr_ids)}

    tr_indices = np.where(train_mask & np.isin(labels, meta_tr_ids))[0]
    val_tr_idx = {c: np.where(train_mask & (labels == c))[0] for c in meta_val_ids}
    val_te_idx = {c: np.where(test_mask & (labels == c))[0] for c in meta_val_ids}
    val_all_idx = np.concatenate([val_tr_idx[c] for c in meta_val_ids] + [val_te_idx[c] for c in meta_val_ids])

    # Neo văn bản của 12 lớp Meta-Train (trung bình 4 giai đoạn, cố định)
    meta_tr_txt_anchors = F.normalize(stage_embs_dev[meta_tr_ids].mean(dim=1), dim=-1)  # [12, 512]
    val_txt_anchors = F.normalize(stage_embs_dev[meta_val_ids].mean(dim=1), dim=-1)    # [4, 512]

    def quick_meta_val_score():
        model.eval()
        sub_rows = [rows[i] for i in val_all_idx]
        z_sub = extract_all_images_with_model(model, sub_rows, eval_preprocess, device, batch_size=64).to(device)
        idx_map = {orig_i: k for k, orig_i in enumerate(val_all_idx)}
        q_pos = [idx_map[i] for c in meta_val_ids for i in val_te_idx[c]]
        q_y = np.concatenate([np.full(len(val_te_idx[c]), lc, dtype=np.int64) for lc, c in enumerate(meta_val_ids)])
        qz = z_sub[q_pos]

        rng_v = np.random.default_rng(777)
        baccs = []
        for _ in range(100):
            protos = []
            for lc, c in enumerate(meta_val_ids):
                ch = rng_v.choice(val_tr_idx[c], size=5, replace=False)
                vp = F.normalize(z_sub[[idx_map[x] for x in ch]].mean(dim=0), dim=-1)
                hp = F.normalize(0.5 * vp + 0.5 * val_txt_anchors[lc], dim=-1)
                protos.append(hp)
            pr = (25.0 * (qz @ torch.stack(protos, dim=0).T)).argmax(dim=-1).cpu().numpy()
            baccs.append(balanced_accuracy_score(q_y, pr) * 100.0)
        return float(np.mean(baccs))

    best_val = quick_meta_val_score()
    best_trainable_state = [p.detach().cpu().clone() for p in trainable_params]
    print(f"  -> [Step 0] Meta-Val 5-shot BAcc trước fine-tune: {best_val:.2f}%", flush=True)

    use_amp = device.type == "cuda"
    optimizer.zero_grad()

    with ThreadPoolExecutor(max_workers=6) as pool:
        for step in range(1, total_steps + 1):
            model.train()
            loss_accum = 0.0
            for _ in range(grad_accum):
                batch_sel = rng.choice(tr_indices, size=batch_size, replace=False)
                b_imgs = load_batch_images(rows, batch_sel, train_tfm, pool).to(device, non_blocking=True)
                b_y = torch.tensor([local_map[labels[i]] for i in batch_sel], dtype=torch.long, device=device)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    img_emb = F.normalize(model.encode_image(b_imgs).float(), dim=-1)
                    logits_ce = 25.0 * (img_emb @ meta_tr_txt_anchors.T)
                    ce_loss = F.cross_entropy(logits_ce, b_y, label_smoothing=0.05)

                    l2_sp = sum(((p.float() - p0.float()) ** 2).sum() for p, p0 in zip(trainable_params, theta_0))
                    loss = (ce_loss + 0.02 * l2_sp) / grad_accum

                scaler.scale(loss).backward()
                loss_accum += float(loss.item())

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if step % 60 == 0 or step == total_steps:
                v_sc = quick_meta_val_score()
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_amp else 0.0
                print(f"  -> [Step {step:3d}/{total_steps}] Loss={loss_accum:.4f} | Meta-Val 5-shot BAcc={v_sc:.2f}% | Peak VRAM={peak_mb:.0f} MB", flush=True)
                if v_sc > best_val:
                    best_val = v_sc
                    best_trainable_state = [p.detach().cpu().clone() for p in trainable_params]

    for p, saved_p in zip(trainable_params, best_trainable_state):
        p.data.copy_(saved_p.to(device))
    model.eval()
    return model, eval_preprocess, best_val


def main():
    t0 = time.time()
    with PreventWindowsSleep():
        record_or_verify_protected_files("verify")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"=== BƯỚC 5 V2: FINE-TUNE NHẸ BIOMEDCLIP ViT TRÊN 4GB VRAM - BẢNG ĐỘC LẬP ({device}) ===", flush=True)

        rows = load_manifest_rows()
        bclip_v1 = torch.load(V1_FEAT_DIR / "biomedclip_features.pt", map_location="cpu", weights_only=False)
        stage_pkg = torch.load(FEAT_V2_DIR / "stage_text_embs_v2.pt", map_location="cpu", weights_only=False)

        labels = bclip_v1["labels"].numpy()
        splits = np.array(bclip_v1["splits"])
        clean_sem_mask = bclip_v1["clean_semantic_mask"].numpy()
        train_mask = (splits == "train") & clean_sem_mask
        test_mask = splits == "test"

        cls_to_id = {c: i for i, c in enumerate(ALL_CLASSES)}
        base_cls_ids = [cls_to_id[c] for c in BASE_CLASSES]
        meta_tr_ids = [cls_to_id[c] for c in META_TRAIN_CLASSES]
        novel_cls_ids = [cls_to_id[c] for c in NOVEL_CLASSES]
        stage_embs_dev = stage_pkg["stage_text_features"].float().to(device)

        used_cfg = {"n_blocks": 2, "batch_size": 16, "grad_accum": 4, "steps": 240}
        try:
            ft_model, eval_tfm, best_val_score = run_partial_finetune(
                rows, train_mask, test_mask, labels, stage_embs_dev, cls_to_id, device,
                n_blocks=2, batch_size=16, grad_accum=4, total_steps=240,
            )
        except torch.cuda.OutOfMemoryError:
            print("[CẢNH BÁO VRAM] Đụng trần 4GB khi mở 2 block -> Tự động dọn VRAM và hạ xuống 1 block cuối (batch_size=8)...", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            used_cfg = {"n_blocks": 1, "batch_size": 8, "grad_accum": 8, "steps": 240}
            ft_model, eval_tfm, best_val_score = run_partial_finetune(
                rows, train_mask, test_mask, labels, stage_embs_dev, cls_to_id, device,
                n_blocks=1, batch_size=8, grad_accum=8, total_steps=240,
            )

        print("\n-> Đang trích xuất đặc trưng toàn bộ 14.797 ảnh từ BiomedCLIP đã Fine-tune (1-crop)...", flush=True)
        ft_feats = extract_all_images_with_model(ft_model, rows, eval_tfm, device, batch_size=64)
        torch.save(
            {"image_features": ft_feats, "meta_val_best_5shot": best_val_score, "finetune_config": used_cfg},
            FEAT_V2_DIR / "biomedclip_finetuned_1crop.pt",
        )

        del ft_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ft_dev = F.normalize(ft_feats.to(device), dim=-1)
        hp_ft = tune_samp_hyperparams_on_meta_val(
            ft_dev, stage_embs_dev, labels, train_mask, test_mask, cls_to_id, temp=25.0
        )

        ft_rows = []
        zs_ft = evaluate_samp_variant_5seeds(
            ft_dev, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
            k_shot=0, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED, cfg=hp_ft[0],
        )
        ft_rows.append(flatten_metric_row("Fine-Tuned ViT (Bảng riêng)", "FT-BiomedCLIP + 4 giai đoạn (0-shot)", "1-crop-FT", 0, zs_ft))

        for sh in SHOT_LIST:
            vis_ft = evaluate_visual_few_shot_5seeds(
                {sd: ft_dev for sd in SEEDS_V2}, {sd: 25.0 for sd in SEEDS_V2},
                labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            ft_rows.append(flatten_metric_row("Fine-Tuned ViT (Bảng riêng)", "FT-BiomedCLIP + ProtoNet (Visual Only)", "1-crop-FT", sh, vis_ft))

            samp_ft = evaluate_samp_variant_5seeds(
                ft_dev, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED, cfg={"use_text": True, **hp_ft[sh]},
            )
            ft_rows.append(flatten_metric_row("Fine-Tuned ViT (Bảng riêng)", "FT-BiomedCLIP + SAMP Đầy đủ", "1-crop-FT", sh, samp_ft))

        csv_path = TABLES_V2_DIR / "step5_vit_finetune_separate.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(ft_rows[0].keys()))
            w.writeheader()
            w.writerows(ft_rows)

        with open(TABLES_V2_DIR / "step5_vit_finetune_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "finetune_config": used_cfg,
                    "meta_val_best_5shot_bacc": round(best_val_score, 3),
                    "selected_hparams_from_meta_val": hp_ft,
                    "results": ft_rows,
                    "elapsed_seconds": round(time.time() - t0, 1),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        verify_after = record_or_verify_protected_files("verify")
        print("\n=== KẾT QUẢ BƯỚC 5 (FINE-TUNE NHẸ ViT - BẢNG ĐỘC LẬP, 5 SEEDS) ===", flush=True)
        for r in ft_rows:
            print(
                f"  {r['method']:45s} | {r['shot']:2d}-shot | BAcc: {r['bacc_mean']:6.2f} ± {r['bacc_seed_std']:4.2f}% "
                f"| F1: {r['mf1_mean']:6.2f}% | AUROC: {r['auroc_mean']:6.2f}%",
                flush=True,
            )
        print(f"[OK] Lưu tại: {csv_path} | SHA-256: {verify_after['status']} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
