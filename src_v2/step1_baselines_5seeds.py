# -*- coding: utf-8 -*-
"""
BƯỚC 1 (Ưu tiên số 1 - Chạy đầu tiên):
  - Khóa mã băm SHA-256 của toàn bộ file V1 và .docx.
  - Chạy đầy đủ Baselines 3 hướng trên đặc trưng 1-crop sạch với 5 RANDOM SEEDS ĐỘC LẬP (42, 123, 2027, 7, 99):
      (1) Hướng 1 - CNN Metric Learning:
            + ResNet-50 + ProtoNet (huấn luyện độc lập 5 mô hình theo 5 seeds trên Meta-Train, chọn trên Meta-Val)
            + ResNet-50 + SimpleShot (dịch tâm Meta-Train + L2 Nearest Centroid, Wang et al. 2019)
      (2) Hướng 2 - VLM Zero-Shot (0-shot):
            + BiomedCLIP Zero-Shot (chỉ truyền tên bệnh đơn giản)
            + BiomedCLIP Zero-Shot (mô tả triệu chứng gộp V1)
      (3) Hướng 3 - VLM Few-Shot thuần ảnh:
            + BiomedCLIP + ProtoNet (Visual Only)
            + BiomedCLIP + SimpleShot (Centered Visual Only - dịch tâm Meta-Train)
  - Báo cáo đầy đủ: Mean ± SD (giữa 5 trung bình seed) VÀ 95% CI (trên 1.000 episodes).
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from v2_config import (
    ALL_CLASSES,
    BASE_CLASSES,
    CKPT_V2_DIR,
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
    summarize_5seed_episodes,
)


class ResNetProtoHeadV2(nn.Module):
    """Mạng chiếu Metric chuẩn hóa cho Baseline ResNet-50 (2048-d -> 256-d)."""

    def __init__(self, in_dim: int = 2048, out_dim: int = 256, dropout: float = 0.35):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.drop(self.norm_in(x))), dim=-1)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_np(probs: np.ndarray, y_true: np.ndarray):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    mf1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
    try:
        auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro") * 100.0
    except ValueError:
        auroc = float("nan")
    return acc, bacc, mf1, auroc


def prebuild_val_episodes(tr_pools: dict, te_pools: dict, target_classes: list, k_shot: int, n_episodes: int = 150, seed: int = 777):
    rng = np.random.default_rng(seed)
    all_te_idx, all_te_y = [], []
    for local_y, c_id in enumerate(target_classes):
        idxs = te_pools[c_id]
        all_te_idx.append(idxs)
        all_te_y.extend([local_y] * len(idxs))
    q_idx = np.concatenate(all_te_idx)
    q_y = np.array(all_te_y, dtype=np.int64)

    episodes = []
    for _ in range(n_episodes):
        sup_idx, sup_y = [], []
        for local_y, c_id in enumerate(target_classes):
            chosen_s = rng.choice(tr_pools[c_id], size=k_shot, replace=False)
            sup_idx.append(chosen_s)
            sup_y.extend([local_y] * len(chosen_s))
        episodes.append((np.concatenate(sup_idx), np.array(sup_y, dtype=np.int64)))
    return episodes, q_idx, q_y


def train_single_resnet_protonet(res_feats_dev, labels, train_mask, test_mask, cls_to_id, device, seed: int):
    """Huấn luyện 1 mô hình ResNet-50 + ProtoNet độc lập theo seed chỉ trên Meta-Train, dừng sớm trên Meta-Val."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = ResNetProtoHeadV2(in_dim=2048, out_dim=256, dropout=0.35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)

    meta_tr_ids = [cls_to_id[c] for c in META_TRAIN_CLASSES]
    meta_val_ids = [cls_to_id[c] for c in META_VAL_CLASSES]

    tr_pools = {c: np.where(train_mask & (labels == c))[0] for c in meta_tr_ids}
    val_tr_pools = {c: np.where(train_mask & (labels == c))[0] for c in meta_val_ids}
    val_te_pools = {c: np.where(test_mask & (labels == c))[0] for c in meta_val_ids}
    fixed_val_eps, vq_idx, vq_y = prebuild_val_episodes(val_tr_pools, val_te_pools, meta_val_ids, k_shot=5, n_episodes=150, seed=700 + seed)

    def eval_val():
        model.eval()
        with torch.no_grad():
            z_all = model(res_feats_dev)
            vq_z = z_all[vq_idx]
            sc = float(model.scale.clamp(1.0, 40.0).item())
            baccs = []
            for vs_idx, vs_y in fixed_val_eps:
                vs_z = z_all[vs_idx]
                vp = torch.stack([F.normalize(vs_z[vs_y == c].mean(dim=0), dim=-1) for c in range(len(meta_val_ids))], dim=0)
                preds = (sc * (vq_z @ vp.T)).argmax(dim=-1).cpu().numpy()
                baccs.append(balanced_accuracy_score(vq_y, preds) * 100.0)
        return float(np.mean(baccs))

    best_val = eval_val()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    n_way = 8
    for ep in range(1, 601):
        model.train()
        sampled_cls = rng.choice(meta_tr_ids, size=n_way, replace=False).tolist()
        sup_idx, sup_y, qry_idx, qry_y = [], [], [], []
        for ly, cid in enumerate(sampled_cls):
            chosen = rng.choice(tr_pools[cid], size=20, replace=False)
            sup_idx.append(chosen[:5])
            sup_y.extend([ly] * 5)
            qry_idx.append(chosen[5:])
            qry_y.extend([ly] * 15)
        s_idx = np.concatenate(sup_idx)
        q_idx = np.concatenate(qry_idx)
        s_y_t = torch.tensor(sup_y, dtype=torch.long, device=device)
        q_y_t = torch.tensor(qry_y, dtype=torch.long, device=device)

        s_z = model(res_feats_dev[s_idx])
        q_z = model(res_feats_dev[q_idx])
        protos = torch.stack([F.normalize(s_z[s_y_t == c].mean(dim=0), dim=-1) for c in range(n_way)], dim=0)
        logits = model.scale.clamp(1.0, 40.0) * (q_z @ protos.T)
        loss = F.cross_entropy(logits, q_y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 75 == 0:
            v_score = eval_val()
            if v_score > best_val:
                best_val = v_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


@torch.no_grad()
def evaluate_zero_shot_5seeds(img_feats_dev, text_feats_dev, labels, test_mask, target_cls_ids, temp=25.0):
    te_idx_list, y_true_list = [], []
    for lc, cid in enumerate(target_cls_ids):
        idxs = np.where(test_mask & (labels == cid))[0]
        te_idx_list.append(idxs)
        y_true_list.extend([lc] * len(idxs))
    te_idx = np.concatenate(te_idx_list)
    y_true = np.array(y_true_list, dtype=np.int64)

    q_z = F.normalize(img_feats_dev[te_idx], dim=-1)
    t_p = F.normalize(text_feats_dev[target_cls_ids], dim=-1)
    probs = F.softmax(temp * (q_z @ t_p.T), dim=-1).cpu().numpy()
    acc, bacc, mf1, auroc = compute_metrics_np(probs, y_true)
    return {
        "acc": {"mean": acc, "seed_std": 0.0, "ep_ci95": 0.0},
        "bacc": {"mean": bacc, "seed_std": 0.0, "ep_ci95": 0.0},
        "mf1": {"mean": mf1, "seed_std": 0.0, "ep_ci95": 0.0},
        "auroc": {"mean": auroc, "seed_std": 0.0, "ep_ci95": 0.0},
    }


@torch.no_grad()
def evaluate_visual_few_shot_5seeds(
    seed_to_z_all: dict,
    seed_to_temp: dict,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    target_cls_ids: list,
    k_shot: int,
    seeds: list,
    n_episodes_per_seed: int,
):
    """
    Hỗ trợ cả mô hình cố định (truyền cùng 1 tensor z_all cho mọi seed) LẪN mô hình train theo từng seed
    (mỗi seed có không gian đặc trưng z_all riêng như ResNet-50 + ProtoNet).
    """
    tr_pools = {c: np.where(train_mask & (labels == c))[0] for c in target_cls_ids}
    te_pools = {c: np.where(test_mask & (labels == c))[0] for c in target_cls_ids}

    all_te_idx, all_te_y = [], []
    for lc, cid in enumerate(target_cls_ids):
        idxs = te_pools[cid]
        all_te_idx.append(idxs)
        all_te_y.extend([lc] * len(idxs))
    all_te_idx = np.concatenate(all_te_idx)
    all_te_y = np.array(all_te_y, dtype=np.int64)
    te_set = set(all_te_idx.tolist())

    acc_by_s, bacc_by_s, mf1_by_s, auc_by_s = {}, {}, {}, {}
    for sd in seeds:
        z_all = seed_to_z_all[sd]
        temp = seed_to_temp[sd]
        q_z = F.normalize(z_all[all_te_idx], dim=-1)
        rng = np.random.default_rng(sd + k_shot * 17)

        accs, baccs, mf1s, aucs = [], [], [], []
        for _ in range(n_episodes_per_seed):
            protos = []
            for cid in target_cls_ids:
                chosen_s = rng.choice(tr_pools[cid], size=k_shot, replace=False)
                # Kiểm tra tự động 0% giao nhau giữa support và test query
                assert te_set.isdisjoint(chosen_s.tolist()), "RÒ RỈ DỮ LIỆU GIỮA SUPPORT VÀ QUERY!"
                s_z = F.normalize(z_all[chosen_s], dim=-1)
                protos.append(F.normalize(s_z.mean(dim=0), dim=-1))
            protos_t = torch.stack(protos, dim=0)
            probs = F.softmax(temp * (q_z @ protos_t.T), dim=-1).cpu().numpy()
            acc, bacc, mf1, auc = compute_metrics_np(probs, all_te_y)
            accs.append(acc)
            baccs.append(bacc)
            mf1s.append(mf1)
            aucs.append(auc)

        acc_by_s[sd] = accs
        bacc_by_s[sd] = baccs
        mf1_by_s[sd] = mf1s
        auc_by_s[sd] = aucs

    return {
        "acc": summarize_5seed_episodes(acc_by_s),
        "bacc": summarize_5seed_episodes(bacc_by_s),
        "mf1": summarize_5seed_episodes(mf1_by_s),
        "auroc": summarize_5seed_episodes(auc_by_s),
    }


def flatten_metric_row(family: str, method: str, crop_mode: str, shot: int, stats: dict) -> dict:
    return {
        "family": family,
        "method": method,
        "crop_mode": crop_mode,
        "shot": shot,
        "bacc_mean": round(stats["bacc"]["mean"], 3),
        "bacc_seed_std": round(stats["bacc"]["seed_std"], 3),
        "bacc_ep_ci95": round(stats["bacc"]["ep_ci95"], 3),
        "mf1_mean": round(stats["mf1"]["mean"], 3),
        "mf1_seed_std": round(stats["mf1"]["seed_std"], 3),
        "mf1_ep_ci95": round(stats["mf1"]["ep_ci95"], 3),
        "auroc_mean": round(stats["auroc"]["mean"], 3),
        "auroc_seed_std": round(stats["auroc"]["seed_std"], 3),
        "acc_mean": round(stats["acc"]["mean"], 3),
        "acc_seed_std": round(stats["acc"]["seed_std"], 3),
    }


def main():
    t0 = time.time()
    with PreventWindowsSleep():
        lock_info = record_or_verify_protected_files("record")
        verify_info = record_or_verify_protected_files("verify")
        print(f"[SHA-256 LOCK] {lock_info['status']} -> {verify_info['status']} ({verify_info['checked_count']} tệp được bảo vệ)", flush=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"=== BƯỚC 1 V2: CHẠY BASELINES 3 HƯỚNG x 5 SEEDS TRÊN {device} ===", flush=True)

        res_pkg = torch.load(V1_FEAT_DIR / "resnet50_features.pt", map_location="cpu", weights_only=False)
        bclip_pkg = torch.load(V1_FEAT_DIR / "biomedclip_features.pt", map_location="cpu", weights_only=False)

        res_feats = res_pkg["features"].float()
        bclip_img = F.normalize(bclip_pkg["image_features"].float(), dim=-1)
        simple_txt = F.normalize(bclip_pkg["simple_text_features"].float(), dim=-1)
        clin_txt = F.normalize(bclip_pkg["clinical_text_features"].float(), dim=-1)
        labels = bclip_pkg["labels"].numpy()
        splits = np.array(bclip_pkg["splits"])
        clean_sem_mask = bclip_pkg["clean_semantic_mask"].numpy()

        train_mask = (splits == "train") & clean_sem_mask
        test_mask = splits == "test"

        cls_to_id = {c: i for i, c in enumerate(ALL_CLASSES)}
        meta_tr_ids = [cls_to_id[c] for c in META_TRAIN_CLASSES]
        novel_cls_ids = [cls_to_id[c] for c in NOVEL_CLASSES]

        res_feats_dev = res_feats.to(device)
        bclip_img_dev = bclip_img.to(device)
        simple_txt_dev = simple_txt.to(device)
        clin_txt_dev = clin_txt.to(device)

        # Tính tâm toàn cục CHỈ trên tập train của 12 lớp Meta-Train (cho SimpleShot Baseline công bằng)
        meta_tr_sample_mask = train_mask & np.isin(labels, meta_tr_ids)
        meta_tr_idx = torch.tensor(np.where(meta_tr_sample_mask)[0], device=device)

        res_raw_norm = F.normalize(res_feats_dev, dim=-1)
        res_global_mean = res_raw_norm[meta_tr_idx].mean(dim=0, keepdim=True)
        res_centered_dev = F.normalize(res_raw_norm - res_global_mean, dim=-1)

        bclip_global_mean = bclip_img_dev[meta_tr_idx].mean(dim=0, keepdim=True)
        bclip_centered_dev = F.normalize(bclip_img_dev - bclip_global_mean, dim=-1)

        # 1. Huấn luyện 5 mô hình ResNet-50 + ProtoNet theo 5 seeds độc lập
        print("\n[1/3] Đang huấn luyện 5 mô hình ResNet-50 + ProtoNet ứng với 5 random seeds...", flush=True)
        res_z_by_seed = {}
        res_temp_by_seed = {}
        res_ckpt_bundle = {}
        for sd in SEEDS_V2:
            m_sd, val_bacc = train_single_resnet_protonet(res_feats_dev, labels, train_mask, test_mask, cls_to_id, device, seed=sd)
            with torch.no_grad():
                res_z_by_seed[sd] = m_sd(res_feats_dev)
                res_temp_by_seed[sd] = float(m_sd.scale.clamp(1.0, 40.0).item())
            res_ckpt_bundle[str(sd)] = {"state_dict": m_sd.state_dict(), "meta_val_5shot_bacc": round(val_bacc, 3)}
            print(f"  - Seed {sd:4d}: Meta-Val 5-shot BAcc = {val_bacc:.2f}% | scale = {res_temp_by_seed[sd]:.2f}", flush=True)

        torch.save(res_ckpt_bundle, CKPT_V2_DIR / "resnet50_protonet_5seeds.pt")

        # 2. Đánh giá Hướng 2: Zero-Shot Baselines (0-shot)
        print("\n[2/3] Đang đánh giá Hướng 2 (BiomedCLIP Zero-Shot Baselines)...", flush=True)
        rows_out = []
        full_detail = {}

        zs_simp = evaluate_zero_shot_5seeds(bclip_img_dev, simple_txt_dev, labels, test_mask, novel_cls_ids, temp=25.0)
        rows_out.append(flatten_metric_row("Hướng 2: VLM Zero-Shot", "BiomedCLIP Zero-Shot (Chỉ tên bệnh)", "1-crop", 0, zs_simp))
        full_detail["zs_simple"] = zs_simp

        zs_clin = evaluate_zero_shot_5seeds(bclip_img_dev, clin_txt_dev, labels, test_mask, novel_cls_ids, temp=25.0)
        rows_out.append(flatten_metric_row("Hướng 2: VLM Zero-Shot", "BiomedCLIP Zero-Shot (Triệu chứng gộp V1)", "1-crop", 0, zs_clin))
        full_detail["zs_clinical_v1"] = zs_clin

        # 3. Đánh giá Hướng 1 & Hướng 3 qua 1, 5, 10-shot (5 seeds x 200 = 1.000 episodes)
        print("\n[3/3] Đang đánh giá Hướng 1 & Hướng 3 trên 1, 5, 10-shot (5 seeds x 200 = 1.000 episodes)...", flush=True)
        for sh in SHOT_LIST:
            # ResNet-50 + ProtoNet (5 mô hình độc lập)
            r_proto = evaluate_visual_few_shot_5seeds(
                res_z_by_seed, res_temp_by_seed, labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            rows_out.append(flatten_metric_row("Hướng 1: CNN Metric", "ResNet-50 + ProtoNet (Episodic)", "1-crop", sh, r_proto))
            full_detail[f"resnet_protonet_{sh}shot"] = r_proto

            # ResNet-50 + SimpleShot (Centered L2)
            r_ss = evaluate_visual_few_shot_5seeds(
                {sd: res_centered_dev for sd in SEEDS_V2}, {sd: 20.0 for sd in SEEDS_V2},
                labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            rows_out.append(flatten_metric_row("Hướng 1: CNN Metric", "ResNet-50 + SimpleShot (Centered)", "1-crop", sh, r_ss))
            full_detail[f"resnet_simpleshot_{sh}shot"] = r_ss

            # BiomedCLIP + ProtoNet (Visual Only - Raw)
            b_proto = evaluate_visual_few_shot_5seeds(
                {sd: bclip_img_dev for sd in SEEDS_V2}, {sd: 25.0 for sd in SEEDS_V2},
                labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            rows_out.append(flatten_metric_row("Hướng 3: VLM Visual FSL", "BiomedCLIP + ProtoNet (Visual Only)", "1-crop", sh, b_proto))
            full_detail[f"biomedclip_protonet_{sh}shot"] = b_proto

            # BiomedCLIP + SimpleShot (Centered Visual Only)
            b_ss = evaluate_visual_few_shot_5seeds(
                {sd: bclip_centered_dev for sd in SEEDS_V2}, {sd: 25.0 for sd in SEEDS_V2},
                labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            rows_out.append(flatten_metric_row("Hướng 3: VLM Visual FSL", "BiomedCLIP + SimpleShot (Centered Visual)", "1-crop", sh, b_ss))
            full_detail[f"biomedclip_simpleshot_{sh}shot"] = b_ss

        csv_path = TABLES_V2_DIR / "step1_baselines_5seeds.csv"
        fieldnames = list(rows_out[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)

        json_path = TABLES_V2_DIR / "step1_baselines_5seeds.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seeds": SEEDS_V2,
                    "episodes_per_seed": N_EPISODES_PER_SEED,
                    "total_episodes": len(SEEDS_V2) * N_EPISODES_PER_SEED,
                    "elapsed_seconds": round(time.time() - t0, 1),
                    "summary_rows": rows_out,
                    "detailed_per_seed": full_detail,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Kiểm tra lại SHA-256 cuối Bước 1
        verify_after = record_or_verify_protected_files("verify")
        print("\n=== KẾT QUẢ BƯỚC 1 (BASELINES 3 HƯỚNG, 5 SEEDS, MEAN ± SEED_SD) ===", flush=True)
        for r in rows_out:
            print(
                f"  {r['method']:42s} | {r['shot']:2d}-shot | BAcc: {r['bacc_mean']:6.2f} ± {r['bacc_seed_std']:4.2f}% "
                f"(CI95: ±{r['bacc_ep_ci95']:4.2f}%) | F1: {r['mf1_mean']:6.2f} ± {r['mf1_seed_std']:4.2f}% | AUROC: {r['auroc_mean']:6.2f}%",
                flush=True,
            )
        print(f"\n[OK] Lưu tại: {csv_path} | SHA-256 Check: {verify_after['status']} | Thời gian: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
