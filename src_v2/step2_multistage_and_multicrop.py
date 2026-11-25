# -*- coding: utf-8 -*-
"""
BƯỚC 2 V2 (Ưu tiên số 2):
  - Phần 2A (trên đặc trưng 1-crop đã trích sẵn -> công bằng tuyệt đối với Bước 1):
      + Mã hóa từ điển 4 giai đoạn bệnh lý da liễu (cấp/ướt, bán cấp/mài, mạn/khô vảy, cào xước/phân biệt) cho 22 lớp.
      + Khảo sát từng nấc Ablation (6 phiên bản A1 -> A6) trên 5 random seeds (1.000 episodes):
          A1: BiomedCLIP Visual ProtoNet
          A2: + Mô tả triệu chứng gộp V1
          A3: + Đa nguyên mẫu 4 giai đoạn bệnh (LogSumExp Stage-Aware Matching)
          A4: + Khử thiên lệch tiền nghiệm văn bản (Domain-Prior De-biasing tính từ Meta-Train)
          A5: + Dịch tâm mặt cầu đặc trưng ảnh (Base-Mean Spherical Centering từ Meta-Train)
          A6: SAMP Đầy đủ (+ Support Sub-Stage Cache & Empirical-Bayes 1/K Calibration)
      + Toàn bộ siêu tham số CHỈ tìm trên 4 lớp Meta-Val (Rule 2: 0% nhìn trộm Novel).
  - Phần 2B (Trích xuất 5-Crop cho CẢ ResNet-50 & BiomedCLIP -> Rule 4: Công bằng tuyệt đối):
      + Trích xuất 5-crop (TL, TR, BL, BR, Center) ở fp16 cho cả ResNet-50 và BiomedCLIP trên 14.797 ảnh sạch.
      + Huấn luyện lại 5 mô hình ResNet-50+ProtoNet trên 5-crop và đánh giá lại đồng bộ toàn bộ Baselines + SAMP qua 5 seeds.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from step1_baselines_5seeds import (
    evaluate_visual_few_shot_5seeds,
    evaluate_zero_shot_5seeds,
    flatten_metric_row,
    prebuild_val_episodes,
    train_single_resnet_protonet,
)
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
    SRC_V2_DIR,
    TABLES_V2_DIR,
    V1_FEAT_DIR,
    V1_OUT_DIR,
    record_or_verify_protected_files,
    summarize_5seed_episodes,
)


def compute_metrics_with_preds(probs: np.ndarray, y_true: np.ndarray):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    mf1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
    try:
        auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro") * 100.0
    except ValueError:
        auroc = float("nan")
    return acc, bacc, mf1, auroc, preds


@torch.no_grad()
def encode_4stage_text_bank(device: torch.device) -> dict:
    """Mã hóa bộ mô tả 4 giai đoạn bệnh bằng PubMedBERT của BiomedCLIP (lưu cache để dùng lại cho Bước 4 & Web Demo)."""
    cache_pt = FEAT_V2_DIR / "stage_text_embs_v2.pt"
    stages_json_path = SRC_V2_DIR / "clinical_stages_v2.json"
    stages_db = json.loads(stages_json_path.read_text(encoding="utf-8"))

    if cache_pt.exists():
        pkg = torch.load(cache_pt, map_location="cpu", weights_only=False)
        if pkg.get("classes") == ALL_CLASSES:
            return pkg

    print("\n[Text Encoder] Đang mã hóa bộ mô tả 4 giai đoạn bệnh cho 22 lớp bằng BiomedCLIP...", flush=True)
    model, _, _ = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID)
    model = model.to(device).eval()

    stage_ids = [s["id"] for s in stages_db["stage_definitions"]]
    all_stage_embs = []  # [22, 4, 512]
    for cls_name in ALL_CLASSES:
        cls_stages = stages_db["classes"][cls_name]["stages"]
        texts = [f"clinical dermatology photograph showing {cls_stages[sid]}" for sid in stage_ids]
        tok = tokenizer(texts, context_length=256).to(device)
        emb = F.normalize(model.encode_text(tok).float(), dim=-1)  # [4, 512]
        all_stage_embs.append(emb.cpu())

    stage_tensor = torch.stack(all_stage_embs, dim=0)  # [22, 4, 512]
    pkg = {
        "classes": ALL_CLASSES,
        "stage_ids": stage_ids,
        "stage_definitions": stages_db["stage_definitions"],
        "stage_text_features": stage_tensor,
        "stage_mean_features": F.normalize(stage_tensor.mean(dim=1), dim=-1),
    }
    torch.save(pkg, cache_pt)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pkg


@torch.no_grad()
def compute_stage_text_logits(
    q_raw: torch.Tensor,
    stage_embs_sub: torch.Tensor,
    meta_train_raw_pool: torch.Tensor,
    tau_stage: float = 0.5,
    prior_weight: float = 0.5,
    temp: float = 25.0,
) -> torch.Tensor:
    """
    Tính điểm tương đồng văn bản đa giai đoạn (Stage-Aware LogSumExp + Domain-Prior De-biasing):
      - q_raw: [N_q, 512] (đã L2-normalize)
      - stage_embs_sub: [C, S, 512] (S=4 giai đoạn cho C lớp đích, hoặc S=1 nếu dùng mô tả gộp V1)
      - meta_train_raw_pool: [N_meta_tr, 512] (CHỈ lấy từ 12 lớp Meta-Train để ước lượng độ lệch tần suất từ vựng chung)
    """
    # raw_sim: [N_q, C, S]
    raw_sim = torch.einsum("nd,csd->ncs", q_raw, stage_embs_sub)
    if prior_weight > 1e-6 and meta_train_raw_pool is not None:
        # Độ tương đồng nền trung bình của từng câu mô tả trên tập ảnh da phổ biến (Meta-Train)
        bg_sim = torch.einsum("md,csd->mcs", meta_train_raw_pool, stage_embs_sub).mean(dim=0, keepdim=True)  # [1, C, S]
        sim_adj = raw_sim - prior_weight * bg_sim
    else:
        sim_adj = raw_sim

    scaled = temp * sim_adj  # [N_q, C, S]
    if stage_embs_sub.size(1) == 1 or tau_stage <= 1e-4:
        return scaled.mean(dim=-1)  # [N_q, C]
    # LogSumExp có chuẩn hóa số lượng giai đoạn S: tự động nhấn mạnh giai đoạn (ướt/mài/khô/xước) khớp nhất với ảnh q
    S = stage_embs_sub.size(1)
    return (torch.logsumexp(tau_stage * scaled, dim=-1) - np.log(float(S))) / tau_stage


@torch.no_grad()
def compute_samp_visual_logits(
    q_vis: torch.Tensor,
    sup_vis_by_cls: list,
    base_vis_means: torch.Tensor,
    calib_gamma: float = 0.20,
    tip_lambda: float = 0.25,
    tip_beta: float = 3.5,
    temp: float = 25.0,
) -> torch.Tensor:
    """
    Tính điểm tương đồng thị giác trong không gian mặt cầu đã dịch tâm:
      - Thành phần 1: Nguyên mẫu tâm lớp hiệu chỉnh Empirical-Bayes 1/K từ top-3 lớp Base lân cận.
      - Thành phần 2: Bộ nhớ đệm phi tham số trên từng ảnh support (Tip-Adapter Radial Kernel),
        giữ lại tính đa đỉnh khi K ảnh mẫu nằm ở các giai đoạn ướt/khô khác nhau của cùng 1 bệnh.
    """
    protos = []
    cache_scores = []
    b_norm = F.normalize(base_vis_means, dim=-1) if base_vis_means is not None else None

    for s_c in sup_vis_by_cls:
        # s_c: [K, D] đã chuẩn hóa trong không gian hiện hành
        k_shots = s_c.size(0)
        emp_p = F.normalize(s_c.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
        if calib_gamma > 1e-6 and b_norm is not None:
            sim_b = emp_p @ b_norm.T  # [1, N_base]
            top_v, top_i = torch.topk(sim_b, k=min(3, b_norm.size(0)), dim=-1)
            w_b = F.softmax(top_v * 15.0, dim=-1)
            neigh = F.normalize((w_b.unsqueeze(-1) * b_norm[top_i]).sum(dim=1), dim=-1)
            eff_g = calib_gamma / float(k_shots)
            p_c = F.normalize((1.0 - eff_g) * emp_p + eff_g * neigh, dim=-1).squeeze(0)
        else:
            p_c = emp_p.squeeze(0)
        protos.append(p_c)

        if tip_lambda > 1e-6:
            # Kernel phi tham số tới từng mẫu support của lớp c: trung bình exp(-beta * (1 - cos))
            sim_q_s = q_vis @ s_c.T  # [N_q, K]
            aff = torch.exp(-tip_beta * (1.0 - sim_q_s)).mean(dim=-1)  # [N_q]
            cache_scores.append(aff)

    protos_t = torch.stack(protos, dim=0)  # [C, D]
    proto_logits = temp * (q_vis @ protos_t.T)  # [N_q, C]
    if tip_lambda > 1e-6:
        cache_mat = torch.stack(cache_scores, dim=-1)  # [N_q, C]
        return (1.0 - tip_lambda) * proto_logits + tip_lambda * (temp * cache_mat)
    return proto_logits


@torch.no_grad()
def tune_samp_hyperparams_on_meta_val(
    raw_img_dev: torch.Tensor,
    stage_embs_dev: torch.Tensor,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    cls_to_id: dict,
    temp: float = 25.0,
) -> dict:
    """
    LUẬT SỐ 2: Toàn bộ việc dò siêu tham số của SAMP diễn ra 100% trên 4 lớp Meta-Val,
    sử dụng thống kê nền từ 12 lớp Meta-Train. Tuyệt đối KHÔNG chạm vào 6 lớp Novel!
    """
    meta_tr_ids = [cls_to_id[c] for c in META_TRAIN_CLASSES]
    meta_val_ids = [cls_to_id[c] for c in META_VAL_CLASSES]

    meta_tr_mask = train_mask & np.isin(labels, meta_tr_ids)
    meta_tr_idx = torch.tensor(np.where(meta_tr_mask)[0], device=raw_img_dev.device)
    meta_tr_pool = raw_img_dev[meta_tr_idx]
    global_mean = meta_tr_pool.mean(dim=0, keepdim=True)

    val_tr_pools = {c: np.where(train_mask & (labels == c))[0] for c in meta_val_ids}
    val_te_pools = {c: np.where(test_mask & (labels == c))[0] for c in meta_val_ids}

    # Bước A: Chọn (tau_stage, prior_weight) cho nhánh văn bản 4 giai đoạn trên tập test của 4 lớp Meta-Val (0-shot Meta-Val)
    vq_all_idx = np.concatenate([val_te_pools[c] for c in meta_val_ids])
    vq_all_y = np.concatenate([np.full(len(val_te_pools[c]), i, dtype=np.int64) for i, c in enumerate(meta_val_ids)])
    vq_cls_counts = np.bincount(vq_all_y, minlength=len(meta_val_ids)).astype(np.float64)
    vq_raw = raw_img_dev[vq_all_idx]
    val_stage_sub = stage_embs_dev[meta_val_ids]  # [4, 4, 512]

    def _fast_bacc(preds_arr: np.ndarray) -> float:
        corr = np.bincount(vq_all_y[preds_arr == vq_all_y], minlength=len(meta_val_ids))
        return float((corr / vq_cls_counts).mean() * 100.0)

    best_txt_bacc = -1.0
    best_tau_s, best_prior_w = 0.5, 0.5
    for tau_s in [0.0, 0.25, 0.50, 1.0]:
        for pw in [0.0, 0.35, 0.65, 0.85]:
            t_logits = compute_stage_text_logits(vq_raw, val_stage_sub, meta_tr_pool, tau_stage=tau_s, prior_weight=pw, temp=temp)
            bacc_v = _fast_bacc(t_logits.argmax(dim=-1).cpu().numpy())
            if bacc_v > best_txt_bacc:
                best_txt_bacc = bacc_v
                best_tau_s, best_prior_w = tau_s, pw

    # Tính trước điểm văn bản tối ưu trên Meta-Val
    val_txt_logits_fixed = compute_stage_text_logits(
        vq_raw, val_stage_sub, meta_tr_pool, tau_stage=best_tau_s, prior_weight=best_prior_w, temp=temp
    )

    # Bước B: Với mỗi mức K-shot (1, 5, 10), tìm (center_beta, alpha_vis, calib_gamma, tip_lambda) trên 120 episodes Meta-Val
    best_by_shot = {
        0: {"tau_stage": best_tau_s, "prior_weight": best_prior_w, "meta_val_0shot_bacc": round(best_txt_bacc, 3)}
    }
    labels_dev = torch.tensor(labels, device=raw_img_dev.device)

    for k_shot in SHOT_LIST:
        val_eps, _, _ = prebuild_val_episodes(val_tr_pools, val_te_pools, meta_val_ids, k_shot=k_shot, n_episodes=120, seed=880 + k_shot)
        best_score = -1.0
        best_cfg = None

        for c_beta in [0.0, 0.6, 1.0]:
            z_space = F.normalize(raw_img_dev - c_beta * global_mean, dim=-1)
            vq_vis = z_space[vq_all_idx]
            # Tâm 12 lớp Meta-Train trong không gian đã dịch tâm
            b_means = torch.stack(
                [F.normalize(z_space[meta_tr_idx][labels_dev[meta_tr_idx] == cid].mean(dim=0), dim=-1) for cid in meta_tr_ids],
                dim=0,
            )
            for alpha_v in [0.25, 0.40, 0.55, 0.70]:
                for c_gamma in [0.0, 0.15, 0.30]:
                    for t_lam in ([0.0] if k_shot == 1 else [0.0, 0.20, 0.35]):
                        ep_baccs = []
                        for vs_idx, vs_y in val_eps:
                            sup_list = [z_space[vs_idx][vs_y == cl] for cl in range(len(meta_val_ids))]
                            v_logits = compute_samp_visual_logits(
                                vq_vis, sup_list, b_means, calib_gamma=c_gamma, tip_lambda=t_lam, tip_beta=3.5, temp=temp
                            )
                            comb = alpha_v * v_logits + (1.0 - alpha_v) * val_txt_logits_fixed
                            pr = comb.argmax(dim=-1).cpu().numpy()
                            ep_baccs.append(_fast_bacc(pr))
                        m_sc = float(np.mean(ep_baccs))
                        if m_sc > best_score:
                            best_score = m_sc
                            best_cfg = {
                                "tau_stage": best_tau_s,
                                "prior_weight": best_prior_w,
                                "center_beta": c_beta,
                                "alpha_vis": alpha_v,
                                "calib_gamma": c_gamma,
                                "tip_lambda": t_lam,
                                "meta_val_bacc": round(m_sc, 3),
                            }
        best_by_shot[k_shot] = best_cfg

    return best_by_shot


@torch.no_grad()
def evaluate_samp_variant_5seeds(
    raw_img_dev: torch.Tensor,
    stage_embs_dev: torch.Tensor,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    base_cls_ids: list,
    meta_tr_ids: list,
    target_cls_ids: list,
    k_shot: int,
    seeds: list,
    n_episodes_per_seed: int,
    cfg: dict,
    temp: float = 25.0,
):
    """Đánh giá 1 phiên bản cấu hình trên 6 lớp Novel qua 5 seeds x 200 = 1.000 episodes."""
    meta_tr_mask = train_mask & np.isin(labels, meta_tr_ids)
    meta_tr_idx = torch.tensor(np.where(meta_tr_mask)[0], device=raw_img_dev.device)
    meta_tr_pool = raw_img_dev[meta_tr_idx]
    global_mean = meta_tr_pool.mean(dim=0, keepdim=True)

    te_pools = {c: np.where(test_mask & (labels == c))[0] for c in target_cls_ids}
    all_te_idx = np.concatenate([te_pools[c] for c in target_cls_ids])
    all_te_y = np.concatenate([np.full(len(te_pools[c]), i, dtype=np.int64) for i, c in enumerate(target_cls_ids)])
    q_raw = raw_img_dev[all_te_idx]
    stage_sub = stage_embs_dev[target_cls_ids]

    txt_logits = compute_stage_text_logits(
        q_raw,
        stage_sub,
        meta_tr_pool,
        tau_stage=cfg.get("tau_stage", 0.0),
        prior_weight=cfg.get("prior_weight", 0.0),
        temp=temp,
    )

    if k_shot == 0:
        probs = F.softmax(txt_logits, dim=-1).cpu().numpy()
        acc, bacc, mf1, auc, preds = compute_metrics_with_preds(probs, all_te_y)
        per_cls_rec = {
            ALL_CLASSES[cid]: round(float((preds[all_te_y == lc] == lc).mean() * 100.0), 2)
            for lc, cid in enumerate(target_cls_ids)
        }
        return {
            "acc": {"mean": acc, "seed_std": 0.0, "ep_ci95": 0.0},
            "bacc": {"mean": bacc, "seed_std": 0.0, "ep_ci95": 0.0},
            "mf1": {"mean": mf1, "seed_std": 0.0, "ep_ci95": 0.0},
            "auroc": {"mean": auc, "seed_std": 0.0, "ep_ci95": 0.0},
            "per_class_recall": per_cls_rec,
            "y_true": all_te_y,
            "preds": preds,
        }

    c_beta = cfg.get("center_beta", 0.0)
    alpha_v = cfg.get("alpha_vis", 1.0)
    c_gamma = cfg.get("calib_gamma", 0.0)
    t_lam = cfg.get("tip_lambda", 0.0)
    use_text = cfg.get("use_text", True)

    z_space = F.normalize(raw_img_dev - c_beta * global_mean, dim=-1)
    q_vis = z_space[all_te_idx]

    base_tr_mask = train_mask & np.isin(labels, base_cls_ids)
    base_tr_idx = torch.tensor(np.where(base_tr_mask)[0], device=raw_img_dev.device)
    labels_dev = torch.tensor(labels, device=raw_img_dev.device)
    base_means = torch.stack(
        [F.normalize(z_space[base_tr_idx][labels_dev[base_tr_idx] == cid].mean(dim=0), dim=-1) for cid in base_cls_ids],
        dim=0,
    )

    tr_pools = {c: np.where(train_mask & (labels == c))[0] for c in target_cls_ids}
    te_set = set(all_te_idx.tolist())

    acc_by_s, bacc_by_s, mf1_by_s, auc_by_s = {}, {}, {}, {}
    agg_probs = np.zeros((len(all_te_y), len(target_cls_ids)), dtype=np.float64)
    total_eps = 0

    for sd in seeds:
        rng = np.random.default_rng(sd + k_shot * 17)
        accs, baccs, mf1s, aucs = [], [], [], []
        for _ in range(n_episodes_per_seed):
            sup_list = []
            for cid in target_cls_ids:
                chosen_s = rng.choice(tr_pools[cid], size=k_shot, replace=False)
                assert te_set.isdisjoint(chosen_s.tolist())
                sup_list.append(z_space[chosen_s])

            v_logits = compute_samp_visual_logits(
                q_vis, sup_list, base_means, calib_gamma=c_gamma, tip_lambda=t_lam, tip_beta=3.5, temp=temp
            )
            comb_logits = (alpha_v * v_logits + (1.0 - alpha_v) * txt_logits) if use_text else v_logits
            probs = F.softmax(comb_logits, dim=-1).cpu().numpy()
            acc, bacc, mf1, auc, _ = compute_metrics_with_preds(probs, all_te_y)
            accs.append(acc)
            baccs.append(bacc)
            mf1s.append(mf1)
            aucs.append(auc)
            agg_probs += probs
            total_eps += 1

        acc_by_s[sd] = accs
        bacc_by_s[sd] = baccs
        mf1_by_s[sd] = mf1s
        auc_by_s[sd] = aucs

    mean_preds = (agg_probs / max(1, total_eps)).argmax(axis=1)
    per_cls_rec = {
        ALL_CLASSES[cid]: round(float((mean_preds[all_te_y == lc] == lc).mean() * 100.0), 2)
        for lc, cid in enumerate(target_cls_ids)
    }
    return {
        "acc": summarize_5seed_episodes(acc_by_s),
        "bacc": summarize_5seed_episodes(bacc_by_s),
        "mf1": summarize_5seed_episodes(mf1_by_s),
        "auroc": summarize_5seed_episodes(auc_by_s),
        "per_class_recall": per_cls_rec,
        "y_true": all_te_y,
        "preds": mean_preds,
    }


# =============================================================================
# PHẦN 2B: TRÍCH XUẤT ĐẶC TRƯNG 5-CROP CÔNG BẰNG CHO CẢ RESNET-50 & BIOMEDCLIP
# =============================================================================
def load_manifest_rows():
    manifest_path = V1_OUT_DIR / "clean_manifest.csv"
    rows = []
    cls_to_idx = {c: i for i, c in enumerate(ALL_CLASSES)}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["label"] = cls_to_idx[r["class"]]
            rows.append(r)
    return rows


def iter_5crop_batches(rows, resize_size: int, crop_size: int, norm_mean, norm_std, interp, batch_size: int = 32, max_workers: int = 8):
    """Giải mã ảnh và cắt 5 góc chuẩn (TL, TR, BL, BR, Center) bằng ThreadPool để tối ưu tốc độ trên Windows."""
    five_crop_tfm = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interp),
        transforms.FiveCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=norm_mean, std=norm_std)(transforms.ToTensor()(c)) for c in crops])),
    ])

    def _load(idx):
        p = DATA_DIR / rows[idx]["rel_path"]
        with Image.open(p) as im:
            return five_crop_tfm(im.convert("RGB")), idx

    n = len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in range(0, n, batch_size):
            b_idxs = range(start, min(start + batch_size, n))
            loaded = list(pool.map(_load, b_idxs))
            # [B, 5, 3, H, W]
            yield torch.stack([x[0] for x in loaded], dim=0), [x[1] for x in loaded]


@torch.no_grad()
def extract_fair_5crop_features(device: torch.device):
    """Trích xuất 5-crop cho CẢ ResNet-50 lẫn BiomedCLIP để đảm bảo Rule 4 (so sánh công bằng 100%)."""
    res_5c_path = FEAT_V2_DIR / "resnet50_5crop.pt"
    bclip_5c_path = FEAT_V2_DIR / "biomedclip_5crop.pt"
    rows = load_manifest_rows()
    n_total = len(rows)
    use_amp = device.type == "cuda"

    if not res_5c_path.exists():
        print(f"\n[5-Crop 1/2] Đang trích xuất 5-Crop cho ResNet-50 trên {n_total} ảnh...", flush=True)
        t0 = time.time()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        resnet.fc = torch.nn.Identity()
        resnet = resnet.to(device).eval()

        feats_5c = torch.zeros((n_total, 2048), dtype=torch.float32)
        n_b = (n_total + 31) // 32
        for crops5, idxs in tqdm(
            iter_5crop_batches(rows, 256, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), InterpolationMode.BILINEAR, batch_size=32),
            total=n_b, desc="ResNet50-5Crop", ncols=90,
        ):
            bsz = crops5.size(0)
            flat = crops5.view(bsz * 5, 3, 224, 224).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out = F.normalize(resnet(flat).float(), dim=-1).view(bsz, 5, 2048)
                avg = F.normalize(out.mean(dim=1), dim=-1)
            feats_5c[idxs] = avg.cpu()

        torch.save({"features": feats_5c}, res_5c_path)
        print(f"-> Xong ResNet-50 5-Crop trong {time.time() - t0:.1f}s | Lưu tại: {res_5c_path}", flush=True)
        del resnet
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()

    if not bclip_5c_path.exists():
        print(f"\n[5-Crop 2/2] Đang trích xuất 5-Crop cho BiomedCLIP trên {n_total} ảnh...", flush=True)
        t0 = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
        model = model.to(device).eval()
        # Lấy chuẩn mean/std chính xác từ preprocess của BiomedCLIP
        norm_op = next(t for t in preprocess.transforms if isinstance(t, transforms.Normalize))
        b_mean, b_std = norm_op.mean, norm_op.std

        img_5c = torch.zeros((n_total, 512), dtype=torch.float32)
        n_b = (n_total + 31) // 32
        for crops5, idxs in tqdm(
            iter_5crop_batches(rows, 256, 224, b_mean, b_std, InterpolationMode.BICUBIC, batch_size=32),
            total=n_b, desc="BiomedCLIP-5Crop", ncols=90,
        ):
            bsz = crops5.size(0)
            flat = crops5.view(bsz * 5, 3, 224, 224).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                emb = F.normalize(model.encode_image(flat).float(), dim=-1).view(bsz, 5, 512)
                avg = F.normalize(emb.mean(dim=1), dim=-1)
            img_5c[idxs] = avg.cpu()

        torch.save({"image_features": img_5c}, bclip_5c_path)
        print(f"-> Xong BiomedCLIP 5-Crop trong {time.time() - t0:.1f}s | Lưu tại: {bclip_5c_path}", flush=True)
        del model
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()

    return (
        torch.load(res_5c_path, map_location="cpu", weights_only=False)["features"].float(),
        torch.load(bclip_5c_path, map_location="cpu", weights_only=False)["image_features"].float(),
    )


def main():
    t0 = time.time()
    with PreventWindowsSleep():
        record_or_verify_protected_files("verify")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"=== BƯỚC 2 V2: ĐA NGUYÊN MẪU 4 GIAI ĐOẠN & ĐỐI CHỨNG CÔNG BẰNG 1-CROP vs 5-CROP ({device}) ===", flush=True)

        # Nạp đặc trưng 1-crop sạch từ V1 (CHỈ ĐỌC)
        bclip_v1 = torch.load(V1_FEAT_DIR / "biomedclip_features.pt", map_location="cpu", weights_only=False)
        img_1crop = F.normalize(bclip_v1["image_features"].float(), dim=-1)
        simple_txt = F.normalize(bclip_v1["simple_text_features"].float(), dim=-1)
        v1_clin_txt = F.normalize(bclip_v1["clinical_text_features"].float(), dim=-1)
        labels = bclip_v1["labels"].numpy()
        splits = np.array(bclip_v1["splits"])
        clean_sem_mask = bclip_v1["clean_semantic_mask"].numpy()

        train_mask = (splits == "train") & clean_sem_mask
        test_mask = splits == "test"

        cls_to_id = {c: i for i, c in enumerate(ALL_CLASSES)}
        base_cls_ids = [cls_to_id[c] for c in BASE_CLASSES]
        meta_tr_ids = [cls_to_id[c] for c in META_TRAIN_CLASSES]
        novel_cls_ids = [cls_to_id[c] for c in NOVEL_CLASSES]

        # 1. Mã hóa ngân hàng câu chữ 4 giai đoạn bệnh
        stage_pkg = encode_4stage_text_bank(device)
        stage_embs_dev = stage_pkg["stage_text_features"].to(device)      # [22, 4, 512]
        v1_clin_3d_dev = v1_clin_txt.unsqueeze(1).to(device)               # [22, 1, 512]
        simple_txt_dev = simple_txt.to(device)
        img_1crop_dev = img_1crop.to(device)

        # 2. PHẦN 2A: Dò siêu tham số trên Meta-Val & Chạy Ablation A1 -> A6 trên 1-crop (5 seeds)
        print("\n[Phần 2A] Đang dò siêu tham số SAMP trên 4 lớp Meta-Val (1-crop)...", flush=True)
        hparams_1crop = tune_samp_hyperparams_on_meta_val(
            img_1crop_dev, stage_embs_dev, labels, train_mask, test_mask, cls_to_id, temp=25.0
        )
        print(f"-> Siêu tham số chọn từ Meta-Val (1-crop): {hparams_1crop}", flush=True)

        ablation_1crop_rows = []
        per_class_evolution = {}

        # Đánh giá 0-shot của mô tả 4 giai đoạn (có & không có Prior De-biasing)
        zs_stage_raw = evaluate_samp_variant_5seeds(
            img_1crop_dev, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
            k_shot=0, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            cfg={"tau_stage": hparams_1crop[0]["tau_stage"], "prior_weight": 0.0},
        )
        r_zsa3 = flatten_metric_row("Zero-Shot 4 giai đoạn", "ZS-A3: BiomedCLIP + 4 giai đoạn bệnh", "1-crop", 0, zs_stage_raw)
        r_zsa3["lupus_recall"] = zs_stage_raw["per_class_recall"]["Lupus"]
        ablation_1crop_rows.append(r_zsa3)

        zs_stage_debias = evaluate_samp_variant_5seeds(
            img_1crop_dev, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
            k_shot=0, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            cfg=hparams_1crop[0],
        )
        r_zsa4 = flatten_metric_row("Zero-Shot 4 giai đoạn", "ZS-A4: BiomedCLIP + 4 giai đoạn + Prior De-biasing", "1-crop", 0, zs_stage_debias)
        r_zsa4["lupus_recall"] = zs_stage_debias["per_class_recall"]["Lupus"]
        ablation_1crop_rows.append(r_zsa4)
        per_class_evolution["0shot_samp"] = zs_stage_debias["per_class_recall"]

        for sh in SHOT_LIST:
            hp = hparams_1crop[sh]
            variants = [
                ("A1. BiomedCLIP + ProtoNet (Chỉ ảnh)", v1_clin_3d_dev, {"use_text": False, "center_beta": 0.0, "calib_gamma": 0.0, "tip_lambda": 0.0}),
                ("A2. + Mô tả triệu chứng 1 câu (V1)", v1_clin_3d_dev, {"use_text": True, "alpha_vis": hp["alpha_vis"], "tau_stage": 0.0, "prior_weight": 0.0, "center_beta": 0.0, "calib_gamma": 0.0, "tip_lambda": 0.0}),
                ("A3. + Đa nguyên mẫu 4 giai đoạn (ướt/mài/khô/xước)", stage_embs_dev, {"use_text": True, "alpha_vis": hp["alpha_vis"], "tau_stage": hp["tau_stage"], "prior_weight": 0.0, "center_beta": 0.0, "calib_gamma": 0.0, "tip_lambda": 0.0}),
                ("A4. + Khử thiên lệch từ vựng (Prior De-biasing)", stage_embs_dev, {"use_text": True, "alpha_vis": hp["alpha_vis"], "tau_stage": hp["tau_stage"], "prior_weight": hp["prior_weight"], "center_beta": 0.0, "calib_gamma": 0.0, "tip_lambda": 0.0}),
                ("A5. + Dịch tâm mặt cầu (Base-Mean Centering)", stage_embs_dev, {"use_text": True, "alpha_vis": hp["alpha_vis"], "tau_stage": hp["tau_stage"], "prior_weight": hp["prior_weight"], "center_beta": hp["center_beta"], "calib_gamma": 0.0, "tip_lambda": 0.0}),
                ("A6. SAMP Đầy đủ (+ Cache & Calibration)", stage_embs_dev, {"use_text": True, **hp}),
            ]
            for v_name, st_tensor, cfg_v in variants:
                res_v = evaluate_samp_variant_5seeds(
                    img_1crop_dev, st_tensor, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
                    k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED, cfg=cfg_v,
                )
                row_flat = flatten_metric_row("Ablation 1-crop", v_name, "1-crop", sh, res_v)
                row_flat["lupus_recall"] = res_v["per_class_recall"]["Lupus"]
                ablation_1crop_rows.append(row_flat)
                if v_name.startswith("A1") or v_name.startswith("A6"):
                    per_class_evolution[f"{v_name[:2]}_{sh}shot"] = res_v["per_class_recall"]

        csv_2a = TABLES_V2_DIR / "step2a_ablation_1crop_5seeds.csv"
        fields_2a = list(dict.fromkeys(k for r in ablation_1crop_rows for k in r.keys()))
        with open(csv_2a, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields_2a)
            w.writeheader()
            w.writerows(ablation_1crop_rows)

        print("\n=== KẾT QUẢ PHẦN 2A (ABLATION TRÊN 1-CROP, 5 SEEDS, MEAN ± SEED_SD) ===", flush=True)
        for r in ablation_1crop_rows:
            lup = f" | Lupus: {r.get('lupus_recall', '—')}%" if "lupus_recall" in r else ""
            print(
                f"  {r['method']:52s} | {r['shot']:2d}-shot | BAcc: {r['bacc_mean']:6.2f} ± {r['bacc_seed_std']:4.2f}% "
                f"| F1: {r['mf1_mean']:6.2f}% | AUROC: {r['auroc_mean']:6.2f}%{lup}",
                flush=True,
            )

        # 3. PHẦN 2B: Trích xuất 5-Crop cho CẢ ResNet-50 & BiomedCLIP và so sánh công bằng toàn bộ mô hình
        print("\n[Phần 2B] Bắt đầu trích xuất và đánh giá công bằng 5-Crop cho CẢ Baselines & SAMP...", flush=True)
        res_5crop, bclip_5crop = extract_fair_5crop_features(device)
        res_5c_dev = res_5crop.to(device)
        bclip_5c_dev = F.normalize(bclip_5crop.to(device), dim=-1)

        # Huấn luyện 5 mô hình ResNet-50 + ProtoNet độc lập trên đặc trưng 5-crop
        print("-> Đang huấn luyện 5 mô hình ResNet-50 + ProtoNet trên đặc trưng 5-Crop...", flush=True)
        res5c_z_by_seed, res5c_temp_by_seed = {}, {}
        for sd in SEEDS_V2:
            m_sd, vb = train_single_resnet_protonet(res_5c_dev, labels, train_mask, test_mask, cls_to_id, device, seed=sd)
            with torch.no_grad():
                res5c_z_by_seed[sd] = m_sd(res_5c_dev)
                res5c_temp_by_seed[sd] = float(m_sd.scale.clamp(1.0, 40.0).item())

        # Chọn lại siêu tham số SAMP độc lập trên Meta-Val của 5-crop
        hparams_5crop = tune_samp_hyperparams_on_meta_val(
            bclip_5c_dev, stage_embs_dev, labels, train_mask, test_mask, cls_to_id, temp=25.0
        )
        print(f"-> Siêu tham số chọn từ Meta-Val (5-crop): {hparams_5crop}", flush=True)

        meta_tr_mask = train_mask & np.isin(labels, meta_tr_ids)
        meta_tr_idx = torch.tensor(np.where(meta_tr_mask)[0], device=device)
        res5c_norm = F.normalize(res_5c_dev, dim=-1)
        res5c_centered = F.normalize(res5c_norm - res5c_norm[meta_tr_idx].mean(dim=0, keepdim=True), dim=-1)
        bclip5c_centered = F.normalize(bclip_5c_dev - bclip_5c_dev[meta_tr_idx].mean(dim=0, keepdim=True), dim=-1)

        fair_5crop_rows = []
        # 0-shot trên 5-crop
        zs5_simp = evaluate_zero_shot_5seeds(bclip_5c_dev, simple_txt_dev, labels, test_mask, novel_cls_ids, temp=25.0)
        fair_5crop_rows.append(flatten_metric_row("Zero-Shot (5-crop)", "BiomedCLIP Zero-Shot (Chỉ tên bệnh)", "5-crop", 0, zs5_simp))

        zs5_clin1 = evaluate_zero_shot_5seeds(bclip_5c_dev, v1_clin_txt.to(device), labels, test_mask, novel_cls_ids, temp=25.0)
        fair_5crop_rows.append(flatten_metric_row("Zero-Shot (5-crop)", "BiomedCLIP Zero-Shot (Triệu chứng gộp V1)", "5-crop", 0, zs5_clin1))

        zs5_samp = evaluate_samp_variant_5seeds(
            bclip_5c_dev, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
            k_shot=0, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED, cfg=hparams_5crop[0],
        )
        fair_5crop_rows.append(flatten_metric_row("Zero-Shot (5-crop)", "SAMP Zero-Shot (4 giai đoạn + Prior De-biasing)", "5-crop", 0, zs5_samp))

        cm_samp_5crop_5shot = None
        for sh in SHOT_LIST:
            r5_p = evaluate_visual_few_shot_5seeds(
                res5c_z_by_seed, res5c_temp_by_seed, labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            fair_5crop_rows.append(flatten_metric_row("Baseline (5-crop)", "ResNet-50 + ProtoNet", "5-crop", sh, r5_p))

            r5_ss = evaluate_visual_few_shot_5seeds(
                {sd: res5c_centered for sd in SEEDS_V2}, {sd: 20.0 for sd in SEEDS_V2},
                labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            fair_5crop_rows.append(flatten_metric_row("Baseline (5-crop)", "ResNet-50 + SimpleShot", "5-crop", sh, r5_ss))

            b5_p = evaluate_visual_few_shot_5seeds(
                {sd: bclip_5c_dev for sd in SEEDS_V2}, {sd: 25.0 for sd in SEEDS_V2},
                labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            fair_5crop_rows.append(flatten_metric_row("Baseline (5-crop)", "BiomedCLIP + ProtoNet (Visual Only)", "5-crop", sh, b5_p))

            b5_ss = evaluate_visual_few_shot_5seeds(
                {sd: bclip5c_centered for sd in SEEDS_V2}, {sd: 25.0 for sd in SEEDS_V2},
                labels, train_mask, test_mask, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED,
            )
            fair_5crop_rows.append(flatten_metric_row("Baseline (5-crop)", "BiomedCLIP + SimpleShot (Centered Visual)", "5-crop", sh, b5_ss))

            s5_full = evaluate_samp_variant_5seeds(
                bclip_5c_dev, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
                k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED, cfg={"use_text": True, **hparams_5crop[sh]},
            )
            fair_5crop_rows.append(flatten_metric_row("Proposed (5-crop)", "SAMP Đầy đủ (4 giai đoạn + Centering + Calibration)", "5-crop", sh, s5_full))
            if sh == 5:
                cm_samp_5crop_5shot = confusion_matrix(s5_full["y_true"], s5_full["preds"], labels=list(range(len(novel_cls_ids)))).tolist()
                per_class_evolution["A6_5crop_5shot"] = s5_full["per_class_recall"]

        csv_2b = TABLES_V2_DIR / "step2b_all_methods_5crop_5seeds.csv"
        with open(csv_2b, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(fair_5crop_rows[0].keys()))
            w.writeheader()
            w.writerows(fair_5crop_rows)

        # Lưu gói suy luận SAMP V2 phục vụ Bước 4 (PAD-UFES-20) & Bước 6 (FastAPI Web Demo)
        meta_tr_mask_1c = train_mask & np.isin(labels, meta_tr_ids)
        meta_tr_pool_1c = img_1crop_dev[torch.tensor(np.where(meta_tr_mask_1c)[0], device=device)]
        global_mean_1c = meta_tr_pool_1c.mean(dim=0, keepdim=True).cpu()
        bg_stage_prior_1c = torch.einsum("md,csd->mcs", meta_tr_pool_1c, stage_embs_dev).mean(dim=0).cpu()  # [22, 4]

        # Tính tâm tất cả 22 lớp cho Web Demo live inference
        c_beta_demo = hparams_1crop[5]["center_beta"]
        z_1c_centered = F.normalize(img_1crop - c_beta_demo * global_mean_1c, dim=-1)
        all_cls_centered_means = torch.stack(
            [F.normalize(z_1c_centered[train_mask & (labels == i)].mean(dim=0), dim=-1) for i in range(len(ALL_CLASSES))],
            dim=0,
        )
        torch.save(
            {
                "classes": ALL_CLASSES,
                "stage_ids": stage_pkg["stage_ids"],
                "stage_definitions": stage_pkg["stage_definitions"],
                "stage_text_features": stage_pkg["stage_text_features"],
                "concept_features": bclip_v1["concept_features"],
                "concept_meta": bclip_v1["concept_meta"],
                "meta_train_global_mean": global_mean_1c,
                "meta_train_raw_sample": meta_tr_pool_1c[:512].cpu(),
                "bg_stage_prior": bg_stage_prior_1c,
                "all_cls_centered_means": all_cls_centered_means,
                "base_cls_centered_means": all_cls_centered_means[base_cls_ids],
                "hparams_1crop": hparams_1crop,
                "hparams_5crop": hparams_5crop,
            },
            CKPT_V2_DIR / "samp_v2_bundle.pt",
        )

        with open(TABLES_V2_DIR / "step2_full_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hparams_1crop_from_meta_val": hparams_1crop,
                    "hparams_5crop_from_meta_val": hparams_5crop,
                    "ablation_1crop_rows": ablation_1crop_rows,
                    "fair_5crop_rows": fair_5crop_rows,
                    "per_class_recall_evolution": per_class_evolution,
                    "confusion_matrix_samp_5crop_5shot": cm_samp_5crop_5shot,
                    "elapsed_seconds": round(time.time() - t0, 1),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        verify_after = record_or_verify_protected_files("verify")
        print("\n=== KẾT QUẢ PHẦN 2B (SO SÁNH CÔNG BẰNG TRÊN 5-CROP, 5 SEEDS) ===", flush=True)
        for r in fair_5crop_rows:
            print(
                f"  {r['method']:52s} | {r['shot']:2d}-shot | BAcc: {r['bacc_mean']:6.2f} ± {r['bacc_seed_std']:4.2f}% "
                f"| F1: {r['mf1_mean']:6.2f}% | AUROC: {r['auroc_mean']:6.2f}%",
                flush=True,
            )
        print(f"\n[OK] Hoàn tất Bước 2 trong {time.time() - t0:.1f}s | SHA-256: {verify_after['status']}", flush=True)


if __name__ == "__main__":
    main()
