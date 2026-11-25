# -*- coding: utf-8 -*-
"""
Bước 5B (Bổ sung 15 giây vào bảng riêng Bước 5):
  - Kiểm chứng kỹ thuật WiSE-FT (Wortsman et al., CVPR 2022): Nội suy đặc trưng giữa Frozen BiomedCLIP
    và Fine-Tuned BiomedCLIP theo hệ số rho chọn hoàn toàn trên Meta-Val, khắc phục hiện tượng lệch không gian mở khi fine-tune.
"""
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from step1_baselines_5seeds import flatten_metric_row, prebuild_val_episodes
from step2_multistage_and_multicrop import evaluate_samp_variant_5seeds, tune_samp_hyperparams_on_meta_val
from v2_config import (
    ALL_CLASSES,
    BASE_CLASSES,
    FEAT_V2_DIR,
    META_TRAIN_CLASSES,
    META_VAL_CLASSES,
    N_EPISODES_PER_SEED,
    NOVEL_CLASSES,
    SEEDS_V2,
    SHOT_LIST,
    TABLES_V2_DIR,
    V1_FEAT_DIR,
    record_or_verify_protected_files,
)


def main():
    record_or_verify_protected_files("verify")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bclip_v1 = torch.load(V1_FEAT_DIR / "biomedclip_features.pt", map_location="cpu", weights_only=False)
    ft_pkg = torch.load(FEAT_V2_DIR / "biomedclip_finetuned_1crop.pt", map_location="cpu", weights_only=False)
    stage_pkg = torch.load(FEAT_V2_DIR / "stage_text_embs_v2.pt", map_location="cpu", weights_only=False)

    z_froz = F.normalize(bclip_v1["image_features"].float().to(device), dim=-1)
    z_ft = F.normalize(ft_pkg["image_features"].float().to(device), dim=-1)
    stage_embs_dev = stage_pkg["stage_text_features"].float().to(device)

    labels = bclip_v1["labels"].numpy()
    splits = np.array(bclip_v1["splits"])
    clean_sem_mask = bclip_v1["clean_semantic_mask"].numpy()
    train_mask = (splits == "train") & clean_sem_mask
    test_mask = splits == "test"

    cls_to_id = {c: i for i, c in enumerate(ALL_CLASSES)}
    base_cls_ids = [cls_to_id[c] for c in BASE_CLASSES]
    meta_tr_ids = [cls_to_id[c] for c in META_TRAIN_CLASSES]
    meta_val_ids = [cls_to_id[c] for c in META_VAL_CLASSES]
    novel_cls_ids = [cls_to_id[c] for c in NOVEL_CLASSES]

    # Chọn hệ số pha trộn WiSE-FT rho in {0.15, 0.30, 0.50} CHỈ trên 4 lớp Meta-Val (5-shot)
    val_tr_pools = {c: np.where(train_mask & (labels == c))[0] for c in meta_val_ids}
    val_te_pools = {c: np.where(test_mask & (labels == c))[0] for c in meta_val_ids}
    val_eps, vq_idx, vq_y = prebuild_val_episodes(val_tr_pools, val_te_pools, meta_val_ids, k_shot=5, n_episodes=120, seed=905)
    vq_counts = np.bincount(vq_y, minlength=len(meta_val_ids)).astype(np.float64)
    val_txt_mean = F.normalize(stage_embs_dev[meta_val_ids].mean(dim=1), dim=-1)

    best_rho, best_v = 0.25, -1.0
    for rho in [0.15, 0.25, 0.40, 0.55]:
        z_mix = F.normalize((1.0 - rho) * z_froz + rho * z_ft, dim=-1)
        qz = z_mix[vq_idx]
        scs = []
        for vs_i, vs_y in val_eps:
            vp = torch.stack([F.normalize(z_mix[vs_i][vs_y == cl].mean(dim=0), dim=-1) for cl in range(len(meta_val_ids))], dim=0)
            hp = F.normalize(0.4 * vp + 0.6 * val_txt_mean, dim=-1)
            pr = (25.0 * (qz @ hp.T)).argmax(dim=-1).cpu().numpy()
            corr = np.bincount(vq_y[pr == vq_y], minlength=len(meta_val_ids))
            scs.append(float((corr / vq_counts).mean() * 100.0))
        mv = float(np.mean(scs))
        if mv > best_v:
            best_v = mv
            best_rho = rho

    print(f"[WiSE-FT] Chọn từ Meta-Val: rho = {best_rho} (Meta-Val 5-shot BAcc = {best_v:.2f}%)", flush=True)
    z_wise = F.normalize((1.0 - best_rho) * z_froz + best_rho * z_ft, dim=-1)
    hp_wise = tune_samp_hyperparams_on_meta_val(z_wise, stage_embs_dev, labels, train_mask, test_mask, cls_to_id, temp=25.0)

    summary_p = TABLES_V2_DIR / "step5_vit_finetune_summary.json"
    s5 = json.loads(summary_p.read_text(encoding="utf-8"))
    existing_rows = [r for r in s5["results"] if "WiSE-FT" not in r["method"]]

    wise_rows = []
    zs_w = evaluate_samp_variant_5seeds(
        z_wise, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
        k_shot=0, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED, cfg=hp_wise[0],
    )
    wise_rows.append(flatten_metric_row("Fine-Tuned ViT (Bảng riêng)", f"WiSE-FT (rho={best_rho}) + 4 giai đoạn (0-shot)", "1-crop-WiSE", 0, zs_w))

    for sh in SHOT_LIST:
        rw = evaluate_samp_variant_5seeds(
            z_wise, stage_embs_dev, labels, train_mask, test_mask, base_cls_ids, meta_tr_ids, novel_cls_ids,
            k_shot=sh, seeds=SEEDS_V2, n_episodes_per_seed=N_EPISODES_PER_SEED, cfg={"use_text": True, **hp_wise[sh]},
        )
        wise_rows.append(flatten_metric_row("Fine-Tuned ViT (Bảng riêng)", f"WiSE-FT (rho={best_rho}) + SAMP Đầy đủ", "1-crop-WiSE", sh, rw))

    all_s5_rows = existing_rows + wise_rows
    s5["wise_ft_rho_from_meta_val"] = best_rho
    s5["results"] = all_s5_rows
    summary_p.write_text(json.dumps(s5, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_p = TABLES_V2_DIR / "step5_vit_finetune_separate.csv"
    with open(csv_p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_s5_rows[0].keys()))
        w.writeheader()
        w.writerows(all_s5_rows)

    for r in wise_rows:
        print(f"  {r['method']:48s} | {r['shot']:2d}-shot | BAcc: {r['bacc_mean']:6.2f} ± {r['bacc_seed_std']:4.2f}% | AUROC: {r['auroc_mean']:6.2f}%", flush=True)


if __name__ == "__main__":
    main()
