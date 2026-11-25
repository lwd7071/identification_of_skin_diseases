# -*- coding: utf-8 -*-
r"""
BƯỚC 4 V2 (Kiểm chứng ngoài miền sơ bộ trên PAD-UFES-20 chia tách nghiêm ngặt theo patient_id):
  - Tải tuần tự từng mảnh Parquet của chehablab/PAD-UFES-20 (mở 100%, không cần token),
    lưu ảnh chuẩn 384px + metadata đầy đủ (patient_id, lesion_id, diagnostic, fitspatrick)
    rồi xóa ngay file tạm .parquet (chỉ tốn ~110 MB trên ổ D:\).
  - Ánh xạ rõ ràng 6 lớp tổn thương da chụp bằng điện thoại (ACK, BCC, MEL, NEV, SCC, SEK).
  - Chuyển giao Ngoài miền Thực sự (True Cross-Domain Transfer):
      + Dùng nguyên mô hình ResNet-50+ProtoNet và siêu tham số SAMP đã khóa trên SkinDisease Meta-Val
        (KHÔNG train lại, KHÔNG dò lại siêu tham số trên PAD-UFES-20).
      + Chia tách theo BỆNH NHÂN (patient_id): tập bệnh nhân lấy mẫu Support và tập bệnh nhân Query Test
        tách rời hoàn toàn 100% (assert overlap == 0 ở mọi seed & mọi episode).
  - Phân tích phụ công bằng theo nhóm tông màu da Fitzpatrick (I-II, III-IV, V-VI).
"""
import csv
import gc
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torchvision import models
from tqdm import tqdm

from step1_baselines_5seeds import ResNetProtoHeadV2, flatten_metric_row
from step2_multistage_and_multicrop import compute_samp_visual_logits, compute_stage_text_logits
from v2_config import (
    BIOMEDCLIP_MODEL_ID,
    CKPT_V2_DIR,
    N_EPISODES_PER_SEED,
    PAD_UFES_DIR,
    PreventWindowsSleep,
    SEEDS_V2,
    SHOT_LIST,
    TABLES_V2_DIR,
    record_or_verify_protected_files,
    summarize_5seed_episodes,
)

PAD_CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]

PAD_CLASS_MAPPING_INFO = {
    "ACK": {
        "full_name": "Actinic Keratosis (Dày sừng ánh sáng)",
        "skindisease_relation": "Tương ứng lớp Base 'Actinic_Keratosis' (khác thiết bị: ảnh điện thoại lâm sàng Brazil)",
        "simple_prompt": "this is a dermatological photograph of actinic keratosis",
        "single_clinical_prompt": "clinical smartphone photograph of actinic keratosis showing rough gritty scaly erythematous macule or papule on sun-exposed skin",
        "stages": [
            "erythematous inflamed actinic keratosis macule with tender pink-red base and fragile thin surface on sun-damaged skin",
            "solar keratosis lesion covered by adherent rough yellow-white or tan keratotic crust and sandpaper-like scale",
            "chronic hyperkeratotic actinic keratosis with hard dry thick adherent scale on photoaged forehead, nose, scalp, or forearm",
            "multiple flat to slightly raised scaly premalignant solar keratoses on chronically UV-exposed fair skin without pearly rolled border",
        ],
    },
    "BCC": {
        "full_name": "Basal Cell Carcinoma (Ung thư biểu mô tế bào đáy)",
        "skindisease_relation": "Lớp con chuyên biệt thuộc nhóm 'SkinCancer' (tách chi tiết 3 loại ung thư da)",
        "simple_prompt": "this is a dermatological photograph of basal cell carcinoma",
        "single_clinical_prompt": "clinical smartphone photograph of basal cell carcinoma showing pearly translucent papule or nodule with arborizing telangiectasia and central ulceration",
        "stages": [
            "nodular basal cell carcinoma presenting as shiny translucent pearly pink-flesh papule with surface oozing or pinpoint bleeding",
            "ulcerated rodent-ulcer basal cell carcinoma with rolled pearly border and central reddish-brown hemorrhagic crust",
            "superficial or pigmented basal cell carcinoma plaque with shiny thin border, focal brown-blue pigment flecks, and fine scaling",
            "slow-growing friable malignant skin lesion on nose, cheek, eyelid, ear, or neck showing prominent branching arborizing telangiectasia blood vessels",
        ],
    },
    "MEL": {
        "full_name": "Melanoma (U hắc tố ác tính)",
        "skindisease_relation": "Lớp con chuyên biệt thuộc nhóm 'SkinCancer' (tách chi tiết 3 loại ung thư da)",
        "simple_prompt": "this is a dermatological photograph of malignant melanoma",
        "single_clinical_prompt": "clinical smartphone photograph of cutaneous malignant melanoma showing asymmetry, irregular notched border, and multicolor black-brown pigmentation",
        "stages": [
            "invasive nodular or amelanotic melanoma presenting as rapidly growing asymmetrical dark-black or pink-red bleeding nodule",
            "ulcerated malignant melanoma plaque with focal hemorrhagic crusting and irregular Polychromatic brown-black-blue-pink zones",
            "superficial spreading or lentigo maligna melanoma macule with jagged notched borders, pigment variegation, and eccentric dark blotches",
            "atypical asymmetrical melanocytic malignancy fulfilling ABCDE criteria with diameter over 6mm, uneven border, and whitish-gray regression areas",
        ],
    },
    "NEV": {
        "full_name": "Melanocytic Nevus (Nốt ruồi sắc tố lành tính)",
        "skindisease_relation": "Tương ứng lớp Novel 'Moles' (cùng bản chất nốt ruồi sắc tố lành tính)",
        "simple_prompt": "this is a dermatological photograph of benign melanocytic nevus",
        "single_clinical_prompt": "clinical smartphone photograph of a benign melanocytic nevus showing symmetrical round uniformly tan-brown mole with smooth regular border",
        "stages": [
            "benign junctional or halo melanocytic nevus presenting as flat symmetrical round evenly brown macule on intact skin",
            "compound melanocytic nevus showing small dome-shaped uniformly light-to-dark brown papule with sharp regular border",
            "mature intradermal benign nevus appearing as soft flesh-colored or tan raised dome-shaped mole",
            "solitary small symmetrical benign pigmented mole with homogeneous color and smooth surface without ulceration or waxy keratin plugs",
        ],
    },
    "SCC": {
        "full_name": "Squamous Cell Carcinoma (Ung thư biểu mô tế bào vảy)",
        "skindisease_relation": "Lớp con chuyên biệt thuộc nhóm 'SkinCancer' (tách chi tiết 3 loại ung thư da)",
        "simple_prompt": "this is a dermatological photograph of cutaneous squamous cell carcinoma",
        "single_clinical_prompt": "clinical smartphone photograph of cutaneous squamous cell carcinoma showing firm indurated hyperkeratotic nodule or ulcerated scaly plaque",
        "stages": [
            "invasive ulcerated cutaneous squamous cell carcinoma presenting as friable erythematous fleshy tumor with purulent or bloody exudate",
            "crusted squamous cell carcinoma or keratoacanthoma dome-shaped nodule with central crater filled with thick yellow-brown keratin crust",
            "Bowen disease squamous cell carcinoma in situ or hyperkeratotic SCC plaque with thick rough adherent scale and firm indurated base",
            "rapidly enlarging tender keratotic ulcerated skin malignancy on sun-damaged ear, lip, scalp, hand dorsum, or forearm",
        ],
    },
    "SEK": {
        "full_name": "Seborrheic Keratosis (Dày sừng tiết bã)",
        "skindisease_relation": "Tương ứng lớp Novel 'Seborrh_Keratoses' (cùng bản chất dày sừng tiết bã lành tính)",
        "simple_prompt": "this is a dermatological photograph of seborrheic keratosis",
        "single_clinical_prompt": "clinical smartphone photograph of seborrheic keratosis showing well-demarcated waxy brown stuck-on verrucous plaque",
        "stages": [
            "irritated inflamed seborrheic keratosis with erythematous base and friable warty surface after clothing friction",
            "tan-brown waxy stuck-on seborrheic keratosis papule or plaque with greasy keratin plugs and rough crumbly surface",
            "mature dark-brown hyperkeratotic seborrheic keratosis plaque with sharp demarcated stuck-on border, fissures, and horn pseudocysts",
            "benign pasted-on dull waxy verrucous pigmented plaque on face, trunk, or limbs of older adults without arborizing vessels",
        ],
    },
}


def download_and_unpack_pad_ufes_20():
    """Tải từng shard Parquet của chehablab/PAD-UFES-20, giải nén ảnh 384px và xóa ngay shard tạm."""
    meta_csv = PAD_UFES_DIR / "pad_ufes20_clean_metadata.csv"
    img_dir = PAD_UFES_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if meta_csv.exists():
        with open(meta_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) >= 2200:
            print(f"[PAD-UFES-20] Đã có sẵn {len(rows)} ảnh sạch tại {PAD_UFES_DIR}", flush=True)
            return rows

    print("\n[PAD-UFES-20] Đang lấy danh sách Parquet shards từ HuggingFace (chehablab/PAD-UFES-20)...", flush=True)
    api_url = "https://huggingface.co/api/datasets/chehablab/PAD-UFES-20"
    with urllib.request.urlopen(urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=15) as r:
        repo_info = json.loads(r.read().decode("utf-8"))

    parquet_files = sorted(
        s["rfilename"] for s in repo_info.get("siblings", []) if s.get("rfilename", "").endswith(".parquet")
    )
    assert len(parquet_files) > 0, "Không tìm thấy file parquet trong chehablab/PAD-UFES-20"

    rows_saved = []
    seen_img_ids = set()

    for idx_s, rel_pq in enumerate(parquet_files, 1):
        dl_url = f"https://huggingface.co/datasets/chehablab/PAD-UFES-20/resolve/main/{rel_pq}"
        tmp_pq = PAD_UFES_DIR / "_tmp_shard.parquet"
        print(f"  -> [{idx_s}/{len(parquet_files)}] Đang tải {rel_pq}...", flush=True)
        req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_pq, "wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)

        table = pq.read_table(tmp_pq)
        pyd = table.to_pydict()
        n_rows = len(pyd["image_id"])

        for i in range(n_rows):
            img_id = str(pyd["image_id"][i]).strip()
            diag = str(pyd["diagnostic"][i]).strip().upper()
            pat_id = str(pyd["patient_id"][i]).strip()
            if diag not in PAD_CLASSES or not img_id or img_id in seen_img_ids:
                continue

            img_struct = pyd["image"][i]
            img_bytes = img_struct["bytes"] if isinstance(img_struct, dict) else img_struct
            with Image.open(io.BytesIO(img_bytes)) as im:
                rgb = im.convert("RGB")
                rgb.thumbnail((384, 384), Image.Resampling.LANCZOS)
                out_jpg = img_dir / f"{img_id}.jpg"
                rgb.save(out_jpg, format="JPEG", quality=92)

            seen_img_ids.add(img_id)
            rows_saved.append({
                "image_id": img_id,
                "rel_img": f"images/{img_id}.jpg",
                "patient_id": pat_id if pat_id else f"UNK_{img_id}",
                "lesion_id": str(pyd.get("lesion_id", [""] * n_rows)[i]).strip(),
                "diagnostic": diag,
                "fitspatrick": str(pyd.get("fitspatrick", [""] * n_rows)[i]).strip(),
                "region": str(pyd.get("region", [""] * n_rows)[i]).strip(),
                "age": str(pyd.get("age", [""] * n_rows)[i]).strip(),
                "biopsed": str(pyd.get("biopsed", [""] * n_rows)[i]).strip(),
            })

        del table, pyd
        gc.collect()
        if tmp_pq.exists():
            tmp_pq.unlink()

    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_saved[0].keys()))
        w.writeheader()
        w.writerows(rows_saved)

    print(f"-> Hoàn tất giải nén gọn {len(rows_saved)} ảnh PAD-UFES-20 (đã xóa toàn bộ shard tạm)!", flush=True)
    return rows_saved


@torch.no_grad()
def extract_pad_features(rows: list, device: torch.device) -> dict:
    cache_pt = PAD_UFES_DIR / "pad_ufes20_features.pt"
    if cache_pt.exists():
        return torch.load(cache_pt, map_location="cpu", weights_only=False)

    n = len(rows)
    use_amp = device.type == "cuda"

    def _iter_batches(tfm, bsz=64):
        def _ld(i):
            with Image.open(PAD_UFES_DIR / rows[i]["rel_img"]) as im:
                return tfm(im.convert("RGB")), i
        with ThreadPoolExecutor(max_workers=8) as pool:
            for st in range(0, n, bsz):
                bi = range(st, min(st + bsz, n))
                loaded = list(pool.map(_ld, bi))
                yield torch.stack([x[0] for x in loaded], dim=0), [x[1] for x in loaded]

    print("\n[PAD-UFES-20] Đang trích xuất đặc trưng ResNet-50 & BiomedCLIP...", flush=True)
    w_res = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=w_res)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    res_feats = torch.zeros((n, 2048), dtype=torch.float32)
    for imgs, idxs in tqdm(_iter_batches(w_res.transforms()), total=(n + 63) // 64, desc="PAD-ResNet50", ncols=90):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            res_feats[idxs] = resnet(imgs).float().cpu()
    del resnet
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()

    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID)
    model = model.to(device).eval()

    bclip_feats = torch.zeros((n, 512), dtype=torch.float32)
    for imgs, idxs in tqdm(_iter_batches(preprocess), total=(n + 63) // 64, desc="PAD-BiomedCLIP", ncols=90):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            emb = F.normalize(model.encode_image(imgs).float(), dim=-1)
        bclip_feats[idxs] = emb.cpu()

    simple_txts = [PAD_CLASS_MAPPING_INFO[c]["simple_prompt"] for c in PAD_CLASSES]
    single_clin_txts = [PAD_CLASS_MAPPING_INFO[c]["single_clinical_prompt"] for c in PAD_CLASSES]
    simple_embs = F.normalize(model.encode_text(tokenizer(simple_txts, context_length=256).to(device)).float(), dim=-1).cpu()
    single_clin_embs = F.normalize(model.encode_text(tokenizer(single_clin_txts, context_length=256).to(device)).float(), dim=-1).cpu()

    stage_embs = []
    for c in PAD_CLASSES:
        st_prompts = [f"clinical dermatology photograph showing {s}" for s in PAD_CLASS_MAPPING_INFO[c]["stages"]]
        e_st = F.normalize(model.encode_text(tokenizer(st_prompts, context_length=256).to(device)).float(), dim=-1)
        stage_embs.append(e_st.cpu())
    stage_embs = torch.stack(stage_embs, dim=0)  # [6, 4, 512]

    del model
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()

    pkg = {
        "resnet_features": res_feats,
        "biomedclip_features": bclip_feats,
        "simple_text_features": simple_embs,
        "single_clinical_features": single_clin_embs,
        "stage_text_features": stage_embs,
    }
    torch.save(pkg, cache_pt)
    return pkg


def make_patient_disjoint_split(rows: list, seed: int):
    """
    Chia danh sách BỆNH NHÂN (patient_id) thành 2 nửa độc lập (50% Support Pool Patients / 50% Query Test Patients)
    theo phân tầng nhãn chính của bệnh nhân -> đảm bảo 0% trùng patient_id giữa Support và Query!
    """
    rng = np.random.default_rng(seed)
    pat_to_indices = {}
    pat_to_main_cls = {}
    cls_to_id = {c: i for i, c in enumerate(PAD_CLASSES)}

    for idx, r in enumerate(rows):
        pid = r["patient_id"]
        pat_to_indices.setdefault(pid, []).append(idx)
        pat_to_main_cls[pid] = cls_to_id[r["diagnostic"]]

    cls_to_pats = {i: [] for i in range(len(PAD_CLASSES))}
    for pid, cid in pat_to_main_cls.items():
        cls_to_pats[cid].append(pid)

    sup_pats, qry_pats = set(), set()
    for cid, pids in cls_to_pats.items():
        pids_arr = np.array(sorted(pids))
        rng.shuffle(pids_arr)
        mid = len(pids_arr) // 2
        sup_pats.update(pids_arr[:mid].tolist())
        qry_pats.update(pids_arr[mid:].tolist())

    assert sup_pats.isdisjoint(qry_pats), "LỖI NGHIÊM TRỌNG: TRÙNG PATIENT_ID!"

    sup_mask = np.array([r["patient_id"] in sup_pats for r in rows], dtype=bool)
    qry_mask = np.array([r["patient_id"] in qry_pats for r in rows], dtype=bool)
    assert not np.any(sup_mask & qry_mask)
    return sup_mask, qry_mask, len(sup_pats), len(qry_pats)


def compute_metrics_np(probs: np.ndarray, y_true: np.ndarray):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    mf1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro") * 100.0
    except ValueError:
        auc = float("nan")
    return acc, bacc, mf1, auc, preds


def fitzpatrick_group(val_str: str) -> str:
    try:
        v = int(float(val_str))
        if v in (1, 2):
            return "Fitzpatrick I-II (Da sáng)"
        if v in (3, 4):
            return "Fitzpatrick III-IV (Da trung bình)"
        if v in (5, 6):
            return "Fitzpatrick V-VI (Da sẫm màu)"
    except Exception:
        pass
    return "Unknown"


def main():
    t0 = time.time()
    with PreventWindowsSleep():
        record_or_verify_protected_files("verify")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"=== BƯỚC 4 V2: KIỂM CHỨNG NGOÀI MIỀN SƠ BỘ TRÊN PAD-UFES-20 THEO PATIENT_ID ({device}) ===", flush=True)

        rows = download_and_unpack_pad_ufes_20()
        feat_pkg = extract_pad_features(rows, device)

        cls_to_id = {c: i for i, c in enumerate(PAD_CLASSES)}
        labels = np.array([cls_to_id[r["diagnostic"]] for r in rows], dtype=np.int64)
        fitz_groups = np.array([fitzpatrick_group(r["fitspatrick"]) for r in rows])

        # Nạp mô hình & thống kê đã khóa từ tập nguồn SkinDisease (KHÔNG dò lại trên PAD-UFES-20!)
        res_ckpts = torch.load(CKPT_V2_DIR / "resnet50_protonet_5seeds.pt", map_location="cpu", weights_only=False)
        samp_bundle = torch.load(CKPT_V2_DIR / "samp_v2_bundle.pt", map_location="cpu", weights_only=False)

        source_global_mean = samp_bundle["meta_train_global_mean"].to(device)      # [1, 512] từ SkinDisease Meta-Train
        source_raw_pool = samp_bundle["meta_train_raw_sample"].to(device)          # [M, 512] từ SkinDisease Meta-Train
        source_base_means = samp_bundle["base_cls_centered_means"].to(device)      # [16, 512] từ SkinDisease Base
        locked_hparams = samp_bundle["hparams_1crop"]

        res_raw_dev = feat_pkg["resnet_features"].float().to(device)
        bclip_raw_dev = F.normalize(feat_pkg["biomedclip_features"].float().to(device), dim=-1)
        simple_txt_dev = feat_pkg["simple_text_features"].float().to(device)
        single_clin_dev = feat_pkg["single_clinical_features"].float().to(device)
        stage_embs_dev = feat_pkg["stage_text_features"].float().to(device)

        # Chiếu ResNet-50 qua 5 checkpoint đã train trên SkinDisease
        res_z_by_seed, res_scale_by_seed = {}, {}
        for sd in SEEDS_V2:
            m = ResNetProtoHeadV2().to(device)
            m.load_state_dict(res_ckpts[str(sd)]["state_dict"])
            m.eval()
            with torch.no_grad():
                res_z_by_seed[sd] = m(res_raw_dev)
                res_scale_by_seed[sd] = float(m.scale.clamp(1.0, 40.0).item())

        # Tạo trước 5 phép chia bệnh nhân độc lập cho 5 seeds
        splits_by_seed = {}
        for sd in SEEDS_V2:
            sm, qm, n_sp, n_qp = make_patient_disjoint_split(rows, seed=sd)
            splits_by_seed[sd] = (sm, qm)
        print(f"-> Đã tạo 5 phép chia bệnh nhân độc lập (mỗi seed: ~{n_sp} bệnh nhân Support / ~{n_qp} bệnh nhân Query, overlap=0).", flush=True)

        cross_rows = []

        # 1. Đánh giá 0-shot trên tập Query Patients của từng seed (5 seeds)
        for m_name, mode_tag, txt_t, cfg_0 in [
            ("BiomedCLIP Zero-Shot (Chỉ tên bệnh)", "0-shot", simple_txt_dev.unsqueeze(1), {"tau_stage": 0.0, "prior_weight": 0.0}),
            ("BiomedCLIP Zero-Shot (Mô tả 1 câu)", "0-shot", single_clin_dev.unsqueeze(1), {"tau_stage": 0.0, "prior_weight": 0.0}),
            ("SAMP Zero-Shot (4 giai đoạn - khóa tham số)", "0-shot", stage_embs_dev, locked_hparams[0]),
        ]:
            acc_s, bacc_s, mf1_s, auc_s = {}, {}, {}, {}
            for sd in SEEDS_V2:
                _, qm = splits_by_seed[sd]
                q_idx = np.where(qm)[0]
                qy = labels[q_idx]
                t_logits = compute_stage_text_logits(
                    bclip_raw_dev[q_idx], txt_t, source_raw_pool,
                    tau_stage=cfg_0["tau_stage"], prior_weight=cfg_0["prior_weight"], temp=25.0,
                )
                probs = F.softmax(t_logits, dim=-1).cpu().numpy()
                acc, bacc, mf1, auc, _ = compute_metrics_np(probs, qy)
                acc_s[sd], bacc_s[sd], mf1_s[sd], auc_s[sd] = [acc], [bacc], [mf1], [auc]

            stats = {
                "acc": summarize_5seed_episodes(acc_s),
                "bacc": summarize_5seed_episodes(bacc_s),
                "mf1": summarize_5seed_episodes(mf1_s),
                "auroc": summarize_5seed_episodes(auc_s),
            }
            cross_rows.append(flatten_metric_row("PAD-UFES-20 Cross-Domain", m_name, "patient-disjoint", 0, stats))

        # 2. Đánh giá Few-Shot (1, 5, 10-shot) chia tách theo patient_id + thống kê Fitzpatrick ở 5-shot
        fitz_breakdown_5shot = {
            "Fitzpatrick I-II (Da sáng)": {"ResNet-50 + ProtoNet": [], "BiomedCLIP + ProtoNet": [], "SAMP (Đề xuất)": []},
            "Fitzpatrick III-IV (Da trung bình)": {"ResNet-50 + ProtoNet": [], "BiomedCLIP + ProtoNet": [], "SAMP (Đề xuất)": []},
            "Fitzpatrick V-VI (Da sẫm màu)": {"ResNet-50 + ProtoNet": [], "BiomedCLIP + ProtoNet": [], "SAMP (Đề xuất)": []},
        }

        for sh in SHOT_LIST:
            hp = locked_hparams[sh]
            c_beta = hp["center_beta"]
            bclip_samp_space = F.normalize(bclip_raw_dev - c_beta * source_global_mean, dim=-1)

            methods_sh = [
                ("ResNet-50 + ProtoNet (Trained on SkinDisease)", "resnet"),
                ("BiomedCLIP + ProtoNet (Visual Only)", "bclip_vis"),
                ("SAMP (4 giai đoạn + Centering + Calibration)", "samp"),
            ]
            for m_label, m_code in methods_sh:
                acc_s, bacc_s, mf1_s, auc_s = {}, {}, {}, {}
                for sd in SEEDS_V2:
                    sm, qm = splits_by_seed[sd]
                    q_idx = np.where(qm)[0]
                    qy = labels[q_idx]
                    q_fitz = fitz_groups[q_idx]
                    sup_pools = {c: np.where(sm & (labels == c))[0] for c in range(len(PAD_CLASSES))}
                    rng = np.random.default_rng(sd + sh * 29)

                    if m_code == "samp":
                        txt_logits_q = compute_stage_text_logits(
                            bclip_raw_dev[q_idx], stage_embs_dev, source_raw_pool,
                            tau_stage=hp["tau_stage"], prior_weight=hp["prior_weight"], temp=25.0,
                        )

                    accs, baccs, mf1s, aucs = [], [], [], []
                    agg_probs_seed = np.zeros((len(qy), len(PAD_CLASSES)), dtype=np.float64)

                    for _ in range(N_EPISODES_PER_SEED):
                        chosen_by_c = [rng.choice(sup_pools[c], size=min(sh, len(sup_pools[c])), replace=False) for c in range(len(PAD_CLASSES))]
                        if m_code == "resnet":
                            z_sd = res_z_by_seed[sd]
                            sc_sd = res_scale_by_seed[sd]
                            protos = torch.stack([F.normalize(z_sd[ch].mean(dim=0), dim=-1) for ch in chosen_by_c], dim=0)
                            probs = F.softmax(sc_sd * (z_sd[q_idx] @ protos.T), dim=-1).cpu().numpy()
                        elif m_code == "bclip_vis":
                            protos = torch.stack([F.normalize(bclip_raw_dev[ch].mean(dim=0), dim=-1) for ch in chosen_by_c], dim=0)
                            probs = F.softmax(25.0 * (bclip_raw_dev[q_idx] @ protos.T), dim=-1).cpu().numpy()
                        else:
                            sup_vecs = [bclip_samp_space[ch] for ch in chosen_by_c]
                            v_logits = compute_samp_visual_logits(
                                bclip_samp_space[q_idx], sup_vecs, source_base_means,
                                calib_gamma=hp["calib_gamma"], tip_lambda=hp["tip_lambda"], tip_beta=3.5, temp=25.0,
                            )
                            comb = hp["alpha_vis"] * v_logits + (1.0 - hp["alpha_vis"]) * txt_logits_q
                            probs = F.softmax(comb, dim=-1).cpu().numpy()

                        acc, bacc, mf1, auc, _ = compute_metrics_np(probs, qy)
                        accs.append(acc)
                        baccs.append(bacc)
                        mf1s.append(mf1)
                        aucs.append(auc)
                        agg_probs_seed += probs

                    acc_s[sd], bacc_s[sd], mf1_s[sd], auc_s[sd] = accs, baccs, mf1s, aucs

                    if sh == 5:
                        seed_preds = (agg_probs_seed / N_EPISODES_PER_SEED).argmax(axis=1)
                        short_key = "ResNet-50 + ProtoNet" if m_code == "resnet" else ("BiomedCLIP + ProtoNet" if m_code == "bclip_vis" else "SAMP (Đề xuất)")
                        for fg in fitz_breakdown_5shot:
                            f_mask = q_fitz == fg
                            if np.sum(f_mask) >= 10:
                                fitz_breakdown_5shot[fg][short_key].append(float(balanced_accuracy_score(qy[f_mask], seed_preds[f_mask]) * 100.0))

                stats = {
                    "acc": summarize_5seed_episodes(acc_s),
                    "bacc": summarize_5seed_episodes(bacc_s),
                    "mf1": summarize_5seed_episodes(mf1_s),
                    "auroc": summarize_5seed_episodes(auc_s),
                }
                cross_rows.append(flatten_metric_row("PAD-UFES-20 Cross-Domain", m_label, "patient-disjoint", sh, stats))

        csv_path = TABLES_V2_DIR / "step4_pad_ufes20_crossdomain.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
            w.writeheader()
            w.writerows(cross_rows)

        fitz_summary = {
            fg: {
                mk: {"mean": round(float(np.mean(vs)), 2), "std": round(float(np.std(vs, ddof=1)), 2) if len(vs) > 1 else 0.0}
                for mk, vs in mdict.items() if len(vs) > 0
            }
            for fg, mdict in fitz_breakdown_5shot.items()
        }

        with open(TABLES_V2_DIR / "step4_pad_ufes20_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "note": "Kiểm chứng ngoài miền sơ bộ (6 lớp tổn thương da chụp bằng điện thoại, chia tách theo patient_id, không tinh chỉnh lại siêu tham số).",
                    "total_images": len(rows),
                    "unique_patients": len({r['patient_id'] for r in rows}),
                    "class_mapping": {k: {"full_name": v["full_name"], "relation": v["skindisease_relation"]} for k, v in PAD_CLASS_MAPPING_INFO.items()},
                    "results": cross_rows,
                    "fitzpatrick_5shot_bacc": fitz_summary,
                    "elapsed_seconds": round(time.time() - t0, 1),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        verify_after = record_or_verify_protected_files("verify")
        print("\n=== KẾT QUẢ BƯỚC 4 (PAD-UFES-20 CROSS-DOMAIN SƠ BỘ, CHIA THEO PATIENT_ID, 5 SEEDS) ===", flush=True)
        for r in cross_rows:
            print(
                f"  {r['method']:48s} | {r['shot']:2d}-shot | BAcc: {r['bacc_mean']:6.2f} ± {r['bacc_seed_std']:4.2f}% "
                f"| F1: {r['mf1_mean']:6.2f}% | AUROC: {r['auroc_mean']:6.2f}%",
                flush=True,
            )
        print(f"-> Phân rã Fitzpatrick (5-shot BAcc): {fitz_summary}", flush=True)
        print(f"[OK] Lưu tại: {csv_path} | SHA-256: {verify_after['status']} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
