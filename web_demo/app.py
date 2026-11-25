# -*- coding: utf-8 -*-
"""
FastAPI Web Demo — Phân loại ảnh bệnh da liễu ít mẫu theo Đa giai đoạn lâm sàng (SAMP + BiomedCLIP).
Chạy server:
  & "d:\\NCKH\\New folder\\.venv\\Scripts\\python.exe" -m uvicorn app:app --app-dir "d:\\NCKH\\New folder\\web_demo" --host 127.0.0.1 --port 8000
"""
import csv
import io
import json
import os
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"
for _cd in (CACHE_DIR / "hf" / "hub", CACHE_DIR / "torch", CACHE_DIR / "tmp"):
    _cd.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hf" / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["TEMP"] = str(CACHE_DIR / "tmp")

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DATA_DIR = Path(os.environ.get("SKIN_DATA_DIR", str(ROOT_DIR.parent / "Data" / "SkinDisease")))
BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class DermFewShotEngine:
    """Bộ máy suy luận SAMP nạp 1 lần trên GPU/CPU, hỗ trợ phân tích giai đoạn & đăng ký lớp mới tức thì."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        # Nạp mô hình gốc BiomedCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
        self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID)
        self.model = self.model.to(self.device).eval()

        norm_op = next(t for t in self.preprocess.transforms if isinstance(t, transforms.Normalize))
        self.five_crop_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=norm_op.mean, std=norm_op.std)(transforms.ToTensor()(c)) for c in crops])),
        ])

        # Nạp từ điển giai đoạn bệnh & gói trọng số SAMP V2 (hoặc fallback sang V1 nếu cần)
        stages_path = ROOT_DIR / "src_v2" / "clinical_stages_v2.json"
        self.stages_db = json.loads(stages_path.read_text(encoding="utf-8"))
        self.stage_defs = self.stages_db["stage_definitions"]

        v2_bundle_path = ROOT_DIR / "outputs_v2" / "checkpoints" / "samp_v2_bundle.pt"
        bclip_v1_path = ROOT_DIR / "outputs" / "features" / "biomedclip_features.pt"
        v1_concepts_json = json.loads((ROOT_DIR / "src" / "clinical_concepts.json").read_text(encoding="utf-8"))
        self.concept_meta = v1_concepts_json["concepts_list"]

        if v2_bundle_path.exists():
            bundle = torch.load(v2_bundle_path, map_location="cpu", weights_only=False)
            self.base_classes = list(bundle["classes"])
            self.stage_text_embs = bundle["stage_text_features"].float().to(self.device)  # [22, 4, 512]
            self.concept_embs = F.normalize(bundle["concept_features"].float().to(self.device), dim=-1)  # [18, 512]
            self.global_mean = bundle["meta_train_global_mean"].float().to(self.device)  # [1, 512]
            self.bg_pool = bundle["meta_train_raw_sample"].float().to(self.device)  # [M, 512]
            self.cls_vis_protos = F.normalize(bundle["all_cls_centered_means"].float().to(self.device), dim=-1)  # [22, 512]
            self.base_only_protos = F.normalize(bundle["base_cls_centered_means"].float().to(self.device), dim=-1)
            hp5 = bundle["hparams_1crop"][5]
            self.center_beta = float(hp5["center_beta"])
            # Trong chế độ phân loại hợp nhất 22+ lớp trên Web Demo, cân bằng 55% nguyên mẫu ảnh + 45% đa giai đoạn văn bản
            # để vừa giữ giải thích giai đoạn bệnh, vừa đảm bảo lớp mới đăng ký từ 1-5 ảnh mẫu nhận diện chính xác ngay lập tức
            self.alpha_vis = 0.55
            self.tau_stage = float(hp5["tau_stage"])
            self.prior_weight = float(hp5["prior_weight"])
            self.calib_gamma = 0.15
        else:
            # Fallback an toàn nếu chạy thử trước khi Bước 2B kết thúc
            v1_pkg = torch.load(bclip_v1_path, map_location="cpu", weights_only=False)
            self.base_classes = list(self.stages_db["classes"].keys())
            self.concept_embs = F.normalize(v1_pkg["concept_features"].float().to(self.device), dim=-1)
            img_all = F.normalize(v1_pkg["image_features"].float(), dim=-1)
            labels_all = v1_pkg["labels"].numpy()
            splits_all = np.array(v1_pkg["splits"])
            tr_m = splits_all == "train"
            self.global_mean = img_all[tr_m].mean(dim=0, keepdim=True).to(self.device)
            self.bg_pool = img_all[tr_m][:512].to(self.device)
            self.center_beta = 0.6
            z_cent = F.normalize(img_all.to(self.device) - self.center_beta * self.global_mean, dim=-1)
            self.cls_vis_protos = torch.stack([F.normalize(z_cent[tr_m & (labels_all == i)].mean(dim=0), dim=-1) for i in range(len(self.base_classes))], dim=0)
            self.base_only_protos = self.cls_vis_protos[:16]
            self.alpha_vis = 0.5
            self.tau_stage = 0.5
            self.prior_weight = 0.5
            self.calib_gamma = 0.2
            st_list = []
            for cname in self.base_classes:
                txts = [f"clinical dermatology photograph showing {self.stages_db['classes'][cname]['stages'][sd['id']]}" for sd in self.stage_defs]
                tok = self.tokenizer(txts, context_length=256).to(self.device)
                with torch.no_grad():
                    st_list.append(F.normalize(self.model.encode_text(tok).float(), dim=-1))
            self.stage_text_embs = torch.stack(st_list, dim=0)

        # Bộ nhớ lưu các lớp bệnh mới do người dùng đăng ký Few-Shot trực tiếp trên Web
        self.enrolled_classes = []  # list of dicts: {id, vi_name, shot_count, vis_proto, stage_emb}
        self.sample_items = self._load_novel_samples()

    def _load_novel_samples(self) -> list:
        manifest_p = ROOT_DIR / "outputs" / "clean_manifest.csv"
        novel_targets = ["Seborrh_Keratoses", "Moles", "Sun_Sunlight_Damage", "Lupus", "Rosacea", "Candidiasis"]
        candidates = {c: [] for c in novel_targets}
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["split"] == "test" and r["class"] in candidates:
                        img_p = DATA_DIR / r["rel_path"]
                        if img_p.exists() and len(candidates[r["class"]]) < 8:
                            candidates[r["class"]].append(img_p)
        # Chọn mẫu test điển hình rõ nét cho từng lớp Novel phục vụ nút bấm thử nhanh trên giao diện
        results = []
        cls_to_idx = {c: i for i, c in enumerate(self.base_classes)}
        for c in novel_targets:
            if not candidates[c]:
                continue
            best_p = candidates[c][0]
            best_sc = -1e9
            cid = cls_to_idx[c]
            for p_cand in candidates[c]:
                with Image.open(p_cand) as im:
                    emb = self.encode_pil_1crop(im)
                    cent = F.normalize(emb - self.center_beta * self.global_mean, dim=-1)
                    sc = float((cent @ self.cls_vis_protos[cid]).item())
                    if sc > best_sc:
                        best_sc = sc
                        best_p = p_cand
            results.append({
                "class_id": c,
                "vi_name": self.stages_db["classes"][c]["vi_name"],
                "abs_path": str(best_p),
            })
        return results

    @torch.no_grad()
    def encode_pil_1crop(self, img: Image.Image) -> torch.Tensor:
        rgb = img.convert("RGB")
        t_img = self.preprocess(rgb).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
            emb = F.normalize(self.model.encode_image(t_img).float(), dim=-1)
        return emb  # [1, 512]

    @torch.no_grad()
    def predict_image(self, img: Image.Image) -> dict:
        q_raw = self.encode_pil_1crop(img)  # [1, 512]
        q_cent = F.normalize(q_raw - self.center_beta * self.global_mean, dim=-1)  # [1, 512]

        all_ids = list(self.base_classes) + [c["id"] for c in self.enrolled_classes]
        all_vi = [self.stages_db["classes"][c]["vi_name"] for c in self.base_classes] + [c["vi_name"] for c in self.enrolled_classes]

        if self.enrolled_classes:
            ext_vis = torch.stack([c["vis_proto"] for c in self.enrolled_classes], dim=0)
            ext_stg = torch.stack([c["stage_emb"] for c in self.enrolled_classes], dim=0)
            vis_protos = torch.cat([self.cls_vis_protos, ext_vis], dim=0)
            stg_embs = torch.cat([self.stage_text_embs, ext_stg], dim=0)
        else:
            vis_protos = self.cls_vis_protos
            stg_embs = self.stage_text_embs

        temp = 25.0
        vis_logits = temp * (q_cent @ vis_protos.T)  # [1, C]

        raw_sim = torch.einsum("nd,csd->ncs", q_raw, stg_embs)  # [1, C, 4]
        bg_sim = torch.einsum("md,csd->mcs", self.bg_pool, stg_embs).mean(dim=0, keepdim=True)  # [1, C, 4]
        adj_sim = raw_sim - self.prior_weight * bg_sim
        scaled_stg = temp * adj_sim
        if self.tau_stage > 1e-4:
            txt_logits = (torch.logsumexp(self.tau_stage * scaled_stg, dim=-1) - np.log(4.0)) / self.tau_stage
        else:
            txt_logits = scaled_stg.mean(dim=-1)

        total_logits = self.alpha_vis * vis_logits + (1.0 - self.alpha_vis) * txt_logits  # [1, C]
        probs = F.softmax(total_logits.squeeze(0), dim=-1).cpu().numpy()

        top_indices = np.argsort(-probs)[:5]
        predictions = []
        for rank_i in top_indices:
            stg_scores = F.softmax(scaled_stg[0, rank_i], dim=-1).cpu().numpy()
            best_stg_idx = int(stg_scores.argmax())
            stage_breakdown = [
                {
                    "stage_id": self.stage_defs[s_i]["id"],
                    "vi_label": self.stage_defs[s_i]["vi_label"],
                    "weight_pct": round(float(stg_scores[s_i]) * 100.0, 1),
                }
                for s_i in range(len(self.stage_defs))
            ]
            predictions.append({
                "class_id": str(all_ids[int(rank_i)]),
                "vi_name": str(all_vi[int(rank_i)]),
                "is_custom_enrolled": bool(int(rank_i) >= len(self.base_classes)),
                "probability_pct": round(float(probs[int(rank_i)]) * 100.0, 2),
                "matched_stage": str(self.stage_defs[best_stg_idx]["vi_label"]),
                "stage_breakdown": stage_breakdown,
            })

        # Tính Top-5 dấu hiệu hình thái lâm sàng SKINCON trên ảnh
        con_sims = (q_raw.squeeze(0) @ self.concept_embs.T).cpu().numpy()
        top_con_idx = np.argsort(-con_sims)[:5]
        concepts_detected = [
            {
                "id": self.concept_meta[ci]["id"],
                "vi": self.concept_meta[ci]["vi"],
                "en": self.concept_meta[ci]["en"],
                "cosine_score": round(float(con_sims[ci]), 3),
            }
            for ci in top_con_idx
        ]

        top1_p = float(probs[int(top_indices[0])])
        return {
            "top_predictions": predictions,
            "detected_skincon_concepts": concepts_detected,
            "low_confidence_warning": bool(top1_p < 0.35),
            "device": str(self.device),
            "total_active_classes": int(len(all_ids)),
        }

    @torch.no_grad()
    def enroll_new_disease(self, class_id: str, vi_name: str, clinical_description: str, images: List[Image.Image]) -> dict:
        if len(images) == 0:
            raise ValueError("Cần ít nhất 1 ảnh mẫu để đăng ký lớp bệnh mới.")

        sup_raw = torch.cat([self.encode_pil_1crop(im) for im in images], dim=0)  # [K, 512]
        sup_cent = F.normalize(sup_raw - self.center_beta * self.global_mean, dim=-1)  # [K, 512]
        emp_p = F.normalize(sup_cent.mean(dim=0, keepdim=True), dim=-1)  # [1, 512]

        # Hiệu chỉnh phân phối Empirical-Bayes 1/K mượn từ top-3 lớp Base gần nhất
        k_shots = sup_cent.size(0)
        sim_b = emp_p @ self.base_only_protos.T
        top_v, top_i = torch.topk(sim_b, k=3, dim=-1)
        w_b = F.softmax(top_v * 15.0, dim=-1)
        neigh = F.normalize((w_b.unsqueeze(-1) * self.base_only_protos[top_i]).sum(dim=1), dim=-1)
        eff_g = self.calib_gamma / float(k_shots)
        vis_proto = F.normalize((1.0 - eff_g) * emp_p + eff_g * neigh, dim=-1).squeeze(0)

        base_desc = clinical_description.strip() or f"cutaneous skin lesion of {class_id.replace('_', ' ')}"
        stage_prompts = [
            f"clinical dermatology photograph showing acute erythematous or exudative stage of {base_desc}",
            f"clinical dermatology photograph showing subacute crusted or scaling stage of {base_desc}",
            f"clinical dermatology photograph showing chronic dry hyperkeratotic or pigmented stage of {base_desc}",
            f"clinical dermatology photograph showing secondary excoriated or characteristic anatomical presentation of {base_desc}",
        ]
        tok = self.tokenizer(stage_prompts, context_length=256).to(self.device)
        stg_emb = F.normalize(self.model.encode_text(tok).float(), dim=-1)  # [4, 512]

        # Ghi đè nếu trùng ID hoặc thêm mới
        self.enrolled_classes = [c for c in self.enrolled_classes if c["id"] != class_id]
        self.enrolled_classes.append({
            "id": class_id,
            "vi_name": vi_name,
            "shot_count": k_shots,
            "vis_proto": vis_proto,
            "stage_emb": stg_emb,
        })
        return {"enrolled_id": class_id, "vi_name": vi_name, "shot_count": k_shots, "total_custom_classes": len(self.enrolled_classes)}


app = FastAPI(title="NCKH 2027 - Phân loại bệnh da liễu ít mẫu (SAMP + BiomedCLIP)")
engine: Optional[DermFewShotEngine] = None


def get_engine() -> DermFewShotEngine:
    global engine
    if engine is None:
        engine = DermFewShotEngine()
    return engine


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/api/classes")
def list_classes():
    eng = get_engine()
    built_in = [
        {"id": c, "vi_name": eng.stages_db["classes"][c]["vi_name"], "type": "novel" if i >= 16 else "base"}
        for i, c in enumerate(eng.base_classes)
    ]
    custom = [{"id": c["id"], "vi_name": c["vi_name"], "type": "custom_few_shot", "shots": c["shot_count"]} for c in eng.enrolled_classes]
    return {"built_in_count": len(built_in), "custom_count": len(custom), "built_in": built_in, "custom": custom, "sample_tests": eng.sample_items}


@app.get("/api/sample_image/{idx}")
def get_sample_image(idx: int):
    eng = get_engine()
    if idx < 0 or idx >= len(eng.sample_items):
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu")
    return FileResponse(eng.sample_items[idx]["abs_path"])


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải ảnh hợp lệ.")
    return get_engine().predict_image(img)


@app.post("/api/enroll")
async def enroll_endpoint(
    class_id: str = Form(...),
    vi_name: str = Form(...),
    clinical_description: str = Form(""),
    files: List[UploadFile] = File(...),
):
    pil_imgs = []
    for uf in files[:10]:
        b = await uf.read()
        try:
            pil_imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
        except Exception:
            continue
    if not pil_imgs:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh mẫu hợp lệ nào.")
    clean_id = class_id.strip().replace(" ", "_")
    return get_engine().enroll_new_disease(clean_id, vi_name.strip(), clinical_description, pil_imgs)


@app.post("/api/reset_enrolled")
def reset_enrolled():
    eng = get_engine()
    eng.enrolled_classes.clear()
    return {"status": "reset_ok", "custom_count": 0}
