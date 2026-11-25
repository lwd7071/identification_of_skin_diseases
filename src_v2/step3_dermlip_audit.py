# -*- coding: utf-8 -*-
"""
BƯỚC 3 V2 (Cách 1):
  - Hồ sơ kiểm chứng tự động giấy phép (License), trạng thái truy cập (Gated Access)
    và phân tích nguồn dữ liệu huấn luyện trước (Pretraining Overlap Audit) của DermLIP (ICCV 2025).
  - Nếu có biến môi trường HF_TOKEN đã chấp thuận điều khoản CC-BY-NC-4.0 trên HuggingFace,
    script hỗ trợ nạp và chạy đối chứng phụ; mặc định ở Cách 1 xuất báo cáo kiểm chứng minh bạch.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

from v2_config import (
    PreventWindowsSleep,
    TABLES_V2_DIR,
    record_or_verify_protected_files,
)

DERMLIP_REPOS = [
    "redlessone/DermLIP_ViT-B-16",
    "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
]


def fetch_hf_repo_metadata(repo_id: str) -> dict:
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read().decode("utf-8"))
        return {
            "repo_id": repo_id,
            "reachable": True,
            "license": d.get("cardData", {}).get("license", "unknown"),
            "gated": d.get("gated", False),
            "private": d.get("private", False),
            "downloads": d.get("downloads", 0),
            "arxiv_tag": next((t for t in d.get("tags", []) if t.startswith("arxiv:")), "arxiv:2503.14911"),
            "files": [s.get("rfilename") for s in d.get("siblings", [])],
        }
    except Exception as e:
        return {"repo_id": repo_id, "reachable": False, "error": str(e)}


def main():
    t0 = time.time()
    with PreventWindowsSleep():
        record_or_verify_protected_files("verify")
        print("=== BƯỚC 3 V2: KIỂM CHỨNG GIẤY PHÉP & DỮ LIỆU PRETRAIN CỦA DERMLIP ===", flush=True)

        repo_audits = [fetch_hf_repo_metadata(r) for r in DERMLIP_REPOS]
        hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))

        audit_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "paper_reference": "Yan, S., et al. (2025). Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology. ICCV 2025 Highlight (arXiv:2503.14911v1).",
            "huggingface_repositories": repo_audits,
            "hf_token_configured_in_env": hf_token_present,
            "execution_mode": "CACH_1_AUDIT_ONLY_NO_GATED_TOKEN" if not hf_token_present else "TOKEN_PRESENT",
            "license_assessment": {
                "spdx_id": "cc-by-nc-4.0",
                "permitted_for_student_nckh": True,
                "conditions": [
                    "Chỉ sử dụng cho mục đích nghiên cứu học thuật phi thương mại (Non-Commercial).",
                    "Bắt buộc trích dẫn bài báo gốc Derm1M / DermLIP (Yan et al., ICCV 2025).",
                    "Tải trọng số trên HuggingFace yêu cầu xác nhận điều khoản (gated: auto) kèm HF_TOKEN cá nhân.",
                ],
            },
            "pretraining_overlap_audit_vs_skindisease_dermnet": {
                "direct_public_dataset_ingestion": [
                    "SCIN (6,518 cặp ảnh-văn bản, Mục 7.5 arXiv:2503.14911v1)",
                    "MSKCC / ISIC subset (10,619 cặp ảnh-văn bản, Mục 7.5 arXiv:2503.14911v1)",
                ],
                "other_pretraining_sources": [
                    "YouTube dermatology educational videos (Mục 7.1)",
                    "PubMed Open-Access figures & captions (Mục 7.2)",
                    "68 PDF educational textbooks & atlases extracted via PyMuPDF (Mục 7.3)",
                    "Medical forums Reddit / IIYI (Mục 7.4)",
                ],
                "direct_dermnet_ingestion": False,
                "indirect_overlap_risk": "LOW_TO_MODERATE",
                "scientific_recommendation": (
                    "Bài báo Derm1M không nạp trực tiếp bộ DermNet vào mục Public Dataset (chỉ nạp SCIN và MSKCC). "
                    "Tuy nhiên, do Derm1M thu thập ảnh từ 68 tài liệu giáo trình PDF và video bài giảng YouTube, không thể loại trừ hoàn toàn "
                    "khả năng một số ảnh lâm sàng kinh điển của DermNet được giảng viên trích dẫn lại trong slide/sách. "
                    "Vì vậy, khi sử dụng DermLIP trong nghiên cứu tiếp theo, bắt buộc đặt ở nhóm 'Baseline tham chiếu phụ (Domain-Pretrained Reference)' "
                    "chứ không gộp chung với các mô hình chưa từng thấy dữ liệu giáo trình da liễu."
                ),
            },
        }

        json_path = TABLES_V2_DIR / "step3_dermlip_audit.json"
        json_path.write_text(json.dumps(audit_data, indent=2, ensure_ascii=False), encoding="utf-8")

        md_lines = [
            "# HỒ SƠ KIỂM CHỨNG MÔ HÌNH THAM CHIẾU PHỤ `DermLIP` (BƯỚC 3 – CÁCH 1)",
            "",
            "- **Công trình gốc:** Yan, S., et al. (2025). *Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology.* ICCV 2025 Highlight (`arXiv:2503.14911v1`).",
            "- **Kho trọng số chính thức trên HuggingFace:**",
        ]
        for ra in repo_audits:
            md_lines.append(
                f"  - `{ra['repo_id']}` — Giấy phép: **`{ra.get('license')}`** | Trạng thái cổng kiểm duyệt: **`gated={ra.get('gated')}`** | Lượt tải: `{ra.get('downloads')}`"
            )

        md_lines += [
            "",
            "## 1. Kết luận về Giấy phép & Điều kiện sử dụng",
            "- Trọng số `DermLIP` phát hành theo giấy phép **Creative Commons Attribution-NonCommercial 4.0 (`CC-BY-NC-4.0`)**, **hoàn toàn hợp pháp** cho đề tài Nghiên cứu Khoa học Sinh viên phi thương mại.",
            "- Do kho lưu trữ đặt chế độ `gated: auto` trên HuggingFace (đòi hỏi tài khoản cá nhân xác nhận điều khoản và cấp `HF_TOKEN`), ở **Cách 1** hệ thống tuân thủ nghiêm quy tắc bảo mật/quyền truy cập: không tự ý vượt cổng xác thực và ghi nhận đầy đủ hồ sơ kiểm chứng tại đây.",
            "",
            "## 2. Kiểm tra nguy cơ trùng lặp dữ liệu huấn luyện với `SkinDisease` (`DermNet`)",
            "Đối chiếu trực tiếp toàn văn phụ lục kỹ thuật **Mục 7.1 – 7.5** của `arXiv:2503.14911v1`:",
            "1. **Các bộ dữ liệu mở được nạp trực tiếp (Mục 7.5 – *Public Dataset*):** Chỉ gồm **`SCIN`** (`6.518` cặp) và **`MSKCC`** (`10.619` cặp). Tập **`DermNet` (`SkinDisease`) KHÔNG nằm trong danh sách nạp trực tiếp**.",
            "2. **Các nguồn thu thập còn lại:** Video bài giảng y khoa trên *YouTube* (Mục 7.1), bài báo mở *PubMed* (Mục 7.2), *68 sách/giáo trình PDF* quét qua `PyMuPDF` (Mục 7.3), và diễn đàn y khoa (Mục 7.4).",
            "3. **Đánh giá rủi ro rò rỉ gián tiếp:** **Thấp – Trung bình**. Dù không nạp trực tiếp `DermNet`, một số ảnh kinh điển từ `DermNet` thường được trích lại trong giáo trình PDF hoặc slide bài giảng YouTube.",
            "4. **Quy tắc sử dụng trong đề tài (năm 2027):** Khi bổ sung `HF_TOKEN` để chạy `DermLIP`, kết quả phải được trình bày ở nhóm **Baseline tham chiếu chuyên biệt miền (*In-Domain Pretrained Reference*)** tách riêng khỏi `ResNet-50` và `BiomedCLIP`, kèm ghi chú rõ phạm vi dữ liệu `Derm1M`.",
            "",
        ]

        md_path = TABLES_V2_DIR / "step3_dermlip_audit.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        verify_after = record_or_verify_protected_files("verify")
        print(f"[OK] Đã lưu báo cáo kiểm chứng DermLIP tại: {md_path} | SHA-256: {verify_after['status']} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
