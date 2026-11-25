# -*- coding: utf-8 -*-
"""
Bước 1: Kiểm tra chất lượng ảnh, khử trùng lặp rò rỉ dữ liệu (train-test deduplication)
và xuất báo cáo thống kê + biểu đồ EDA cho tập dữ liệu SkinDisease (22 lớp).
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    ALL_CLASSES,
    BASE_CLASSES,
    DATA_DIR,
    FIG_DIR,
    META_TRAIN_CLASSES,
    META_VAL_CLASSES,
    NOVEL_CLASSES,
    OUT_DIR,
)


def compute_image_info(item):
    """Đọc 1 ảnh, kiểm tra lỗi, tính MD5 và 64-bit dHash."""
    split, cls_name, fpath = item
    rel_path = str(fpath.relative_to(DATA_DIR))
    try:
        raw_bytes = fpath.read_bytes()
        md5_hex = hashlib.md5(raw_bytes).hexdigest()
        with Image.open(fpath) as img:
            img.verify()
        with Image.open(fpath) as img:
            img_rgb = img.convert("RGB")
            w, h = img_rgb.size
            if w < 32 or h < 32:
                return {"ok": False, "reason": f"too_small_{w}x{h}", "rel_path": rel_path, "split": split, "class": cls_name}
            # 64-bit difference hash (9x8 grayscale)
            gray = img_rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            arr = np.asarray(gray, dtype=np.int16)
            diff = arr[:, 1:] > arr[:, :-1]
            bits = diff.flatten()
            dhash_int = 0
            for b in bits:
                dhash_int = (dhash_int << 1) | int(b)
            dhash_hex = f"{dhash_int:016x}"
        return {
            "ok": True,
            "split": split,
            "class": cls_name,
            "rel_path": rel_path,
            "width": w,
            "height": h,
            "md5": md5_hex,
            "dhash": dhash_hex,
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "rel_path": rel_path, "split": split, "class": cls_name}


def assign_role(cls_name: str) -> str:
    if cls_name in META_TRAIN_CLASSES:
        return "base_train"
    if cls_name in META_VAL_CLASSES:
        return "base_val"
    if cls_name in NOVEL_CLASSES:
        return "novel"
    raise ValueError(f"Unknown class: {cls_name}")


def plot_eda(raw_counts, clean_counts):
    """Vẽ biểu đồ phân phối 22 lớp (phân biệt Base và Novel) lưu thành PNG độ phân giải cao."""
    with open(Path(__file__).parent / "clinical_concepts.json", "r", encoding="utf-8") as f:
        meta = json.load(f)["classes"]

    classes_sorted = sorted(ALL_CLASSES, key=lambda c: clean_counts["train"][c] + clean_counts["test"][c], reverse=True)
    labels = [f"{c}\n({meta[c]['vi_name']})" for c in classes_sorted]
    tr_vals = [clean_counts["train"][c] for c in classes_sorted]
    te_vals = [clean_counts["test"][c] for c in classes_sorted]
    is_novel = [c in NOVEL_CLASSES for c in classes_sorted]

    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=220)

    y_pos = np.arange(len(classes_sorted))
    colors_tr = ["#E05A47" if nov else "#2B6CB0" for nov in is_novel]
    colors_te = ["#F6AD55" if nov else "#63B3ED" for nov in is_novel]

    ax.barh(y_pos, tr_vals, color=colors_tr, edgecolor="white", height=0.72, label="Train (Clean)")
    ax.barh(y_pos, te_vals, left=tr_vals, color=colors_te, edgecolor="white", height=0.72, label="Test (Clean)")

    for idx, (tr, te, nov) in enumerate(zip(tr_vals, te_vals, is_novel)):
        tag = " [NOVEL]" if nov else ""
        ax.text(tr + te + 15, idx, f"{tr + te} ({tr}/{te}){tag}", va="center", fontsize=8.5, fontweight="bold" if nov else "normal")

    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Số lượng ảnh sau khi làm sạch & khử trùng lặp", fontsize=10.5, fontweight="bold")
    ax.set_title(
        "Phân phối dữ liệu 22 lớp bệnh da liễu (16 lớp Base màu xanh – 6 lớp Novel màu đỏ cam)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0, max(tr + te for tr, te in zip(tr_vals, te_vals)) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_png = FIG_DIR / "fig1_eda_class_distribution.png"
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main():
    print("=== BƯỚC 1: QUÉT DỮ LIỆU, LỌC LỖI & KHỬ TRÙNG LẶP ===")
    items = []
    for split in ("train", "test"):
        split_dir = DATA_DIR / split
        for cls_name in ALL_CLASSES:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing folder: {cls_dir}")
            for fpath in sorted(cls_dir.iterdir()):
                if fpath.is_file():
                    items.append((split, cls_name, fpath))

    print(f"Tổng số file quét được: {len(items)}")
    with ThreadPoolExecutor(max_workers=12) as ex:
        records = list(ex.map(compute_image_info, items))

    corrupted = [r for r in records if not r["ok"]]
    valid = [r for r in records if r["ok"]]
    print(f"- Ảnh hợp lệ: {len(valid)} | Ảnh lỗi/quá nhỏ: {len(corrupted)}")

    raw_counts = {"train": Counter(), "test": Counter()}
    for r in valid:
        raw_counts[r["split"]][r["class"]] += 1

    # 1. Khử trùng lặp nội bộ trong từng split (cùng class + cùng dHash giữ 1 bản)
    dedup_within_removed = 0
    seen_split_dhash = set()
    unique_within = []
    # Ưu tiên duyệt test trước để giữ nguyên tập test tối đa
    valid_sorted = sorted(valid, key=lambda x: (0 if x["split"] == "test" else 1, x["rel_path"]))
    for r in valid_sorted:
        key = (r["split"], r["class"], r["dhash"])
        if key in seen_split_dhash:
            dedup_within_removed += 1
            continue
        seen_split_dhash.add(key)
        unique_within.append(r)

    # 2. Khử rò rỉ dữ liệu giữa train và test (nếu ảnh train có cùng MD5 hoặc cùng dHash với bất kỳ ảnh test nào -> loại khỏi train)
    test_md5 = {r["md5"] for r in unique_within if r["split"] == "test"}
    test_dhash = {r["dhash"] for r in unique_within if r["split"] == "test"}

    clean_records = []
    leakage_removed = []
    for r in unique_within:
        if r["split"] == "train" and (r["md5"] in test_md5 or r["dhash"] in test_dhash):
            leakage_removed.append(r)
            continue
        r["role"] = assign_role(r["class"])
        clean_records.append(r)

    clean_counts = {"train": Counter(), "test": Counter()}
    for r in clean_records:
        clean_counts[r["split"]][r["class"]] += 1

    # Kiểm tra chéo: đảm bảo overlap = 0
    final_train_dhash = {r["dhash"] for r in clean_records if r["split"] == "train"}
    final_test_dhash = {r["dhash"] for r in clean_records if r["split"] == "test"}
    assert len(final_train_dhash & final_test_dhash) == 0, "Train-test dHash leakage still exists!"

    # Xuất clean_manifest.csv
    manifest_path = OUT_DIR / "clean_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rel_path", "split", "class", "role", "width", "height", "md5", "dhash"],
        )
        writer.writeheader()
        for r in clean_records:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # Xuất báo cáo thống kê chi tiết
    class_summary = []
    for c in ALL_CLASSES:
        class_summary.append({
            "class": c,
            "role": assign_role(c),
            "raw_train": raw_counts["train"][c],
            "raw_test": raw_counts["test"][c],
            "clean_train": clean_counts["train"][c],
            "clean_test": clean_counts["test"][c],
            "removed_train": raw_counts["train"][c] - clean_counts["train"][c],
            "removed_test": raw_counts["test"][c] - clean_counts["test"][c],
        })

    report = {
        "total_scanned_files": len(items),
        "corrupted_or_tiny_removed": len(corrupted),
        "within_split_duplicates_removed": dedup_within_removed,
        "train_test_leakage_removed_from_train": len(leakage_removed),
        "clean_total": len(clean_records),
        "clean_train_total": sum(clean_counts["train"].values()),
        "clean_test_total": sum(clean_counts["test"].values()),
        "verified_train_test_dhash_overlap": len(final_train_dhash & final_test_dhash),
        "corrupted_examples": corrupted[:10],
        "class_summary": class_summary,
    }
    report_path = OUT_DIR / "eda_dedup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fig_path = plot_eda(raw_counts, clean_counts)
    print(f"- Đã loại trùng nội bộ: {dedup_within_removed} ảnh")
    print(f"- Đã loại rò rỉ train-test (khỏi train): {len(leakage_removed)} ảnh")
    print(f"- Tổng sạch cuối cùng: {len(clean_records)} ảnh (Train: {report['clean_train_total']}, Test: {report['clean_test_total']})")
    print(f"- Đã lưu manifest: {manifest_path}")
    print(f"- Đã lưu báo cáo: {report_path}")
    print(f"- Đã lưu biểu đồ EDA: {fig_path}")


if __name__ == "__main__":
    main()
