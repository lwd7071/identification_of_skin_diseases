# HỒ SƠ KIỂM CHỨNG MÔ HÌNH THAM CHIẾU PHỤ `DermLIP` (BƯỚC 3 – CÁCH 1)

- **Công trình gốc:** Yan, S., et al. (2025). *Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology.* ICCV 2025 Highlight (`arXiv:2503.14911v1`).
- **Kho trọng số chính thức trên HuggingFace:**
  - `redlessone/DermLIP_ViT-B-16` — Giấy phép: **`cc-by-nc-4.0`** | Trạng thái cổng kiểm duyệt: **`gated=auto`** | Lượt tải: `3764`
  - `redlessone/DermLIP_PanDerm-base-w-PubMed-256` — Giấy phép: **`cc-by-nc-4.0`** | Trạng thái cổng kiểm duyệt: **`gated=auto`** | Lượt tải: `129`

## 1. Kết luận về Giấy phép & Điều kiện sử dụng
- Trọng số `DermLIP` phát hành theo giấy phép **Creative Commons Attribution-NonCommercial 4.0 (`CC-BY-NC-4.0`)**, **hoàn toàn hợp pháp** cho đề tài Nghiên cứu Khoa học Sinh viên phi thương mại.
- Do kho lưu trữ đặt chế độ `gated: auto` trên HuggingFace (đòi hỏi tài khoản cá nhân xác nhận điều khoản và cấp `HF_TOKEN`), ở **Cách 1** hệ thống tuân thủ nghiêm quy tắc bảo mật/quyền truy cập: không tự ý vượt cổng xác thực và ghi nhận đầy đủ hồ sơ kiểm chứng tại đây.

## 2. Kiểm tra nguy cơ trùng lặp dữ liệu huấn luyện với `SkinDisease` (`DermNet`)
Đối chiếu trực tiếp toàn văn phụ lục kỹ thuật **Mục 7.1 – 7.5** của `arXiv:2503.14911v1`:
1. **Các bộ dữ liệu mở được nạp trực tiếp (Mục 7.5 – *Public Dataset*):** Chỉ gồm **`SCIN`** (`6.518` cặp) và **`MSKCC`** (`10.619` cặp). Tập **`DermNet` (`SkinDisease`) KHÔNG nằm trong danh sách nạp trực tiếp**.
2. **Các nguồn thu thập còn lại:** Video bài giảng y khoa trên *YouTube* (Mục 7.1), bài báo mở *PubMed* (Mục 7.2), *68 sách/giáo trình PDF* quét qua `PyMuPDF` (Mục 7.3), và diễn đàn y khoa (Mục 7.4).
3. **Đánh giá rủi ro rò rỉ gián tiếp:** **Thấp – Trung bình**. Dù không nạp trực tiếp `DermNet`, một số ảnh kinh điển từ `DermNet` thường được trích lại trong giáo trình PDF hoặc slide bài giảng YouTube.
4. **Quy tắc sử dụng trong đề tài (năm 2027):** Khi bổ sung `HF_TOKEN` để chạy `DermLIP`, kết quả phải được trình bày ở nhóm **Baseline tham chiếu chuyên biệt miền (*In-Domain Pretrained Reference*)** tách riêng khỏi `ResNet-50` và `BiomedCLIP`, kèm ghi chú rõ phạm vi dữ liệu `Derm1M`.
