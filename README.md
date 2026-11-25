# Phân Loại Ảnh Bệnh Da Liễu Ít Mẫu (BiomedCLIP + Đa Giai Đoạn SAMP)

Mã nguồn thực nghiệm và ứng dụng **FastAPI Web Demo** cho đề xuất Đề tài Nghiên cứu Khoa học Sinh viên năm 2027 (Trường Đại học Công nghệ Kỹ thuật TP.HCM).

## 1. Điểm nhấn phương pháp & Kết quả chính (5 Random Seeds × 200 = 1.000 Episodes)

- **Làm sạch & khử rò rỉ dữ liệu:** Quét `15.444` ảnh lâm sàng (`SkinDisease` - 22 lớp), loại bỏ `479` ảnh trùng nội bộ, `168` ảnh rò rỉ `dHash` giữa `train–test` và `17` ảnh gần trùng ngữ nghĩa (`Cosine > 0.985`), giữ lại `14.797` ảnh sạch (`overlap = 0`).
- **Đa nguyên mẫu 4 giai đoạn bệnh (`SAMP`):** Biểu diễn mỗi bệnh bằng 4 thể lâm sàng riêng (*cấp tính rỉ dịch ướt – bán cấp đóng mài – mạn tính khô vảy/dày sừng – cào xước thứ phát & dấu hiệu phân biệt vị trí*), kết hợp khử độ lệch từ vựng nền từ `Meta-Train` và hiệu chỉnh nguyên mẫu `1/K`.
- **Kết quả trên 6 lớp bệnh hiếm (`1-crop` công bằng, `Mean ± SD` qua 5 seeds):**
  - `0-shot`: tăng từ `41,14%` (tên bệnh) và `50,56%` (1 câu gộp) lên **`61,17%`**.
  - `1-shot`: đạt **`59,45 ± 0,20%`** (so với `29,57%` của `ResNet-50 + ProtoNet` và `40,47%` của `BiomedCLIP Visual`).
  - `5-shot` / `10-shot`: đạt **`62,42 ± 0,08%`** và **`62,64 ± 0,11%`** (`AUROC = 89,83%`).
- **Kiểm chứng ngoài miền theo `patient_id`:** Đánh giá độc lập trên `2.298` ảnh điện thoại của `1.373` bệnh nhân (`PAD-UFES-20`, `0%` trùng bệnh nhân giữa support và query).

## 2. Cấu trúc thư mục

- `src/` — Tiền xử lý dữ liệu: quét lọc trùng lặp (`MD5 + dHash + Cosine`), khử rò rỉ `train–test` và trích xuất đặc trưng cơ sở.
- `src_v2/` — Thuật toán chính `SAMP` & chuỗi thực nghiệm `5 seeds`: từ điển 4 giai đoạn `clinical_stages_v2.json`, các mô hình đối chứng, so sánh công bằng `1-crop` vs `5-crop`, kiểm định `DermLIP`, kiểm chứng ngoài miền `PAD-UFES-20` theo `patient_id` và tinh chỉnh nhẹ `ViT + WiSE-FT`.
- `web_demo/` — Ứng dụng Web `FastAPI` đối chiếu giai đoạn tổn thương, hiển thị dấu hiệu `SKINCON` và đăng ký lớp bệnh mới trực tiếp từ `1–5` ảnh mẫu.
- `outputs_v2/` — Bảng số liệu kết quả thực nghiệm (`.csv`, `.json`), bộ biểu đồ trực quan (`.png`) và gói trọng số gọn nhẹ `checkpoints/samp_v2_bundle.pt` phục vụ chạy ngay Web Demo.

## 3. Cài đặt & Chạy ứng dụng FastAPI Web Demo

```powershell
# Cài đặt thư viện
pip install -r requirements.txt

# Khởi động FastAPI Web Demo tại http://127.0.0.1:8000
python -m uvicorn app:app --app-dir web_demo --host 127.0.0.1 --port 8000
```
