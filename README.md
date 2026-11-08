# CeNN + MobileNetV2 cho webcam Logitech C270 (train trên Kaggle)

| File | Vai trò |
|---|---|
| `train_repo_only_kaggle.ipynb` | **Notebook Kaggle 1**: chỉ dùng ảnh trong repo GitHub |
| `train_external_kaggle.ipynb` | **Notebook Kaggle 2**: ảnh repo + ảnh ngoài, có chống lệch (giới hạn ≤ 2 lần số ảnh repo, ảnh ngoài luôn bị làm nhiễu kiểu C270, fine-tune chỉ dùng ảnh repo) |
| `cenn.py` | Lớp `CeNN` (CeNN rời rạc, template A/B/z học được), augmentation `webcam_degrade` mô phỏng C270, `build_model`, `export_templates` (JSON + header C) |
| `download_external.py` | Tải và lọc dữ liệu ngoài: TCMP-300 (đọc luồng file 12,8 GB, chỉ giữ loài khớp) và NB-TCM-CHM |
| `c270_tool.py` | Chạy trên máy có cắm C270: thu ảnh test, nhận dạng thời gian thực |
| `notebook_src.py`, `build_notebook.py` | Mã nguồn chung của 2 notebook (dạng `# %%`). Sau khi sửa `cenn.py`, `download_external.py` hoặc `notebook_src.py`, chạy `python build_notebook.py` để sinh lại cả 2 `.ipynb` |

Cả hai notebook đều tự chứa đủ code (tự clone repo, tự ghi `cenn.py`), dùng chung seed và có **tập val/test giống hệt nhau**. Vì vậy kết quả so sánh trực tiếp được với nhau.

## So sánh 2 notebook (kiểm tra ảnh ngoài có gây lệch không)

1. Chạy cả hai notebook (*Save Version → Save & Run All*).
2. Ở trang output của notebook `repo_only`, chọn *New Dataset*. Sau đó *Add Input* dataset này vào notebook `external` và chạy lại mục 7 (hoặc làm ngược lại).
3. Mục 7 sẽ sinh `compare_external_vs_repo_per_class.csv` và biểu đồ `delta_recall_*.png`. Nếu **recall của các lớp không có ảnh ngoài giảm** ở bản external, nghĩa là ảnh ngoài đang gây lệch.

Mỗi notebook cũng tự xuất `per_class_metrics.csv` (precision, recall, F1 và tỷ lệ chấp nhận sai ở α cho từng lớp), nên có thể tự ghép hai file này bằng Excel.

## Chạy trên Kaggle

1. Vào kaggle.com → *Create → New Notebook → File → Import Notebook*, chọn một trong hai file `.ipynb`.
2. *Settings*: **Accelerator = GPU T4 x2 hoặc P100**, **Internet = On**. Bật Internet cần tài khoản Kaggle đã xác minh số điện thoại.
3. (Tùy chọn) Upload tập test webcam thành một Kaggle Dataset, giữ cấu trúc `C270_TEST/<fresh|dried>/<tên_lớp>/*.jpg`, rồi *Add Input*. Notebook tự tìm thư mục `C270_TEST`.
4. Chạy *Run All*, hoặc *Save Version → Save & Run All* để chạy nền (tối đa 12 giờ).
5. Tải kết quả từ tab *Output*: `experiments_cenn/<RUN>/export/<state>/` gồm `model.keras`, `class_names.json`, `config.json`, `.tflite`, `cenn_templates.h`.

Mẹo (bản external): sau lần chạy đầu, tạo Dataset từ output `ExternalData/` rồi *Add Input* ở các lần sau. Notebook sẽ dùng luôn bản đó, khỏi đọc lại luồng 12,8 GB. Muốn chạy nhanh để thử code thì đặt `DOWNLOAD_TCMP = False`.

## Thu ảnh test và chạy webcam (trên máy local)

```bash
pip install opencv-python tensorflow
python c270_tool.py capture --state dried --cls cam_thao --out C270_TEST
python c270_tool.py predict --model export/dried/model.keras --classes export/dried/class_names.json --size 128
```

Nên thu mỗi lớp 20–30 ảnh, dùng nhiều mẫu vật khác nhau, thay đổi nền, ánh sáng và góc chụp. `c270_tool.py` cần nằm cùng thư mục với `cenn.py`.

## Ánh xạ dữ liệu ngoài

Ánh xạ nằm trong `TCMP_MAP` / `NBTCM_MAP` của `download_external.py`. Các loài `approx=True` (chỉ gần đúng: bồ công anh, tam thất, huyết đằng, sơn tra ≈ táo mèo, sa nhân) bị notebook **bỏ qua mặc định** (`INCLUDE_APPROX=False`). Ảnh TCMP-300 của các loài mà repo chỉ có ảnh khô được gắn `state=plant`, và chỉ được dùng khi bật `USE_PLANT_FOR_DRIED=True`.

Cả hai bộ đều theo giấy phép **CC BY 4.0**, khi công bố cần trích dẫn:
- Zhang et al., *TCMP-300*, Scientific Data (2025), doi:10.1038/s41597-025-05522-7
- *NB-TCM-CHM*, Mendeley Data, doi:10.17632/2kjmzjyrmd.3

## Khối CeNN

```text
x[t+1] = x[t] + h·(−x[t] + A∗y[t] + B∗u + z),   y = clip(x, −1, 1),   T bước
out    = clip(u + γ·Proj(y), −1, 1)   (γ khởi tạo 0 nên ban đầu = MobileNetV2 gốc)
```

Tế bào 0 được khởi tạo bằng template phát hiện biên kinh điển, các tế bào còn lại khởi tạo ngẫu nhiên nhỏ. Sau khi train, `export_templates` ghi A, B, z ra `cenn_templates.json` / `cenn_templates.h` để nhúng lên vi điều khiển hoặc FPGA. Với 6 tế bào, 4 bước, ảnh 128 px, khối CeNN tốn khoảng 24 triệu MAC (MobileNetV2 ở 128 px tốn khoảng 100 triệu).
