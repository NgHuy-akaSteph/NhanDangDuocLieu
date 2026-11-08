# Báo cáo dataset và hướng triển khai nhận dạng dược liệu

> Phạm vi: nội dung hiện có trong repo, gồm thư mục `DuocLieu/` và notebook `train.ipynb`. Các số liệu dataset dưới đây được đếm trực tiếp từ file hiện tại; output đã lưu trong notebook có thể phản ánh một phiên bản dữ liệu cũ.

## 1. Bài toán của repo

Repo xây dựng bộ phân loại **tên dược liệu từ ảnh**. Ảnh được tổ chức theo hai trạng thái `fresh` (tươi) và `dried` (khô). Notebook huấn luyện **một mô hình cho mỗi trạng thái**; đầu vào khi dự đoán gồm ảnh và trạng thái đã biết, đầu ra gồm tên lớp dự đoán, độ tin cậy và quyết định chấp nhận kết quả tại chỗ hay chuyển tiếp.

Về mặt thuật toán, đây là **phân loại ảnh nhiều lớp có giám sát**. Mỗi thư mục tên dược liệu là một nhãn lớp. Code không tự phân loại trạng thái tươi/khô, không xác định vùng dược liệu trong ảnh và không suy ra công dụng dược học của mẫu.

Hàm `student_predict()` trả `accept_local` nếu xác suất lớp cao nhất đạt ngưỡng `ALPHA_THRESHOLD`; ngược lại trả `send_to_gemini`. Đây mới là **quyết định định tuyến**: repo chưa có mã gọi Gemini hoặc triển khai dịch vụ suy luận hoàn chỉnh.

## 2. Dataset thực tế

### 2.1. Cấu trúc và quy mô

```text
DuocLieu/
  <ten_duoc_lieu>/
    raw/
      fresh/    # ảnh tươi, nếu có
      dried/    # ảnh khô, nếu có
```

Có **30 tên dược liệu** và **696 file ảnh** trong cây thư mục. Bộ lọc của notebook chỉ nhận đuôi `.jpg`, `.jpeg`, `.png`, `.bmp`, vì vậy nó bỏ qua một file `.webp` và quét **695 ảnh**: **283 tươi**, **412 khô**. Các ảnh có đuôi được chấp nhận đều mở được bằng Pillow khi kiểm tra repo hiện tại.

| Dược liệu (tên thư mục) | Tươi | Khô | Tổng được notebook quét |
|---|---:|---:|---:|
| `Vo_buoi` | 11 | 15 | 26 |
| `bo_cong_anh` | 7 | 11 | 18 |
| `cam_thao` | 3 | 39 | 42 |
| `cau_ky_tu` | 0 | 16 | 16 |
| `che_day` | 0 | 15 | 15 |
| `dinh_lang` | 107 | 6 | 113 |
| `do_trong_bac` | 0 | 39 | 39 |
| `don_do` | 0 | 12 | 12 |
| `gai_bo_ket` | 0 | 13 | 13 |
| `gung` | 13 | 0 | 13 |
| `ha_thu_o_do` | 2 | 16 | 18 |
| `hoai_son` | 0 | 14 | 14 |
| `huyet_dang` | 0 | 28 | 28 |
| `la_mo_long` | 23 | 0 | 23 |
| `la_xa_den` | 26 | 6 | 32 |
| `lac_tien` | 3 | 3 | 6 |
| `long_nhan` | 0 | 16 | 16 |
| `ma_de` | 14 | 19 | 33 |
| `ngai_tuong` | 0 | 26 | 26 |
| `nu_tam_that_bac` | 7 | 13 | 20 |
| `qua_la_han` | 0 | 9 | 9 |
| `que_chi` | 0 | 6 | 6 |
| `rau_ma` | 28 | 11 | 39 |
| `sa` | 7 | 3 | 10 |
| `tam_sen` | 1 | 6 | 7 |
| `tao_do` | 0 | 7 | 7 |
| `tao_meo` | 4 | 23 | 27 |
| `thuc_dia` | 0 | 13 | 13 |
| `tia_to` | 27 | 0 | 27 |
| `vo_quyt` | 0 | 27 | 27 |
| **Tổng** | **283** | **412** | **695** |

`ha_thu_o_do/raw/dried/` còn một file `.webp`, nên tổng file thực tế của lớp này là 19.

### 2.2. Kích thước và định dạng

Trên 695 ảnh được notebook quét, chiều rộng nằm trong **75–5.472 px**, chiều cao **88–3.648 px**; trung vị lần lượt khoảng **709 px** và **493 px**. Ảnh có nhiều tỷ lệ khung hình, không có một kích thước gốc thống nhất. Dung lượng file trung vị khoảng **172 KB**.

Định dạng đọc từ nội dung file gồm **385 JPEG, 286 PNG và 24 WebP**. Có **32 file có đuôi không khớp định dạng nội dung**; trong đó một số file `.jpg` thực chất là WebP. Vì vậy việc Pillow `verify()` thành công chưa bảo đảm bước `tf.io.decode_image()` của TensorFlow sẽ đọc được mọi file. Cần kiểm tra decode bằng chính pipeline TensorFlow trước khi coi một lần chạy là hoàn chỉnh.

### 2.3. Những lớp thực sự được huấn luyện

Notebook yêu cầu **ít nhất 7 ảnh trong từng cặp (trạng thái, dược liệu)**. Điều kiện này được xét độc lập cho `fresh` và `dried`, nên cùng một dược liệu có thể được học ở một nhánh nhưng không ở nhánh kia.

| Nhánh | Số lớp đủ điều kiện | Số ảnh đủ điều kiện | Train theo cách chia hiện tại | Validation theo cách chia hiện tại |
|---|---:|---:|---:|---:|
| Tươi | 11 | 270 | 214 | 56 |
| Khô | 21 | 382 | 304 | 78 |

Tổng cộng **43 ảnh** thuộc các cặp có dưới 7 ảnh nên không được đưa vào huấn luyện. Hai nhánh có thể chia sẻ tên lớp nhưng **không chia sẻ cùng bộ nhãn số**: nhãn được đánh số từ danh sách lớp đã sắp xếp riêng cho từng nhánh.

### 2.4. Giới hạn chất lượng dữ liệu

- Dữ liệu **mất cân bằng lớn**. Ví dụ, `dinh_lang` tươi có 107 ảnh, còn một số lớp đủ điều kiện chỉ có 7 ảnh. Code có dùng trọng số lớp để giảm ảnh hưởng của việc này, nhưng không tạo thêm thông tin độc lập cho các lớp ít ảnh.
- Có **file trùng hệt nội dung**. Khi tái hiện cách chia ngẫu nhiên của notebook trên dữ liệu hiện tại, có **11 nhóm ảnh trùng hệt xuất hiện ở cả train và validation**, liên quan tới 12 ảnh validation. Do đó số đo validation có nguy cơ cao hơn khả năng tổng quát hóa thực tế.
- Không có tập **test độc lập**. Repo cũng không cung cấp nguồn ảnh, cách thu thập, kiểm tra nhãn, giấy phép sử dụng, hay quy tắc để tránh các ảnh rất giống nhau của cùng một mẫu xuất hiện ở cả hai tập.
- Output cũ lưu trong notebook ghi `nu_tam_that_bac` là 0 ảnh tươi và 9 ảnh khô, trong khi file hiện tại là 7 ảnh tươi và 13 ảnh khô. Vì thế cần chạy lại toàn bộ thống kê và huấn luyện trước khi dùng kết quả cũ để báo cáo hiệu năng.

## 3. Luồng xử lý trong `train.ipynb`

1. Gắn Google Drive trong Colab, đặt `PROJECT_DIR = /content/drive/MyDrive/NCKH2026`, tìm ảnh ở `DuocLieu/` và tạo thư mục `experiments/<RUN_ID>/` để lưu kết quả. Notebook hiện phụ thuộc đường dẫn này; clone repo về máy rồi chạy nguyên trạng sẽ không tự tìm thấy dữ liệu.
2. Quét hai thư mục `fresh` và `dried` của từng dược liệu, kiểm tra ảnh, tạo `dataset_summary.csv`.
3. Với từng nhánh, giữ các lớp có ít nhất 7 ảnh. Xáo ảnh **trong từng lớp** bằng seed cố định, lấy khoảng 20% cho validation và tối thiểu 2 ảnh validation/lớp. Lưu `split_manifest.csv` cùng `class_names.json`.
4. Tạo `tf.data.Dataset`: đọc file, decode thành 3 kênh RGB, co ảnh bằng `resize_with_pad` về hình vuông, cache, xáo tập train, gom batch 8 và prefetch.
5. Huấn luyện MobileNetV2 riêng cho mỗi trạng thái ở bốn kích thước đầu vào **96, 128, 160, 224 px**: tổng cộng theo thiết kế là **8 thí nghiệm**.
6. Vẽ lịch sử học và tính các chỉ số validation; chuẩn bị hàm dự đoán với mô hình cỡ **128 px**.

Notebook có **hai định nghĩa `make_dataset()`**. Định nghĩa thứ hai ghi đè định nghĩa thứ nhất và thực hiện chức năng tương đương. Đây là phần code lặp, có thể gây khó theo dõi khi sửa pipeline.

## 4. Transfer learning với MobileNetV2: triển khai cụ thể

### 4.1. Vì sao dùng transfer learning?

Dataset chỉ có vài trăm ảnh, nhiều lớp chỉ có 7–15 ảnh. Huấn luyện một mạng CNN từ đầu trong điều kiện đó dễ học thuộc ảnh train. MobileNetV2 đã học các đặc trưng ảnh tổng quát từ ImageNet, nên notebook tận dụng nó như **bộ trích đặc trưng có sẵn** và chỉ học lại phần đầu ra cho các lớp dược liệu. MobileNetV2 cũng tương đối gọn, phù hợp với mục tiêu dự đoán tại chỗ.

### 4.2. Kiến trúc được tạo cho từng thí nghiệm

```text
Ảnh gốc
  → đọc RGB, resize_with_pad thành S × S × 3; giá trị pixel float32 khoảng 0–255
  → tăng cường ảnh (chỉ lúc train)
  → MobileNetV2.preprocess_input: đưa giá trị pixel về khoảng [-1, 1]
  → MobileNetV2 pretrained ImageNet, bỏ classifier gốc, đóng băng backbone
  → GlobalAveragePooling2D
  → Dropout(0.25)
  → Dense(số lớp của trạng thái, softmax)
  → vector xác suất cho từng dược liệu
```

Trong `build_model(num_classes, state, image_size)`, backbone được tạo bằng `tf.keras.applications.MobileNetV2(input_shape=(S, S, 3), alpha=1.0, include_top=False, weights="imagenet")`. `include_top=False` loại lớp phân loại ImageNet 1.000 lớp. `alpha=1.0` giữ độ rộng chuẩn của MobileNetV2. Một lần dựng mô hình tương ứng với **một kích thước S** và **một trạng thái**; lớp Dense cuối có `num_classes` đầu ra riêng cho nhánh đó.

`base_model.trainable = False` làm trọng số pretrained của backbone **không cập nhật** khi train. `base_model(x, training=False)` còn gọi backbone ở chế độ suy luận, giữ hành vi ổn định của các lớp Batch Normalization. Phần được học chủ yếu là lớp Dense cuối; `GlobalAveragePooling2D` gộp bản đồ đặc trưng theo không gian thành một vector và `Dropout(0.25)` chỉ bỏ ngẫu nhiên một phần đặc trưng lúc train để giảm học thuộc.

Notebook **không có giai đoạn fine-tuning**: không mở lại các tầng MobileNetV2 để cập nhật với learning rate thấp. Vì vậy, mô tả chính xác phương pháp hiện tại là *fixed feature extractor + classifier mới*, không phải fine-tune toàn bộ mạng.

### 4.3. Tăng cường dữ liệu và hàm mục tiêu

Lớp augmentation gồm lật ngang, xoay nhẹ (`0.08` theo tham số Keras), zoom 10% và thay đổi tương phản 10%. Các lớp này nằm **trong model**, nên chỉ biến đổi ngẫu nhiên khi train; validation và `student_predict()` chạy model với `training=False`. Cách này giúp mô hình gặp nhiều biến thể của ảnh train, nhưng không thay thế ảnh chụp độc lập.

Model dùng `Adam(learning_rate=1e-3)`, `sparse_categorical_crossentropy` vì nhãn là **số nguyên** thay vì one-hot, cùng hai chỉ số `accuracy` và `top2_accuracy`. `top2_accuracy` cho biết nhãn đúng có nằm trong hai lớp xác suất cao nhất hay không.

Trọng số của từng lớp được tính từ số ảnh train:

```text
weight(lớp) = tổng số ảnh train / (số lớp × số ảnh train của lớp)
```

Do đó lỗi ở lớp ít ảnh đóng góp nhiều hơn vào loss. Trọng số được truyền vào `model.fit(..., class_weight=...)`; nó **không thay đổi số ảnh** và không xử lý được nhãn sai hoặc ảnh trùng.

### 4.4. Vòng huấn luyện và lưu kết quả

Với mỗi cặp `(image_size, state)`, notebook tạo model mới, train tối đa **30 epoch**, rồi đánh giá trên validation của chính nhánh đó. Các callback gồm:

- `ModelCheckpoint`: lưu `best_model.keras` khi `val_loss` giảm.
- `EarlyStopping`: dừng sau 6 epoch không cải thiện `val_loss`, khôi phục trọng số tốt nhất.
- `ReduceLROnPlateau`: giảm learning rate một nửa sau 2 epoch không cải thiện, tối thiểu `1e-6`.
- `CSVLogger`: lưu lịch sử từng epoch.

Model tốt nhất được tải lại từ file và đánh giá. Kết quả thiết kế để ghi ra `training_summary.csv`; mỗi thí nghiệm còn có lịch sử train và file dự đoán validation. Các output này được đặt dưới Google Drive, **không có model đã train trong repo**.

### 4.5. Dự đoán và ngưỡng tin cậy

`student_predict(image_path, state)` tải model của nhánh tương ứng ở cỡ ảnh **128 × 128**, áp dụng cùng bước resize và lấy lớp có xác suất softmax cao nhất. Nếu xác suất này `>= ALPHA_THRESHOLD`, hàm trả `accept_local`; nếu thấp hơn thì trả `send_to_gemini`.

Giá trị **được code sử dụng là 0,85**. Tiêu đề phần đánh giá và tên file `size_comparison_alpha_095.csv` lại ghi **0,95**, nên tên gọi hiện không khớp phép tính. Xác suất softmax cũng chưa được kiểm tra hiệu chuẩn; một dự đoán 0,85 không tự chứng minh rằng mô hình đúng 85% trong thực tế.

Phần đánh giá dự kiến tính accuracy toàn bộ, macro F1, tỷ lệ được xử lý tại chỗ, tỷ lệ chuyển tiếp, accuracy trên nhóm được chấp nhận và tỷ lệ chấp nhận sai. Đây đều là số đo trên **validation**, chưa phải kết quả trên tập test độc lập.

## 5. Trạng thái triển khai và cách hiểu kết quả

Output đã lưu trong notebook cho thấy quá trình huấn luyện đi tới thí nghiệm `fresh` 224 px rồi dừng giữa chừng. Chưa có output của phần tổng hợp kết quả, so sánh bốn kích thước, đánh giá ngưỡng hay chạy dự đoán. Mặc dù code đã gán `SELECTED_IMAGE_SIZE = 128`, repo **chưa có bằng chứng hoàn chỉnh rằng 128 px là cấu hình tốt nhất**.

Để báo cáo hiệu năng đáng tin cậy, cần tối thiểu: sửa/loại ảnh trùng trước khi chia tập, xác nhận các file có định dạng nội dung bất thường đọc được bởi TensorFlow, chạy lại notebook trên dataset hiện tại, rồi đánh giá trên tập test độc lập. Đây là các giới hạn của kết quả hiện có, không phải các bước đã được notebook thực hiện.
