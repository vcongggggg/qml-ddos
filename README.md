# Đồ Án Cuối Kỳ: Nghiên Cứu và Đánh Giá Quantum Machine Learning trong Phát Hiện Tấn Công DDoS (NISQ-Era Evaluation)

## Môn học: Mã Hóa & Mật Mã Học (CC102)
**Giảng viên hướng dẫn:** PGS. TS. Nguyễn Tấn Khôi  
**Sinh viên thực hiện:**
- **Ngô Văn Công** (MSSV: 102230284)
- **Nguyễn Công Minh** (MSSV: 102230306)

**Bài báo nghiên cứu tham khảo:**
*“Benchmarking quantum machine learning for detecting DDoS cyber-attacks: a NISQ-era evaluation”*  
Tác giả: Ricardo S. Alonso, Guillermo Rivas, Rodrigo Gil-Merino, Diego Valdeolmillos, Javier Prieto.  
Nguồn tham khảo mã nguồn gốc: `https://github.com/ralorin/qml-ddos`

---

## 1. Các Hạng Mục Trọng Tâm Đã Hoàn Thiện (Cập Nhật Theo Nhận Xét Tiến Độ 17/11)

Dự án đã được nâng cấp toàn diện để giải quyết trực tiếp các câu hỏi phản biện và yêu cầu của Giảng viên:
1. **Bộ 5 Sơ đồ Mạch Lượng tử & Giải thích Cơ chế Cổng (`+ Sơ đồ mạch?`, `+ Hiểu sơ đồ, cơ chế`):**
   - Xuất tự động sơ đồ mạch độ phân giải cao (300 DPI) cho toàn bộ các kiến trúc trong bài báo:
     - `circuit_1_qsvm_angle_kernel.png`: Mạch Quantum Kernel với `AngleEmbedding(x)` + `Adjoint(AngleEmbedding(z))` (8 Qubits, Depth = 2, 0 CNOT).
     - `circuit_2_qsvm_amplitude_kernel.png`: Mạch Quantum Kernel với `AmplitudeEmbedding` (nén 8 đặc trưng vào 3 Qubits: `2^3 = 8`).
     - `circuit_3_zz_feature_map.png`: Mạch `ZZFeatureMap` vướng víu phi tuyến cặp `CZ - RZ(2(pi-xi)(pi-xj)) - CZ`.
     - `circuit_4_qnn_twolocal_angle.png`: Mạng `QNN Two-Local` (8 Qubits, 16 tham số `RY`, 7 cổng `CNOT` chuỗi bậc thang, đo toán tử chiếu `Hermitian M = |0><0|`).
     - `circuit_5_qnn_treetensor_angle.png`: Mạng `QNN Tree-Tensor Network (TTN)` (8 Qubits, 15 tham số `RY`, 7 cổng `CNOT` phân cấp cây nhị phân `K = 3` tầng).
   - Bảng thông số kỹ thuật mạch chi tiết: `results/bang_thong_so_kien_truc_mach_luong_tu.csv`.
2. **Hệ thống Chỉ số Thực nghiệm Toàn diện (`+ Kết quả chỉ số?`, `+ Thời gian chạy?`):**
   - Thống kê đầy đủ 7 chỉ số: `Accuracy`, `Macro Precision`, `Macro Recall`, `Macro F1-Score`, `Per-class F1 (Syn & Benign)`, `ROC-AUC`, `Train Time (s)` và `Inference Time (s)` trên cả **12 mô hình của bài báo gốc** (`results/bang_tong_hop_benchmark_chinh_thuc_bai_bao.csv`) lẫn **7 mô hình chạy tái hiện trực tiếp trên máy trạm** (`results/benchmark_summary.csv`).
3. **Phân tích So sánh Ưu / Nhược điểm (`+ Phân tích ưu/nhược`):**
   - Phân tích định lượng Ưu/Nhược điểm của 3 kỹ thuật nhúng (`Angle` vs `Amplitude` vs `ZZ Feature Map`), 2 kiến trúc biến phân (`Two-Local` vs `Tree-Tensor`), và 3 họ mô hình (`Classical ML` vs `QSVM` vs `QNN`) trong tài liệu `TAI_LIEU_LY_THUYET_DE_TAI_QML_DDOS.md`.
4. **Thực nghiệm Mở rộng Phân loại Đa lớp (`+ Multiclass`):**
   - Mở rộng từ phân loại nhị phân (`SYN Flood` vs `Benign`) sang phân loại đa lớp **4 nhãn lưu lượng mạng (`SYN Flood`, `Benign`, `UDP Flood`, `DNS Amplification`)** bằng `One-vs-Rest Quantum Kernel QSVM` và `4-Wire Pauli-Z Hybrid QNN` (`results/multiclass_benchmark_summary.csv` & `results/multiclass_confusion_matrices.png`).

---

## 2. Bảng Tóm Tắt Kết Quả Thực Nghiệm 12 Mô Hình Bài Báo Gốc

| Mô hình Thực nghiệm | Họ Giải thuật | Tập Dữ liệu | Số Qubit | Accuracy | Macro Precision | Macro Recall | Macro F1 | F1 (Benign) | ROC-AUC | Train Time (s) | Inference Time (s) |
| :--- | :--- | :---: | :-: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical SVM (RBF)** | Classical ML | FULL (100k) | 0 | 0.9926 | 0.9700 | 1.0000 | **0.9800** | 0.9600 | 0.9954 | **0.98** | **0.13** |
| **Classical ANN (8-8-16-8-1)** | Classical ML | FULL (100k) | 0 | **0.9941** | 0.9700 | 0.9900 | **0.9800** | **0.9700** | **0.9988** | 21.00 | 0.39 |
| **QSVM (Angle Embedding)** | Quantum SVM | SMALL (1k) | 8 | **0.9900** | 0.9500 | **0.9900** | **0.9700** | **0.9500** | 0.9944 | 1265.59 | 1576.28 |
| **QSVM (Amplitude 3-Qubit)** | Quantum SVM | SMALL (1k) | 3 | 0.9700 | 0.9300 | 0.8900 | 0.9100 | 0.8400 | 0.8944 | 4432.02 | 5502.47 |
| **QSVM (ZZ Feature Map)** | Quantum SVM | SMALL (1k) | 8 | 0.9700 | **0.9800** | 0.8500 | 0.9000 | 0.8200 | 0.8500 | 18115.92 | 23312.32 |
| **QNN (Two-Local + Angle)** | Quantum NN | SMALL (1k) | 8 | 0.9700 | **0.9800** | 0.8500 | 0.9000 | 0.8200 | 0.9933 | 978.96 | **6.94** |
| **QNN (Tree-Tensor + Angle)** | Quantum NN | SMALL (1k) | 8 | 0.9700 | **0.9800** | 0.8500 | 0.9000 | 0.8200 | **0.9956** | 3607.15 | **8.78** |
| **QNN (Two-Local + ZZ Map)** | Quantum NN | SMALL (1k) | 8 | 0.9200 | 0.7800 | 0.9600 | 0.8300 | 0.7100 | 0.9889 | 2409.97 | 27.09 |
| **QNN (Tree-Tensor + ZZ Map)** | Quantum NN | SMALL (1k) | 8 | 0.9200 | 0.7800 | 0.9600 | 0.8300 | 0.7100 | 0.9911 | 2795.10 | 26.24 |
| **QNN (Two-Local + Amp 3Q)** | Quantum NN | SMALL (1k) | 3 | 0.4300 | 0.4100 | 0.2400 | 0.3000 | 0.0000 | 0.2100 | 125.96 | 6.13 |
| **QNN (Two-Local + Amp 4Q)** | Quantum NN | SMALL (1k) | 4 | 0.9000 | 0.4500 | 0.5000 | 0.4700 | 0.0000 | 0.7178 | 342.10 | 7.45 |
| **QNN (Tree-Tensor + Amp 4Q)** | Quantum NN | SMALL (1k) | 4 | 0.9000 | 0.4500 | 0.5000 | 0.4700 | 0.0000 | 0.7678 | 418.50 | 8.12 |

---

## 3. Cấu Trúc Thư Mục Dự Án

```text
qml-ddos/
├── 01_train_models_qml_ddos.ipynb                 # [NOTEBOOK 1] Tiền xử lý PCA, Sơ đồ mạch, Huấn luyện 7 Model & Lưu trọng số
├── 02_test_and_evaluate_qml_ddos.ipynb            # [NOTEBOOK 2] Nạp Model đã Train, Kiểm thử Độc lập & Vẽ Biểu đồ Đánh giá
├── models/
│   └── trained_qml_ddos_bundle.pkl                # Gói lưu trữ 7 mô hình, bộ chuẩn hóa PCA và trọng số góc quay lượng tử đã học
├── TAI_LIEU_LY_THUYET_DE_TAI_QML_DDOS.md           # Tài liệu lý thuyết, sơ đồ mạch, phân tích ưu/nhược từ A-Z
├── generate_quantum_circuits_and_paper_metrics.py # Script xuất 5 sơ đồ mạch lượng tử & bảng chỉ số 12 mô hình
├── export_paper_and_reproduction_benchmark.py     # Script xuất dữ liệu đối sánh chuẩn bài báo gốc
├── run_benchmark_all.py                           # Pipeline CLI chạy tự động 7 mô hình Binary + 4 mô hình Multiclass
├── run_dataset_preprocessing.py                   # Script tiền xử lý, chuẩn hóa MaxAbsScaler và PCA 8 chiều
├── download_syn_dataset.py                        # Script tải bộ dữ liệu gốc CICDDoS2019
├── dataset/                                       # Thư mục dữ liệu mẫu (tiny & small)
├── results/                                       # Toàn bộ sơ đồ mạch, biểu đồ ROC, Confusion Matrix và bảng CSV
│   ├── circuit_1_qsvm_angle_kernel.png            # Sơ đồ mạch 1: QSVM Angle Embedding Kernel (8 Qubits)
│   ├── circuit_2_qsvm_amplitude_kernel.png        # Sơ đồ mạch 2: QSVM Amplitude Embedding Kernel (3 Qubits)
│   ├── circuit_3_zz_feature_map.png               # Sơ đồ mạch 3: Cấu trúc ZZ Feature Map
│   ├── circuit_4_qnn_twolocal_angle.png           # Sơ đồ mạch 4: Mạng QNN Two-Local + Angle (16 tham số)
│   ├── circuit_5_qnn_treetensor_angle.png         # Sơ đồ mạch 5: Mạng QNN Tree-Tensor + Angle (15 tham số)
│   ├── bang_thong_so_kien_truc_mach_luong_tu.csv  # Bảng thông số kỹ thuật 8 cấu hình mạch lượng tử
│   ├── bang_tong_hop_benchmark_chinh_thuc_bai_bao.csv # Bảng đầy đủ 7 chỉ số của 12 mô hình bài báo
│   ├── benchmark_summary.csv                      # Bảng kết quả chạy thực nghiệm 7 mô hình trên máy trạm
│   ├── multiclass_benchmark_summary.csv           # Bảng kết quả mở rộng phân loại đa lớp (4 lớp DDoS)
│   ├── bieu_do_phan_hoa_day_du_bai_bao.png        # Biểu đồ đối sánh 5 chỉ số trên 12 mô hình bài báo
│   ├── bieu_do_thoi_gian_huan_luyen_log10.png     # Biểu đồ thời gian Train vs Inference (Log10)
│   ├── benchmark_metrics_comparison.png           # Biểu đồ 5 chỉ số của 7 mô hình chạy trên máy trạm
│   ├── benchmark_time_comparison.png              # Biểu đồ thời gian chạy thực tế trên máy trạm
│   ├── confusion_matrices_all_models.png          # Ma trận nhầm lẫn của 7 mô hình Binary
│   ├── multiclass_confusion_matrices.png          # Ma trận nhầm lẫn của 4 mô hình Multiclass
│   └── roc_curves_all_models.png                  # Đường cong ROC-AUC của 7 mô hình
└── dev/                                           # 29 Jupyter Notebooks gốc của tác giả bài báo
```

---

## 4. Hướng Dẫn Chạy Thực Nghiệm (2 Chế Độ: Jupyter Notebook Tách Biệt hoặc CLI Tự Động)

### Cách 1: Chạy Tương Tác Từng Bước Qua 2 File Jupyter Notebook (Khuyên dùng khi thuyết trình / demo)
1. **Bước Huấn luyện (`01_train_models_qml_ddos.ipynb`):**
   - Nạp dữ liệu huấn luyện `dataset/tr_day_syn_small.csv`, fit `MaxAbsScaler` và `PCA (80 -> 8 chiều)`.
   - Vẽ trực tiếp sơ đồ mạch lượng tử, huấn luyện cả 7 mô hình và xuất gói trọng số ra `models/trained_qml_ddos_bundle.pkl`.
2. **Bước Kiểm thử độc lập (`02_test_and_evaluate_qml_ddos.ipynb`):**
   - Nạp trực tiếp model đã huấn luyện từ `models/trained_qml_ddos_bundle.pkl` (không cần train lại).
   - Chạy kiểm thử trên tập `dataset/test_day_syn_small.csv`, xuất bảng chỉ số, Confusion Matrix và ROC-AUC ngay trên giao diện Notebook.

### Cách 2: Chạy Tự Động Hóa Qua Command Line (CLI Script)
```bash
# 1. Xuất toàn bộ sơ đồ mạch lượng tử và bảng chỉ số chuẩn 12 mô hình bài báo
python export_paper_and_reproduction_benchmark.py

# 2. Chạy pipeline thực nghiệm 7 mô hình Binary + 4 mô hình Multiclass trên máy trạm
python run_benchmark_all.py
```
