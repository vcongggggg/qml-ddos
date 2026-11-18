# Đồ Án Cuối Kỳ: Nghiên Cứu và Đánh Giá Quantum Machine Learning trong Phát Hiện Tấn Công DDoS

## Môn học: Mã Hóa & Mật Mã Học
**Sinh viên thực hiện:**
- **Ngô Văn Công** (MSSV: 102230284)
- **Nguyễn Công Minh** (MSSV: 102230306)

**Bài báo nghiên cứu tham khảo:**
*“Benchmarking quantum machine learning for detecting DDoS cyber-attacks: a NISQ-era evaluation”*  
Tác giả: Ricardo S. Alonso, Guillermo Rivas, Rodrigo Gil-Merino, Diego Valdeolmillos, Javier Prieto.  
Nguồn tham khảo mã nguồn gốc: `https://github.com/ralorin/qml-ddos`

---

## 1. Giới Thiệu Đề Tài
Trong bối cảnh bùng nổ của các thiết bị Internet of Things (IoT) với tài nguyên phần cứng hạn chế, các cuộc tấn công từ chối dịch vụ phân tán (DDoS - đặc biệt là biến thể SYN Flood) ngày càng trở nên nguy hiểm và khó phát hiện.

Đề tài tập trung vào việc:
1. Nghiên cứu cơ sở lý thuyết mật mã và tính toán lượng tử ứng dụng trong an toàn thông tin (Quantum Machine Learning - QML).
2. Tái hiện và đo lường thực nghiệm (benchmarking) hiệu năng phát hiện tấn công DDoS trên bộ dữ liệu chuẩn `CICDDoS2019`.
3. So sánh đối đầu giữa các mô hình học máy Cổ điển (Classical ML) và mô hình Học máy Lượng tử (Quantum ML) trong kỷ nguyên thiết bị lượng tử có nhiễu trung gian (NISQ - Noisy Intermediate-Scale Quantum).

---

## 2. Các Mô Hình Được Đánh Giá

### Mô Hình Cổ Điển (Classical Models):
- **Classical SVM (Support Vector Machine):** Sử dụng RBF Kernel.
- **Classical ANN (Artificial Neural Network):** Mạng MLP kiến trúc 8 x 8 x 16 x 8 x 1 nơ-ron.

### Mô Hình Học Máy Lượng Tử (Quantum Models):
- **QSVM (Quantum Support Vector Machine):**
  - Angle Embedding Quantum Kernel (8 qubits)
  - Amplitude Embedding Quantum Kernel (3 qubits)
  - ZZ Feature Map Quantum Kernel
- **QNN / VQC (Variational Quantum Classifier):**
  - Angle Embedding kết hợp Two-Local Variational Form
  - Angle / Amplitude kết hợp Tree Tensor Network Form

---

## 3. Cấu Trúc Thư Mục Dự Án

```
qml-ddos/
├── TAI_LIEU_LY_THUYET_DE_TAI_QML_DDOS.md  # Tài liệu lý thuyết chi tiết từ A-Z theo yêu cầu môn học
├── download_syn_dataset.py               # Script tải bộ dữ liệu gốc CICDDoS2019
├── run_dataset_preprocessing.py          # Script tiền xử lý, chuẩn hóa và trích chọn 8 đặc trưng tối ưu
├── run_benchmark_all.py                  # Script chạy toàn bộ pipeline benchmark tự động
├── export_paper_and_reproduction_benchmark.py # Script xuất số liệu đối sánh thực nghiệm
├── dataset/                              # Thư mục dữ liệu mẫu
│   ├── tr_day_syn_tiny.csv               # Bộ train cực nhỏ (chạy test nhanh)
│   ├── test_day_syn_tiny.csv             # Bộ test cực nhỏ
│   ├── tr_day_syn_small.csv              # Bộ train mẫu chuẩn
│   └── test_day_syn_small.csv            # Bộ test mẫu chuẩn
├── results/                              # Kết quả xuất tự động (biểu đồ và số liệu)
│   ├── roc_curves_all_models.png         # Đường cong ROC-AUC so sánh tất cả mô hình
│   ├── confusion_matrices_all_models.png # Ma trận nhầm lẫn (Confusion Matrix)
│   ├── quantum_circuit_angle_embedding.png # Sơ đồ mạch lượng tử Angle Embedding
│   ├── quantum_circuit_vqc_twolocal.png    # Sơ đồ mạch lượng tử VQC Two-Local
│   ├── benchmark_metrics_comparison.png  # Biểu đồ so sánh Accuracy, F1, Precision, Recall
│   ├── benchmark_time_comparison.png     # Biểu đồ thời gian huấn luyện và dự đoán
│   └── bang_tong_hop_benchmark_chinh_thuc_bai_bao.csv # Bảng số liệu chi tiết
└── dev/                                  # Các notebook Jupyter thực nghiệm từng phần
```

---

## 4. Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### Yêu Cầu Môi Trường:
- Python 3.9 - 3.11
- Cài đặt các thư viện phụ thuộc:
```bash
pip install qiskit qiskit-machine-learning qiskit-aer
pip install scikit-learn pandas numpy matplotlib seaborn pylatexenc
```

### Chạy Pipeline Benchmark Toàn Diện:
Chỉ cần chạy lệnh sau để tự động huấn luyện, đánh giá và sinh toàn bộ biểu đồ:
```bash
python run_benchmark_all.py
```
Toàn bộ biểu đồ phân tích và file số liệu sẽ tự động được lưu vào thư mục `results/`.

### Xem Tài Liệu Lý Thuyết Đầy Đủ:
Mở file `TAI_LIEU_LY_THUYET_DE_TAI_QML_DDOS.md` để xem chi tiết cơ sở toán học lượng tử, phân tích 8 đặc trưng chọn lọc và lý giải nguyên nhân rào cản tính toán trong kỷ nguyên NISQ.
