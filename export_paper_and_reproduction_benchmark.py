import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = r"C:\Study\HocKy6\MH&MM\qml-ddos\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------------------------------
# DỮ LIỆU ĐỐI SÁNH THỰC NGHIỆM CHUẨN XÁC TỪ TOÀN BỘ 12 NOTEBOOK BÀI BÁO (TẬP LỚN FULL & SMALL)
# -------------------------------------------------------------------------------------
data_full_benchmark = [
    {
        "Mô hình": "Classical SVM",
        "Họ giải thuật": "Classical ML",
        "Tập dữ liệu": "FULL (100.000 mẫu)",
        "Độ chính xác (Acc)": 0.9947,
        "F1-Score": 0.9850,
        "Thời gian Huấn luyện (s)": 0.19,
        "Thời gian Suy luận (s)": 0.05,
        "Ghi chú": "Mô hình cổ điển chạy nhanh nhất, chính xác cao nhất"
    },
    {
        "Mô hình": "Classical ANN (MLP)",
        "Họ giải thuật": "Classical ML",
        "Tập dữ liệu": "FULL (100.000 mẫu)",
        "Độ chính xác (Acc)": 0.9941,
        "F1-Score": 0.9800,
        "Thời gian Huấn luyện (s)": 4.31,
        "Thời gian Suy luận (s)": 0.39,
        "Ghi chú": "Mạng nơ-ron sâu 8-8-16-8-1"
    },
    {
        "Mô hình": "QSVM (Angle Embedding)",
        "Họ giải thuật": "Quantum SVM",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.9900,
        "F1-Score": 0.9700,
        "Thời gian Huấn luyện (s)": 1265.59,
        "Thời gian Suy luận (s)": 1576.28,
        "Ghi chú": "Mô hình lượng tử xuất sắc nhất của bài báo"
    },
    {
        "Mô hình": "QSVM (Amplitude Embedding)",
        "Họ giải thuật": "Quantum SVM",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.9700,
        "F1-Score": 0.9100,
        "Thời gian Huấn luyện (s)": 4432.02,
        "Thời gian Suy luận (s)": 5502.47,
        "Ghi chú": "Mạch mã hóa biên độ 3 Qubit"
    },
    {
        "Mô hình": "QSVM (ZZ Feature Map)",
        "Họ giải thuật": "Quantum SVM",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.9700,
        "F1-Score": 0.9000,
        "Thời gian Huấn luyện (s)": 18115.92,
        "Thời gian Suy luận (s)": 23312.32,
        "Ghi chú": "Thời gian tính toán lâu nhất (hơn 5 giờ)"
    },
    {
        "Mô hình": "QNN (Two-Local + Angle)",
        "Họ giải thuật": "Quantum NN (VQC)",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.9700,
        "F1-Score": 0.9000,
        "Thời gian Huấn luyện (s)": 978.96,
        "Thời gian Suy luận (s)": 6.94,
        "Ghi chú": "Mạng nơ-ron biến phân 8 Qubit"
    },
    {
        "Mô hình": "QNN (Tree Tensor + Angle)",
        "Họ giải thuật": "Quantum NN (VQC)",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.9700,
        "F1-Score": 0.9000,
        "Thời gian Huấn luyện (s)": 3607.15,
        "Thời gian Suy luận (s)": 8.78,
        "Ghi chú": "Mạng Tensor dạng cây"
    },
    {
        "Mô hình": "QNN (Two-Local + ZZ Map)",
        "Họ giải thuật": "Quantum NN (VQC)",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.9200,
        "F1-Score": 0.8300,
        "Thời gian Huấn luyện (s)": 2409.97,
        "Thời gian Suy luận (s)": 27.09,
        "Ghi chú": "Bị suy giảm độ chính xác do nhiễu cổng CNOT"
    },
    {
        "Mô hình": "QNN (Tree Tensor + ZZ Map)",
        "Họ giải thuật": "Quantum NN (VQC)",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.9200,
        "F1-Score": 0.8300,
        "Thời gian Huấn luyện (s)": 2795.10,
        "Thời gian Suy luận (s)": 26.24,
        "Ghi chú": "Cấu trúc cây kết hợp ZZ Map"
    },
    {
        "Mô hình": "QNN (Two-Local + Amplitude)",
        "Họ giải thuật": "Quantum NN (VQC)",
        "Tập dữ liệu": "SMALL (1.000 mẫu)",
        "Độ chính xác (Acc)": 0.4300,
        "F1-Score": 0.3000,
        "Thời gian Huấn luyện (s)": 125.96,
        "Thời gian Suy luận (s)": 6.13,
        "Ghi chú": "Mô hình thất bại hoàn toàn, không hội tụ"
    }
]

df = pd.DataFrame(data_full_benchmark)

# 1. Lưu CSV
csv_out = os.path.join(RESULTS_DIR, "bang_tong_hop_benchmark_chinh_thuc_bai_bao.csv")
df.to_csv(csv_out, index=False, encoding='utf-8-sig')
print(f"[OK] Đã lưu bảng tổng hợp chính thức: {csv_out}")

# 2. Vẽ biểu đồ phân hóa F1-Score và Accuracy chính thức của bài báo
plt.figure(figsize=(12, 6), dpi=300)
df_melt = df.melt(id_vars=["Mô hình", "Họ giải thuật"], value_vars=["Độ chính xác (Acc)", "F1-Score"],
                  var_name="Chỉ số", value_name="Điểm số")

palette = {"Classical ML": "#1f77b4", "Quantum SVM": "#2ca02c", "Quantum NN (VQC)": "#d62728"}

ax = sns.barplot(data=df_melt, x="Mô hình", y="Điểm số", hue="Chỉ số", palette="tab10")
plt.title("Chuẩn Đối Sánh Đầy Đủ Tất Cả Mô Hình Cổ Điển vs Lượng Tử (Công Bố Bài Báo qml-ddos)", fontsize=13, fontweight='bold', pad=15)
plt.ylim(0, 1.15)
plt.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='Ngưỡng chuẩn 90%')
plt.ylabel("Điểm số [0.0 - 1.0]", fontsize=10, fontweight='bold')
plt.xlabel("Mô hình Thực nghiệm", fontsize=10, fontweight='bold')
plt.xticks(rotation=28, ha="right", fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc="lower left", framealpha=0.9)
plt.tight_layout()
p_metrics = os.path.join(RESULTS_DIR, "bieu_do_phan_hoa_day_du_bai_bao.png")
plt.savefig(p_metrics)
plt.close()
print(f"[OK] Đã lưu biểu đồ phân hóa F1-Score: {p_metrics}")

# 3. Vẽ biểu đồ so sánh thời gian thực thi (Thang đo Log10)
plt.figure(figsize=(11, 5.5), dpi=300)
sns.barplot(data=df, x="Mô hình", y="Thời gian Huấn luyện (s)", hue="Họ giải thuật", dodge=False, palette="Set2")
plt.title("So sánh Thời gian Huấn luyện Thực tế (Thang đo Log10 giây) - Tử Huyệt Kỷ Nguyên NISQ", fontsize=13, fontweight='bold', pad=15)
plt.yscale("log")
plt.ylabel("Thời gian Huấn luyện (giây) - Log10", fontsize=10, fontweight='bold')
plt.xlabel("Mô hình Thực nghiệm", fontsize=10, fontweight='bold')
plt.xticks(rotation=28, ha="right", fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
p_time = os.path.join(RESULTS_DIR, "bieu_do_thoi_gian_huan_luyen_log10.png")
plt.savefig(p_time)
plt.close()
print(f"[OK] Đã lưu biểu đồ thời gian: {p_time}")

print("\n>>> ĐÃ XUẤT TOÀN BỘ KẾT QUẢ ĐỐI SÁNH KHOA HỌC CHUẨN XÁC 100%!")
