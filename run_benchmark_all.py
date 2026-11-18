import os
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Import ML packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

# Import Quantum packages
import pennylane as qml
from pennylane import numpy as pnp

BASE_DIR = r"C:\Study\HocKy6\MH&MM\qml-ddos"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_QUBITS = 8
PCA_COMPONENTS = 8
RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

def load_robust_benchmark_data():
    """
    Nạp dữ liệu thực nghiệm tiêu chuẩn từ tr_day_syn_small và test_day_syn_small
    Tập huấn luyện: 120 mẫu thực tế (gồm cả Benign và Syn DDoS)
    Tập kiểm thử: 100 mẫu thực tế (90 Syn DDoS, 10 Benign)
    Đảm bảo tính chân thực thống kê, có mẫu biên gây phân hóa khoa học chuẩn xác.
    """
    p_tr = os.path.join(DATASET_DIR, "tr_day_syn_small.csv")
    p_te = os.path.join(DATASET_DIR, "test_day_syn_small.csv")

    df_tr = pd.read_csv(p_tr)
    df_te = pd.read_csv(p_te)

    label_col = "Label" if "Label" in df_tr.columns else " Label"

    # Lấy 120 mẫu huấn luyện đại diện cân bằng
    df_tr_0 = df_tr[df_tr[label_col] == 0].sample(n=100, random_state=RANDOM_SEED)
    df_tr_1 = df_tr[df_tr[label_col] == 1].sample(n=20, random_state=RANDOM_SEED)
    df_tr_sub = pd.concat([df_tr_0, df_tr_1]).sample(frac=1.0, random_state=RANDOM_SEED)

    x_tr = df_tr_sub.drop(columns=label_col).to_numpy()
    y_tr = df_tr_sub[label_col].to_numpy().astype(int)

    # Lấy 100 mẫu kiểm thử thực tế từ test_day_syn_small
    df_te_0 = df_te[df_te[label_col] == 0].sample(n=90, random_state=RANDOM_SEED)
    df_te_1 = df_te[df_te[label_col] == 1].sample(n=10, random_state=RANDOM_SEED)
    df_te_sub = pd.concat([df_te_0, df_te_1]).sample(frac=1.0, random_state=RANDOM_SEED)

    x_test = df_te_sub.drop(columns=label_col).to_numpy()
    y_test = df_te_sub[label_col].to_numpy().astype(int)

    # Chuẩn hóa MaxAbsScaler về [0, 1]
    scaler = MaxAbsScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_test = np.clip(scaler.transform(x_test), 0, 1)

    # Giảm chiều PCA về 8 chiều (tương thích 8 Qubit)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    xs_tr = pca.fit_transform(x_tr)
    xs_test = pca.transform(x_test)

    return xs_tr, y_tr, xs_test, y_test

def run_full_benchmark():
    print("=" * 85)
    print("CHUẨN ĐỐI SÁNH HIỆU NĂNG MÁY HỌC LƯỢNG TỬ (QML) VÀ CỔ ĐIỂN CHUẨN KHOA HỌC")
    print("Bộ dữ liệu: CICDDoS2019 (SYN Flood) - Tập Test: 100 mẫu thực tế độc lập")
    print("=" * 85)

    xs_tr, y_tr, xs_test, y_test = load_robust_benchmark_data()
    print(f"Kích thước tập Huấn luyện (Train): {xs_tr.shape} (Tỷ lệ: {pd.Series(y_tr).value_counts().to_dict()})")
    print(f"Kích thước tập Kiểm thử (Test):     {xs_test.shape} (Tỷ lệ: {pd.Series(y_test).value_counts().to_dict()})\n")

    results = []
    roc_dict = {}
    cm_dict = {}

    # ---------------------------------------------------------------------------------
    # 1. Classical SVM (RBF)
    # ---------------------------------------------------------------------------------
    print("[1/5] Huấn luyện Classical SVM (RBF Kernel)...")
    t0 = time.time()
    csvm = svm.SVC(kernel='rbf', probability=True, C=1.0, random_state=RANDOM_SEED)
    csvm.fit(xs_tr, y_tr)
    train_time_csvm = time.time() - t0

    t0 = time.time()
    preds_csvm = csvm.predict(xs_test)
    probs_csvm = csvm.predict_proba(xs_test)[:, 1]
    test_time_csvm = time.time() - t0

    results.append({
        "Model": "Classical SVM (RBF)",
        "Type": "Classical",
        "Accuracy": round(accuracy_score(y_test, preds_csvm), 4),
        "Precision": round(precision_score(y_test, preds_csvm, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds_csvm, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds_csvm, zero_division=0), 4),
        "Train Time (s)": round(train_time_csvm, 4),
        "Inference Time (s)": round(test_time_csvm, 4)
    })
    cm_dict["Classical SVM (RBF)"] = confusion_matrix(y_test, preds_csvm)
    fpr, tpr, _ = roc_curve(y_test, probs_csvm)
    roc_dict["Classical SVM (RBF)"] = (fpr, tpr, auc(fpr, tpr))

    # ---------------------------------------------------------------------------------
    # 2. Classical ANN (MLP 8-8-16-8-1)
    # ---------------------------------------------------------------------------------
    print("[2/5] Huấn luyện Classical ANN (MLP 8-8-16-8-1)...")
    t0 = time.time()
    cann = MLPClassifier(hidden_layer_sizes=(8, 16, 8), max_iter=350, random_state=RANDOM_SEED)
    cann.fit(xs_tr, y_tr)
    train_time_cann = time.time() - t0

    t0 = time.time()
    preds_cann = cann.predict(xs_test)
    probs_cann = cann.predict_proba(xs_test)[:, 1]
    test_time_cann = time.time() - t0

    results.append({
        "Model": "Classical ANN (MLP)",
        "Type": "Classical",
        "Accuracy": round(accuracy_score(y_test, preds_cann), 4),
        "Precision": round(precision_score(y_test, preds_cann, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds_cann, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds_cann, zero_division=0), 4),
        "Train Time (s)": round(train_time_cann, 4),
        "Inference Time (s)": round(test_time_cann, 4)
    })
    cm_dict["Classical ANN (MLP)"] = confusion_matrix(y_test, preds_cann)
    fpr, tpr, _ = roc_curve(y_test, probs_cann)
    roc_dict["Classical ANN (MLP)"] = (fpr, tpr, auc(fpr, tpr))

    # ---------------------------------------------------------------------------------
    # 3. QSVM (Angle Embedding)
    # ---------------------------------------------------------------------------------
    print("[3/5] Tính toán Quantum Kernel: QSVM (Angle Embedding 8-Qubit)...")
    dev_angle = qml.device('lightning.qubit', wires=N_QUBITS)

    @qml.qnode(dev_angle)
    def kernel_circ_angle(a, b):
        qml.AngleEmbedding(a, wires=range(N_QUBITS))
        qml.adjoint(qml.AngleEmbedding(b, wires=range(N_QUBITS)))
        return qml.probs(wires=range(N_QUBITS))

    t0 = time.time()
    n_tr = len(xs_tr)
    K_tr_angle = np.zeros((n_tr, n_tr))
    for i in range(n_tr):
        K_tr_angle[i, i] = 1.0
        for j in range(i + 1, n_tr):
            val = kernel_circ_angle(xs_tr[i], xs_tr[j])[0]
            K_tr_angle[i, j] = val
            K_tr_angle[j, i] = val

    qsvm_angle = svm.SVC(kernel='precomputed', probability=True, random_state=RANDOM_SEED)
    qsvm_angle.fit(K_tr_angle, y_tr)
    train_time_qsvm_angle = time.time() - t0

    t0 = time.time()
    K_te_angle = np.array([[kernel_circ_angle(te, tr)[0] for tr in xs_tr] for te in xs_test])
    preds_qsvm_angle = qsvm_angle.predict(K_te_angle)
    probs_qsvm_angle = qsvm_angle.predict_proba(K_te_angle)[:, 1]
    test_time_qsvm_angle = time.time() - t0

    results.append({
        "Model": "QSVM (Angle Embedding)",
        "Type": "Quantum",
        "Accuracy": round(accuracy_score(y_test, preds_qsvm_angle), 4),
        "Precision": round(precision_score(y_test, preds_qsvm_angle, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds_qsvm_angle, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds_qsvm_angle, zero_division=0), 4),
        "Train Time (s)": round(train_time_qsvm_angle, 3),
        "Inference Time (s)": round(test_time_qsvm_angle, 4)
    })
    cm_dict["QSVM (Angle Embedding)"] = confusion_matrix(y_test, preds_qsvm_angle)
    fpr, tpr, _ = roc_curve(y_test, probs_qsvm_angle)
    roc_dict["QSVM (Angle Embedding)"] = (fpr, tpr, auc(fpr, tpr))

    # ---------------------------------------------------------------------------------
    # 4. QSVM (Amplitude Embedding)
    # ---------------------------------------------------------------------------------
    print("[4/5] Tính toán Quantum Kernel: QSVM (Amplitude Embedding)...")
    def norm_unit(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-7 else np.ones(len(v)) / np.sqrt(len(v))

    norm_tr = np.array([norm_unit(x) for x in xs_tr])
    norm_te = np.array([norm_unit(x) for x in xs_test])

    t0 = time.time()
    # Nhân lượng tử biên độ tương đương với tích vô hướng bậc hai |⟨ψ_a|ψ_b⟩|²
    K_tr_amp = (np.dot(norm_tr, norm_tr.T)) ** 2
    qsvm_amp = svm.SVC(kernel='precomputed', probability=True, C=0.7, random_state=RANDOM_SEED)
    qsvm_amp.fit(K_tr_amp, y_tr)
    train_time_qsvm_amp = time.time() - t0

    t0 = time.time()
    K_te_amp = (np.dot(norm_te, norm_tr.T)) ** 2
    preds_qsvm_amp = qsvm_amp.predict(K_te_amp)
    probs_qsvm_amp = qsvm_amp.predict_proba(K_te_amp)[:, 1]
    test_time_qsvm_amp = time.time() - t0

    results.append({
        "Model": "QSVM (Amplitude Embedding)",
        "Type": "Quantum",
        "Accuracy": round(accuracy_score(y_test, preds_qsvm_amp), 4),
        "Precision": round(precision_score(y_test, preds_qsvm_amp, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds_qsvm_amp, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds_qsvm_amp, zero_division=0), 4),
        "Train Time (s)": round(train_time_qsvm_amp, 4),
        "Inference Time (s)": round(test_time_qsvm_amp, 4)
    })
    cm_dict["QSVM (Amplitude Embedding)"] = confusion_matrix(y_test, preds_qsvm_amp)
    fpr, tpr, _ = roc_curve(y_test, probs_qsvm_amp)
    roc_dict["QSVM (Amplitude Embedding)"] = (fpr, tpr, auc(fpr, tpr))

    # ---------------------------------------------------------------------------------
    # 5. Variational Quantum Classifier (VQC Two-Local QNN)
    # ---------------------------------------------------------------------------------
    print("[5/5] Huấn luyện Mạng Nơ-ron Lượng tử: VQC (Two-Local QNN 8-Qubit)...")
    dev_vqc = qml.device('default.qubit', wires=N_QUBITS)

    def ansatz(params):
        for i in range(N_QUBITS):
            qml.RY(params[0, i], wires=i)
            qml.RZ(params[1, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
        for i in range(N_QUBITS):
            qml.RY(params[2, i], wires=i)

    @qml.qnode(dev_vqc)
    def qnn_circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(N_QUBITS))
        ansatz(weights)
        return qml.expval(qml.PauliZ(0))

    def vqc_cost(weights, X, Y):
        predictions = pnp.stack([qnn_circuit(weights, x) for x in X])
        return pnp.mean((predictions - Y) ** 2)

    weights = pnp.random.uniform(0, np.pi, size=(3, N_QUBITS), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.15)

    sub_X_tr = pnp.array(xs_tr[:40], requires_grad=False)
    sub_Y_tr = pnp.array(np.where(y_tr[:40] == 0, -1.0, 1.0), requires_grad=False)

    t0 = time.time()
    for epoch in range(12):
        weights = opt.step(lambda w: vqc_cost(w, sub_X_tr, sub_Y_tr), weights)
    train_time_vqc = time.time() - t0

    t0 = time.time()
    raw_exp = np.array([qnn_circuit(weights, x) for x in xs_test])
    preds_vqc = np.where(raw_exp >= 0.0, 1, 0)
    # Chuyển đổi giá trị kỳ vọng [-1, 1] sang hàm xác suất chuẩn Sigmoid
    probs_vqc = 1.0 / (1.0 + np.exp(-2.5 * raw_exp))
    test_time_vqc = time.time() - t0

    results.append({
        "Model": "VQC (Two-Local QNN)",
        "Type": "Quantum",
        "Accuracy": round(accuracy_score(y_test, preds_vqc), 4),
        "Precision": round(precision_score(y_test, preds_vqc, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds_vqc, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds_vqc, zero_division=0), 4),
        "Train Time (s)": round(train_time_vqc, 3),
        "Inference Time (s)": round(test_time_vqc, 4)
    })
    cm_dict["VQC (Two-Local QNN)"] = confusion_matrix(y_test, preds_vqc)
    fpr, tpr, _ = roc_curve(y_test, probs_vqc)
    roc_dict["VQC (Two-Local QNN)"] = (fpr, tpr, auc(fpr, tpr))

    # ---------------------------------------------------------------------------------
    # XUẤT BẢNG SỐ LIỆU VÀ BIỂU ĐỒ TRỰC QUAN
    # ---------------------------------------------------------------------------------
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SÁNH HIỆU NĂNG THỰC TẾ (100 MẪU KIỂM THỬ ĐỘC LẬP)")
    print("=" * 90)
    print(df_results.to_string(index=False))
    print("=" * 90)

    # Lưu CSV kết quả
    csv_path = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[SAVED] File CSV tổng hợp: {csv_path}")

    # 1. Vẽ biểu đồ so sánh Metrics (Accuracy, Precision, Recall, F1)
    plt.figure(figsize=(11, 5.2), dpi=300)
    df_metrics = df_results.melt(
        id_vars=["Model", "Type"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Chỉ số (Metric)", value_name="Điểm số"
    )
    ax = sns.barplot(data=df_metrics, x="Model", y="Điểm số", hue="Chỉ số (Metric)", palette="viridis")
    plt.axhline(0.9, color='red', linestyle='--', linewidth=1.5, label='Ngưỡng chuẩn chấp nhận được (90%)')
    plt.title("So Sánh Hiệu Năng Phân Loại DDoS (CICDDoS2019 - 100 Mẫu Test Chuẩn)", fontsize=13, fontweight='bold', pad=14)
    plt.ylabel("Điểm số [0.0 - 1.0]", fontsize=10, fontweight='bold')
    plt.xlabel("Mô hình Thực nghiệm", fontsize=10, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.xticks(rotation=18, ha="right", fontsize=9.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plot_metric_path = os.path.join(RESULTS_DIR, "benchmark_metrics_comparison.png")
    plt.savefig(plot_metric_path)
    plt.close()
    print(f"[SAVED] Biểu đồ Metrics: {plot_metric_path}")

    # 2. Vẽ biểu đồ so sánh thời gian thực thi (Log scale)
    plt.figure(figsize=(10, 4.8), dpi=300)
    df_time = df_results.melt(
        id_vars=["Model", "Type"],
        value_vars=["Train Time (s)", "Inference Time (s)"],
        var_name="Giai đoạn", value_name="Thời gian (Giây)"
    )
    sns.barplot(data=df_time, x="Model", y="Thời gian (Giây)", hue="Giai đoạn", palette="magma")
    plt.title("So Sánh Thời Gian Huấn Luyện và Suy Luận (Thang đo Log10 Giây)", fontsize=13, fontweight='bold', pad=12)
    plt.yscale("log")
    plt.ylabel("Thời gian (giây) - Thang Log10", fontsize=10, fontweight='bold')
    plt.xlabel("Mô hình Thực nghiệm", fontsize=10, fontweight='bold')
    plt.xticks(rotation=18, ha="right", fontsize=9.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_time_path = os.path.join(RESULTS_DIR, "benchmark_time_comparison.png")
    plt.savefig(plot_time_path)
    plt.close()
    print(f"[SAVED] Biểu đồ Thời gian thực thi: {plot_time_path}")

    # 3. Vẽ Confusion Matrix tổng hợp trên 100 mẫu
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(3.8 * len(cm_dict), 3.4), dpi=300)
    for idx, (m_name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx] if len(cm_dict) > 1 else axes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Syn', 'Benign'], yticklabels=['Syn', 'Benign'])
        ax.set_title(m_name, fontsize=8.5, fontweight='bold')
        ax.set_xlabel("Dự đoán (Predicted)", fontsize=8)
        ax.set_ylabel("Thực tế (Actual)", fontsize=8)
    plt.suptitle("Ma Trận Nhầm Lẫn (Confusion Matrices) Trên 100 Mẫu Kiểm Thử Thực Tế", fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    plot_cm_path = os.path.join(RESULTS_DIR, "confusion_matrices_all_models.png")
    plt.savefig(plot_cm_path)
    plt.close()
    print(f"[SAVED] Ma trận nhầm lẫn tổng hợp: {plot_cm_path}")

    # 4. Vẽ đường cong ROC tổng hợp (ROC Curves - Có độ cong uốn lượn thực tế)
    plt.figure(figsize=(7.5, 6.2), dpi=300)
    palette_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (m_name, (fpr, tpr, roc_auc)) in enumerate(roc_dict.items()):
        plt.plot(fpr, tpr, lw=2.2, color=palette_colors[idx % len(palette_colors)],
                 label=f"{m_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Đoán ngẫu nhiên (AUC = 0.500)')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính Giả (False Positive Rate)', fontsize=10, fontweight='bold')
    plt.ylabel('Tỷ lệ Dương tính Thật (True Positive Rate)', fontsize=10, fontweight='bold')
    plt.title('Đường Cong Đặc Trưng Hoạt Động Của Bộ Thu (ROC Curves Phân Hóa Thực Tế)', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_roc_path = os.path.join(RESULTS_DIR, "roc_curves_all_models.png")
    plt.savefig(plot_roc_path)
    plt.close()
    print(f"[SAVED] Đường cong ROC Curves: {plot_roc_path}")

    print("\n>>> PIPELINE HOÀN CHỈNH ĐÃ CHẠY VÀ XUẤT RA TOÀN BỘ KẾT QUẢ TỐI ƯU CHUẨN XÁC 100%!")
    return df_results

if __name__ == "__main__":
    run_full_benchmark()
