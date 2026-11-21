import os
import sys
import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Study\HocKy6\MH&MM\qml-ddos"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_QUBITS = 8
K_TTN = math.ceil(math.log2(N_QUBITS))

# ==============================================================================
# 1. ĐỊNH NGHĨA CÁC MẠCH LƯỢNG TỬ CHUẨN XÁC 100% THEO MÃ NGUỒN BÀI BÁO GỐC
# ==============================================================================

def ZZFeatureMap(nqubits, data):
    nload = min(len(data), nqubits)
    for i in range(nload):
        qml.Hadamard(i)
        qml.RZ(2.0 * data[i], wires=i)
    for pair in list(combinations(range(nload), 2)):
        q0, q1 = pair[0], pair[1]
        qml.CZ(wires=[q0, q1])
        qml.RZ(2.0 * (np.pi - data[q0]) * (np.pi - data[q1]), wires=q1)
        qml.CZ(wires=[q0, q1])

def QNNTwoLocal(nqubits, theta, repetitions=1):
    for r in range(repetitions):
        for i in range(nqubits):
            qml.RY(theta[r * nqubits + i], wires=i)
        for i in range(nqubits - 1):
            qml.CNOT(wires=[i, i + 1])
    for i in range(nqubits):
        qml.RY(theta[repetitions * nqubits + i], wires=i)

def QNNTreeTensor(k, theta):
    nqubits = 2 ** k
    theta_i = 0
    for i in range(nqubits):
        qml.RY(theta[theta_i], wires=i)
        theta_i += 1
    for r in range(1, k + 1):
        for s in range(2 ** (k - r)):
            ctrl = 1 + s * (2 ** r) + (2 ** (r - 1)) - 1
            targ = 1 + s * (2 ** r) - 1
            qml.CNOT(wires=[ctrl, targ])
            qml.RY(theta[theta_i], wires=targ)
            theta_i += 1

state_0 = np.array([[1.0], [0.0]])
M_HERMITIAN = state_0 @ np.conj(state_0).T

dev_8 = qml.device("default.qubit", wires=N_QUBITS)
dev_3 = qml.device("default.qubit", wires=3)
dev_4 = qml.device("default.qubit", wires=4)

# Mạch 1: QSVM Angle Embedding Kernel (8 Qubit)
@qml.qnode(dev_8)
def circ_qsvm_angle(a, b):
    qml.AngleEmbedding(a, wires=range(N_QUBITS))
    qml.adjoint(qml.AngleEmbedding(b, wires=range(N_QUBITS)))
    return qml.probs(wires=range(N_QUBITS))

# Mạch 2: QSVM Amplitude Embedding Kernel (3 Qubit cho 8 đặc trưng)
@qml.qnode(dev_3)
def circ_qsvm_amplitude(a, b):
    qml.AmplitudeEmbedding(a, wires=range(3), pad_with=0.0, normalize=True)
    qml.adjoint(qml.AmplitudeEmbedding(b, wires=range(3), pad_with=0.0, normalize=True))
    return qml.probs(wires=range(3))

# Mạch 3: ZZ Feature Map (Minh họa trên 4 Qubit để thấy rõ cấu trúc cặp CZ-RZ-CZ không bị quá dài, và bản 8 Qubit)
@qml.qnode(dev_4)
def circ_zz_feature_map_4q(a):
    ZZFeatureMap(4, a[:4])
    return qml.probs(wires=range(4))

# Mạch 4: QNN Two-Local + Angle Embedding (8 Qubit, 16 tham số)
@qml.qnode(dev_8)
def circ_qnn_twolocal(x, theta):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    QNNTwoLocal(N_QUBITS, theta, repetitions=1)
    return qml.expval(qml.Hermitian(M_HERMITIAN, wires=[0]))

# Mạch 5: QNN Tree-Tensor Network + Angle Embedding (8 Qubit, 15 tham số)
@qml.qnode(dev_8)
def circ_qnn_treetensor(x, theta):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    QNNTreeTensor(K_TTN, theta)
    return qml.expval(qml.Hermitian(M_HERMITIAN, wires=[0]))

def export_all_circuit_diagrams():
    print("=== ĐANG VẼ VÀ XUẤT BỘ SƠ ĐỒ MẠCH LƯỢNG TỬ CHUẨN BÀI BÁO ===")
    sample_x = np.round(np.linspace(0.15, 0.85, N_QUBITS), 2)
    sample_z = np.round(np.linspace(0.25, 0.95, N_QUBITS), 2)
    theta_tl = np.round(np.linspace(0.1, 1.6, 16), 2)
    theta_ttn = np.round(np.linspace(0.1, 1.5, 15), 2)

    # 1. QSVM Angle Embedding Kernel
    fig, ax = qml.draw_mpl(circ_qsvm_angle, decimals=2, style="black_white")(sample_x, sample_z)
    fig.suptitle("Sơ đồ mạch 1: QSVM Angle Embedding Kernel — U(x) * U†(z) (8 Qubits, Depth = 2, CNOT = 0)",
                 fontsize=12, fontweight="bold", y=1.05)
    p1 = os.path.join(RESULTS_DIR, "circuit_1_qsvm_angle_kernel.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Cập nhật luôn file cũ để tương thích ngược
    fig, ax = qml.draw_mpl(circ_qsvm_angle, decimals=2, style="black_white")(sample_x, sample_z)
    fig.suptitle("Sơ đồ mạch QSVM Angle Embedding Kernel — U(x) * U†(z) (8 Qubits)",
                 fontsize=12, fontweight="bold", y=1.05)
    fig.savefig(os.path.join(RESULTS_DIR, "quantum_circuit_angle_embedding.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p1}")

    # 2. QSVM Amplitude Embedding Kernel (3 Qubits)
    fig, ax = qml.draw_mpl(circ_qsvm_amplitude, decimals=2, style="black_white")(sample_x, sample_z)
    fig.suptitle("Sơ đồ mạch 2: QSVM Amplitude Embedding Kernel — Mã hóa 8 đặc trưng vào 2^3 = 8 biên độ của 3 Qubits",
                 fontsize=11, fontweight="bold", y=1.08)
    p2 = os.path.join(RESULTS_DIR, "circuit_2_qsvm_amplitude_kernel.png")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p2}")

    # 3. ZZ Feature Map
    fig, ax = qml.draw_mpl(circ_zz_feature_map_4q, decimals=2, style="black_white")(sample_x[:4])
    fig.suptitle("Sơ đồ mạch 3: Cấu trúc ZZ Feature Map — Tầng H-RZ nối tiếp chuỗi vướng víu cặp CZ-RZ(2(π-xi)(π-xj))-CZ",
                 fontsize=11, fontweight="bold", y=1.08)
    p3 = os.path.join(RESULTS_DIR, "circuit_3_zz_feature_map.png")
    fig.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p3}")

    # 4. QNN Two-Local Ansatz (8 Qubits, 16 params)
    fig, ax = qml.draw_mpl(circ_qnn_twolocal, decimals=2, style="black_white")(sample_x, theta_tl)
    fig.suptitle("Sơ đồ mạch 4: Mạng QNN Two-Local + Angle Embedding (8 Qubits, 16 tham số RY, 7 cổng CNOT chuỗi bậc thang)",
                 fontsize=11.5, fontweight="bold", y=1.04)
    p4 = os.path.join(RESULTS_DIR, "circuit_4_qnn_twolocal_angle.png")
    fig.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Cập nhật luôn file cũ
    fig, ax = qml.draw_mpl(circ_qnn_twolocal, decimals=2, style="black_white")(sample_x, theta_tl)
    fig.suptitle("Sơ đồ mạch QNN Two-Local + Angle Embedding (8 Qubits, 16 tham số RY, 7 cổng CNOT)",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(os.path.join(RESULTS_DIR, "quantum_circuit_vqc_twolocal.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p4}")

    # 5. QNN Tree-Tensor Network Ansatz (8 Qubits, 15 params)
    fig, ax = qml.draw_mpl(circ_qnn_treetensor, decimals=2, style="black_white")(sample_x, theta_ttn)
    fig.suptitle("Sơ đồ mạch 5: Mạng QNN Tree-Tensor Network (TTN) + Angle Embedding (8 Qubits, 15 tham số RY, Cây nhị phân K=3 tầng)",
                 fontsize=11.5, fontweight="bold", y=1.04)
    p5 = os.path.join(RESULTS_DIR, "circuit_5_qnn_treetensor_angle.png")
    fig.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {p5}")

    # 6. Tạo bảng so sánh thông số kỹ thuật mạch (Circuit Architecture Summary CSV)
    circuit_specs = [
        {
            "Kiến trúc Mạch": "QSVM - Angle Embedding Kernel",
            "Loại Mô hình": "Quantum Kernel (QSVM)",
            "Số Qubit": 8,
            "Cổng Nhúng (Feature Map)": "8 cổng RX(x_i) + 8 cổng RX†(z_i)",
            "Số cổng 2-Qubit (CNOT/CZ)": 0,
            "Số tham số huấn luyện (Theta)": 0,
            "Độ sâu mạch (Circuit Depth)": 2,
            "Phép đo đầu ra": "Xác suất P(|00000000>) = |<psi(x)|psi(z)>|^2"
        },
        {
            "Kiến trúc Mạch": "QSVM - Amplitude Embedding Kernel",
            "Loại Mô hình": "Quantum Kernel (QSVM)",
            "Số Qubit": 3,
            "Cổng Nhúng (Feature Map)": "MottonenStatePreparation (2^3 = 8 biên độ)",
            "Số cổng 2-Qubit (CNOT/CZ)": 14,
            "Số tham số huấn luyện (Theta)": 0,
            "Độ sâu mạch (Circuit Depth)": 18,
            "Phép đo đầu ra": "Xác suất P(|000>) = |<psi(x)|psi(z)>|^2"
        },
        {
            "Kiến trúc Mạch": "QSVM - ZZ Feature Map Kernel",
            "Loại Mô hình": "Quantum Kernel (QSVM)",
            "Số Qubit": 8,
            "Cổng Nhúng (Feature Map)": "8 H + 8 RZ + 28 cặp (CZ - RZ - CZ) x 2 (Adjoint)",
            "Số cổng 2-Qubit (CNOT/CZ)": 112,
            "Số tham số huấn luyện (Theta)": 0,
            "Độ sâu mạch (Circuit Depth)": 88,
            "Phép đo đầu ra": "Xác suất P(|00000000>) = |<psi(x)|psi(z)>|^2"
        },
        {
            "Kiến trúc Mạch": "QNN - Two-Local + Angle Embedding",
            "Loại Mô hình": "Variational QNN (VQC)",
            "Số Qubit": 8,
            "Cổng Nhúng (Feature Map)": "8 cổng RX(x_i)",
            "Số cổng 2-Qubit (CNOT/CZ)": 7,
            "Số tham số huấn luyện (Theta)": 16,
            "Độ sâu mạch (Circuit Depth)": 10,
            "Phép đo đầu ra": "Kỳ vọng <q0 | M | q0> với M = |0><0|"
        },
        {
            "Kiến trúc Mạch": "QNN - Tree-Tensor (TTN) + Angle Embedding",
            "Loại Mô hình": "Variational QNN (VQC)",
            "Số Qubit": 8,
            "Cổng Nhúng (Feature Map)": "8 cổng RX(x_i)",
            "Số cổng 2-Qubit (CNOT/CZ)": 7,
            "Số tham số huấn luyện (Theta)": 15,
            "Độ sâu mạch (Circuit Depth)": 8,
            "Phép đo đầu ra": "Kỳ vọng <q0 | M | q0> với M = |0><0|"
        },
        {
            "Kiến trúc Mạch": "QNN - Two-Local + ZZ Feature Map",
            "Loại Mô hình": "Variational QNN (VQC)",
            "Số Qubit": 8,
            "Cổng Nhúng (Feature Map)": "8 H + 8 RZ + 28 cặp (CZ - RZ - CZ)",
            "Số cổng 2-Qubit (CNOT/CZ)": 63,
            "Số tham số huấn luyện (Theta)": 16,
            "Độ sâu mạch (Circuit Depth)": 52,
            "Phép đo đầu ra": "Kỳ vọng <q0 | M | q0> với M = |0><0|"
        },
        {
            "Kiến trúc Mạch": "QNN - Tree-Tensor (TTN) + ZZ Feature Map",
            "Loại Mô hình": "Variational QNN (VQC)",
            "Số Qubit": 8,
            "Cổng Nhúng (Feature Map)": "8 H + 8 RZ + 28 cặp (CZ - RZ - CZ)",
            "Số cổng 2-Qubit (CNOT/CZ)": 63,
            "Số tham số huấn luyện (Theta)": 15,
            "Độ sâu mạch (Circuit Depth)": 50,
            "Phép đo đầu ra": "Kỳ vọng <q0 | M | q0> với M = |0><0|"
        },
        {
            "Kiến trúc Mạch": "QNN - Two-Local + Amplitude (3-Qubit / 4-Qubit)",
            "Loại Mô hình": "Variational QNN (VQC)",
            "Số Qubit": 3,
            "Cổng Nhúng (Feature Map)": "MottonenStatePreparation (Chuẩn hóa L2)",
            "Số cổng 2-Qubit (CNOT/CZ)": 9,
            "Số tham số huấn luyện (Theta)": 6,
            "Độ sâu mạch (Circuit Depth)": 14,
            "Phép đo đầu ra": "Kỳ vọng <q0 | M | q0> (Dính Barren Plateaus)"
        }
    ]
    df_specs = pd.DataFrame(circuit_specs)
    specs_csv = os.path.join(RESULTS_DIR, "bang_thong_so_kien_truc_mach_luong_tu.csv")
    df_specs.to_csv(specs_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVED] Bảng thông số kỹ thuật mạch: {specs_csv}")

# ==============================================================================
# 2. XUẤT BẢNG CHỈ SỐ ĐẦY ĐỦ TỪ 12 NOTEBOOK GỐC CỦA TÁC GIẢ BÀI BÁO
# ==============================================================================
def export_complete_paper_metrics():
    print("\n=== ĐANG XUẤT BẢNG ĐẦY ĐỦ CHỈ SỐ TỪ 12 NOTEBOOK BÀI BÁO GỐC ===")
    data_full_benchmark = [
        {
            "Mô hình": "Classical SVM (RBF)",
            "Họ giải thuật": "Classical ML",
            "Tập dữ liệu": "FULL (80k Train / 10k Test)",
            "Số Qubit": 0,
            "Độ chính xác (Acc)": 0.9926,
            "Precision (Macro)": 0.9700,
            "Recall (Macro)": 1.0000,
            "F1-Score (Macro)": 0.9800,
            "F1 (Syn - Lớp 0)": 1.0000,
            "F1 (Benign - Lớp 1)": 0.9600,
            "ROC-AUC": 0.9954,
            "Thời gian Huấn luyện (s)": 0.98,
            "Thời gian Suy luận (s)": 0.13,
            "Ghi chú": "Mô hình cổ điển nhanh nhất, cân bằng hoàn hảo trên 100.000 mẫu"
        },
        {
            "Mô hình": "Classical ANN (MLP 8-8-16-8-1)",
            "Họ giải thuật": "Classical ML",
            "Tập dữ liệu": "FULL (80k Train / 10k Test)",
            "Số Qubit": 0,
            "Độ chính xác (Acc)": 0.9941,
            "Precision (Macro)": 0.9700,
            "Recall (Macro)": 0.9900,
            "F1-Score (Macro)": 0.9800,
            "F1 (Syn - Lớp 0)": 1.0000,
            "F1 (Benign - Lớp 1)": 0.9700,
            "ROC-AUC": 0.9988,
            "Thời gian Huấn luyện (s)": 21.00,
            "Thời gian Suy luận (s)": 0.39,
            "Ghi chú": "Mạng nơ-ron sâu 361 tham số, đạt Accuracy và ROC-AUC cao nhất toàn bài báo"
        },
        {
            "Mô hình": "QSVM (Angle Embedding)",
            "Họ giải thuật": "Quantum SVM",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 8,
            "Độ chính xác (Acc)": 0.9900,
            "Precision (Macro)": 0.9500,
            "Recall (Macro)": 0.9900,
            "F1-Score (Macro)": 0.9700,
            "F1 (Syn - Lớp 0)": 0.9900,
            "F1 (Benign - Lớp 1)": 0.9500,
            "ROC-AUC": 0.9944,
            "Thời gian Huấn luyện (s)": 1265.59,
            "Thời gian Suy luận (s)": 1576.28,
            "Ghi chú": "Mô hình lượng tử xuất sắc nhất (mạch nông Depth=2, 0 CNOT, kháng nhiễu tốt nhất)"
        },
        {
            "Mô hình": "QSVM (Amplitude Embedding)",
            "Họ giải thuật": "Quantum SVM",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 3,
            "Độ chính xác (Acc)": 0.9700,
            "Precision (Macro)": 0.9300,
            "Recall (Macro)": 0.8900,
            "F1-Score (Macro)": 0.9100,
            "F1 (Syn - Lớp 0)": 0.9800,
            "F1 (Benign - Lớp 1)": 0.8400,
            "ROC-AUC": 0.8944,
            "Thời gian Huấn luyện (s)": 4432.02,
            "Thời gian Suy luận (s)": 5502.47,
            "Ghi chú": "Tiết kiệm qubit nhất (3 Qubit cho 8 chiều), nhưng tốn chi phí phân rã cổng khởi tạo"
        },
        {
            "Mô hình": "QSVM (ZZ Feature Map)",
            "Họ giải thuật": "Quantum SVM",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 8,
            "Độ chính xác (Acc)": 0.9700,
            "Precision (Macro)": 0.9800,
            "Recall (Macro)": 0.8500,
            "F1-Score (Macro)": 0.9000,
            "F1 (Syn - Lớp 0)": 0.9800,
            "F1 (Benign - Lớp 1)": 0.8200,
            "ROC-AUC": 0.8500,
            "Thời gian Huấn luyện (s)": 18115.92,
            "Thời gian Suy luận (s)": 23312.32,
            "Ghi chú": "Mạch sâu 112 cổng CZ, chạy lâu nhất (>11 giờ tổng cộng cho Train + Test)"
        },
        {
            "Mô hình": "QNN (Two-Local + Angle)",
            "Họ giải thuật": "Quantum NN (VQC)",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 8,
            "Độ chính xác (Acc)": 0.9700,
            "Precision (Macro)": 0.9800,
            "Recall (Macro)": 0.8500,
            "F1-Score (Macro)": 0.9000,
            "F1 (Syn - Lớp 0)": 0.9800,
            "F1 (Benign - Lớp 1)": 0.8200,
            "ROC-AUC": 0.9933,
            "Thời gian Huấn luyện (s)": 978.96,
            "Thời gian Suy luận (s)": 6.94,
            "Ghi chú": "16 tham số RY + 7 CNOT; Suy luận cực nhanh (6.94s) nhờ không phải tính ma trận Kernel"
        },
        {
            "Mô hình": "QNN (Tree-Tensor + Angle)",
            "Họ giải thuật": "Quantum NN (VQC)",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 8,
            "Độ chính xác (Acc)": 0.9700,
            "Precision (Macro)": 0.9800,
            "Recall (Macro)": 0.8500,
            "F1-Score (Macro)": 0.9000,
            "F1 (Syn - Lớp 0)": 0.9800,
            "F1 (Benign - Lớp 1)": 0.8200,
            "ROC-AUC": 0.9956,
            "Thời gian Huấn luyện (s)": 3607.15,
            "Thời gian Suy luận (s)": 8.78,
            "Ghi chú": "15 tham số cấu trúc cây phân cấp; đạt ROC-AUC cao nhất trong các mô hình lượng tử (0.9956)"
        },
        {
            "Mô hình": "QNN (Two-Local + ZZ Map)",
            "Họ giải thuật": "Quantum NN (VQC)",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 8,
            "Độ chính xác (Acc)": 0.9200,
            "Precision (Macro)": 0.7800,
            "Recall (Macro)": 0.9600,
            "F1-Score (Macro)": 0.8300,
            "F1 (Syn - Lớp 0)": 0.9500,
            "F1 (Benign - Lớp 1)": 0.7100,
            "ROC-AUC": 0.9889,
            "Thời gian Huấn luyện (s)": 2409.97,
            "Thời gian Suy luận (s)": 27.09,
            "Ghi chú": "Độ sâu mạch lớn làm giảm Precision lớp Benign xuống 0.56 (nhiều cảnh báo giả)"
        },
        {
            "Mô hình": "QNN (Tree-Tensor + ZZ Map)",
            "Họ giải thuật": "Quantum NN (VQC)",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 8,
            "Độ chính xác (Acc)": 0.9200,
            "Precision (Macro)": 0.7800,
            "Recall (Macro)": 0.9600,
            "F1-Score (Macro)": 0.8300,
            "F1 (Syn - Lớp 0)": 0.9500,
            "F1 (Benign - Lớp 1)": 0.7100,
            "ROC-AUC": 0.9911,
            "Thời gian Huấn luyện (s)": 2795.10,
            "Thời gian Suy luận (s)": 26.24,
            "Ghi chú": "Chứng minh hiệu năng QNN phụ thuộc vào Feature Map hơn là Ansatz (y hệt Two-Local + ZZ)"
        },
        {
            "Mô hình": "QNN (Two-Local + Amplitude 3Q)",
            "Họ giải thuật": "Quantum NN (VQC)",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 3,
            "Độ chính xác (Acc)": 0.4300,
            "Precision (Macro)": 0.4100,
            "Recall (Macro)": 0.2400,
            "F1-Score (Macro)": 0.3000,
            "F1 (Syn - Lớp 0)": 0.6000,
            "F1 (Benign - Lớp 1)": 0.0000,
            "ROC-AUC": 0.2100,
            "Thời gian Huấn luyện (s)": 125.96,
            "Thời gian Suy luận (s)": 6.13,
            "Ghi chú": "Sụp đổ do Barren Plateaus & mất thông tin biên độ (không phát hiện được mẫu Benign nào)"
        },
        {
            "Mô hình": "QNN (Two-Local + Amplitude 4Q Padding)",
            "Họ giải thuật": "Quantum NN (VQC)",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 4,
            "Độ chính xác (Acc)": 0.9000,
            "Precision (Macro)": 0.4500,
            "Recall (Macro)": 0.5000,
            "F1-Score (Macro)": 0.4700,
            "F1 (Syn - Lớp 0)": 0.9500,
            "F1 (Benign - Lớp 1)": 0.0000,
            "ROC-AUC": 0.7178,
            "Thời gian Huấn luyện (s)": 342.10,
            "Thời gian Suy luận (s)": 7.45,
            "Ghi chú": "Bẫy Accuracy giả tạo (90%): Mô hình đoán tất cả là lớp đa số Syn (F1 lớp Benign = 0.00)"
        },
        {
            "Mô hình": "QNN (Tree-Tensor + Amplitude 4Q Padding)",
            "Họ giải thuật": "Quantum NN (VQC)",
            "Tập dữ liệu": "SMALL (800 Train / 100 Test)",
            "Số Qubit": 4,
            "Độ chính xác (Acc)": 0.9000,
            "Precision (Macro)": 0.4500,
            "Recall (Macro)": 0.5000,
            "F1-Score (Macro)": 0.4700,
            "F1 (Syn - Lớp 0)": 0.9500,
            "F1 (Benign - Lớp 1)": 0.0000,
            "ROC-AUC": 0.7678,
            "Thời gian Huấn luyện (s)": 418.50,
            "Thời gian Suy luận (s)": 8.12,
            "Ghi chú": "Tương tự Two-Local 4Q: Rơi vào cực tiểu cục bộ lớp đa số do dữ liệu mất cân bằng 90:10"
        }
    ]

    df = pd.DataFrame(data_full_benchmark)
    csv_out = os.path.join(RESULTS_DIR, "bang_tong_hop_benchmark_chinh_thuc_bai_bao.csv")
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"[SAVED] Bảng tổng hợp 12 mô hình bài báo: {csv_out}")

    # Vẽ biểu đồ đối sánh toàn diện 4 chỉ số (Accuracy, Macro Precision, Macro Recall, Macro F1, ROC-AUC)
    plt.figure(figsize=(14, 6.5), dpi=300)
    df_melt = df.melt(
        id_vars=["Mô hình", "Họ giải thuật"],
        value_vars=["Độ chính xác (Acc)", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "ROC-AUC"],
        var_name="Chỉ số Đánh giá", value_name="Điểm số"
    )
    sns.barplot(data=df_melt, x="Mô hình", y="Điểm số", hue="Chỉ số Đánh giá", palette="viridis")
    plt.axhline(0.9, color="red", linestyle="--", alpha=0.6, label="Ngưỡng chuẩn 90%")
    plt.title("Đối Sánh Toàn Diện 5 Chỉ Số Hiệu Năng Trên Cả 12 Mô Hình Của Bài Báo (CICDDoS2019)",
              fontsize=13, fontweight="bold", pad=14)
    plt.ylim(0, 1.15)
    plt.ylabel("Điểm số [0.0 - 1.0]", fontsize=10, fontweight="bold")
    plt.xlabel("Mô hình Thực nghiệm (Classical vs QSVM vs QNN)", fontsize=10, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=8.8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower left", ncol=3, framealpha=0.92, fontsize=8.5)
    plt.tight_layout()
    p_metrics = os.path.join(RESULTS_DIR, "bieu_do_phan_hoa_day_du_bai_bao.png")
    plt.savefig(p_metrics)
    plt.close()
    print(f"[SAVED] Biểu đồ 5 chỉ số 12 mô hình: {p_metrics}")

    # Vẽ biểu đồ so sánh kép: Thời gian Huấn luyện vs Thời gian Suy luận (Thang đo Log10)
    plt.figure(figsize=(13, 5.8), dpi=300)
    df_time_melt = df.melt(
        id_vars=["Mô hình", "Họ giải thuật"],
        value_vars=["Thời gian Huấn luyện (s)", "Thời gian Suy luận (s)"],
        var_name="Giai đoạn Thực thi", value_name="Thời gian (Giây)"
    )
    sns.barplot(data=df_time_melt, x="Mô hình", y="Thời gian (Giây)", hue="Giai đoạn Thực thi", palette="magma")
    plt.yscale("log")
    plt.title("So Sánh Thời Gian Huấn Luyện vs Suy Luận (Thang Log10 Giây) — Phân Tích Đặc Tính QSVM vs QNN",
              fontsize=12.5, fontweight="bold", pad=14)
    plt.ylabel("Thời gian (giây) - Thang Log10", fontsize=10, fontweight="bold")
    plt.xlabel("Mô hình Thực nghiệm", fontsize=10, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=8.8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", framealpha=0.92)
    plt.tight_layout()
    p_time = os.path.join(RESULTS_DIR, "bieu_do_thoi_gian_huan_luyen_log10.png")
    plt.savefig(p_time)
    plt.close()
    print(f"[SAVED] Biểu đồ thời gian Train vs Inference: {p_time}")

if __name__ == "__main__":
    export_all_circuit_diagrams()
    export_complete_paper_metrics()
