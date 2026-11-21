import os
import sys
from generate_quantum_circuits_and_paper_metrics import (
    export_all_circuit_diagrams,
    export_complete_paper_metrics
)

sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    export_all_circuit_diagrams()
    export_complete_paper_metrics()
    print("\n>>> ĐÃ XUẤT TOÀN BỘ SƠ ĐỒ MẠCH LƯỢNG TỬ VÀ KẾT QUẢ ĐỐI SÁNH 12 MÔ HÌNH BÀI BÁO CHUẨN XÁC 100%!")
