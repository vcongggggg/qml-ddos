# TÀI LIỆU TOÀN DIỆN CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN NGHIÊN CỨU
## ĐỀ TÀI 2.7: BENCHMARKING QUANTUM MACHINE LEARNING FOR DETECTING DDOS CYBER-ATTACKS: A NISQ-ERA EVALUATION

* **Học phần:** Mã hóa và Mật mã học (CC102)
* **Giảng viên hướng dẫn:** PGS. TS. Nguyễn Tấn Khôi
* **Sinh viên thực hiện:** 
  1. Ngô Văn Công — MSSV: 102230284 (Nhóm MH&MM23.12 / 22Nh14)
  2. Nguyễn Công Minh — MSSV: 102230306
* **Mã nguồn tham khảo chính thức:** [https://github.com/ralorin/qml-ddos](https://github.com/ralorin/qml-ddos)
* **Nền tảng thực thi & mô phỏng:** PennyLane, Qiskit & Qiskit-Aer, PyTorch, Scikit-learn

---

# MỤC LỤC TỔNG QUAN

1. [PHẦN I: TỔNG QUAN ĐỀ TÀI & NHIỆM VỤ NGHIÊN CỨU](#phần-i-tổng-quan-đề-tài--nhiệm-vụ-nghiên-cứu)
2. [PHẦN II: CƠ SỞ LÝ THUYẾT AN NINH MẠNG & TẤN CÔNG DDOS](#phần-ii-cơ-sở-lý-thuyết-an-ninh-mạng--tấn-công-ddos)
3. [PHẦN III: NỀN TẢNG ĐIỆN TOÁN LƯỢNG TỬ & KỶ NGUYÊN NISQ](#phần-iii-nền-tảng-điện-toán-lượng-tử--kỷ-nguyên-nisq)
4. [PHẦN IV: CÁC KỸ THUẬT NHÚNG DỮ LIỆU LƯỢNG TỬ (QUANTUM FEATURE MAPS)](#phần-iv-các-kỹ-thuật-nhúng-dữ-liệu-lượng-tử-quantum-feature-maps)
5. [PHẦN V: CÁC MÔ HÌNH HỌC MÁY LƯỢNG TỬ (QSVM & QNN/VQC)](#phần-v-các-mô-hình-học-máy-lượng-tử-qsvm--qnnvqc)
6. [PHẦN VI: CÁC MÔ HÌNH HỌC MÁY CỔ ĐIỂN ĐỐI CHUẨN (CLASSICAL BASELINES)](#phần-vi-các-mô-hình-học-máy-cổ-điển-đối-chuẩn-classical-baselines)
7. [PHẦN VII: TỔNG HỢP CÁC BÀI BÁO KHOA HỌC LIÊN QUAN (LITERATURE REVIEW)](#phần-vii-tổng-hợp-các-bài-báo-khoa-học-liên-quan-literature-review)
8. [PHẦN VIII: HƯỚNG DẪN TRIỂN KHAI SOURCE CODE & QUY TRÌNH THỰC NGHIỆM](#phần-viii-hướng-dẫn-triển-khai-source-code--quy-trình-thực-nghiệm)
9. [PHẦN IX: BỘ CÂU HỎI VẤN ĐÁP BẢO VỆ ĐỒ ÁN VỚI THẦY KHÔI (Q&A)](#phần-ix-bộ-câu-hỏi-vấn-đáp-bảo-vệ-đồ-án-với-thầy-khôi-qa)

---

# PHẦN I: TỔNG QUAN ĐỀ TÀI & NHIỆM VỤ NGHIÊN CỨU

### 1. Bối cảnh đề tài
Sự bùng nổ của mạng lưới Internet vạn vật (IoT) kết nối hàng chục tỷ thiết bị thông minh nhưng có năng lực bảo mật yếu kém, tạo điều kiện cho tin tặc xây dựng các mạng máy tính ma (Botnet) khổng lồ để phát động các đợt tấn công từ chối dịch vụ phân tán (DDoS). Mặc dù các giải thuật Học máy Cổ điển (Classical Machine Learning) đã được ứng dụng rộng rãi trong các hệ thống phát hiện xâm nhập mạng (IDS), tiềm năng thực tế của **Học máy Lượng tử (Quantum Machine Learning - QML)** cho nhiệm vụ này vẫn chưa được kiểm chứng hệ thống dưới các điều kiện thực nghiệm có kiểm soát.

Đề tài **2.7: "Benchmarking quantum machine learning for detecting DDoS cyber-attacks: a NISQ-era evaluation"** thực hiện nghiên cứu đối chuẩn thực nghiệm chuyên sâu, so sánh trực diện giữa các mô hình học máy có giám sát Cổ điển và Lượng tử trên tập dữ liệu chuẩn quốc tế **CICDDoS2019**.

### 2. Ba nhiệm vụ trọng tâm do Giảng viên yêu cầu:
1. **Tìm hiểu lý thuyết:** Nắm vững bản chất toán học và vật lý của cơ chế tấn công SYN Flood, các cổng lượng tử, kỹ thuật mã hóa dữ liệu cổ điển vào không gian Hilbert (Feature Maps), giải thuật QSVM và mạng nơ-ron lượng tử biến phân (VQC).
2. **Tìm kiếm các bài báo liên quan cùng chủ đề:** Tổng hợp các công trình khoa học quốc tế uy tín làm nền tảng lý luận (IBM Quantum, Google Quantum AI, Nature, IEEE).
3. **Triển khai và chạy source tham khảo:** Vận hành thành công kho mã nguồn [ralorin/qml-ddos](https://github.com/ralorin/qml-ddos), tự động hóa pipeline tiền xử lý dữ liệu và xuất biểu đồ đánh giá đối chuẩn khoa học.

---

# PHẦN II: CƠ SỞ LÝ THUYẾT AN NINH MẠNG & TẤN CÔNG DDOS

### 1. Cơ chế giao thức TCP Bắt tay Ba bước (Three-way Handshake)
Trong mạng máy tính truyền thống, giao thức điều khiển truyền vận (TCP) thiết lập kết nối tin cậy giữa Máy khách (Client) và Máy chủ (Server) thông qua quy trình bắt tay 3 bước:
* **Bước 1 (SYN):** Client gửi gói tin có cờ `SYN = 1` kèm số thứ tự khởi tạo (Sequence Number) yêu cầu mở kết nối.
* **Bước 2 (SYN-ACK):** Server nhận được, cấp phát bộ đệm bộ nhớ (Buffer / Transmission Control Block - TCB) và gửi lại gói tin có cờ `SYN = 1, ACK = 1`. Kết nối lúc này ở trạng thái "nửa mở" (Half-open connection).
* **Bước 3 (ACK):** Client gửi lại gói tin có cờ `ACK = 1` để xác nhận. Kết nối chính thức hoàn tất (Established) và bắt đầu truyền dữ liệu.

### 2. Bản chất Tấn công DDoS dạng SYN Flood
* **Cơ chế khai thác:** Kẻ tấn công lợi dụng điểm yếu trong việc duy trì trạng thái nửa mở của Server. Botnet sử dụng các địa chỉ IP giả mạo (IP Spoofing) đồng loạt gửi hàng triệu gói tin `SYN` đến Server, nhưng **cố tình không bao giờ phản hồi gói tin `ACK` cuối cùng**.
* **Hậu quả:** Server mở hàng ngàn ô nhớ tạm trong hàng đợi kết nối (SYN Backlog Queue) và giữ trạng thái chờ đợi cho đến khi hết thời gian chờ (Timeout). Khi hàng đợi đầy, Server cạn kiệt tài nguyên bộ nhớ RAM và từ chối toàn bộ các yêu cầu kết nối hợp lệ của người dùng bình thường.

### 3. Bộ dữ liệu chuẩn quốc tế CICDDoS2019
Được phát triển bởi Viện An ninh mạng Canada (Canadian Institute for Cybersecurity - CIC, Đại học New Brunswick):
* Ghi lại lưu lượng mạng PCAP thực tế mô phỏng môi trường tấn công mạng hiện đại, được trích xuất thành các file CSV với hơn **80 đặc trưng luồng mạng (Network Flow Features)**.
* Gồm 2 ngày thu thập:
  * Ngày 12/01/2019 (`01-12/Syn.csv`): Sử dụng làm **tập Huấn luyện (Training set)**.
  * Ngày 11/03/2019 (`03-11/Syn.csv`): Sử dụng làm **tập Kiểm thử (Testing set)** độc lập.
* **Mã hóa nhãn (Label Encoding):**
  * Nhãn `0` (`Syn`): Đại diện cho gói tin tấn công DDoS SYN Flood (chiếm 90% dữ liệu thực nghiệm).
  * Nhãn `1` (`Benign`): Đại diện cho lưu lượng mạng thông thường hợp lệ (chiếm 10% dữ liệu).

### 4. Quy trình Tiền xử lý Dữ liệu & Giảm chiều PCA
1. **Làm sạch dữ liệu:** Loại bỏ các cột metadata mang tính định danh tĩnh không có tính quy luật tổng quát hóa: `Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Timestamp`, `SimillarHTTP`.
2. **Chuẩn hóa giá trị (MaxAbsScaler):** Dữ liệu mạng có khoảng giá trị chênh lệch rất lớn (từ 0 bytes đến hàng triệu bytes). Bộ chuẩn hóa co giãn tất cả giá trị về đoạn `[0, 1]`, đảm bảo tương thích hoàn hảo với miền góc quay lượng tử.
3. **Phân tích thành phần chính (PCA - Principal Component Analysis):**
   * *Nguyên lý:* Không vứt bỏ cột nào, mà thực hiện phép chiếu ma trận trực giao kết hợp tuyến tính cả 80 thuộc tính gốc thành **8 thành phần chính (PC1 đến PC8)**.
   * *Bảo toàn thông tin:* **8 thành phần này giữ lại tới 91.6% lượng thông tin phương sai (Explained Variance)** của toàn bộ 80 cột ban đầu.
   * *Ý nghĩa vật lý:*
     * **Qubit 0 (PC1 - chiếm 46.6% thông tin):** Đại diện cho thời gian luồng (`Flow Duration`) và khoảng cách thời gian giữa các gói tin (`Fwd IAT Total`).
     * **Qubit 1 (PC2 - chiếm 13.2% thông tin):** Đại diện cho sự vắng mặt của cờ xác nhận (`ACK Flag Count`) và hướng luồng (`Inbound`).
     * **Qubit 2 đến 7 (PC3-PC8):** Giám sát kích thước gói tin (`Packet Length Min/Max`) và độ trễ phản hồi máy chủ (`Bwd IAT`).

---

# PHẦN III: NỀN TẢNG ĐIỆN TOÁN LƯỢNG TỬ & KỶ NGUYÊN NISQ

### 1. Đơn vị thông tin Qubit & Mặt cầu Bloch
* Trong máy tính cổ điển, một bit chỉ có thể nhận một trong hai trạng thái rời rạc: `0` hoặc `1`.
* Một **Qubit (Quantum Bit)** là một hệ lượng tử hai mức, có thể tồn tại ở trạng thái chồng chập tuyến tính giữa hai trạng thái cơ sở `|0>` và `|1>`:
  ```text
  |psi> = alpha * |0> + beta * |1>
  ```
  Trong đó `alpha` và `beta` là các biên độ xác suất phức thỏa mãn điều kiện chuẩn hóa:
  ```text
  |alpha|^2 + |beta|^2 = 1
  ```
* **Mặt cầu Bloch (Bloch Sphere):** Biểu diễn hình học không gian 3 chiều của trạng thái đơn qubit:
  * Cực Bắc của mặt cầu tương ứng với trạng thái `|0>`.
  * Cực Nam của mặt cầu tương ứng với trạng thái `|1>`.
  * Khoảng cách góc quay từ cực Bắc xuống cực Nam **đúng bằng góc pi radian (180 độ)**.

### 2. Các nguyên lý lượng tử trụ cột
* **Chồng chập lượng tử (Superposition):** Một thanh ghi gồm `n` qubit có thể biểu diễn đồng thời `2^n` trạng thái cơ sở cùng một lúc. Ví dụ: 8 qubit chứa đồng thời 256 trạng thái, tạo tiềm năng xử lý dữ liệu song song cực lớn.
* **Vướng víu lượng tử (Quantum Entanglement):** Hiện tượng hai hoặc nhiều qubit liên kết chặt chẽ với nhau mà trạng thái của hệ không thể phân tích thành tích tensor của từng qubit riêng lẻ. Được tạo ra bởi các cổng 2-qubit (như cổng CNOT). Khi đo lường một qubit, trạng thái của qubit vướng víu với nó sẽ lập tức sụp đổ tương ứng.

### 3. Kỷ nguyên NISQ (Noisy Intermediate-Scale Quantum)
Thuật ngữ do Giáo sư John Preskill (Viện Công nghệ California - Caltech) đặt ra năm 2018 để mô tả giai đoạn phát triển phần cứng lượng tử hiện tại:
* **Quy mô trung bình (Intermediate-scale):** Số lượng qubit vật lý chỉ dao động từ vài chục đến vài trăm qubit (chưa đủ hàng triệu qubit để chạy các thuật toán lượng tử hoàn hảo như Shor phá mã RSA).
* **Nhiễu phần cứng (Noisy):** Các cổng lượng tử có tỷ lệ sai số vật lý cao, môi trường dễ gây mất kết hợp lượng tử (Decoherence).
* **Chưa có mã sửa lỗi (No Fault-Tolerance):** Chưa thể triển khai mã sửa lỗi lượng tử bề mặt (Surface Codes).
* **Ràng buộc thiết kế mạch:** Các mạch lượng tử trong giai đoạn NISQ bắt buộc phải có **độ sâu nông (Shallow Circuit Depth)**. Nếu mạch quá sâu với nhiều tầng cổng CNOT, trạng thái lượng tử sẽ bị phân rã thành nhiễu trắng vô nghĩa trước khi kịp đo lường.

---

# PHẦN IV: CÁC KỸ THUẬT NHÚNG DỮ LIỆU LƯỢNG TỬ (QUANTUM FEATURE MAPS)

Kỹ thuật nhúng dữ liệu (Quantum Feature Map / Embedding) là chiếc cầu nối biến vector đặc trưng cổ điển số thực `x = [x0, x1, ..., x7]` thành trạng thái lượng tử `|psi(x)>` trong không gian Hilbert. Đề tài khảo sát 3 kỹ thuật cốt lõi:

### 1. Kỹ thuật Angle Embedding (Nhúng Góc quay)
* **Cơ chế:** Sử dụng trực tiếp `n` đặc trưng cổ điển làm các góc quay độc lập cho `n` qubit thông qua các cổng quay một qubit:
  ```text
  |psi(x)> = U_angle(x) |00...0> = (Rx(x0) (x) Rx(x1) (x) ... (x) Rx(x7)) |00...0>
  ```
* **Số lượng Qubit cần dùng:** Cần đúng **n Qubit** cho `n` đặc trưng (8 đặc trưng cần đúng 8 Qubit).
* **Đặc tính kỹ thuật:**
  * Mạch hoàn toàn độc lập, không sử dụng cổng vướng víu 2-qubit (CNOT = 0).
  * Độ sâu mạch nông nhất (Circuit Depth = 1).
  * Kháng nhiễu tốt nhất trong điều kiện phần cứng NISQ.
* **Kết quả thực nghiệm:** Là kỹ thuật nhúng **tốt nhất trong toàn bộ nhánh lượng tử**, giúp QSVM đạt `Accuracy = 99.0%` và `F1-Score = 0.9474`.

### 2. Kỹ thuật Amplitude Embedding (Nhúng Biên độ)
* **Cơ chế:** Mã hóa các giá trị của vector `x` trực tiếp vào các hệ số biên độ xác suất của hàm sóng lượng tử:
  ```text
  |psi(x)> = sum_{i=0}^{2^n - 1} x_i |i>
  ```
  Với điều kiện vector đã được chuẩn hóa đơn vị: `sum |x_i|^2 = 1`.
* **Số lượng Qubit cần dùng:** Cực kỳ tiết kiệm: Chỉ cần **log2(n) Qubit**. Với 8 đặc trưng, chỉ cần đúng **3 Qubit** vì `2^3 = 8`.
* **Đặc tính kỹ thuật:**
  * Giúp tối ưu hóa số lượng qubit vật lý của máy tính lượng tử.
  * *Nhược điểm nghiêm trọng:* Mạch khởi tạo trạng thái (State Preparation) yêu cầu phân rã thành chuỗi cổng lượng tử phức tạp. Khi kết hợp với mạng nơ-ron biến phân, nó gây ra hiện tượng **Barren Plateaus** làm triệt tiêu gradient.

### 3. Kỹ thuật ZZ Feature Map (Nhúng Vướng víu Phi tuyến)
* **Cơ chế:** Đề xuất bởi IBM Quantum (Havlíček et al., Nature 2019). Sau khi quay cổng đơn qubit, áp dụng thêm các cổng vướng víu CNOT xen kẽ cổng quay 2-qubit `Rzz`:
  ```text
  Góc quay vướng víu = 2 * (pi - x_i) * (pi - x_j)
  ```
* **Số lượng Qubit cần dùng:** Cần **n Qubit** cho `n` đặc trưng (8 Qubit).
* **Đặc tính kỹ thuật:**
  * Tạo ra các liên kết tương quan bậc hai phi tuyến giữa các cặp đặc trưng gói tin trong không gian Hilbert mà máy tính cổ điển rất khó mô phỏng.
  * *Nhược điểm:* Mạch rất sâu với **28 cổng CNOT** liên kết chéo. Thời gian tính toán ma trận bị bùng nổ lên tới **hơn 5 giờ** và tích tụ nhiều nhiễu cổng vật lý làm suy giảm độ chính xác của mạng nơ-ron xuống còn 83%.

---

# PHẦN V: CÁC MÔ HÌNH HỌC MÁY LƯỢNG TỬ (QSVM & QNN/VQC)

---

### 1. Thuật toán QSVM (Quantum Support Vector Machine)
* **Nguyên lý Hàm nhân Lượng tử (Quantum Kernel Trick):**
  Trong máy học cổ điển, thuật toán SVM ánh xạ dữ liệu sang không gian nhiều chiều nhờ hàm nhân (như RBF). Trong QSVM, không gian nhiều chiều chính là không gian Hilbert đa chiều của hệ lượng tử ($2^8 = 256$ chiều).
* **Đo lường độ tương đồng:** Mạch lượng tử tính trực tiếp độ chồng lấn (Quantum Overlap) giữa trạng thái của hai gói tin mạng `x_i` và `x_j`:
  ```text
  K(x_i, x_j) = |<psi(x_i) | psi(x_j)>|^2
  ```
* **Quy trình phân loại:**
  1. Mạch lượng tử tính toán toàn bộ ma trận Kernel Gram kích thước `N x N`.
  2. Nạp ma trận này vào bộ giải quy hoạch toàn phương cổ điển (Quadratic Programming Solver) để tìm siêu phẳng phân cách tối ưu có lề an toàn rộng nhất.
* **Ưu điểm vượt trội:** Bài toán tối ưu của SVM là **bài toán tối ưu lồi (Convex Optimization)** toàn cục, luôn đảm bảo tìm được nghiệm tối ưu duy nhất mà không bị rơi vào điểm cực tiểu cục bộ.

---

### 2. Thuật toán VQC / QNN (Variational Quantum Classifier / Quantum Neural Network)
* **Kiến trúc Lai Lượng tử - Cổ điển (Quantum-Classical Hybrid Loop):**
  Mô hình hoạt động theo vòng lặp khép kín:
  1. **Tầng nhúng (Feature Map):** Khối `AngleEmbedding` nạp vector gói tin vào 8 Qubit.
  2. **Tầng biến phân (Ansatz):** Chuỗi các cổng xoay có tham số `RY(theta)`, `RZ(theta)` xen kẽ các cổng vướng víu CNOT. Có 2 dạng kiến trúc:
     * *Two-Local:* Tầng cổng quay xen kẽ tầng CNOT mắc nối tiếp vòng tròn (`Circular Entanglement`).
     * *Tree-Tensor:* Cấu trúc mạng dạng cây nhị phân, giảm dần số qubit qua từng tầng.
  3. **Đo lường (Measurement):** Đo giá trị kỳ vọng toán tử Pauli-Z trên qubit đầu tiên: `E[Z]`. Giá trị đo được nằm trong đoạn `[-1.0, +1.0]`.
  4. **Tối ưu hóa cổ điển (Classical Optimizer):** Máy tính cổ điển tính hàm mất mát (Mean Squared Error / Cross Entropy) và cập nhật tham số góc quay `theta` bằng thuật toán Gradient Descent.

---

### 3. Quy tắc Dịch chuyển Tham số (Parameter-Shift Rule)
Khác với mạng nơ-ron cổ điển có thể nhìn vào trọng số để tính đạo hàm bằng Lan truyền ngược (Backpropagation), mạch lượng tử là một hộp đen vật lý (đo vào sẽ làm sập hàm sóng). Để tính đạo hàm giải tích chính xác của hàm mất mát theo tham số `theta_i`, máy tính lượng tử thực thi quy tắc:
```text
d<Z>/d(theta_i) = [<Z>(theta_i + pi/2) - <Z>(theta_i - pi/2)] / 2
```
* **Ý nghĩa:** Muốn cập nhật 1 tham số góc quay, CPU phải chạy mạch lượng tử **2 lần** (một lần cộng `pi/2`, một lần trừ `pi/2`). Với mạch 24 tham số, mỗi bước học cần chạy mạch tới 48 lần.

---

### 4. Hiện tượng Barren Plateaus (Triệt Tiêu Gradient Lượng Tử)
* **Bản chất khoa học:** Một hiện tượng bệnh lý phổ biến trong mạng nơ-ron lượng tử biến phân (McClean et al., Nature Communications 2018). Khi số lượng qubit tăng hoặc khi sử dụng mạch nhúng biên độ (Amplitude Embedding) không phù hợp, phương sai của đạo hàm hàm mất mát bị triệt tiêu về 0 theo cấp số nhân:
  ```text
  Var[ d(Cost)/d(theta) ] -> 0
  ```
* **Hậu quả thực tế:** Bề mặt hàm mất mát trở nên hoàn toàn phẳng lì như một sa mạc (Bình nguyên cằn cỗi). Thuật toán Gradient Descent hoàn toàn bị "mù", không xác định được hướng đi để hạ thấp sai số huấn luyện.
* **Minh chứng trong đề tài:** Mô hình `QNN (Two-Local + Amplitude)` bị dính Barren Plateaus, dẫn đến độ chính xác sụp đổ về **43.0%** (tương đương đoán mò ngẫu nhiên).

---

# PHẦN VI: CÁC MÔ HÌNH HỌC MÁY CỔ ĐIỂN ĐỐI CHUẨN (CLASSICAL BASELINES)

---

### 1. Classical SVM (Support Vector Machine)
* Sử dụng hàm nhân RBF (Radial Basis Function / Gaussian Kernel):
  ```text
  K_RBF(x, z) = exp( -gamma * ||x - z||^2 )
  ```
* **Đặc tính:** Tìm ranh giới phân tách phi tuyến tối ưu trên không gian 8 chiều số thực.
* **Hiệu năng:** Chạy cực nhanh (**0.002 giây** cho 100 mẫu, **0.19 giây** cho 100.000 mẫu), độ chính xác đạt tới **98% - 99.47%**.

---

### 2. Classical ANN (Multi-Layer Perceptron)
* **Kiến trúc mạng sâu 5 tầng:** `8 x 8 x 16 x 8 x 1`
  * Tầng vào: 8 nơ-ron nhận 8 thành phần PCA.
  * Tầng ẩn 1: 8 nơ-ron trích xuất sơ bộ.
  * Tầng ẩn 2: 16 nơ-ron mở rộng không gian để kết hợp chéo các đặc trưng.
  * Tầng ẩn 3: 8 nơ-ron cô đọng thông tin tổng hợp.
  * Tầng ra: 1 nơ-ron sử dụng hàm kích hoạt Sigmoid xuất xác suất nhãn DDoS.
* **Hàm kích hoạt tầng ẩn:** `ReLU(x) = max(0, x)` giúp tính đạo hàm nhanh chóng và tránh hiện tượng bão hòa gradient.
* **Thuật toán tối ưu:** `Adam` (Adaptive Moment Estimation) tự động điều chỉnh tốc độ học cho từng trọng số.

---

# PHẦN VII: TỔNG HỢP CÁC BÀI BÁO KHOA HỌC LIÊN QUAN (LITERATURE REVIEW)

Để đáp ứng yêu cầu *"Tìm kiếm các bài báo liên quan cùng chủ đề"* của Thầy Khôi, dưới đây là 5 công trình khoa học quốc tế uy tín làm nền tảng lý luận cho đề tài:

1. **Bài báo gốc của đề tài:**
   * **Tên bài báo:** *Benchmarking quantum machine learning for detecting DDoS cyber-attacks: a NISQ-era evaluation*
   * **Tác giả:** Ricardo S. Alonso, Guillermo Rivas, Rodrigo Gil-Merino, Diego Valdeolmillos, Javier Prieto (2023).
   * **Đơn vị:** Đại học Quốc tế La Rioja (UNIR) và Đại học Salamanca, Tây Ban Nha.
   * **Đóng góp:** Công trình đầu tiên thực hiện đo kiểm đối chuẩn có kiểm soát giữa Classical ML và QML trên tập dữ liệu `CICDDoS2019`.

2. **Bài báo nền tảng về Không gian Đặc trưng Lượng tử (IBM Quantum):**
   * **Tên bài báo:** *Supervised learning with quantum-enhanced feature spaces*
   * **Tác giả:** Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta (2019).
   * **Tạp chí:** **Nature**, Vol. 567, pp. 209–212.
   * **Đóng góp:** Phát minh ra cấu trúc mạch **ZZ Feature Map** và chứng minh về mặt toán học rằng máy tính lượng tử có thể phân loại dữ liệu trong không gian Hilbert đa chiều mà máy tính cổ điển không thể mô phỏng hiệu quả.

3. **Bài báo nền tảng về Hiện tượng Barren Plateaus (Google Quantum AI):**
   * **Tên bài báo:** *Barren plateaus in quantum neural network training landscapes*
   * **Tác giả:** Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven (2018).
   * **Tạp chí:** **Nature Communications**, Vol. 9, Article 4812.
   * **Đóng góp:** Chứng minh định lý toán học rằng các mạng nơ-ron lượng tử ngẫu nhiên sẽ bị triệt tiêu đạo hàm theo hàm mũ khi số qubit tăng hoặc khi mạch quá sâu.

4. **Bài báo về Bộ dữ liệu Chuẩn CICDDoS2019:**
   * **Tên bài báo:** *Developing Realistic Distributed Denial of Service (DDoS) Dataset for Machine Learning-based Intrusion Detection System*
   * **Tác giả:** H. J. Hadi, U. Hayat, N. Musthaq, F. B. Hussain, Y. Cao (2022).
   * **Hội thảo:** *2022 9th International Conference on Internet of Things: Systems, Management and Security (IOTSMS)*, pp. 1-6. IEEE.
   * **Đóng góp:** Thu thập và chuẩn hóa bộ dữ liệu CICDDoS2019 phản ánh trung thực lưu lượng tấn công mạng hiện đại.

5. **Bài báo về Học máy Lượng tử trong Không gian Hilbert (Xanadu / PennyLane):**
   * **Tên bài báo:** *Quantum Machine Learning in Feature Hilbert Spaces*
   * **Tác giả:** Maria Schuld, Nathan Killoran (2019).
   * **Tạp chí:** **Physical Review Letters**, Vol. 122, 040504.
   * **Đóng góp:** Đặt nền móng lý thuyết thống nhất cho các thuật toán Quantum Kernel và phương pháp học có giám sát lượng tử trên thư viện PennyLane.

---

# PHẦN VIII: HƯỚNG DẪN TRIỂN KHAI SOURCE CODE & QUY TRÌNH THỰC NGHIỆM

Kho mã nguồn chính thức được tổ chức tại thư mục:  
`C:\Study\HocKy6\MH&MM\qml-ddos`

### 1. Cấu trúc thư mục dự án
* `dataset/`: Chứa dữ liệu gốc `CICDDoS2019` (~2.5 GB) và các tập đã tiền xử lý (`tr_day_syn_small.csv`, `test_day_syn_small.csv`).
* `dev/`: 29 file Jupyter Notebooks thử nghiệm chi tiết từng mô hình độc lập của tác giả.
* `results/`: Toàn bộ các bảng CSV tổng hợp và 8 biểu đồ kết quả độ phân giải cao phục vụ báo cáo.
* `download_syn_dataset.py`: Script tự động tải bộ dữ liệu `Syn.csv`.
* `run_dataset_preprocessing.py`: Script tiền xử lý dữ liệu và xuất các tập dữ liệu phân tầng.
* `run_benchmark_all.py`: **Pipeline thực thi chính**, chạy tự động đo kiểm cả 5 mô hình đại diện và xuất toàn bộ biểu đồ.
* `export_paper_and_reproduction_benchmark.py`: Tái hiện đầy đủ số liệu 10 mô hình của bài báo quốc tế.

### 2. Lệnh thực thi nhanh bằng PowerShell
```powershell
# Bước 1: Di chuyển vào thư mục dự án
cd "C:\Study\HocKy6\MH&MM\qml-ddos"

# Bước 2: Chạy pipeline đối chuẩn chuẩn khoa học
python run_benchmark_all.py

# Bước 3: Xuất bảng so sánh toàn diện bài báo
python export_paper_and_reproduction_benchmark.py
```

---

# PHẦN IX: BỘ CÂU HỎI VẤN ĐÁP BẢO VỆ ĐỒ ÁN VỚI THẦY KHÔI (Q&A)

### Câu 1: Tại sao nhóm lại chọn 8 Qubit mà không chọn 6, 7 hay 9 Qubit?
* **Trả lời:**
  * 8 Qubit tương ứng với 8 thành phần chính của thuật toán PCA, giúp giữ lại tới **91.6% lượng thông tin phương sai** của toàn bộ 80 đặc trưng mạng gốc.
  * Con số 8 là lũy thừa hoàn hảo của 2 (`2^3 = 8`), cho phép nén vừa khít vào **3 Qubit** trong mạch `Amplitude Embedding` mà không bị dư thừa trạng thái rác.
  * 8 Qubit tạo ra không gian Hilbert $2^8 = 256$ trạng thái, vừa đủ nhẹ để máy tính mô phỏng chạy mượt mà trong vài chục giây mà không bị tràn bộ nhớ RAM.

### Câu 2: Giữa Cổ điển và Lượng tử, mô hình nào tốt hơn trong thực tế?
* **Trả lời:**
  * Trong điều kiện phần cứng hiện tại (kỷ nguyên NISQ), **mô hình Cổ điển (Classical SVM) vẫn hoàn toàn vượt trội**.
  * Cổ điển đạt độ chính xác **99.47%** trên 100.000 mẫu chỉ trong **0.19 giây**.
  * Trong khi đó, mô hình lượng tử QSVM mất từ vài chục phút đến hơn 5 giờ cho 1.000 mẫu. Do đó, QML chưa thể áp dụng cho các hệ thống giám sát mạng thời gian thực (Real-time IDS) ở thời điểm hiện tại.

### Câu 3: Khâu nào trong mạch lượng tử quyết định độ chính xác của mô hình?
* **Trả lời:**
  * Thực nghiệm đối chuẩn chứng minh: **Kỹ thuật nhúng dữ liệu (Feature Map) quyết định tới 90% hiệu năng của mô hình**, chứ không phải hình dạng mạch biến phân (Ansatz).
  * Minh chứng: Cả hai kiến trúc `Two-Local` và `Tree-Tensor` khi dùng chung kỹ thuật `Angle Embedding` đều đạt độ chính xác y hệt nhau (97%, F1 = 0.90). Nhưng khi chuyển sang `ZZ Feature Map`, cả hai đều bị tụt dốc xuống 92% (F1 = 0.83).

### Câu 4: Tại sao mô hình QNN Amplitude lại thất bại nặng nề (chỉ đạt 43% độ chính xác)?
* **Trả lời:**
  * Do mạch bị dính hiện tượng vật lý lượng tử kinh điển gọi là **Barren Plateaus (Triệt tiêu Gradient Lượng tử)**.
  * Khi nén 8 đặc trưng vào biên độ của 3 Qubit, đạo hàm của hàm mất mát bị triệt tiêu về 0 theo cấp số nhân, khiến thuật toán Gradient Descent không thể tìm thấy hướng đi để cập nhật trọng số. Mạng bị "mù" và chỉ còn khả năng đoán mò ngẫu nhiên.

### Câu 5: Tại sao trên tập mẫu nhỏ cục bộ (100 mẫu), QSVM Angle lại đạt AUC cao hơn Classical SVM (0.996 so với 0.989)?
* **Trả lời:**
  * Đây là ưu thế học trên dữ liệu ít mẫu (**Few-shot Learning**) của lượng tử. Khi dữ liệu ít, hàm nhân lượng tử ánh xạ các điểm dữ liệu vào không gian Hilbert 256 chiều, giúp các điểm dữ liệu thưa thớt dễ dàng được phân tách với lề an toàn (Margin) rộng và thoáng hơn so với hàm nhân RBF cổ điển trong không gian 8 chiều.

---
*Tài liệu được biên soạn phục vụ học tập, nghiên cứu và báo cáo cuối kỳ môn Mã hóa và Mật mã học — Đại học Bách Khoa, ĐH Đà Nẵng.*
