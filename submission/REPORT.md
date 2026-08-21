# Lab 21 — Evaluation Report

**Họ tên**: Lương Quốc Khánh  **MSSV**: 2A202601713  **Ngày**: 2026-08-21  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 (Colab)`

> Mọi con số dưới đây được lấy từ `results/`; report này được sinh sau NB5 để tránh chép nhầm số.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | `1024` — p95 đo được là `98`, suggested `256` |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2` / `30` |

P95 chỉ là 98 token và giá trị gợi ý theo dữ liệu là 256. Tôi vẫn giữ `max_length=1024` vì đây là cấu hình cố định của tier T4 trong lab và cần giữ cùng một cấu hình cho cả bốn run để phép đối chứng không đổi thêm một biến. Với `max=101`, lựa chọn này không gây truncation; đổi lại nó bảo thủ hơn mức cần thiết về padding/VRAM, và đây là trade-off tôi ghi nhận thay vì coi 1024 là con số được suy ra trực tiếp từ p95.

**Template có giữ khối `<think>` không?** **có** — `results/template_check.json`.  
Template giữ được nội dung reasoning giả trong phép thử, nên không có bằng chứng rằng `<think>` bị xoá ở bước render. Corpus core chỉ chứa JSON trả lời nên tôi vẫn dùng `assistant-only` và không dựa vào reasoning trace.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

3–5 dòng đầu của đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn supervise chứa câu trả lời assistant và không chứa fragment của ticket, vì vậy loss mask phù hợp với mục tiêu SFT: model bị phạt khi sinh sai JSON, không bị dạy viết lại prompt.

---

## 3. Ba baseline (NB2 đo trước khi train; NB5 chấm fine-tune)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3240.1 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1074.3 |
| (c) LoRA fine-tune | 0.9700 | 0.4556 | 1.0000 | 1431.0 |

**(b) có thật sự mạnh hơn (a) không?** **có**. Baseline (b) đã được đóng băng trước NB3 (`optimized_prompt_sha=719e74d3b6232053`), và tôi không làm yếu prompt sau khi nhìn thấy kết quả train.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target (NB5 §4)** | steps | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6255 | 0.9700 | 30 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5385 | 0.9750 | 30 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | 0.0000 | 30 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.9400 | 30 | 7.09 |

**4.1 — Vị trí vs rank.** `attn_only` có 32,456,704 trainable params so với 32,464,896 của `correct`, lệch chỉ khoảng 0.025%, nên đây là đối chứng về vị trí chứ không phải ngân sách. Trên target, `attn_only` đạt 0.9750 còn `correct` đạt 0.9700: về số học `attn_only` nhỉnh hơn 0.005, tương đương đúng thêm khoảng **1 field trên 200 field được chấm** (50 mẫu × 4 trường). Train loss cũng xếp `attn_only` trước (0.5385 < 0.6255). Vì biên target rất nhỏ, tôi không coi đây là bằng chứng đủ mạnh rằng attention-only tốt hơn một cách hệ thống; kết quả hợp lý hơn là **matched-rank attention-only không thua trong bài toán hẹp này**. Do đó rank lớn tự nó không chứng minh ưu thế, và placement phải được phán bằng target trên frozen eval chứ không bằng train loss hay số tham số.

**4.2 — Learning rate.** `wrong_lr` chỉ đổi LR từ 1e-4 xuống 1e-5 nhưng final loss là 1.5704 so với 0.6255 của `correct`; target tương ứng là 0.0000 so với 0.9700. Nếu chỉ nhìn một đường loss mà không biết LR, tôi có thể kết luận sai rằng LoRA/placement không học được hoặc cần tăng rank. Đối chứng này tách nguyên nhân: learning-rate scale là biến đủ lớn để làm thay đổi động lực học dù mọi phần còn lại giữ nguyên và cùng step budget.

**4.3 — QLoRA.** QLoRA dùng 7.09 GB so với 12.01 GB của 16-bit, tiết kiệm khoảng **4.92 GB (41.0%)**. Trên target, QLoRA **thua** `correct` (0.9400 so với 0.9700), còn train loss là 0.7058. Vì vậy dữ liệu của run này ủng hộ việc thận trọng với QLoRA: VRAM giảm nhưng chất lượng không vượt cấu hình 16-bit, nhưng kết luận chỉ áp dụng cho đúng model/corpus/budget đã đo.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**  
`target Δ = +0.205` · `regression Δ = -0.302` · `valid_trace_rate = 0.000`

Cổng hồi quy **FAILED**, nhưng nguyên nhân không phải fine-tune không học task. Ngược lại, target tăng từ 0.765 lên 0.970 (`Δtarget=+0.205`) và format vẫn đạt 1.000. Điểm làm run bị loại là **regression giảm từ 0.7578 xuống 0.4556 (`Δ=-0.302`)**, vượt rất xa tolerance 0.020 của gate. Vì regression set chỉ gồm 15 câu, tôi diễn giải đây là một **regression nghiêm trọng trên proxy general-capability của lab**, không khẳng định rộng hơn rằng model đã “quên mọi kiến thức”. Tôi giữ nguyên FAILED thay vì nới ngưỡng: theo contract đã đóng băng, một adapter giỏi hơn trên ticket nhưng làm giảm mạnh năng lực ngoài miền vẫn chưa đủ an toàn để deploy. Baseline (b) không mạnh hơn fine-tune trên target, nhưng nó giữ regression tốt hơn và có latency thấp hơn (1074 ms so với 1431 ms), nên tổng thể nó là lựa chọn an toàn hơn theo chính gate của bài. Hướng sửa hợp lý tiếp theo là thêm 1–5% replay dữ liệu phổ thông rồi train lại trên cùng frozen eval, không sửa prompt (b) hay eval để “cứu” verdict.

---

## 6. Định tính — có cả ca thắng và ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm … | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment":` | ❌ FT thua nhãn (0.75/1.00); còn ít nhất một lỗi field |
| 2 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho … | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sen` | ❌ FT thua nhãn (0.75/1.00); còn ít nhất một lỗi field |
| 3 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | `{"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment"` | ❌ FT thua nhãn (0.75/1.00); còn ít nhất một lỗi field |
| 4 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé. Shop xem giúp. | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sent` | ✅ FT đúng đủ 4 trường |
| 5 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. Mong shop phản hồi.… | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | `{"intent": "doi_tra", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sen` | ✅ FT đúng đủ 4 trường |

Các chuỗi dự đoán trong bảng là **preview đã bị cắt ngắn để trình bày**; `ft_score` được tính từ output đầy đủ trước khi lưu preview, nên dấu cắt giữa JSON không đồng nghĩa với lỗi parse. Các ca tệ nhất được chọn từ đầu `results/qualitative.json`, không cherry-pick ca thắng. Mẫu lỗi được đọc theo field score: khi `ft_score < 1`, ít nhất một trong `intent`, `urgency`, `product`, `sentiment` sai hoặc output không parse đúng. Tôi ưu tiên sửa bằng dữ liệu phản ánh đúng kiểu lỗi field đó; không sửa `eval_target.jsonl` sau khi đã thấy điểm.

---

## 7. Kết luận & điều tôi học được

Tôi không nên deploy ngay adapter hiện tại dựa trên gate đã định trước, chứ không dựa vào việc loss giảm đẹp. Bài học lớn nhất là thứ tự kiểm chứng quan trọng hơn việc quét hyperparameter: NB1 chứng minh loss thực sự chỉ rơi vào lượt assistant (supervised_fraction=0.4149), NB2 đóng băng một prompt baseline đủ mạnh trước khi tôi nhìn thấy kết quả train, và NB5 mới quyết định fine-tune có tạo giá trị hay không. Trong autopsy, `attn_only` thắng `correct` trên target dù ngân sách tham số được matched; điều này cho thấy rank không thể được diễn giải tách khỏi vị trí gắn adapter. `wrong_lr` dùng LR 1e-5 thay vì 1e-4 và có final loss 1.5704 so với 0.6255 của `correct`, nên learning-rate scale là một đòn bẩy có thể làm một cấu hình hợp lý trông như không học. QLoRA tiết kiệm khoảng 4.92 GB (41.0%) VRAM so với 16-bit nhưng trên target nó thua `correct`, vì vậy tiết kiệm bộ nhớ phải được cân cùng chất lượng thay vì mặc định 4-bit là tốt hơn. Quan trọng nhất vẫn là mask và thiết kế phép đo: mask sai làm toàn bộ thí nghiệm vô nghĩa, còn baseline yếu hoặc eval bị chỉnh sau khi train sẽ tạo ra một chiến thắng giả. Nếu cần cải thiện tiếp, tôi sẽ ưu tiên phân tích các lỗi field trong qualitative set, bổ sung dữ liệu đúng loại lỗi và giữ nguyên frozen eval để kiểm tra nhân quả, thay vì chỉ tăng rank hoặc train lâu hơn.

**Ba điều tôi học được:**
1. Một loss curve đẹp không chứng minh task performance; autopsy phải được xếp hạng bằng target, không bằng `final_loss`.
2. `assistant-only` chỉ có ý nghĩa khi tôi nhìn trực tiếp phần token được supervise; một cờ thư viện không thay thế được `mask_proof.json`.
3. Mốc đúng để đánh giá giá trị fine-tune là base + prompt tử tế (b), không phải prompt ngây thơ (a); FAILED vẫn là kết quả hữu ích nếu phép đo được giữ nguyên.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** phân tích lỗi theo từng field trên các ca thua, bổ sung một lượng nhỏ dữ liệu đúng lỗi (và replay 1–5% nếu regression giảm), rồi chạy lại training trên cùng frozen eval để kiểm tra thay đổi có thật sự mang tính nhân quả hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
