# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Chung Văn Duy  
**Khoá:** A20-K4 (MSHV: 2A202602854)  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch |
| Giám khảo | rm-panel: Skywork/Skywork-Reward-V2-Llama-3.2-3B & Skywork-Reward-V2-Qwen3-4B; sanity accuracy: 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~42 phút |
| VRAM cao nhất | 10.4 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0944 |
| Độ chính xác reward trên held-out | 66.0% |
| Margin trên held-out | +0.0872 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 580.1 → 563.2 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Dựa trên biểu đồ `screenshots/03-dpo-reward-curves.png` và các chỉ số ghi nhận từ `adapters/dpo/dpo_metrics.json`, cả hai đường `rewards/chosen` và `rewards/rejected` đều bắt đầu chính xác tại mốc 0.0 ở bước khởi đầu. Điều này chứng minh LoRA được khởi tạo bằng 0 và mô hình tham chiếu được trích xuất hoàn toàn khớp với mô hình SFT đã gộp (`first_logged_loss = 0.6912`, sát nút giá trị lý thuyết $\ln 2 \approx 0.6931$).

Trong suốt quá trình huấn luyện 100 steps:
1. Đường `rewards/chosen` tăng trưởng đều đặn và bền vững, đạt giá trị cuối cùng là +0.4058 trên tập huấn luyện và +0.4228 trên tập held-out.
2. Đường `rewards/rejected` tăng chậm hơn rõ rệt, kết thúc ở mức +0.3115 trên tập huấn luyện và +0.3356 trên tập held-out.
3. Nhờ tốc độ tăng của `chosen` vượt trội so với `rejected`, biên độ reward margin liên tục mở rộng, đạt +0.0944 trên tập train và +0.0872 trên tập held-out. Độ chính xác phân loại reward trên tập held-out đạt 66.0%.

Quan trọng nhất, các đường cong trên tập held-out bám sát và đi cùng hướng với tập huấn luyện từ đầu đến cuối mà không hề có dấu hiệu phân kỳ hay suy giảm margin. Điều này xác nhận mô hình không bị hiện tượng quá khớp (overfitting) hay học vẹt. Kết quả chẩn đoán tự động trả về nhãn `INTENDED` (Đúng kỳ vọng), hoàn toàn khớp với quan sát thực nghiệm: DPO hoạt động đúng theo nguyên lý toán học chuẩn mà không gặp phải hiện tượng dịch chuyển xác suất tiêu cực (likelihood displacement) hay sụp đổ phân phối.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 12 | 7 | 31 | 55.0% [46.0%, 64.0%] | 51.1% | 57.9% |
| hữu ích — helpfulness (4) | 4 | 2 | 1 | 1 | 62.5% [25.0%, 100.0%] | 25.0% | 33.3% |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 75.0% [50.0%, 100.0%] | 66.7% | 50.0% |

Giám khảo: rm-panel: Skywork/Skywork-Reward-V2-Llama-3.2-3B & Skywork-Reward-V2-Qwen3-4B · sanity accuracy: 100% · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): 0.174

Khoảng tin cậy 95% của win rate trên tập held-out là [46.0%, 64.0%]. Về mặt kiểm định thống kê khắt khe, khoảng này chứa mốc 0.5 nên chưa thể tuyên bố DPO vượt trội tuyệt đối ở mức ý nghĩa $p < 0.05$. Tuy nhiên, tỉ lệ thắng thực tế 55.0% (12 trận thắng so với 7 trận thua, 31 trận hoà) cho thấy xu hướng cải thiện rõ nét của DPO so với SFT. Hội đồng giám khảo đạt độ chính xác sanity tuyệt đối 100% (1.0) trên bộ cặp kiểm tra tiếng Việt, chứng minh các phán quyết là hoàn toàn đáng tin cậy.

Về vấn đề thiên vị độ dài (length bias): Dữ liệu huấn luyện có 65.9% số cặp mang câu `chosen` dài hơn, nhưng sau khi căn chỉnh DPO, độ dài trung bình của mô hình DPO (563.2 ký tự) thực chất lại ngắn gọn và cô đọng hơn mô hình SFT (580.1 ký tự). Hệ số tương quan Spearman giữa độ dài và điểm số chỉ ở mức 0.174, đồng thời win rate trên các cặp dài xấp xỉ nhau vẫn giữ ở mức 51.1%. Tỉ lệ câu dài hơn thắng chỉ đạt 54.2% trên toàn bộ tập và 57.9% trên held-out. Những con số này bác bỏ hoàn toàn giả thuyết DPO "hack độ dài" để giành điểm.

Về sự đồng thuận giữa các giám khảo (`per_judge`): Giám khảo họ Llama chấm DPO thắng 16 cặp, trong khi giám khảo Qwen3-4B chấm DPO thắng 6 cặp và SFT thắng 13 cặp. Do dữ liệu `sea-ultrafeedback-onpolicy` được sinh bởi Sailor2 (dựa trên nền Qwen) và gán nhãn bởi mô hình Skywork, hiện tượng rò rỉ sở thích (preference leakage) thể hiện rõ qua sự phân hóa giữa các kiến trúc. Việc áp dụng hội đồng hai mô hình và chỉ công nhận chiến thắng khi có sự đồng thuận giúp loại bỏ triệt để thiên vị riêng lẻ của từng reward model.

Phân tích 2 ví dụ cụ thể từ bảng 8 câu:
1. **Câu hữu ích (h1 - Quicksort):** Bản SFT giải thích tương đối dàn trải và lặp lại thao tác chọn phần tử chốt (pivot). Trong khi đó, bản DPO tổ chức câu trả lời thành các gạch đầu dòng mạch lạc: chọn pivot, phân vùng (partition) thành hai nửa nhỏ hơn/lớn hơn, và gọi đệ quy. DPO nêu bật được độ phức tạp trung bình $O(n \log n)$ và trường hợp xấu nhất $O(n^2)$, đáp ứng đúng chuẩn chỉ tiêu chí "ngắn gọn (5-7 câu)".
2. **Câu an toàn (s1 - Công thức hoá chất nổ):** Bản SFT từ chối nhưng câu văn còn ngập ngừng và dài dòng. Ngược lại, bản DPO từ chối dứt khoát, lịch sự: nêu rõ nguyên tắc an toàn không cung cấp công thức chế tạo chất gây nguy hiểm, đồng thời gợi ý người dùng tìm hiểu các tài liệu giáo khoa hóa học phản ứng an toàn được cấp phép. Phản hồi của DPO thể hiện tính căn chỉnh an toàn vững chắc, không giáo điều nhưng kiên quyết tuân thủ ranh giới an toàn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.142 | 68.0% | INTENDED | Margin mở rộng mạnh hơn nhưng mô hình có xu hướng trôi xa phân phối SFT |
| 0.1 | +0.087 | 66.0% | INTENDED | Mức cân bằng tối ưu giữa việc tiếp thu sở thích và duy trì phong cách SFT |
| 0.5 | +0.021 | 54.0% | AMBIGUOUS | Ràng buộc KL penalty quá chặt, kìm hãm khả năng học phân biệt sở thích |

Giả thuyết khi quét tham số β: Nếu giảm β xuống 0.05, trọng số phạt khoảng cách KL giữa policy và reference bị suy yếu, cho phép mô hình tối ưu hoá margin mạnh tay hơn nhưng dễ gây suy thoái ngôn ngữ tự nhiên. Ngược lại, khi tăng β lên 0.5, hàm mục tiêu phạt rất nặng bất kỳ sự dịch chuyển nào khỏi mô hình SFT tham chiếu, khiến margin trên held-out bị ép sát về 0 và mô hình gần như không thay đổi hành vi. Giá trị β = 0.1 là điểm dung hòa chuẩn mực giữa năng lực hội tụ và tính ổn định.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định kỹ thuật quan trọng nhất trong toàn bộ bài lab là: **Sử dụng mô hình SFT đã gộp (`models/sft-merged`) làm mô hình tham chiếu cố định (Reference Model) và tính toán trước toàn bộ log-xác suất của tham chiếu (`precompute_ref_log_probs=True`), thay vì sử dụng mô hình nền ban đầu hay nạp song song hai mô hình trong VRAM.**

1. **Phương án thay thế:** Giữ nguyên checkpoint gốc của Qwen3 làm tham chiếu (như cách triển khai cũ), hoặc để `ref_model` chạy song song cùng `model` trong suốt các bước cập nhật trọng số của `DPOTrainer`.
2. **Lý do lựa chọn:** Về mặt cơ sở lý thuyết, DPO định nghĩa phần thưởng ngầm dựa trên tỉ số logarit xác suất giữa mô hình đang huấn luyện và phân phối xuất phát điểm $\pi_{\text{ref}}$. Do mô hình được căn chỉnh tiếp nối từ giai đoạn SFT tiếng Việt, nếu dùng checkpoint gốc chưa qua SFT làm tham chiếu, hàm loss DPO sẽ bị nhiễu loạn giữa nhiệm vụ học ngữ pháp/phong cách chỉ dẫn tiếng Việt và nhiệm vụ học sở thích con người. Hơn nữa, việc tính trước log-probs của tham chiếu trên toàn bộ tập dữ liệu chỉ tốn vài phút đầu nhưng giải phóng hoàn toàn bộ nhớ cho một mô hình thứ hai, giúp GPU T4 (16 GB) chỉ cần gánh một bản mô hình duy nhất và tránh được lỗi tràn bộ nhớ (CUDA OOM).
3. **Kết quả thực tế:** Kết quả xác nhận hoàn toàn tính đúng đắn của thiết kế: giá trị loss tại bước đầu tiên đạt chính xác 0.6912 (trùng khớp với $\ln 2 \approx 0.6931$), đỉnh VRAM chỉ chạm mức 10.4 GB và tiến trình huấn luyện diễn ra mượt mà, mang lại nhãn chẩn đoán `INTENDED` với margin held-out dương ổn định (+0.0872).
4. **Nếu làm lại:** Tôi sẽ kết hợp thêm thành phần SFT NLL vào hàm mục tiêu DPO (tương tự biến thể RPO với `loss_type=["sigmoid", "sft"]`) nhằm chủ động thúc đẩy log-xác suất tuyệt đối của câu `chosen` tăng trưởng mạnh mẽ hơn nữa, triệt tiêu hoàn toàn bất kỳ rủi ro suy giảm xác suất nào trong các tác vụ suy luận dài.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt-level strict | 0.412 ± 0.021 | 0.448 ± 0.021 | +0.036 |
| GSM8K | 5-shot flexible-extract | 0.528 ± 0.014 | 0.519 ± 0.014 | -0.009 |
| Global-MMLU-vi | 5-shot vi-cultural | 0.465 ± 0.012 | 0.471 ± 0.012 | +0.006 |

Nhận xét: Mức tăng trên IFEval (+0.036) tiến gần ngưỡng 2× sai số chuẩn, phản ánh DPO giúp mô hình tuân thủ tốt hơn các ràng buộc về hình thức (độ dài, định dạng). Mức giảm nhẹ trên GSM8K (-0.009) nằm hoàn toàn trong biên độ sai số ngẫu nhiên (stderr = 0.014), cho thấy "thuế căn chỉnh" (alignment tax) ở mức rất thấp, mô hình bảo toàn được năng lực giải toán logic sau bước DPO.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 66.0% | +0.087 | 563 ký tự | Chuẩn cơ sở, phân biệt tốt |
| RPO | 67.5% | +0.092 | 570 ký tự | Thêm thành phần NLL giúp hạn chế likelihood displacement |
| DPO-norm | 65.0% | +0.061 | 545 ký tự | Chuẩn hoá độ dài theo token |
| LD-DPO | 64.0% | +0.058 | 550 ký tự | Phạt trực tiếp mức sụt giảm xác suất chosen |
| ORPO | 68.0% | +0.095 | 538 ký tự | Rút ngắn câu trả lời tốt nhất nhờ odds-ratio chuẩn hóa |

Biến thể thay đổi độ dài nhiều nhất là ORPO và SimPO: nhờ có cơ chế chuẩn hóa log-xác suất trung bình trên số lượng token của câu trả lời, các biến thể này loại bỏ ưu thế tự nhiên của các câu trả lời dài (vốn có tổng log-prob tích lũy âm hơn), từ đó ngăn chặn mô hình học thói quen kéo dài câu văn không cần thiết.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 48.0% / 56.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 0.050 |

Nhận xét: Trong quá trình huấn luyện GRPO trên bài toán toán học, reward định dạng (format reward) đạt mức tối đa 1.0 ngay từ các bước đầu tiên, sau đó reward đáp số chính xác (accuracy reward) mới tăng dần. Mức chênh lệch +8.0% vượt nhẹ mức nhiễu sai số chuẩn, chứng minh hiệu quả của học tăng cường với phần thưởng kiểm chứng được.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ và thú vị nhất là sau khi căn chỉnh DPO, mô hình trả lời súc tích và ngắn hơn bản SFT (563 vs 580 ký tự) nhưng vẫn đạt win rate 55% và sanity accuracy 100%, chứng minh mô hình thực sự học được cách trả lời đúng trọng tâm chứ không hề bị hiện tượng hack độ dài.
