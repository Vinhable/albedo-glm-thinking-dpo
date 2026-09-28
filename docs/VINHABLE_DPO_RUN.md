# Train thử DPO GLM-có-thinking (vinhable_single / vinhable_double) trên 8×H200

Mục tiêu: kiểm tra **tín hiệu học** của hai bộ `vinhable_*` trên King 127 trước khi đầu tư sinh
data lớn. Không nhằm tạo model để nộp.

## Bài học từ các lần chạy của lab và cách code này xử lý

| Lab gặp | Ở đây |
|---|---|
| Lỗi ở một GPU nhưng không có traceback (`error_file: <N/A>`) | `main` bọc `@record`, `TORCHELASTIC_ERROR_FILE`, log riêng từng GPU (`rank<N>.log`) |
| OOM ở bước optimizer đầu (Adam moments khi train full-weight) | Chỉ train attention, linear attention, shared expert, norm (~1,4B); expert, router, embedding, lm_head, vision, MTP đóng băng |
| Full-weight LR 2e-6 trôi như nhiễu; export bf16 làm tròn mất độ trôi nhỏ | Đo độ trôi trên fp32 master (`safe_get_full_fp32_param`) so với đường nhiễu √Σlr²; LR mặc định 5e-6 |
| Log-prob reference ghép nhầm vì khoá trùng | Khoá theo `sample_uid`, kiểm tra trùng khi đọc data và khi gộp reference |
| Dev lệch −0,03 giữa lượt reference và eval | Reference và policy dùng chung một hàm; cổng bước 0 (lệch mỗi token < 1e-3) chặn trước khi update |
| Che `</think>` khỏi loss nên model không bao giờ đóng think | Lượt được tính loss luôn có `</think>` và `<|im_end|>`; lượt không đóng think thì không tính loss |
| Render cả quỹ đạo nên model thấy reasoning cũ | Mỗi lượt được tính loss là một chuỗi riêng, lịch sử render qua chat template production (tự bỏ reasoning cũ) |
| Grad norm ≈ 100 do cộng log-prob hàng nghìn token | Chuẩn hoá theo độ dài (`--agg mean`, N0 = 512) + NLL 0,1 trên chosen |
| Image Shadeform không có `nvcc` | Optimizer là `torch.optim.AdamW` (không op DeepSpeed nào cần build); vLLM tắt sampler FlashInfer |
| Một GPU bỏ một lượt forward làm ZeRO-3 treo | Mỗi GPU chạy cùng số forward/backward mỗi bước (chuỗi giả bù cho GPU ít việc) |
| Máy bị xoá khi idle, VM khởi động lâu | Mỗi stage chạy `setsid nohup`; kết quả nhỏ kéo về máy ngay |

Phát hiện thêm khi chuẩn bị: ZeRO-3 chỉ cộng dồn gradient giữa các lần backward nếu forward đi qua
`engine.forward` (nơi nó tăng micro-step và gắn hook backward). Gọi thẳng module con thì mỗi lần
backward **ghi đè** gradient trước. Trainer bọc model trong một module trả về tổng log-prob và luôn
gọi qua engine.

## Các file

| File | Vai trò |
|---|---|
| `scripts/vinhable_dpo_data.py` | Render mỗi lượt được tính loss thành chuỗi riêng, đúng như production |
| `scripts/prepare_vinhable_dpo.py` | Kiểm hợp đồng data, ước tính token và thời gian (CPU) |
| `scripts/train_vinhable_dpo.py` | Trainer (pha `reference`, `train`, `export`), DeepSpeed ZeRO-3 |
| `scripts/reassemble_trained_checkpoint.py` | Ghép export gọn vào checkpoint King → checkpoint đầy đủ, kiểm tên/shape/metadata |
| `scripts/thinking_check_vllm.py` | Kiểm thinking bằng vLLM, so với King |
| `scripts/run_vinhable_dpo_8xh200.sh` | Các stage trên máy |
| `ops/vinhable-dpo/machine.sh` | Phía máy local: đẩy code, chạy stage, xem log, kéo kết quả |
| `tests/test_vinhable_dpo_data.py`, `tests/test_vinhable_dpo_trainer.py` | Test CPU (render; đạo hàm DPO; gradient hai lượt = autograd trực tiếp) |

## Chạy

Tất cả lệnh chạy từ Git Bash ở gốc repo. `HOST`, `PORT`, `SSH_USER` là thông tin máy Shadeform.

```bash
export HOST=<ip> PORT=<port> SSH_USER=<user>
bash ops/vinhable-dpo/machine.sh push                 # đẩy code (không có khoá hay token nào)
bash ops/vinhable-dpo/machine.sh stage setup          # ~1 giờ: venv, King 127 (~70 GB), 2 dataset
bash ops/vinhable-dpo/machine.sh tail setup
DATASET=single bash ops/vinhable-dpo/machine.sh stage prep
DATASET=single bash ops/vinhable-dpo/machine.sh stage smoke        # ~20–30 phút, dòng dài nhất
DATASET=single bash ops/vinhable-dpo/machine.sh stage reference    # ~15–20 phút
DATASET=single bash ops/vinhable-dpo/machine.sh stage train        # ~2 giờ (2 epoch)
DATASET=single bash ops/vinhable-dpo/machine.sh stage reassemble
DATASET=single bash ops/vinhable-dpo/machine.sh stage think        # King + checkpoint, vLLM
bash ops/vinhable-dpo/machine.sh pull                 # metrics, log, kết quả về E:
```

Lặp lại với `DATASET=double` nếu còn thời gian (~1,7× lâu hơn).

Các núm chỉnh qua biến môi trường của stage, ví dụ
`bash ops/vinhable-dpo/machine.sh stage train "LR=1e-5 EPOCHS=3"`: `LR`, `EPOCHS`, `ROWS_PER_RANK`,
`BETA`, `AGG`, `N0`, `LS`, `NLL`, `MAX_SEQ`, `EVAL_EVERY`, `DRIFT_EVERY`, `EXPORT_STEPS`.

## Smoke phải đạt trước khi train

1. `step-0 gate`: lệch |policy − reference| mỗi token < 1e-3.
2. Không OOM ở hai bước trên các dòng dài nhất; ghi lại `peak_vram_gib` và `tokens_per_s`.
3. `drift_rms` > 0 sau 2 bước (gradient thật sự tới được trọng số).
4. Export và reassemble thành công (1.045 tensor, metadata giữ nguyên).

Nếu OOM: giảm `MAX_SEQ` (ví dụ 49152). Nếu cổng bước 0 trượt: dừng lại, không train.

## Tiêu chí go/no-go (đọc từ `metrics.jsonl`)

| Chỉ số | Không có tín hiệu (như `gen_vinhable`) | Có tín hiệu |
|---|---|---|
| `drift_over_noise` | ≈ 1,0 | > 1,5 và tăng dần |
| `increment_cos_total` | ≈ 0 | dương rõ |
| `dev_accuracy` | ≈ 0,5 | ≥ 0,60–0,65 |
| `dev_chosen_logratio` | — | không giảm đều (nếu cả hai log-ratio cùng giảm là likelihood collapse) |
| Thinking check | — | đóng think ≈ King, không có think rỗng, reasoning không co lại, không lượt nào bị cắt |

So sánh `single` và `double` trên cùng các chỉ số.

## Thời gian và chi phí (Shadeform ~$32/giờ)

| Việc | single | double |
|---|---|---|
| Reference | ~15–20 phút | ~30 phút |
| Train 2 epoch | ~2 giờ | ~3,5 giờ |
| Thinking check (2 model) | ~30 phút | ~20 phút |

Setup + smoke + single ≈ 4–5 giờ (~$130–160); thêm double ≈ +4 giờ.

## Rủi ro còn lại

- Vòng train tuỳ biến trên DeepSpeed ZeRO-3 mới chỉ chạy thử trên CPU với model nhỏ; smoke là lần
  đầu trên GPU thật.
- `flash-linear-attention` phải cài được (Triton, không cần `nvcc`); thiếu nó thì linear attention
  chạy bản PyTorch, rất chậm. `causal-conv1d` không cài (cần build), dùng bản PyTorch.
- vLLM trên image không có `nvcc`: nếu lỗi, xem lại cấu hình backend GDN (lab dùng Triton).
