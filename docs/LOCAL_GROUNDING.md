# Chạy grounding simulator ở máy local

Production không để LLM bịa observation khi có thể tránh. Service repo-context
(`src/repo_context_service/`) tải snapshot repo thật của task từ GitHub rồi **chạy thật** các lệnh đọc
(`cat`, `grep`, `find`, `ls`, `sed`, một phần `git`) trên snapshot đó, có tính cả các file model đã sửa
trước đó. Chỉ khi không chạy thật được thì simulator mới gọi DeepSeek. Tài liệu này dựng lại đúng
service đó ở máy local để pilot sinh data ([scripts/pilot_glm_thinking_teacher.py](../scripts/pilot_glm_thinking_teacher.py))
có observation giống production.

## Cần chuẩn bị (một lần)

| Thứ cần | Ở đâu | Ghi chú |
|---|---|---|
| Manifest production | `E:\albedo-storage-temp\repo-context\manifest.json` | Công khai tại `https://albedo.tech/datasets/manifest.json`, sha256 phải là `e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40`. Dùng để tra `sample_id` → `instance_id`, nên **không cần** tải dataset gốc (Open-SWE-Traces riêng đã 42,6 GB) |
| GitHub token | dòng `ALBEDO_REPO_CONTEXT_GITHUB_TOKEN=...` trong `.env` ở gốc repo | Fine-grained token, chỉ cần quyền đọc repo public. `.env` đã nằm trong `.gitignore`. Tự ghi vào file, không dán vào chat hay dòng lệnh. Thiếu token thì GitHub chỉ cho 60 request/giờ và task open-swe-traces, swe-hero không ground được |
| Cache snapshot | `E:\albedo-storage-temp\repo-context\cache` | Mỗi snapshot ≤ 500 MB, giới hạn cache mặc định 15 GB (`MAX_CACHE_GB`) |

Service chạy thẳng bằng Python của Windows (đã có fastapi, uvicorn, httpx, pyarrow). Không dùng
WSL: ổ ảo của WSL nằm trên C:, mà C: gần đầy.

Tải manifest (khi mạng tới albedo.tech ổn định):

```bash
py -3 -c "import urllib.request,hashlib; d=urllib.request.urlopen('https://albedo.tech/datasets/manifest.json',timeout=300).read(); open(r'E:/albedo-storage-temp/repo-context/manifest.json','wb').write(d); print(hashlib.sha256(d).hexdigest())"
```

## Chạy

```bash
# cửa sổ 1: service (giữ mở)
bash scripts/run_repo_context_local.sh

# cửa sổ 2: kiểm tra
curl http://127.0.0.1:8093/healthz        # github_token_set và manifest_configured phải là true
py -3 scripts/check_local_grounding.py --pilot E:/albedo-storage-temp/glm-thinking-pilot-20260927
```

`check_local_grounding.py` phát lại rollout King của các task pilot qua simulator production nối với
service local, với một client LLM từ chối mọi lời gọi, nên **không tốn tiền**. Lượt nào được ground
chính xác sẽ đem so với observation production đã ghi. Kết quả nằm ở `grounding-check.json`:
`grounded_share` (tỷ lệ lượt không cần LLM) và `match_rate_when_grounded` (tỷ lệ khớp nguyên văn).
Khớp cao nghĩa là snapshot đúng commit và service chạy đúng như production.

Pilot có grounding:

```bash
py -3 scripts/pilot_glm_thinking_teacher.py run --out E:/albedo-storage-temp/glm-thinking-pilot-20260927 \
  --repo-context-url http://127.0.0.1:8093
```

Nếu không truyền `--repo-context-url`, pilot từ chối chạy, trừ khi thêm `--allow-ungrounded`.

## Giới hạn

- Snapshot giải nén trên Windows: file có tên chứa `:` hoặc trùng tên khác hoa/thường sẽ lỗi hoặc
  bị ghi đè. Hiếm với repo Python, nhưng `check_local_grounding.py` sẽ lộ ra nếu có.
- Không đặt `ALBEDO_REPO_CONTEXT_DATASET_ROOT`: nhánh dự phòng "trajectory block" (dùng khi không có
  snapshot) sẽ không có. Bản dataset production dựng lại đầy đủ nằm trên Hessian:
  `~/truong/albedo-storage/production-datasets-e3cff617` (xem `docs/PRODUCTION_SNAPSHOT.md`).
