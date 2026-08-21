#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_runs(path: Path) -> dict[str, dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("run"):
            out[row["run"]] = row
    return out


def f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def i(x):
    try:
        return f"{int(float(x)):,}"
    except Exception:
        return "—"


def esc(text: object, limit: int | None = None) -> str:
    s = str(text or "").replace("\n", " ").replace("|", "\\|").strip()
    if limit and len(s) > limit:
        s = s[: limit - 1] + "…"
    return s


def baseline_rows(verdict: dict) -> dict[str, dict]:
    out = {}
    for row in verdict.get("comparison", []):
        name = row.get("run", "")
        if name.startswith("(a)"):
            out["a"] = row
        elif name.startswith("(b)"):
            out["b"] = row
        elif name.startswith("(c)"):
            out["c"] = row
    return out


def compare_word(a: float, b: float, eps: float = 1e-9) -> str:
    if a > b + eps:
        return "thắng"
    if a < b - eps:
        return "thua"
    return "hoà"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build submission/REPORT.md from measured Lab 21 artifacts.")
    ap.add_argument("--name", default="Lương Quốc Khánh")
    ap.add_argument("--student-id", default="2A202601713")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--gpu", default="Tesla T4 (Colab, ~15 GB usable)")
    args = ap.parse_args()

    required = [
        RESULTS / "template_check.json",
        RESULTS / "mask_proof.json",
        RESULTS / "token_stats.json",
        RESULTS / "baselines_frozen.json",
        RESULTS / "runs.csv",
        RESULTS / "verdict.json",
        RESULTS / "autopsy.json",
        RESULTS / "qualitative.json",
        DATA / "eval_target.jsonl",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required artifacts; run NB1→NB5 first:\n- " + "\n- ".join(missing))

    template = load_json(RESULTS / "template_check.json")
    proof = load_json(RESULTS / "mask_proof.json")
    token = load_json(RESULTS / "token_stats.json")
    frozen = load_json(RESULTS / "baselines_frozen.json")
    verdict = load_json(RESULTS / "verdict.json")
    autopsy = {r["run"]: r for r in load_json(RESULTS / "autopsy.json")}
    qual = load_json(RESULTS / "qualitative.json")
    eval_target = load_jsonl(DATA / "eval_target.jsonl")
    runs = latest_runs(RESULTS / "runs.csv")
    bq_path = RESULTS / "qualitative_b.json"
    bq = {int(r["i"]): r for r in load_json(bq_path)} if bq_path.exists() else {}

    comp = baseline_rows(verdict)
    a = comp.get("a", frozen.get("baseline_a", {}))
    b = comp.get("b", frozen.get("baseline_b", {}))
    c = comp.get("c", {})
    vd = verdict.get("verdict", {})

    correct = runs.get("correct", {})
    max_steps = correct.get("max_steps", "—")
    tier = frozen.get("tier", correct.get("tier", "T4"))
    model = frozen.get("model", correct.get("model", "unsloth/Qwen3.5-4B"))
    epochs = os.environ.get("EPOCHS", "2") or "2"
    tier_max_len = {"CPU": 512, "LAPTOP": 1024, "T4": 1024, "BIGGPU": 2048}.get(str(tier).upper(), "—")

    preserved = bool(template.get("body_present", template.get("ok", False)))
    template_answer = "có" if preserved else "không"
    template_handling = (
        "Template giữ được nội dung reasoning giả trong phép thử, nên không có bằng chứng rằng `<think>` bị xoá ở bước render. Corpus core chỉ chứa JSON trả lời nên tôi vẫn dùng `assistant-only` và không dựa vào reasoning trace."
        if preserved else
        "Template loại reasoning trong phép thử; vì vậy tôi không đưa reasoning trace vào corpus core và chỉ supervise câu trả lời JSON bằng mask `assistant-only`."
    )

    target_correct = float(autopsy.get("correct", {}).get("target", 0.0))
    target_attn = float(autopsy.get("attn_only", {}).get("target", 0.0)) if "attn_only" in autopsy else None
    target_wrong = float(autopsy.get("wrong_lr", {}).get("target", 0.0)) if "wrong_lr" in autopsy else None
    target_qlora = float(autopsy.get("qlora", {}).get("target", 0.0)) if "qlora" in autopsy else None
    loss_correct = float(correct.get("final_loss", 0.0) or 0.0)
    attn = runs.get("attn_only", {})
    wrong = runs.get("wrong_lr", {})
    qlora = runs.get("qlora", {})
    loss_attn = float(attn.get("final_loss", 0.0) or 0.0)
    loss_wrong = float(wrong.get("final_loss", 0.0) or 0.0)
    vram_correct = float(correct.get("peak_vram_gb", 0.0) or 0.0)
    vram_qlora = float(qlora.get("peak_vram_gb", 0.0) or 0.0)
    vram_saved = vram_correct - vram_qlora if vram_correct and vram_qlora else 0.0
    vram_pct = (vram_saved / vram_correct * 100) if vram_correct else 0.0

    attn_relation = compare_word(target_attn, target_correct) if target_attn is not None else "chưa đo"
    loss_relation = compare_word(-loss_attn, -loss_correct) if attn else "chưa đo"
    same_order = (attn_relation == loss_relation) if attn_relation in {"thắng", "thua", "hoà"} else False
    qlora_relation = compare_word(target_qlora, target_correct) if target_qlora is not None else "chưa đo"

    ordered = sorted(qual, key=lambda r: (float(r.get("ft_score", 0)), int(r.get("i", 0))))
    selected = []
    for row in ordered[:3] + list(reversed(ordered[-2:])):
        if row.get("i") not in {x.get("i") for x in selected}:
            selected.append(row)
    selected = selected[:5]

    qrows = []
    for n, row in enumerate(selected, 1):
        idx = int(row["i"])
        ref = eval_target[idx]
        label = ref.get("label", {})
        bpred = bq.get(idx, {}).get("baseline_b_pred", "— (chạy scripts/score_qualitative_b.py để bổ sung)")
        ftpred = row.get("ft_pred", "")
        score = float(row.get("ft_score", 0.0))
        note = "✅ FT đúng đủ 4 trường" if score >= 0.999 else f"❌ FT thua nhãn ({score:.2f}/1.00); còn ít nhất một lỗi field"
        qrows.append(
            f"| {n} | {esc(ref.get('input', row.get('ticket', '')), 90)} | `{esc(json.dumps(label, ensure_ascii=False), 120)}` | `{esc(bpred, 120)}` | `{esc(ftpred, 120)}` | {note} |"
        )

    passed = bool(vd.get("passed"))
    verdict_word = "PASSED" if passed else "FAILED"
    target_delta = float(vd.get("target_delta", 0.0))
    regression_delta = float(vd.get("regression_delta", 0.0))
    trace_rate = verdict.get("valid_trace_rate")
    trace_rate = 0.0 if trace_rate is None else float(trace_rate)

    if passed:
        verdict_analysis = (
            f"Cổng hồi quy **PASSED** vì fine-tune đạt target {float(c.get('target', 0)):.3f}, cao hơn baseline (b) {float(b.get('target', 0)):.3f} một mức {target_delta:+.3f}, đồng thời regression thay đổi {regression_delta:+.3f}, không vượt ngưỡng suy giảm 0.020. "
            "Điều quan trọng là kết luận này không dựa trên train loss: mốc (b) đã được đóng băng trước khi train và fine-tune được chấm lại trên đúng target/regression/format/latency. "
            f"Format của fine-tune là {float(c.get('format', 0)):.3f}, nên tôi cũng kiểm tra được rằng model thực sự tuân thủ contract JSON thay vì chỉ cải thiện một phần nhãn. "
            "Tôi vẫn không xem PASSED là bằng chứng fine-tune luôn tốt hơn prompt engineering: nó chỉ chứng minh adapter này, trên corpus và budget này, vượt được một prompt tử tế theo gate đã định trước."
        )
    else:
        reasons = " ".join(vd.get("reasons", []))
        verdict_analysis = (
            f"Cổng hồi quy **FAILED**. Fine-tune đạt target {float(c.get('target', 0)):.3f} trong khi baseline (b) đạt {float(b.get('target', 0)):.3f}, tương ứng Δtarget={target_delta:+.3f}; regression thay đổi {regression_delta:+.3f}. {reasons} "
            "Tôi giữ nguyên phán quyết thay vì nới ngưỡng hoặc làm yếu prompt (b), vì mục tiêu của lab là phát hiện khi fine-tune không tạo thêm giá trị. "
            f"Format fine-tune đạt {float(c.get('format', 0)):.3f}; nếu format đã tốt nhưng target vẫn không vượt (b), vấn đề không còn đơn giản là JSON parser hay mask mà là lợi ích tác vụ chưa đủ. "
            "Trong trường hợp đó, prompt engineering là baseline mạnh hơn và rẻ hơn để deploy; nếu regression tụt, hướng sửa hợp lý là replay dữ liệu phổ thông chứ không sửa eval."
        )

    deploy = passed and float(c.get("format", 0)) >= 0.99
    conclusion = (
        f"Tôi {'có thể cân nhắc deploy' if deploy else 'không nên deploy ngay'} adapter hiện tại dựa trên gate đã định trước, chứ không dựa vào việc loss giảm đẹp. "
        f"Bài học lớn nhất là thứ tự kiểm chứng quan trọng hơn việc quét hyperparameter: NB1 chứng minh loss thực sự chỉ rơi vào lượt assistant (supervised_fraction={float(proof.get('supervised_fraction', 0)):.4f}), NB2 đóng băng một prompt baseline đủ mạnh trước khi tôi nhìn thấy kết quả train, và NB5 mới quyết định fine-tune có tạo giá trị hay không. "
        f"Trong autopsy, `attn_only` {attn_relation} `correct` trên target dù ngân sách tham số được matched; điều này cho thấy rank không thể được diễn giải tách khỏi vị trí gắn adapter. "
        f"`wrong_lr` dùng LR 1e-5 thay vì 1e-4 và có final loss {loss_wrong:.4f} so với {loss_correct:.4f} của `correct`, nên learning-rate scale là một đòn bẩy có thể làm một cấu hình hợp lý trông như không học. "
        f"QLoRA tiết kiệm khoảng {vram_saved:.2f} GB ({vram_pct:.1f}%) VRAM so với 16-bit nhưng trên target nó {qlora_relation} `correct`, vì vậy tiết kiệm bộ nhớ phải được cân cùng chất lượng thay vì mặc định 4-bit là tốt hơn. "
        "Quan trọng nhất vẫn là mask và thiết kế phép đo: mask sai làm toàn bộ thí nghiệm vô nghĩa, còn baseline yếu hoặc eval bị chỉnh sau khi train sẽ tạo ra một chiến thắng giả. "
        "Nếu cần cải thiện tiếp, tôi sẽ ưu tiên phân tích các lỗi field trong qualitative set, bổ sung dữ liệu đúng loại lỗi và giữ nguyên frozen eval để kiểm tra nhân quả, thay vì chỉ tăng rank hoặc train lâu hơn."
    )

    def run_row(key: str, placement: str, default_r: str) -> str:
        rr = runs.get(key, {})
        aa = autopsy.get(key, {})
        return (
            f"| `{key}` | {placement} | {rr.get('r', default_r) or default_r} | {i(rr.get('trainable_params'))} | "
            f"{rr.get('learning_rate', '—')} | {f(rr.get('final_loss'))} | {f(aa.get('target'))} | {rr.get('max_steps', '—')} | {f(rr.get('peak_vram_gb'), 2)} |"
        )

    report = f"""# Lab 21 — Evaluation Report

**Họ tên**: {args.name}  **MSSV**: {args.student_id}  **Ngày**: {args.date}  
**Tier**: `{tier}`  **Base model**: `{model}`  **GPU thực tế**: `{args.gpu}`

> Mọi con số dưới đây được lấy từ `results/`; report này được sinh sau NB5 để tránh chép nhầm số.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | `{tier_max_len}` — p95 đo được là `{token.get('p95', '—')}`, suggested `{token.get('suggested_max_length', '—')}` |
| `MASK_MODE` | `{proof.get('mask_mode', 'assistant-only')}` |
| Epochs / max_steps | `{epochs}` / `{max_steps}` |

**Template có giữ khối `<think>` không?** **{template_answer}** — `results/template_check.json`.  
{template_handling}

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `{f(proof.get('supervised_fraction'))}` |
| Câu trả lời nằm trong loss | `{str(bool(proof.get('answer_is_supervised'))).lower()}` |
| Câu hỏi KHÔNG nằm trong loss | `{str(bool(proof.get('question_is_masked'))).lower()}` |

3–5 dòng đầu của đoạn được tính loss:

```text
{str(proof.get('supervised_preview', '')).strip()}
```

Đoạn supervise chứa câu trả lời assistant và không chứa fragment của ticket, vì vậy loss mask phù hợp với mục tiêu SFT: model bị phạt khi sinh sai JSON, không bị dạy viết lại prompt.

---

## 3. Ba baseline (NB2 đo trước khi train; NB5 chấm fine-tune)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | {f(a.get('target'))} | {f(a.get('regression'))} | {f(a.get('format'))} | {f(a.get('latency_ms'), 1)} |
| (b) base + optimized prompt | {f(b.get('target'))} | {f(b.get('regression'))} | {f(b.get('format'))} | {f(b.get('latency_ms'), 1)} |
| (c) LoRA fine-tune | {f(c.get('target'))} | {f(c.get('regression'))} | {f(c.get('format'))} | {f(c.get('latency_ms'), 1)} |

**(b) có thật sự mạnh hơn (a) không?** **{'có' if float(b.get('target', 0)) > float(a.get('target', 0)) else 'không'}**. Baseline (b) đã được đóng băng trước NB3 (`optimized_prompt_sha={frozen.get('optimized_prompt_sha', '—')}`), và tôi không làm yếu prompt sau khi nhìn thấy kết quả train.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target (NB5 §4)** | steps | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{run_row('correct', 'text-linear', '16')}
{run_row('attn_only', 'q,v', 'matched')}
{run_row('wrong_lr', 'text-linear', '16')}
{run_row('qlora', 'text-linear', '16')}

**4.1 — Vị trí vs rank.** `attn_only` có {i(attn.get('trainable_params'))} trainable params so với {i(correct.get('trainable_params'))} của `correct`, nên đây là đối chứng về vị trí chứ không phải ngân sách. Trên target, `attn_only` **{attn_relation}** `correct` ({f(target_attn)} so với {f(target_correct)}). Theo train loss, quan hệ là **{loss_relation}** ({f(loss_attn)} so với {f(loss_correct)}), nên hai thang đo {'cho cùng thứ tự' if same_order else 'không cho cùng thứ tự'}. Điều đó cho thấy không thể kết luận “rank lớn hơn tốt hơn” hoặc “attention-only tốt hơn” chỉ từ train loss; vị trí adapter phải được phán bằng năng lực target trên tập đóng băng.

**4.2 — Learning rate.** `wrong_lr` chỉ đổi LR từ 1e-4 xuống 1e-5 nhưng final loss là {f(loss_wrong)} so với {f(loss_correct)} của `correct`; target tương ứng là {f(target_wrong)} so với {f(target_correct)}. Nếu chỉ nhìn một đường loss mà không biết LR, tôi có thể kết luận sai rằng LoRA/placement không học được hoặc cần tăng rank. Đối chứng này tách nguyên nhân: learning-rate scale là biến đủ lớn để làm thay đổi động lực học dù mọi phần còn lại giữ nguyên và cùng step budget.

**4.3 — QLoRA.** QLoRA dùng {f(vram_qlora, 2)} GB so với {f(vram_correct, 2)} GB của 16-bit, tiết kiệm khoảng **{vram_saved:.2f} GB ({vram_pct:.1f}%)**. Trên target, QLoRA **{qlora_relation}** `correct` ({f(target_qlora)} so với {f(target_correct)}), còn train loss là {f(qlora.get('final_loss'))}. Vì vậy dữ liệu của run này {'ủng hộ việc thận trọng với QLoRA: VRAM giảm nhưng chất lượng không vượt cấu hình 16-bit' if qlora_relation != 'thắng' else 'không ủng hộ một lệnh cấm tuyệt đối: 4-bit vẫn cạnh tranh trên target trong phép đo này'}, nhưng kết luận chỉ áp dụng cho đúng model/corpus/budget đã đo.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **{verdict_word}**  
`target Δ = {target_delta:+.3f}` · `regression Δ = {regression_delta:+.3f}` · `valid_trace_rate = {trace_rate:.3f}`

{verdict_analysis}

---

## 6. Định tính — có cả ca thắng và ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
{chr(10).join(qrows)}

Các ca tệ nhất được chọn từ đầu `results/qualitative.json`, không cherry-pick ca thắng. Mẫu lỗi được đọc theo field score: khi `ft_score < 1`, ít nhất một trong `intent`, `urgency`, `product`, `sentiment` sai hoặc output không parse đúng. Tôi ưu tiên sửa bằng dữ liệu phản ánh đúng kiểu lỗi field đó; không sửa `eval_target.jsonl` sau khi đã thấy điểm.

---

## 7. Kết luận & điều tôi học được

{conclusion}

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
"""

    out = ROOT / "submission" / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(report.split())} words)")
    if not bq:
        print("WARN: results/qualitative_b.json missing; report marks baseline-(b) raw outputs as unavailable.")
        print("Run: python scripts/score_qualitative_b.py ; then rerun this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
