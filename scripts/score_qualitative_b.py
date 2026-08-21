#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import generate, report
from labkit.config import get_tier


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    qpath = ROOT / "results" / "qualitative.json"
    if not qpath.exists():
        raise SystemExit("Run NB5 first: results/qualitative.json is missing")

    qual = json.loads(qpath.read_text(encoding="utf-8"))
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    ordered = sorted(qual, key=lambda r: (float(r.get("ft_score", 0)), int(r.get("i", 0))))
    selected = []
    for row in ordered[:3] + list(reversed(ordered[-2:])):
        if row.get("i") not in {x.get("i") for x in selected}:
            selected.append(row)
    selected = selected[:5]
    indices = [int(r["i"]) for r in selected]

    tier = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
    model, tok = generate.load_base(tier)
    preds, latency = generate.generate_batch(
        model, tok, [target[idx]["input"] for idx in indices],
        system=generate.OPTIMIZED_PROMPT, label="report/baseline-b-qualitative"
    )
    rows = [
        {"i": idx, "baseline_b_pred": pred, "latency_ms_batch_avg": round(latency, 1)}
        for idx, pred in zip(indices, preds)
    ]
    report.write_json(rows, "qualitative_b.json", results_dir=ROOT / "results")
    print("saved results/qualitative_b.json for indices", indices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
