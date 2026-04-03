"""Quick dataset analysis before training."""
import json
from pathlib import Path
from collections import Counter

jsonl = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed/dataset.jsonl")
records = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").strip().split("\n")]

labels   = Counter(r["label"] for r in records)
priority = Counter(r["priority"] for r in records)
lengths  = [len(r["text"]) for r in records]

print(f"Total records : {len(records)}")
print(f"\nLabel distribution:")
for lbl, cnt in labels.most_common():
    pct = cnt / len(records) * 100
    print(f"  {lbl:<20} {cnt:>5}  ({pct:.1f}%)")

print(f"\nPriority distribution:")
for pri, cnt in priority.most_common():
    print(f"  {pri:<10} {cnt:>5}")

print(f"\nText length stats (chars):")
print(f"  min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//len(lengths)}")
p50 = sorted(lengths)[len(lengths)//2]
p95 = sorted(lengths)[int(len(lengths)*0.95)]
print(f"  p50={p50}, p95={p95}")

# HIGH priority label breakdown
high = [r for r in records if r["priority"] == "HIGH"]
high_labels = Counter(r["label"] for r in high)
print(f"\nHIGH priority label breakdown ({len(high)} records):")
for lbl, cnt in high_labels.most_common():
    print(f"  {lbl:<20} {cnt:>5}")

# Token estimate (rough: 1 token ≈ 4 chars)
print(f"\nApprox token count at p95 cutoff: {p95//4} tokens")
print(f"Recommended max_length for DeBERTa: 512 tokens")
