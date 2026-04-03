"""Print a clean summary of the processed dataset."""
import json
from pathlib import Path

summary = json.loads(
    Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed/dataset_summary.json")
    .read_text(encoding="utf-8")
)

t = summary["totals"]
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Processed at  : {summary['processed_at']}")
print(f"Total files   : {summary['total_files']}")
print(f"Priority cutoff: {summary['priority_cutoff']} (Jan 15, 2026)")
print()
print("Priority Distribution")
print(f"  HIGH (>= Jan 15 2026) : {t['high_priority']} files")
print(f"  LOW  (<  Jan 15 2026) : {t['low_priority']} files")
print(f"  UNKNOWN               : {t['unknown_priority']} files")
print()
print("Segment Counts")
print(f"  AI-generated segments : {t['ai_generated_segments']}")
print(f"  Human segments        : {t['human_segments']}")
print(f"  Formula segments      : {t['formula_segments']}")
print(f"  Total training records: {t['ai_generated_segments'] + t['human_segments'] + t['formula_segments']}")
print(f"  Pages via OCR         : {t['ocr_pages']}")
print()

# Per-priority breakdown
high_files = [f for f in summary["files"] if f["priority"] == "HIGH"]
low_files  = [f for f in summary["files"] if f["priority"] == "LOW"]

print("HIGH Priority Files (post Jan-15 retrain, more accurate labels):")
for f in sorted(high_files, key=lambda x: x["submission_date"] or ""):
    ai_pct = f"AI={f['ai_percentage_reported']}%" if f["ai_percentage_reported"] is not None else "AI=N/A"
    segs = f["segments"]
    print(f"  [{f['submission_date']}] {ai_pct:8s} {segs:4d} segs | {f['file_name'][:55]}")

print()
print("LOW Priority Files (pre-retrain model, less accurate labels):")
for f in sorted(low_files, key=lambda x: x["submission_date"] or ""):
    ai_pct = f"AI={f['ai_percentage_reported']}%" if f["ai_percentage_reported"] is not None else "AI=N/A"
    segs = f["segments"]
    print(f"  [{f['submission_date']}] {ai_pct:8s} {segs:4d} segs | {f['file_name'][:55]}")

print()
print("Output files:")
out = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed")
for p in sorted(out.iterdir()):
    print(f"  {p.name:55s}  {p.stat().st_size/1024:>8.1f} KB")
