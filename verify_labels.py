"""Verify all three labels are correct in the processed output."""
import json
from pathlib import Path

f = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed/Chapter 2-LR.json")
data = json.loads(f.read_text(encoding="utf-8"))

ai   = [s for s in data["segments"] if s["label"] == "ai_generated"]
para = [s for s in data["segments"] if s["label"] == "ai_paraphrased"]
hum  = [s for s in data["segments"] if s["label"] == "human"]

print(f"File: {data['file_name']}")
print(f"Priority: {data['priority']}  |  Submission: {data['submission_date']}  |  Turnitin AI%: {data['ai_percentage_reported']}")
print(f"\nAI={len(ai)}, Paraphrased={len(para)}, Human={len(hum)}\n")

print("=== AI_GENERATED samples ===")
for s in ai[:3]:
    print(f"  [pg{s['page_num']}] {s['text'][:120]}")

print("\n=== AI_PARAPHRASED samples ===")
for s in para:
    print(f"  [pg{s['page_num']}] {s['text'][:300]}")

print("\n=== HUMAN samples ===")
for s in hum[:3]:
    print(f"  [pg{s['page_num']}] {s['text'][:120]}")

# Also check 4_5.json which has formulas inside highlights
f2 = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed/4_5.json")
data2 = json.loads(f2.read_text(encoding="utf-8"))
ai2   = [s for s in data2["segments"] if s["label"] == "ai_generated"]
para2 = [s for s in data2["segments"] if s["label"] == "ai_paraphrased"]
hum2  = [s for s in data2["segments"] if s["label"] == "human"]
print(f"\n=== 4_5.pdf (formula test) ===")
print(f"AI={len(ai2)}, Paraphrased={len(para2)}, Human={len(hum2)}")
print("Sample AI segments (may include formulas inside highlights):")
for s in ai2[:5]:
    print(f"  [pg{s['page_num']}] {s['text'][:150]}")
