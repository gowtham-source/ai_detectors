import json
from pathlib import Path

f = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed/4_5.json")
data = json.loads(f.read_text(encoding="utf-8"))
print("File:", data["file_name"])
print("Priority:", data["priority"])
print("Submission:", data["submission_date"])
print("AI%:", data["ai_percentage_reported"])
print("Total segs:", len(data["segments"]))

ai_segs = [s for s in data["segments"] if s["label"] == "ai_generated"]
hum_segs = [s for s in data["segments"] if s["label"] == "human"]
form_segs = [s for s in data["segments"] if s["label"] == "formula"]
print(f"AI={len(ai_segs)}, Human={len(hum_segs)}, Formula={len(form_segs)}")

print("\n=== AI sample ===")
print(ai_segs[0]["text"][:300])
print("\n=== HUMAN sample ===")
print(hum_segs[0]["text"][:300])
if form_segs:
    print("\n=== FORMULA sample ===")
    print(form_segs[0]["text"][:200])

jl = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed/dataset.jsonl")
lines = jl.read_text(encoding="utf-8").strip().split("\n")
print(f"\n=== JSONL: {len(lines)} total records ===")
first = json.loads(lines[0])
print("Keys:", list(first.keys()))
print(json.dumps(first, indent=2)[:500])
