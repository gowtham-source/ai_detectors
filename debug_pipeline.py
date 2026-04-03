"""Debug: find exact line causing 'Document has no get_text' error."""
import fitz
import re
from pathlib import Path

p = list(Path("c:/Users/gowth/Downloads/ai_plag_detector/data/raw_data/raw_data").glob("4*.pdf"))[0]
print("File:", p.name)
doc = fitz.open(str(p))
print("Pages:", len(doc))

# Step through each page
for pg in range(2, min(8, len(doc))):
    page = doc[pg]
    print(f"\n--- Page {pg+1} type: {type(page)} ---")
    
    # raw text
    raw = page.get_text("text")
    print(f"  text len: {len(raw)}")
    
    # words
    words = page.get_text("words")
    print(f"  words count: {len(words)}")
    
    # rawdict WITHOUT flags param (simpler)
    rd = page.get_text("rawdict")
    print(f"  rawdict blocks: {len(rd['blocks'])}")
    
    # drawings
    drawings = page.get_drawings()
    print(f"  drawings: {len(drawings)}")
    ai_draws = [d for d in drawings if d.get("fill") and len(d["fill"]) == 3 and
                abs(d["fill"][0] - 0.32) < 0.05]
    print(f"  AI-colored drawings: {len(ai_draws)}")

doc.close()
print("\nAll pages scanned successfully.")
