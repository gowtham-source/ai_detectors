"""Inspect all unique fill colors across all pages of Chapter 2-LR.pdf to find the purple AI-paraphrase color."""
import fitz
from pathlib import Path

PDF = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/raw_data/raw_data/Chapter 2-LR.pdf")

doc = fitz.open(str(PDF))
print(f"Pages: {len(doc)}")

all_colors = {}
for pg in range(len(doc)):
    page = doc[pg]
    for d in page.get_drawings():
        fill = d.get("fill")
        if fill and len(fill) == 3 and fill != (1,1,1) and fill != (0,0,0):
            key = tuple(round(c, 4) for c in fill)
            all_colors[key] = all_colors.get(key, 0) + 1

print("\nAll unique non-white/non-black fill colors (color: count):")
for color, count in sorted(all_colors.items(), key=lambda x: -x[1]):
    r, g, b = color
    print(f"  RGB({r:.4f}, {g:.4f}, {b:.4f})  →  hex #{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}  count={count}")

doc.close()
