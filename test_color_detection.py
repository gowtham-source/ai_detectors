"""
Deep-dive into how Turnitin stores highlighted (AI-detected) text in PDF.
Checks: annotations, drawings, color fills, span colors, markup.
"""
import fitz
from pathlib import Path

TEST_FILE = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/raw_data/raw_data/4_5.pdf")


def inspect_page(doc, page_num):
    page = doc[page_num]
    print(f"\n{'='*60}")
    print(f"PAGE {page_num + 1}")

    # 1. Annotations (highlights, underlines, etc.)
    annots = list(page.annots())
    print(f"  Annotations count: {len(annots)}")
    for a in annots[:5]:
        print(f"    type={a.type}, color={a.colors}, rect={a.rect}, content={a.info.get('content','')[:60]}")

    # 2. Drawings / paths (filled rectangles = highlight backgrounds)
    drawings = page.get_drawings()
    filled_rects = [d for d in drawings if d.get("fill") is not None and d.get("fill") != (1,1,1)]
    print(f"  Non-white filled drawings: {len(filled_rects)}")
    for d in filled_rects[:5]:
        print(f"    fill={d.get('fill')}, rect={d.get('rect')}")

    # 3. Text with color info via rawdict
    blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    colored_spans = []
    all_spans = []
    for block in blocks:
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    color = span.get("color", 0)
                    txt = span.get("text", "").strip()
                    if txt:
                        all_spans.append((color, txt))
                        if color != 0:
                            colored_spans.append((hex(color), txt))

    color_set = set(c for c, _ in all_spans)
    print(f"  Unique span colors: {color_set}")
    print(f"  Colored (non-black) spans: {len(colored_spans)}")
    for c, t in colored_spans[:5]:
        print(f"    {c}: {t[:80]}")

    # 4. Check if drawings overlap with text (highlight = rect over text)
    if filled_rects and all_spans:
        print(f"\n  Checking highlight overlap with text...")
        # Get text with bounding boxes
        words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_num)
        for d in filled_rects[:3]:
            drect = fitz.Rect(d["rect"])
            overlapping = [w[4] for w in words if fitz.Rect(w[:4]).intersects(drect)]
            if overlapping:
                print(f"    Fill {d.get('fill')} covers words: {' '.join(overlapping[:10])}")

    # 5. HTML extraction (sometimes preserves colors)
    html = page.get_text("html")
    import re
    colored_html = re.findall(r'color:\s*rgb\(([^)]+)\)[^>]*>([^<]{5,})', html)
    print(f"\n  HTML color spans: {len(colored_html)}")
    for rgb, text in colored_html[:5]:
        print(f"    rgb({rgb}): {text[:80]}")


def main():
    print(f"Inspecting: {TEST_FILE.name}")
    doc = fitz.open(str(TEST_FILE))
    print(f"Total pages: {len(doc)}")

    # Check pages 3, 4, 5 (content pages - 0-indexed: 2,3,4)
    for pg in [2, 3, 4]:
        if pg < len(doc):
            inspect_page(doc, pg)

    doc.close()


if __name__ == "__main__":
    main()
