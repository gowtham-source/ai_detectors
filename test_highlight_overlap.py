"""
Verify that Turnitin AI-highlight = filled cyan rect overlapping text words.
Color: (0.32, 0.78, 0.86) approx = Turnitin's AI highlight blue.
"""
import fitz
from pathlib import Path

TEST_FILE = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/raw_data/raw_data/4_5.pdf")

# Turnitin AI highlight color (approximate)
TURNITIN_AI_COLOR = (0.3203125, 0.77734375, 0.85546875)


def is_ai_highlight_color(fill_color, tol=0.02):
    """Check if a fill color matches Turnitin's AI highlight."""
    if fill_color is None or len(fill_color) != 3:
        return False
    return all(abs(fill_color[i] - TURNITIN_AI_COLOR[i]) < tol for i in range(3))


def extract_page_segments(doc, page_num):
    """
    Returns list of dicts with:
      - text: the sentence/word
      - label: 'ai_generated' or 'human'
      - bbox: bounding box
    """
    page = doc[page_num]

    # Get all highlight rectangles
    drawings = page.get_drawings()
    ai_rects = []
    for d in drawings:
        if is_ai_highlight_color(d.get("fill")):
            ai_rects.append(fitz.Rect(d["rect"]))

    # Get all words with positions
    words = page.get_text("words")  # (x0,y0,x1,y1, word, block_no, line_no, word_no)

    # Tag each word as AI or human based on rect overlap
    tagged_words = []
    for w in words:
        word_rect = fitz.Rect(w[0], w[1], w[2], w[3])
        word_text = w[4]
        is_ai = any(word_rect.intersects(r) for r in ai_rects)
        tagged_words.append({
            "text": word_text,
            "label": "ai_generated" if is_ai else "human",
            "block": w[5],
            "line": w[6],
            "bbox": (w[0], w[1], w[2], w[3])
        })

    # Group consecutive words of same label into segments
    segments = []
    if not tagged_words:
        return segments

    current_label = tagged_words[0]["label"]
    current_words = [tagged_words[0]["text"]]

    for tw in tagged_words[1:]:
        if tw["label"] == current_label:
            current_words.append(tw["text"])
        else:
            segments.append({
                "text": " ".join(current_words),
                "label": current_label
            })
            current_label = tw["label"]
            current_words = [tw["text"]]

    segments.append({"text": " ".join(current_words), "label": current_label})
    return segments


def main():
    doc = fitz.open(str(TEST_FILE))
    print(f"File: {TEST_FILE.name}  |  Pages: {len(doc)}")
    print(f"Testing pages 3-6 (0-indexed: 2-5)\n")

    for pg in range(2, min(6, len(doc))):
        segs = extract_page_segments(doc, pg)
        print(f"\n{'='*60}")
        print(f"PAGE {pg+1}  |  Segments: {len(segs)}")
        ai_count = sum(1 for s in segs if s["label"] == "ai_generated")
        human_count = sum(1 for s in segs if s["label"] == "human")
        print(f"  AI segments: {ai_count}  |  Human segments: {human_count}")
        print()
        for s in segs:
            label_tag = "[AI]  " if s["label"] == "ai_generated" else "[HUM] "
            preview = s["text"][:120].replace("\n", " ")
            print(f"  {label_tag} {preview}")

    doc.close()


if __name__ == "__main__":
    main()
