"""
AI Plagiarism Detector - Inference & Visualization
====================================================
Load a PDF, run the trained RoBERTa model on each sentence,
produce an overall AI score, and generate a Turnitin-style
color-coded HTML report.

Colors (matching Turnitin convention):
  - Cyan   (#51c6da) → AI-generated text
  - Purple (#b68bfb) → AI-paraphrased text
  - White  (no bg)   → Human-written text

Usage:
  uv run inference.py --pdf "path/to/document.pdf"
  uv run inference.py --pdf "path/to/document.pdf" --model "models/roberta_ai_detector"
"""

import argparse
import json
import re
import sys
import logging
import html as html_lib
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = Path("models/roberta_ai_detector")
MAX_LENGTH        = 256
INFER_THRESHOLD   = 0.65   # min confidence to predict non-human
MIN_SEGMENT_CHARS = 20     # skip very short segments

LABEL2ID = {"human": 0, "ai_generated": 1, "ai_paraphrased": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Colors for HTML visualization
COLORS = {
    "ai_generated":   {"bg": "#d4f5fa", "border": "#51c6da", "tag_bg": "#51c6da", "tag_text": "#fff"},
    "ai_paraphrased": {"bg": "#ece0ff", "border": "#b68bfb", "tag_bg": "#b68bfb", "tag_text": "#fff"},
    "human":          {"bg": "transparent", "border": "transparent", "tag_bg": "#e0e0e0", "tag_text": "#555"},
}

# Colors for PDF highlight annotations (RGB 0-1 float)
PDF_COLORS = {
    "ai_generated":   (0.32, 0.78, 0.85),   # Cyan  #51c6da
    "ai_paraphrased": (0.71, 0.55, 0.98),   # Purple #b68bfb
}

# ─── PDF Text Extraction ────────────────────────────────────────────────────

def extract_pages_text(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.
    Returns list of {page_num, raw_text, sentences}.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for pg_idx in range(len(doc)):
        page = doc[pg_idx]
        text = page.get_text("text").strip()
        if len(text) < 10:
            continue
        sentences = split_into_sentences(text)
        pages.append({
            "page_num": pg_idx + 1,
            "raw_text": text,
            "sentences": sentences,
        })
    doc.close()
    return pages


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex-based heuristics.
    Handles abbreviations, decimals, etc.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split on sentence-ending punctuation followed by space + uppercase
    # or newline boundaries from PDF block extraction
    raw_splits = re.split(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+',
        text
    )

    sentences = []
    for s in raw_splits:
        s = s.strip()
        if len(s) >= MIN_SEGMENT_CHARS:
            sentences.append(s)
        elif sentences:
            # Merge very short fragments with previous sentence
            sentences[-1] += " " + s
    return sentences


# ─── Model Inference ─────────────────────────────────────────────────────────

def load_model(model_dir: Path, device: torch.device):
    """Load tokenizer and model from saved checkpoint."""
    log.info(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return tokenizer, model


def classify_sentences(
    sentences: list[str],
    tokenizer,
    model,
    device: torch.device,
    threshold: float = INFER_THRESHOLD,
    batch_size: int = 16,
) -> list[dict]:
    """
    Classify each sentence. Returns list of:
      {text, label, confidence, probabilities}
    """
    results = []

    for i in range(0, len(sentences), batch_size):
        batch_texts = sentences[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
        probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        for j, (text, prob_row) in enumerate(zip(batch_texts, probs)):
            human_p = prob_row[LABEL2ID["human"]]
            non_h_p = 1.0 - human_p

            if non_h_p >= threshold:
                pred_label = ID2LABEL[int(np.argmax(prob_row[1:])) + 1]
            else:
                pred_label = "human"

            confidence = float(np.max(prob_row))
            results.append({
                "text": text,
                "label": pred_label,
                "confidence": confidence,
                "probabilities": {
                    "human": float(prob_row[0]),
                    "ai_generated": float(prob_row[1]),
                    "ai_paraphrased": float(prob_row[2]),
                },
            })

    return results


# ─── Score Computation ───────────────────────────────────────────────────────

def compute_ai_score(results: list[dict]) -> dict:
    """Compute overall AI score as % of characters flagged as AI."""
    total_chars = sum(len(r["text"]) for r in results)
    ai_gen_chars = sum(len(r["text"]) for r in results if r["label"] == "ai_generated")
    ai_para_chars = sum(len(r["text"]) for r in results if r["label"] == "ai_paraphrased")
    human_chars = sum(len(r["text"]) for r in results if r["label"] == "human")

    ai_total = ai_gen_chars + ai_para_chars
    return {
        "overall_ai_percent": round(ai_total / max(total_chars, 1) * 100, 1),
        "ai_generated_percent": round(ai_gen_chars / max(total_chars, 1) * 100, 1),
        "ai_paraphrased_percent": round(ai_para_chars / max(total_chars, 1) * 100, 1),
        "human_percent": round(human_chars / max(total_chars, 1) * 100, 1),
        "total_sentences": len(results),
        "ai_generated_sentences": sum(1 for r in results if r["label"] == "ai_generated"),
        "ai_paraphrased_sentences": sum(1 for r in results if r["label"] == "ai_paraphrased"),
        "human_sentences": sum(1 for r in results if r["label"] == "human"),
    }


# ─── HTML Report Generation ─────────────────────────────────────────────────

def generate_html_report(
    pdf_name: str,
    pages_results: list[dict],
    score: dict,
    output_path: Path,
):
    """Generate a Turnitin-style HTML report with color-coded text."""

    # Build the highlighted text body
    body_sections = []
    for page_data in pages_results:
        page_num = page_data["page_num"]
        results = page_data["results"]

        sentences_html = []
        for r in results:
            label = r["label"]
            conf = r["confidence"]
            colors = COLORS[label]
            escaped = html_lib.escape(r["text"])

            if label == "human":
                sentences_html.append(
                    f'<span class="sentence human" title="Human ({conf:.0%})">{escaped}</span> '
                )
            else:
                tag_label = "AI Generated" if label == "ai_generated" else "AI Paraphrased"
                sentences_html.append(
                    f'<span class="sentence {label}" '
                    f'style="background:{colors["bg"]};border-bottom:2px solid {colors["border"]}" '
                    f'title="{tag_label} ({conf:.0%})">'
                    f'{escaped}'
                    f'<sup class="tag" style="background:{colors["tag_bg"]};color:{colors["tag_text"]}">'
                    f'{conf:.0%}</sup>'
                    f'</span> '
                )

        body_sections.append(f"""
        <div class="page">
            <div class="page-header">Page {page_num}</div>
            <div class="page-content">{"".join(sentences_html)}</div>
        </div>
        """)

    # Donut chart values
    ai_gen_pct = score["ai_generated_percent"]
    ai_para_pct = score["ai_paraphrased_percent"]
    human_pct = score["human_percent"]
    overall_pct = score["overall_ai_percent"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Detection Report - {html_lib.escape(pdf_name)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f5f7fa;
    color: #2d3748;
    line-height: 1.6;
  }}
  .container {{ max-width: 1000px; margin: 0 auto; padding: 24px; }}

  /* Header */
  .header {{
    background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
    color: white;
    padding: 32px;
    border-radius: 12px;
    margin-bottom: 24px;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 4px; }}
  .header .subtitle {{ opacity: 0.85; font-size: 0.9rem; }}

  /* Score Panel */
  .score-panel {{
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 24px;
    background: white;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 24px;
    align-items: center;
  }}
  .donut-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .donut {{
    width: 150px;
    height: 150px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: 700;
    color: #1a365d;
  }}
  .donut-label {{ margin-top: 8px; font-size: 0.85rem; color: #718096; font-weight: 500; }}
  .breakdown {{ display: flex; flex-direction: column; gap: 12px; }}
  .breakdown-item {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-radius: 8px;
    background: #f7fafc;
  }}
  .breakdown-dot {{
    width: 14px;
    height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .breakdown-label {{ flex: 1; font-size: 0.9rem; }}
  .breakdown-value {{ font-weight: 700; font-size: 1.1rem; }}
  .breakdown-count {{ font-size: 0.8rem; color: #a0aec0; }}

  /* Legend */
  .legend {{
    display: flex;
    gap: 20px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    color: #4a5568;
  }}
  .legend-swatch {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #e2e8f0;
  }}

  /* Page blocks */
  .page {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 16px;
    overflow: hidden;
  }}
  .page-header {{
    background: #edf2f7;
    padding: 10px 20px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .page-content {{
    padding: 20px;
    font-size: 0.95rem;
    line-height: 1.8;
  }}

  /* Sentence spans */
  .sentence {{
    border-radius: 3px;
    padding: 1px 0;
    transition: opacity 0.15s;
    cursor: default;
  }}
  .sentence:hover {{ opacity: 0.8; }}
  .sentence.ai_generated {{
    background: #d4f5fa;
    border-bottom: 2px solid #51c6da;
    padding: 2px 4px;
    border-radius: 3px;
  }}
  .sentence.ai_paraphrased {{
    background: #ece0ff;
    border-bottom: 2px solid #b68bfb;
    padding: 2px 4px;
    border-radius: 3px;
  }}
  .tag {{
    font-size: 0.65rem;
    padding: 1px 4px;
    border-radius: 3px;
    margin-left: 2px;
    vertical-align: super;
    font-weight: 600;
  }}

  /* Responsive */
  @media (max-width: 640px) {{
    .score-panel {{ grid-template-columns: 1fr; }}
    .donut-wrap {{ margin-bottom: 12px; }}
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>AI Content Detection Report</h1>
    <div class="subtitle">{html_lib.escape(pdf_name)}</div>
  </div>

  <div class="score-panel">
    <div class="donut-wrap">
      <div class="donut" style="background: conic-gradient(
        #51c6da 0% {ai_gen_pct}%,
        #b68bfb {ai_gen_pct}% {ai_gen_pct + ai_para_pct}%,
        #e2e8f0 {ai_gen_pct + ai_para_pct}% 100%
      );">
        <div style="background:white;width:100px;height:100px;border-radius:50%;display:flex;align-items:center;justify-content:center;">
          {overall_pct}%
        </div>
      </div>
      <div class="donut-label">Overall AI Score</div>
    </div>
    <div class="breakdown">
      <div class="breakdown-item">
        <div class="breakdown-dot" style="background:#51c6da"></div>
        <div class="breakdown-label">AI Generated</div>
        <div class="breakdown-value">{ai_gen_pct}%</div>
        <div class="breakdown-count">{score["ai_generated_sentences"]} sentences</div>
      </div>
      <div class="breakdown-item">
        <div class="breakdown-dot" style="background:#b68bfb"></div>
        <div class="breakdown-label">AI Paraphrased</div>
        <div class="breakdown-value">{ai_para_pct}%</div>
        <div class="breakdown-count">{score["ai_paraphrased_sentences"]} sentences</div>
      </div>
      <div class="breakdown-item">
        <div class="breakdown-dot" style="background:#e2e8f0"></div>
        <div class="breakdown-label">Human Written</div>
        <div class="breakdown-value">{human_pct}%</div>
        <div class="breakdown-count">{score["human_sentences"]} sentences</div>
      </div>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:#d4f5fa;border-color:#51c6da"></div>
      AI Generated
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:#ece0ff;border-color:#b68bfb"></div>
      AI Paraphrased
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:#fff;border-color:#e2e8f0"></div>
      Human Written
    </div>
    <div class="legend-item" style="margin-left:auto;color:#a0aec0;font-size:0.8rem;">
      Hover over text for confidence scores
    </div>
  </div>

  {"".join(body_sections)}

  <div style="text-align:center;padding:24px;color:#a0aec0;font-size:0.8rem;">
    Generated by AI Plagiarism Detector &bull; Model: RoBERTa-base &bull; Threshold: {INFER_THRESHOLD}
  </div>

</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    log.info(f"HTML report saved: {output_path}")


# ─── JSON Report ─────────────────────────────────────────────────────────────

def save_json_report(pdf_name: str, pages_results: list[dict], score: dict, output_path: Path):
    """Save detailed JSON report."""
    report = {
        "file": pdf_name,
        "score": score,
        "pages": [],
    }
    for page_data in pages_results:
        report["pages"].append({
            "page_num": page_data["page_num"],
            "sentences": [
                {
                    "text": r["text"],
                    "label": r["label"],
                    "confidence": round(r["confidence"], 4),
                    "probabilities": {k: round(v, 4) for k, v in r["probabilities"].items()},
                }
                for r in page_data["results"]
            ],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"JSON report saved: {output_path}")


# ─── PDF Report (Turnitin-style highlights on original PDF) ─────────────────

def _overlap_ratio(r1: fitz.Rect, r2: fitz.Rect) -> float:
    """Compute overlap ratio between two rects."""
    if r1.is_empty or r2.is_empty:
        return 0.0
    ix = max(0, min(r1.x1, r2.x1) - max(r1.x0, r2.x0))
    iy = max(0, min(r1.y1, r2.y1) - max(r1.y0, r2.y0))
    intersection = ix * iy
    area1 = (r1.x1 - r1.x0) * (r1.y1 - r1.y0)
    if area1 == 0:
        return 0.0
    return intersection / area1


def _search_and_highlight(page: fitz.Page, sentence: str, color: tuple, added_rects: list):
    """
    Search for a sentence on a PDF page and add highlight annotations.
    Uses full-sentence search first, then falls back to word chunks.
    Returns number of quads highlighted.
    """
    highlighted = 0

    # Try full sentence search first
    quads = page.search_for(sentence, quads=True)
    if quads:
        for q in quads:
            rect = q.rect
            if any(rect.intersects(r) and _overlap_ratio(rect, r) > 0.8 for r in added_rects):
                continue
            annot = page.add_highlight_annot(q)
            annot.set_colors(stroke=color)
            annot.set_opacity(0.35)
            annot.update()
            added_rects.append(rect)
            highlighted += 1
        return highlighted

    # Fallback: search for chunks of ~6 words to handle line breaks
    words = sentence.split()
    chunk_size = min(6, len(words))
    i = 0
    while i < len(words) - 2:
        chunk = " ".join(words[i:i + chunk_size])
        quads = page.search_for(chunk, quads=True)
        if quads:
            for q in quads:
                rect = q.rect
                if any(rect.intersects(r) and _overlap_ratio(rect, r) > 0.8 for r in added_rects):
                    continue
                annot = page.add_highlight_annot(q)
                annot.set_colors(stroke=color)
                annot.set_opacity(0.35)
                annot.update()
                added_rects.append(rect)
                highlighted += 1
            i += chunk_size
        else:
            if chunk_size > 3:
                chunk_size -= 1
            else:
                i += 1
                chunk_size = min(6, len(words) - i)
    return highlighted


def _draw_turnitin_footer(page: fitz.Page, page_label: str, total_pages: int, page_no: int):
    """Draw Turnitin-style footer: logo area left, page info center, submission ID right."""
    W, H = 595, 842
    footer_y = H - 28

    # Dark footer bar
    page.draw_rect(fitz.Rect(0, H - 40, W, H), color=None, fill=(0.18, 0.18, 0.2))

    # "turnitin" logo text (blue-ish)
    page.insert_text(
        fitz.Point(20, footer_y), "turnitin",
        fontsize=9, fontname="hebo", color=(0.32, 0.78, 0.85)
    )

    # Page info center
    center_text = f"Page {page_no} of {total_pages}  -  {page_label}"
    page.insert_text(
        fitz.Point(200, footer_y), center_text,
        fontsize=8, fontname="helv", color=(0.8, 0.8, 0.8)
    )

    # Right side label
    page.insert_text(
        fitz.Point(430, footer_y), "AI Writing Detection",
        fontsize=8, fontname="helv", color=(0.8, 0.8, 0.8)
    )


def _create_cover_page(doc: fitz.Document, pdf_name: str, score: dict):
    """
    Insert two pages at the start:
      Page 1 - Cover Page  (Turnitin style: white, doc details)
      Page 2 - AI Writing Overview  (% detected + detection groups)
    """
    W, H = 595, 842
    total_pages = len(doc) + 2  # +2 for the two pages we're adding
    GRAY    = (0.45, 0.45, 0.45)
    DGRAY   = (0.2,  0.2,  0.2)
    LGRAY   = (0.85, 0.85, 0.85)
    CYAN    = (0.32, 0.78, 0.85)
    PURPLE  = (0.71, 0.55, 0.98)
    BLACK   = (0.0,  0.0,  0.0)
    WHITE   = (1.0,  1.0,  1.0)
    DIVIDER = (0.88, 0.88, 0.88)

    # ── Page 1: Cover Page ────────────────────────────────────────────────────
    p1 = doc.new_page(pno=0, width=W, height=H)

    # "A I" initials box (top-left, bold large text like Turnitin's "A B")
    stem_initials = "".join(w[0].upper() for w in pdf_name.replace("-", " ").replace("_", " ").split()[:2])
    p1.insert_text(fitz.Point(36, 55), stem_initials, fontsize=28, fontname="hebo", color=BLACK)

    # Document name below initials (smaller, gray)
    # Truncate if too long
    display_name = pdf_name if len(pdf_name) <= 55 else pdf_name[:52] + "..."
    p1.insert_text(fitz.Point(36, 80), display_name, fontsize=11, fontname="helv", color=DGRAY)

    # Quick Submit row icons (simulated as small squares + text)
    y_sub = 105
    for label in ["Quick Submit", "Quick Submit"]:
        p1.draw_rect(fitz.Rect(36, y_sub - 9, 46, y_sub + 1), color=LGRAY, fill=LGRAY)
        p1.insert_text(fitz.Point(52, y_sub), label, fontsize=9, fontname="helv", color=GRAY)
        y_sub += 18
    # Author row
    p1.draw_rect(fitz.Rect(36, y_sub - 9, 46, y_sub + 1), color=LGRAY, fill=LGRAY)
    p1.insert_text(fitz.Point(52, y_sub), "AI Detection Report", fontsize=9, fontname="helv", color=GRAY)

    # Horizontal divider
    p1.draw_line(fitz.Point(36, 158), fitz.Point(W - 36, 158), color=DIVIDER, width=0.8)

    # "Document Details" heading
    p1.insert_text(fitz.Point(36, 182), "Document Details", fontsize=13, fontname="hebo", color=BLACK)

    # Metadata table (left column: labels, right column: values)
    import datetime as _dt
    now_str = _dt.datetime.now().strftime("%b %d, %Y, %I:%M %p")
    total_words = score["total_sentences"] * 18  # rough estimate
    total_chars = score["total_sentences"] * 95

    meta_rows = [
        ("Submission ID",   f"oid::{abs(hash(pdf_name)) % 9999999999:010d}"),
        ("Submission Date", now_str),
        ("Download Date",   now_str),
        ("File Name",       display_name),
        ("File Size",       "—"),
    ]

    y_meta = 210
    for key, val in meta_rows:
        p1.insert_text(fitz.Point(36, y_meta), key, fontsize=9, fontname="helv", color=GRAY)
        p1.insert_text(fitz.Point(36, y_meta + 14), val, fontsize=10, fontname="helv", color=DGRAY)
        y_meta += 42

    # Right-side stats box (pages / words / chars)
    stats_x, stats_y = 340, 210
    stats = [
        (f"{len(doc)} Pages",),
        (f"{total_words:,} Words",),
        (f"{total_chars:,} Characters",),
    ]
    box_h = 34
    for (stat_text,) in stats:
        box_rect = fitz.Rect(stats_x, stats_y, stats_x + 140, stats_y + box_h)
        p1.draw_rect(box_rect, color=DIVIDER, fill=WHITE, width=0.8)
        p1.insert_text(fitz.Point(stats_x + 10, stats_y + 22), stat_text,
                       fontsize=10, fontname="helv", color=DGRAY)
        stats_y += box_h + 6

    _draw_turnitin_footer(p1, "Cover Page", total_pages, 1)

    # ── Page 2: AI Writing Overview ───────────────────────────────────────────
    p2 = doc.new_page(pno=1, width=W, height=H)

    overall = score["overall_ai_percent"]
    ai_gen_pct  = score["ai_generated_percent"]
    ai_gen_cnt  = score["ai_generated_sentences"]
    ai_para_pct = score["ai_paraphrased_percent"]
    ai_para_cnt = score["ai_paraphrased_sentences"]

    # Big headline
    headline = f"{overall}% detected as AI"
    p2.insert_text(fitz.Point(36, 80), headline, fontsize=26, fontname="hebo", color=BLACK)

    # Subtitle
    subtitle = ("The percentage indicates the combined amount of likely AI-generated text as\n"
                "well as likely AI-generated text that was also likely AI-paraphrased.")
    p2.insert_textbox(
        fitz.Rect(36, 90, 310, 145), subtitle,
        fontsize=9, fontname="helv", color=GRAY
    )

    # Caution box (right side, cyan border)
    caution_rect = fitz.Rect(320, 60, W - 36, 145)
    p2.draw_rect(caution_rect, color=CYAN, fill=(0.92, 0.98, 1.0), width=1.0)
    p2.insert_textbox(
        fitz.Rect(328, 70, W - 44, 140),
        "Caution: Review required.\n\nIt is essential to understand the limitations of AI detection "
        "before making decisions about a student's work.",
        fontsize=8, fontname="helv", color=(0.1, 0.3, 0.4)
    )

    # Divider
    p2.draw_line(fitz.Point(36, 165), fitz.Point(W - 36, 165), color=DIVIDER, width=0.8)

    # "Detection Groups" heading
    p2.insert_text(fitz.Point(36, 195), "Detection Groups", fontsize=13, fontname="hebo", color=BLACK)

    # Group 1 — AI-generated only
    g_y = 225
    # Cyan circle icon
    p2.draw_circle(fitz.Point(52, g_y + 6), 10, color=None, fill=CYAN)
    p2.insert_text(fitz.Point(48, g_y + 10), "AI", fontsize=6, fontname="hebo", color=WHITE)

    count_label = f"{ai_gen_cnt}  AI-generated only  {ai_gen_pct}%"
    p2.insert_text(fitz.Point(70, g_y + 11), count_label, fontsize=11, fontname="hebo", color=DGRAY)
    g_y += 24
    p2.insert_textbox(
        fitz.Rect(70, g_y, 480, g_y + 30),
        "Likely AI-generated text from a large-language model.",
        fontsize=9, fontname="helv", color=GRAY
    )

    # Group 2 — AI-paraphrased
    g_y += 48
    p2.draw_circle(fitz.Point(52, g_y + 6), 10, color=None, fill=PURPLE)
    p2.insert_text(fitz.Point(46, g_y + 10), "AP", fontsize=6, fontname="hebo", color=WHITE)

    count_label2 = f"{ai_para_cnt}  AI-generated text that was AI-paraphrased  {ai_para_pct}%"
    p2.insert_text(fitz.Point(70, g_y + 11), count_label2, fontsize=11, fontname="hebo", color=DGRAY)
    g_y += 24
    p2.insert_textbox(
        fitz.Rect(70, g_y, 480, g_y + 40),
        "Likely AI-generated text that was likely revised using an AI-paraphrase tool or word spinner.",
        fontsize=9, fontname="helv", color=GRAY
    )

    _draw_turnitin_footer(p2, "AI Writing Overview", total_pages, 2)


def generate_pdf_report(
    pdf_path: str,
    pages_results: list[dict],
    score: dict,
    output_path: Path,
):
    """
    Generate a Turnitin-style PDF with highlight annotations on the original document.
    Adds a cover page with score summary.
    """
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).name

    # Build mapping: page_num -> list of (sentence_text, label, confidence)
    page_highlights = {}
    for page_data in pages_results:
        pg = page_data["page_num"]
        for r in page_data["results"]:
            if r["label"] != "human":
                page_highlights.setdefault(pg, []).append((r["text"], r["label"], r["confidence"]))

    # Add highlight annotations to each page
    total_highlighted = 0
    for pg_num, highlights in page_highlights.items():
        pg_idx = pg_num - 1
        if pg_idx < 0 or pg_idx >= len(doc):
            continue
        page = doc[pg_idx]
        added_rects = []

        for sentence, label, conf in highlights:
            color = PDF_COLORS.get(label)
            if not color:
                continue
            n = _search_and_highlight(page, sentence, color, added_rects)
            total_highlighted += n

    log.info(f"  PDF: {total_highlighted} text regions highlighted across {len(page_highlights)} pages")

    # Insert cover page at the beginning
    _create_cover_page(doc, pdf_name, score)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()
    log.info(f"PDF report saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Plagiarism Detector - Inference")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_DIR), help="Path to trained model directory")
    parser.add_argument("--threshold", type=float, default=INFER_THRESHOLD, help="Confidence threshold for non-human prediction")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        log.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    model_dir = Path(args.model)
    if not model_dir.exists():
        log.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    tokenizer, model = load_model(model_dir, device)

    # Extract text from PDF
    log.info(f"Processing PDF: {pdf_path.name}")
    pages = extract_pages_text(str(pdf_path))
    total_sentences = sum(len(p["sentences"]) for p in pages)
    log.info(f"Extracted {total_sentences} sentences from {len(pages)} pages")

    if total_sentences == 0:
        log.warning("No text extracted from PDF. The file may be image-only.")
        sys.exit(1)

    # Classify all sentences
    all_sentences = []
    sentence_page_map = []  # track which page each sentence belongs to
    for page_data in pages:
        for sent in page_data["sentences"]:
            all_sentences.append(sent)
            sentence_page_map.append(page_data["page_num"])

    log.info(f"Running inference on {len(all_sentences)} sentences...")
    results = classify_sentences(all_sentences, tokenizer, model, device, threshold=args.threshold)

    # Group results back by page
    pages_results = []
    for page_data in pages:
        page_num = page_data["page_num"]
        page_results = []
        for r, pg in zip(results, sentence_page_map):
            if pg == page_num:
                page_results.append(r)
        pages_results.append({"page_num": page_num, "results": page_results})
        # Remove processed results from front
        results = results[len(page_results):]
        sentence_page_map = sentence_page_map[len(page_results):]

    # Compute score
    all_results = [r for pr in pages_results for r in pr["results"]]
    score = compute_ai_score(all_results)

    # Print summary
    log.info(f"\n{'='*55}")
    log.info(f"AI DETECTION RESULTS: {pdf_path.name}")
    log.info(f"{'='*55}")
    log.info(f"  Overall AI Score : {score['overall_ai_percent']}%")
    log.info(f"  AI Generated     : {score['ai_generated_percent']}% ({score['ai_generated_sentences']} sentences)")
    log.info(f"  AI Paraphrased   : {score['ai_paraphrased_percent']}% ({score['ai_paraphrased_sentences']} sentences)")
    log.info(f"  Human Written    : {score['human_percent']}% ({score['human_sentences']} sentences)")
    log.info(f"  Total Sentences  : {score['total_sentences']}")

    # Generate reports
    stem = pdf_path.stem
    html_path = output_dir / f"{stem}_report.html"
    json_path = output_dir / f"{stem}_report.json"

    pdf_report_path = output_dir / f"{stem}_report.pdf"

    generate_html_report(pdf_path.name, pages_results, score, html_path)
    generate_pdf_report(str(pdf_path), pages_results, score, pdf_report_path)
    save_json_report(pdf_path.name, pages_results, score, json_path)

    log.info(f"\nDone!")
    log.info(f"  HTML report : {html_path}")
    log.info(f"  PDF report  : {pdf_report_path}")
    log.info(f"  JSON report : {json_path}")


if __name__ == "__main__":
    main()
