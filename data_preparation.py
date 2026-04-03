"""
AI Plagiarism Detector - Data Preparation Pipeline
====================================================
Processes Turnitin AI-detection PDFs from raw_data folder.

Steps:
  1. Parse submission date from page 1 → assign priority
       HIGH  : Submission date >= Jan 15, 2026  (after model retrain)
       LOW   : Submission date <  Jan 15, 2026  (older model)
  2. Skip pages 1-2 (Turnitin cover + AI overview)
  3. From page 3+: extract text segments tagged as:
       - ai_generated   : text under CYAN highlight  (#51c6da) — Turnitin AI flag
       - ai_paraphrased : text under PURPLE highlight (#b68bfb) — AI + paraphrase tool
       - human          : unhighlighted text
     Formulas/equations inside a highlight rect inherit that highlight's label.
     Formulas outside any highlight are labelled human.
  4. If a page has no text layer → fallback to GLM-OCR (zai-org/GLM-OCR)
  5. Save structured JSON output per file + combined dataset JSONL

Output:
  data/processed/
    <file_stem>.json        per-file structured result
    dataset.jsonl           combined training dataset
    dataset_summary.json    statistics
"""

import json
import re
import sys
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

RAW_DATA_DIR = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/raw_data/raw_data")
OUTPUT_DIR   = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed")

# Priority cutoff: submissions on/after this date → HIGH priority
PRIORITY_CUTOFF = date(2026, 1, 15)

# Turnitin highlight colors in RGB float (0-1)
# Cyan  #51c6da → AI-generated only
TURNITIN_AI_COLOR         = (0.3203125, 0.77734375, 0.85546875)
# Purple #b68bfb → AI-generated + AI-paraphrased
TURNITIN_PARAPHRASE_COLOR = (0.7148,    0.5469,     0.9844)
COLOR_TOLERANCE = 0.04  # tolerance for color matching

# Minimum text length to keep a segment (filter noise)
MIN_SEGMENT_CHARS = 10

# Turnitin page footer pattern to strip
TURNITIN_FOOTER_RE = re.compile(
    r"Page\s+\d+\s+of\s+\d+\s+-\s+AI\s+Writing\s+Submission\s+Submission\s+ID\s+[\w:]+",
    re.IGNORECASE
)

# Labels
LABEL_AI_GENERATED   = "ai_generated"
LABEL_AI_PARAPHRASED = "ai_paraphrased"
LABEL_HUMAN          = "human"

# Month name → number mapping
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

# ─── GLM-OCR lazy loader ──────────────────────────────────────────────────────

_glm_model = None
_glm_processor = None


def get_glm_ocr():
    """Lazy-load GLM-OCR model only when needed (fallback for image-only pages)."""
    global _glm_model, _glm_processor
    if _glm_model is None:
        log.info("Loading GLM-OCR model (zai-org/GLM-OCR) - first use only...")
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            MODEL_ID = "zai-org/GLM-OCR"
            _glm_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            _glm_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            log.info("GLM-OCR loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load GLM-OCR: {e}")
            return None, None
    return _glm_model, _glm_processor


def ocr_page_with_glm(page: fitz.Page) -> str:
    """Render page to image and run GLM-OCR on it."""
    model, processor = get_glm_ocr()
    if model is None:
        return ""
    try:
        from PIL import Image
        import io, torch

        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024)
        text = processor.decode(outputs[0], skip_special_tokens=True)
        return text
    except Exception as e:
        log.warning(f"GLM-OCR failed on page: {e}")
        return ""


# ─── Utility functions ────────────────────────────────────────────────────────

def color_label(fill_color, tol=COLOR_TOLERANCE) -> Optional[str]:
    """
    Map a PDF fill color to a Turnitin highlight label.
    Returns 'ai_generated', 'ai_paraphrased', or None (not a Turnitin highlight).
    """
    if not fill_color or len(fill_color) != 3:
        return None
    if all(abs(fill_color[i] - TURNITIN_AI_COLOR[i]) < tol for i in range(3)):
        return LABEL_AI_GENERATED
    if all(abs(fill_color[i] - TURNITIN_PARAPHRASE_COLOR[i]) < tol for i in range(3)):
        return LABEL_AI_PARAPHRASED
    return None


def clean_text(text: str) -> str:
    """Remove Turnitin footer watermarks and normalize whitespace."""
    text = TURNITIN_FOOTER_RE.sub("", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


def parse_submission_date(text: str) -> Optional[date]:
    """
    Extract submission date from Turnitin cover page text.
    Handles formats like:
      - "Jan 8, 2026, 9:59 AM GMT+6"
      - "December 11, 2025, 5:48 PM GMT+2"
    """
    pattern = r"(?:Submission\s+Date[^\n]*\n)?\s*([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})"
    matches = re.findall(pattern, text, re.IGNORECASE)
    for month_str, day_str, year_str in matches:
        month = MONTH_MAP.get(month_str.lower())
        if month:
            try:
                return date(int(year_str), month, int(day_str))
            except ValueError:
                continue
    return None


def assign_priority(submission_date: Optional[date]) -> str:
    """
    HIGH  : submission >= Jan 15 2026 (post-retrain, accurate model)
    LOW   : submission <  Jan 15 2026 (older model, less accurate)
    UNKNOWN: no date found
    """
    if submission_date is None:
        return "UNKNOWN"
    return "HIGH" if submission_date >= PRIORITY_CUTOFF else "LOW"


# ─── Core per-page extraction ─────────────────────────────────────────────────

def extract_highlight_rects(page: fitz.Page) -> list[tuple]:
    """
    Get all Turnitin-highlight rects from a page's drawings.
    Returns list of (fitz.Rect, label) tuples.
    """
    rects = []
    for d in page.get_drawings():
        lbl = color_label(d.get("fill"))
        if lbl is not None:
            rects.append((fitz.Rect(d["rect"]), lbl))
    return rects


def word_highlight_label(word_rect: fitz.Rect, highlight_rects: list) -> Optional[str]:
    """
    Return the highlight label for a word rect, or None if not highlighted.
    ai_paraphrased takes precedence over ai_generated if both overlap.
    """
    matched = None
    for hr, lbl in highlight_rects:
        if word_rect.intersects(hr):
            if lbl == LABEL_AI_PARAPHRASED:
                return LABEL_AI_PARAPHRASED  # highest priority
            matched = lbl
    return matched


def extract_page_segments(page: fitz.Page, page_num: int) -> list[dict]:
    """
    Extract labeled text segments from a single page.
    Labels: ai_generated | ai_paraphrased | human
    Formulas inside a highlight inherit that highlight's label.
    Formulas outside any highlight → human.
    """
    # Check if page has a usable text layer
    raw_text = page.get_text("text").strip()
    if len(raw_text) < 20:
        log.info(f"  Page {page_num+1}: sparse text ({len(raw_text)} chars) → GLM-OCR fallback")
        ocr_text = ocr_page_with_glm(page)
        if ocr_text:
            return [{
                "text": clean_text(ocr_text),
                "label": LABEL_HUMAN,  # OCR can't distinguish color; label conservatively
                "page_num": page_num + 1,
                "source": "glm_ocr"
            }]
        return []

    highlight_rects = extract_highlight_rects(page)  # list of (Rect, label)

    # Get word-level data with bounding boxes
    words = page.get_text("words")
    # words: (x0, y0, x1, y1, word_text, block_no, line_no, word_no)

    # Tag each word — highlight label takes priority; no highlight → human
    tagged = []
    for w in words:
        word_text = w[4]
        if not word_text.strip():
            continue
        word_rect = fitz.Rect(w[0], w[1], w[2], w[3])

        hl_label = word_highlight_label(word_rect, highlight_rects) if highlight_rects else None
        label = hl_label if hl_label is not None else LABEL_HUMAN

        tagged.append({
            "text": word_text,
            "label": label,
            "block": w[5],
            "line": w[6],
        })

    # Group consecutive same-label words into segments
    segments = []
    if not tagged:
        return segments

    cur_label = tagged[0]["label"]
    cur_words = [tagged[0]["text"]]

    for tw in tagged[1:]:
        if tw["label"] == cur_label:
            cur_words.append(tw["text"])
        else:
            seg_text = clean_text(" ".join(cur_words))
            if len(seg_text) >= MIN_SEGMENT_CHARS:
                segments.append({
                    "text": seg_text,
                    "label": cur_label,
                    "page_num": page_num + 1,
                    "source": "direct_pdf"
                })
            cur_label = tw["label"]
            cur_words = [tw["text"]]

    # Last group
    seg_text = clean_text(" ".join(cur_words))
    if len(seg_text) >= MIN_SEGMENT_CHARS:
        segments.append({
            "text": seg_text,
            "label": cur_label,
            "page_num": page_num + 1,
            "source": "direct_pdf"
        })

    return segments


# ─── Per-file processing ──────────────────────────────────────────────────────

def process_pdf(pdf_path: Path) -> dict:
    """
    Full processing of a single Turnitin PDF.
    Returns structured result dict.
    """
    log.info(f"Processing: {pdf_path.name}")
    result = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "submission_date": None,
        "priority": "UNKNOWN",
        "total_pages": 0,
        "ai_percentage_reported": None,
        "segments": [],
        "stats": {
            "ai_generated_segments": 0,
            "ai_paraphrased_segments": 0,
            "human_segments": 0,
            "ocr_pages": 0,
            "total_chars_ai": 0,
            "total_chars_paraphrased": 0,
            "total_chars_human": 0,
        },
        "errors": []
    }

    try:
        doc = fitz.open(str(pdf_path))
        result["total_pages"] = len(doc)

        # ── Page 1: Extract submission date ───────────────────────────────────
        if len(doc) >= 1:
            page1_text = doc[0].get_text("text")
            sub_date = parse_submission_date(page1_text)
            result["submission_date"] = sub_date.isoformat() if sub_date else None
            result["priority"] = assign_priority(sub_date)
            log.info(f"  Submission date: {result['submission_date']} → Priority: {result['priority']}")

        # ── Page 2: Extract overall AI % if available ─────────────────────────
        if len(doc) >= 2:
            page2_text = doc[1].get_text("text")
            pct_match = re.search(r"(\d{1,3})%\s+detected\s+as\s+AI", page2_text, re.IGNORECASE)
            if pct_match:
                result["ai_percentage_reported"] = int(pct_match.group(1))
                log.info(f"  Turnitin AI score: {result['ai_percentage_reported']}%")

        # ── Pages 3+: Content extraction ──────────────────────────────────────
        all_segments = []
        for pg in range(2, len(doc)):  # 0-indexed, so page 3 = index 2
            page_segs = extract_page_segments(doc[pg], pg)
            ocr_segs = [s for s in page_segs if s.get("source") == "glm_ocr"]
            if ocr_segs:
                result["stats"]["ocr_pages"] += 1
            all_segments.extend(page_segs)

        doc.close()

        result["segments"] = all_segments

        # ── Compute stats ──────────────────────────────────────────────────────
        for seg in all_segments:
            lbl = seg["label"]
            char_count = len(seg["text"])
            if lbl == LABEL_AI_GENERATED:
                result["stats"]["ai_generated_segments"] += 1
                result["stats"]["total_chars_ai"] += char_count
            elif lbl == LABEL_AI_PARAPHRASED:
                result["stats"]["ai_paraphrased_segments"] += 1
                result["stats"]["total_chars_paraphrased"] += char_count
            elif lbl == LABEL_HUMAN:
                result["stats"]["human_segments"] += 1
                result["stats"]["total_chars_human"] += char_count

        log.info(
            f"  Segments → AI: {result['stats']['ai_generated_segments']}, "
            f"Paraphrased: {result['stats']['ai_paraphrased_segments']}, "
            f"Human: {result['stats']['human_segments']}"
        )

    except Exception as e:
        import traceback
        log.error(f"  ERROR processing {pdf_path.name}: {e}\n{traceback.format_exc()}")
        result["errors"].append(str(e))

    return result


# ─── Training dataset builder ─────────────────────────────────────────────────

def build_training_record(segment: dict, file_meta: dict) -> dict:
    """
    Convert a segment + file metadata into a training sample.
    Format suitable for fine-tuning a text classifier.
    """
    return {
        "text": segment["text"],
        "label": segment["label"],          # ai_generated | human | formula
        "priority": file_meta["priority"],  # HIGH | LOW | UNKNOWN
        "source_file": file_meta["file_name"],
        "page_num": segment["page_num"],
        "submission_date": file_meta["submission_date"],
        "turnitin_ai_score": file_meta["ai_percentage_reported"],
        "extraction_source": segment.get("source", "direct_pdf"),
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = OUTPUT_DIR,
    test_single: bool = False,
    test_file_name: Optional[str] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "dataset.jsonl"
    summary_path = out_dir / "dataset_summary.json"

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        log.error(f"No PDF files found in {raw_dir}")
        return

    if test_single:
        if test_file_name:
            matches = [f for f in pdf_files if test_file_name.lower() in f.name.lower()]
            pdf_files = matches[:1] if matches else pdf_files[:1]
        else:
            pdf_files = pdf_files[:1]
        log.info(f"TEST MODE: processing only → {pdf_files[0].name}")

    log.info(f"Found {len(pdf_files)} PDF files to process")
    log.info(f"Priority cutoff: {PRIORITY_CUTOFF} (>= HIGH, < LOW)")

    summary = {
        "processed_at": datetime.now().isoformat(),
        "total_files": len(pdf_files),
        "priority_cutoff": PRIORITY_CUTOFF.isoformat(),
        "files": [],
        "totals": {
            "high_priority": 0,
            "low_priority": 0,
            "unknown_priority": 0,
            "total_segments": 0,
            "ai_generated_segments": 0,
            "ai_paraphrased_segments": 0,
            "human_segments": 0,
            "ocr_pages": 0,
        }
    }

    training_records_written = 0

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for pdf_path in pdf_files:
            result = process_pdf(pdf_path)

            # Save per-file JSON
            per_file_path = out_dir / f"{pdf_path.stem}.json"
            with open(per_file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Write training records to JSONL
            for seg in result["segments"]:
                record = build_training_record(seg, result)
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                training_records_written += 1

            # Update summary
            pri = result["priority"]
            summary["totals"][f"{pri.lower()}_priority"] = (
                summary["totals"].get(f"{pri.lower()}_priority", 0) + 1
            )
            summary["totals"]["total_segments"] += len(result["segments"])
            for k in ["ai_generated_segments", "ai_paraphrased_segments", "human_segments", "ocr_pages"]:
                summary["totals"][k] += result["stats"][k]

            summary["files"].append({
                "file_name": result["file_name"],
                "submission_date": result["submission_date"],
                "priority": result["priority"],
                "ai_percentage_reported": result["ai_percentage_reported"],
                "total_pages": result["total_pages"],
                "segments": len(result["segments"]),
                "stats": result["stats"][0] if isinstance(result["stats"], tuple) else result["stats"],
                "errors": result["errors"],
            })

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("\n" + "="*60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Files processed  : {len(pdf_files)}")
    log.info(f"  Training records : {training_records_written}")
    log.info(f"  HIGH priority    : {summary['totals']['high_priority']} files")
    log.info(f"  LOW priority     : {summary['totals']['low_priority']} files")
    log.info(f"  UNKNOWN priority : {summary['totals']['unknown_priority']} files")
    log.info(f"  AI segments         : {summary['totals']['ai_generated_segments']}")
    log.info(f"  AI-paraph segments  : {summary['totals']['ai_paraphrased_segments']}")
    log.info(f"  Human segments      : {summary['totals']['human_segments']}")
    log.info(f"  OCR pages           : {summary['totals']['ocr_pages']}")
    log.info(f"\n  Output dir  : {out_dir}")
    log.info(f"  Dataset JSONL : {jsonl_path}")
    log.info(f"  Summary JSON  : {summary_path}")
    log.info("="*60)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Turnitin AI-report data preparation pipeline")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only the first (or specified) PDF file"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific filename (partial match) to test with --test"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all PDF files in the raw data directory"
    )
    args = parser.parse_args()

    if args.all:
        run_pipeline(test_single=False)
    else:
        # Default: test mode with one file
        run_pipeline(
            test_single=True,
            test_file_name=args.file
        )
