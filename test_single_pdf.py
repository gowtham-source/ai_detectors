"""
Test script: inspect one PDF file to understand structure before building full pipeline.
Tests direct text extraction via PyMuPDF (fitz) - no OCR needed if text layer exists.
"""
import sys
import re
from pathlib import Path

TEST_FILE = Path(r"C:\Users\gowth\Downloads\ai_plag_detector\data\raw_data\raw_data\4_5.pdf")

def test_pymupdf(pdf_path: Path):
    try:
        import fitz  # PyMuPDF
        print(f"\n{'='*60}")
        print(f"[PyMuPDF] Opening: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        print(f"  Total pages: {len(doc)}")

        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text = page.get_text("text")
            print(f"\n--- PAGE {page_num + 1} (raw text, first 800 chars) ---")
            print(text[:800])
            print(f"  [Char count: {len(text)}]")

            # Also check for colored spans on page 3+
            if page_num >= 2:
                print(f"\n--- PAGE {page_num + 1} COLORED SPANS ---")
                blocks = page.get_text("rawdict")["blocks"]
                colored_segments = []
                for block in blocks:
                    if block.get("type") == 0:  # text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                color = span.get("color", 0)
                                txt = span.get("text", "").strip()
                                if txt and color != 0:
                                    colored_segments.append({
                                        "text": txt,
                                        "color": hex(color)
                                    })
                if colored_segments:
                    print(f"  Found {len(colored_segments)} colored spans (AI-detected):")
                    for seg in colored_segments[:5]:
                        print(f"    color={seg['color']} | {seg['text'][:100]}")
                else:
                    print("  No colored spans found on this page.")
        doc.close()
        return True
    except ImportError:
        print("[PyMuPDF] NOT installed.")
        return False
    except Exception as e:
        print(f"[PyMuPDF] Error: {e}")
        return False


def test_pdfplumber(pdf_path: Path):
    try:
        import pdfplumber
        print(f"\n{'='*60}")
        print(f"[pdfplumber] Opening: {pdf_path.name}")
        with pdfplumber.open(str(pdf_path)) as pdf:
            print(f"  Total pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages[:3]):
                text = page.extract_text() or ""
                print(f"\n--- PAGE {i+1} (first 600 chars) ---")
                print(text[:600])
        return True
    except ImportError:
        print("[pdfplumber] NOT installed.")
        return False
    except Exception as e:
        print(f"[pdfplumber] Error: {e}")
        return False


def extract_submission_date_test(pdf_path: Path):
    """Try to pull submission date from first 2 pages."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        full_text = ""
        for page_num in range(min(2, len(doc))):
            full_text += doc[page_num].get_text("text")
        doc.close()

        # Turnitin format: "Jan 8, 2026, 9:59 AM GMT+6"
        pattern = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),\s+(\d{4})"
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        print(f"\n[Date Extraction] Found dates: {matches}")

        # Submission Date label search
        sub_match = re.search(r"Submission Date[^\n]*\n([^\n]+)", full_text, re.IGNORECASE)
        if sub_match:
            print(f"  Submission Date line: {sub_match.group(1).strip()}")
    except Exception as e:
        print(f"[Date Extraction] Error: {e}")


if __name__ == "__main__":
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        sys.exit(1)

    print(f"Testing with: {TEST_FILE.name}")
    print(f"File size: {TEST_FILE.stat().st_size / 1024:.1f} KB")

    pymupdf_ok = test_pymupdf(TEST_FILE)
    test_pdfplumber(TEST_FILE)
    extract_submission_date_test(TEST_FILE)

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"  PyMuPDF available: {pymupdf_ok}")
    print("  If text is present above -> direct PDF extraction works, no OCR needed.")
    print("  If blank -> need GLM-OCR fallback.")
