"""
AI Humaniser - CLI Entry Point
================================
Reads a text file or raw text, runs the multi-agent humanisation pipeline,
and outputs the humanised text + a detailed report.

Usage:
  uv run humaniser/humanise.py --input "path/to/paper.txt" --provider gemini
  uv run humaniser/humanise.py --input "path/to/paper.txt" --provider openai --model gpt-4o
  uv run humaniser/humanise.py --input "path/to/paper.txt" --provider ollama --model llama3
  uv run humaniser/humanise.py --input "path/to/paper.txt" --provider claude
  uv run humaniser/humanise.py --input "path/to/paper.txt" --provider groq --model llama3-70b-8192

  # Also accepts PDF input directly
  uv run humaniser/humanise.py --input "path/to/paper.pdf" --provider gemini

Environment variables for API keys:
  OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: uv add pymupdf")
    doc = fitz.open(pdf_path)
    pages_text = []
    for pg in range(len(doc)):
        text = doc[pg].get_text("text").strip()
        if len(text) > 20:
            pages_text.append(text)
    doc.close()
    return "\n\n".join(pages_text)


def load_input(path: str) -> str:
    p = Path(path)
    if not p.exists():
        log.error(f"Input file not found: {path}")
        sys.exit(1)
    if p.suffix.lower() == ".pdf":
        log.info("Extracting text from PDF...")
        return extract_text_from_pdf(path)
    return p.read_text(encoding="utf-8")


def save_report(result, output_dir: Path, stem: str):
    """Save humanised text + JSON metadata report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plain text output
    txt_path = output_dir / f"{stem}_humanised.txt"
    txt_path.write_text(result.humanised_text, encoding="utf-8")
    log.info(f"Humanised text saved: {txt_path}")

    # JSON report
    report = {
        "original_ai_score":  result.original_ai_score,
        "final_ai_score":     result.final_ai_score,
        "reduction":          round(result.original_ai_score - result.final_ai_score, 1),
        "total_iterations":   result.total_iterations,
        "all_passed":         result.passed,
        "sentences": [
            {
                "original":     s.text,
                "rewritten":    s.rewritten,
                "label":        s.label,
                "confidence":   round(s.confidence, 4),
                "ai_patterns":  s.ai_patterns,
                "strategy":     s.strategy,
                "passed":       s.passed,
                "iterations":   s.iterations,
            }
            for s in result.sentences
        ],
    }
    json_path = output_dir / f"{stem}_humanised_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Report saved: {json_path}")

    return txt_path, json_path


def print_summary(result):
    """Print a terminal summary table."""
    reduction = result.original_ai_score - result.final_ai_score
    passed_count = sum(1 for s in result.sentences if s.passed)
    total = len(result.sentences)

    print()
    print("=" * 60)
    print("  HUMANISATION RESULTS")
    print("=" * 60)
    print(f"  Original AI Score  : {result.original_ai_score}%")
    print(f"  Final AI Score     : {result.final_ai_score}%")
    print(f"  Reduction          : -{reduction:.1f}%")
    print(f"  Sentences passed   : {passed_count}/{total}")
    print(f"  Iterations used    : {result.total_iterations}")
    print(f"  All passed         : {'YES ✓' if result.passed else 'PARTIAL'}")
    print("=" * 60)
    print()

    # Show before/after for AI-flagged sentences that were rewritten
    rewritten = [(s.text, s.rewritten, s.label, s.passed) for s in result.sentences
                 if s.rewritten and s.rewritten != s.text][:5]
    if rewritten:
        print("Sample rewrites:")
        for orig, new, label, passed in rewritten:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n  [{label.upper()}] {status}")
            print(f"  BEFORE: {orig[:120]}{'...' if len(orig) > 120 else ''}")
            print(f"  AFTER:  {new[:120]}{'...' if len(new) > 120 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Humaniser - Multi-agent pipeline to remove AI detection flags"
    )
    parser.add_argument("--input",     type=str, required=True, help="Input .txt or .pdf file path")
    parser.add_argument("--provider",  type=str, default="gemini",
                        choices=["openai", "gemini", "claude", "ollama", "groq"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model",     type=str, default=None,
                        help="Model name override (e.g. gpt-4o, gemini-1.5-pro, llama3)")
    parser.add_argument("--api-key",   type=str, default=None,
                        help="API key (or set env: OPENAI_API_KEY / GEMINI_API_KEY / etc.)")
    parser.add_argument("--base-url",  type=str, default=None,
                        help="Custom base URL (for ollama or proxied endpoints)")
    parser.add_argument("--max-iter",  type=int, default=3,
                        help="Max humanisation iterations per sentence (default: 3)")
    parser.add_argument("--target",    type=float, default=10.0,
                        help="Target AI score %% to achieve (default: 10.0)")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Detector confidence threshold (default: 0.65)")
    parser.add_argument("--model-dir", type=str, default="models/roberta_ai_detector",
                        help="Path to trained RoBERTa detector model")
    parser.add_argument("--output-dir",type=str, default="reports/humanised",
                        help="Output directory for results (default: reports/humanised)")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="LLM temperature (default: 0.85)")
    args = parser.parse_args()

    # Lazy import here to avoid loading torch just for --help
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from humaniser.agents import HumaniserOrchestrator
    from humaniser.llm_backends import LLMConfig

    # Load input
    input_text = load_input(args.input)
    stem = Path(args.input).stem

    log.info(f"Input loaded: {len(input_text)} chars from {args.input}")

    # Build LLM config
    config = LLMConfig(
        provider    = args.provider,
        model       = args.model,
        api_key     = args.api_key,
        base_url    = args.base_url,
        temperature = args.temperature,
        max_tokens  = 512,
    )

    # Build orchestrator
    orchestrator = HumaniserOrchestrator(
        llm_config          = config,
        model_dir           = args.model_dir,
        detector_threshold  = args.threshold,
        max_iterations      = args.max_iter,
        target_ai_score     = args.target,
    )

    # Run pipeline
    result = orchestrator.run(input_text)

    # Output
    print_summary(result)
    save_report(result, Path(args.output_dir), stem)


if __name__ == "__main__":
    main()
