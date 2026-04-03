"""
Humaniser Pipeline - Programmatic API
======================================
Use this module to call the humaniser from other Python scripts.

Example:
    from humaniser.pipeline import humanise_text, humanise_pdf

    result = humanise_text(
        text="The proposed framework leverages state-of-the-art...",
        provider="gemini",
        api_key="YOUR_KEY",
    )
    print(result.humanised_text)
    print(f"AI score: {result.original_ai_score}% → {result.final_ai_score}%")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .agents import HumaniserOrchestrator, HumanisationResult
from .llm_backends import LLMConfig


def humanise_text(
    text: str,
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.85,
    model_dir: str | Path = "models/roberta_ai_detector",
    detector_threshold: float = 0.65,
    max_iterations: int = 3,
    target_ai_score: float = 10.0,
) -> HumanisationResult:
    """
    Humanise a raw text string.

    Args:
        text:               Input text (any length)
        provider:           LLM provider: "openai" | "gemini" | "claude" | "ollama" | "groq"
        model:              Model name override (uses provider default if None)
        api_key:            API key (reads from env if None)
        base_url:           Custom base URL (for Ollama or proxied endpoints)
        temperature:        LLM temperature for rewriting (0.7–1.0 recommended)
        model_dir:          Path to trained RoBERTa detector checkpoint
        detector_threshold: Confidence threshold for AI detection (0.0–1.0)
        max_iterations:     Max rewrite loops per sentence
        target_ai_score:    Stop when overall AI score drops below this %

    Returns:
        HumanisationResult with .humanised_text, .original_ai_score, .final_ai_score, etc.
    """
    config = LLMConfig(
        provider    = provider,
        model       = model,
        api_key     = api_key,
        base_url    = base_url,
        temperature = temperature,
        max_tokens  = 512,
    )
    orchestrator = HumaniserOrchestrator(
        llm_config          = config,
        model_dir           = model_dir,
        detector_threshold  = detector_threshold,
        max_iterations      = max_iterations,
        target_ai_score     = target_ai_score,
    )
    return orchestrator.run(text)


def humanise_pdf(
    pdf_path: str | Path,
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.85,
    model_dir: str | Path = "models/roberta_ai_detector",
    detector_threshold: float = 0.65,
    max_iterations: int = 3,
    target_ai_score: float = 10.0,
) -> HumanisationResult:
    """
    Extract text from a PDF and humanise it.
    Same arguments as humanise_text(), plus pdf_path.
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    for pg in range(len(doc)):
        t = doc[pg].get_text("text").strip()
        if len(t) > 20:
            pages.append(t)
    doc.close()
    full_text = "\n\n".join(pages)
    return humanise_text(
        text                = full_text,
        provider            = provider,
        model               = model,
        api_key             = api_key,
        base_url            = base_url,
        temperature         = temperature,
        model_dir           = model_dir,
        detector_threshold  = detector_threshold,
        max_iterations      = max_iterations,
        target_ai_score     = target_ai_score,
    )
