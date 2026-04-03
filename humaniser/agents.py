"""
Multi-Agent Humanisation Pipeline
===================================
Agents work in sequence orchestrated by HumaniserOrchestrator:

  Agent 1 — AnalyserAgent    : Detects AI-flagged sentences via RoBERTa
  Agent 2 — StrategistAgent  : Plans a per-sentence humanisation strategy
                                based on known AI writing patterns
  Agent 3 — RewriterAgent    : Calls an LLM to rewrite with the strategy
  Agent 4 — VerifierAgent    : Re-runs detector; marks sentences that pass
  Orchestrator               : Loops Strategist→Rewriter→Verifier until
                                all sentences pass or max iterations hit.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .llm_backends import BaseLLMBackend, LLMConfig, create_backend

log = logging.getLogger(__name__)

# ─── Labels ──────────────────────────────────────────────────────────────────

LABEL2ID = {"human": 0, "ai_generated": 1, "ai_paraphrased": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ─── AI writing pattern fingerprints (from data analysis) ────────────────────

AI_PATTERNS = {
    "overused_openers": [
        r"^(the proposed|this paper|in this paper|in this work|this study)",
        r"^(to (comprehensively|further|effectively|systematically|rigorously))",
        r"^(furthermore|moreover|notably|importantly|in addition)",
        r"^(the results (demonstrate|show|indicate|reveal))",
        r"^(an ablation study)",
    ],
    "filler_phrases": [
        "state-of-the-art", "robust framework", "novel approach", "significant improvement",
        "leverages", "utilizes", "exhibits", "demonstrates superior", "outperforms",
        "to address this", "to tackle this", "to mitigate this challenge",
        "in the realm of", "in the domain of", "plays a crucial role",
        "it is worth noting", "it should be noted", "as can be seen",
    ],
    "passive_constructions": [
        r"\bis (proposed|presented|introduced|described|utilized|employed)\b",
        r"\bwere (obtained|achieved|conducted|performed|evaluated)\b",
        r"\bhas been (proposed|shown|demonstrated|reported)\b",
    ],
    "hyperbolic_adjectives": [
        "comprehensive", "sophisticated", "groundbreaking", "cutting-edge",
        "unprecedented", "synergistic", "holistic", "multifaceted",
    ],
}


def detect_ai_patterns(text: str) -> list[str]:
    """Return list of pattern categories detected in text."""
    found = []
    t_lower = text.lower()

    for pattern in AI_PATTERNS["overused_openers"]:
        if re.search(pattern, t_lower):
            found.append("overused_opener")
            break

    for phrase in AI_PATTERNS["filler_phrases"]:
        if phrase in t_lower:
            found.append("filler_phrase")
            break

    for pattern in AI_PATTERNS["passive_constructions"]:
        if re.search(pattern, t_lower):
            found.append("passive_voice")
            break

    for word in AI_PATTERNS["hyperbolic_adjectives"]:
        if word in t_lower:
            found.append("hyperbolic_language")
            break

    return list(set(found))


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Sentence:
    text: str
    label: str = "human"
    confidence: float = 0.0
    ai_patterns: list[str] = field(default_factory=list)
    strategy: str = ""
    rewritten: Optional[str] = None
    passed: bool = False
    iterations: int = 0


@dataclass
class HumanisationResult:
    original_text: str
    humanised_text: str
    sentences: list[Sentence]
    original_ai_score: float
    final_ai_score: float
    total_iterations: int
    passed: bool


# ─── Agent 1: Analyser ───────────────────────────────────────────────────────

class AnalyserAgent:
    """
    Loads the trained RoBERTa model and classifies each sentence.
    Returns list of Sentence objects with label + confidence.
    """

    def __init__(
        self,
        model_dir: str | Path = "models/roberta_ai_detector",
        device: Optional[torch.device] = None,
        threshold: float = 0.65,
        max_length: int = 256,
        batch_size: int = 32,
    ):
        self.threshold  = threshold
        self.max_length = max_length
        self.batch_size = batch_size
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"[Analyser] Loading model from {model_dir} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    def analyse(self, sentences: list[str]) -> list[Sentence]:
        results = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for text, prob in zip(batch, probs):
                human_p = prob[LABEL2ID["human"]]
                non_h   = 1.0 - human_p
                if non_h >= self.threshold:
                    pred = ID2LABEL[int(np.argmax(prob[1:])) + 1]
                else:
                    pred = "human"

                s = Sentence(
                    text       = text,
                    label      = pred,
                    confidence = float(np.max(prob)),
                    ai_patterns= detect_ai_patterns(text),
                    passed     = pred == "human",
                )
                results.append(s)
        return results

    def ai_score(self, sentences: list[Sentence]) -> float:
        """Return % of characters that are AI-flagged."""
        total = sum(len(s.text) for s in sentences)
        ai    = sum(len(s.text) for s in sentences if s.label != "human")
        return round(ai / max(total, 1) * 100, 1)


# ─── Agent 2: Strategist ─────────────────────────────────────────────────────

class StrategistAgent:
    """
    Analyses each AI-flagged sentence's patterns and assigns a targeted
    humanisation strategy string that the RewriterAgent will use as guidance.
    """

    STRATEGY_MAP = {
        "overused_opener": (
            "Rewrite using a direct, active-voice opening. "
            "Avoid starting with 'The proposed', 'This paper', 'In this work'. "
            "Start with a specific observation, data point, or concrete subject."
        ),
        "passive_voice": (
            "Convert passive constructions to active voice. "
            "Make the actor/subject explicit. "
            "Use present tense where appropriate."
        ),
        "filler_phrase": (
            "Remove vague filler phrases like 'state-of-the-art', 'robust framework', "
            "'plays a crucial role'. Replace with specific, measurable descriptions."
        ),
        "hyperbolic_language": (
            "Tone down hyperbolic adjectives (comprehensive, sophisticated, groundbreaking). "
            "Use precise, neutral, specific language instead."
        ),
    }

    DEFAULT_STRATEGY = (
        "Rephrase this sentence in a natural, conversational academic style. "
        "Vary sentence structure, use concrete examples or specifics, "
        "and avoid typical AI writing patterns."
    )

    def plan(self, sentence: Sentence) -> str:
        if not sentence.ai_patterns:
            return self.DEFAULT_STRATEGY

        strategies = [self.STRATEGY_MAP[p] for p in sentence.ai_patterns if p in self.STRATEGY_MAP]
        if not strategies:
            return self.DEFAULT_STRATEGY

        base = " ".join(strategies)
        # Augment with confidence-based intensity
        if sentence.confidence > 0.9:
            base += " Significantly restructure the sentence — change word order, split or merge clauses."
        elif sentence.confidence > 0.75:
            base += " Moderately rephrase — change at least 60% of the vocabulary while preserving meaning."
        else:
            base += " Light rephrase — vary a few key words and the sentence rhythm."

        return base


# ─── Agent 3: Rewriter ───────────────────────────────────────────────────────

REWRITER_SYSTEM = """You are an expert academic writing editor. Your task is to rewrite AI-generated academic text so it reads as naturally human-written, while fully preserving the original technical meaning, accuracy, and quality.

Rules:
- Keep all technical terms, numbers, citations, and domain-specific vocabulary exactly as-is
- Do NOT add new information or change the meaning
- Do NOT make it less technical or dumb it down
- Vary sentence structure, rhythm, and vocabulary
- Use active voice where natural
- Avoid: "Furthermore", "Moreover", "Notably", "It is worth noting", "This paper proposes"
- Output ONLY the rewritten sentence. No explanations, no quotes, no prefixes."""


class RewriterAgent:
    """
    Calls the LLM backend to rewrite a sentence using the strategy from StrategistAgent.
    """

    def __init__(self, backend: BaseLLMBackend):
        self.backend = backend

    def rewrite(self, sentence: Sentence) -> str:
        prompt = f"""Strategy: {sentence.strategy}

Original sentence:
\"\"\"{sentence.text}\"\"\"

Rewrite this sentence following the strategy above. Output ONLY the rewritten sentence."""

        try:
            result = self.backend.complete(prompt, system=REWRITER_SYSTEM)
            # Strip any accidental quotes wrapping the output
            result = result.strip().strip('"').strip("'").strip()
            return result
        except Exception as e:
            log.warning(f"[Rewriter] LLM call failed: {e}")
            return sentence.text  # fallback: keep original


# ─── Agent 4: Verifier ───────────────────────────────────────────────────────

class VerifierAgent:
    """
    Re-runs the detector on rewritten sentences.
    Updates sentence.label, sentence.confidence, sentence.passed.
    """

    def __init__(self, analyser: AnalyserAgent):
        self.analyser = analyser

    def verify(self, sentences: list[Sentence]) -> list[Sentence]:
        # Only re-check sentences that haven't passed yet
        pending = [(i, s) for i, s in enumerate(sentences) if not s.passed]
        if not pending:
            return sentences

        texts   = [s.rewritten or s.text for _, s in pending]
        checked = self.analyser.analyse(texts)

        for (i, orig_s), checked_s in zip(pending, checked):
            sentences[i].label      = checked_s.label
            sentences[i].confidence = checked_s.confidence
            sentences[i].ai_patterns= checked_s.ai_patterns
            sentences[i].passed     = checked_s.passed
            if checked_s.passed and orig_s.rewritten:
                sentences[i].text = orig_s.rewritten  # promote rewrite as new text

        return sentences


# ─── Orchestrator ────────────────────────────────────────────────────────────

class HumaniserOrchestrator:
    """
    Multi-agent loop:
      1. Analyser  → detect all AI sentences
      2. Strategist→ plan per-sentence strategy
      3. Rewriter  → rewrite with LLM
      4. Verifier  → check again
      Repeat 2-4 until all pass or max_iterations reached.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        model_dir: str | Path = "models/roberta_ai_detector",
        detector_threshold: float = 0.65,
        max_iterations: int = 3,
        target_ai_score: float = 10.0,
    ):
        self.max_iterations    = max_iterations
        self.target_ai_score   = target_ai_score

        # Initialise all agents
        self.analyser    = AnalyserAgent(model_dir=model_dir, threshold=detector_threshold)
        self.strategist  = StrategistAgent()
        self.backend     = create_backend(llm_config)
        self.rewriter    = RewriterAgent(self.backend)
        self.verifier    = VerifierAgent(self.analyser)

        log.info(
            f"[Orchestrator] Ready | LLM: {llm_config.provider}/{llm_config.resolved_model()} | "
            f"max_iter: {max_iterations} | target AI%: {target_ai_score}"
        )

    def run(self, text: str) -> HumanisationResult:
        """
        Run the full multi-agent pipeline on the input text.
        Returns HumanisationResult with humanised text + metadata.
        """
        # Split text into sentences
        raw_sentences = _split_sentences(text)
        log.info(f"[Orchestrator] Input: {len(raw_sentences)} sentences")

        # Agent 1: Initial analysis
        sentences = self.analyser.analyse(raw_sentences)
        original_ai_score = self.analyser.ai_score(sentences)
        ai_count = sum(1 for s in sentences if not s.passed)
        log.info(f"[Analyser]    AI score: {original_ai_score}% | AI sentences: {ai_count}")

        total_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            pending = [s for s in sentences if not s.passed]
            if not pending:
                log.info(f"[Orchestrator] All sentences passed after iteration {iteration - 1}")
                break

            current_score = self.analyser.ai_score(sentences)
            if current_score <= self.target_ai_score:
                log.info(f"[Orchestrator] AI score {current_score}% ≤ target {self.target_ai_score}% — done")
                break

            log.info(f"[Orchestrator] Iteration {iteration} | AI score: {current_score}% | Pending: {len(pending)}")
            total_iterations = iteration

            # Agent 2: Plan strategies
            for s in pending:
                s.strategy = self.strategist.plan(s)

            # Agent 3: Rewrite
            for s in pending:
                rewritten = self.rewriter.rewrite(s)
                s.rewritten = rewritten
                s.iterations += 1
                log.debug(f"  [Rewriter] '{s.text[:60]}...' → '{rewritten[:60]}...'")

            # Agent 4: Verify
            sentences = self.verifier.verify(sentences)

            passed_now = sum(1 for s in sentences if s.passed)
            log.info(f"[Verifier]    Passed: {passed_now}/{len(sentences)}")

        # Compose final text: use rewritten if available, else original
        final_parts = []
        for s in sentences:
            text_to_use = s.rewritten if (s.rewritten and s.passed) else s.text
            final_parts.append(text_to_use)

        humanised_text = " ".join(final_parts)
        final_ai_score = self.analyser.ai_score(sentences)
        all_passed = all(s.passed for s in sentences)

        log.info(
            f"[Orchestrator] Done | Original AI: {original_ai_score}% → Final AI: {final_ai_score}% | "
            f"Passed: {sum(s.passed for s in sentences)}/{len(sentences)}"
        )

        return HumanisationResult(
            original_text    = text,
            humanised_text   = humanised_text,
            sentences        = sentences,
            original_ai_score= original_ai_score,
            final_ai_score   = final_ai_score,
            total_iterations = total_iterations,
            passed           = all_passed,
        )


# ─── Sentence splitter ───────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping them at a reasonable length."""
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+', text)
    result = []
    for p in parts:
        p = p.strip()
        if len(p) >= 15:
            result.append(p)
        elif result:
            result[-1] += " " + p
    return result
