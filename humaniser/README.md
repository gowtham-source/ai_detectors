# AI Humaniser — Multi-Agent Paraphrasing System

Removes AI detection flags from academic text while preserving full technical quality and meaning.

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│  Agent 1: Analyser  │  ← RoBERTa detector: classifies each sentence
└────────┬────────────┘
         │ AI-flagged sentences
         ▼
┌─────────────────────┐
│ Agent 2: Strategist │  ← Pattern analysis → per-sentence strategy
└────────┬────────────┘      (passive voice, filler phrases, openers, hyperbole)
         │
         ▼
┌─────────────────────┐
│  Agent 3: Rewriter  │  ← Calls LLM with targeted strategy prompt
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 4: Verifier  │  ← Re-runs detector on rewritten text
└────────┬────────────┘
         │ still flagged? → loop back (max N iterations)
         ▼
    Final Text
```

## Supported LLM Backends

| Provider   | `--provider` | Default Model           | Install                          |
|------------|-------------|--------------------------|----------------------------------|
| OpenAI     | `openai`    | `gpt-4o-mini`            | `uv add openai`                  |
| Google     | `gemini`    | `gemini-1.5-flash`       | `uv add google-generativeai`     |
| Anthropic  | `claude`    | `claude-3-haiku-20240307`| `uv add anthropic`               |
| Groq       | `groq`      | `llama3-70b-8192`        | `uv add groq`                    |
| Ollama     | `ollama`    | `llama3` (local)         | *(no install, needs Ollama running)* |

## Setup

```bash
# Install for your preferred backend (pick one or more)
uv add google-generativeai     # Gemini
uv add openai                  # OpenAI
uv add anthropic               # Claude
uv add groq                    # Groq (free tier available)
# OR install all at once:
uv add "ai-plag-detector[all-llm]"
```

Set your API key as an environment variable:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-key"
$env:OPENAI_API_KEY = "your-key"
$env:ANTHROPIC_API_KEY = "your-key"
$env:GROQ_API_KEY = "your-key"
```

## CLI Usage

```bash
# Gemini (default, free tier available)
uv run humaniser/humanise.py --input paper.txt --provider gemini

# OpenAI GPT-4o
uv run humaniser/humanise.py --input paper.txt --provider openai --model gpt-4o

# Claude Sonnet
uv run humaniser/humanise.py --input paper.txt --provider claude --model claude-3-5-sonnet-20241022

# Groq (fast + free tier)
uv run humaniser/humanise.py --input paper.txt --provider groq

# Local Ollama (no API key needed)
uv run humaniser/humanise.py --input paper.txt --provider ollama --model llama3

# PDF input directly
uv run humaniser/humanise.py --input paper.pdf --provider gemini

# Custom options
uv run humaniser/humanise.py \
  --input paper.txt \
  --provider gemini \
  --max-iter 4 \
  --target 5.0 \
  --threshold 0.60 \
  --temperature 0.9
```

## Python API

```python
from humaniser.pipeline import humanise_text, humanise_pdf

# From text
result = humanise_text(
    text="The proposed framework leverages state-of-the-art deep learning...",
    provider="gemini",
    api_key="YOUR_KEY",          # or set GEMINI_API_KEY env var
    max_iterations=3,
    target_ai_score=10.0,
)
print(result.humanised_text)
print(f"AI score: {result.original_ai_score}% → {result.final_ai_score}%")

# From PDF
result = humanise_pdf("paper.pdf", provider="gemini")
```

## Options

| Flag             | Default                          | Description                                      |
|------------------|----------------------------------|--------------------------------------------------|
| `--input`        | *(required)*                     | `.txt` or `.pdf` input file                      |
| `--provider`     | `gemini`                         | LLM provider                                     |
| `--model`        | provider default                 | Model name override                              |
| `--api-key`      | from env var                     | API key                                          |
| `--max-iter`     | `3`                              | Max rewrite loops per sentence                   |
| `--target`       | `10.0`                           | Stop when AI score drops below this %            |
| `--threshold`    | `0.65`                           | Detector confidence threshold                    |
| `--temperature`  | `0.85`                           | LLM temperature (higher = more variation)        |
| `--model-dir`    | `models/roberta_ai_detector`     | Trained RoBERTa detector path                    |
| `--output-dir`   | `reports/humanised`              | Output directory                                 |

## Output

- `reports/humanised/<stem>_humanised.txt` — final humanised text
- `reports/humanised/<stem>_humanised_report.json` — per-sentence details:
  - original text, rewritten text, label, confidence, AI patterns detected, strategy used

## AI Pattern Detection

The Strategist agent detects these patterns (from corpus analysis of 1,045 AI-generated segments):

| Pattern              | Examples                                                    |
|----------------------|-------------------------------------------------------------|
| `overused_opener`    | "The proposed...", "This paper...", "In this work..."       |
| `filler_phrase`      | "state-of-the-art", "robust framework", "plays a crucial role" |
| `passive_voice`      | "is proposed", "were obtained", "has been demonstrated"    |
| `hyperbolic_language`| "comprehensive", "sophisticated", "groundbreaking"          |
