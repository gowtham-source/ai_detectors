"""
LLM Backend Registry
====================
Pluggable LLM adapters. Each backend exposes a single method:
    complete(prompt: str, system: str) -> str

Supported backends (set via LLMConfig.provider):
  - "openai"   : OpenAI GPT-4o / GPT-4-turbo / GPT-3.5-turbo
  - "gemini"   : Google Gemini 1.5 Pro / Flash
  - "claude"   : Anthropic Claude 3 (Sonnet / Haiku / Opus)
  - "ollama"   : Local Ollama (llama3, mistral, phi3, etc.)
  - "groq"     : Groq API (llama3-70b, mixtral, gemma2)

Configure via environment variables or pass LLMConfig directly.
"""

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "openai"            # openai | gemini | claude | ollama | groq
    model: Optional[str] = None         # auto-selected per provider if None
    api_key: Optional[str] = None       # reads from env if None
    base_url: Optional[str] = None      # for ollama or custom endpoints
    temperature: float = 0.85           # higher = more human-like variation
    max_tokens: int = 1024
    timeout: int = 60

    # Defaults per provider
    _DEFAULTS: dict = field(default_factory=lambda: {
        "openai":  {"model": "gpt-4o-mini",       "env": "OPENAI_API_KEY"},
        "gemini":  {"model": "gemini-1.5-flash",  "env": "GEMINI_API_KEY"},
        "claude":  {"model": "claude-3-haiku-20240307", "env": "ANTHROPIC_API_KEY"},
        "ollama":  {"model": "llama3",             "env": None,  "base_url": "http://localhost:11434"},
        "groq":    {"model": "llama3-70b-8192",   "env": "GROQ_API_KEY"},
    }, repr=False)

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return self._DEFAULTS.get(self.provider, {}).get("model", "")

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        env_var = self._DEFAULTS.get(self.provider, {}).get("env")
        if env_var:
            return os.environ.get(env_var)
        return None

    def resolved_base_url(self) -> Optional[str]:
        if self.base_url:
            return self.base_url
        return self._DEFAULTS.get(self.provider, {}).get("base_url")


# ─── Base class ──────────────────────────────────────────────────────────────

class BaseLLMBackend:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError


# ─── OpenAI ──────────────────────────────────────────────────────────────────

class OpenAIBackend(BaseLLMBackend):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: uv add openai")
        self._client = OpenAI(
            api_key=config.resolved_api_key(),
            base_url=config.resolved_base_url(),
            timeout=config.timeout,
        )

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.config.resolved_model(),
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()


# ─── Google Gemini ───────────────────────────────────────────────────────────

class GeminiBackend(BaseLLMBackend):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: uv add google-generativeai")
        genai.configure(api_key=config.resolved_api_key())
        gen_config = genai.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        self._model = genai.GenerativeModel(
            model_name=config.resolved_model(),
            generation_config=gen_config,
        )

    def complete(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        response = self._model.generate_content(full_prompt)
        return response.text.strip()


# ─── Anthropic Claude ────────────────────────────────────────────────────────

class ClaudeBackend(BaseLLMBackend):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: uv add anthropic")
        self._client = anthropic.Anthropic(
            api_key=config.resolved_api_key(),
            timeout=config.timeout,
        )

    def complete(self, prompt: str, system: str = "") -> str:
        kwargs = {
            "model": self.config.resolved_model(),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text.strip()


# ─── Ollama (local) ──────────────────────────────────────────────────────────

class OllamaBackend(BaseLLMBackend):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._base_url = config.resolved_base_url() or "http://localhost:11434"

    def complete(self, prompt: str, system: str = "") -> str:
        import urllib.request
        payload = {
            "model": self.config.resolved_model(),
            "prompt": f"{system}\n\n{prompt}".strip() if system else prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as r:
            result = json.loads(r.read())
        return result.get("response", "").strip()


# ─── Groq ────────────────────────────────────────────────────────────────────

class GroqBackend(BaseLLMBackend):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: uv add groq")
        self._client = Groq(
            api_key=config.resolved_api_key(),
            timeout=config.timeout,
        )

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.config.resolved_model(),
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()


# ─── Factory ─────────────────────────────────────────────────────────────────

_BACKEND_MAP = {
    "openai": OpenAIBackend,
    "gemini": GeminiBackend,
    "claude": ClaudeBackend,
    "ollama": OllamaBackend,
    "groq":   GroqBackend,
}


def create_backend(config: LLMConfig) -> BaseLLMBackend:
    """Factory: return the correct backend for the given config."""
    provider = config.provider.lower()
    cls = _BACKEND_MAP.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_BACKEND_MAP)}")
    log.info(f"LLM backend: {provider} / {config.resolved_model()}")
    return cls(config)
