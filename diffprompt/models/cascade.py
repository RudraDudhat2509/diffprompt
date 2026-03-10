"""
Cascade model client.
Tries Ollama first (local, free), falls back to Groq, escalates to Claude Haiku.
"""
from __future__ import annotations
import httpx
import os
from typing import Optional


OLLAMA_BASE = "http://localhost:11434"
GROQ_BASE   = "https://api.groq.com/openai/v1"


async def call_ollama(model: str, prompt: str, system: Optional[str] = None) -> Optional[str]:
    """Call a local Ollama model. Returns None if Ollama is not running."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": model, "messages": messages, "stream": False}
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
    except Exception:
        return None


async def call_groq(model: str, prompt: str, system: Optional[str] = None) -> Optional[str]:
    """Call Groq API. Returns None if API key missing or call fails."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{GROQ_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "max_tokens": 2000}
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


async def call_cascade(
    prompt: str,
    system: Optional[str] = None,
    local_model: str = "qwen2.5:7b",
    groq_model: str = "llama-3.3-70b-versatile",
    local_only: bool = False,
) -> tuple[str, str]:
    """
    Try models in order: Ollama → Groq → error.
    Returns (output, model_used).
    """
    # Try local first
    result = await call_ollama(local_model, prompt, system)
    if result:
        return result, f"local/{local_model}"

    if local_only:
        raise RuntimeError("Ollama unavailable and --local-only is set. Is Ollama running?")

    # Fall back to Groq
    result = await call_groq(groq_model, prompt, system)
    if result:
        return result, f"groq/{groq_model}"

    raise RuntimeError(
        "All models failed. Check: is Ollama running? Is GROQ_API_KEY set?"
    )
