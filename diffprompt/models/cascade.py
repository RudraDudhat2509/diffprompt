"""
Cascade model client.
Tries Ollama first (local, free), falls back to Groq.

Key design decisions:
- Exponential backoff on transient failures (not just silently returning None)
- call_groq_only() used by judge escalation so it doesn't re-try Ollama
"""
from __future__ import annotations
import asyncio
import httpx
import os
from typing import Optional


OLLAMA_BASE = "http://localhost:11434"
GROQ_BASE   = "https://api.groq.com/openai/v1"

_MAX_RETRIES = 2
_RETRY_DELAY = 1.0  # seconds


async def call_ollama(
    model: str,
    prompt: str,
    system: Optional[str] = None,
) -> Optional[str]:
    """Call a local Ollama model. Returns None if Ollama is not running."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
        except httpx.ConnectError:
            return None  # Ollama not running — don't retry
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
    return None

async def call_groq(
    model: str,
    prompt: str,
    system: Optional[str] = None,
) -> Optional[str]:
    """Call Groq API with loud error logging and backoff."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n[DEBUG] The code cannot find your GROQ_API_KEY. VS Code is not loading it.")
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    _MAX_RETRIES = 5  # Increased from 2 to handle rate limits better

    for attempt in range(_MAX_RETRIES):
        try:
            # We keep your trust_env=False fix here
            async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
                r = await client.post(
                    f"{GROQ_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": messages, "max_tokens": 2000},
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                print(f"\n[DEBUG] HTTP 401: Your Groq API key is rejected.")
                return None
            elif e.response.status_code in (429, 503):
                # We read the exact header Groq sends to know how long to wait
                wait_time = float(e.response.headers.get("retry-after", _RETRY_DELAY * (2 ** attempt)))
                print(f"\n[DEBUG] Rate limit hit. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"\n[DEBUG] Groq API returned HTTP {e.response.status_code}: {e.response.text}")
                return None
        except httpx.ConnectError:
             print("\n[DEBUG] ConnectError: Still blocked from the internet. Are you behind a corporate firewall?")
             return None
        except Exception as e:
            print(f"\n[DEBUG] Unexpected error: {str(e)}")
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY)
                
    return None


async def call_cascade(
    prompt: str,
    system: Optional[str] = None,
    local_model: str = "qwen2.5:7b",
    groq_model: str = "llama-3.1-8b-instant",
    local_only: bool = False,
) -> tuple[str, str]:
    """
    Try models in order: Ollama → Groq → error.
    Returns (output, model_used).
    """
    result = await call_ollama(local_model, prompt, system)
    if result:
        return result, f"local/{local_model}"

    if local_only:
        raise RuntimeError(
            "Ollama unavailable and --local-only is set. Is Ollama running?"
        )

    result = await call_groq(groq_model, prompt, system)
    if result:
        return result, f"groq/{groq_model}"

    raise RuntimeError(
        "All models failed. Check: is Ollama running? Is GROQ_API_KEY set?\n"
        "  → Install Ollama: https://ollama.ai\n"
        "  → Get free Groq key: https://console.groq.com"
    )


async def call_groq_only(
    prompt: str,
    system: Optional[str] = None,
    groq_model: str = "llama-3.3-70b-versatile",
) -> Optional[str]:
    """
    Directly calls Groq, skipping Ollama.
    Used by judge escalation — we explicitly want the larger cloud model,
    not a retry of the same local model that gave a low-confidence result.
    """
    return await call_groq(groq_model, prompt, system)