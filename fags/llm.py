"""Thin Groq client used by the LLM experiments.

Reads GROQ_API_KEY from the environment or from the project's git-ignored
.env file. Every call is cached on disk (results/v2/llm_cache.jsonl) keyed
by (model, messages, temperature, seed), so experiments are reproducible and
re-runs cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "results" / "v2" / "llm_cache.jsonl"


class QuotaExhausted(RuntimeError):
    """Daily request/token quota used up. Re-run later: cached calls are free."""


def _load_key() -> str:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is empty - paste it into the project's .env file.")
    return key


class GroqLLM:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        from groq import Groq
        self.client = Groq(api_key=_load_key())
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self._cache: dict[str, str] = {}
        # One cache file per model, so parallel runs on different models never
        # append to the same file. The shared legacy file is still read.
        self.cache_path = CACHE_PATH.with_name(f"llm_cache_{model.split('/')[-1]}.jsonl")
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        for path in (CACHE_PATH, self.cache_path):
            if path.exists():
                for line in path.open(encoding="utf-8"):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn line from an interrupted write
                    if rec.get("model") == model:
                        self._cache[rec["key"]] = rec["text"]
        self.calls = self.cache_hits = 0

    def chat(self, messages: list[dict], seed: int = 0, temperature: float | None = None) -> str:
        temp = self.temperature if temperature is None else temperature
        key = hashlib.sha256(json.dumps([self.model, messages, temp, seed, self.max_tokens]).encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        from groq import RateLimitError, APIStatusError, APIConnectionError
        for attempt in range(30):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temp,
                    max_tokens=self.max_tokens, seed=seed,
                )
                break
            except RateLimitError as e:
                msg = str(e).lower()
                if "per day" in msg or "(rpd)" in msg or "(tpd)" in msg:
                    raise QuotaExhausted(str(e)[:300]) from e
                retry = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
                time.sleep(float(retry) + 1 if retry else min(60, 5 * (attempt + 1)))
            except (APIConnectionError, APIStatusError) as e:  # transient server/network errors
                if isinstance(e, APIStatusError) and e.status_code < 500:
                    raise
                time.sleep(min(60, 2 ** min(attempt, 5) * 3))
        else:
            # Persistent rate limiting: stop gracefully so the run can resume later.
            raise QuotaExhausted("still rate-limited after 30 retries")
        text = resp.choices[0].message.content or ""
        self.calls += 1
        self._cache[key] = text
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": self.model, "text": text}) + "\n")
        return text

    @staticmethod
    def list_models() -> list[str]:
        from groq import Groq
        return sorted(m.id for m in Groq(api_key=_load_key()).models.list().data)
