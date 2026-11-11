"""Unified Gemini API client with caching, exponential backoff, and rate-limit handling."""

import hashlib
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from google import genai
from google.genai import types

from src.config import (
    GEMINI_MODEL,
    GEMINI_EMBEDDING_MODEL,
    CACHE_DIR,
    get_gemini_api_key,
)

RAW_API_CACHE_PATH = CACHE_DIR / "gemini_raw_cache.json"


class GeminiClient:
    """Wrapper around Google GenAI client with disk caching and rate-limit backoff."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL, mock_mode: bool = False):
        self.api_key = api_key or get_gemini_api_key()
        self.model = model
        self.embedding_model = GEMINI_EMBEDDING_MODEL
        self.mock_mode = mock_mode or (os.getenv("GEMINI_MOCK", "0") == "1") or (self.api_key and self.api_key.upper() == "MOCK")
        self._client: Optional[genai.Client] = None
        self._cache = self._load_cache()

    def _get_client(self) -> Optional[genai.Client]:
        """Lazily initialize client with validation."""
        if self.mock_mode:
            return None
        if not self.api_key:
            self.api_key = get_gemini_api_key()
        if not self.api_key:
            if os.getenv("GEMINI_MOCK", "0") == "1":
                self.mock_mode = True
                return None
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable, "
                "add it to .env, or enter it in the Streamlit sidebar."
            )
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _load_cache(self) -> Dict[str, Any]:
        if RAW_API_CACHE_PATH.exists():
            try:
                with open(RAW_API_CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(RAW_API_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save raw cache: {e}")

    @staticmethod
    def _hash_key(prefix: str, payload: str) -> str:
        return f"{prefix}::{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
        use_cache: bool = True,
        max_retries: int = 6,
        initial_backoff: float = 2.0,
    ) -> str:
        """
        Generate text response with exponential backoff on rate limits and disk caching.
        """
        cache_key = self._hash_key(
            f"gen::{self.model}::temp={temperature}",
            f"{system_instruction or ''}___{prompt}"
        )

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if self.mock_mode:
            text = self._mock_generate(prompt)
            self._cache[cache_key] = text
            self._save_cache()
            return text

        client = self._get_client()
        backoff = initial_backoff

        for attempt in range(1, max_retries + 1):
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                )
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                try:
                    text = response.text or ""
                except Exception:
                    text = ""
                self._cache[cache_key] = text
                self._save_cache()
                time.sleep(0.5)  # Pace requests to respect free-tier RPM
                return text

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "resource_exhausted" in err_str
                    or "quota" in err_str
                    or "rate" in err_str
                )

                if is_rate_limit and attempt < max_retries:
                    sleep_time = backoff + random.uniform(0.5, 2.0)
                    print(f"Rate limited (attempt {attempt}/{max_retries}). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    backoff *= 2.0
                elif attempt < max_retries and ("503" in err_str or "unavailable" in err_str or "connection" in err_str):
                    sleep_time = backoff + random.uniform(0.5, 1.5)
                    print(f"Transient error (attempt {attempt}/{max_retries}). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    backoff *= 1.5
                else:
                    raise RuntimeError(f"Gemini API generation failed after {attempt} attempts: {e}")

        raise RuntimeError("Max retries exceeded for text generation.")

    def _mock_generate(self, prompt: str) -> str:
        """Simulate realistic model responses during offline testing or smoke tests."""
        p_lower = prompt.lower()
        if "faithfulness" in p_lower:
            return json.dumps({
                "statements": [
                    {"statement": "Key point from answer", "supported": True, "reason": "Mentioned in context"},
                    {"statement": "Secondary procedural step", "supported": True, "reason": "Mentioned in context"}
                ],
                "faithfulness_score": 0.85
            })
        elif "answer relevancy" in p_lower:
            return json.dumps({
                "generated_questions": [
                    "What are the main details outlined in the answer?",
                    "How is the process conducted?",
                    "What conditions apply here?"
                ]
            })
        elif "impartial judge" in p_lower:
            h = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16)
            verdicts = ["correct", "partly", "incorrect"]
            chosen = verdicts[h % len(verdicts)]
            return json.dumps({
                "verdict": chosen,
                "reason": f"Simulated judge rationale: consistent with {chosen} criteria."
            })
        elif "passage relevance" in p_lower:
            h = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16)
            return json.dumps({
                "relevant": (h % 3 != 0),
                "reason": "Passage contains matching entities and instructions."
            })
        return "Simulated response"

    def get_embedding(
        self,
        text: str,
        use_cache: bool = True,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
    ) -> List[float]:
        """
        Generate text embeddings with disk caching and rate limit backoff.
        """
        cache_key = self._hash_key(f"emb::{self.embedding_model}", text)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if self.mock_mode:
            h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
            rng = random.Random(h)
            emb = [rng.gauss(0.0, 1.0) for _ in range(64)]
            norm = sum(x*x for x in emb) ** 0.5
            emb = [x / norm for x in emb]
            self._cache[cache_key] = emb
            self._save_cache()
            return emb

        client = self._get_client()
        backoff = initial_backoff

        for attempt in range(1, max_retries + 1):
            try:
                res = client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                )
                # Res may contain embedding or embeddings
                emb = []
                if hasattr(res, "embedding") and res.embedding:
                    emb = list(res.embedding.values)
                elif hasattr(res, "embeddings") and res.embeddings:
                    emb = list(res.embeddings[0].values)
                
                if emb:
                    self._cache[cache_key] = emb
                    self._save_cache()
                    time.sleep(0.3)  # Respect free-tier rate limit
                    return emb
                else:
                    raise ValueError("Empty embedding returned by API")

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "resource_exhausted" in err_str
                    or "quota" in err_str
                    or "rate" in err_str
                )
                if is_rate_limit and attempt < max_retries:
                    sleep_time = backoff + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                    backoff *= 2.0
                elif attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 1.5
                else:
                    raise RuntimeError(f"Gemini embedding failed: {e}")

        raise RuntimeError("Max retries exceeded for embedding generation.")
