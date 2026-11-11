"""Retrieval Metrics Module.

Computes Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG)
over the retrieved passage list. Because RAGTruth does not contain passage-level ground truth
relevance annotations, this module implements a documented proxy:
A passage is classified as relevant (1) if it directly supports the factual claims
in the answer or question context, determined via a cached Gemini binary judgment call.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RETRIEVAL_CACHE_PATH,
    SAMPLE_CSV_PATH,
    DATA_DIR,
)
from src.gemini_client import GeminiClient

PASSAGE_PROXY_PROMPT = """You are evaluating passage relevance in a Question-Answering RAG system.

Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Passage to Evaluate:
\"\"\"
{passage}
\"\"\"

Does this specific passage provide relevant supporting information or evidence for the question and answer?
Answer strictly in valid JSON:
{{
  "relevant": true or false,
  "reason": "short explanation"
}}
"""


def parse_passages(text: str) -> List[str]:
    """Parse a concatenated RAGTruth passage string into individual passages."""
    parts = [p.strip() for p in re.split(r'passage\s+\d+\s*:', text, flags=re.IGNORECASE) if p.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return parts


def compute_reciprocal_rank(relevance_scores: List[int]) -> float:
    """Compute Reciprocal Rank (RR = 1/k for first relevant passage, or 0.0)."""
    for idx, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (idx + 1)
    return 0.0


def compute_ndcg_at_k(relevance_scores: List[int], k: Optional[int] = None) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    if k is not None:
        relevance_scores = relevance_scores[:k]
    if not relevance_scores or sum(relevance_scores) == 0:
        return 0.0

    # DCG
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    # IDCG (ideal sorted relevance)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


class RetrievalEvaluator:
    """Computes retrieval metrics using cached proxy relevance calls."""

    def __init__(self, api_key: Optional[str] = None, cache_path: Path = RETRIEVAL_CACHE_PATH):
        self.client = GeminiClient(api_key=api_key)
        self.cache_path = cache_path
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save retrieval cache: {e}")

    def evaluate_passage_relevance(self, question: str, answer: str, passage: str) -> bool:
        """Evaluate single passage relevance via Gemini binary judgment."""
        prompt = PASSAGE_PROXY_PROMPT.format(question=question, answer=answer, passage=passage)
        raw = self.client.generate_text(prompt, temperature=0.0)
        try:
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                data = json.loads(cleaned)
            except Exception:
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if match:
                    data = json.loads(match.group(0))
                else:
                    data = {}

            return bool(data.get("relevant", False))
        except Exception:
            lower = raw.lower()
            if '"relevant": true' in lower or 'relevant: true' in lower:
                return True
            elif '"relevant": false' in lower or 'relevant: false' in lower:
                return False
            return "true" in lower and "false" not in lower

    def evaluate_item(self, item_id: str, question: str, answer: str, passages_text: str) -> Dict[str, Any]:
        """Compute RR, NDCG, and proxy passage labels for a single item with caching."""
        if item_id in self.cache:
            entry = self.cache[item_id]
            if ("reciprocal_rank" in entry or "mrr" in entry) and "ndcg" in entry:
                # Ensure reciprocal_rank key is populated
                if "reciprocal_rank" not in entry and "mrr" in entry:
                    entry["reciprocal_rank"] = entry["mrr"]
                return entry

        passages = parse_passages(passages_text)
        relevance_vector = []

        # Evaluate each passage (typically 3-10 passages)
        for p in passages:
            is_rel = self.evaluate_passage_relevance(question=question, answer=answer, passage=p)
            relevance_vector.append(1 if is_rel else 0)

        rr = compute_reciprocal_rank(relevance_vector)
        ndcg = compute_ndcg_at_k(relevance_vector)

        entry = {
            "id": item_id,
            "passage_count": len(passages),
            "relevance_vector": relevance_vector,
            "reciprocal_rank": round(rr, 4),
            "ndcg": round(ndcg, 4),
            "relevant_count": sum(relevance_vector),
        }
        self.cache[item_id] = entry
        self._save_cache()
        return entry

    def run_eval(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """Run retrieval evaluation across dataframe with progress tracking."""
        records = df.to_dict(orient="records")
        if limit:
            records = records[:limit]

        results = []
        total = len(records)
        for idx, r in enumerate(records):
            item_id = str(r["id"])
            res = self.evaluate_item(
                item_id=item_id,
                question=r["question"],
                answer=r["answer"],
                passages_text=r["retrieved passages"],
            )

            r_copy = dict(r)
            r_copy["reciprocal_rank"] = res["reciprocal_rank"]
            r_copy["ndcg"] = res["ndcg"]
            r_copy["passage_count"] = res["passage_count"]
            r_copy["relevant_passage_count"] = res["relevant_count"]
            results.append(r_copy)

            if progress_callback:
                progress_callback(idx + 1, total)

        res_df = pd.DataFrame(results)
        out_csv = DATA_DIR / "retrieval_results.csv"
        res_df.to_csv(out_csv, index=False)
        return res_df


def smoke_test_retrieval(sample_csv: Path = SAMPLE_CSV_PATH, n: int = 5):
    """Run a small smoke test on 5 items."""
    print(f"Running Retrieval smoke test on first {n} items...")
    df = pd.read_csv(sample_csv).head(n)
    evaluator = RetrievalEvaluator()
    res = evaluator.run_eval(df)
    print("\nSmoke test completed successfully!")
    print(res[["id", "derived_label", "reciprocal_rank", "ndcg", "relevant_passage_count"]])


if __name__ == "__main__":
    smoke_test_retrieval()
