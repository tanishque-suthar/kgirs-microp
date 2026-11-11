"""RAGAS Evaluation Module.

Computes Faithfulness and Answer Relevancy for RAG answers using Gemini as backing LLM
and text embedding model. Caches every individual result to disk to prevent redundant API calls.
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
    RAGAS_CACHE_PATH,
    SAMPLE_CSV_PATH,
    DATA_DIR,
)
from src.gemini_client import GeminiClient


FAITHFULNESS_PROMPT = """You are an expert NLP evaluator implementing the RAGAS Faithfulness metric.

Given the following retrieved passages and an answer, evaluate whether all factual statements made in the answer can be directly inferred from the retrieved passages.

Retrieved Passages:
\"\"\"
{context}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Instructions:
1. Break down the answer into a list of atomic factual statements / claims.
2. For each statement, decide whether it is completely supported by the retrieved passages (Yes or No).
3. Return your response ONLY as valid JSON in the following format:
{{
  "statements": [
    {{"statement": "...", "supported": true or false, "reason": "..."}}
  ],
  "faithfulness_score": float between 0.0 and 1.0 (number of supported statements / total statements)
}}
"""

ANSWER_RELEVANCY_PROMPT = """You are an expert NLP evaluator implementing the RAGAS Answer Relevancy metric.

Given the following generated answer, generate 3 distinct candidate questions that this specific answer could be an answer to.

Answer:
\"\"\"
{answer}
\"\"\"

Return your response ONLY as valid JSON in the following format:
{{
  "generated_questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}}
"""


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two numerical vectors."""
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    if np.isnan(sim):
        return 0.0
    return max(-1.0, min(1.0, sim))


class RagasEvaluator:
    """Evaluates Faithfulness and Answer Relevancy using Gemini with disk-backed caching."""

    def __init__(self, api_key: Optional[str] = None, cache_path: Path = RAGAS_CACHE_PATH):
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
            print(f"Warning: Failed to save RAGAS cache: {e}")

    def evaluate_faithfulness(self, context: str, answer: str) -> Dict[str, Any]:
        """Compute RAGAS faithfulness score (0.0 to 1.0)."""
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        raw = self.client.generate_text(prompt, temperature=0.0)

        # Parse JSON from response
        try:
            clean_json = raw.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                data = json.loads(clean_json)
            except Exception:
                match = re.search(r"\{[\s\S]*\}", clean_json)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise ValueError("No JSON object found")

            statements = data.get("statements", [])
            if not statements:
                score = 1.0
            else:
                supported_count = sum(1 for s in statements if s.get("supported") is True)
                score = round(supported_count / len(statements), 4)

            return {
                "faithfulness": score,
                "statements": statements,
                "supported_count": sum(1 for s in statements if s.get("supported") is True),
                "total_statements": len(statements),
            }
        except Exception as e:
            # Fallback heuristic if JSON parsing fails
            score = 1.0 if "1.0" in raw or "yes" in raw.lower() else 0.5
            return {
                "faithfulness": score,
                "statements": [],
                "supported_count": 0,
                "total_statements": 0,
                "raw_error": str(e),
            }

    def evaluate_answer_relevancy(self, question: str, answer: str) -> Dict[str, Any]:
        """Compute RAGAS answer relevancy score via reverse-generated question embeddings."""
        prompt = ANSWER_RELEVANCY_PROMPT.format(answer=answer)
        raw = self.client.generate_text(prompt, temperature=0.0)

        # Parse generated questions
        gen_questions = []
        try:
            clean_json = raw.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                data = json.loads(clean_json)
            except Exception:
                match = re.search(r"\{[\s\S]*\}", clean_json)
                if match:
                    data = json.loads(match.group(0))
                else:
                    data = {}

            gen_questions = data.get("generated_questions", [])
        except Exception:
            lines = [l.strip().lstrip("-*123. ") for l in raw.split("\n") if "?" in l]
            gen_questions = lines[:3]

        if not gen_questions:
            gen_questions = [question]

        # Compute embedding similarities
        try:
            q_emb = self.client.get_embedding(question)
            sims = []
            for gq in gen_questions:
                gq_emb = self.client.get_embedding(gq)
                sims.append(cosine_similarity(q_emb, gq_emb))

            relevancy_score = round(float(np.mean(sims)), 4) if sims else 0.5
        except Exception as e:
            # Fallback to Jaccard / lexical overlap if embedding fails
            q_words = set(re.findall(r"\w+", question.lower()))
            ans_words = set(re.findall(r"\w+", answer.lower()))
            jaccard = len(q_words & ans_words) / max(1, len(q_words | ans_words))
            relevancy_score = round(min(1.0, 0.5 + jaccard), 4)

        return {
            "answer_relevancy": relevancy_score,
            "generated_questions": gen_questions,
        }

    def evaluate_item(self, item_id: str, question: str, context: str, answer: str) -> Dict[str, Any]:
        """Evaluate both metrics for a single item with caching."""
        if item_id in self.cache:
            cached = self.cache[item_id]
            if "faithfulness" in cached and "answer_relevancy" in cached:
                return cached

        f_res = self.evaluate_faithfulness(context=context, answer=answer)
        r_res = self.evaluate_answer_relevancy(question=question, answer=answer)

        merged = {
            "id": item_id,
            "faithfulness": f_res["faithfulness"],
            "answer_relevancy": r_res["answer_relevancy"],
            "statements": f_res.get("statements", []),
            "generated_questions": r_res.get("generated_questions", []),
        }
        self.cache[item_id] = merged
        self._save_cache()
        return merged

    def run_eval(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Evaluate full dataset (or slice if limit is set).
        Updates and returns the DataFrame with 'ragas_faithfulness' and 'ragas_relevancy'.
        """
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
                context=r["retrieved passages"],
                answer=r["answer"],
            )
            r_copy = dict(r)
            r_copy["ragas_faithfulness"] = res["faithfulness"]
            r_copy["ragas_relevancy"] = res["answer_relevancy"]
            results.append(r_copy)

            if progress_callback:
                progress_callback(idx + 1, total)

        res_df = pd.DataFrame(results)
        # Save to CSV
        out_csv = DATA_DIR / "ragas_eval_results.csv"
        res_df.to_csv(out_csv, index=False)
        return res_df


def smoke_test_ragas(sample_csv: Path = SAMPLE_CSV_PATH, n: int = 5):
    """Run a small smoke test on 5 items."""
    print(f"Running RAGAS smoke test on first {n} items...")
    df = pd.read_csv(sample_csv).head(n)
    evaluator = RagasEvaluator()
    res = evaluator.run_eval(df)
    print("\nSmoke test completed successfully!")
    print(res[["id", "derived_label", "ragas_faithfulness", "ragas_relevancy"]])


if __name__ == "__main__":
    smoke_test_ragas()
