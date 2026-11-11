"""LLM Judge Module using Gemini.

Evaluates RAG answer quality using a temperature 0 prompt, comparing against context.
Features:
- Position-bias test: evaluates each answer twice with rubric options listed in forward
  vs reversed order.
- Verbosity bias test: appends neutral fluff to a subset (~20 answers) and measures score change.
- Exponential backoff and full disk caching for every judge decision.
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    LLM_JUDGE_CACHE_PATH,
    SAMPLE_CSV_PATH,
    DATA_DIR,
    RANDOM_SEED,
)
from src.gemini_client import GeminiClient

# Forward Rubric: Correct -> Partly -> Incorrect
JUDGE_PROMPT_FORWARD = """You are an expert impartial judge evaluating the factual accuracy and faithfulness of a generated answer relative to the retrieved context passages.

Context Passages:
\"\"\"
{context}
\"\"\"

Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Evaluate the answer against the context passages according to the following rubric:
1. "correct": The answer is completely faithful to the provided context with no hallucinations, no unsupported claims, and no factual contradictions.
2. "partly": The answer is mostly accurate but has a minor flaw, slight unsupported detail, or 1 small minor hallucination.
3. "incorrect": The answer contains major factual contradictions, significant baseless claims, or multiple hallucinations.

Respond strictly in valid JSON format:
{{
  "verdict": "correct" or "partly" or "incorrect",
  "reason": "1-2 sentence concise justification"
}}
"""

# Reversed Rubric: Incorrect -> Partly -> Correct (Swapped order to test position bias)
JUDGE_PROMPT_REVERSED = """You are an expert impartial judge evaluating the factual accuracy and faithfulness of a generated answer relative to the retrieved context passages.

Context Passages:
\"\"\"
{context}
\"\"\"

Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Evaluate the answer against the context passages according to the following rubric:
1. "incorrect": The answer contains major factual contradictions, significant baseless claims, or multiple hallucinations.
2. "partly": The answer is mostly accurate but has a minor flaw, slight unsupported detail, or 1 small minor hallucination.
3. "correct": The answer is completely faithful to the provided context with no hallucinations, no unsupported claims, and no factual contradictions.

Respond strictly in valid JSON format:
{{
  "verdict": "incorrect" or "partly" or "correct",
  "reason": "1-2 sentence concise justification"
}}
"""

NEUTRAL_FLUFF_TEXT = (
    "\n\nAdditionally, it is standard knowledge that general context and background factors "
    "must be examined holistically across similar standard situations without altering the "
    "specific factual determinations established above."
)

LABEL_TO_ORDINAL = {
    "incorrect": 0,
    "partly": 1,
    "correct": 2,
}


def parse_judge_json(raw: str) -> Dict[str, str]:
    """Parse JSON response from the judge model into verdict and reason."""
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    if isinstance(data, dict):
        verdict = str(data.get("verdict", "")).strip().lower()
        reason = str(data.get("reason", "")).strip()
        if verdict in ["correct", "partly", "incorrect"]:
            return {"verdict": verdict, "reason": reason}
        if "incorrect" in verdict or "not correct" in verdict or "false" in verdict:
            return {"verdict": "incorrect", "reason": reason}
        elif "partly" in verdict or "partial" in verdict:
            return {"verdict": "partly", "reason": reason}
        elif "correct" in verdict or "true" in verdict:
            return {"verdict": "correct", "reason": reason}
        return {"verdict": "partly", "reason": reason}

    # Heuristic fallback if JSON fails
    lower = raw.lower()
    if '"verdict": "incorrect"' in lower or "verdict: incorrect" in lower or '"verdict": "not correct"' in lower:
        verdict = "incorrect"
    elif '"verdict": "partly"' in lower or "verdict: partly" in lower or "partially" in lower:
        verdict = "partly"
    elif '"verdict": "correct"' in lower or "verdict: correct" in lower:
        verdict = "correct"
    else:
        verdict = "partly"
    return {"verdict": verdict, "reason": raw[:120].strip()}


class LLMJudge:
    """Gemini LLM Judge with position bias check and verbosity perturbation."""

    def __init__(self, api_key: Optional[str] = None, cache_path: Path = LLM_JUDGE_CACHE_PATH):
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
            print(f"Warning: Failed to save LLM judge cache: {e}")

    def judge_single(self, context: str, question: str, answer: str, reversed_order: bool = False) -> Dict[str, str]:
        """Execute a single judgment call with specified rubric order."""
        template = JUDGE_PROMPT_REVERSED if reversed_order else JUDGE_PROMPT_FORWARD
        prompt = template.format(context=context, question=question, answer=answer)
        raw = self.client.generate_text(prompt, temperature=0.0)
        return parse_judge_json(raw)

    def evaluate_item(
        self,
        item_id: str,
        context: str,
        question: str,
        answer: str,
        run_fluff: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate an item in forward and reversed order (position-bias test),
        and optionally with neutral fluff appended (verbosity-bias test).
        """
        if item_id in self.cache:
            entry = self.cache[item_id]
            # If entry already has both orders and fluff is not requested or already present
            if "verdict_forward" in entry and "verdict_reversed" in entry:
                if not run_fluff or "verdict_fluff" in entry:
                    return entry

        # 1. Forward order judgment
        res_fwd = self.judge_single(context, question, answer, reversed_order=False)

        # 2. Reversed order judgment
        res_rev = self.judge_single(context, question, answer, reversed_order=True)

        entry = {
            "id": item_id,
            "verdict_forward": res_fwd["verdict"],
            "reason_forward": res_fwd["reason"],
            "verdict_reversed": res_rev["verdict"],
            "reason_reversed": res_rev["reason"],
            "position_flipped": res_fwd["verdict"] != res_rev["verdict"],
            "judge_verdict": res_fwd["verdict"],  # Primary verdict
        }

        # 3. Fluff test if requested
        if run_fluff:
            fluff_answer = answer + NEUTRAL_FLUFF_TEXT
            res_fluff = self.judge_single(context, question, fluff_answer, reversed_order=False)
            entry["verdict_fluff"] = res_fluff["verdict"]
            entry["reason_fluff"] = res_fluff["reason"]
            score_orig = LABEL_TO_ORDINAL.get(res_fwd["verdict"], 1)
            score_fluff = LABEL_TO_ORDINAL.get(res_fluff["verdict"], 1)
            entry["fluff_score_delta"] = score_fluff - score_orig
        else:
            entry["verdict_fluff"] = None
            entry["reason_fluff"] = None
            entry["fluff_score_delta"] = None

        self.cache[item_id] = entry
        self._save_cache()
        return entry

    def run_eval(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
        fluff_subset_size: int = 20,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Run judge evaluation across dataframe.
        Randomly selects fluff_subset_size items for verbosity test using fixed seed.
        """
        records = df.to_dict(orient="records")
        if limit:
            records = records[:limit]

        # Select fixed subset for fluff testing
        rng = random.Random(RANDOM_SEED)
        fluff_ids = set()
        if len(records) > 0:
            fluff_count = min(fluff_subset_size, len(records))
            all_ids = [str(r["id"]) for r in records]
            fluff_ids = set(rng.sample(all_ids, fluff_count))

        results = []
        total = len(records)
        for idx, r in enumerate(records):
            item_id = str(r["id"])
            is_fluff = item_id in fluff_ids

            res = self.evaluate_item(
                item_id=item_id,
                context=r["retrieved passages"],
                question=r["question"],
                answer=r["answer"],
                run_fluff=is_fluff,
            )

            r_copy = dict(r)
            r_copy["judge_verdict_forward"] = res["verdict_forward"]
            r_copy["judge_reason_forward"] = res["reason_forward"]
            r_copy["judge_verdict_reversed"] = res["verdict_reversed"]
            r_copy["judge_reason_reversed"] = res["reason_reversed"]
            r_copy["position_flipped"] = res["position_flipped"]
            r_copy["judge_verdict"] = res["judge_verdict"]
            r_copy["is_fluff_tested"] = is_fluff
            r_copy["judge_verdict_fluff"] = res.get("verdict_fluff")
            r_copy["fluff_score_delta"] = res.get("fluff_score_delta")
            results.append(r_copy)

            if progress_callback:
                progress_callback(idx + 1, total)

        res_df = pd.DataFrame(results)
        out_csv = DATA_DIR / "llm_judge_results.csv"
        res_df.to_csv(out_csv, index=False)
        return res_df


def smoke_test_judge(sample_csv: Path = SAMPLE_CSV_PATH, n: int = 5):
    """Run a small smoke test on 5 items."""
    print(f"Running LLM Judge smoke test on first {n} items...")
    df = pd.read_csv(sample_csv).head(n)
    judge = LLMJudge()
    res = judge.run_eval(df, fluff_subset_size=2)
    print("\nSmoke test completed successfully!")
    print(res[["id", "derived_label", "judge_verdict_forward", "judge_verdict_reversed", "position_flipped"]])


if __name__ == "__main__":
    smoke_test_judge()
