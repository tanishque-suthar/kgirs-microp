"""Data sampling module for RAGTruth QA subset.

Loads QA pairs and responses from the official RAGTruth corpus,
derives a 3-class ground truth hallucination label, performs
reproducible stratified sampling (40 correct / 30 partly / 30 incorrect),
and saves the output to data/sample_100.csv.
"""

import json
import os
import sys
import random
import urllib.request
from pathlib import Path

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.config import (
    DATA_DIR,
    SAMPLE_CSV_PATH,
    RANDOM_SEED,
    STRATIFIED_COUNTS,
    RAGTRUTH_SOURCE_URL,
    RAGTRUTH_RESPONSE_URL,
)


def derive_label(labels: list[dict], max_minor_length: int = 80) -> str:
    """
    Derive a 3-class hallucination label ('correct', 'partly', 'incorrect')
    from RAGTruth hallucination span annotations.

    Rule:
    - 'correct': No hallucinated spans detected (len(labels) == 0).
    - 'partly': Exactly 1 span that is either subtle or short
      (label_type contains 'Subtle' OR character length <= max_minor_length).
    - 'incorrect': Multiple hallucinated spans (len(labels) >= 2) OR
      1 evident and long hallucinated span (> max_minor_length).

    Parameters:
    -----------
    labels : list[dict]
        List of hallucinated spans from RAGTruth response annotation.
        Each span contains 'text', 'start', 'end', and 'label_type'.
    max_minor_length : int, default=80
        Maximum character length of a single span to still be considered minor/partly.

    Returns:
    --------
    str: 'correct', 'partly', or 'incorrect'
    """
    if not labels or len(labels) == 0:
        return "correct"

    if len(labels) == 1:
        span = labels[0]
        text = span.get("text", "")
        label_type = span.get("label_type", "")
        is_subtle = "Subtle" in label_type

        # If it's subtle OR short in length, classify as partly
        if is_subtle or len(text) <= max_minor_length:
            return "partly"
        return "incorrect"

    # Multiple spans -> major hallucination
    return "incorrect"


def fetch_or_load_ragtruth_qa(cache_file: Path = DATA_DIR / "ragtruth_qa_full.json") -> list[dict]:
    """
    Fetch and parse the QA subset of RAGTruth, caching the joined QA records locally.
    Returns a list of dicts with question, passages, answer, labels, etc.
    """
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Fetching QA records from RAGTruth repository...")

    # Step 1: Download & parse source_info.jsonl for task_type == 'QA'
    qa_sources = {}
    req_source = urllib.request.Request(
        RAGTRUTH_SOURCE_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req_source) as resp:
        for line in resp:
            rec = json.loads(line.decode("utf-8"))
            if rec.get("task_type") == "QA":
                sid = str(rec["source_id"])
                s_info = rec.get("source_info", {})
                qa_sources[sid] = {
                    "question": s_info.get("question", "").strip(),
                    "passages": s_info.get("passages", "").strip(),
                    "source": rec.get("source", "MARCO")
                }

    print(f"Loaded {len(qa_sources)} QA source contexts. Streaming responses...")

    # Step 2: Download & filter response.jsonl matching QA source IDs
    qa_records = []
    req_resp = urllib.request.Request(
        RAGTRUTH_RESPONSE_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req_resp) as resp:
        for line in resp:
            r = json.loads(line.decode("utf-8"))
            sid = str(r.get("source_id", ""))
            if sid in qa_sources:
                ans = r.get("response", "").strip()
                # Skip degenerate empty responses or low-quality refusals
                if not ans or r.get("quality") == "truncated":
                    continue
                
                labels = r.get("labels", [])
                derived = derive_label(labels)
                qa_records.append({
                    "id": str(r.get("id")),
                    "source_id": sid,
                    "model": r.get("model", ""),
                    "question": qa_sources[sid]["question"],
                    "retrieved passages": qa_sources[sid]["passages"],
                    "answer": ans,
                    "derived_label": derived,
                    "raw_labels": labels,
                    "answer_length": len(ans),
                })

    print(f"Loaded {len(qa_records)} total QA model responses.")
    # Cache locally for instant reuse
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(qa_records, f, indent=2)

    return qa_records


def sample_stratified_100(
    records: list[dict] = None,
    seed: int = RANDOM_SEED,
    target_counts: dict = None,
    save_path: Path = SAMPLE_CSV_PATH
) -> pd.DataFrame:
    """
    Sample 100 answers stratified by derived_label (~40 correct / 30 partly / 30 incorrect).
    Saves to data/sample_100.csv.
    """
    if target_counts is None:
        target_counts = STRATIFIED_COUNTS

    if records is None:
        records = fetch_or_load_ragtruth_qa()

    # Group by derived label
    groups = {"correct": [], "partly": [], "incorrect": []}
    for r in records:
        lbl = r["derived_label"]
        if lbl in groups:
            groups[lbl].append(r)

    rng = random.Random(seed)
    sampled_records = []

    for label, count in target_counts.items():
        pool = groups[label]
        if len(pool) < count:
            raise ValueError(f"Not enough instances for label '{label}': {len(pool)} available, {count} requested.")
        sampled = rng.sample(pool, count)
        sampled_records.extend(sampled)

    # Shuffle the combined sample using the fixed seed so classes are intermixed
    rng.shuffle(sampled_records)

    # Format into DataFrame with the exact columns specified
    df = pd.DataFrame([
        {
            "id": r["id"],
            "question": r["question"],
            "retrieved passages": r["retrieved passages"],
            "answer": r["answer"],
            "derived_label": r["derived_label"],
            "answer_length": r["answer_length"],
        }
        for r in sampled_records
    ])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Successfully saved {len(df)} stratified samples to {save_path}.")
    print("Class breakdown:\n", df["derived_label"].value_counts())
    return df


if __name__ == "__main__":
    sample_stratified_100()
