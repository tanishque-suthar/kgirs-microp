"""Configuration and constants for the RAG evaluation project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from a .env file if present
load_dotenv()

# Random Seed for reproducibility across sampling and bootstrapping
RANDOM_SEED = 42

# Gemini model settings
# gemini-2.5-flash is the current recommended free/fast model
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# Stratified sample counts
SAMPLE_SIZE = 100
STRATIFIED_COUNTS = {
    "correct": 40,
    "partly": 30,
    "incorrect": 30
}

# Blind annotation subset size
ANNOTATION_SUBSET_SIZE = 35

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

SAMPLE_CSV_PATH = DATA_DIR / "sample_100.csv"
MY_LABELS_CSV_PATH = DATA_DIR / "my_labels.csv"
RAGAS_CACHE_PATH = CACHE_DIR / "ragas_eval_cache.json"
LLM_JUDGE_CACHE_PATH = CACHE_DIR / "llm_judge_cache.json"
RETRIEVAL_CACHE_PATH = CACHE_DIR / "retrieval_cache.json"

# Remote source files for RAGTruth dataset
RAGTRUTH_SOURCE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
RAGTRUTH_RESPONSE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"


def get_gemini_api_key() -> str:
    """
    Retrieve the Gemini API key from environment variables or Streamlit secrets.
    Never hardcodes secrets. Returns empty string if not configured.
    """
    # 1. Check environment variable
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key.strip()

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    return ""
