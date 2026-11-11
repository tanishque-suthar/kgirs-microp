"""UI helper functions and shared data loaders for Streamlit pages.

Strict constraint: NO EMOJIS in any UI text or labels.
"""

import io
import sys
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DATA_DIR,
    SAMPLE_CSV_PATH,
    MY_LABELS_CSV_PATH,
    RAGAS_CACHE_PATH,
    LLM_JUDGE_CACHE_PATH,
    RETRIEVAL_CACHE_PATH,
    RANDOM_SEED,
    ANNOTATION_SUBSET_SIZE,
    get_gemini_api_key,
)


def load_dataset() -> Optional[pd.DataFrame]:
    """Load sample_100.csv if it exists."""
    if SAMPLE_CSV_PATH.exists():
        try:
            return pd.read_csv(SAMPLE_CSV_PATH)
        except Exception as e:
            st.error(f"Failed to load sample dataset: {e}")
            return None
    return None


def load_ragas_data() -> Optional[pd.DataFrame]:
    """Load ragas_eval_results.csv or cache."""
    csv_path = DATA_DIR / "ragas_eval_results.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return None


def load_judge_data() -> Optional[pd.DataFrame]:
    """Load llm_judge_results.csv."""
    csv_path = DATA_DIR / "llm_judge_results.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return None


def load_retrieval_data() -> Optional[pd.DataFrame]:
    """Load retrieval_results.csv."""
    csv_path = DATA_DIR / "retrieval_results.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return None


def load_my_labels() -> pd.DataFrame:
    """Load user's manual blind annotations from data/my_labels.csv."""
    if MY_LABELS_CSV_PATH.exists():
        try:
            df = pd.read_csv(MY_LABELS_CSV_PATH)
            # Ensure required columns exist
            for col in ["id", "my_label", "timestamp"]:
                if col not in df.columns:
                    df[col] = ""
            df["id"] = df["id"].astype(str)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["id", "my_label", "timestamp"])


def get_annotation_subset(df: pd.DataFrame, subset_size: int = ANNOTATION_SUBSET_SIZE) -> pd.DataFrame:
    """Select a deterministic subset of 35 answers for blind annotation."""
    if len(df) <= subset_size:
        return df.copy()
    return df.sample(n=subset_size, random_state=RANDOM_SEED).copy()


def render_sidebar():
    """Render shared sidebar with API key check, pipeline runner buttons, and export option."""
    st.sidebar.title("Pipeline Controls")

    # 1. API Key Status & Input
    st.sidebar.subheader("Gemini Configuration")
    api_key = get_gemini_api_key()
    if not api_key:
        user_key = st.sidebar.text_input(
            "Enter GEMINI_API_KEY",
            type="password",
            help="Free Gemini API key from Google AI Studio. Stored in session only."
        )
        if user_key.strip():
            import os
            os.environ["GEMINI_API_KEY"] = user_key.strip()
            st.sidebar.success("API Key set for current session.")
            api_key = user_key.strip()
    else:
        st.sidebar.info("API Key: Configured")

    # 2. Pipeline Execution
    st.sidebar.subheader("Execute Pipeline Stages")

    # Stage 1: Sampling
    if st.sidebar.button("Run / Refresh Sampling (100 items)"):
        with st.sidebar.status("Running stratified sampling..."):
            from src.sampling import sample_stratified_100
            try:
                sample_stratified_100()
                st.sidebar.success("Sample 100 generated.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Sampling error: {e}")

    # Stage 2: RAGAS Evaluation
    if st.sidebar.button("Run / Refresh RAGAS Eval"):
        df = load_dataset()
        if df is None:
            st.sidebar.warning("Please run sampling first.")
        elif not api_key:
            st.sidebar.error("GEMINI_API_KEY required.")
        else:
            prog_bar = st.sidebar.progress(0.0)
            status_text = st.sidebar.empty()

            def update_progress(curr, total):
                prog_bar.progress(curr / total)
                status_text.text(f"RAGAS: {curr}/{total}")

            from src.ragas_eval import RagasEvaluator
            try:
                evaluator = RagasEvaluator(api_key=api_key)
                evaluator.run_eval(df, progress_callback=update_progress)
                st.sidebar.success("RAGAS evaluation complete.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"RAGAS evaluation error: {e}")

    # Stage 3: LLM Judge Evaluation
    if st.sidebar.button("Run / Refresh LLM Judge"):
        df = load_dataset()
        if df is None:
            st.sidebar.warning("Please run sampling first.")
        elif not api_key:
            st.sidebar.error("GEMINI_API_KEY required.")
        else:
            prog_bar = st.sidebar.progress(0.0)
            status_text = st.sidebar.empty()

            def update_progress(curr, total):
                prog_bar.progress(curr / total)
                status_text.text(f"Judge: {curr}/{total}")

            from src.llm_judge import LLMJudge
            try:
                judge = LLMJudge(api_key=api_key)
                judge.run_eval(df, fluff_subset_size=20, progress_callback=update_progress)
                st.sidebar.success("LLM Judge evaluation complete.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Judge evaluation error: {e}")

    # Stage 4: Retrieval Metrics
    if st.sidebar.button("Run / Refresh Retrieval Metrics"):
        df = load_dataset()
        if df is None:
            st.sidebar.warning("Please run sampling first.")
        elif not api_key:
            st.sidebar.error("GEMINI_API_KEY required.")
        else:
            prog_bar = st.sidebar.progress(0.0)
            status_text = st.sidebar.empty()

            def update_progress(curr, total):
                prog_bar.progress(curr / total)
                status_text.text(f"Retrieval: {curr}/{total}")

            from src.retrieval_metrics import RetrievalEvaluator
            try:
                evaluator = RetrievalEvaluator(api_key=api_key)
                evaluator.run_eval(df, progress_callback=update_progress)
                st.sidebar.success("Retrieval metrics complete.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Retrieval error: {e}")

    # 3. Export Data
    st.sidebar.subheader("Export Report Data")
    zip_buffer = create_export_bundle()
    if zip_buffer:
        st.sidebar.download_button(
            label="Export report data (ZIP)",
            data=zip_buffer,
            file_name="rag_evaluation_results_bundle.zip",
            mime="application/zip",
            help="Download all result tables and cached evaluations in CSV format."
        )


def create_export_bundle() -> Optional[bytes]:
    """Bundle all available CSV results into a ZIP file for export."""
    buffer = io.BytesIO()
    has_files = False

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Check and add all result CSVs
        for filename in [
            "sample_100.csv",
            "ragas_eval_results.csv",
            "llm_judge_results.csv",
            "retrieval_results.csv",
            "my_labels.csv",
        ]:
            p = DATA_DIR / filename
            if p.exists():
                zf.write(p, arcname=filename)
                has_files = True

    if has_files:
        buffer.seek(0)
        return buffer.getvalue()
    return None
