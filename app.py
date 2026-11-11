"""Page 1: Overview - How much can we trust automatic evaluation of RAG answers?

Main entrypoint for the Streamlit research application.
Strict constraint: NO EMOJIS in UI.
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui_helpers import render_sidebar, load_dataset
from src.sampling import derive_label


def render_overview():
    """Render the contents of Page 1: Overview."""
    st.title("Page 1: Overview")
    st.subheader("Research Investigation: How Much Can We Trust Automatic Evaluation of RAG Answers?")

    st.markdown("""
### 1. Research Question
Retrieval-Augmented Generation (RAG) pipelines increasingly rely on automated evaluators—such as
RAGAS (Faithfulness, Answer Relevancy) and LLM-as-a-Judge prompting—to validate answer accuracy
without continuous human oversight.

This project investigates:
1. **Correlation with Human Ground Truth**: Do automatic metrics (RAGAS, LLM Judge) reliably separate
   flawless answers from minor hallucinations and major factual contradictions?
2. **Systematic Biases**: Are LLM judges susceptible to **position bias** (altering judgments when option
   rubric orders are reversed) and **verbosity bias** (rewarding neutral, uninformative fluff)?
3. **Retrieval Coupling**: Does retrieval quality (MRR, NDCG) correlate with final answer correctness?
4. **Inter-Annotator Calibration**: How does automated evaluation compare with human manual relabelling?
""")

    st.divider()

    # Dataset Summary
    st.markdown("### 2. Dataset and Stratified Sampling")

    df = load_dataset()
    if df is None:
        st.warning("Sample dataset not found. Click 'Run / Refresh Sampling (100 items)' in the sidebar to generate data/sample_100.csv.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sample Size", f"{len(df)} items")
        with col2:
            st.metric("Source Corpus", "RAGTruth (QA Subset)")
        with col3:
            st.metric("Avg Answer Length", f"{int(df['answer_length'].mean())} chars")
        with col4:
            st.metric("Random Seed", "42 (Deterministic)")

        st.markdown("#### Sample Stratification Breakdown")
        counts = df["derived_label"].value_counts().reset_index()
        counts.columns = ["Derived Label", "Count"]

        fig = px.bar(
            counts,
            x="Derived Label",
            y="Count",
            color="Derived Label",
            color_discrete_map={
                "correct": "#2ecc71",
                "partly": "#f39c12",
                "incorrect": "#e74c3c"
            },
            text="Count",
            title="Stratified Ground Truth Distribution (n=100)"
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Label Class",
            yaxis_title="Number of Answers",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Preview Sampled Dataset (First 10 Rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # Ground Truth Label Rule
    st.markdown("### 3. Three-Class Label Derivation Rule")
    st.markdown("""
The ground truth labels are derived directly from the human-annotated word-level hallucination spans
in the official RAGTruth corpus:
""")

    st.code(f"""
Config Rule (src/sampling.py):

{derive_label.__doc__}
""", language="python")

    st.markdown("""
#### Class Definitions:
- **correct**: Exactly 0 hallucinated spans identified by RAGTruth annotators. The response is completely faithful to the retrieved passages.
- **partly**: Exactly 1 hallucinated span that is either subtle (`label_type` contains "Subtle") or short (span length <= 80 characters). Represents minor inaccuracies or minor extrapolations.
- **incorrect**: 2 or more hallucinated spans, or 1 major/long evident hallucination (`label_type` is "Evident" and span length > 80 characters). Represents severe factual distortions.
""")

    st.divider()

    st.markdown("""
### Navigation Guide
Use the sidebar navigation to explore the investigation pages:
- **Page 2: Annotate** - Blind relabelling tool for a random 35-item subset to compute human agreement.
- **Page 3: RAGAS vs Human** - Faithfulness and Answer Relevancy correlation analysis with 95% bootstrap CIs.
- **Page 4: LLM Judge** - LLM-as-a-judge accuracy, quadratic Cohen's kappa, position bias, and verbosity tests.
- **Page 5: Agreement** - Inter-annotator agreement (pairwise Cohen's kappa, Fleiss' kappa) across RAGTruth, User, and Gemini.
- **Page 6: Retrieval** - MRR and NDCG analysis against answer quality with proxy relevance disclosure.
- **Page 7: Limitations** - Methodological limitations, wide confidence intervals, and confounding risks.
""")


if __name__ == "__main__" or "streamlit" in sys.modules:
    st.set_page_config(
        page_title="RAG Evaluation Trust Study",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_sidebar()
    render_overview()
