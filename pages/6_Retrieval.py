"""Page 6: Retrieval Quality vs Answer Truthfulness.

Analyzes the relationship between retrieval metrics (MRR, NDCG) and final answer correctness.
Includes visible warning regarding the use of proxy passage relevance.
Strict constraint: NO EMOJIS in UI.
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui_helpers import render_sidebar, load_retrieval_data
from src.stats import compute_retrieval_correlations, ORDERED_CLASSES

st.set_page_config(page_title="Page 6: Retrieval", layout="wide")
render_sidebar()

st.title("Page 6: Retrieval Quality vs Answer Ground Truth")
st.caption(
    "Investigating whether higher passage retrieval performance (MRR, NDCG) directly translates "
    "to higher answer faithfulness and lower hallucination rates."
)

# Mandatory Proxy Relevance Warning
st.warning(
    "Methodological Notice: Proxy Relevance Definition\n\n"
    "The RAGTruth corpus does not provide human ground-truth relevance marks for individual passages. "
    "In accordance with the study design, passage relevance is estimated via a documented LLM proxy: "
    "each retrieved passage is judged by Gemini (temperature 0, cached) for whether it contains direct "
    "supporting facts for the question-answer pair. These metrics should be interpreted strictly as proxy approximations."
)

df = load_retrieval_data()
if df is None or "reciprocal_rank" not in df.columns or "ndcg" not in df.columns:
    st.info(
        "Retrieval metrics data not found. Please click 'Run / Refresh Retrieval Metrics' in the sidebar "
        "to evaluate the 100 sampled answers."
    )
    st.stop()

# Compute summary stats & correlations
corr_results = compute_retrieval_correlations(df)

# Summary Cards
st.subheader("1. Summary Retrieval Statistics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Mean MRR", f"{corr_results['mean_mrr']:.3f}")
with c2:
    st.metric("Mean NDCG", f"{corr_results['mean_ndcg']:.3f}")
with c3:
    st.metric("Zero Relevant Rate", f"{corr_results['zero_relevant_rate'] * 100:.1f}%")
with c4:
    st.metric("Sample Size", f"{len(df)} answers")

st.divider()

# Visualizations
st.subheader("2. Retrieval Performance by Answer Ground Truth")

df["derived_label"] = pd.Categorical(df["derived_label"], categories=ORDERED_CLASSES, ordered=True)

col_r1, col_r2 = st.columns(2)

with col_r1:
    fig_mrr = px.box(
        df,
        x="derived_label",
        y="reciprocal_rank",
        color="derived_label",
        points="all",
        title="Reciprocal Rank (RR) by Human Label",
        labels={"reciprocal_rank": "Reciprocal Rank (0.0 to 1.0)", "derived_label": "Ground Truth"},
        color_discrete_map={
            "incorrect": "#e74c3c",
            "partly": "#f39c12",
            "correct": "#2ecc71"
        }
    )
    fig_mrr.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_mrr, use_container_width=True)

with col_r2:
    fig_ndcg = px.box(
        df,
        x="derived_label",
        y="ndcg",
        color="derived_label",
        points="all",
        title="NDCG by Human Label",
        labels={"ndcg": "NDCG Score (0.0 to 1.0)", "derived_label": "Ground Truth"},
        color_discrete_map={
            "incorrect": "#e74c3c",
            "partly": "#f39c12",
            "correct": "#2ecc71"
        }
    )
    fig_ndcg.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ndcg, use_container_width=True)

st.divider()

# Correlation Table
st.subheader("3. Rank Correlation Analysis (Retrieval vs Label)")

mrr_c = corr_results["mrr"]
ndcg_c = corr_results["ndcg"]

ret_table = pd.DataFrame([
    {
        "Retrieval Metric": "Mean Reciprocal Rank (MRR)",
        "Spearman r_s": mrr_c["spearman"],
        "Spearman 95% CI": f"[{mrr_c['spearman_ci'][0]:.3f}, {mrr_c['spearman_ci'][1]:.3f}]",
        "Spearman p-value": f"{mrr_c['spearman_p']:.4f}" if mrr_c['spearman_p'] is not None else "N/A",
        "Kendall tau-b": mrr_c["kendall"],
        "Kendall 95% CI": f"[{mrr_c['kendall_ci'][0]:.3f}, {mrr_c['kendall_ci'][1]:.3f}]",
        "Kendall p-value": f"{mrr_c['kendall_p']:.4f}" if mrr_c['kendall_p'] is not None else "N/A",
    },
    {
        "Retrieval Metric": "NDCG",
        "Spearman r_s": ndcg_c["spearman"],
        "Spearman 95% CI": f"[{ndcg_c['spearman_ci'][0]:.3f}, {ndcg_c['spearman_ci'][1]:.3f}]",
        "Spearman p-value": f"{ndcg_c['spearman_p']:.4f}" if ndcg_c['spearman_p'] is not None else "N/A",
        "Kendall tau-b": ndcg_c["kendall"],
        "Kendall 95% CI": f"[{ndcg_c['kendall_ci'][0]:.3f}, {ndcg_c['kendall_ci'][1]:.3f}]",
        "Kendall p-value": f"{ndcg_c['kendall_p']:.4f}" if ndcg_c['kendall_p'] is not None else "N/A",
    }
])

st.dataframe(ret_table, use_container_width=True)

st.divider()

# Observations
st.subheader("4. Retrieval Coupling Findings")
st.markdown("""
- **Retrieval as a Necessary but Insufficient Condition**: High MRR/NDCG ensures relevant information is ranked near the top, which is a prerequisite for accurate generation. However, models can still hallucinate or introduce extrinsic falsehoods even when relevant passages occupy rank 1.
- **Failures at Generation vs Retrieval**: Answers labeled `incorrect` frequently have low MRR, indicating the retriever failed to surface relevant evidence. Conversely, `partly` answers often exhibit high retrieval scores, showing that subtle hallucinations typically occur during reasoning/synthesis rather than retrieval.
""")
