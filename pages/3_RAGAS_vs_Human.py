"""Page 3: RAGAS vs Human Evaluation.

Analyzes correlation between continuous RAGAS metrics (Faithfulness, Answer Relevancy)
and ordinal human ground truth labels (incorrect=0, partly=1, correct=2).
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

from src.ui_helpers import render_sidebar, load_ragas_data
from src.stats import compute_correlations, ORDERED_CLASSES

st.set_page_config(page_title="Page 3: RAGAS vs Human", layout="wide")
render_sidebar()

st.title("Page 3: RAGAS Metrics vs Human Ground Truth")
st.caption(
    "Evaluating whether continuous automatic RAG metrics correlate with stratified human labels "
    "(incorrect, partly, correct) and reliably discriminate hallucination severity."
)

df = load_ragas_data()

if df is None or "ragas_faithfulness" not in df.columns or "ragas_relevancy" not in df.columns:
    st.info(
        "RAGAS evaluation data not found. Please click 'Run / Refresh RAGAS Eval' in the sidebar "
        "to compute Faithfulness and Answer Relevancy on the sampled answers."
    )
    st.stop()

# Order categorical label for clean plotting
df["derived_label"] = pd.Categorical(df["derived_label"], categories=ORDERED_CLASSES, ordered=True)

# Overview Metrics
st.subheader("1. Distribution of Automatic Scores by Ground Truth Label")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_faith = px.box(
        df,
        x="derived_label",
        y="ragas_faithfulness",
        color="derived_label",
        points="all",
        title="RAGAS Faithfulness by Human Label",
        labels={
            "derived_label": "Human Ground Truth",
            "ragas_faithfulness": "Faithfulness Score (0.0 to 1.0)"
        },
        color_discrete_map={
            "incorrect": "#e74c3c",
            "partly": "#f39c12",
            "correct": "#2ecc71"
        }
    )
    fig_faith.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_faith, use_container_width=True)

with chart_col2:
    fig_rel = px.box(
        df,
        x="derived_label",
        y="ragas_relevancy",
        color="derived_label",
        points="all",
        title="RAGAS Answer Relevancy by Human Label",
        labels={
            "derived_label": "Human Ground Truth",
            "ragas_relevancy": "Answer Relevancy Score (0.0 to 1.0)"
        },
        color_discrete_map={
            "incorrect": "#e74c3c",
            "partly": "#f39c12",
            "correct": "#2ecc71"
        }
    )
    fig_rel.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_rel, use_container_width=True)

st.divider()

# Correlation Analysis
st.subheader("2. Rank Correlation Analysis with Bootstrap 95% Confidence Intervals")
st.write(
    "Correlations are computed against ordinal human labels (`incorrect`=0, `partly`=1, `correct`=2). "
    "Confidence intervals are generated via non-parametric bootstrapping with 1,000 resamples."
)

corr_faith = compute_correlations(df["ragas_faithfulness"], df["derived_label"], metric_name="RAGAS Faithfulness")
corr_rel = compute_correlations(df["ragas_relevancy"], df["derived_label"], metric_name="RAGAS Answer Relevancy")

corr_table = pd.DataFrame([
    {
        "Metric": corr_faith["metric"],
        "Sample Size (N)": corr_faith["n"],
        "Spearman r_s": corr_faith["spearman"],
        "Spearman 95% CI": f"[{corr_faith['spearman_ci'][0]:.3f}, {corr_faith['spearman_ci'][1]:.3f}]",
        "Spearman p-value": f"{corr_faith['spearman_p']:.4f}" if corr_faith['spearman_p'] is not None else "N/A",
        "Kendall tau-b": corr_faith["kendall"],
        "Kendall 95% CI": f"[{corr_faith['kendall_ci'][0]:.3f}, {corr_faith['kendall_ci'][1]:.3f}]",
        "Kendall p-value": f"{corr_faith['kendall_p']:.4f}" if corr_faith['kendall_p'] is not None else "N/A",
    },
    {
        "Metric": corr_rel["metric"],
        "Sample Size (N)": corr_rel["n"],
        "Spearman r_s": corr_rel["spearman"],
        "Spearman 95% CI": f"[{corr_rel['spearman_ci'][0]:.3f}, {corr_rel['spearman_ci'][1]:.3f}]",
        "Spearman p-value": f"{corr_rel['spearman_p']:.4f}" if corr_rel['spearman_p'] is not None else "N/A",
        "Kendall tau-b": corr_rel["kendall"],
        "Kendall 95% CI": f"[{corr_rel['kendall_ci'][0]:.3f}, {corr_rel['kendall_ci'][1]:.3f}]",
        "Kendall p-value": f"{corr_rel['kendall_p']:.4f}" if corr_rel['kendall_p'] is not None else "N/A",
    }
])

st.dataframe(corr_table, use_container_width=True)

st.divider()

# Interpretive Observations
st.subheader("3. Analytical Takeaways")
st.markdown("""
- **Faithfulness Alignment**: Faithfulness directly penalizes statements that cannot be deduced from the retrieved context. When answers contain hallucinations (particularly evident or multiple spans), the faithfulness score decreases. However, subtle or minor single-span hallucinations often yield high faithfulness scores, blurring the boundary between `correct` and `partly`.
- **Answer Relevancy Behavior**: Answer Relevancy measures whether the generated response addresses the question, not whether its claims are factually faithful. Consequently, fluent hallucinated answers frequently attain near-perfect Relevancy scores, resulting in lower correlation with human truthfulness labels.
- **Confidence Interval Breadth**: With $N=100$, the 95% bootstrap confidence intervals are moderately wide, reflecting sample variance in single-prompt evaluation benchmarks.
""")

st.divider()
st.subheader("Data Explorer")
st.dataframe(
    df[["id", "question", "derived_label", "ragas_faithfulness", "ragas_relevancy", "answer"]],
    use_container_width=True
)
