"""Page 4: LLM Judge Evaluation.

Analyzes performance of Gemini as an LLM judge:
- Confusion matrix heatmap and classification metrics (Accuracy, Quadratic Kappa)
- Position bias analysis (Option swap flip rate)
- Verbosity bias analysis (Answer length correlation and Fluff padding experiment)
Strict constraint: NO EMOJIS in UI.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui_helpers import render_sidebar, load_judge_data
from src.stats import compute_judge_metrics, compute_verbosity_bias, ORDERED_CLASSES

st.set_page_config(page_title="Page 4: LLM Judge", layout="wide")
render_sidebar()

st.title("Page 4: LLM-as-a-Judge Evaluation")
st.caption(
    "Assessing direct LLM classification performance, alignment with human ground truth, "
    "and vulnerability to position bias and verbosity bias."
)

df = load_judge_data()
if df is None or "judge_verdict" not in df.columns:
    st.info(
        "LLM Judge data not found. Please click 'Run / Refresh LLM Judge' in the sidebar "
        "to evaluate the 100 sampled answers."
    )
    st.stop()

# Compute core judge metrics
metrics = compute_judge_metrics(df)
verb_results = compute_verbosity_bias(df)

# Overview Metric Cards
st.subheader("1. Core Performance & Bias Summary")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
with m_col2:
    st.metric(
        "Quadratic Cohen's Kappa",
        f"{metrics['cohen_kappa_quadratic']:.3f}",
        help="Penalizes distance between ordinal categories (incorrect vs correct is penalized more than incorrect vs partly)."
    )
with m_col3:
    st.metric("Unweighted Kappa", f"{metrics['cohen_kappa_unweighted']:.3f}")
with m_col4:
    flip_pct = metrics['position_bias_flip_rate'] * 100
    st.metric(
        "Position-Bias Flip Rate",
        f"{flip_pct:.1f}%",
        help="Percentage of answers where reversing rubric option order changed the LLM's classification."
    )

st.divider()

# Confusion Matrix Heatmap
st.subheader("2. Confusion Matrix (Human Ground Truth vs LLM Judge)")

cm = np.array(metrics["confusion_matrix"])

fig_cm = px.imshow(
    cm,
    x=ORDERED_CLASSES,
    y=ORDERED_CLASSES,
    color_continuous_scale="Blues",
    text_auto=True,
    labels=dict(x="LLM Judge Verdict", y="Human Ground Truth", color="Count"),
    title="Confusion Matrix (Rows: Ground Truth, Columns: Judge Verdict)"
)
fig_cm.update_layout(
    xaxis=dict(side="bottom"),
    yaxis=dict(autorange="reversed"),
    margin=dict(l=40, r=40, t=60, b=40)
)
st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# Position Bias Deep Dive
st.subheader("3. Position Bias Analysis")
st.markdown("""
To measure position bias, each answer was evaluated twice with identical criteria, but the rubric options
were presented in swapped orders:
- **Order A (Forward)**: 1. `correct`, 2. `partly`, 3. `incorrect`
- **Order B (Reversed)**: 1. `incorrect`, 2. `partly`, 3. `correct`
""")

flips_df = df[df.get("position_flipped", False) == True]
st.write(f"Total Flips Detected: **{len(flips_df)} of {len(df)} answers ({metrics['position_bias_flip_rate'] * 100:.1f}%)**")

if not flips_df.empty:
    with st.expander("Inspect Flipped Items", expanded=False):
        st.dataframe(
            flips_df[[
                "id", "derived_label",
                "judge_verdict_forward", "judge_reason_forward",
                "judge_verdict_reversed", "judge_reason_reversed"
            ]],
            use_container_width=True
        )

st.divider()

# Verbosity Bias Analysis
st.subheader("4. Verbosity Bias Analysis")
st.markdown("""
Language models frequently exhibit **verbosity bias**—favoring longer, more elaborate responses even when
no additional factual content is provided. We test this through two complementary angles:
1. **Length-vs-Score Correlation**: Checking if answer length correlates with judge score within each ground truth category.
2. **Neutral Fluff Perturbation**: Appending uninformative, neutral fluff to answers in a subset (~20 items) and re-evaluating.
""")

col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown("#### Length vs Score Correlation within Ground Truth Classes")
    by_class = verb_results.get("by_class", {})
    class_rows = []
    for cls in ORDERED_CLASSES:
        info = by_class.get(cls, {})
        class_rows.append({
            "Ground Truth Class": cls,
            "Sample Size (N)": info.get("n", 0),
            "Spearman Correlation (Length vs Score)": info.get("spearman", 0.0),
            "p-value": f"{info.get('p_value', 1.0):.4f}" if "p_value" in info else "N/A"
        })
    st.dataframe(pd.DataFrame(class_rows), use_container_width=True)

    fig_len = px.box(
        df,
        x="derived_label",
        y="answer_length",
        color="judge_verdict",
        category_orders={"derived_label": ORDERED_CLASSES, "judge_verdict": ORDERED_CLASSES},
        title="Answer Length Distribution by Judge Verdict within Class",
        labels={"answer_length": "Answer Length (characters)", "derived_label": "Ground Truth"}
    )
    fig_len.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_len, use_container_width=True)

with col_v2:
    st.markdown("#### Fluff-Padding Perturbation Experiment")
    fluff_stats = verb_results.get("fluff_test")
    if fluff_stats:
        f1, f2, f3 = st.columns(3)
        with f1:
            st.metric("Fluff Items Tested", f"{fluff_stats['n_tested']}")
        with f2:
            st.metric("Mean Score Delta", f"{fluff_stats['mean_delta']:+.3f}")
        with f3:
            st.metric("Verdict Unchanged", f"{fluff_stats['pct_unchanged']:.1f}%")

        f_breakdown = pd.DataFrame([
            {"Outcome": "Score Increased", "Percentage": f"{fluff_stats['pct_increased']:.1f}%"},
            {"Outcome": "Score Unchanged", "Percentage": f"{fluff_stats['pct_unchanged']:.1f}%"},
            {"Outcome": "Score Decreased", "Percentage": f"{fluff_stats['pct_decreased']:.1f}%"},
        ])
        st.dataframe(f_breakdown, use_container_width=True)

        fluff_subset = df[df.get("is_fluff_tested", False) == True]
        with st.expander("Inspect Fluff Perturbation Samples", expanded=False):
            st.dataframe(
                fluff_subset[[
                    "id", "derived_label", "judge_verdict", "judge_verdict_fluff", "fluff_score_delta"
                ]],
                use_container_width=True
            )
    else:
        st.write("Fluff testing was not performed or results are unavailable.")
