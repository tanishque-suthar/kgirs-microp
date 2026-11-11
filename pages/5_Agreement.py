"""Page 5: Inter-Annotator Agreement (Mini-IAA).

Computes Pairwise Quadratic-Weighted Cohen's Kappa and Fleiss' Kappa across three raters:
- Rater A: RAGTruth-derived ground truth
- Rater B: User manual blind relabels (from data/my_labels.csv)
- Rater C: Gemini LLM Judge
Strict constraint: NO EMOJIS in UI.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui_helpers import render_sidebar, load_dataset, load_judge_data, load_my_labels, get_annotation_subset
from src.stats import compute_inter_annotator_agreement, ORDERED_CLASSES

st.set_page_config(page_title="Page 5: Agreement", layout="wide")
render_sidebar()

st.title("Page 5: Inter-Annotator Agreement (Mini-IAA)")
st.caption(
    "Exploratory agreement analysis comparing human ground truth (Rater A), "
    "manual blind user annotation (Rater B), and Gemini LLM Judge (Rater C)."
)

st.info(
    "Notice: This is a mini-IAA conducted on the blind relabelling subset (n=35 items). "
    "Because sample size is limited, these kappa values serve as an exploratory diagnostic rather "
    "than a definitive multi-annotator benchmark."
)

sample_df = load_dataset()
judge_df = load_judge_data()
my_labels_df = load_my_labels()

if sample_df is None:
    st.warning("Sample dataset not found. Please run sampling first from the sidebar.")
    st.stop()

# Join datasets on id
subset_df = get_annotation_subset(sample_df, subset_size=35)[["id", "derived_label"]].copy()
subset_df["id"] = subset_df["id"].astype(str)

if judge_df is not None and "judge_verdict" in judge_df.columns:
    j_sub = judge_df[["id", "judge_verdict"]].copy()
    j_sub["id"] = j_sub["id"].astype(str)
    merged = pd.merge(subset_df, j_sub, on="id", how="left")
else:
    merged = subset_df.copy()
    merged["judge_verdict"] = None

if not my_labels_df.empty and "my_label" in my_labels_df.columns:
    m_sub = my_labels_df[["id", "my_label"]].copy()
    m_sub["id"] = m_sub["id"].astype(str)
    merged = pd.merge(merged, m_sub, on="id", how="left")
else:
    merged["my_label"] = None

annotated_count = merged["my_label"].dropna().count()
total_target = len(merged)

st.write(f"User Relabelling Progress: **{annotated_count} of {total_target} items completed**")

if annotated_count < 5:
    st.warning(
        f"Only {annotated_count} manual annotations found in data/my_labels.csv. "
        "Please visit Page 2 (Annotate) to provide at least 5-10 ratings to compute full 3-way agreement."
    )
    # If judge data exists, show 2-way agreement between RAGTruth and Gemini
    if "judge_verdict" in merged.columns and merged["judge_verdict"].dropna().count() >= 5:
        from sklearn.metrics import cohen_kappa_score
        from src.stats import encode_labels
        valid = merged.dropna(subset=["derived_label", "judge_verdict"])
        k2 = cohen_kappa_score(
            encode_labels(valid["derived_label"]),
            encode_labels(valid["judge_verdict"]),
            weights="quadratic"
        )
        st.subheader("Preliminary 2-Rater Agreement (RAGTruth vs Gemini Judge)")
        st.metric("Quadratic Cohen's Kappa (A vs C)", f"{k2:.4f}")
    st.stop()

# Compute 3-way agreement
valid_merged = merged.dropna(subset=["derived_label", "my_label", "judge_verdict"])
iaa = compute_inter_annotator_agreement(
    derived_series=valid_merged["derived_label"],
    my_series=valid_merged["my_label"],
    judge_series=valid_merged["judge_verdict"],
)

if iaa.get("status") != "complete":
    st.warning("Insufficient overlapping annotations to compute 3-way agreement.")
    st.stop()

# Helper for Landis & Koch benchmark scale
def interpret_kappa(val: float) -> str:
    if val < 0.0:
        return "Poor (< 0.0)"
    elif val <= 0.20:
        return "Slight (0.01 - 0.20)"
    elif val <= 0.40:
        return "Fair (0.21 - 0.40)"
    elif val <= 0.60:
        return "Moderate (0.41 - 0.60)"
    elif val <= 0.80:
        return "Substantial (0.61 - 0.80)"
    else:
        return "Almost Perfect (0.81 - 1.00)"

st.subheader("1. Summary Metrics")
c1, c2 = st.columns(2)
with c1:
    st.metric("Overlapping Items Evaluated", f"{iaa['n']} items")
with c2:
    st.metric(
        "Fleiss' Kappa (All 3 Raters)",
        f"{iaa['fleiss_kappa']:.4f}",
        help="Overall chance-adjusted agreement across RAGTruth, User, and Gemini."
    )

st.divider()

# Pairwise Kappa Table
st.subheader("2. Pairwise Quadratic Cohen's Kappa Table")

pair_data = []
for pair_name, k_val in iaa["pairwise"].items():
    pair_data.append({
        "Rater Comparison": pair_name,
        "Quadratic Cohen's Kappa": f"{k_val:.4f}",
        "Interpretation (Landis & Koch)": interpret_kappa(k_val),
    })

st.dataframe(pd.DataFrame(pair_data), use_container_width=True)

st.divider()

# Inspection Table
st.subheader("3. Per-Item Relabeling Comparisons")
st.dataframe(
    valid_merged[["id", "derived_label", "my_label", "judge_verdict"]].rename(columns={
        "derived_label": "Rater A (RAGTruth)",
        "my_label": "Rater B (User)",
        "judge_verdict": "Rater C (Gemini Judge)",
    }),
    use_container_width=True
)
