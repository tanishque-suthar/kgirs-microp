"""Page 2: Annotate - Blind Relabelling Interface.

Presents a blind random subset of 35 answers for manual evaluation.
Hides ground truth and judge verdicts to prevent bias.
Saves progress incrementally to data/my_labels.csv.
Strict constraint: NO EMOJIS in UI.
"""

from datetime import datetime
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MY_LABELS_CSV_PATH, ANNOTATION_SUBSET_SIZE
from src.ui_helpers import render_sidebar, load_dataset, load_my_labels, get_annotation_subset

st.set_page_config(page_title="Page 2: Annotate", layout="wide")
render_sidebar()

st.title("Page 2: Annotate (Blind Relabelling)")
st.caption(
    f"Blind evaluation task: Relabel a random subset of {ANNOTATION_SUBSET_SIZE} answers. "
    "Ground truth labels and automatic judge scores are strictly hidden."
)

df = load_dataset()
if df is None:
    st.warning("Sample dataset not found. Please run sampling first from the sidebar.")
    st.stop()

# Deterministic subset of 35 items
subset_df = get_annotation_subset(df, subset_size=ANNOTATION_SUBSET_SIZE).reset_index(drop=True)
my_labels_df = load_my_labels()

# Map existing labels by id
existing_labels = {}
if not my_labels_df.empty:
    for _, row in my_labels_df.iterrows():
        existing_labels[str(row["id"])] = str(row["my_label"]).strip().lower()

# Calculate completed count
total_items = len(subset_df)
completed_count = sum(1 for _, row in subset_df.iterrows() if str(row["id"]) in existing_labels)
pct_complete = completed_count / total_items if total_items > 0 else 0.0

# Progress Bar
st.progress(pct_complete)
st.write(f"Progress: {completed_count} of {total_items} items annotated ({int(pct_complete * 100)}%)")

# Determine active item index (resume where left off)
if "annotate_index" not in st.session_state:
    # Find first unannotated index
    first_unannotated = 0
    for idx, row in subset_df.iterrows():
        if str(row["id"]) not in existing_labels:
            first_unannotated = idx
            break
    st.session_state.annotate_index = first_unannotated

# Bound index
current_idx = max(0, min(st.session_state.annotate_index, total_items - 1))
st.session_state.annotate_index = current_idx

current_row = subset_df.iloc[current_idx]
item_id = str(current_row["id"])
current_saved_label = existing_labels.get(item_id, None)

st.divider()

# Navigation Header
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col1:
    if st.button("Previous Item", disabled=(current_idx == 0)):
        st.session_state.annotate_index -= 1
        st.rerun()

with nav_col2:
    st.markdown(f"**Item {current_idx + 1} of {total_items}** (ID: `{item_id}`)")

with nav_col3:
    if st.button("Next Item", disabled=(current_idx >= total_items - 1)):
        st.session_state.annotate_index += 1
        st.rerun()

# Item Display
st.markdown("#### Question")
st.info(current_row["question"])

st.markdown("#### Retrieved Passages (Evidence Context)")
with st.container(height=240):
    st.markdown(current_row["retrieved passages"])

st.markdown("#### Generated Answer to Evaluate")
st.code(current_row["answer"], language="text")

# Annotation Controls
st.divider()
st.markdown("#### Your Evaluation")
st.write("Determine whether the generated answer is supported by the retrieved passages above:")

options = ["correct", "partly", "incorrect"]
default_idx = options.index(current_saved_label) if current_saved_label in options else 0

selected_label = st.radio(
    "Select verdict:",
    options=options,
    index=default_idx,
    format_func=lambda x: {
        "correct": "correct (Completely faithful to passages, no factual errors)",
        "partly": "partly (Mostly accurate, but has 1 minor flaw or slight unsupported detail)",
        "incorrect": "incorrect (Major factual contradictions, baseless claims, or multiple errors)"
    }[x],
    key=f"radio_label_{item_id}"
)

save_col1, save_col2 = st.columns([1, 3])
with save_col1:
    if st.button("Save & Next Item", type="primary"):
        # Save to my_labels_df
        timestamp = datetime.now().isoformat()
        existing_labels[item_id] = selected_label

        # Rebuild or update DataFrame
        records = []
        for sid, lbl in existing_labels.items():
            records.append({
                "id": sid,
                "my_label": lbl,
                "timestamp": timestamp if sid == item_id else my_labels_df.loc[my_labels_df["id"] == sid, "timestamp"].values[0] if sid in my_labels_df["id"].values else timestamp
            })

        new_df = pd.DataFrame(records)
        MY_LABELS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(MY_LABELS_CSV_PATH, index=False)

        st.toast(f"Saved Item {item_id} as '{selected_label}'")

        # Advance to next item if not at end
        if current_idx < total_items - 1:
            st.session_state.annotate_index += 1
        st.rerun()

with save_col2:
    if current_saved_label:
        st.write(f"Current saved label for this item: **{current_saved_label}**")
    else:
        st.write("This item has not yet been annotated.")

st.divider()
st.subheader("Your Progress Summary")
if not my_labels_df.empty:
    st.dataframe(my_labels_df, use_container_width=True)
else:
    st.write("No annotations recorded yet.")
