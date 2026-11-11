"""Page 1: Overview mirror page."""

import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui_helpers import render_sidebar
from app import render_overview

st.set_page_config(
    page_title="Page 1: Overview",
    layout="wide",
    initial_sidebar_state="expanded"
)
render_sidebar()
render_overview()
