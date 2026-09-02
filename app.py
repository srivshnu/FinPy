"""
app.py — Application Entry Point.
Single-page Agentic Banking Copilot.
"""
import streamlit as st
import importlib.util, os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="FinPy — AI Banking Copilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(__file__)

def load_page(rel_path, name):
    path = os.path.join(BASE_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

load_page("ui/copilot.py", "copilot").render()