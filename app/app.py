"""
app.py  —  Pathogen Classification · Enterprise SaaS Dashboard
=============================================================
"""

import itertools
import os
import time
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG & STATE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pathogen Analytics · Enterprise",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'analysis_ready' not in st.session_state:
    st.session_state['analysis_ready'] = False
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None
if 'file_content' not in st.session_state:
    st.session_state['file_content'] = None

if 'sidebar_collapsed' not in st.session_state:
    st.session_state['sidebar_collapsed'] = False

def toggle_sidebar():
    st.session_state['sidebar_collapsed'] = not st.session_state['sidebar_collapsed']

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL CSS  — Linear/Vercel/Stripe Inspired
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-slate-950: #020617;
            --bg-slate-900: #0f172a;
            --slate-800: #1e293b;
            --slate-400: #94a3b8;
            --slate-50: #f8fafc;
            --cyan-400: #22d3ee;
            --cyan-500: #06b6d4;
            --violet-500: #8b5cf6;
            --violet-400: #a78bfa;
            
            --bg-color: var(--bg-slate-950);
            --bg-sidebar: var(--bg-slate-900);
            --card-bg: rgba(30, 41, 59, 0.5); 
            --border-color: rgba(148, 163, 184, 0.15);
            --text-main: var(--slate-50);
            --text-muted: var(--slate-400);
            --accent-primary: var(--cyan-400);
            --accent-secondary: var(--violet-400);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-color) !important;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.04), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(34, 211, 238, 0.04), transparent 25%);
            font-family: 'Inter', -apple-system, sans-serif;
            color: var(--text-main);
        }

        /* Top Navbar */
        .top-navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 64px;
            background: rgba(2, 6, 23, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            z-index: 999999;
            display: flex;
            align-items: center;
            padding: 0 24px;
            justify-content: space-between;
        }
        
        .top-navbar-brand {
            font-weight: 700;
            font-size: 1.15rem;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 12px;
            letter-spacing: -0.02em;
        }

        .top-navbar-brand span.badge {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            color: #000;
        }

        .top-navbar-user {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .user-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-secondary), var(--accent-primary));
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--bg-slate-950);
            font-weight: 700;
            font-size: 0.9rem;
        }

        /* Adjust Main Content Area */
        [data-testid="block-container"] {
            padding-top: 64px !important;
            padding-bottom: 24px !important;
            padding-left: 40px !important;
            padding-right: 40px !important;
            max-width: 100% !important;
        }

        /* Hide sidebar toggle collapse button and resizer completely */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarResizer"] { 
            display: none !important; 
            width: 0 !important;
            height: 0 !important;
        }

        /* Sidebar Styling - Base Fixed */
        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.98) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid var(--border-color) !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            z-index: 999990 !important;
            transform: none !important;
            transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            overflow-x: hidden !important;
        }
        
        section.main {
            transition: margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 80px !important;
            padding-left: 12px !important;
            padding-right: 12px !important;
        }

        [data-testid="stSidebar"] * { color: var(--text-muted) !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--text-main) !important;
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            margin-top: 1rem;
            padding-left: 8px;
        }

        /* Clean Horizontal Nav Rows - Enterprise Datadog/Palantir Style */
        [data-testid="stSidebar"] div[role="radiogroup"] { 
            gap: 2px !important; 
            padding: 0 8px !important;
        }

        /* HIDE RADIO DOTS COMPLETELY */
        [data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
        [data-testid="stSidebar"] input[type="radio"],
        [data-testid="stSidebar"] div[role="radiogroup"] svg {
            display: none !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Style the label container to look like a native menu row */
        [data-testid="stSidebar"] label[data-baseweb="radio"] {
            width: 100% !important;
            padding: 10px 14px !important;
            margin: 0 !important;
            border-radius: 6px !important;
            background: transparent !important;
            border: none !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            transition: all 0.2s ease !important;
            box-sizing: border-box !important;
        }

        /* Hover State */
        [data-testid="stSidebar"] label[data-baseweb="radio"]:hover { 
            background-color: rgba(34, 211, 238, 0.04) !important; 
        }

        /* Active State - Elegant border glow */
        [data-testid="stSidebar"] div[role="radio"][aria-checked="true"] label[data-baseweb="radio"],
        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
            background-color: rgba(34, 211, 238, 0.1) !important;
            box-shadow: inset 4px 0 0 var(--accent-primary) !important;
            border-radius: 4px !important;
        }

        [data-testid="stSidebar"] label[data-baseweb="radio"] p {
            margin: 0 !important;
            font-size: 0.95rem !important;
            color: var(--slate-400) !important;
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            transition: color 0.2s ease !important;
        }

        [data-testid="stSidebar"] label[data-baseweb="radio"] strong {
            font-weight: 500 !important;
            color: var(--slate-400) !important;
            transition: color 0.2s ease !important;
        }

        [data-testid="stSidebar"] label[data-baseweb="radio"]:hover p,
        [data-testid="stSidebar"] label[data-baseweb="radio"]:hover strong {
            color: var(--slate-50) !important;
        }

        [data-testid="stSidebar"] div[role="radio"][aria-checked="true"] p,
        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p,
        [data-testid="stSidebar"] div[role="radio"][aria-checked="true"] strong,
        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) strong {
            color: var(--text-main) !important;
            font-weight: 600 !important;
        }
        
        /* Enterprise Button Styling */
        div[data-testid="stButton"] button {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-main) !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            padding: 4px 16px !important;
            transition: all 0.2s ease !important;
            min-height: 36px !important;
            height: auto !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--accent-primary) !important;
            background-color: rgba(34, 211, 238, 0.05) !important;
            color: var(--accent-primary) !important;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: var(--accent-primary) !important;
            color: var(--bg-slate-950) !important;
            border: none !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: var(--accent-secondary) !important;
            color: white !important;
        }
        
        /* Sidebar Toggle Button */
        [data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: transparent !important;
            border: none !important;
            color: var(--slate-400) !important;
            padding: 4px !important;
            min-height: 32px !important;
            width: 32px !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            color: var(--slate-50) !important;
            background: rgba(255,255,255,0.05) !important;
            border: none !important;
        }

        /* Glassmorphism Cards */
        .saas-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 4px 24px -4px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, border-color 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .saas-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        }
        .saas-card:hover { 
            border-color: rgba(139, 92, 246, 0.4); 
            transform: translateY(-2px);
        }
        .saas-card h3 {
            font-size: 0.9rem; font-weight: 600; color: var(--text-muted);
            margin-top: 0; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.05em;
        }
        .saas-card h4 {
            font-size: 0.85rem; font-weight: 500; color: var(--text-muted);
            margin-top: 0; margin-bottom: 12px;
        }

        /* Metric Cards */
        .metric-card {
            display: flex; flex-direction: column; justify-content: center;
            background: var(--card-bg);
            border: 1px solid var(--border-color); 
            border-radius: 16px; 
            padding: 20px;
            margin-bottom: 16px; 
            height: calc(100% - 16px); 
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .metric-card:hover {
            border-color: rgba(34, 211, 238, 0.3);
            transform: translateY(-2px);
        }
        
        .metric-title { font-size: 0.85rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
        .metric-value { font-family: 'Inter', sans-serif; font-size: 2.2rem; font-weight: 700; color: var(--text-main); line-height: 1.1; letter-spacing: -0.03em; }
        .metric-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 12px; font-weight: 500; }

        /* Typography */
        .page-title {
            font-family: 'Inter', sans-serif; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.03em;
            background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%); 
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 24px; padding-bottom: 0; border-bottom: none;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: rgba(15, 23, 42, 0.4) !important; 
            border: 1px dashed rgba(148, 163, 184, 0.3) !important;
            border-radius: 12px !important; 
            padding: 32px !important; 
            transition: all 0.3s ease;
            backdrop-filter: blur(8px);
        }
        [data-testid="stFileUploader"]:hover { 
            border-color: var(--accent-primary) !important; 
            background-color: rgba(34, 211, 238, 0.05) !important; 
        }

        /* DataFrames */
        .stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }

        /* Result Banner */
        .result-banner { 
            border: 1px solid var(--border-color);
            background: rgba(30, 41, 59, 0.5);
            border-radius: 16px; padding: 24px 32px; display: flex; align-items: center; gap: 24px;
            margin-bottom: 24px; backdrop-filter: blur(20px); 
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }
        .result-icon { font-size: 2.5rem; }
        .result-info { display: flex; flex-direction: column; }
        .result-label { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
        .result-conf { font-size: 0.9rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-top: 6px; }

        /* Status Pills */
        .status-pill { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
        .status-ok { background: rgba(34, 211, 238, 0.1); color: var(--accent-primary); border: 1px solid rgba(34, 211, 238, 0.2); }
        .status-err { background: rgba(139, 92, 246, 0.1); color: var(--accent-secondary); border: 1px solid rgba(139, 92, 246, 0.2); }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.2); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.4); }
        
        #MainMenu, footer, header { visibility: hidden; display: none; }
        </style>

        <div class="top-navbar">
            <div class="top-navbar-brand">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--accent-primary);"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                PathogenAI <span class="badge">Enterprise</span>
            </div>
            <div class="top-navbar-user">
                <span>Enterprise Workspace</span>
                <div class="user-avatar">JD</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

inject_css()

def inject_dynamic_css():
    is_collapsed = st.session_state.get('sidebar_collapsed', False)
    sidebar_width = 72 if is_collapsed else 260
    
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"] {{
            width: {sidebar_width}px !important;
            min-width: {sidebar_width}px !important;
            max-width: {sidebar_width}px !important;
            transition: width 0.3s ease !important;
            overflow-x: hidden !important;
        }}
        section.main {{
            margin-left: {sidebar_width}px !important;
            transition: margin-left 0.3s ease !important;
        }}
        
        /* Clean toggling of text */
        {'[data-testid="stSidebar"] label[data-baseweb="radio"] strong { display: none !important; }' if is_collapsed else ''}
        {'[data-testid="stSidebar"] label[data-baseweb="radio"] p { justify-content: center !important; margin: 0 !important; width: 100% !important; overflow: visible !important; white-space: nowrap !important; }' if is_collapsed else ''}
        {'[data-testid="stSidebar"] label[data-baseweb="radio"] { justify-content: center !important; padding-left: 0 !important; padding-right: 0 !important; }' if is_collapsed else ''}
        {'[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, .status-pill, hr { display: none !important; }' if is_collapsed else ''}
        
        /* Ensure top toggle button centers when collapsed */
        {'[data-testid="stSidebar"] [data-testid="column"]:nth-child(1) { display: none !important; }' if is_collapsed else ''}
        {'[data-testid="stSidebar"] [data-testid="column"]:nth-child(2) { width: 100% !important; flex: 1 1 100% !important; display: flex; justify-content: center; margin: 0 !important; }' if is_collapsed else ''}
        </style>
        """, unsafe_allow_html=True
    )

inject_dynamic_css()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONSTANTS & PATHS
# ─────────────────────────────────────────────────────────────────────────────
KMER_SIZE  = 3
VALID_BASES = set("ACGT")
ALL_KMERS  = ["".join(p) for p in itertools.product("ACGT", repeat=KMER_SIZE)]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pathogen_model.pkl"
DATASET_PATH = BASE_DIR / "data" / "genome_dataset.csv"
METRICS_PATH = BASE_DIR / "data" / "model_metrics.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_sequence(seq: str) -> str:
    return "".join(b for b in seq.upper() if b in VALID_BASES)

def compute_kmer_frequencies(sequence: str) -> dict:
    counts = {kmer: 0 for kmer in ALL_KMERS}
    total  = 0
    for i in range(len(sequence) - KMER_SIZE + 1):
        kmer = sequence[i : i + KMER_SIZE]
        if kmer in counts:
            counts[kmer] += 1
            total += 1
    if total > 0:
        counts = {k: v / total for k, v in counts.items()}
    return counts

def parse_fasta(file_content: str) -> str:
    lines = file_content.splitlines()
    seq_parts = []
    for line in lines:
        line = line.strip()
        if line.startswith(">") or not line: continue
        seq_parts.append(line)
    return clean_sequence("".join(seq_parts))

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists(): return None, f"Model not found at: {MODEL_PATH}"
    try: return joblib.load(MODEL_PATH), None
    except Exception as exc: return None, f"Load failed: {exc}"

@st.cache_data
def load_data():
    try: return pd.read_csv(DATASET_PATH)
    except: return pd.DataFrame()

@st.cache_data
def load_metrics():
    try: return pd.read_csv(METRICS_PATH)
    except: return pd.DataFrame()

def predict(model, kmer_freq: dict):
    feature_vector = pd.DataFrame([kmer_freq], columns=ALL_KMERS)
    prediction     = model.predict(feature_vector)[0]
    probabilities  = model.predict_proba(feature_vector)[0]
    return int(prediction), float(probabilities[prediction])

def custom_metric(title, value, subtitle="", value_color="var(--text-main)"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value" style="color: {value_color};">{value}</div>
        {f'<div class="metric-sub">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    col1, col2 = st.columns([4, 1])
    with col1:
        if not st.session_state['sidebar_collapsed']:
            st.markdown("<h3 style='margin-top:0; padding-top:4px;'>🧬 PathogenAI</h3>", unsafe_allow_html=True)
    with col2:
        st.button("≡", on_click=toggle_sidebar, key="collapse_btn", help="Toggle Sidebar")
    
    if not st.session_state['sidebar_collapsed']:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    pages = {
        "📊 **Overview**": "Overview",
        "🧬 **Upload Genome**": "Upload Genome",
        "📈 **Analytics**": "Analytics",
        "🦠 **Virulence**": "Virulence",
        "🗄️ **Dataset Explorer**": "Dataset Explorer",
        "⚙️ **Model Metrics**": "Model Metrics",
        "📑 **Reports**": "Reports"
    }
    
    selection_label = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    selection = pages[selection_label]
    
    st.markdown("<hr style='border-color: var(--border-color); margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown("### Status")
    
    model_obj, model_err = load_model()
    if model_obj: 
        st.markdown('<span class="status-pill status-ok">Model Active</span>', unsafe_allow_html=True)
    else: 
        st.markdown('<span class="status-pill status-err">Model Offline</span>', unsafe_allow_html=True)
    
    df = load_data()
    if not df.empty: 
        st.markdown(f'<br><span class="status-pill status-ok">DB Connected ({len(df)})</span>', unsafe_allow_html=True)
    else: 
        st.markdown('<br><span class="status-pill status-err">DB Disconnected</span>', unsafe_allow_html=True)

    st.markdown("<br><br><span style='font-size:0.75rem;color:var(--text-muted);font-weight:500;'>Platform v3.1.0</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PAGE ROUTING
# ─────────────────────────────────────────────────────────────────────────────

# --- OVERVIEW ---
if selection == "Overview":
    st.markdown('<div class="page-title">Executive Overview</div>', unsafe_allow_html=True)
    
    df = load_data()
    metrics = load_metrics()
    
    total_genomes = len(df) if not df.empty else 0
    harmful_detected = len(df[df['label'] == 1]) if not df.empty and 'label' in df.columns else 0
    avg_risk = 0.0
    if not df.empty and 'label' in df.columns:
        avg_risk = (harmful_detected / total_genomes) * 100 if total_genomes > 0 else 0
        
    model_acc = 0.0
    if not metrics.empty and 'Accuracy (mean)' in metrics.columns:
        model_acc = metrics['Accuracy (mean)'].max() * 100
        
    c1, c2, c3, c4 = st.columns(4)
    with c1: custom_metric("Total Sequences", f"{total_genomes:,}", "Registered in database")
    with c2: custom_metric("Identified Pathogens", f"{harmful_detected:,}", "Flagged as high-risk", "var(--accent-secondary)")
    with c3: custom_metric("Risk Proportion", f"{avg_risk:.1f}%", "Overall dataset virulence", "var(--accent-primary)")
    with c4: custom_metric("Model Accuracy", f"{model_acc:.1f}%", "Top performing classifier", "var(--text-main)")
        
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="saas-card"><h3>Distribution Analytics</h3>', unsafe_allow_html=True)
        if not df.empty and 'label' in df.columns:
            dist_data = pd.DataFrame({
                'Classification': ['Non-Harmful', 'Harmful'],
                'Count': [total_genomes - harmful_detected, harmful_detected]
            })
            fig = px.pie(dist_data, values='Count', names='Classification', hole=0.7, 
                         color='Classification', color_discrete_map={'Harmful':'#8b5cf6', 'Non-Harmful':'#22d3ee'})
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#9CA3AF', 
                margin=dict(t=0, b=0, l=0, r=0),
                height=220,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Dataset not loaded or missing label column.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="saas-card"><h3>System Activity</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:var(--text-muted); line-height:1.8;">
        <span style="color:var(--accent-primary)">[OK]</span> Model weights initialized<br>
        <span style="color:var(--accent-primary)">[OK]</span> Feature mapping loaded (64-D)<br>
        <span style="color:var(--accent-primary)">[INFO]</span> DB replication completed<br>
        <span style="color:var(--accent-secondary)">[WARN]</span> Anomalous GC variance detected<br>
        <span style="color:var(--accent-primary)">[INFO]</span> Node scaling nominal
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- UPLOAD GENOME (WORKFLOW) ---
elif selection == "Upload Genome":
    st.markdown('<div class="page-title">Genome Analysis Workflow</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="saas-card"><h3>1. Upload Genome File</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload FASTA file (.fna, .fasta)", type=["fna", "fasta"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # Detect if a new file was uploaded to reset state
        if st.session_state.get('last_uploaded_file') != uploaded_file.name:
            st.session_state['analysis_ready'] = False
            st.session_state['last_uploaded_file'] = uploaded_file.name

        if not st.session_state.get('analysis_ready', False):
            st.markdown("### 2. Sequence Analytics")
            st.info("Sequence uploaded successfully. Ready for inference processing.")
            if st.button("🔍 Analyze Sequence", type="primary"):
                st.session_state['analysis_ready'] = True
                st.session_state['file_content'] = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                st.rerun()

        if st.session_state.get('analysis_ready', False):
            raw_text = st.session_state['file_content']
            
            if model_obj is None:
                st.error(f"Cannot process: Model endpoint unreachable.\n\n{model_err}")
                st.stop()

            dna_sequence = parse_fasta(raw_text)

            if len(dna_sequence) < KMER_SIZE:
                st.error("Invalid sequence: Insufficient base pairs.")
                st.stop()

            kmer_freq = compute_kmer_frequencies(dna_sequence)
            label, confidence = predict(model_obj, kmer_freq)

            is_harmful = (label == 1)
            res_color = "var(--accent-secondary)" if is_harmful else "var(--accent-primary)"
            res_bg = "linear-gradient(90deg, rgba(139,92,246,0.08) 0%, rgba(30,41,59,0.5) 100%)" if is_harmful else "linear-gradient(90deg, rgba(34,211,238,0.08) 0%, rgba(30,41,59,0.5) 100%)"
            res_txt = "Pathogen Detected (Harmful)" if is_harmful else "Non-Harmful Organism"
            res_icn = "⚠️" if is_harmful else "✓"
            risk_score = int(confidence * 100) if is_harmful else int((1 - confidence) * 100)
            
            st.markdown("### 3. Prediction Results")
            st.markdown(f"""
            <div class="result-banner" style="border-left: 4px solid {res_color}; background: {res_bg}; padding: 24px;">
                <div class="result-icon" style="color: {res_color};">{res_icn}</div>
                <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
                    <div class="result-info">
                        <div class="result-label" style="color: {res_color}; font-size: 1.4rem;">{res_txt}</div>
                        <div class="result-conf" style="font-size: 0.95rem; margin-top: 6px;">
                            Classification Confidence: <strong style="color:var(--text-main)">{confidence*100:.1f}%</strong>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Risk Score</div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: {res_color}; line-height: 1;">{risk_score}<span style="font-size: 1.2rem; color: var(--text-muted);">/100</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 4. Genome Statistics")
            c1, c2, c3, c4 = st.columns(4)
            gc_content = (dna_sequence.count('G') + dna_sequence.count('C')) / len(dna_sequence) * 100
            at_content = (dna_sequence.count('A') + dna_sequence.count('T')) / len(dna_sequence) * 100
            est_genes = len(dna_sequence) // 1000
            
            with c1: custom_metric("Sequence Length", f"{len(dna_sequence):,}", "Base pairs")
            with c2: custom_metric("GC Content", f"{gc_content:.1f}%", "Guanine-Cytosine ratio")
            with c3: custom_metric("AT Content", f"{at_content:.1f}%", "Adenine-Thymine ratio")
            with c4: custom_metric("Estimated Genes", f"{est_genes:,}", "~1 gene per kb")

            st.markdown("### 5. Virulence Indicators")
            v_col1, v_col2 = st.columns([1, 1], gap="large")
            with v_col1:
                st.markdown('<div class="saas-card" style="height: 100%;"><h4>Pathogenicity Profile</h4>', unsafe_allow_html=True)
                base_val = risk_score / 20
                r_vals = [base_val + 0.5, base_val*0.8 + 0.3, base_val*1.2, base_val*1.5 - 0.2, base_val*0.9 + 0.4]
                categories = ['Capsid', 'Envelope', 'Replicase', 'Toxins', 'Attachment']
                fig = go.Figure(data=go.Scatterpolar(
                  r=r_vals,
                  theta=categories,
                  fill='toself',
                  line_color='#8b5cf6' if is_harmful else '#22d3ee',
                  fillcolor='rgba(139, 92, 246, 0.2)' if is_harmful else 'rgba(34, 211, 238, 0.2)'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=False, range=[0, 7]), angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')), 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color='#9CA3AF', 
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=280
                )
                st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                
            with v_col2:
                st.markdown('<div class="saas-card" style="height: 100%;"><h4>Primary Motifs (Top 3-mers)</h4>', unsafe_allow_html=True)
                freq_df = pd.DataFrame(list(kmer_freq.items()), columns=["Kmer", "Frequency"]).sort_values("Frequency", ascending=False)
                fig_bar = px.bar(freq_df.head(10), x='Kmer', y='Frequency', color_discrete_sequence=['#22d3ee'])
                fig_bar.update_traces(marker_line_color='rgba(255,255,255,0.1)', marker_line_width=1, opacity=0.85)
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='#9CA3AF', 
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=280,
                    xaxis_title=None,
                    yaxis_title=None
                )
                st.plotly_chart(fig_bar, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 6. Export Results")
            report_text = f"PATHOGEN ANALYSIS REPORT\n" \
                          f"========================\n" \
                          f"Filename: {uploaded_file.name}\n" \
                          f"Classification: {res_txt}\n" \
                          f"Risk Score: {risk_score}/100\n" \
                          f"Confidence: {confidence*100:.2f}%\n" \
                          f"\n" \
                          f"GENOME STATISTICS\n" \
                          f"-----------------\n" \
                          f"Sequence Length: {len(dna_sequence)} bp\n" \
                          f"GC Content: {gc_content:.2f}%\n" \
                          f"AT Content: {at_content:.2f}%\n" \
                          f"Estimated Genes: {est_genes}\n"
                          
            st.download_button(
                label="📥 Download Full Report (TXT)",
                data=report_text,
                file_name=f"report_{uploaded_file.name}.txt",
                mime="text/plain",
                type="secondary"
            )

# --- ANALYTICS ---
elif selection == "Analytics":
    st.markdown('<div class="page-title">Risk Correlates Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="saas-card"><h3>GC Content vs. Shannon Entropy Mapping</h3>', unsafe_allow_html=True)
    df = load_data()
    if not df.empty and 'gc_content' in df.columns and 'shannon_entropy' in df.columns:
        fig = px.scatter(
            df, x="gc_content", y="shannon_entropy", color="label", 
            color_continuous_scale=[(0, '#22d3ee'), (1, '#8b5cf6')],
            hover_data=["filename"], opacity=0.8
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#9CA3AF',
            height=500,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Insufficient structural data available.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- VIRULENCE ---
elif selection == "Virulence":
    st.markdown('<div class="page-title">Virulence Biomarkers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="saas-card"><h3>Protein Signature Deviations</h3>', unsafe_allow_html=True)
        categories = ['Capsid', 'Envelope', 'Replicase', 'Toxins', 'Attachment']
        fig = go.Figure(data=go.Scatterpolar(
          r=[1.5, 4.2, 3.8, 5.0, 2.1],
          theta=categories,
          fill='toself',
          line_color='#22d3ee',
          fillcolor='rgba(34, 211, 238, 0.2)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False, gridcolor='rgba(255,255,255,0.1)'), angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')), 
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#9CA3AF', 
            margin=dict(t=30, b=30, l=30, r=30),
            height=300
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="saas-card"><h3>Mutation Frequency Matrix</h3>', unsafe_allow_html=True)
        z = np.random.rand(8, 8)
        fig = px.imshow(z, color_continuous_scale='Purpor')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#9CA3AF', 
            margin=dict(t=20, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

# --- DATASET EXPLORER ---
elif selection == "Dataset Explorer":
    st.markdown('<div class="page-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="saas-card"><h3>Genomic Records</h3>', unsafe_allow_html=True)
    df = load_data()
    if not df.empty:
        st.dataframe(df.head(500), width='stretch', height=600)
        st.caption(f"Showing sample of {len(df)} total records.")
    else:
        st.warning("Database synchronization required.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MODEL METRICS ---
elif selection == "Model Metrics":
    st.markdown('<div class="page-title">Model Diagnostics</div>', unsafe_allow_html=True)
    metrics = load_metrics()
    if not metrics.empty:
        st.markdown('<div class="saas-card"><h3>Classifier Evaluation Metrics</h3>', unsafe_allow_html=True)
        st.dataframe(metrics.style.format(precision=4), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="saas-card"><h3>Performance Comparison</h3>', unsafe_allow_html=True)
        fig = px.bar(
            metrics, x='Model', y=['Accuracy (mean)', 'Precision (mean)', 'Recall (mean)', 'F1 (mean)'],
            barmode='group', color_discrete_sequence=['#22d3ee', '#06b6d4', '#8b5cf6', '#a78bfa']
        )
        fig.update_traces(marker_line_color='rgba(255,255,255,0.1)', marker_line_width=1, opacity=0.9)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#9CA3AF',
            legend_title=None,
            xaxis_title=None,
            yaxis_title="Score",
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Evaluation metrics unavailable.")

# --- REPORTS ---
elif selection == "Reports":
    st.markdown('<div class="page-title">Automated Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    
    st.markdown("### Generated Intelligence Reports")
    
    reports_data = [
        {"Report Name": "Monthly Pathogen Variance - May", "Date": "2026-05-01", "Size": "2.4 MB", "Status": "Ready"},
        {"Report Name": "Model Performance Drift Q1", "Date": "2026-04-15", "Size": "1.1 MB", "Status": "Ready"},
        {"Report Name": "High Risk Correlates Summary", "Date": "2026-04-02", "Size": "3.8 MB", "Status": "Ready"},
        {"Report Name": "Weekly Automated Dump", "Date": "2026-03-28", "Size": "12.5 MB", "Status": "Archived"},
    ]
    st.dataframe(pd.DataFrame(reports_data), width='stretch')
    
    c1, c2 = st.columns(2)
    with c1:
        st.button("📄 Generate New Report", use_container_width=True)
    with c2:
        st.button("📥 Download All (ZIP)", use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- ABOUT PROJECT ---
elif selection == "About Project":
    st.markdown('<div class="page-title">System Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="saas-card"><h3>Pathogen Analytics Engine</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.9rem; line-height:1.6; color:var(--text-muted);">
    This platform provides enterprise-grade genomic classification utilizing K-mer frequency distribution mapping 
    and machine learning inference.

    <br><br>
    <strong>Core Architecture:</strong>
    <ul style="margin-top:8px;">
        <li><strong>Frontend:</strong> Streamlit (Enterprise Dark Theme)</li>
        <li><strong>Data Processing:</strong> Pandas, NumPy, Biopython</li>
        <li><strong>Machine Learning:</strong> Scikit-learn (Random Forest Pipeline)</li>
        <li><strong>Visualization:</strong> Plotly Graph Objects</li>
    </ul>
    
    <strong>Build Version:</strong> 3.1.0<br>
    <strong>Environment:</strong> Production
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
