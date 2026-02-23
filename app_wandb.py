import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import torch
import textwrap

# Import modules
from data_analyzer import DatasetAnalyzer
from brain import MetaLearner
from engine_stream import DynamicTrainer # Updated Engine
from algorithm_recommender import recommend_algorithms
from sklearn_engine import train_and_package, package_to_joblib_bytes

# === UI CONFIGURATION ===
st.set_page_config(page_title="MetaTune Workspace", page_icon="⚡", layout="wide")

# Custom CSS for WandB Aesthetic
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;600;700&family=Share+Tech+Mono&family=Bebas+Neue&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════
   ROOT TOKENS
═══════════════════════════════════════ */
:root {
  --void: #03040A;
  --deep: #080B14;
  --surface: #0D1220;
  --panel: #111827;
  --border: #1a2540;
  --dna-green: #00FF88;
  --neural-amber: #FFB800;
  --quantum-magenta: #FF006E;
  --bio-cyan: #00D4FF;
  --evolution-purple: #9B5DE5;
  --text-primary: #E8EEF4;
  --text-secondary: #7A8BA0;
  --text-dim: #3D4F66;
  --font-display: 'Bebas Neue', sans-serif;
  --font-tech: 'Chakra Petch', sans-serif;
  --font-mono: 'Share Tech Mono', monospace;
}

/* ═══════════════════════════════════════
   BASE OVERRIDES
═══════════════════════════════════════ */
.stApp { background-color: var(--void) !important; color: var(--text-primary); font-family: var(--font-tech); }
.main .block-container { padding-top: 2rem; max-width: 100%; }

/* ═══════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════ */
::-webkit-scrollbar { width: 2px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--dna-green); }

/* ═══════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════ */
div[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--dna-green) !important;
  border-radius: 0 !important;
  padding: 16px 20px !important;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
  transition: border-left-color 0.3s, transform 0.2s !important;
  font-family: var(--font-mono) !important;
}
div[data-testid="stMetric"]:hover {
  border-left-color: var(--bio-cyan) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="stMetricLabel"] {
  font-family: var(--font-mono) !important;
  font-size: 9px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}
div[data-testid="stMetricValue"] {
  font-family: var(--font-display) !important;
  font-size: 28px !important;
  color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════
   HEADERS
═══════════════════════════════════════ */
h1 { font-family: var(--font-display) !important; font-size: 56px !important; letter-spacing: 4px !important; color: var(--text-primary) !important; }
h2 { font-family: var(--font-display) !important; font-size: 36px !important; letter-spacing: 3px !important; color: var(--dna-green) !important; }
h3 { font-family: var(--font-tech) !important; font-weight: 600 !important; letter-spacing: 2px !important; color: var(--text-secondary) !important; }

/* ═══════════════════════════════════════
   BUTTONS — CINEMATIC
═══════════════════════════════════════ */
.stButton > button {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  background: linear-gradient(90deg, var(--quantum-magenta) 0%, #9B00FF 100%) !important;
  color: var(--text-primary) !important;
  border: none !important;
  border-radius: 0 !important;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%) !important;
  padding: 14px 32px !important;
  transition: all 0.3s !important;
  animation: ignitePulse 3s ease-in-out infinite !important;
  width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 40px rgba(255, 0, 110, 0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ═══════════════════════════════════════
   DOWNLOAD BUTTON
═══════════════════════════════════════ */
.stDownloadButton > button {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  background: transparent !important;
  color: var(--dna-green) !important;
  border: 1px solid var(--dna-green) !important;
  border-radius: 0 !important;
  transition: all 0.3s !important;
}
.stDownloadButton > button:hover {
  background: rgba(0,255,136,0.08) !important;
  box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
}

/* ═══════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════ */
[data-testid="stFileUploader"] {
  background: var(--panel) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 0 !important;
  padding: 16px !important;
  font-family: var(--font-mono) !important;
  transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--dna-green) !important; }

/* ═══════════════════════════════════════
   SELECTBOX
═══════════════════════════════════════ */
[data-testid="stSelectbox"] > div {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  letter-spacing: 1px !important;
}

/* ═══════════════════════════════════════
   INFO / WARNING / SUCCESS BANNERS
═══════════════════════════════════════ */
[data-testid="stAlert"] {
  border-radius: 0 !important;
  border: none !important;
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 1px !important;
}
.stSuccess { border-left: 3px solid var(--dna-green) !important; background: rgba(0,255,136,0.06) !important; }
.stWarning { border-left: 3px solid var(--neural-amber) !important; background: rgba(255,184,0,0.06) !important; }
.stInfo    { border-left: 3px solid var(--bio-cyan) !important; background: rgba(0,212,255,0.06) !important; }
.stError   { border-left: 3px solid var(--quantum-magenta) !important; background: rgba(255,0,110,0.06) !important; }

/* ═══════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--deep) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-tech) !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-family: var(--font-display) !important;
  letter-spacing: 3px !important;
}

/* ═══════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════ */
[data-testid="stProgressBar"] > div {
  background: var(--border) !important;
  border-radius: 0 !important;
  height: 3px !important;
}
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--dna-green), var(--bio-cyan)) !important;
  border-radius: 0 !important;
  box-shadow: 0 0 10px rgba(0,255,136,0.5) !important;
}

/* ═══════════════════════════════════════
   SPINNER
═══════════════════════════════════════ */
[data-testid="stSpinner"] { font-family: var(--font-mono) !important; font-size: 11px !important; letter-spacing: 2px !important; color: var(--dna-green) !important; }

/* ═══════════════════════════════════════
   EXPANDER
═══════════════════════════════════════ */
[data-testid="stExpander"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
}
[data-testid="stExpander"] summary { font-family: var(--font-mono) !important; letter-spacing: 2px !important; }

/* ═══════════════════════════════════════
   INPUT TEXT
═══════════════════════════════════════ */
[data-testid="stTextInput"] input {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--dna-green) !important; box-shadow: 0 0 10px rgba(0,255,136,0.2) !important; }

/* ═══════════════════════════════════════
   ANIMATIONS
═══════════════════════════════════════ */
@keyframes ignitePulse {
  0%, 100% { box-shadow: 0 0 10px rgba(255,0,110,0.3); }
  50% { box-shadow: 0 0 30px rgba(255,0,110,0.7), 0 0 60px rgba(155,0,255,0.3); }
}
@keyframes neuralPulse {
  0%, 100% { box-shadow: 0 0 5px var(--dna-green), 0 0 15px var(--dna-green); }
  50% { box-shadow: 0 0 20px var(--dna-green), 0 0 50px var(--dna-green); }
}
@keyframes scanSweep {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
@keyframes matrixFlicker {
  0%, 100% { opacity: 1; }
  33% { opacity: 0.4; }
  66% { opacity: 0.8; }
}
@keyframes slideUpFadeIn {
  from { transform: translateY(24px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
@keyframes orbitalSpin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
@keyframes orbitalSpinReverse {
  from { transform: rotate(0deg); }
  to { transform: rotate(-360deg); }
}
@keyframes heartbeat {
  0%, 100% { transform: scale(1); }
  14% { transform: scale(1.3); }
  28% { transform: scale(1); }
  42% { transform: scale(1.3); }
}
@keyframes borderTrace {
  0% { background-position: 0% 0%; }
  100% { background-position: 200% 0%; }
}
@keyframes barFillAnim {
  from { width: 0; }
}
@keyframes dataPacketFlow {
  0% { left: -10px; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { left: calc(100% + 10px); opacity: 0; }
}
@keyframes glitchText {
  0%, 90%, 100% { text-shadow: none; clip-path: none; }
  92% { text-shadow: -2px 0 var(--quantum-magenta), 2px 0 var(--bio-cyan); clip-path: inset(10% 0 85% 0); }
  94% { text-shadow: 2px 0 var(--neural-amber), -2px 0 var(--dna-green); clip-path: inset(50% 0 30% 0); }
  96% { clip-path: none; text-shadow: none; }
}
@keyframes numberRoll {
  from { transform: translateY(-20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# === SIDEBAR: PROJECT CONFIG ===
with st.sidebar:
    # ── BRAND HEADER ──
    st.markdown("""
    <div style="padding: 24px 0 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px;">
      <div style="font-family:var(--font-display); font-size:32px; letter-spacing:4px; color:var(--dna-green); text-shadow:0 0 20px rgba(0,255,136,0.4);">METATUNE</div>
      <div style="font-family:var(--font-mono); font-size:9px; letter-spacing:4px; color:var(--text-dim); margin-top:4px;">AUTOMATIC HYPERPARAMETER OPTIMIZATION</div>
      <div style="
        height:1px;
        background: linear-gradient(90deg, var(--dna-green), transparent);
        margin-top:16px;
        box-shadow: 0 0 8px rgba(0,255,136,0.3);
      "></div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATA SECTION HEADER ──
    st.markdown("""
    <div style="font-family:var(--font-mono); font-size:9px; letter-spacing:4px; color:var(--text-dim); text-transform:uppercase; margin-bottom:12px;">
      ◈ DATA INGESTION
    </div>
    """, unsafe_allow_html=True)

    # ── KEEP ORIGINAL WIDGET (DO NOT REMOVE) ──
    uploaded_file = st.file_uploader("DROP CSV DATASET", type=['csv'])

    # ── KEEP ORIGINAL WIDGET (DO NOT REMOVE) ──
    target_col = st.text_input("TARGET COLUMN (optional)", help="Leave empty for auto-detection")

    # ── SYSTEM STATUS ──
    st.markdown("""
    <div style="
      margin-top: 32px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      font-family: var(--font-mono);
      font-size: 9px;
      letter-spacing: 4px;
      color: var(--text-dim);
      text-transform: uppercase;
      margin-bottom: 12px;
    ">◈ SYSTEM STATUS</div>
    """, unsafe_allow_html=True)

    # ── STATUS INDICATOR (keep original reference) ──
    status_indicator = st.empty()
    status_indicator.markdown("""
    <div style="
      font-family:var(--font-mono); font-size:10px; letter-spacing:2px;
      color:var(--dna-green); padding:10px 0;
      display:flex; align-items:center; gap:8px;
    ">
      <span style="
        width:6px; height:6px; background:var(--dna-green);
        border-radius:50%; display:inline-block;
        box-shadow:0 0 8px var(--dna-green);
        animation: heartbeat 2s infinite;
      "></span>
      SYSTEM IDLE — AWAITING DATA
    </div>
    """, unsafe_allow_html=True)

# === MAIN WORKSPACE ===
st.markdown("""
<div style="
  padding: 48px 0 32px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
  position: relative;
  overflow: hidden;
">
  <!-- Scan line effect -->
  <div style="
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(0,255,136,0.03) 50%, transparent 100%);
    animation: scanSweep 4s linear infinite;
    pointer-events: none;
  "></div>

  <div style="
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 6px;
    color: var(--dna-green);
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 16px;
  ">
    <span style="display:inline-block; width:40px; height:1px; background:var(--dna-green); box-shadow:0 0 8px var(--dna-green);"></span>
    BILEVEL OPTIMIZATION ENGINE — DATASET INTELLIGENCE SYSTEM
  </div>

  <div style="
    font-family: var(--font-display);
    font-size: clamp(48px, 6vw, 96px);
    line-height: 0.92;
    letter-spacing: 2px;
    color: var(--text-primary);
    animation: glitchText 8s infinite;
  ">
    META<span style="color: var(--dna-green); text-shadow: 0 0 40px rgba(0,255,136,0.4);">TUNE</span>
  </div>

  <div style="
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--text-dim);
    margin-top: 12px;
    text-transform: uppercase;
  ">
    EVERY DATASET HAS A DNA — WE READ IT, PRESCRIBE IT, EVOLVE IT
  </div>
</div>
""", unsafe_allow_html=True)

if uploaded_file:
    # Save temp file
    with open("temp.csv", "wb") as f: f.write(uploaded_file.getbuffer())
    
    # 1. ANALYSIS ROW
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # ── PHASE HEADER ──
        st.markdown("""
        <div style="margin-bottom:24px;">
          <div style="
            display:flex; align-items:center; gap:12px;
            font-family:var(--font-mono); font-size:9px;
            letter-spacing:4px; color:var(--dna-green);
            text-transform:uppercase; margin-bottom:16px;
          ">
            <div style="
              width:28px; height:28px; border:1px solid var(--dna-green);
              border-radius:50%; display:flex; align-items:center; justify-content:center;
              font-family:var(--font-display); font-size:14px;
            ">1</div>
            FORENSIC DNA SCAN
          </div>
          <div style="font-family:var(--font-display); font-size:32px; letter-spacing:2px; color:var(--text-primary);">
            DATASET <span style="color:var(--dna-green);">GENOME</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("Sequencing Genome..."):
            analyzer = DatasetAnalyzer("temp.csv", target_col if target_col else None)
            analyzer.load_data()
            dna = analyzer.analyze()
        
        # Radar Chart for DNA
        categories = ['Skewness', 'Entropy', 'Sparsity', 'Imbalance', 'Dimensionality']
        values = [
            min(dna['mean_skewness'], 5)/5, 
            min(dna['target_entropy'], 2)/2,
            dna['sparsity'],
            min(dna['class_imbalance_ratio'], 10)/10,
            min(dna['dimensionality'], 1)
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            line=dict(color='#00FF88', width=2),
            fillcolor='rgba(0, 255, 136, 0.08)',
            marker=dict(color='#00FF88', size=6)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    showticklabels=False,
                    gridcolor='rgba(26,37,64,0.8)',
                    linecolor='rgba(26,37,64,0.8)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(26,37,64,0.8)',
                    linecolor='rgba(26,37,64,0.5)',
                    tickfont=dict(family='Share Tech Mono', size=10, color='#7A8BA0')
                ),
                bgcolor='rgba(8,11,20,0.0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Share Tech Mono', color='#7A8BA0'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=240,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── DNA METRIC BARS (add AFTER radar, BEFORE existing metrics) ──
        st.markdown(f"""
        <div style="
          background: var(--panel);
          border: 1px solid var(--border);
          padding: 16px 20px;
          margin-top: 8px;
          clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
        ">
          <div style="font-family:var(--font-mono); font-size:8px; letter-spacing:4px; color:var(--text-dim); margin-bottom:14px; text-transform:uppercase;">GENOME READOUT</div>

          {''.join([
            f'''<div style="margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:9px; letter-spacing:1px; color:var(--text-dim); margin-bottom:4px; text-transform:uppercase;">
                <span>{name}</span><span style="color:var(--text-secondary);">{val:.4f}</span>
              </div>
              <div style="background:rgba(26,37,64,0.6); height:2px; border-radius:1px; overflow:hidden;">
                <div style="height:100%; width:{int(pct*100)}%; background:linear-gradient(90deg,{color1},{color2}); animation:barFillAnim 1.2s ease-out forwards;"></div>
              </div>
            </div>'''
            for name, val, pct, color1, color2 in [
              ('TARGET ENTROPY',    dna.get('target_entropy',0),         min(dna.get('target_entropy',0)/2,1),        '#00FF88','#00D4FF'),
              ('SPARSITY',          dna.get('sparsity',0),               min(dna.get('sparsity',0),1),                '#FFB800','#FF006E'),
              ('IMBALANCE RATIO',   dna.get('class_imbalance_ratio',0),  min(dna.get('class_imbalance_ratio',0)/10,1),'#FF006E','#9B5DE5'),
              ('DIMENSIONALITY',    dna.get('dimensionality',0),         min(dna.get('dimensionality',0),1),          '#00D4FF','#00FF88'),
              ('TASK DIFFICULTY',   dna.get('task_difficulty_score',0),  min(dna.get('task_difficulty_score',0)/3,1), '#9B5DE5','#FF006E'),
            ]
          ])}

          <div style="
            margin-top:14px; padding-top:12px; border-top:1px solid var(--border);
            font-family:var(--font-mono); font-size:9px; letter-spacing:2px;
            color:{'var(--dna-green)' if dna.get('task_type')=='classification' else 'var(--neural-amber)'};
            text-transform:uppercase;
          ">
            ◈ TASK TYPE: {dna.get('task_type','UNKNOWN').upper()}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            INSTANCES: {dna.get('n_instances',0):,}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            FEATURES: {dna.get('n_features',0)}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="margin-bottom:20px;">
          <div style="
            display:flex; align-items:center; gap:12px;
            font-family:var(--font-mono); font-size:9px;
            letter-spacing:4px; color:var(--neural-amber);
            text-transform:uppercase; margin-bottom:16px;
          ">
            <div style="
              width:28px; height:28px; border:1px solid var(--neural-amber);
              border-radius:50%; display:flex; align-items:center; justify-content:center;
              font-family:var(--font-display); font-size:14px; color:var(--neural-amber);
            ">2</div>
            NEURAL PRESCRIPTION ENGINE
          </div>
          <div style="font-family:var(--font-display); font-size:28px; letter-spacing:2px; color:var(--text-primary);">
            META-BRAIN <span style="color:var(--neural-amber);">PREDICTION</span>
          </div>
          <div style="height:1px; background:linear-gradient(90deg, var(--neural-amber), transparent); margin-top:12px; box-shadow:0 0 8px rgba(255,184,0,0.3);"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # === VIZIER WORKFLOW INTEGRATION ===
        from vizier_stub import Study, Measurement
        from designer_brain import MetaLearningDesigner
        
        # 1. Create Designer & Study
        designer = MetaLearningDesigner(dna)
        study = Study(designer=designer, study_id="meta_tune_session")
        
        # 2. Get Suggestions (Trials)
        # In a real loop, we might ask for multiple, but here we do 1 for the demo
        trials = study.suggest(count=1)
        active_trial = trials[0]
        params = active_trial.parameters
        
        # ── CINEMATIC METRIC DISPLAY ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TRIAL", f"#{active_trial.id}")
        m2.metric("LEARNING RATE", f"{params['learning_rate']:.2e}")
        m3.metric("L2 REG", f"{params['weight_decay_l2']:.2e}")
        m4.metric("OPTIMIZER", params['optimizer_type'].upper())
        
        # ── DESIGNER INSIGHT BANNER ──
        entropy_val = dna.get('target_entropy', 0)
        regime = 'HIGH ENTROPY DETECTED' if entropy_val > 1.0 else 'STANDARD PROFILE'
        accent = 'var(--quantum-magenta)' if entropy_val > 1.0 else 'var(--dna-green)'
        st.markdown(f"""
        <div style="
          background: rgba(8,11,20,0.8);
          border: 1px solid var(--border);
          border-left: 3px solid {accent};
          padding: 14px 18px;
          margin: 12px 0;
          font-family: var(--font-mono);
          font-size: 11px;
          letter-spacing: 1px;
          color: var(--text-secondary);
          clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
        ">
          <span style="color:{accent}; font-weight:700;">◈ BRAIN SIGNAL: {regime}</span><br>
          Entropy Ω = {entropy_val:.3f} →
          {'Enforcing stricter regularization. High variance in target distribution detected.' if entropy_val > 1.0 else 'Stability confirmed. Standard optimization protocols engaged.'}
          Trial #{active_trial.id} parameters generated.
        </div>
        """, unsafe_allow_html=True)

        # ── ALGORITHM SECTION HEADER ──
        st.markdown("""
        <div style="
          font-family:var(--font-mono); font-size:9px;
          letter-spacing:4px; color:var(--text-dim);
          text-transform:uppercase; margin:20px 0 12px 0;
          display:flex; align-items:center; gap:12px;
        ">
          <span style="flex:1; height:1px; background:var(--border);"></span>
          ◈ ALGORITHM SELECTION
          <span style="flex:1; height:1px; background:var(--border);"></span>
        </div>
        """, unsafe_allow_html=True)
        raw = recommend_algorithms(dna, params)
        recommendations = raw["recommendations"] if isinstance(raw, dict) else raw
        algo_label_to_id = {c["label"]: c["id"] for c in recommendations}
        default_algo_label = recommendations[0]["label"] if recommendations else "Neural Network (PyTorch)"
        selected_algo_label = st.selectbox(
            "Select algorithm to train/deploy",
            options=list(algo_label_to_id.keys()) if recommendations else [default_algo_label],
            index=0,
        )
        selected_algorithm_id = algo_label_to_id.get(selected_algo_label, "pytorch_mlp")
        selected_reason = next((c["reason"] for c in recommendations if c["id"] == selected_algorithm_id), "")
        if selected_reason:
            st.markdown(f"""
            <div style="
              background:var(--panel); border:1px solid var(--border);
              border-left:3px solid var(--bio-cyan);
              padding:12px 16px; margin-top:8px;
              font-family:var(--font-mono); font-size:10px; line-height:1.8;
              color:var(--text-secondary);
              clip-path: polygon(0 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%);
            ">
              <span style="color:var(--bio-cyan); letter-spacing:2px;">WHY:</span> {selected_reason}
            </div>
            """, unsafe_allow_html=True)

        # ── DEPLOY PATH INDICATOR ──
        if selected_algorithm_id != "pytorch_mlp":
            st.markdown("""
            <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--dna-green);margin-top:8px;display:flex;align-items:center;gap:8px;">
              <span style="width:6px;height:6px;background:var(--dna-green);border-radius:50%;box-shadow:0 0 8px var(--dna-green);display:inline-block;"></span>
              DEPLOYABLE PATH — .JOBLIB PACKAGE OUTPUT
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--neural-amber);margin-top:8px;display:flex;align-items:center;gap:8px;">
              <span style="width:6px;height:6px;background:var(--neural-amber);border-radius:50%;box-shadow:0 0 8px var(--neural-amber);display:inline-block;animation:heartbeat 2s infinite;"></span>
              PYTORCH PATH — .PTH WEIGHTS OUTPUT
            </div>
            """, unsafe_allow_html=True)

    # 2. TRAINING ROW (THE LIVE PART)
    st.markdown("---")
    st.markdown("""
<div style="
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  padding: 32px 0;
  margin: 40px 0;
  text-align: center;
">
  <div style="
    font-family:var(--font-display); font-size:48px; letter-spacing:3px;
    color:var(--text-dim); margin-bottom:8px;
    animation:glitchText 6s infinite;
  ">PHASE 03</div>
  <div style="
    font-family:var(--font-mono); font-size:10px; letter-spacing:5px;
    color:var(--text-dim); text-transform:uppercase; margin-bottom:24px;
  ">BILEVEL EVOLUTION ENGINE — AWAITING IGNITION</div>
</div>
""", unsafe_allow_html=True)

    col_btn, col_txt = st.columns([1, 4])
    with col_btn:
        start_btn = st.button("⌬ IGNITE ENGINE")   # ← only label change, same variable name
    with col_txt:
        st.markdown(f"""
        <div style="
          padding: 14px 20px;
          font-family:var(--font-mono); font-size:10px; letter-spacing:2px;
          color:var(--text-dim); line-height:1.8;
        ">
          ALGORITHM: <span style="color:var(--text-primary);">{selected_algo_label.upper()}</span>
          &nbsp;·&nbsp;
          TRIAL: <span style="color:var(--neural-amber);">#{active_trial.id}</span>
          &nbsp;·&nbsp;
          PATH: <span style="color:{'var(--dna-green)' if selected_algorithm_id != 'pytorch_mlp' else 'var(--neural-amber)'}">
            {'SKLEARN/JOBLIB' if selected_algorithm_id != 'pytorch_mlp' else 'PYTORCH/PTH'}
          </span>
        </div>
        """, unsafe_allow_html=True)
    
    if start_btn:
        if selected_algorithm_id != "pytorch_mlp":
            status_indicator.markdown(f"""
<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--neural-amber);padding:8px 0;display:flex;align-items:center;gap:8px;">
  <span style="width:6px;height:6px;background:var(--neural-amber);border-radius:50%;box-shadow:0 0 8px var(--neural-amber);display:inline-block;animation:heartbeat 1.5s infinite;"></span>
  TRAINING TRIAL #{active_trial.id}...
</div>
""", unsafe_allow_html=True)
            # ── PRE-TRAINING LAUNCH SEQUENCE ──────────────────────────────────
            phase_cards_html = ""
            for i, (em, label) in enumerate([('🧬','DATA PREP'),('🏗️','BUILDING'),('🎯','FITTING'),('📦','PACKAGING')]):
                bg = '#00FFFF11' if i == 0 else '#ffffff05'
                border = '#00FFFF' if i == 0 else '#333'
                color = '#00FFFF' if i == 0 else '#444'
                anim = 'animation: neuralPulse 2s infinite;' if i == 0 else ''
                phase_cards_html += f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;padding:12px;text-align:center;{anim}">
    <div style="font-size:20px;margin-bottom:4px;">{em}</div>
    <div style="color:{color};font-size:10px;letter-spacing:1px;">{label}</div>
</div>"""

            launch_container = st.empty()
            launch_container.markdown(f"""
<div style="
  background: linear-gradient(135deg, var(--void) 0%, var(--deep) 100%);
  border: 1px solid rgba(0,255,136,0.2);
  padding: 40px;
  position: relative;
  overflow: hidden;
  clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 0 100%);
  font-family: var(--font-mono);
">
  <!-- Animated scan line -->
  <div style="
    position:absolute; top:0; left:0; right:0; bottom:0;
    background:linear-gradient(90deg,transparent 0%,rgba(0,255,136,0.04) 50%,transparent 100%);
    animation:scanSweep 2s linear infinite; pointer-events:none;
  "></div>

  <!-- Header row -->
  <div style="display:flex; align-items:center; gap:20px; margin-bottom:32px;">
    <!-- Orbital spinner -->
    <div style="width:52px; height:52px; position:relative; flex-shrink:0; display:flex; align-items:center; justify-content:center;">
      <div style="position:absolute; inset:0; border:2px solid var(--dna-green); border-top-color:transparent; border-radius:50%; animation:orbitalSpin 1s linear infinite;"></div>
      <div style="position:absolute; inset:8px; border:2px solid rgba(0,255,136,0.3); border-bottom-color:transparent; border-radius:50%; animation:orbitalSpinReverse 0.7s linear infinite;"></div>
      <span style="font-size:16px; position:relative; z-index:1;">⌬</span>
    </div>
    <div>
      <div style="color:var(--dna-green); font-size:14px; font-weight:700; letter-spacing:4px; text-transform:uppercase; animation:glitchText 4s infinite;">NEURAL ENGINE IGNITED</div>
      <div style="color:var(--text-dim); font-size:9px; letter-spacing:3px; margin-top:4px;">
        TRIAL #{active_trial.id} · {selected_algo_label.upper()} · DEPLOYABLE SKLEARN PATH
      </div>
    </div>
    <div style="margin-left:auto; display:flex; align-items:center; gap:8px;">
      <span style="width:8px; height:8px; background:#00FF41; border-radius:50%; display:inline-block; box-shadow:0 0 10px #00FF41; animation:heartbeat 1.5s infinite;"></span>
      <span style="color:#00FF41; font-size:9px; letter-spacing:3px;">LIVE</span>
    </div>
  </div>

  <!-- Phase sequence -->
  <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:32px;">
    {''.join([
      f'''<div style="
        background:{'rgba(0,255,136,0.08)' if i==0 else 'rgba(255,255,255,0.02)'};
        border:1px solid {'var(--dna-green)' if i==0 else 'var(--border)'};
        padding:14px; text-align:center;
        {'animation:neuralPulse 2s infinite;' if i==0 else ''}
      ">
        <div style="font-size:18px; margin-bottom:6px;">{em}</div>
        <div style="color:{'var(--dna-green)' if i==0 else 'var(--text-dim)'}; font-size:8px; letter-spacing:2px;">{label}</div>
      </div>'''
      for i,(em,label) in enumerate([('🧬','DATA PREP'),('🏗️','BUILDING'),('🎯','FITTING'),('📦','PACKAGING')])
    ])}
  </div>

  <!-- Animated progress bar -->
  <div style="background:rgba(26,37,64,0.6); height:4px; border-radius:2px; overflow:hidden; margin-bottom:20px;">
    <div style="height:100%; background:linear-gradient(90deg,var(--dna-green),var(--bio-cyan),var(--dna-green)); background-size:200% 100%; animation:borderTrace 1.5s linear infinite;"></div>
  </div>

  <!-- Terminal readout -->
  <div style="background:rgba(0,0,0,0.4); border:1px solid rgba(26,37,64,0.6); padding:14px; font-size:11px; color:#00FF41; line-height:2;">
    <div style="animation:matrixFlicker 0.5s infinite;">▶ Initializing preprocessing pipeline...</div>
    <div style="animation:matrixFlicker 0.5s 0.15s infinite; opacity:0.8;">▶ Splitting train/validation (80/20)...</div>
    <div style="color:var(--neural-amber); animation:matrixFlicker 0.6s 0.3s infinite;">◈ Anti-overfitting regularization: ACTIVE</div>
    <div style="animation:matrixFlicker 0.5s 0.45s infinite; opacity:0.6;">▶ Fitting {selected_algo_label.upper()} estimator...</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── ACTUAL TRAINING (runs while animation shows) ───────────────────
            with st.spinner(""):
                package, training_results = train_and_package(
                    data_path="temp.csv",
                    dna=dna,
                    algorithm_id=selected_algorithm_id,
                    target_col=(target_col if target_col else None),
                    hyperparameters=params,
                )
                payload = package_to_joblib_bytes(package)

            # ── CLEAR LAUNCH ANIMATION ─────────────────────────────────────────
            launch_container.empty()

            if (training_results is not None
                    and training_results.get('final_metric') is not None):

                final_metric   = training_results.get('final_metric', 0.0)
                metric_name    = training_results.get('metric_name', 'Metric')
                training_time  = training_results.get('training_time', 0.0)

                score_pct = final_metric * 100
                score_color = '#00FF41' if score_pct >= 85 else ('#FFB800' if score_pct >= 70 else '#FF006E')
                score_label = 'EXCELLENT' if score_pct >= 85 else ('GOOD' if score_pct >= 70 else 'DEVELOPING')

                st.markdown(f"""
<div style="
  background:linear-gradient(135deg,var(--void),var(--deep));
  border:1px solid rgba({('0,255,65' if score_pct>=85 else '255,184,0' if score_pct>=70 else '255,0,110')},0.25);
  padding:40px;
  margin:16px 0;
  position:relative;
  overflow:hidden;
  clip-path:polygon(0 0,calc(100% - 24px) 0,100% 24px,100% 100%,0 100%);
  animation:slideUpFadeIn 0.6s ease-out;
">
  <!-- Glow backdrop -->
  <div style="position:absolute;top:-80px;left:50%;width:400px;height:400px;background:radial-gradient(circle,{score_color}0A 0%,transparent 70%);transform:translateX(-50%);pointer-events:none;"></div>

  <!-- Status bar -->
  <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:32px; flex-wrap:wrap; gap:20px;">
    <div>
      <div style="font-family:var(--font-mono);color:{score_color};font-size:9px;letter-spacing:5px;margin-bottom:6px;text-transform:uppercase;">
        TRAINING COMPLETE · TRIAL #{active_trial.id} · {('SKLEARN' if selected_algorithm_id!='pytorch_mlp' else 'PYTORCH')} ENGINE
      </div>
      <div style="font-family:var(--font-display);font-size:36px;letter-spacing:2px;color:var(--text-primary);">{selected_algo_label.upper()}</div>
    </div>
    <!-- Score ring -->
    <div style="width:88px;height:88px;position:relative;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
      <div style="position:absolute;inset:0;border:2px solid {score_color};border-right-color:transparent;border-radius:50%;animation:orbitalSpin 2s linear infinite;"></div>
      <div style="position:absolute;inset:10px;border:1px solid rgba({('0,255,65' if score_pct>=85 else '255,184,0' if score_pct>=70 else '255,0,110')},0.3);border-left-color:transparent;border-radius:50%;animation:orbitalSpinReverse 3s linear infinite;"></div>
      <div style="text-align:center;">
        <div style="font-family:var(--font-display);font-size:20px;color:{score_color};animation:numberRoll 0.5s ease-out;">{score_pct:.1f}%</div>
        <div style="font-family:var(--font-mono);font-size:7px;color:var(--text-dim);letter-spacing:1px;">{score_label}</div>
      </div>
    </div>
  </div>

  <!-- 3-column metric grid -->
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:28px;">
    {''.join([
      f'''<div style="
        background:var(--panel);border:1px solid {bc}33;
        border-left:3px solid {bc};padding:20px;text-align:center;
        clip-path:polygon(0 0,calc(100% - 10px) 0,100% 10px,100% 100%,0 100%);
      ">
        <div style="font-family:var(--font-mono);color:var(--text-dim);font-size:8px;letter-spacing:3px;margin-bottom:10px;text-transform:uppercase;">{lb}</div>
        <div style="font-family:var(--font-display);font-size:28px;color:{bc};">{vl}</div>
        <div style="font-family:var(--font-mono);font-size:8px;color:var(--text-dim);margin-top:6px;letter-spacing:1px;">{sl}</div>
      </div>'''
      for lb,vl,bc,sl in [
        (metric_name.upper(),        f'{final_metric:.4f}',    score_color,             'PRIMARY OBJECTIVE'),
        ('TRAIN TIME',               f'{training_time:.2f}s',  'var(--evolution-purple)','WALL CLOCK'),
        ('ALGORITHM',                selected_algorithm_id.upper(), 'var(--bio-cyan)',   'DEPLOYABLE ✓'),
      ]
    ])}
  </div>

  <!-- Performance bar -->
  <div style="margin-bottom:24px;">
    <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--text-dim);margin-bottom:8px;">
      <span>MODEL PERFORMANCE INDEX</span>
      <span style="color:{score_color};">{score_pct:.1f}%</span>
    </div>
    <div style="background:rgba(26,37,64,0.6);height:6px;border-radius:3px;overflow:hidden;">
      <div style="height:100%;width:{min(score_pct,100):.0f}%;background:linear-gradient(90deg,{score_color},{score_color}AA);border-radius:3px;box-shadow:0 0 12px {score_color}66;transition:width 1s ease-out;"></div>
    </div>
  </div>

  <!-- Hyperparameter readout -->
  <div style="background:rgba(0,0,0,0.3);border:1px solid var(--border);padding:16px;">
    <div style="font-family:var(--font-mono);font-size:8px;letter-spacing:4px;color:var(--text-dim);margin-bottom:12px;text-transform:uppercase;">HYPERPARAMETER CONFIGURATION — TRIAL #{active_trial.id}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;">
      {''.join([
        f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(26,37,64,0.4);">
          <span style="font-family:var(--font-mono);color:var(--text-dim);font-size:9px;letter-spacing:1px;">{str(k).upper()[:14]}</span>
          <span style="font-family:var(--font-mono);color:var(--bio-cyan);font-size:10px;font-weight:700;">{''.join([f'{v:.4f}' if isinstance(v,float) else str(v)])}</span>
        </div>'''
        for k,v in list(params.items())[:6]
      ])}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                # ── VIZIER TRIAL TRACKING (keep existing logic) ────────────────
                if 'study' not in st.session_state:
                    from vizier_stub import Study
                    st.session_state['study'] = Study(name="metatune_session")
                from vizier_stub import Trial
                _run_id = len(st.session_state['study'].get_trials())
                _trial = Trial(id=_run_id, parameters=params)
                _trial.complete(
                    metric_value=training_results.get('final_metric', 0.0),
                    elapsed_secs=training_results.get('training_time', 0.0)
                )
                st.session_state['study'].add_trial(_trial)
                active_trial.complete(
                    metric_value=training_results.get('final_metric', 0.0),
                    elapsed_secs=training_results.get('training_time', 0.0)
                )
                designer.update(active_trial, study.trials)
                status_indicator.success(f"Trial #{active_trial.id} Completed")

                # ── BEST RUN PANEL (animated version) ─────────────────────────
                if 'study' in st.session_state:
                    _optimal = st.session_state['study'].optimal_trials()
                    if _optimal:
                        _best = _optimal[0]
                        st.markdown(f"""
    <div style="
      background:linear-gradient(135deg,rgba(0,255,65,0.04),var(--void));
      border:1px solid rgba(0,255,65,0.2);
      border-left:4px solid #00FF41;
      padding:20px 24px;
      margin-top:16px;
      display:flex; align-items:center; gap:20px;
      clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%);
    ">
      <div style="font-size:32px;animation:heartbeat 2s infinite;flex-shrink:0;">🏆</div>
      <div>
        <div style="font-family:var(--font-mono);color:#00FF41;font-size:9px;letter-spacing:4px;text-transform:uppercase;margin-bottom:4px;">PERSONAL BEST · TRIAL #{_best.id}</div>
        <div style="font-family:var(--font-display);font-size:32px;color:var(--text-primary);">{_best.final_measurement:.4f}</div>
        <div style="font-family:var(--font-mono);color:var(--text-dim);font-size:9px;letter-spacing:1px;margin-top:4px;">{_best.elapsed_secs:.1f}s training time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
                        with st.expander("◈ BEST HYPERPARAMETERS"):
                            st.json(_best.parameters)

                # ── DOWNLOAD BUTTON ────────────────────────────────────────────
                st.download_button(
                    label="📦 Download Deployable Package (.joblib)",
                    data=payload,
                    file_name=f"metatune_{selected_algorithm_id}_package.joblib",
                    mime="application/octet-stream",
                )

            else:
                st.markdown("""
<div style="
    background: #1A0A0A; border: 1px solid #FF4B4B44;
    border-left: 4px solid #FF4B4B; border-radius: 10px; padding: 20px;
    animation: slideUpFadeIn 0.5s ease-out;
">
    <div style="color: #FF4B4B; font-size: 14px; font-weight: bold;">
        ⚠️ Training did not complete.
    </div>
    <div style="color: #888; font-size: 12px; margin-top: 6px;">
        Check your dataset and hyperparameters.
    </div>
</div>
                """, unsafe_allow_html=True)

        else:
            status_indicator.markdown(f"""
<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--neural-amber);padding:8px 0;display:flex;align-items:center;gap:8px;">
  <span style="width:6px;height:6px;background:var(--neural-amber);border-radius:50%;box-shadow:0 0 8px var(--neural-amber);display:inline-block;animation:heartbeat 1.5s infinite;"></span>
  TRAINING TRIAL #{active_trial.id}...
</div>
""", unsafe_allow_html=True)
            
            # Layout for Live Graphs
            g1, g2 = st.columns(2)
            with g1: 
                st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:4px;color:var(--bio-cyan);text-transform:uppercase;margin-bottom:8px;">◈ LOSS CONVERGENCE</div>', unsafe_allow_html=True)
                loss_chart = st.empty()
            with g2: 
                st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:4px;color:var(--neural-amber);text-transform:uppercase;margin-bottom:8px;">◈ ADAPTIVE REGULARIZATION — LIVE</div>', unsafe_allow_html=True)
                reg_chart = st.empty()
                
            progress_bar = st.progress(0)
            
            # Data Buffers
            history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'l2': []}
            
            trainer = DynamicTrainer("temp.csv", dna, params, target_col if target_col else None)
            
            # === THE TRAINING LOOP ===
            final_metric = 0.0
            start_time = time.time()
            for stats in trainer.run(epochs=30):
                # Update Data
                history['epoch'].append(stats['epoch'])
                history['train_loss'].append(stats['train_loss'])
                history['val_loss'].append(stats['val_loss'])
                history['l2'].append(stats['current_l2'])
                
                final_metric = stats['metric']
                
                # Show Adaptation Messages
                if stats.get('adaptation'):
                    st.toast(stats['adaptation'], icon="🧠")
                
                # 1. Loss Chart (Multi-line)
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=history['epoch'], y=history['train_loss'], 
                                            mode='lines', name='Train', line=dict(color='#00FFFF', width=2)))
                fig_loss.add_trace(go.Scatter(x=history['epoch'], y=history['val_loss'], 
                                            mode='lines', name='Val', line=dict(color='#FF00FF', width=2)))
                fig_loss.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(8,11,20,0.8)',
                    font=dict(family='Share Tech Mono', color='#3D4F66', size=10),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    xaxis=dict(showgrid=False, title=dict(text='EPOCH', font=dict(size=8, family='Share Tech Mono')), color='#3D4F66'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(26,37,64,0.5)', title=dict(text='LOSS', font=dict(size=8, family='Share Tech Mono')), color='#3D4F66'),
                    legend=dict(orientation='h', y=1.1, font=dict(family='Share Tech Mono', size=9), bgcolor='rgba(0,0,0,0)'),
                )
                loss_chart.plotly_chart(fig_loss, use_container_width=True, key=f"loss_{stats['epoch']}")
                
                # 2. Adaptation Chart (Proving Bilevel Opt)
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=history['epoch'], y=history['l2'], 
                                           mode='lines+markers', name='L2 Reg', 
                                           line=dict(color='#FFFF00', width=3)))
                fig_reg.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(8,11,20,0.8)',
                    font=dict(family='Share Tech Mono', color='#3D4F66', size=10),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    xaxis=dict(showgrid=False, title=dict(text='EPOCH', font=dict(size=8, family='Share Tech Mono')), color='#3D4F66'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(26,37,64,0.5)', title=dict(text='WEIGHT DECAY', font=dict(size=8, family='Share Tech Mono')), color='#3D4F66'),
                    legend=dict(orientation='h', y=1.1, font=dict(family='Share Tech Mono', size=9), bgcolor='rgba(0,0,0,0)'),
                )
                reg_chart.plotly_chart(fig_reg, use_container_width=True, key=f"reg_{stats['epoch']}")
                
                progress_bar.progress(stats['epoch'] / 30)
                time.sleep(0.02) # Yield for rendering
                
            # === COMPLETE THE TRIAL ===
            metric_key = 'Accuracy' if dna.get('task_type') == 'classification' else 'R2 Score'
            training_results = {
                'final_metric': final_metric,
                'metric_name': metric_key,
                'training_time': time.time() - start_time
            }
            
            if (training_results is not None and training_results.get('final_metric') is not None):
                # --- Vizier Trial Tracking ---
                if 'study' not in st.session_state:
                    from vizier_stub import Study
                    st.session_state['study'] = Study(name="metatune_session")

                from vizier_stub import Trial
                _run_id = len(st.session_state['study'].get_trials())
                _trial = Trial(id=_run_id, parameters=params)
                _trial.complete(
                    metric_value=training_results.get('final_metric', 0.0),
                    elapsed_secs=training_results.get('training_time', 0.0)
                )
                st.session_state['study'].add_trial(_trial)
                # --- End Trial Tracking ---

                active_trial.complete(metric_value=training_results.get('final_metric', 0.0), elapsed_secs=training_results.get('training_time', 0.0))
                designer.update(active_trial, study.trials)
                
                status_indicator.markdown(f"""
<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--dna-green);padding:8px 0;display:flex;align-items:center;gap:8px;">
  <span style="width:6px;height:6px;background:var(--dna-green);border-radius:50%;box-shadow:0 0 8px var(--dna-green);display:inline-block;"></span>
  TRIAL #{active_trial.id} COMPLETE
</div>
""", unsafe_allow_html=True)
                st.balloons()
                
                final_score_pct = training_results.get('final_metric', 0.0) * 100
                sc = '#00FF41' if final_score_pct >= 85 else ('#FFB800' if final_score_pct >= 70 else '#FF006E')

                st.markdown(f"""
<div style="
  background:linear-gradient(135deg,var(--void),var(--deep));
  border:1px solid rgba(0,212,255,0.2);
  border-top:2px solid var(--bio-cyan);
  padding:32px;
  margin:16px 0;
  clip-path:polygon(0 0,calc(100% - 20px) 0,100% 20px,100% 100%,0 100%);
  animation:slideUpFadeIn 0.6s ease-out;
">
  <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
    <!-- Orbital spinner (static-looking for complete state) -->
    <div style="width:64px;height:64px;position:relative;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
      <div style="position:absolute;inset:0;border:2px solid var(--bio-cyan);border-right-color:transparent;border-radius:50%;animation:orbitalSpin 2s linear infinite;"></div>
      <span style="font-size:24px;position:relative;">🧠</span>
    </div>
    <div style="flex:1;">
      <div style="font-family:var(--font-mono);color:var(--bio-cyan);font-size:9px;letter-spacing:5px;margin-bottom:4px;text-transform:uppercase;">
        PYTORCH ENGINE · TRIAL #{active_trial.id} · COMPLETE
      </div>
      <div style="font-family:var(--font-display);font-size:36px;letter-spacing:2px;color:var(--text-primary);">
        {training_results.get('metric_name', 'METRIC')}: <span style="color:{sc};">{training_results.get('final_metric', 0.0):.4f}</span>
      </div>
      <!-- Performance bar -->
      <div style="margin-top:12px;background:rgba(26,37,64,0.6);border-radius:2px;height:4px;width:min(400px,100%);overflow:hidden;">
        <div style="height:100%;width:{min(final_score_pct,100):.0f}%;background:linear-gradient(90deg,var(--bio-cyan),var(--dna-green));border-radius:2px;box-shadow:0 0 8px rgba(0,212,255,0.5);"></div>
      </div>
      <div style="font-family:var(--font-mono);color:var(--text-dim);font-size:9px;margin-top:8px;letter-spacing:1px;">
        Trained in {training_results.get('training_time', 0.0):.2f}s ·
        {training_results.get('total_epochs', 30)} epochs ·
        Best at epoch {training_results.get('best_epoch', 0) + 1}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            else:
                st.error("⚠️ Training did not complete. Check your dataset and hyperparameters.")
            
            # Save Model Button
            torch.save(trainer.model.state_dict(), "best_model.pth")
            
            with open("best_model.pth", "rb") as f:
                st.download_button(
                    label="💾 Download Trained Model (.pth)",
                    data=f,
                    file_name="meta_tune_model.pth",
                    mime="application/octet-stream"
                )
