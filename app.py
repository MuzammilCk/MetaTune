import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import torch

# Import your modules
from data_analyzer import DatasetAnalyzer
from brain import MetaLearner
from engine import DynamicTrainer

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="MetaTune AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS FOR "OUT OF WORLD" LOOK ===
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

# === SIDEBAR ===
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
    st.markdown("### **AI-Driven Hyperparameter Optimization**")
    st.markdown("---")
    st.info("💡 **System Status:** Online")
    st.info("🛡️ **Bilevel Optimization:** Active")
    st.info("🧠 **Meta-Learning Core:** Ready")

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

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    # Save file temporarily
    with open("temp_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # === PHASE 1: DIAGNOSIS ===
    st.markdown("---")
    st.header("1. 🧬 Forensic Data Diagnosis")
    
    with st.spinner('Scanning dataset DNA...'):
        analyzer = DatasetAnalyzer("temp_data.csv")
        analyzer.load_data()
        dna = analyzer.analyze()
        time.sleep(1) # Dramatic pause for effect
    
    # Display DNA Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (Instances)", dna['n_instances'])
    col2.metric("Features", dna['n_features'])
    col3.metric("Complexity (Entropy)", f"{dna['target_entropy']:.3f}")
    col4.metric("Noise (Imbalance)", f"{dna['class_imbalance_ratio']:.2f}")
    
    # Show "Brain" Logic
    if dna['target_entropy'] > 1.0:
        st.warning(f"⚠️ High Entropy Detected ({dna['target_entropy']:.2f}). MetaTune will enforce stricter regularization.")
    else:
        st.success("✅ Dataset Stability Confirmed. Standard optimization protocols engaged.")

    # === PHASE 2: PRESCRIPTION ===
    st.markdown("---")
    st.header("2. 🧠 Neural Hyperparameter Prediction")
    
    if st.button("Query Meta-Learner"):
        with st.spinner('Meta-Brain is calculating optimal geometry...'):
            brain = MetaLearner()
            # Check for saved brain
            if not os.path.exists("meta_brain.pkl"):
                brain.train(epochs=10) # Quick boot for demo
            else:
                brain = MetaLearner.load("meta_brain.pkl")
            
            params = brain.predict(dna)
            time.sleep(1.5) # Thinking time
            
        # Display Predicted Params
        st.markdown("#### ✨ AI-Generated Configuration")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Learning Rate", f"{params['learning_rate']:.5f}")
        p_col2.metric("L2 Regularization", f"{params['weight_decay_l2']:.5f}")
        p_col3.metric("Batch Size", params['batch_size'])
        p_col4.metric("Optimizer", params['optimizer_type'].upper())
        
        # Save params to session state for next step
        st.session_state['params'] = params
        st.session_state['ready_to_train'] = True

    # === PHASE 3: EXECUTION ===
    if st.session_state.get('ready_to_train'):
        st.markdown("---")
        st.header("3. 🚀 Dynamic Bilevel Training")
        
        if st.button("Start Training Engine"):
            # Containers for real-time updates
            chart_col, log_col = st.columns([2, 1])
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Chart placeholders
            with chart_col:
                chart_placeholder = st.empty()
            
            # Real-time data storage
            loss_data = {"epoch": [], "train_loss": [], "val_loss": []}
            
            # Callback function to update UI
            def update_ui(epoch, total_epochs, t_loss, v_loss, metric):
                progress = float(epoch) / total_epochs
                progress_bar.progress(progress)
                status_text.markdown(f"**Epoch {epoch}/{total_epochs}** | Accuracy: **{metric:.2%}**")
                
                # Update Chart
                loss_data["epoch"].append(epoch)
                loss_data["train_loss"].append(t_loss)
                loss_data["val_loss"].append(v_loss)
                
                chart_df = pd.DataFrame(loss_data).set_index("epoch")
                chart_placeholder.line_chart(chart_df)

            # Run Engine
            trainer = DynamicTrainer(
                "temp_data.csv", 
                dna, 
                st.session_state['params'], 
                progress_callback=update_ui
            )
            
            result = trainer.run(epochs=30)
            
            st.success(f"🏆 Training Complete! Final Accuracy: {result['final_metric']:.4f}")
            st.balloons()

            # === NEW: SAVE & DOWNLOAD MODEL ===
            # 1. Save the model locally
            torch.save(trainer.model.state_dict(), "trained_model.pth")
            
            # 2. Create a download button
            with open("trained_model.pth", "rb") as f:
                st.download_button(
                    label="💾 Download Trained Model (.pth)",
                    data=f,
                    file_name="my_metatune_model.pth",
                    mime="application/octet-stream"
                )
