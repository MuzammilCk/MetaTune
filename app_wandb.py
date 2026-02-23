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
<style>
    /* Dark Mode Base */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00FFFF; /* Cyan Accent */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; }
    h1 { color: #FAFAFA; }
    .highlight { color: #FFD700; }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF00FF 100%);
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.5);
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }

/* ============================================================
   METATUNE NEURAL MISSION CONTROL — ANIMATION SYSTEM
   ============================================================ */

/* Pulsing glow for active elements */
@keyframes neuralPulse {
    0%   { box-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 20px #00FFFF; }
    50%  { box-shadow: 0 0 20px #00FFFF, 0 0 40px #00FFFF, 0 0 80px #00FFFF; }
    100% { box-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 20px #00FFFF; }
}

/* Horizontal scan line sweep */
@keyframes scanSweep {
    0%   { transform: translateX(-100%); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { transform: translateX(100%); opacity: 0; }
}

/* Matrix rain digit flicker */
@keyframes matrixFlicker {
    0%, 100% { opacity: 1; color: #00FF41; }
    25%       { opacity: 0.4; color: #00FFFF; }
    50%       { opacity: 1; color: #00FF41; }
    75%       { opacity: 0.7; color: #FFB800; }
}

/* Status bar fill animation */
@keyframes barFill {
    from { width: 0%; }
    to   { width: var(--target-width, 100%); }
}

/* Orbital ring spin */
@keyframes orbitalSpin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* Counter-spin for inner ring */
@keyframes orbitalSpinReverse {
    from { transform: rotate(0deg); }
    to   { transform: rotate(-360deg); }
}

/* Data packet travel along wire */
@keyframes dataPacket {
    0%   { left: -10px; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { left: calc(100% + 10px); opacity: 0; }
}

/* Heartbeat for live indicators */
@keyframes heartbeat {
    0%, 100% { transform: scale(1);   opacity: 1; }
    14%       { transform: scale(1.3); opacity: 0.8; }
    28%       { transform: scale(1);   opacity: 1; }
    42%       { transform: scale(1.3); opacity: 0.8; }
    70%       { transform: scale(1);   opacity: 1; }
}

/* Radar sweep */
@keyframes radarSweep {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* Glitch effect for header text */
@keyframes glitchText {
    0%, 100% { text-shadow: 2px 0 #FF00FF, -2px 0 #00FFFF; clip-path: none; }
    20%       { text-shadow: -2px 0 #FF00FF, 2px 0 #00FFFF; clip-path: inset(10% 0 85% 0); }
    40%       { text-shadow: 2px 0 #FFB800, -2px 0 #00FF41; clip-path: inset(50% 0 30% 0); }
    60%       { text-shadow: -2px 0 #00FFFF, 2px 0 #FF00FF; clip-path: inset(80% 0 5% 0); }
    80%       { text-shadow: 2px 2px #FF00FF; clip-path: none; }
}

/* Fade-in slide up for result cards */
@keyframes slideUpFadeIn {
    from { transform: translateY(30px); opacity: 0; }
    to   { transform: translateY(0px);  opacity: 1; }
}

/* Number odometer roll */
@keyframes numberRoll {
    from { transform: translateY(-100%); }
    to   { transform: translateY(0%); }
}

/* Border trace animation */
@keyframes borderTrace {
    0%   { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

/* IGNITE button idle pulse */
.stButton > button {
    animation: ignitePulse 3s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}

@keyframes ignitePulse {
    0%, 100% { box-shadow: 0 0 10px rgba(255, 0, 255, 0.4); }
    50%       { box-shadow: 0 0 30px rgba(255, 0, 255, 0.9), 0 0 60px rgba(255, 75, 75, 0.5); }
}

/* Metric card hover glow */
div[data-testid="stMetric"] {
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    border-left-color: #FF00FF !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)


# === SIDEBAR: PROJECT CONFIG ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("MetaTune / Core")
    st.markdown("Automatic Hyperparameter Optimization")
    
    st.markdown("---")
    st.markdown("### 📂 Data Ingestion")
    uploaded_file = st.file_uploader("Drop CSV Dataset", type=['csv'])
    
    target_col = st.text_input("Target Column (Optional)", help="Leave empty for auto-detection")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    status_indicator = st.empty()
    status_indicator.info("System Idle")

# === MAIN WORKSPACE ===
st.title("⚡ MetaTune Experiment Dashboard")

if uploaded_file:
    # Save temp file
    with open("temp.csv", "wb") as f: f.write(uploaded_file.getbuffer())
    
    # 1. ANALYSIS ROW
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧬 Dataset DNA")
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
            r=values, theta=categories, fill='toself', 
            line_color='#00FFFF', fillcolor='rgba(0, 255, 255, 0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                bgcolor='#1E1E1E'
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color="white", margin=dict(l=20, r=20, t=20, b=20), height=250
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.subheader("🧠 Meta-Learning Designer")
        
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
        
        # Metric Grid
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trial ID", f"#{active_trial.id}")
        m2.metric("Learning Rate", f"{params['learning_rate']:.2e}")
        m3.metric("L2 Regularization", f"{params['weight_decay_l2']:.2e}")
        m4.metric("Optimizer", params['optimizer_type'].upper())
        
        st.info(f"💡 **Designer Reasoning:** Detected entropy of {dna['target_entropy']:.2f}. " 
                f"{'High regularization' if dna['target_entropy'] > 1.0 else 'Standard profile'} suggested for Trial {active_trial.id}.")

        st.markdown("---")
        st.subheader("🧩 Algorithm Recommendation")
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
            st.info(f"**Why this algorithm?** {selected_reason}")
        if selected_algorithm_id != "pytorch_mlp":
            st.success("Deployable output: a single `.joblib` package containing preprocessing + model (+ label encoder if classification).")
        else:
            st.warning("PyTorch path exports weights (`.pth`). For deployment you will still need to replicate preprocessing at inference.")

    # 2. TRAINING ROW (THE LIVE PART)
    st.markdown("---")
    col_btn, col_txt = st.columns([1, 4])
    with col_btn:
        start_btn = st.button("🚀 IGNITE ENGINE")
    
    if start_btn:
        if selected_algorithm_id != "pytorch_mlp":
            status_indicator.warning(
                f"Training & packaging (deployable) for Trial #{active_trial.id}..."
            )

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
            launch_container.markdown(textwrap.dedent(f"""
            <div style="
                background: linear-gradient(135deg, #0A0A0F 0%, #0D1117 50%, #0A0A0F 100%);
                border: 1px solid #00FFFF33;
                border-radius: 12px;
                padding: 32px;
                position: relative;
                overflow: hidden;
                font-family: 'Courier New', monospace;
            ">
                <!-- Scan line sweep -->
                <div style="
                    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                    background: linear-gradient(90deg, transparent 0%, rgba(0,255,255,0.05) 50%, transparent 100%);
                    animation: scanSweep 2s linear infinite;
                    pointer-events: none;
                "></div>

                <!-- Header -->
                <div style="
                    display: flex; align-items: center; gap: 16px; margin-bottom: 28px;
                ">
                    <div style="
                        width: 48px; height: 48px;
                        border: 2px solid #00FFFF;
                        border-radius: 50%;
                        border-top-color: transparent;
                        animation: orbitalSpin 1s linear infinite;
                        display: flex; align-items: center; justify-content: center;
                        position: relative;
                    ">
                        <div style="
                            position: absolute;
                            width: 32px; height: 32px;
                            border: 2px solid #FF00FF;
                            border-radius: 50%;
                            border-bottom-color: transparent;
                            animation: orbitalSpinReverse 0.7s linear infinite;
                        "></div>
                        <span style="font-size: 14px;">⚡</span>
                    </div>
                    <div>
                        <div style="
                            color: #00FFFF; font-size: 18px; font-weight: bold;
                            letter-spacing: 4px; text-transform: uppercase;
                            animation: glitchText 4s infinite;
                        ">NEURAL ENGINE IGNITED</div>
                        <div style="color: #666; font-size: 11px; letter-spacing: 2px;">
                            TRIAL #{active_trial.id} · ALGORITHM: {selected_algo_label.upper()} · MODE: DEPLOYABLE
                        </div>
                    </div>
                    <div style="margin-left: auto; text-align: right;">
                        <div style="
                            width: 12px; height: 12px; background: #00FF41;
                            border-radius: 50%; display: inline-block;
                            animation: heartbeat 1.5s ease-in-out infinite;
                            box-shadow: 0 0 10px #00FF41;
                        "></div>
                        <span style="color: #00FF41; font-size: 11px; letter-spacing: 2px; margin-left: 6px;">LIVE</span>
                    </div>
                </div>

                <!-- Phase indicators -->
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px;">
                    {phase_cards_html}
                </div>

                <!-- Animated progress bar -->
                <div style="margin-bottom: 16px;">
                    <div style="
                        display: flex; justify-content: space-between;
                        color: #666; font-size: 10px; letter-spacing: 2px;
                        margin-bottom: 6px;
                    ">
                        <span>PROCESSING</span><span style="color: #00FFFF; animation: matrixFlicker 0.5s infinite;">RUNNING...</span>
                    </div>
                    <div style="
                        background: #111; border-radius: 4px;
                        height: 6px; overflow: hidden; position: relative;
                    ">
                        <div style="
                            height: 100%; width: 100%;
                            background: linear-gradient(90deg, #00FFFF, #FF00FF, #00FFFF);
                            background-size: 200% 100%;
                            animation: borderTrace 1.5s linear infinite;
                            border-radius: 4px;
                        "></div>
                    </div>
                </div>

                <!-- Data stream readout -->
                <div style="
                    background: #050505; border: 1px solid #1a1a1a;
                    border-radius: 6px; padding: 12px;
                    font-family: 'Courier New', monospace; font-size: 11px;
                    color: #00FF41; line-height: 1.8;
                ">
                    <span style="animation: matrixFlicker 0.3s infinite; animation-delay: 0.0s;">▶ Initializing preprocessing pipeline...</span><br>
                    <span style="animation: matrixFlicker 0.3s infinite; animation-delay: 0.1s;">▶ Splitting train/validation sets (80/20)...</span><br>
                    <span style="color: #FFB800; animation: matrixFlicker 0.4s infinite; animation-delay: 0.3s;">◈ Anti-overfitting regularization: ACTIVE</span>
                </div>
            </div>
            """), unsafe_allow_html=True)

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

                hyperparam_html = ""
                for i, (k, v) in enumerate(list(params.items())[:6]):
                    val_str = f'{v:.4f}' if isinstance(v, float) else str(v)
                    delay = f'{i*0.1:.1f}s'
                    hyperparam_html += f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="color:#555;font-size:10px;font-family:'Courier New',monospace;">{str(k).upper()[:16]}</span>
                        <span style="color:#00FFFF;font-size:11px;font-weight:bold;font-family:'Courier New',monospace;animation:matrixFlicker 2s infinite;animation-delay:{delay};">{val_str}</span>
                    </div>"""

                # ── RESULTS EXPLOSION ANIMATION ────────────────────────────────
                # Score color based on performance
                score_pct = final_metric * 100
                if score_pct >= 85:
                    score_color = '#00FF41'
                    score_label = 'EXCELLENT'
                    score_ring_color = '#00FF41'
                elif score_pct >= 70:
                    score_color = '#FFB800'
                    score_label = 'GOOD'
                    score_ring_color = '#FFB800'
                else:
                    score_color = '#FF4B4B'
                    score_label = 'TRAINING'
                    score_ring_color = '#FF4B4B'

                st.markdown(textwrap.dedent(f"""
                <div style="
                    background: linear-gradient(135deg, #0A0A0F 0%, #0D1117 100%);
                    border: 1px solid {score_color}44;
                    border-radius: 16px;
                    padding: 32px;
                    margin: 16px 0;
                    animation: slideUpFadeIn 0.6s ease-out;
                    position: relative;
                    overflow: hidden;
                ">
                    <!-- Victory glow backdrop -->
                    <div style="
                        position: absolute; top: -50%; left: 50%;
                        width: 300px; height: 300px;
                        background: radial-gradient(circle, {score_color}15 0%, transparent 70%);
                        transform: translateX(-50%);
                        pointer-events: none;
                    "></div>

                    <!-- Header row -->
                    <div style="
                        display: flex; align-items: center;
                        justify-content: space-between; margin-bottom: 28px;
                    ">
                        <div>
                            <div style="
                                font-family: 'Courier New', monospace;
                                color: {score_color}; font-size: 11px;
                                letter-spacing: 4px; margin-bottom: 4px;
                            ">TRAINING COMPLETE · TRIAL #{active_trial.id}</div>
                            <div style="
                                color: #FAFAFA; font-size: 22px; font-weight: 700;
                                letter-spacing: 2px;
                            ">{selected_algo_label}</div>
                        </div>
                        <!-- Orbital score ring -->
                        <div style="
                            width: 90px; height: 90px; position: relative;
                            display: flex; align-items: center; justify-content: center;
                        ">
                            <div style="
                                position: absolute; inset: 0;
                                border: 3px solid {score_ring_color};
                                border-radius: 50%;
                                border-right-color: transparent;
                                animation: orbitalSpin 2s linear infinite;
                            "></div>
                            <div style="
                                position: absolute; inset: 8px;
                                border: 2px solid {score_ring_color}44;
                                border-radius: 50%;
                                border-left-color: transparent;
                                animation: orbitalSpinReverse 3s linear infinite;
                            "></div>
                            <div style="text-align: center;">
                                <div style="
                                    color: {score_color}; font-size: 18px; font-weight: 900;
                                    font-family: 'Courier New', monospace;
                                    animation: numberRoll 0.5s ease-out;
                                ">{score_pct:.1f}%</div>
                                <div style="color: #666; font-size: 9px; letter-spacing: 1px;">{score_label}</div>
                            </div>
                        </div>
                    </div>

                    <!-- Metric grid -->
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;">
                        <div style="
                            background: #0D1117; border: 1px solid {score_color}33;
                            border-radius: 10px; padding: 16px; text-align: center;
                            animation: slideUpFadeIn 0.6s ease-out 0.1s both;
                            border-left: 3px solid {score_color};
                        ">
                            <div style="color: #888; font-size: 10px; letter-spacing: 2px; margin-bottom: 8px;">
                                {metric_name.upper()}
                            </div>
                            <div style="color: {score_color}; font-size: 28px; font-weight: 900;
                                        font-family: 'Courier New', monospace;">
                                {final_metric:.4f}
                            </div>
                            <div style="color: #555; font-size: 9px; margin-top: 4px;">PRIMARY OBJECTIVE</div>
                        </div>
                        <div style="
                            background: #0D1117; border: 1px solid #FF00FF33;
                            border-radius: 10px; padding: 16px; text-align: center;
                            animation: slideUpFadeIn 0.6s ease-out 0.2s both;
                            border-left: 3px solid #FF00FF;
                        ">
                            <div style="color: #888; font-size: 10px; letter-spacing: 2px; margin-bottom: 8px;">TRAIN TIME</div>
                            <div style="color: #FF00FF; font-size: 28px; font-weight: 900;
                                        font-family: 'Courier New', monospace;">
                                {training_time:.2f}s
                            </div>
                            <div style="color: #555; font-size: 9px; margin-top: 4px;">WALL CLOCK</div>
                        </div>
                        <div style="
                            background: #0D1117; border: 1px solid #00FFFF33;
                            border-radius: 10px; padding: 16px; text-align: center;
                            animation: slideUpFadeIn 0.6s ease-out 0.3s both;
                            border-left: 3px solid #00FFFF;
                        ">
                            <div style="color: #888; font-size: 10px; letter-spacing: 2px; margin-bottom: 8px;">ALGORITHM</div>
                            <div style="color: #00FFFF; font-size: 14px; font-weight: 900;
                                        font-family: 'Courier New', monospace; margin-top: 8px;">
                                {selected_algorithm_id.upper()}
                            </div>
                            <div style="color: #555; font-size: 9px; margin-top: 4px;">DEPLOYABLE ✓</div>
                        </div>
                    </div>

                    <!-- Animated performance bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="
                            display: flex; justify-content: space-between;
                            color: #555; font-size: 10px; letter-spacing: 2px; margin-bottom: 8px;
                        ">
                            <span>MODEL PERFORMANCE INDEX</span>
                            <span style="color: {score_color};">{score_pct:.1f}%</span>
                        </div>
                        <div style="background: #111; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="
                                height: 100%;
                                width: {min(score_pct, 100):.0f}%;
                                background: linear-gradient(90deg, {score_ring_color}, {score_color});
                                border-radius: 4px;
                                box-shadow: 0 0 10px {score_color}88;
                                transition: width 1s ease-out;
                            "></div>
                        </div>
                    </div>

                    <!-- Hyperparameter readout -->
                    <div style="
                        background: #050505; border: 1px solid #1a1a1a;
                        border-radius: 8px; padding: 16px;
                        animation: slideUpFadeIn 0.6s ease-out 0.4s both;
                    ">
                        <div style="
                            color: #444; font-size: 10px; letter-spacing: 3px;
                            margin-bottom: 10px;
                        ">HYPERPARAMETER CONFIGURATION</div>
                        <div style="
                            display: grid;
                            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
                            gap: 8px;
                        ">
                            {hyperparam_html}
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

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
                        st.markdown(textwrap.dedent(f"""
                        <div style="
                            background: linear-gradient(135deg, #0A1A0A, #0A0A0F);
                            border: 1px solid #00FF4144;
                            border-left: 4px solid #00FF41;
                            border-radius: 10px; padding: 16px;
                            animation: slideUpFadeIn 0.5s ease-out 0.5s both;
                            display: flex; align-items: center; gap: 16px;
                        ">
                            <div style="
                                font-size: 28px;
                                animation: heartbeat 2s ease-in-out infinite;
                            ">🏆</div>
                            <div>
                                <div style="color: #00FF41; font-size: 11px; letter-spacing: 3px;">
                                    PERSONAL BEST
                                </div>
                                <div style="color: #FAFAFA; font-size: 20px; font-weight: 900;
                                            font-family: 'Courier New', monospace;">
                                    {_best.final_measurement:.4f}
                                </div>
                                <div style="color: #555; font-size: 10px;">
                                    Trial #{_best.id} · {_best.elapsed_secs:.1f}s training time
                                </div>
                            </div>
                        </div>
                        """), unsafe_allow_html=True)
                        with st.expander("🔍 Best Hyperparameters"):
                            st.json(_best.parameters)

                # ── DOWNLOAD BUTTON ────────────────────────────────────────────
                st.download_button(
                    label="📦 Download Deployable Package (.joblib)",
                    data=payload,
                    file_name=f"metatune_{selected_algorithm_id}_package.joblib",
                    mime="application/octet-stream",
                )

            else:
                st.markdown(textwrap.dedent("""
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
                """), unsafe_allow_html=True)
        else:
            status_indicator.warning(f"Training Trial #{active_trial.id}...")
            
            # Layout for Live Graphs
            g1, g2 = st.columns(2)
            with g1: 
                st.markdown("**📉 Loss Convergence**")
                loss_chart = st.empty()
            with g2: 
                st.markdown("**🛡️ Adaptive Regularization (Real-Time)**")
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
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E1E1E',
                    font_color='#B0B3B8', margin=dict(l=10, r=10, t=10, b=10), height=300,
                    xaxis=dict(showgrid=False, title="Epoch"), 
                    yaxis=dict(showgrid=True, gridcolor='#333', title="Loss"),
                    legend=dict(orientation="h", y=1.1)
                )
                loss_chart.plotly_chart(fig_loss, use_container_width=True, key=f"loss_{stats['epoch']}")
                
                # 2. Adaptation Chart (Proving Bilevel Opt)
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=history['epoch'], y=history['l2'], 
                                           mode='lines+markers', name='L2 Reg', 
                                           line=dict(color='#FFFF00', width=3)))
                fig_reg.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E1E1E',
                    font_color='#B0B3B8', margin=dict(l=10, r=10, t=10, b=10), height=300,
                    xaxis=dict(showgrid=False, title="Epoch"), 
                    yaxis=dict(showgrid=True, gridcolor='#333', title="Weight Decay Strength"),
                    legend=dict(orientation="h", y=1.1)
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
                
                status_indicator.success(f"Trial #{active_trial.id} Completed")
                st.balloons()
                
                st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0A0A0F, #0D1117);
    border: 1px solid #00FFFF44; border-top: 3px solid #00FFFF;
    border-radius: 12px; padding: 24px;
    animation: slideUpFadeIn 0.6s ease-out;
    display: flex; align-items: center; gap: 24px;
">
    <div style="
        width: 64px; height: 64px;
        border: 3px solid #00FFFF;
        border-radius: 50%; border-right-color: transparent;
        animation: orbitalSpin 1.5s linear infinite;
        flex-shrink: 0; display: flex;
        align-items: center; justify-content: center;
    ">
        <span style="font-size: 24px;">🧠</span>
    </div>
    <div>
        <div style="color: #00FFFF; font-size: 11px; letter-spacing: 4px; margin-bottom: 4px;">
            PYTORCH ENGINE · TRIAL #{active_trial.id} · COMPLETE
        </div>
        <div style="color: #FAFAFA; font-size: 26px; font-weight: 900;
                    font-family: 'Courier New', monospace;">
            {training_results.get('metric_name', 'METRIC')}: {training_results.get('final_metric', 0.0):.4f}
        </div>
        <div style="
            margin-top: 8px;
            background: #111; border-radius: 3px; height: 4px; width: 300px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                width: {min(training_results.get('final_metric', 0) * 100, 100):.0f}%;
                background: linear-gradient(90deg, #00FFFF, #FF00FF);
                border-radius: 3px;
                box-shadow: 0 0 8px #00FFFF88;
            "></div>
        </div>
        <div style="color: #555; font-size: 10px; margin-top: 6px; letter-spacing: 1px;">
            Trained in {training_results.get('training_time', 0.0):.2f}s ·
            {training_results.get('total_epochs', 30)} epochs ·
            Best at epoch {training_results.get('best_epoch', 0) + 1}
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
