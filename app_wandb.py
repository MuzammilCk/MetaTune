import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import torch

# Import modules
from data_analyzer import DatasetAnalyzer
from brain import MetaLearner
from engine_stream import DynamicTrainer # Updated Engine
from algorithm_recommender import recommend_algorithms
from sklearn_engine import train_and_package, package_to_joblib_bytes
from utils.ui_styler import setup_tactical_style, render_tactical_hero

# === UI CONFIGURATION ===
st.set_page_config(page_title="MetaTune Workspace", page_icon="⚡", layout="wide")

# Custom CSS for WandB Aesthetic
# Custom CSS for WandB Aesthetic
setup_tactical_style()

# === SIDEBAR: PROJECT CONFIG ===
with st.sidebar:

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
render_tactical_hero(main_text="METATUNE", sub_text="EXPERIMENT", subtitle="Advanced Hyperparameter Optimization")

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
        recommendations = recommend_algorithms(dna, params)
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
            status_indicator.warning(f"Training & packaging (deployable) for Trial #{active_trial.id}...")
            with st.spinner("Training deployable model..."):
                package, metrics = train_and_package(
                    data_path="temp.csv",
                    dna=dna,
                    algorithm_id=selected_algorithm_id,
                    target_col=(target_col if target_col else None),
                    hyperparameters=params,
                )
                payload = package_to_joblib_bytes(package)

            metric_key = list(metrics.keys())[0] if metrics else "metric"
            metric_value = float(metrics.get(metric_key, 0.0))

            active_trial.complete(Measurement(metrics={metric_key: metric_value}))
            designer.update(active_trial, study.trials)

            status_indicator.success(f"Trial #{active_trial.id} Completed")
            st.success(f"🏆 Final {metric_key.upper()}: {metric_value:.4f}")
            st.download_button(
                label="📦 Download Deployable Package (.joblib)",
                data=payload,
                file_name=f"metatune_{selected_algorithm_id}_package.joblib",
                mime="application/octet-stream",
            )
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
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
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
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#B0B3B8', margin=dict(l=10, r=10, t=10, b=10), height=300,
                    xaxis=dict(showgrid=False, title="Epoch"), 
                    yaxis=dict(showgrid=True, gridcolor='#333', title="Weight Decay Strength"),
                    legend=dict(orientation="h", y=1.1)
                )
                reg_chart.plotly_chart(fig_reg, use_container_width=True, key=f"reg_{stats['epoch']}")
                
                progress_bar.progress(stats['epoch'] / 30)
                time.sleep(0.02) # Yield for rendering
                
            # === COMPLETE THE TRIAL ===
            metric_key = 'accuracy' if dna.get('task_type') == 'classification' else 'r2'
            active_trial.complete(Measurement(metrics={metric_key: final_metric}))
            designer.update(active_trial, study.trials)
            
            status_indicator.success(f"Trial #{active_trial.id} Completed")
            st.balloons()
            
            # Final Metrics
            metric_label = 'Accuracy' if metric_key == 'accuracy' else 'R2'
            st.success(f"🏆 Final {metric_label}: {final_metric:.4f}")
            
            # Save Model Button
            torch.save(trainer.model.state_dict(), "best_model.pth")
            
            with open("best_model.pth", "rb") as f:
                st.download_button(
                    label="💾 Download Trained Model (.pth)",
                    data=f,
                    file_name="meta_tune_model.pth",
                    mime="application/octet-stream"
                )
